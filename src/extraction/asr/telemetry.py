"""ASR pipeline telemetry — per-stage latency ring buffer + structlog events.

Provides a ``@measure("stage_name")`` decorator that wraps any callable and
records elapsed_ms to a fixed-size ring buffer. `get_stats()` computes P50 /
P95 / P99 / max per stage. `reset()` clears all buffers.

Design notes
------------
- Ring buffer size = 1000 per stage (configurable at module level via
  ``RING_BUFFER_SIZE``). At 5 segments per consultation and one event per stage,
  this stores ~200 consultations of history without growing unboundedly.
- No thread locking — the ASR pipeline is single-threaded per session. If
  concurrent sessions are added later, a `threading.Lock` per stage should be
  introduced.
- structlog events fire on every measure exit with `stage` and `elapsed_ms`
  (CLAUDE.md §8).
- All statistics are exact (no approximation) — the ring buffer stores raw
  floats and sorts on-demand. P50/P95/P99 are computed via sorted-index lookup
  (not interpolated), which is fine for small N.

Wire `@measure` around the main pipeline stages:
  normalize, trim, transcribe, align, diarise, cleanup, guard, correct.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RING_BUFFER_SIZE: int = 1000

# ---------------------------------------------------------------------------
# Module-level state — one deque per named stage.
# ---------------------------------------------------------------------------

_buffers: dict[str, deque[float]] = {}


def _get_buffer(stage: str) -> deque[float]:
    """Get or create the ring buffer for a named stage."""
    if stage not in _buffers:
        _buffers[stage] = deque(maxlen=RING_BUFFER_SIZE)
    return _buffers[stage]


# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------

F = TypeVar("F", bound=Callable[..., Any])


def measure(stage: str) -> Callable[[F], F]:
    """Decorator factory. Records elapsed_ms for `stage` on every call.

    Usage::

        @measure("normalize")
        def normalize_audio(path_in: Path, path_out: Path) -> Path:
            ...

    The wrapped function behaves identically to the original; elapsed_ms is
    recorded in the stage's ring buffer and logged via structlog.
    """
    def decorator(fn: F) -> F:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            t0 = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
            finally:
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                buf = _get_buffer(stage)
                buf.append(elapsed_ms)
                logger.debug(
                    "telemetry.measure",
                    stage=stage,
                    elapsed_ms=round(elapsed_ms, 2),
                )
            return result

        return wrapper  # type: ignore[return-value]

    return decorator


def get_stats() -> dict[str, dict[str, float]]:
    """Return per-stage statistics over all recorded samples.

    Returns a mapping::

        {
          "normalize": {"count": 42, "p50": 12.3, "p95": 45.1, "p99": 67.8, "max": 80.0},
          "transcribe": {...},
          ...
        }

    If a stage has no recorded samples, its entry has ``count=0`` and all
    percentile values are 0.0.
    """
    stats: dict[str, dict[str, float]] = {}
    for stage, buf in _buffers.items():
        samples = sorted(buf)
        n = len(samples)
        if n == 0:
            stats[stage] = {"count": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
        else:
            stats[stage] = {
                "count": float(n),
                "p50": _percentile(samples, 50),
                "p95": _percentile(samples, 95),
                "p99": _percentile(samples, 99),
                "max": float(samples[-1]),
            }
    return stats


def reset() -> None:
    """Clear all ring buffers. Useful between benchmark runs."""
    for buf in _buffers.values():
        buf.clear()
    logger.info("telemetry.reset")


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _percentile(sorted_samples: list[float], pct: int) -> float:
    """Compute the `pct`-th percentile from a sorted list using nearest-rank.

    Nearest-rank (not interpolated) — conservative and simple. For small N
    (< 100 samples) the difference from interpolation is negligible.
    """
    n = len(sorted_samples)
    if n == 0:
        return 0.0
    # Nearest-rank formula: ceil(pct / 100 * n) - 1 (0-indexed).
    idx = max(0, int(round(pct / 100.0 * n)) - 1)
    idx = min(idx, n - 1)
    return float(sorted_samples[idx])
