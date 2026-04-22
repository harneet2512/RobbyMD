"""Tests for Part D — ASR pipeline telemetry.

Three tests:
1. @measure decorator records elapsed_ms in the ring buffer.
2. get_stats() computes correct P50 / P95 percentiles.
3. reset() clears all buffers.
"""

from __future__ import annotations

import time

import pytest

import src.extraction.asr.telemetry as telemetry


@pytest.fixture(autouse=True)
def _clear_telemetry() -> None:  # type: ignore[misc]
    """Ensure clean state before and after each test."""
    telemetry.reset()
    yield  # type: ignore[misc]
    telemetry.reset()


def test_measure_decorator_records_elapsed() -> None:
    """@measure should add at least one sample to the stage's buffer."""

    @telemetry.measure("test_stage_alpha")
    def _noop() -> str:
        return "ok"

    _noop()
    stats = telemetry.get_stats()
    assert "test_stage_alpha" in stats, "Stage not registered"
    assert stats["test_stage_alpha"]["count"] == 1.0


def test_stats_compute_correct_percentiles() -> None:
    """get_stats() should return accurate P50 and P95 from recorded samples."""

    @telemetry.measure("test_stage_beta")
    def _sleep_variable(duration_s: float) -> None:
        time.sleep(duration_s)

    # Record 10 samples: 1..10 ms (approximately).
    # We inject samples directly for determinism instead of sleeping.
    import src.extraction.asr.telemetry as _tel
    buf = _tel._get_buffer("test_stage_perc")  # pyright: ignore[reportPrivateUsage]
    for ms in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]:
        buf.append(ms)

    stats = telemetry.get_stats()
    assert "test_stage_perc" in stats
    s = stats["test_stage_perc"]
    assert s["count"] == 10.0
    # P50 of [1..10] via nearest-rank is the 5th value = 5.0
    assert s["p50"] == 5.0, f"Expected P50=5.0, got {s['p50']}"
    # P95 of [1..10] via nearest-rank: ceil(0.95*10)=10th value = 10.0
    assert s["p95"] == 10.0, f"Expected P95=10.0, got {s['p95']}"
    assert s["max"] == 10.0


def test_reset_clears_buffers() -> None:
    """reset() should empty all registered buffers."""
    import src.extraction.asr.telemetry as _tel
    buf = _tel._get_buffer("test_stage_gamma")  # pyright: ignore[reportPrivateUsage]
    buf.append(42.0)
    buf.append(99.0)

    telemetry.reset()
    stats = telemetry.get_stats()
    # After reset, any previously-recorded stage should report count=0.
    if "test_stage_gamma" in stats:
        assert stats["test_stage_gamma"]["count"] == 0.0
