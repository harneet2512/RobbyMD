"""Test bootstrap — makes the `src` package importable without installing.

The worktree's pyproject.toml targets Python 3.11 specifically (see pyproject.toml
line 10); on a host with only 3.12, `pip install -e .` fails the version guard.
This conftest adds the repo root to `sys.path` at collection time so tests run
regardless. When a 3.11 env is available, `pip install -e .[dev]` remains the
recommended path and makes this conftest redundant.

structlog note: this conftest configures structlog to write to os.devnull
during tests to avoid OSError: [Errno 28] on environments with a full C: drive
(the default structlog ConsoleRenderer writes to sys.stdout which may go through
a Windows tempfile on the full drive). Tests that explicitly need to capture
structlog output use `structlog.testing.capture_logs()` which bypasses this.
"""

from __future__ import annotations

import sys
from pathlib import Path

import structlog

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Configure structlog to discard all log output during tests to avoid OSError
# on environments where C: drive is full (structlog's default ConsoleRenderer
# writes to sys.stdout which on Windows may go through a tempfile on C:).
# Tests that need to assert on log output use structlog.testing.capture_logs()
# which uses its own processor chain and bypasses this configuration.
import io as _io

structlog.configure(
    processors=[
        # DropEvent drops the log entry cleanly.
        lambda _logger, _method, event_dict: (_ for _ in ()).throw(  # type: ignore[misc]
            structlog.DropEvent()
        ),
    ],
    wrapper_class=structlog.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(file=_io.StringIO()),
    cache_logger_on_first_use=False,
)
