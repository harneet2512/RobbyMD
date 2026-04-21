"""Test bootstrap — makes the `src` package importable without installing.

The worktree's pyproject.toml targets Python 3.11 specifically (see pyproject.toml
line 10); on a host with only 3.12, `pip install -e .` fails the version guard.
This conftest adds the repo root to `sys.path` at collection time so tests run
regardless. When a 3.11 env is available, `pip install -e .[dev]` remains the
recommended path and makes this conftest redundant.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
