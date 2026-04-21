#!/usr/bin/env bash
# setup.sh — install dev dependencies and fetch published benchmark datasets.
# Idempotent. Safe to run multiple times.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

echo "[setup] Installing Python dev dependencies..."
python -m pip install -e ".[dev]"

# Benchmark data download stubs. Implemented per-eval-adapter in wt-eval worktree.
# DDXPlus:       eval/ddxplus/fetch.py
# LongMemEval-S: eval/longmemeval/fetch.py  (pin commit SHA — re-cleaned Sept 2025)
# ACI-Bench:     eval/aci_bench/fetch.py    (requires UMLS licence for MEDCON)
echo "[setup] Benchmark dataset fetchers not yet implemented (see eval/*/fetch.py)."

echo "[setup] Done."
