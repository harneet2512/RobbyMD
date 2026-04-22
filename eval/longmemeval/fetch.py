"""Download LongMemEval-S into eval/longmemeval/data/.

Clones `xiaowu0162/LongMemEval` and pins the commit SHA. The dataset itself
ships as JSON + JSONL files under `data/` in the upstream repo; we shallow-clone
so we also get the official evaluator script (`src/evaluation/evaluate_qa.py`).

**Sept 2025 caveat**: the authors re-cleaned the dataset on 2025-09-XX. We pin
to HEAD at the time of first fetch and refuse to auto-upgrade; overriding
requires an ADR in `docs/decisions/`.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

LONGMEMEVAL_REPO = "https://github.com/xiaowu0162/LongMemEval.git"
DATA_DIR = Path(__file__).resolve().parent / "data"
CLONE_DIR = DATA_DIR / "LongMemEval-repo"
SHA_FILE = DATA_DIR / ".commit_sha"


def _run(cmd: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({' '.join(cmd)}):\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
    return result.stdout.strip()


def fetch(force: bool = False) -> str:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if CLONE_DIR.exists() and not force:
        sha = _run(["git", "rev-parse", "HEAD"], cwd=CLONE_DIR)
        print(f"[fetch] LongMemEval already cloned at {CLONE_DIR} @ {sha[:12]}")
    else:
        if CLONE_DIR.exists() and force:
            raise RuntimeError(
                f"{CLONE_DIR} already exists; remove manually and rerun with --force"
            )
        print(f"[fetch] cloning {LONGMEMEVAL_REPO} into {CLONE_DIR}")
        _run(["git", "clone", "--depth", "1", LONGMEMEVAL_REPO, str(CLONE_DIR)])
        sha = _run(["git", "rev-parse", "HEAD"], cwd=CLONE_DIR)

    SHA_FILE.write_text(sha + "\n", encoding="utf-8")
    print(f"[fetch] pinned SHA: {sha}")
    print("[fetch] NOTE: authors re-cleaned dataset Sept 2025. Log this SHA in every")
    print("[fetch]       report; pre-2025-09 comparator numbers are NOT compatible.")
    return sha


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-clone even if present")
    args = parser.parse_args(argv)
    try:
        fetch(force=args.force)
    except RuntimeError as e:
        print(f"[fetch] FAIL: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
