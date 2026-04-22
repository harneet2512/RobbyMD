"""Download DDXPlus into eval/ddxplus/data/.

Clones `mila-iqia/ddxplus` and pins the commit SHA to
`eval/ddxplus/data/.commit_sha` so subsequent runs are reproducible
(rules.md §6.3 methodology honesty).

Idempotent: if `.commit_sha` already exists, re-verifies the checkout matches
and exits without re-cloning.

No network access happens at import time; `main()` is the only side-effectful
entry point.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DDXPLUS_REPO = "https://github.com/mila-iqia/ddxplus.git"
DATA_DIR = Path(__file__).resolve().parent / "data"
CLONE_DIR = DATA_DIR / "ddxplus-repo"
SHA_FILE = DATA_DIR / ".commit_sha"


def _run(cmd: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({' '.join(cmd)}):\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
    return result.stdout.strip()


def fetch(force: bool = False) -> str:
    """Clone DDXPlus (or verify an existing clone) and return the pinned SHA."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if CLONE_DIR.exists() and not force:
        sha = _run(["git", "rev-parse", "HEAD"], cwd=CLONE_DIR)
        print(f"[fetch] ddxplus already cloned at {CLONE_DIR} @ {sha[:12]}")
    else:
        if CLONE_DIR.exists() and force:
            # TODO(wt-eval): nuke cleanly on force. For now, require a manual rm.
            raise RuntimeError(
                f"{CLONE_DIR} already exists; remove it manually and rerun with --force"
            )
        print(f"[fetch] cloning {DDXPLUS_REPO} into {CLONE_DIR}")
        _run(["git", "clone", "--depth", "1", DDXPLUS_REPO, str(CLONE_DIR)])
        sha = _run(["git", "rev-parse", "HEAD"], cwd=CLONE_DIR)

    SHA_FILE.write_text(sha + "\n", encoding="utf-8")
    print(f"[fetch] pinned SHA: {sha}")
    print(f"[fetch] wrote {SHA_FILE}")
    print(f"[fetch] data dir: {CLONE_DIR}")
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
