"""Download ACI-Bench into eval/aci_bench/data/.

Clones `wyim/aci-bench` and pins the commit SHA. The dataset ships as CSVs
and directories under `src/data/`; we shallow-clone so we also get the
official MEDIQA-CHAT evaluation scripts.

**MEDCON dependency note**: ACI-Bench's MEDCON implementation requires UMLS
+ QuickUMLS, which is NOT installed by this script. See
`scripts/install_umls.sh` for the UMLS-dependent upgrade path and
`docs/decisions/2026-04-21_medcon-tiered-fallback.md` for the tiered plan.
The T1 default (scispaCy) runs without UMLS.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ACI_BENCH_REPO = "https://github.com/wyim/aci-bench.git"
DATA_DIR = Path(__file__).resolve().parent / "data"
CLONE_DIR = DATA_DIR / "aci-bench-repo"
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
        print(f"[fetch] ACI-Bench already cloned at {CLONE_DIR} @ {sha[:12]}")
    else:
        if CLONE_DIR.exists() and force:
            raise RuntimeError(
                f"{CLONE_DIR} already exists; remove manually and rerun with --force"
            )
        print(f"[fetch] cloning {ACI_BENCH_REPO} into {CLONE_DIR}")
        _run(["git", "clone", "--depth", "1", ACI_BENCH_REPO, str(CLONE_DIR)])
        sha = _run(["git", "rev-parse", "HEAD"], cwd=CLONE_DIR)

    SHA_FILE.write_text(sha + "\n", encoding="utf-8")
    print(f"[fetch] pinned SHA: {sha}")
    print(
        "[fetch] ACI-Bench has 2 test splits: `aci` (66 encounters across "
        "test1/test2/test3) + `virtscribe` (24 encounters). Total: 90 encounters."
    )
    print(
        "[fetch] NOTE: MEDCON needs UMLS (optional — see install_umls.sh). T1 (scispaCy) "
        "runs without UMLS; see docs/decisions/2026-04-21_medcon-tiered-fallback.md."
    )
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
