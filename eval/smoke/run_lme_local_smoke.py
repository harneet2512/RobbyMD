"""Local LongMemEval smoke runner — Azure-backed, no GPU required.

Thin wrapper over `eval.smoke.run_smoke.main` that pins the canonical flags for
the post-bge-m3-fix verification smoke: n=10 stratified across the 6 LME
question types, both baseline + substrate variants, Azure gpt-4o reader +
judge (via `eval._openai_client` routing on `AZURE_OPENAI_ENDPOINT`), fixed
seed, capped budget.

Preflight: validates the Azure env vars are set and that either a Modal
bge-m3 endpoint or a local sentence-transformers+torch path is available.
Fails fast with an enumerated list of missing vars rather than crashing
deep inside the reader or embedding backend.

Usage:
    python -m eval.smoke.run_lme_local_smoke

Exit codes:
    0 — smoke passed (step (a) query-prefix fix is sufficient)
    1 — smoke warned (ANOMALY — manual review before escalating to (b) MMR)
    2 — smoke failed (escalate to step (b) MMR + top-K 40, or (c) reranker)
    3 — preflight failed (env vars missing; no run attempted)

See eval/smoke/README_LME_LOCAL.md for the env-var contract + interpretation.
"""
from __future__ import annotations

import datetime
import os
import sys
from pathlib import Path


REQUIRED_AZURE_VARS: tuple[str, ...] = (
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_GPT4O_LME_DEPLOYMENT",
)


def _preflight() -> list[str]:
    """Return a list of human-readable preflight failures. Empty list = OK."""
    problems: list[str] = []

    missing = [v for v in REQUIRED_AZURE_VARS if not os.environ.get(v)]
    if missing:
        problems.append(
            "Missing Azure env vars: " + ", ".join(missing) + ". "
            "Set them in your shell or .env before running this smoke. "
            "See eval/smoke/README_LME_LOCAL.md for the contract."
        )

    # bge-m3 backend: either Modal (remote) or sentence-transformers (local).
    modal_url = os.environ.get("MODAL_BGE_M3_URL")
    st_available = True
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        st_available = False
    if not modal_url and not st_available:
        problems.append(
            "No bge-m3 backend reachable. Either set MODAL_BGE_M3_URL to your "
            "deployed Modal endpoint, or `pip install sentence-transformers` "
            "locally (slow on CPU — Modal path strongly preferred on laptops)."
        )

    return problems


def _canonical_argv() -> list[str]:
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path("eval/smoke/results") / f"lme_local_{ts}"
    return [
        "--benchmark", "longmemeval",
        "--reader", "gpt-4o-2024-11-20",
        "--variant", "both",
        "--n", "10",
        "--stratified",
        "--seed", "42",
        "--budget-usd", "10",
        "--output-dir", str(out_dir),
    ]


def main() -> int:
    problems = _preflight()
    if problems:
        print("[lme-local-smoke] preflight failed:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 3

    argv = _canonical_argv()
    print(f"[lme-local-smoke] invoking run_smoke.main with: {' '.join(argv)}")

    from eval.smoke.run_smoke import main as run_smoke_main

    rc = run_smoke_main(argv)

    # run_smoke.main returns 0=PASS, 1=ANOMALY, 2=FAIL. Passthrough.
    print(f"[lme-local-smoke] run_smoke exit code: {rc}")
    if rc == 0:
        print(
            "[lme-local-smoke] smoke passed. bge-m3 asymmetric fix (step a) is "
            "sufficient. No need for step (b) MMR or (c) reranker fallback."
        )
    elif rc == 1:
        print(
            "[lme-local-smoke] ANOMALY verdict — inspect the hypotheses.jsonl "
            "under the output dir for per-question detail before deciding "
            "whether to escalate to step (b)."
        )
    else:
        print(
            "[lme-local-smoke] smoke failed. Step (a) query-prefix alone is "
            "not sufficient. Escalate to step (b): top-K 40 + MMR diversification. "
            "Do NOT jump straight to step (c) reranker without measuring (b) first."
        )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
