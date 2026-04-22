"""Top-level DDXPlus runner.

Loops cases, calls either baseline or full, emits raw predictions + a
metrics summary to `eval/reports/<timestamp>/ddxplus/`.

Metrics:
- Top-5 accuracy (LLM judge, pinned DDXPLUS_JUDGE_MODEL) — stubbed here; see LIMITATIONS.md.
- HDF1 (ICD-10 hierarchical F1) — stubbed here; see LIMITATIONS.md.

The judge + HDF1 computation are deliberately deferred — they require the
DDXPlus ICD-10 mapping artefact and a pinned OpenAI key, both out of scope
for Day-1 scaffold. This runner produces the raw predictions + a metrics
skeleton; the scoring pass is a separate commit.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from dataclasses import asdict
from pathlib import Path

from eval.ddxplus.adapter import iter_cases
from eval.ddxplus.baseline import DDXPrediction, predict_differential
from eval.ddxplus.full import FullRunner

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = REPO_ROOT / "eval" / "ddxplus" / "data"
REPORTS_ROOT = REPO_ROOT / "eval" / "reports"


def _load_limitations() -> str:
    """Return the active LIMITATIONS.md text so the run report includes it verbatim."""
    p = Path(__file__).resolve().parent / "LIMITATIONS.md"
    return p.read_text(encoding="utf-8") if p.exists() else ""


def _run_variant(
    variant: str, records_path: Path, evidence_dict_path: Path, limit: int | None
) -> list[DDXPrediction]:
    preds: list[DDXPrediction] = []
    runner = FullRunner(evidence_dict_path=evidence_dict_path) if variant == "full" else None

    for i, case in enumerate(iter_cases(records_path)):
        if limit is not None and i >= limit:
            break
        if variant == "baseline":
            preds.append(predict_differential(case))
        elif variant == "full":
            assert runner is not None
            preds.append(runner.predict_differential(case))
        else:
            raise ValueError(f"unknown variant: {variant}")
    return preds


def _write_report(
    variant: str, preds: list[DDXPrediction], out_dir: Path, limit: int | None
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "predictions.jsonl").write_text(
        "\n".join(json.dumps(asdict(p)) for p in preds) + "\n", encoding="utf-8"
    )
    # Metrics are intentionally TODO — scoring pass lives in a separate commit.
    metrics = {
        "variant": variant,
        "n_cases": len(preds),
        "limit": limit,
        "top_5_accuracy": None,
        "hdf1": None,
        "note": (
            "Scoring deferred — LLM judge (DDXPLUS_JUDGE_MODEL) + HDF1 ICD-10 mapping "
            "pipeline to be implemented in a follow-up commit. See eval/ddxplus/LIMITATIONS.md."
        ),
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "LIMITATIONS.md").write_text(_load_limitations(), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=("baseline", "full"),
        default="baseline",
        help="baseline = Opus 4.7 direct; full = substrate-augmented",
    )
    parser.add_argument(
        "--records",
        type=Path,
        default=DEFAULT_DATA / "h_ddx_730.jsonl",
        help="path to JSONL of DDXPlus records (H-DDx 730 stratified subset)",
    )
    parser.add_argument(
        "--evidence-dict",
        type=Path,
        default=DEFAULT_DATA / "ddxplus-repo" / "release_evidences.json",
        help="path to release_evidences.json from the DDXPlus repo",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="smoke: stop after 5 cases, don't write report files",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="max cases (default: full subset); smaller values violate rules.md §6 if reported as comparator",
    )
    args = parser.parse_args(argv)

    if not args.records.exists():
        print(
            f"[run] records file not found: {args.records}\n"
            f"      run `python -m eval.ddxplus.fetch` and prepare the H-DDx 730 subset first.",
            file=sys.stderr,
        )
        return 2

    limit = 5 if args.dry_run else args.limit
    preds = _run_variant(args.variant, args.records, args.evidence_dict, limit)

    if args.dry_run:
        print(f"[run] dry-run: {len(preds)} predictions generated, not writing report.")
        for p in preds[:3]:
            print(f"  {p.patient_id}: top5={p.top5[:3]}...")
        return 0

    ts = dt.datetime.now(tz=dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = REPORTS_ROOT / ts / "ddxplus" / args.variant
    _write_report(args.variant, preds, out_dir, limit)
    print(f"[run] wrote {len(preds)} predictions to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
