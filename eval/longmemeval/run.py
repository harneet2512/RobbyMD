"""Top-level LongMemEval-S runner.

Loops all 500 questions, calls baseline or full, writes raw predictions +
per-category accuracy (using the **official** LongMemEval evaluator — we do
NOT fork/reimplement the judge).

Per `memory/feedback_full_benchmarks.md`: we run the **full 500 questions**.
No slicing. `--dry-run` and `--limit` exist for smoke; reports from limited
runs are marked `slice=true` in `metrics.json` and must not be shown on the
demo slide (enforced visually in the report template, not structurally here).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from dataclasses import asdict
from pathlib import Path

from eval.longmemeval.adapter import QUESTION_CATEGORIES, iter_questions
from eval.longmemeval.baseline import LongMemEvalPrediction, predict_answer
from eval.longmemeval.full import FullRunner

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = REPO_ROOT / "eval" / "longmemeval" / "data"
REPORTS_ROOT = REPO_ROOT / "eval" / "reports"


def _load_limitations() -> str:
    p = Path(__file__).resolve().parent / "LIMITATIONS.md"
    return p.read_text(encoding="utf-8") if p.exists() else ""


def _load_commit_sha() -> str:
    p = DEFAULT_DATA / ".commit_sha"
    return p.read_text(encoding="utf-8").strip() if p.exists() else "UNKNOWN"


def _run_variant(
    variant: str, questions_path: Path, limit: int | None
) -> list[LongMemEvalPrediction]:
    preds: list[LongMemEvalPrediction] = []
    runner = FullRunner() if variant == "full" else None

    for i, q in enumerate(iter_questions(questions_path)):
        if limit is not None and i >= limit:
            break
        if variant == "baseline":
            preds.append(predict_answer(q))
        elif variant == "full":
            assert runner is not None
            preds.append(runner.predict_answer(q))
        else:
            raise ValueError(f"unknown variant: {variant}")
    return preds


def _metrics_skeleton(preds: list[LongMemEvalPrediction], is_slice: bool) -> dict[str, object]:
    # Official evaluator runs on the full `predictions.jsonl`; this function
    # emits a skeleton + per-category counts so the report directory is
    # non-empty even before the judge runs. Judge scoring is a follow-up
    # commit that imports `LongMemEval-repo/src/evaluation/evaluate_qa.py`.
    by_cat: dict[str, int] = dict.fromkeys(QUESTION_CATEGORIES, 0)
    for p in preds:
        by_cat[p.question_type] = by_cat.get(p.question_type, 0) + 1
    return {
        "n_predictions": len(preds),
        "per_category_count": by_cat,
        "per_category_accuracy": None,  # filled by official evaluator in a follow-up
        "overall_accuracy": None,
        "commit_sha": _load_commit_sha(),
        "slice": is_slice,
        "note": (
            "Scoring deferred — runs the OFFICIAL evaluator "
            "(LongMemEval-repo/src/evaluation/evaluate_qa.py) in a follow-up commit. "
            "Do not infer performance from this skeleton."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("baseline", "full"), default="baseline")
    parser.add_argument(
        "--questions",
        type=Path,
        default=DEFAULT_DATA / "LongMemEval-repo" / "data" / "longmemeval_s.json",
        help="path to longmemeval_s.json (500 questions)",
    )
    parser.add_argument("--dry-run", action="store_true", help="smoke: stop after 5 questions")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="max questions (omit for full 500); smaller values are DEV ONLY per rules.md §6",
    )
    args = parser.parse_args(argv)

    if not args.questions.exists():
        print(
            f"[run] questions file not found: {args.questions}\n"
            f"      run `python -m eval.longmemeval.fetch` first.",
            file=sys.stderr,
        )
        return 2

    limit = 5 if args.dry_run else args.limit
    preds = _run_variant(args.variant, args.questions, limit)

    if args.dry_run:
        print(f"[run] dry-run: {len(preds)} predictions generated, not writing report.")
        return 0

    ts = dt.datetime.now(tz=dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = REPORTS_ROOT / ts / "longmemeval" / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "predictions.jsonl").write_text(
        "\n".join(json.dumps(asdict(p)) for p in preds) + "\n", encoding="utf-8"
    )
    (out_dir / "metrics.json").write_text(
        json.dumps(_metrics_skeleton(preds, is_slice=(limit is not None)), indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "LIMITATIONS.md").write_text(_load_limitations(), encoding="utf-8")
    print(f"[run] wrote {len(preds)} predictions to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
