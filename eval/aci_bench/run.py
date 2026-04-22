"""Top-level ACI-Bench runner.

Loops all 90 test encounters (aci test1/test2/test3 + virtscribe test),
calls baseline or full, computes MEDIQA-CHAT metrics (ROUGE/BERTScore/MEDCON)
and writes to `eval/reports/<timestamp>/aci_bench/`.

Active concept-extractor tier (T0/T1/T2) is selected at startup via
`CONCEPT_EXTRACTOR` env var and **stamped** at the top of the run's
`LIMITATIONS.md` copy + `metrics.json.active_tier`.

Per `memory/feedback_full_benchmarks.md`: we run the full 90-encounter set.
Smaller `--limit` values are dev smoke only.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from dataclasses import asdict
from pathlib import Path

from eval.aci_bench.adapter import ACIEncounter, iter_all_test_encounters
from eval.aci_bench.baseline import ACINotePrediction, predict_note
from eval.aci_bench.extractors import (
    ConceptExtractor,
    NullExtractor,
    build_extractor,
    compute_medcon_f1,
)
from eval.aci_bench.full import FullRunner

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = REPO_ROOT / "eval" / "aci_bench" / "data" / "aci-bench-repo" / "src" / "data"
REPORTS_ROOT = REPO_ROOT / "eval" / "reports"


def _active_tier_header(extractor: ConceptExtractor) -> str:
    return (
        f"# ACTIVE TIER: {extractor.name.upper()}  —  {extractor.label}\n\n"
        f"_Stamped at run time. See full tier table in this file's 3 sections below."
        f" See `docs/decisions/2026-04-21_medcon-tiered-fallback.md` for the decision record._\n\n"
    )


def _load_limitations() -> str:
    p = Path(__file__).resolve().parent / "LIMITATIONS.md"
    return p.read_text(encoding="utf-8") if p.exists() else ""


def _run_variant(
    variant: str,
    encounters: list[ACIEncounter],
    limit: int | None,
) -> list[ACINotePrediction]:
    preds: list[ACINotePrediction] = []
    runner = FullRunner() if variant == "full" else None

    for i, enc in enumerate(encounters):
        if limit is not None and i >= limit:
            break
        if variant == "baseline":
            preds.append(predict_note(enc))
        elif variant == "full":
            assert runner is not None
            preds.append(runner.predict_note(enc))
        else:
            raise ValueError(f"unknown variant: {variant}")
    return preds


def _score_medcon(
    extractor: ConceptExtractor,
    encounters: list[ACIEncounter],
    preds: list[ACINotePrediction],
) -> dict[str, object]:
    """Compute MEDCON F1 per encounter + micro-averaged.

    For T2 (NullExtractor) this returns `{"omitted": true, ...}` per the ADR
    — MEDCON column is omitted on the slide; supplementary metrics kick in.
    """
    if isinstance(extractor, NullExtractor):
        return {
            "omitted": True,
            "reason": "CONCEPT_EXTRACTOR=null — see LIMITATIONS.md §T2 + ADR",
            "per_encounter": [],
            "micro_f1": None,
        }

    per_encounter: list[dict[str, object]] = []
    all_tp = all_gold = all_pred = 0
    for enc, pred in zip(encounters, preds, strict=False):
        gold_cuis = extractor.extract(enc.gold_note)
        pred_cuis = extractor.extract(pred.predicted_note)
        scores = compute_medcon_f1(gold_cuis, pred_cuis)
        per_encounter.append({"encounter_id": enc.encounter_id, **scores})
        all_tp += len(gold_cuis & pred_cuis)
        all_gold += len(gold_cuis)
        all_pred += len(pred_cuis)

    micro_p = all_tp / all_pred if all_pred else 0.0
    micro_r = all_tp / all_gold if all_gold else 0.0
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) else 0.0

    return {
        "omitted": False,
        "extractor_name": extractor.name,
        "extractor_label": extractor.label,
        "per_encounter": per_encounter,
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": micro_f1,
        "n_gold_cuis_total": all_gold,
        "n_pred_cuis_total": all_pred,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("baseline", "full"), default="baseline")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA,
        help="path to aci-bench-repo/src/data (from fetch.py)",
    )
    parser.add_argument("--dry-run", action="store_true", help="smoke: stop after 5 encounters")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="max encounters (omit for full 90); smaller values are DEV ONLY per rules.md §6",
    )
    args = parser.parse_args(argv)

    if not args.data_root.exists():
        print(
            f"[run] data root not found: {args.data_root}\n"
            f"      run `python -m eval.aci_bench.fetch` first.",
            file=sys.stderr,
        )
        return 2

    extractor = build_extractor()
    print(f"[run] MEDCON tier: {extractor.name} — {extractor.label}")

    encounters = list(iter_all_test_encounters(args.data_root))
    print(f"[run] loaded {len(encounters)} encounters from data_root")

    limit = 5 if args.dry_run else args.limit
    encounters_used = encounters[:limit] if limit is not None else encounters

    preds = _run_variant(args.variant, encounters_used, limit=None)

    # ROUGE / BERTScore would be computed by the official ACI-Bench eval
    # script here — deferred to a follow-up commit that imports
    # `aci-bench-repo/evaluation/*.py`. See LIMITATIONS.md.
    medcon = _score_medcon(extractor, encounters_used, preds)

    if args.dry_run:
        print(f"[run] dry-run: {len(preds)} predictions, MEDCON micro-f1={medcon.get('micro_f1')}")
        return 0

    ts = dt.datetime.now(tz=dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = REPORTS_ROOT / ts / "aci_bench" / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "predictions.jsonl").write_text(
        "\n".join(json.dumps(asdict(p)) for p in preds) + "\n", encoding="utf-8"
    )
    metrics = {
        "variant": args.variant,
        "n_encounters": len(preds),
        "is_slice": limit is not None,
        "active_tier": {
            "name": extractor.name,
            "label": extractor.label,
            "semantic_groups": sorted(extractor.semantic_groups),
        },
        "rouge_1": None,
        "rouge_2": None,
        "rouge_l": None,
        "bertscore_division": None,
        "medcon": medcon,
        "note": (
            "ROUGE + BERTScore computation deferred — imports the OFFICIAL ACI-Bench "
            "evaluator in a follow-up commit. MEDCON is computed here via the active "
            "ConceptExtractor tier."
        ),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    (out_dir / "LIMITATIONS.md").write_text(
        _active_tier_header(extractor) + _load_limitations(), encoding="utf-8"
    )
    print(f"[run] wrote {len(preds)} predictions to {out_dir}")
    print(f"[run] active tier: {extractor.name} ({extractor.label})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
