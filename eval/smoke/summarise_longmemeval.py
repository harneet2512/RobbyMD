"""Summarise LongMemEval smoke results per-category + aggregate.

Reads a results.json written by eval/smoke/run_smoke.py and produces:
- per-category accuracy per arm (baseline, substrate)
- per-category delta (substrate − baseline)
- aggregate accuracy per arm + aggregate delta
- knowledge-update delta called out separately (substrate should win here)

Deviation labels (reader model + version, dataset version, seed) are pulled
from config.json sitting alongside results.json in the same dir and echoed
at the top of the output — every number this script prints is attributed.

Usage
-----
    python eval/smoke/summarise_longmemeval.py \
        --results eval/longmemeval/results/20260423_postmerge_lme_stratified_<UTC>_seed42
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

# Allow `python eval/smoke/summarise_longmemeval.py` from repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _load_case_id_to_question_type() -> dict[str, str]:
    """Join key for per-category bucketing.

    CaseResult doesn't carry question_type (would inflate every smoke
    results.json); we re-read the dataset once and build a case_id →
    question_type map. Prefers the cleaned dataset.
    """
    from eval.smoke.run_smoke import _resolve_longmemeval_dataset
    import json as _json

    p = _resolve_longmemeval_dataset()
    if not p.is_file():
        return {}
    data = _json.loads(p.read_text(encoding="utf-8"))
    return {str(o["question_id"]): str(o.get("question_type", "")) for o in data}


def _bucket(rows: list[dict], arm: str, qid_to_type: dict[str, str]) -> dict[str, list[float]]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r.get("variant") != arm:
            continue
        # LongMemEval judge scores land on substrate_score for substrate rows
        # and baseline_score for baseline rows. Unify under "score" here.
        score = r.get("substrate_score") if arm == "substrate" else r.get("baseline_score")
        # Bucket key from the joined map (CaseResult doesn't carry the type itself).
        cat = qid_to_type.get(str(r.get("case_id", "")), "unknown")
        if score is not None:
            buckets[cat].append(float(score))
    return buckets


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="Results directory containing results.json + config.json")
    ns = ap.parse_args(argv)

    rdir = Path(ns.results)
    rpath = rdir / "results.json"
    cpath = rdir / "config.json"
    if not rpath.is_file():
        print(f"ERROR: {rpath} not found")
        return 1

    cfg = json.loads(cpath.read_text(encoding="utf-8")) if cpath.is_file() else {}
    rows = json.loads(rpath.read_text(encoding="utf-8"))

    print(f"=== LongMemEval smoke summary — {rdir.name} ===")
    print(f"reader: {cfg.get('readers', ['?'])}   seed: {cfg.get('seed', '?')}   "
          f"stratified: {cfg.get('stratified', '?')}   n_cases: {cfg.get('n_cases', '?')}")
    print()

    qid_to_type = _load_case_id_to_question_type()
    base_buckets = _bucket(rows, "baseline", qid_to_type)
    sub_buckets = _bucket(rows, "substrate", qid_to_type)
    all_cats = sorted(set(base_buckets) | set(sub_buckets))

    def _mean(x: list[float]) -> float:
        return statistics.fmean(x) if x else 0.0

    print(f"  {'category':<30} {'n':>4}  {'baseline':>10}  {'substrate':>10}  {'delta':>10}")
    total_base, total_sub = 0.0, 0.0
    n_base_total = n_sub_total = 0
    for cat in all_cats:
        b = base_buckets.get(cat, [])
        s = sub_buckets.get(cat, [])
        bm, sm = _mean(b), _mean(s)
        delta = sm - bm if (b and s) else 0.0
        n = max(len(b), len(s))
        print(
            f"  {cat:<30} {n:>4}  {bm:>10.4f}  {sm:>10.4f}  {delta:>+10.4f}"
        )
        total_base += sum(b); total_sub += sum(s)
        n_base_total += len(b); n_sub_total += len(s)

    print()
    agg_b = total_base / n_base_total if n_base_total else 0.0
    agg_s = total_sub / n_sub_total if n_sub_total else 0.0
    print(f"aggregate (baseline n={n_base_total}): {agg_b:.4f}")
    print(f"aggregate (substrate n={n_sub_total}): {agg_s:.4f}")
    print(f"aggregate Δ: {agg_s - agg_b:+.4f}")
    ku_b = _mean(base_buckets.get("knowledge-update", []))
    ku_s = _mean(sub_buckets.get("knowledge-update", []))
    if ku_b or ku_s:
        print(f"knowledge-update Δ (substrate − baseline): {ku_s - ku_b:+.4f}  "
              f"(baseline {ku_b:.4f}, substrate {ku_s:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
