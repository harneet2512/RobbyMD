"""Aggregate n-seed smoke results for the Phase 1.5 multi-seed discipline.

Usage
-----
    python eval/smoke/aggregate_seeds.py --benchmark acibench \
        --pattern "eval/acibench/results/20260423_postmerge_hybrid_phase1*_seed*/results.json"

Reports, per-benchmark:
- Per-case row: {case_id, per-seed deltas + baseline/substrate scores,
  mean_delta, stdev_delta, sign_vote ("win_3_of_3" / "win_2_of_3" / ...)}
- Aggregate: mean baseline / mean substrate / mean delta across all cases
  and all seeds; per-case σ summary.
- Robust-wins, likely-wins, likely-losses per the plan's decision rule.

Intentionally minimal — no external deps beyond stdlib. Reads results.json
files written by `eval/smoke/run_smoke.py` and summarises them. LongMemEval
is supported the same way but `delta` is then an accuracy delta, not MEDCON.
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def _load_results(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _extract_seed_from_path(path: Path) -> int | None:
    """Pull the seed suffix out of a results-dir path like ..._seed43/results.json."""
    for part in reversed(path.parts):
        if "_seed" in part:
            tail = part.rsplit("_seed", 1)[-1]
            try:
                return int(tail)
            except ValueError:
                continue
    return None


def _per_case_by_seed(paths: list[Path]) -> dict[str, dict[int, dict]]:
    """Return {case_id: {seed: {baseline, substrate, delta}}}."""
    per_case: dict[str, dict[int, dict]] = defaultdict(dict)
    for p in paths:
        seed = _extract_seed_from_path(p)
        if seed is None:
            print(f"[agg] WARN seed-suffix missing from {p}; skipping", file=sys.stderr)
            continue
        rows = _load_results(p)
        for row in rows:
            if row.get("variant") != "substrate":
                # Substrate rows carry the delta on ACI-Bench; baseline rows
                # carry baseline-only scores. We only need the substrate row.
                continue
            cid = row.get("case_id")
            if cid is None:
                continue
            per_case[cid][seed] = {
                "baseline": row.get("baseline_score"),
                "substrate": row.get("substrate_score"),
                "delta": row.get("delta"),
            }
    return per_case


def _summarise(per_case: dict[str, dict[int, dict]], *, delta_gate: float = 0.0) -> dict:
    """Compute per-case σ + robust/likely wins; plus global means."""
    case_rows = []
    all_baselines: list[float] = []
    all_substrates: list[float] = []
    all_deltas: list[float] = []
    robust_wins = likely_wins = likely_losses = noisy = 0

    for cid in sorted(per_case):
        per_seed = per_case[cid]
        seeds = sorted(per_seed)
        deltas = [
            per_seed[s]["delta"]
            for s in seeds
            if per_seed[s].get("delta") is not None
        ]
        baselines = [
            per_seed[s]["baseline"]
            for s in seeds
            if per_seed[s].get("baseline") is not None
        ]
        substrates = [
            per_seed[s]["substrate"]
            for s in seeds
            if per_seed[s].get("substrate") is not None
        ]
        all_baselines.extend(baselines)
        all_substrates.extend(substrates)
        all_deltas.extend(deltas)

        mean_d = statistics.fmean(deltas) if deltas else None
        stdev_d = statistics.stdev(deltas) if len(deltas) > 1 else None
        # Sign vote across seeds: how many seeds show delta > gate?
        wins = sum(1 for d in deltas if d > delta_gate)
        losses = sum(1 for d in deltas if d < -delta_gate)
        if wins == len(deltas) and len(deltas) >= 2:
            vote = "robust_win"
            robust_wins += 1
        elif wins >= 2:
            vote = "likely_win"
            likely_wins += 1
        elif losses >= 2:
            vote = "likely_loss"
            likely_losses += 1
        else:
            vote = "mixed"
        if stdev_d is not None and stdev_d > 0.15:
            noisy += 1

        case_rows.append(
            {
                "case_id": cid,
                "seeds": seeds,
                "deltas": deltas,
                "mean_delta": mean_d,
                "stdev_delta": stdev_d,
                "sign_vote": vote,
            }
        )

    return {
        "per_case": case_rows,
        "global": {
            "n_seeds": sorted({s for d in per_case.values() for s in d}),
            "n_cases": len(per_case),
            "mean_baseline": statistics.fmean(all_baselines) if all_baselines else None,
            "mean_substrate": statistics.fmean(all_substrates) if all_substrates else None,
            "mean_delta": statistics.fmean(all_deltas) if all_deltas else None,
            "stdev_delta": (
                statistics.stdev(all_deltas) if len(all_deltas) > 1 else None
            ),
            "robust_wins": robust_wins,
            "likely_wins": likely_wins,
            "likely_losses": likely_losses,
            "noisy_cases": noisy,
        },
    }


def _format(summary: dict, *, benchmark: str) -> str:
    out: list[str] = []
    g = summary["global"]
    out.append(f"=== {benchmark} multi-seed aggregate ===")
    out.append(f"seeds: {g['n_seeds']}  cases: {g['n_cases']}")
    if g["mean_baseline"] is not None:
        out.append(
            f"mean baseline: {g['mean_baseline']:.4f}   "
            f"mean substrate: {g['mean_substrate']:.4f}   "
            f"mean Delta: {g['mean_delta']:+.4f}   "
            f"σ(Delta)global: {g['stdev_delta']:.4f}"
            if g["stdev_delta"] is not None
            else f"mean baseline: {g['mean_baseline']:.4f}   "
                 f"mean substrate: {g['mean_substrate']:.4f}   "
                 f"mean Delta: {g['mean_delta']:+.4f}   σ(Delta)global: n/a"
        )
    out.append(
        f"robust wins: {g['robust_wins']}   "
        f"likely wins (≥2/3): {g['likely_wins']}   "
        f"likely losses (≥2/3): {g['likely_losses']}   "
        f"noisy (σ>0.15): {g['noisy_cases']}"
    )
    out.append("")
    out.append("per-case:")
    out.append(
        f"  {'case_id':<30} {'Delta(42)':>8} {'Delta(43)':>8} {'Delta(44)':>8} "
        f"{'meanDelta':>8} {'σ(Delta)':>8}  vote"
    )
    for row in summary["per_case"]:
        deltas = {s: d for s, d in zip(row["seeds"], row["deltas"])}
        d42 = f"{deltas.get(42, 'na'):+.3f}" if 42 in deltas else "   na  "
        d43 = f"{deltas.get(43, 'na'):+.3f}" if 43 in deltas else "   na  "
        d44 = f"{deltas.get(44, 'na'):+.3f}" if 44 in deltas else "   na  "
        md = f"{row['mean_delta']:+.3f}" if row["mean_delta"] is not None else "   na "
        sd = f"{row['stdev_delta']:.3f}" if row["stdev_delta"] is not None else "  na "
        out.append(
            f"  {row['case_id']:<30} {d42:>8} {d43:>8} {d44:>8} "
            f"{md:>8} {sd:>8}  {row['sign_vote']}"
        )
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="acibench")
    ap.add_argument(
        "--pattern",
        required=True,
        help='glob pattern matching results.json files across seed dirs. '
        'e.g. "eval/acibench/results/*_seed*/results.json"',
    )
    ap.add_argument(
        "--delta-gate",
        type=float,
        default=0.0,
        help="Deltas > this count as wins in the per-case sign vote. Default 0.",
    )
    ap.add_argument("--json", dest="as_json", action="store_true", help="Emit JSON instead of table")
    ns = ap.parse_args(argv)

    matches = sorted(Path(p) for p in glob.glob(ns.pattern))
    if not matches:
        print(f"[agg] ERROR: no results.json matches for {ns.pattern!r}", file=sys.stderr)
        return 1
    print(f"[agg] matched {len(matches)} results.json files", file=sys.stderr)
    for m in matches:
        print(f"[agg]   {m}", file=sys.stderr)

    per_case = _per_case_by_seed(matches)
    summary = _summarise(per_case, delta_gate=ns.delta_gate)

    if ns.as_json:
        print(json.dumps(summary, indent=2))
    else:
        print(_format(summary, benchmark=ns.benchmark))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
