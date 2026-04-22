"""Tests for eval/smoke/aggregate_seeds.py — multi-seed aggregation helper."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.smoke.aggregate_seeds import (
    _extract_seed_from_path,
    _per_case_by_seed,
    _summarise,
)


def _write(results_dir: Path, rows: list[dict]) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    p = results_dir / "results.json"
    p.write_text(json.dumps(rows), encoding="utf-8")
    return p


def _mk_row(case_id: str, baseline: float, substrate: float, variant: str = "substrate") -> dict:
    return {
        "case_id": case_id,
        "benchmark": "acibench",
        "reader": "qwen2.5-14b",
        "variant": variant,
        "baseline_score": baseline,
        "substrate_score": substrate,
        "delta": substrate - baseline,
        "latency_baseline_ms": None,
        "latency_substrate_ms": None,
        "tokens_used_baseline": None,
        "tokens_used_substrate": None,
        "estimated_cost": 0.0,
        "judge_reasoning": None,
        "structural_validity": {},
    }


class TestExtractSeedFromPath:
    def test_seed_suffix_at_end(self) -> None:
        p = Path("eval/acibench/results/20260423_postmerge_hybrid_phase1_abc_seed42/results.json")
        assert _extract_seed_from_path(p) == 42

    def test_missing_seed_returns_none(self) -> None:
        p = Path("eval/acibench/results/no_seed_here/results.json")
        assert _extract_seed_from_path(p) is None


class TestAggregateSeeds:
    def test_three_seed_robust_win(self, tmp_path: Path) -> None:
        p1 = _write(tmp_path / "run_seed42", [_mk_row("c1", 0.5, 0.6)])
        p2 = _write(tmp_path / "run_seed43", [_mk_row("c1", 0.5, 0.65)])
        p3 = _write(tmp_path / "run_seed44", [_mk_row("c1", 0.5, 0.58)])

        per_case = _per_case_by_seed([p1, p2, p3])
        summary = _summarise(per_case)

        assert summary["global"]["n_seeds"] == [42, 43, 44]
        assert summary["global"]["n_cases"] == 1
        assert summary["global"]["robust_wins"] == 1
        row = summary["per_case"][0]
        assert row["sign_vote"] == "robust_win"
        assert row["mean_delta"] == pytest.approx((0.10 + 0.15 + 0.08) / 3)

    def test_likely_loss_when_two_of_three_negative(self, tmp_path: Path) -> None:
        # c1 wins on seed 42, loses on 43 and 44 → likely_loss
        p1 = _write(tmp_path / "run_seed42", [_mk_row("c1", 0.6, 0.7)])
        p2 = _write(tmp_path / "run_seed43", [_mk_row("c1", 0.6, 0.5)])
        p3 = _write(tmp_path / "run_seed44", [_mk_row("c1", 0.6, 0.55)])

        per_case = _per_case_by_seed([p1, p2, p3])
        summary = _summarise(per_case)
        row = summary["per_case"][0]
        assert row["sign_vote"] == "likely_loss"
        assert summary["global"]["likely_losses"] == 1

    def test_noisy_case_flagged(self, tmp_path: Path) -> None:
        # Big per-seed swing so stdev > 0.15 threshold.
        p1 = _write(tmp_path / "run_seed42", [_mk_row("c1", 0.5, 0.7)])   # +0.2
        p2 = _write(tmp_path / "run_seed43", [_mk_row("c1", 0.5, 0.3)])   # -0.2
        p3 = _write(tmp_path / "run_seed44", [_mk_row("c1", 0.5, 0.5)])   # 0

        per_case = _per_case_by_seed([p1, p2, p3])
        summary = _summarise(per_case)
        assert summary["global"]["noisy_cases"] == 1
        # And it's mixed, not robust
        assert summary["per_case"][0]["sign_vote"] == "mixed"

    def test_baseline_rows_ignored(self, tmp_path: Path) -> None:
        # Aggregator reads only variant=="substrate" rows (the substrate row
        # carries the delta). Baseline row duplication shouldn't confuse it.
        p1 = _write(
            tmp_path / "run_seed42",
            [
                _mk_row("c1", 0.5, 0.6, variant="baseline"),
                _mk_row("c1", 0.5, 0.6, variant="substrate"),
            ],
        )
        per_case = _per_case_by_seed([p1])
        summary = _summarise(per_case)
        # Expect exactly one seed's delta recorded per case.
        assert summary["per_case"][0]["deltas"] == [pytest.approx(0.1)]
