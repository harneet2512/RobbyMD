"""Unit tests for the deterministic LR-weighted engine."""

from __future__ import annotations

import math

from src.differential.engine import rank_branches
from src.differential.lr_table import load_lr_table
from src.differential.types import ActiveClaim
from tests.fixtures.loader import LR_TABLE_PATH, load_mid_case_claims


def test_ranking_produces_all_four_branches() -> None:
    table = load_lr_table(LR_TABLE_PATH)
    claims = load_mid_case_claims()
    ranking = rank_branches(claims, table)
    assert {s.branch for s in ranking.scores} == {"cardiac", "pulmonary", "msk", "gi"}
    # Posteriors form a distribution.
    total = sum(s.posterior for s in ranking.scores)
    assert math.isclose(total, 1.0, abs_tol=1e-9)


def test_mid_case_top1_is_cardiac() -> None:
    """With exertional + left-arm + diaphoresis + prior CAD + smoker → cardiac leads."""
    table = load_lr_table(LR_TABLE_PATH)
    claims = load_mid_case_claims()
    ranking = rank_branches(claims, table)
    assert ranking.scores[0].branch == "cardiac"


def test_empty_claims_gives_uniform_posterior() -> None:
    table = load_lr_table(LR_TABLE_PATH)
    ranking = rank_branches([], table)
    posteriors = [s.posterior for s in ranking.scores]
    # With zero log-scores, softmax → uniform.
    assert all(math.isclose(p, 0.25, abs_tol=1e-9) for p in posteriors)


def test_audit_trail_reproduces_log_score() -> None:
    """The sum of applied log-LRs per branch must equal that branch's log_score."""
    table = load_lr_table(LR_TABLE_PATH)
    claims = load_mid_case_claims()
    ranking = rank_branches(claims, table)
    for s in ranking.scores:
        recomputed = sum(a.log_lr for a in s.applied)
        assert math.isclose(recomputed, s.log_score, abs_tol=1e-9)


def test_iteration_order_does_not_affect_output() -> None:
    table = load_lr_table(LR_TABLE_PATH)
    claims = load_mid_case_claims()
    shuffled: list[ActiveClaim] = list(reversed(claims))
    a = rank_branches(claims, table)
    b = rank_branches(shuffled, table)
    assert tuple(s.branch for s in a.scores) == tuple(s.branch for s in b.scores)
    for sa, sb in zip(a.scores, b.scores, strict=True):
        assert math.isclose(sa.log_score, sb.log_score, abs_tol=1e-12)
        assert math.isclose(sa.posterior, sb.posterior, abs_tol=1e-12)


def test_rule_out_feature_depresses_cardiac() -> None:
    """Pleuritic-pain present has LR+ 0.2 on cardiac — a rule-out row.

    With only pleuritic pain present, cardiac should rank below msk/pulmonary.
    """
    table = load_lr_table(LR_TABLE_PATH)
    claims = [
        ActiveClaim(
            claim_id="pp1",
            predicate_path="aggravating_factor=inspiration",
            polarity=True,
            source_turn_id="t1",
        )
    ]
    ranking = rank_branches(claims, table)
    cardiac = ranking.by_branch("cardiac")
    assert cardiac is not None
    # Cardiac log-score should be negative (0.2 < 1 → log < 0) plus pleuritic is shared with msk and pulmonary.
    assert cardiac.log_score < 0
    # The top-1 branch should not be cardiac on pleuritic-pain alone.
    assert ranking.scores[0].branch != "cardiac"


def test_ranking_sorted_posterior_desc_with_branch_tiebreak() -> None:
    table = load_lr_table(LR_TABLE_PATH)
    claims = load_mid_case_claims()
    ranking = rank_branches(claims, table)
    posteriors = [s.posterior for s in ranking.scores]
    assert posteriors == sorted(posteriors, reverse=True)


def test_heart_score_high_dominates_cardiac() -> None:
    """HEART 7-10 LR+ 13 should drive cardiac posterior above 0.6 by itself."""
    table = load_lr_table(LR_TABLE_PATH)
    claims = [
        ActiveClaim(
            claim_id="h1",
            predicate_path="risk_factor=heart_score_high",
            polarity=True,
            source_turn_id="t1",
        )
    ]
    ranking = rank_branches(claims, table)
    assert ranking.scores[0].branch == "cardiac"
    assert ranking.scores[0].posterior > 0.6
