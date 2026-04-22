"""Unit tests for per-branch projections."""

from __future__ import annotations

from src.differential.engine import rank_branches
from src.differential.lr_table import load_lr_table
from src.differential.projection import NodeState, project_branches
from tests.fixtures.loader import LR_TABLE_PATH, load_mid_case_claims


def test_projections_cover_all_branches() -> None:
    table = load_lr_table(LR_TABLE_PATH)
    claims = load_mid_case_claims()
    ranking = rank_branches(claims, table)
    projections = project_branches(claims, table, ranking)
    assert {p.branch for p in projections} == {"cardiac", "pulmonary", "msk", "gi"}


def test_cardiac_nodes_include_evidence_present_for_exertion() -> None:
    table = load_lr_table(LR_TABLE_PATH)
    claims = load_mid_case_claims()
    ranking = rank_branches(claims, table)
    projections = project_branches(claims, table, ranking)
    cardiac = next(p for p in projections if p.branch == "cardiac")
    exertion_nodes = [n for n in cardiac.nodes if n.predicate_path == "aggravating_factor=exertion"]
    assert exertion_nodes
    assert exertion_nodes[0].state == NodeState.EVIDENCE_PRESENT
    assert exertion_nodes[0].matched_claim_id == "c01"


def test_unasked_nodes_exist_for_unseen_features() -> None:
    """UNASKED may match a claim if the claim is absent-polarity and the row
    has no LR- (i.e. we know the feature is absent but the row can't score it).
    Most UNASKED nodes will have no matched claim — feature simply hasn't come up.
    """
    table = load_lr_table(LR_TABLE_PATH)
    claims = load_mid_case_claims()
    ranking = rank_branches(claims, table)
    projections = project_branches(claims, table, ranking)
    cardiac = next(p for p in projections if p.branch == "cardiac")
    unasked = [n for n in cardiac.nodes if n.state == NodeState.UNASKED]
    assert unasked
    # The majority are genuinely un-probed features (no matching claim at all).
    genuinely_unprobed = [n for n in unasked if n.matched_claim_id is None]
    assert len(genuinely_unprobed) >= len(unasked) // 2
    # UNASKED nodes never contribute to the score.
    assert all(n.applied_log_lr is None for n in unasked)


def test_relevant_claims_subset_of_active_claims() -> None:
    table = load_lr_table(LR_TABLE_PATH)
    claims = load_mid_case_claims()
    ranking = rank_branches(claims, table)
    projections = project_branches(claims, table, ranking)
    all_ids = {c.claim_id for c in claims}
    for p in projections:
        assert set(p.relevant_claim_ids).issubset(all_ids)
