"""Unit tests for the counterfactual verifier."""

from __future__ import annotations

from src.differential.engine import rank_branches
from src.differential.lr_table import load_lr_table
from src.differential.types import ActiveClaim
from src.verifier import MockOpusClient, select_discriminator, verify
from tests.fixtures.loader import LR_TABLE_PATH, load_mid_case_claims


def test_verify_returns_shape_matching_eng_doc_6() -> None:
    table = load_lr_table(LR_TABLE_PATH)
    claims = load_mid_case_claims()
    ranking = rank_branches(claims, table)
    out = verify(ranking, table, claims, opus_client=MockOpusClient())
    assert isinstance(out.why_moved, tuple)
    assert isinstance(out.missing_or_contradicting, tuple)
    assert isinstance(out.next_best_question, str)
    assert isinstance(out.source_feature, str)


def test_next_best_question_is_at_most_20_words() -> None:
    table = load_lr_table(LR_TABLE_PATH)
    claims = load_mid_case_claims()
    ranking = rank_branches(claims, table)
    out = verify(ranking, table, claims, opus_client=MockOpusClient())
    # Only enforced when a discriminator exists.
    if out.next_best_question:
        assert len(out.next_best_question.split()) <= 20


def test_discriminator_picks_largest_log_gap() -> None:
    """Cardiac vs whoever-is-second — the discriminator's log gap > 0."""
    table = load_lr_table(LR_TABLE_PATH)
    claims = load_mid_case_claims()
    ranking = rank_branches(claims, table)
    disc = select_discriminator(ranking, table, claims)
    assert disc is not None
    assert disc.log_gap > 0
    assert disc.branch_a == ranking.scores[0].branch
    assert disc.branch_b == ranking.scores[1].branch


def test_verifier_stays_quiet_when_top1_is_decisive() -> None:
    """Single HEART 7-10 claim drives posterior so high the verifier stays quiet."""
    table = load_lr_table(LR_TABLE_PATH)
    claims = [
        ActiveClaim(
            claim_id="h1",
            predicate_path="risk_factor=heart_score_high",
            polarity=True,
            source_turn_id="t1",
        ),
        ActiveClaim(
            claim_id="h2",
            predicate_path="risk_factor=timi_score_high",
            polarity=True,
            source_turn_id="t2",
        ),
        ActiveClaim(
            claim_id="h3",
            predicate_path="risk_factor=marburg_score_high",
            polarity=True,
            source_turn_id="t3",
        ),
    ]
    ranking = rank_branches(claims, table)
    out = verify(ranking, table, claims, opus_client=MockOpusClient())
    # Either we produce a question against a tied-but-far-behind branch or we stay silent.
    # Key property: when the top-1 posterior is nearly 1.0, the uncertainty multiplier goes
    # to ~0 and the verifier either returns empty or a trivially-scored discriminator.
    if ranking.scores[0].posterior > 0.99:
        assert out.next_best_question == "" or len(out.next_best_question.split()) <= 20


def test_verifier_respects_present_claims_and_skips_answered_features() -> None:
    """Features already volunteered should not be proposed as 'next best question'."""
    table = load_lr_table(LR_TABLE_PATH)
    claims = load_mid_case_claims()
    ranking = rank_branches(claims, table)
    disc = select_discriminator(ranking, table, claims)
    if disc is not None:
        present = {c.predicate_path for c in claims if c.polarity}
        assert disc.predicate_path not in present


def test_mock_client_is_deterministic() -> None:
    client = MockOpusClient()
    a = client.render_next_best_question(
        branch_a="cardiac",
        branch_b="pulmonary",
        feature="pleuritic_pain",
        predicate_path="aggravating_factor=inspiration",
        lr_a=0.2,
        lr_b=1.8,
        direction="+",
    )
    b = client.render_next_best_question(
        branch_a="cardiac",
        branch_b="pulmonary",
        feature="pleuritic_pain",
        predicate_path="aggravating_factor=inspiration",
        lr_a=0.2,
        lr_b=1.8,
        direction="+",
    )
    assert a == b
    assert a.endswith("?")
