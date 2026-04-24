"""UNIT tests of individual pipeline stages with synthetic Claim objects.

NOT end-to-end gold-fact survival through the real ingestion pipeline.
For true E2E tracking with deterministic mocks and stage-by-stage gold
instrumentation, see ``tests/e2e/test_adversarial_gold_survival.py``.

These are still valuable regression tests that catch if:
- the verifier drops all evidence (Bug 1, found 2026-04-24)
- the token budget truncates the gold claim
- the confidence threshold kills the correct answer
- the structured bundle format loses critical information
"""
from __future__ import annotations

import sqlite3

import pytest

from eval.longmemeval.context import (
    ContextBundle,
    RetrievedEvidence,
    build_longmemeval_context_v2,
    format_structured_bundle,
)
from eval.longmemeval.evidence_verifier import (
    ClassifiedEvidence,
    EvidenceSufficiency,
    EvidenceType,
    classify_evidence,
    filter_evidence,
)
from eval.longmemeval.question_router import classify_question
from eval.longmemeval.token_budget import allocate_budget, apply_budget
from src.substrate.schema import Claim, ClaimStatus


def _claim(claim_id: str, subject: str, predicate: str, value: str) -> Claim:
    return Claim(
        claim_id=claim_id,
        session_id="test",
        subject=subject,
        predicate=predicate,
        value=value,
        value_normalised=None,
        confidence=0.9,
        source_turn_id="tu_test",
        status=ClaimStatus.ACTIVE,
        created_ts=1000,
    )


class TestGoldClaimSurvivesVerifier:
    """The gold answer-bearing claim must NOT be classified as IRRELEVANT."""

    def test_gold_claim_with_positive_fused_score_not_irrelevant(self) -> None:
        gold = _claim("gold", "user", "user_fact", "Business Administration degree")
        classified = classify_evidence(
            "What degree did the user graduate with?",
            "information_extraction",
            [(gold, 0.048)],  # typical RRF fused score
        )
        assert len(classified) == 1
        assert classified[0].evidence_type != EvidenceType.IRRELEVANT

    def test_gold_claim_survives_filter(self) -> None:
        gold = _claim("gold", "user", "user_fact", "Business Administration degree")
        noise = _claim("noise", "user", "user_goal", "completely unrelated goal")
        classified = classify_evidence(
            "What degree?",
            "information_extraction",
            [(gold, 0.048), (noise, 0.040)],
        )
        kept = filter_evidence(classified)
        gold_kept = [e for e in kept if e.claim.claim_id == "gold"]
        assert len(gold_kept) == 1, "gold claim was dropped by filter_evidence"

    def test_gold_claim_with_zero_fused_score_is_irrelevant(self) -> None:
        claim = _claim("c1", "user", "user_fact", "totally unrelated value")
        classified = classify_evidence(
            "What degree?",
            "information_extraction",
            [(claim, 0.0)],
        )
        assert classified[0].evidence_type == EvidenceType.IRRELEVANT


class TestGoldClaimSurvivesBudget:
    """The gold claim (DIRECT) must never be dropped by the token budget."""

    def test_direct_evidence_never_dropped(self) -> None:
        gold = ClassifiedEvidence(
            claim=_claim("gold", "user", "user_fact", "the answer is 42"),
            fused_score=0.048,
            evidence_type=EvidenceType.DIRECT,
            confidence=0.85,
            conflict_with=None,
            reason="test",
        )
        background = ClassifiedEvidence(
            claim=_claim("bg", "user", "user_goal", "x" * 500),
            fused_score=0.030,
            evidence_type=EvidenceType.BACKGROUND,
            confidence=0.3,
            conflict_with=None,
            reason="test",
        )
        alloc = allocate_budget([gold, background], budget_tokens=10)
        result = apply_budget(alloc)
        gold_kept = [e for e in result.evidence if e.claim.claim_id == "gold"]
        assert len(gold_kept) == 1, "DIRECT evidence was dropped by budget"

    def test_conflict_evidence_never_dropped(self) -> None:
        conflict = ClassifiedEvidence(
            claim=_claim("conflict", "user", "user_fact", "conflicting value"),
            fused_score=0.040,
            evidence_type=EvidenceType.CONFLICT,
            confidence=0.8,
            conflict_with="other_id",
            reason="test",
        )
        alloc = allocate_budget([conflict], budget_tokens=10)
        result = apply_budget(alloc)
        assert len(result.evidence) == 1, "CONFLICT evidence was dropped"


class TestAbstentionBehavior:
    """Abstention based on verifier-assessed evidence sufficiency."""

    def test_direct_evidence_means_sufficient(self) -> None:
        gold = _claim("gold", "user", "user_fact", "answer value")
        classified = classify_evidence(
            "What is the answer?",
            "information_extraction",
            [(gold, 0.048)],
        )
        sufficiency = EvidenceSufficiency.assess(classified)
        assert sufficiency == EvidenceSufficiency.SUFFICIENT
        assert not EvidenceSufficiency.should_abstain(sufficiency)

    def test_empty_evidence_means_insufficient(self) -> None:
        sufficiency = EvidenceSufficiency.assess([])
        assert sufficiency == EvidenceSufficiency.INSUFFICIENT
        assert EvidenceSufficiency.should_abstain(sufficiency)

    def test_only_background_means_insufficient(self) -> None:
        bg = ClassifiedEvidence(
            claim=_claim("bg", "user", "user_fact", "background noise"),
            fused_score=0.0,
            evidence_type=EvidenceType.BACKGROUND,
            confidence=0.3,
            conflict_with=None,
            reason="test",
        )
        sufficiency = EvidenceSufficiency.assess([bg])
        assert sufficiency == EvidenceSufficiency.INSUFFICIENT

    def test_conflict_without_direct_means_conflicted(self) -> None:
        c1 = ClassifiedEvidence(
            claim=_claim("c1", "user", "user_fact", "value A"),
            fused_score=0.05,
            evidence_type=EvidenceType.CONFLICT,
            confidence=0.8,
            conflict_with="c2",
            reason="test",
        )
        sufficiency = EvidenceSufficiency.assess([c1])
        assert sufficiency == EvidenceSufficiency.CONFLICTED
        assert not EvidenceSufficiency.should_abstain(sufficiency)

    def test_supporting_only_means_marginal(self) -> None:
        s = ClassifiedEvidence(
            claim=_claim("s1", "user", "user_fact", "some context"),
            fused_score=0.04,
            evidence_type=EvidenceType.SUPPORTING,
            confidence=0.5,
            conflict_with=None,
            reason="test",
        )
        sufficiency = EvidenceSufficiency.assess([s])
        assert sufficiency == EvidenceSufficiency.MARGINAL
        assert not EvidenceSufficiency.should_abstain(sufficiency)


class TestStructuredBundleFormat:
    """The structured bundle must have the required sections."""

    def test_has_direct_evidence_section(self) -> None:
        direct = ClassifiedEvidence(
            claim=_claim("c1", "user", "user_fact", "answer"),
            fused_score=0.05,
            evidence_type=EvidenceType.DIRECT,
            confidence=0.85,
            conflict_with=None,
            reason="test",
        )
        bundle = ContextBundle(
            question_id="q1",
            question="What?",
            question_type="information_extraction",
            query_variants=("what",),
            evidence=(),
            conflict_notes=(),
            retrieval_confidence=0.5,
            provenance={"retrieval_mode": "current_truth"},
        )
        text = format_structured_bundle(bundle, [direct])
        assert "DIRECT_EVIDENCE" in text

    def test_has_metadata_section(self) -> None:
        ev = ClassifiedEvidence(
            claim=_claim("c1", "user", "user_fact", "value"),
            fused_score=0.05,
            evidence_type=EvidenceType.SUPPORTING,
            confidence=0.5,
            conflict_with=None,
            reason="test",
        )
        bundle = ContextBundle(
            question_id="q1",
            question="What?",
            question_type="information_extraction",
            query_variants=("what",),
            evidence=(),
            conflict_notes=(),
            retrieval_confidence=0.5,
            provenance={"retrieval_mode": "current_truth"},
        )
        text = format_structured_bundle(bundle, [ev])
        assert "METADATA" in text

    def test_claim_values_preserved_in_bundle(self) -> None:
        direct = ClassifiedEvidence(
            claim=_claim("c1", "user", "user_fact", "Business Administration"),
            fused_score=0.05,
            evidence_type=EvidenceType.DIRECT,
            confidence=0.85,
            conflict_with=None,
            reason="test",
        )
        bundle = ContextBundle(
            question_id="q1",
            question="What degree?",
            question_type="information_extraction",
            query_variants=("degree",),
            evidence=(),
            conflict_notes=(),
            retrieval_confidence=0.5,
            provenance={"retrieval_mode": "current_truth"},
        )
        text = format_structured_bundle(bundle, [direct])
        assert "Business Administration" in text


class TestRouterAffectsRetrieval:
    """The router must produce different strategies for different question types."""

    def test_knowledge_update_includes_superseded(self) -> None:
        strategy = classify_question("What changed?", "knowledge_update")
        assert strategy.include_superseded is True
        assert strategy.retrieval_mode.value == "changed_truth"

    def test_information_extraction_excludes_superseded(self) -> None:
        strategy = classify_question("What is the name?", "information_extraction")
        assert strategy.include_superseded is False
        assert strategy.retrieval_mode.value == "current_truth"

    def test_different_types_produce_different_weights(self) -> None:
        ie = classify_question("", "information_extraction")
        ku = classify_question("", "knowledge_update")
        assert ie.weights != ku.weights
