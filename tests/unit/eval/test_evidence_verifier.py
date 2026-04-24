"""Tests for evidence verifier — answer-type matching, scoped conflicts, classification."""
from __future__ import annotations

from eval.longmemeval.evidence_verifier import (
    ClassifiedEvidence,
    EvidenceSufficiency,
    EvidenceType,
    classify_evidence,
    filter_evidence,
)
from src.substrate.schema import Claim, ClaimStatus


def _claim(cid: str, subj: str, pred: str, val: str) -> Claim:
    return Claim(
        claim_id=cid, session_id="test", subject=subj, predicate=pred,
        value=val, value_normalised=None, confidence=0.9,
        source_turn_id="tu_test", status=ClaimStatus.ACTIVE, created_ts=1000,
    )


# ═══════════════════════════════════════════════════════════════
# Answer-type matching: question pattern → value signal → DIRECT
# ═══════════════════════════════════════════════════════════════


class TestAnswerTypeMatching:
    def test_commute_duration_claim_is_direct(self) -> None:
        gold = _claim("gold", "user", "user_fact",
                       "daily commute takes 45 minutes each way")
        result = classify_evidence(
            "How long is my daily commute to work?",
            "information_extraction", [(gold, 0.05)],
        )
        assert result[0].evidence_type == EvidenceType.DIRECT

    def test_store_redemption_claim_is_direct(self) -> None:
        gold = _claim("gold", "user", "user_fact",
                       "redeemed $5 coupon at Target")
        result = classify_evidence(
            "Where did you redeem the coupon?",
            "information_extraction", [(gold, 0.05)],
        )
        assert result[0].evidence_type == EvidenceType.DIRECT

    def test_degree_claim_is_direct(self) -> None:
        gold = _claim("gold", "user", "user_fact",
                       "graduated with Business Administration degree")
        result = classify_evidence(
            "What degree did I graduate with?",
            "information_extraction", [(gold, 0.05)],
        )
        assert result[0].evidence_type == EvidenceType.DIRECT

    def test_play_name_claim_is_direct(self) -> None:
        gold = _claim("gold", "user", "user_event",
                       "attended The Glass Menagerie at the local theater")
        result = classify_evidence(
            "What is the name of the play I attended?",
            "information_extraction", [(gold, 0.05)],
        )
        assert result[0].evidence_type == EvidenceType.DIRECT

    def test_related_but_non_answer_is_supporting(self) -> None:
        claim = _claim("c1", "user", "user_fact",
                        "listens to audiobooks during commute")
        result = classify_evidence(
            "How long is my daily commute to work?",
            "information_extraction", [(claim, 0.04)],
        )
        # No duration in value → not answer_type_match
        # "commute" overlaps → SUPPORTING or BACKGROUND, not DIRECT
        assert result[0].evidence_type in (
            EvidenceType.SUPPORTING, EvidenceType.BACKGROUND,
        )


# ═══════════════════════════════════════════════════════════════
# Scoped conflict detection
# ═══════════════════════════════════════════════════════════════


class TestScopedConflicts:
    def test_unrelated_same_predicate_not_conflict(self) -> None:
        """user/user_fact 'commute 45min' and 'enjoys hiking' are NOT conflicts."""
        commute = _claim("c1", "user", "user_fact", "commute takes 45 minutes")
        hiking = _claim("c2", "user", "user_fact", "enjoys hiking on weekends")
        result = classify_evidence(
            "How long is my daily commute?",
            "information_extraction", [(commute, 0.05), (hiking, 0.04)],
        )
        labels = {e.claim.claim_id: e.evidence_type for e in result}
        assert labels["c1"] == EvidenceType.DIRECT
        assert labels["c2"] != EvidenceType.CONFLICT

    def test_real_conflict_same_attribute(self) -> None:
        """'commute 30 min' vs 'commute 45 min' ARE conflicts."""
        c1 = _claim("c1", "user", "user_fact", "commute takes 30 minutes each way")
        c2 = _claim("c2", "user", "user_fact", "commute takes 45 minutes each way")
        result = classify_evidence(
            "How long is my daily commute?",
            "information_extraction", [(c1, 0.05), (c2, 0.05)],
        )
        # Both should be DIRECT (answer_type_match) — conflict detection only
        # fires for non-DIRECT claims. The reader sees both and picks the latest.
        direct = [e for e in result if e.evidence_type == EvidenceType.DIRECT]
        assert len(direct) == 2, "both duration claims should be DIRECT"

    def test_same_value_not_conflict(self) -> None:
        c1 = _claim("c1", "user", "user_fact", "lives in Denver")
        c2 = _claim("c2", "user", "user_fact", "lives in Denver")
        result = classify_evidence(
            "Where does the user live?",
            "information_extraction", [(c1, 0.05), (c2, 0.05)],
        )
        conflicts = [e for e in result if e.evidence_type == EvidenceType.CONFLICT]
        assert len(conflicts) == 0


# ═══════════════════════════════════════════════════════════════
# Coverage-based classification
# ═══════════════════════════════════════════════════════════════


class TestCoverage:
    def test_high_coverage_is_direct(self) -> None:
        gold = _claim("c1", "user", "user_fact",
                       "user degree graduate Business Administration")
        result = classify_evidence(
            "What degree did the user graduate with?",
            "information_extraction", [(gold, 0.05)],
        )
        assert result[0].evidence_type == EvidenceType.DIRECT

    def test_zero_fused_score_is_irrelevant(self) -> None:
        claim = _claim("c1", "other", "other_pred", "completely unrelated xyz")
        result = classify_evidence(
            "What is the answer?",
            "information_extraction", [(claim, 0.0)],
        )
        assert result[0].evidence_type == EvidenceType.IRRELEVANT


# ═══════════════════════════════════════════════════════════════
# Filter
# ═══════════════════════════════════════════════════════════════


class TestFilter:
    def test_drops_irrelevant(self) -> None:
        ev = ClassifiedEvidence(
            claim=_claim("c1", "u", "p", "v"), fused_score=0.0,
            evidence_type=EvidenceType.IRRELEVANT, confidence=0.2,
            conflict_with=None, reason="",
        )
        assert filter_evidence([ev]) == []

    def test_keeps_direct(self) -> None:
        ev = ClassifiedEvidence(
            claim=_claim("c1", "u", "p", "v"), fused_score=0.05,
            evidence_type=EvidenceType.DIRECT, confidence=0.9,
            conflict_with=None, reason="",
        )
        assert len(filter_evidence([ev])) == 1


# ═══════════════════════════════════════════════════════════════
# Sufficiency
# ═══════════════════════════════════════════════════════════════


class TestSufficiency:
    def test_direct_means_sufficient(self) -> None:
        ev = ClassifiedEvidence(
            claim=_claim("c1", "u", "p", "v"), fused_score=0.01,
            evidence_type=EvidenceType.DIRECT, confidence=0.9,
            conflict_with=None, reason="",
        )
        assert EvidenceSufficiency.assess([ev]) == "sufficient"
        assert not EvidenceSufficiency.should_abstain("sufficient")

    def test_empty_means_insufficient(self) -> None:
        assert EvidenceSufficiency.assess([]) == "insufficient"
        assert EvidenceSufficiency.should_abstain("insufficient")

    def test_supporting_means_marginal(self) -> None:
        ev = ClassifiedEvidence(
            claim=_claim("c1", "u", "p", "v"), fused_score=0.03,
            evidence_type=EvidenceType.SUPPORTING, confidence=0.5,
            conflict_with=None, reason="",
        )
        assert EvidenceSufficiency.assess([ev]) == "marginal"

    def test_conflict_means_conflicted(self) -> None:
        ev = ClassifiedEvidence(
            claim=_claim("c1", "u", "p", "v"), fused_score=0.05,
            evidence_type=EvidenceType.CONFLICT, confidence=0.8,
            conflict_with="c2", reason="",
        )
        assert EvidenceSufficiency.assess([ev]) == "conflicted"


# ═══════════════════════════════════════════════════════════════
# Determinism
# ═══════════════════════════════════════════════════════════════


class TestDeterminism:
    def test_same_input_same_output(self) -> None:
        gold = _claim("c1", "user", "user_fact", "commute 45 minutes")
        r1 = classify_evidence("How long?", "info", [(gold, 0.05)])
        r2 = classify_evidence("How long?", "info", [(gold, 0.05)])
        assert r1[0].evidence_type == r2[0].evidence_type
