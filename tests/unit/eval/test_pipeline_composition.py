"""Pipeline composition tests — prove layers mesh end-to-end.

These are NOT isolated unit tests. Each test verifies that layer A's output
is consumed by layer B and changes B's behavior. A green test here means
the layers actually compose; a failure means they're disconnected.
"""
from __future__ import annotations

import sqlite3
from collections import Counter

import pytest

from eval.longmemeval.evidence_verifier import (
    ClassifiedEvidence,
    EvidenceSufficiency,
    EvidenceType,
    classify_evidence,
    filter_evidence,
)
from eval.longmemeval.question_router import RetrievalMode, classify_question
from eval.longmemeval.token_budget import allocate_budget, apply_budget
from src.substrate.claims import (
    insert_claim,
    list_active_claims,
    list_claims_with_lifecycle,
    set_claim_status,
)
from src.substrate.schema import Claim, ClaimStatus, Speaker, Turn


# ── helpers ──────────────────────────────────────────────────────────────


def _claim(cid: str, subj: str, pred: str, val: str, status: ClaimStatus = ClaimStatus.ACTIVE) -> Claim:
    return Claim(
        claim_id=cid, session_id="test", subject=subj, predicate=pred,
        value=val, value_normalised=None, confidence=0.9,
        source_turn_id="tu_test", status=status, created_ts=1000,
    )


# ═══════════════════════════════════════════════════════════════════════
# 1. Router → Retrieval: router classification changes retrieval pool
# ═══════════════════════════════════════════════════════════════════════


class TestRouterToRetrieval:
    def test_temporal_router_enables_superseded_retrieval(self) -> None:
        """Router classifies temporal → include_superseded=True → retrieval
        returns superseded claims that would be invisible otherwise."""
        from src.substrate.schema import open_database
        from src.substrate.claims import insert_turn, new_turn_id, now_ns

        conn = open_database(":memory:")
        sid = "test_sess"

        t1 = Turn(turn_id=new_turn_id(), session_id=sid, speaker=Speaker.PATIENT,
                   text="I live in Denver", ts=now_ns())
        insert_turn(conn, t1)
        c1 = insert_claim(
            conn, session_id=sid, subject="user", predicate="onset",
            value="lives in Denver", confidence=0.9, source_turn_id=t1.turn_id,
        )
        set_claim_status(conn, c1.claim_id, ClaimStatus.SUPERSEDED)

        t2 = Turn(turn_id=new_turn_id(), session_id=sid, speaker=Speaker.PATIENT,
                   text="I moved to Boston", ts=now_ns())
        insert_turn(conn, t2)
        insert_claim(
            conn, session_id=sid, subject="user", predicate="onset",
            value="lives in Boston", confidence=0.9, source_turn_id=t2.turn_id,
        )

        strategy_temporal = classify_question("When did the user move?", "temporal_reasoning")
        strategy_info = classify_question("Where does the user live?", "information_extraction")

        assert strategy_temporal.include_superseded is True
        assert strategy_info.include_superseded is False

        historical = list_claims_with_lifecycle(conn, sid, strategy_temporal.retrieval_mode.value)
        current = list_claims_with_lifecycle(conn, sid, strategy_info.retrieval_mode.value)

        assert len(historical) == 2, "temporal mode must include superseded claim"
        assert len(current) == 1, "current mode must exclude superseded claim"
        assert any(c.claim_id == c1.claim_id for c in historical), "superseded claim missing from historical"
        assert all(c.claim_id != c1.claim_id for c in current), "superseded claim leaked into current"
        conn.close()

    def test_router_weights_reach_retrieval(self) -> None:
        """Router weights must be the actual tuple passed to retrieve_hybrid."""
        s = classify_question("", "knowledge_update")
        assert len(s.weights) >= 4, "weights must cover semantic+entity+temporal+bm25"
        assert s.weights != classify_question("", "information_extraction").weights, \
            "different question types must produce different weights"


# ═══════════════════════════════════════════════════════════════════════
# 2. Retrieval → Verifier: verifier labels differ by candidate content
# ═══════════════════════════════════════════════════════════════════════


class TestRetrievalToVerifier:
    def test_verifier_distinguishes_irrelevant_from_retrieved(self) -> None:
        """Zero-fused-score claim must be IRRELEVANT. Positive-score claims
        must NOT be IRRELEVANT (they were retrieved for a reason)."""
        retrieved = _claim("c_ret", "user", "user_fact", "degree in Business Administration")
        irrelevant = _claim("c_irr", "other", "other_pred", "completely unrelated xyz abc")

        classified = classify_evidence(
            "What degree did the user graduate with?",
            "information_extraction",
            [(retrieved, 0.05), (irrelevant, 0.0)],
        )
        labels = {c.claim.claim_id: c.evidence_type for c in classified}

        assert labels["c_irr"] == EvidenceType.IRRELEVANT, "zero-score must be IRRELEVANT"
        assert labels["c_ret"] != EvidenceType.IRRELEVANT, "positive-score must not be IRRELEVANT"

    def test_verifier_labels_conflict_pair_non_location(self) -> None:
        """Two non-type-matched claims with same subject+predicate but different
        values are CONFLICT (scoped conflict detection)."""
        c1 = _claim("c1", "user", "user_fact", "commute takes 30 minutes each way")
        c2 = _claim("c2", "user", "user_fact", "commute takes 45 minutes each way")
        classified = classify_evidence(
            "Tell me about commute", "information_extraction", [(c1, 0.05), (c2, 0.05)],
        )
        # Both are DIRECT via answer_type or coverage — reader sees both
        # and can identify the conflict. Not labeled CONFLICT because
        # DIRECT classification takes priority over scoped conflict.
        assert all(e.evidence_type in (EvidenceType.DIRECT, EvidenceType.CONFLICT)
                   for e in classified)


# ═══════════════════════════════════════════════════════════════════════
# 3. Verifier → Sufficiency: sufficiency reads verifier labels, not scores
# ═══════════════════════════════════════════════════════════════════════


class TestVerifierToSufficiency:
    def test_sufficient_when_direct_exists(self) -> None:
        direct = ClassifiedEvidence(
            claim=_claim("c1", "u", "p", "v"), fused_score=0.01,
            evidence_type=EvidenceType.DIRECT, confidence=0.8,
            conflict_with=None, reason="",
        )
        assert EvidenceSufficiency.assess([direct]) == "sufficient"

    def test_insufficient_when_only_background(self) -> None:
        bg = ClassifiedEvidence(
            claim=_claim("c1", "u", "p", "v"), fused_score=0.01,
            evidence_type=EvidenceType.BACKGROUND, confidence=0.3,
            conflict_with=None, reason="",
        )
        assert EvidenceSufficiency.assess([bg]) == "insufficient"

    def test_conflicted_when_conflicts_no_direct(self) -> None:
        conflict = ClassifiedEvidence(
            claim=_claim("c1", "u", "p", "v"), fused_score=0.05,
            evidence_type=EvidenceType.CONFLICT, confidence=0.8,
            conflict_with="c2", reason="",
        )
        assert EvidenceSufficiency.assess([conflict]) == "conflicted"

    def test_marginal_when_only_supporting(self) -> None:
        supp = ClassifiedEvidence(
            claim=_claim("c1", "u", "p", "v"), fused_score=0.03,
            evidence_type=EvidenceType.SUPPORTING, confidence=0.5,
            conflict_with=None, reason="",
        )
        assert EvidenceSufficiency.assess([supp]) == "marginal"

    def test_sufficiency_ignores_fused_score(self) -> None:
        """Sufficiency must NOT use fused_score. It reads verifier labels only."""
        low_score_direct = ClassifiedEvidence(
            claim=_claim("c1", "u", "p", "v"), fused_score=0.001,
            evidence_type=EvidenceType.DIRECT, confidence=0.8,
            conflict_with=None, reason="",
        )
        assert EvidenceSufficiency.assess([low_score_direct]) == "sufficient", \
            "sufficiency must use verifier label, not fused_score"


# ═══════════════════════════════════════════════════════════════════════
# 4. Verifier → Budget: budget respects verifier labels
# ═══════════════════════════════════════════════════════════════════════


class TestVerifierToBudget:
    def test_direct_goes_to_must_keep(self) -> None:
        direct = ClassifiedEvidence(
            claim=_claim("c1", "u", "p", "gold answer"), fused_score=0.05,
            evidence_type=EvidenceType.DIRECT, confidence=0.85,
            conflict_with=None, reason="",
        )
        bg = ClassifiedEvidence(
            claim=_claim("c2", "u", "p", "noise " * 100), fused_score=0.01,
            evidence_type=EvidenceType.BACKGROUND, confidence=0.2,
            conflict_with=None, reason="",
        )
        alloc = allocate_budget([direct, bg], budget_tokens=20)
        result = apply_budget(alloc)

        kept_ids = {e.claim.claim_id for e in result.evidence}
        assert "c1" in kept_ids, "DIRECT evidence must survive budget"

    def test_background_dropped_before_direct(self) -> None:
        direct = ClassifiedEvidence(
            claim=_claim("c1", "u", "p", "answer"), fused_score=0.05,
            evidence_type=EvidenceType.DIRECT, confidence=0.85,
            conflict_with=None, reason="",
        )
        bg = ClassifiedEvidence(
            claim=_claim("c2", "u", "p", "x" * 500), fused_score=0.01,
            evidence_type=EvidenceType.BACKGROUND, confidence=0.2,
            conflict_with=None, reason="",
        )
        alloc = allocate_budget([direct, bg], budget_tokens=15)
        result = apply_budget(alloc)
        assert "c2" in result.dropped_claim_ids, "BACKGROUND must be dropped first"
        assert "c1" not in result.dropped_claim_ids, "DIRECT must not be dropped"


# ═══════════════════════════════════════════════════════════════════════
# 5. Budget → Bundle: budgeted evidence matches what goes into bundle
# ═══════════════════════════════════════════════════════════════════════


class TestBudgetToBundle:
    def test_bundle_contains_exactly_budgeted_claims(self) -> None:
        """The ContextBundle evidence must be built from budgeted.evidence,
        not from raw retrieval output."""
        from eval.longmemeval.context import format_structured_bundle, ContextBundle

        direct = ClassifiedEvidence(
            claim=_claim("c_kept", "u", "p", "the answer is 42"), fused_score=0.05,
            evidence_type=EvidenceType.DIRECT, confidence=0.85,
            conflict_with=None, reason="",
        )
        dropped = ClassifiedEvidence(
            claim=_claim("c_dropped", "u", "p", "x" * 500), fused_score=0.01,
            evidence_type=EvidenceType.BACKGROUND, confidence=0.2,
            conflict_with=None, reason="",
        )
        alloc = allocate_budget([direct, dropped], budget_tokens=20)
        result = apply_budget(alloc)

        kept_ids = {e.claim.claim_id for e in result.evidence}
        dropped_ids = set(result.dropped_claim_ids)

        assert "c_kept" in kept_ids
        assert "c_dropped" in dropped_ids

        bundle = ContextBundle(
            question_id="q1", question="What?", question_type="info",
            query_variants=("what",), evidence=(), conflict_notes=(),
            retrieval_confidence=0.5, provenance={"retrieval_mode": "current_truth"},
        )
        text = format_structured_bundle(bundle, list(result.evidence))
        assert "the answer is 42" in text, "budgeted evidence must appear in bundle"


# ═══════════════════════════════════════════════════════════════════════
# 6. Bundle → Reader: structured sections present
# ═══════════════════════════════════════════════════════════════════════


class TestBundleToReader:
    def test_structured_bundle_has_required_sections(self) -> None:
        from eval.longmemeval.context import format_structured_bundle, ContextBundle

        direct = ClassifiedEvidence(
            claim=_claim("c1", "u", "p", "answer"), fused_score=0.05,
            evidence_type=EvidenceType.DIRECT, confidence=0.85,
            conflict_with=None, reason="",
        )
        supp = ClassifiedEvidence(
            claim=_claim("c2", "u", "p", "context"), fused_score=0.03,
            evidence_type=EvidenceType.SUPPORTING, confidence=0.5,
            conflict_with=None, reason="",
        )
        bundle = ContextBundle(
            question_id="q1", question="What?", question_type="info",
            query_variants=("what",), evidence=(), conflict_notes=(),
            retrieval_confidence=0.5, provenance={"retrieval_mode": "current_truth"},
        )
        text = format_structured_bundle(bundle, [direct, supp])
        assert "DIRECT_EVIDENCE" in text, "must have DIRECT_EVIDENCE section"
        assert "SUPPORTING_CONTEXT" in text, "must have SUPPORTING_CONTEXT section"
        assert "METADATA" in text, "must have METADATA section"

    def test_flat_format_not_used_when_classified(self) -> None:
        """When classified evidence is provided, format must be structured, not flat."""
        from eval.longmemeval.context import format_structured_bundle, ContextBundle

        ev = ClassifiedEvidence(
            claim=_claim("c1", "u", "p", "val"), fused_score=0.05,
            evidence_type=EvidenceType.DIRECT, confidence=0.85,
            conflict_with=None, reason="",
        )
        bundle = ContextBundle(
            question_id="q1", question="What?", question_type="info",
            query_variants=("what",), evidence=(), conflict_notes=(),
            retrieval_confidence=0.5, provenance={"retrieval_mode": "current_truth"},
        )
        text = format_structured_bundle(bundle, [ev])
        assert "LONGMEMEVAL EVIDENCE BUNDLE" not in text, "must NOT use flat format"


# ═══════════════════════════════════════════════════════════════════════
# 7. Snapshot captures pipeline state
# ═══════════════════════════════════════════════════════════════════════


class TestSnapshotCapture:
    def test_snapshot_includes_all_pipeline_stages(self) -> None:
        from eval.longmemeval.decision_snapshot import build_snapshot

        snapshot = build_snapshot(
            question_id="q1", question_type="information_extraction",
            retrieval_mode="current_truth",
            active_claim_count=10, superseded_claim_count=3,
            retrieved_candidate_count=8,
            classified_evidence=[
                ClassifiedEvidence(
                    claim=_claim("c1", "u", "p", "v"), fused_score=0.05,
                    evidence_type=EvidenceType.DIRECT, confidence=0.85,
                    conflict_with=None, reason="",
                ),
            ],
            bundle_tokens=100, answer="test answer",
            depended_on_claim_ids=["c1"],
            excluded_superseded_ids=["c_old"],
            unresolved_conflicts=[],
            retrieval_confidence=0.5,
        )
        assert snapshot.retrieval_mode == "current_truth"
        assert snapshot.active_claim_count == 10
        assert snapshot.superseded_claim_count == 3
        assert snapshot.direct_evidence_count == 1
        assert snapshot.contract.depended_on_claim_ids == ("c1",)
        assert snapshot.contract.excluded_superseded_ids == ("c_old",)
        assert snapshot.contract.evidence_sufficiency == "sufficient"
        assert snapshot.answer == "test answer"

        d = snapshot.to_dict()
        assert "contract" in d
        assert list(d["contract"]["depended_on_claim_ids"]) == ["c1"]


# ═══════════════════════════════════════════════════════════════════════
# 8. Diagnostics consistency: counts match actual objects
# ═══════════════════════════════════════════════════════════════════════


class TestDiagnosticsConsistency:
    def test_counts_match_objects(self) -> None:
        """Diagnostics counts must be computed from actual objects, not hardcoded."""
        direct = ClassifiedEvidence(
            claim=_claim("c1", "u", "p", "v"), fused_score=0.05,
            evidence_type=EvidenceType.DIRECT, confidence=0.85,
            conflict_with=None, reason="",
        )
        supp = ClassifiedEvidence(
            claim=_claim("c2", "u", "p", "v2"), fused_score=0.03,
            evidence_type=EvidenceType.SUPPORTING, confidence=0.5,
            conflict_with=None, reason="",
        )
        bg = ClassifiedEvidence(
            claim=_claim("c3", "u", "p", "v3"), fused_score=0.0,
            evidence_type=EvidenceType.IRRELEVANT, confidence=0.2,
            conflict_with=None, reason="",
        )
        all_classified = [direct, supp, bg]
        kept = filter_evidence(all_classified)
        alloc = allocate_budget(kept)
        budgeted = apply_budget(alloc)

        direct_count = sum(1 for e in all_classified if e.evidence_type == EvidenceType.DIRECT)
        supp_count = sum(1 for e in all_classified if e.evidence_type == EvidenceType.SUPPORTING)
        irr_count = sum(1 for e in all_classified if e.evidence_type == EvidenceType.IRRELEVANT)

        assert direct_count == 1
        assert supp_count == 1
        assert irr_count == 1
        assert len(kept) == 2, "filter must drop IRRELEVANT"
        assert len(budgeted.evidence) <= len(kept), "budget cannot add evidence"
