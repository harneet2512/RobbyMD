"""Tests for protected token budget (Layer 6)."""
from __future__ import annotations

from eval.longmemeval.evidence_verifier import ClassifiedEvidence, EvidenceType
from eval.longmemeval.token_budget import (
    BudgetAllocation,
    allocate_budget,
    apply_budget,
    estimate_tokens,
)
from src.substrate.schema import Claim, ClaimStatus


def _ev(
    claim_id: str,
    value: str,
    etype: EvidenceType,
    fused: float = 0.5,
) -> ClassifiedEvidence:
    claim = Claim(
        claim_id=claim_id,
        session_id="test",
        subject="user",
        predicate="onset",
        value=value,
        value_normalised=None,
        confidence=0.9,
        source_turn_id="tu_test",
        status=ClaimStatus.ACTIVE,
        created_ts=1000,
    )
    return ClassifiedEvidence(
        claim=claim,
        fused_score=fused,
        evidence_type=etype,
        confidence=0.8,
        conflict_with=None,
        reason="test",
    )


class TestEstimateTokens:
    def test_roughly_4_chars_per_token(self) -> None:
        assert estimate_tokens("hello world") == 2  # 11 chars / 4 = 2

    def test_empty_string_returns_1(self) -> None:
        assert estimate_tokens("") == 1

    def test_long_string(self) -> None:
        text = "a" * 400
        assert estimate_tokens(text) == 100


class TestAllocateBudget:
    def test_direct_goes_to_must_keep(self) -> None:
        ev = _ev("c1", "direct fact", EvidenceType.DIRECT)
        alloc = allocate_budget([ev])
        assert len(alloc.must_keep) == 1
        assert alloc.must_keep[0].claim.claim_id == "c1"

    def test_conflict_goes_to_must_keep(self) -> None:
        ev = _ev("c1", "conflict fact", EvidenceType.CONFLICT)
        alloc = allocate_budget([ev])
        assert len(alloc.must_keep) == 1

    def test_supporting_goes_to_compressible(self) -> None:
        ev = _ev("c1", "supporting fact", EvidenceType.SUPPORTING)
        alloc = allocate_budget([ev])
        assert len(alloc.compressible) == 1

    def test_background_goes_to_droppable(self) -> None:
        ev = _ev("c1", "background fact", EvidenceType.BACKGROUND)
        alloc = allocate_budget([ev])
        assert len(alloc.droppable) == 1

    def test_overflow_when_must_keep_exceeds_budget(self) -> None:
        ev = _ev("c1", "x" * 400, EvidenceType.DIRECT)
        alloc = allocate_budget([ev], budget_tokens=10)
        assert alloc.overflow is True


class TestApplyBudget:
    def test_direct_never_dropped(self) -> None:
        direct = _ev("c1", "important direct fact", EvidenceType.DIRECT)
        bg = _ev("c2", "background noise", EvidenceType.BACKGROUND)
        alloc = allocate_budget([direct, bg], budget_tokens=20)
        result = apply_budget(alloc)
        kept_ids = {e.claim.claim_id for e in result.evidence}
        assert "c1" in kept_ids

    def test_background_dropped_first(self) -> None:
        direct = _ev("c1", "fact", EvidenceType.DIRECT)
        bg = _ev("c2", "a" * 400, EvidenceType.BACKGROUND)
        alloc = allocate_budget([direct, bg], budget_tokens=20)
        result = apply_budget(alloc)
        assert "c2" in result.dropped_claim_ids

    def test_budget_expansion_when_must_keep_overflows(self) -> None:
        direct = _ev("c1", "x" * 400, EvidenceType.DIRECT)
        alloc = allocate_budget([direct], budget_tokens=10)
        result = apply_budget(alloc)
        assert result.retried is True
        assert result.budget_tokens > 10

    def test_everything_fits_when_under_budget(self) -> None:
        items = [
            _ev("c1", "short", EvidenceType.DIRECT),
            _ev("c2", "also short", EvidenceType.SUPPORTING),
            _ev("c3", "tiny", EvidenceType.BACKGROUND),
        ]
        alloc = allocate_budget(items, budget_tokens=2000)
        result = apply_budget(alloc)
        assert len(result.dropped_claim_ids) == 0
        assert len(result.evidence) == 3

    def test_claim_ids_preserved(self) -> None:
        items = [
            _ev("c1", "value", EvidenceType.DIRECT),
            _ev("c2", "value2", EvidenceType.SUPPORTING),
        ]
        alloc = allocate_budget(items, budget_tokens=2000)
        result = apply_budget(alloc)
        kept_ids = {e.claim.claim_id for e in result.evidence}
        assert "c1" in kept_ids
        assert "c2" in kept_ids
