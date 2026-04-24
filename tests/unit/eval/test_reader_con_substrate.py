"""Tests for substrate-aware Chain-of-Note (Layer 5)."""
from __future__ import annotations

from eval.longmemeval.reader_con import (
    SUBSTRATE_ANSWER_SYSTEM,
    SUBSTRATE_NOTE_EXTRACTION_SYSTEM,
    _format_classified_evidence_for_extraction,
)
from eval.longmemeval.evidence_verifier import ClassifiedEvidence, EvidenceType
from src.substrate.schema import Claim, ClaimStatus


def _ev(
    claim_id: str,
    value: str,
    status: ClaimStatus = ClaimStatus.ACTIVE,
    etype: EvidenceType = EvidenceType.DIRECT,
    valid_from: int | None = 1000,
) -> ClassifiedEvidence:
    claim = Claim(
        claim_id=claim_id,
        session_id="test",
        subject="user",
        predicate="user_fact",
        value=value,
        value_normalised=None,
        confidence=0.9,
        source_turn_id="tu_test",
        status=status,
        created_ts=1000,
        valid_from_ts=valid_from,
    )
    return ClassifiedEvidence(
        claim=claim,
        fused_score=0.8,
        evidence_type=etype,
        confidence=0.85,
        conflict_with=None,
        reason="test",
    )


class TestFormatClassifiedEvidence:
    def test_includes_status(self) -> None:
        ev = _ev("c1", "lives in Denver", ClaimStatus.SUPERSEDED)
        text = _format_classified_evidence_for_extraction([ev])
        assert "superseded" in text
        assert "c1" in text
        assert "Denver" in text

    def test_includes_active_status(self) -> None:
        ev = _ev("c1", "lives in Boston", ClaimStatus.ACTIVE)
        text = _format_classified_evidence_for_extraction([ev])
        assert "active" in text

    def test_includes_evidence_type(self) -> None:
        ev = _ev("c1", "fact", etype=EvidenceType.CONFLICT)
        text = _format_classified_evidence_for_extraction([ev])
        assert "conflict" in text

    def test_includes_replaced_by_when_provided(self) -> None:
        ev = _ev("c_old", "Denver", ClaimStatus.SUPERSEDED)
        ss_info = {"c_old": "c_new"}
        text = _format_classified_evidence_for_extraction([ev], ss_info)
        assert "replaced_by=c_new" in text

    def test_no_replaced_by_when_not_provided(self) -> None:
        ev = _ev("c1", "Denver")
        text = _format_classified_evidence_for_extraction([ev])
        assert "replaced_by" not in text

    def test_multiple_evidence_items(self) -> None:
        evs = [
            _ev("c1", "old value", ClaimStatus.SUPERSEDED),
            _ev("c2", "new value", ClaimStatus.ACTIVE),
        ]
        text = _format_classified_evidence_for_extraction(evs)
        assert "item_000" in text
        assert "item_001" in text

    def test_empty_list(self) -> None:
        text = _format_classified_evidence_for_extraction([])
        assert text == ""


class TestPrompts:
    def test_substrate_note_extraction_system_mentions_lifecycle(self) -> None:
        assert "superseded" in SUBSTRATE_NOTE_EXTRACTION_SYSTEM
        assert "active" in SUBSTRATE_NOTE_EXTRACTION_SYSTEM

    def test_substrate_answer_system_mentions_updated(self) -> None:
        assert "updated" in SUBSTRATE_ANSWER_SYSTEM
        assert "MOST RECENT" in SUBSTRATE_ANSWER_SYSTEM

    def test_substrate_answer_system_mentions_idk(self) -> None:
        assert "I don't know" in SUBSTRATE_ANSWER_SYSTEM
