"""Tests for decision snapshots and context contracts (Layer 8)."""
from __future__ import annotations

import json

from eval.longmemeval.decision_snapshot import (
    ContextContract,
    DecisionSnapshot,
    build_snapshot,
)


class TestBuildSnapshot:
    def test_captures_all_fields(self) -> None:
        snap = build_snapshot(
            question_id="q1",
            question_type="information_extraction",
            retrieval_mode="current_truth",
            active_claim_count=10,
            superseded_claim_count=3,
            retrieved_candidate_count=8,
            classified_evidence=[],
            bundle_tokens=500,
            answer="42",
            depended_on_claim_ids=["c1", "c2"],
        )
        assert snap.question_id == "q1"
        assert snap.active_claim_count == 10
        assert snap.answer == "42"
        assert snap.snapshot_id.startswith("snap_")
        assert snap.timestamp_ns > 0

    def test_contract_claim_ids(self) -> None:
        snap = build_snapshot(
            question_id="q1",
            question_type="information_extraction",
            retrieval_mode="current_truth",
            active_claim_count=5,
            superseded_claim_count=0,
            retrieved_candidate_count=3,
            classified_evidence=[],
            bundle_tokens=100,
            answer="test",
            depended_on_claim_ids=["c1", "c2", "c3"],
            excluded_superseded_ids=["c_old"],
        )
        assert snap.contract.depended_on_claim_ids == ("c1", "c2", "c3")
        assert snap.contract.excluded_superseded_ids == ("c_old",)

    def test_insufficient_when_no_direct_evidence(self) -> None:
        snap = build_snapshot(
            question_id="q1",
            question_type="abstention",
            retrieval_mode="current_truth",
            active_claim_count=0,
            superseded_claim_count=0,
            retrieved_candidate_count=0,
            classified_evidence=[],
            bundle_tokens=0,
            answer="I don't know",
            depended_on_claim_ids=[],
        )
        assert snap.contract.evidence_sufficiency == "insufficient"

    def test_unique_snapshot_ids(self) -> None:
        snaps = [
            build_snapshot(
                question_id="q1",
                question_type="info",
                retrieval_mode="current_truth",
                active_claim_count=0,
                superseded_claim_count=0,
                retrieved_candidate_count=0,
                classified_evidence=[],
                bundle_tokens=0,
                answer="",
                depended_on_claim_ids=[],
            )
            for _ in range(10)
        ]
        ids = {s.snapshot_id for s in snaps}
        assert len(ids) == 10


class TestSerialization:
    def test_to_json_roundtrip(self) -> None:
        snap = build_snapshot(
            question_id="q1",
            question_type="temporal_reasoning",
            retrieval_mode="historical_truth",
            active_claim_count=5,
            superseded_claim_count=2,
            retrieved_candidate_count=8,
            classified_evidence=[],
            bundle_tokens=300,
            answer="Boston",
            depended_on_claim_ids=["c1"],
            unresolved_conflicts=["c2 vs c3"],
        )
        j = snap.to_json()
        parsed = json.loads(j)
        assert parsed["question_id"] == "q1"
        assert parsed["contract"]["depended_on_claim_ids"] == ["c1"]
        assert parsed["contract"]["unresolved_conflicts"] == ["c2 vs c3"]

    def test_to_dict_has_contract(self) -> None:
        snap = build_snapshot(
            question_id="q1",
            question_type="info",
            retrieval_mode="current_truth",
            active_claim_count=0,
            superseded_claim_count=0,
            retrieved_candidate_count=0,
            classified_evidence=[],
            bundle_tokens=0,
            answer="",
            depended_on_claim_ids=[],
        )
        d = snap.to_dict()
        assert "contract" in d
        assert "snapshot_id" in d["contract"]
