"""Decision snapshots and context contracts (Layer 8).

Persists the full context state at answer time for debugging, replay,
and failure taxonomy. Every answer carries a ContextContract declaring
what evidence it depended on.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from time import time_ns
from typing import Any


@dataclass(frozen=True, slots=True)
class ContextContract:
    """What context the answer actually depended on."""

    snapshot_id: str
    depended_on_claim_ids: tuple[str, ...]
    excluded_superseded_ids: tuple[str, ...]
    unresolved_conflicts: tuple[str, ...]
    retrieval_confidence: float
    evidence_sufficiency: str  # "sufficient" | "marginal" | "insufficient"


@dataclass(frozen=True, slots=True)
class DecisionSnapshot:
    """Full context state at answer time."""

    snapshot_id: str
    question_id: str
    question_type: str
    timestamp_ns: int
    retrieval_mode: str
    active_claim_count: int
    superseded_claim_count: int
    retrieved_candidate_count: int
    verified_evidence_count: int
    direct_evidence_count: int
    conflict_count: int
    bundle_tokens: int
    answer: str
    contract: ContextContract

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["contract"] = asdict(self.contract)
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)


def build_snapshot(
    *,
    question_id: str,
    question_type: str,
    retrieval_mode: str,
    active_claim_count: int,
    superseded_claim_count: int,
    retrieved_candidate_count: int,
    classified_evidence: list[Any],
    bundle_tokens: int,
    answer: str,
    depended_on_claim_ids: list[str],
    excluded_superseded_ids: list[str] | None = None,
    unresolved_conflicts: list[str] | None = None,
    retrieval_confidence: float = 0.0,
) -> DecisionSnapshot:
    """Build a snapshot + contract from pipeline outputs."""
    sid = f"snap_{uuid.uuid4().hex[:12]}"

    direct_count = sum(
        1 for ev in classified_evidence
        if getattr(ev, "evidence_type", None) == "direct"
    )
    conflict_count = sum(
        1 for ev in classified_evidence
        if getattr(ev, "evidence_type", None) == "conflict"
    )

    sufficiency = "sufficient"
    if direct_count == 0:
        sufficiency = "insufficient"
    elif retrieval_confidence < 0.3:
        sufficiency = "marginal"

    contract = ContextContract(
        snapshot_id=sid,
        depended_on_claim_ids=tuple(depended_on_claim_ids),
        excluded_superseded_ids=tuple(excluded_superseded_ids or []),
        unresolved_conflicts=tuple(unresolved_conflicts or []),
        retrieval_confidence=retrieval_confidence,
        evidence_sufficiency=sufficiency,
    )

    return DecisionSnapshot(
        snapshot_id=sid,
        question_id=question_id,
        question_type=question_type,
        timestamp_ns=time_ns(),
        retrieval_mode=retrieval_mode,
        active_claim_count=active_claim_count,
        superseded_claim_count=superseded_claim_count,
        retrieved_candidate_count=retrieved_candidate_count,
        verified_evidence_count=len(classified_evidence),
        direct_evidence_count=direct_count,
        conflict_count=conflict_count,
        bundle_tokens=bundle_tokens,
        answer=answer,
        contract=contract,
    )


__all__ = [
    "ContextContract",
    "DecisionSnapshot",
    "build_snapshot",
]
