"""Question-type router for LongMemEval retrieval strategy selection.

Routes each question to a lifecycle-aware retrieval strategy based on the
benchmark's question_type field and keyword heuristics. Deterministic —
no LLM calls, same input always yields the same strategy.

The five LongMemEval question types map to three retrieval modes:

- CURRENT_TRUTH: only active claims (information_extraction, multi_session, abstention)
- HISTORICAL_TRUTH: active + superseded with temporal windows (temporal_reasoning)
- CHANGED_TRUTH: active + superseded, paired via supersession edges (knowledge_update)
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class RetrievalMode(StrEnum):
    CURRENT_TRUTH = "current_truth"
    HISTORICAL_TRUTH = "historical_truth"
    CHANGED_TRUTH = "changed_truth"


@dataclass(frozen=True, slots=True)
class RetrievalStrategy:
    """Per-question retrieval configuration produced by the router."""

    question_type: str
    retrieval_mode: RetrievalMode
    weights: tuple[float, ...]  # (semantic, entity, temporal, bm25, graph)
    top_k_candidates: int
    top_k_final: int
    include_superseded: bool
    confidence_threshold: float
    temporal_boost: bool
    update_boost: bool


# ── routing table ────────────────────────────────────────────────────────

_STRATEGIES: dict[str, RetrievalStrategy] = {
    "information_extraction": RetrievalStrategy(
        question_type="information_extraction",
        retrieval_mode=RetrievalMode.CURRENT_TRUTH,
        weights=(1.0, 1.0, 0.3, 1.0, 0.0),
        top_k_candidates=16,
        top_k_final=8,
        include_superseded=False,
        confidence_threshold=0.15,
        temporal_boost=False,
        update_boost=False,
    ),
    "multi_session_reasoning": RetrievalStrategy(
        question_type="multi_session_reasoning",
        retrieval_mode=RetrievalMode.CURRENT_TRUTH,
        weights=(1.0, 0.8, 0.5, 0.8, 0.5),
        top_k_candidates=24,
        top_k_final=10,
        include_superseded=False,
        confidence_threshold=0.15,
        temporal_boost=False,
        update_boost=False,
    ),
    "temporal_reasoning": RetrievalStrategy(
        question_type="temporal_reasoning",
        retrieval_mode=RetrievalMode.HISTORICAL_TRUTH,
        weights=(0.8, 0.6, 1.5, 0.8, 0.0),
        top_k_candidates=24,
        top_k_final=10,
        include_superseded=True,
        confidence_threshold=0.15,
        temporal_boost=True,
        update_boost=False,
    ),
    "knowledge_update": RetrievalStrategy(
        question_type="knowledge_update",
        retrieval_mode=RetrievalMode.CHANGED_TRUTH,
        weights=(0.8, 1.0, 1.2, 0.8, 0.0),
        top_k_candidates=24,
        top_k_final=12,
        include_superseded=True,
        confidence_threshold=0.15,
        temporal_boost=False,
        update_boost=True,
    ),
    "abstention": RetrievalStrategy(
        question_type="abstention",
        retrieval_mode=RetrievalMode.CURRENT_TRUTH,
        weights=(1.0, 1.0, 0.3, 0.5, 0.0),
        top_k_candidates=12,
        top_k_final=6,
        include_superseded=False,
        confidence_threshold=0.40,
        temporal_boost=False,
        update_boost=False,
    ),
}

_TEMPORAL_KEYWORDS = frozenset(
    {"when", "before", "after", "first", "last", "date", "time", "earliest", "latest"}
)
_UPDATE_KEYWORDS = frozenset(
    {"change", "changed", "update", "updated", "now", "currently", "new", "previous", "old"}
)
_UPDATE_PHRASES = ("used to",)


def classify_question(question: str, question_type: str = "") -> RetrievalStrategy:
    """Return a retrieval strategy for the given question.

    If ``question_type`` matches one of the five LongMemEval categories,
    uses the pre-defined strategy directly. Otherwise falls back to
    keyword heuristics over the question text.
    """
    if question_type in _STRATEGIES:
        return _STRATEGIES[question_type]

    lowered = question.lower()
    words = set(lowered.split())

    if words & _UPDATE_KEYWORDS or any(p in lowered for p in _UPDATE_PHRASES):
        return _STRATEGIES["knowledge_update"]

    if words & _TEMPORAL_KEYWORDS:
        return _STRATEGIES["temporal_reasoning"]

    return _STRATEGIES["information_extraction"]


__all__ = [
    "RetrievalMode",
    "RetrievalStrategy",
    "classify_question",
]
