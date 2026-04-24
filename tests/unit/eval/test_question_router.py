"""Tests for the question-type router (Layer 1)."""
from __future__ import annotations

import pytest

from eval.longmemeval.question_router import (
    RetrievalMode,
    RetrievalStrategy,
    classify_question,
)


class TestStrategyTable:
    """Each known question_type maps to a deterministic strategy."""

    @pytest.mark.parametrize(
        "qtype",
        [
            "information_extraction",
            "multi_session_reasoning",
            "temporal_reasoning",
            "knowledge_update",
            "abstention",
        ],
    )
    def test_known_types_return_strategy(self, qtype: str) -> None:
        strategy = classify_question("any question", qtype)
        assert isinstance(strategy, RetrievalStrategy)
        assert strategy.question_type == qtype

    def test_knowledge_update_includes_superseded(self) -> None:
        s = classify_question("", "knowledge_update")
        assert s.include_superseded is True
        assert s.retrieval_mode == RetrievalMode.CHANGED_TRUTH

    def test_temporal_reasoning_includes_superseded(self) -> None:
        s = classify_question("", "temporal_reasoning")
        assert s.include_superseded is True
        assert s.retrieval_mode == RetrievalMode.HISTORICAL_TRUTH

    def test_information_extraction_excludes_superseded(self) -> None:
        s = classify_question("", "information_extraction")
        assert s.include_superseded is False
        assert s.retrieval_mode == RetrievalMode.CURRENT_TRUTH

    def test_abstention_has_lowest_top_k(self) -> None:
        strategies = {
            qt: classify_question("", qt)
            for qt in [
                "information_extraction",
                "multi_session_reasoning",
                "temporal_reasoning",
                "knowledge_update",
                "abstention",
            ]
        }
        abstention_k = strategies["abstention"].top_k_final
        for qt, s in strategies.items():
            if qt != "abstention":
                assert s.top_k_final >= abstention_k, f"{qt} top_k_final < abstention"

    def test_abstention_has_highest_confidence_threshold(self) -> None:
        strategies = {
            qt: classify_question("", qt)
            for qt in [
                "information_extraction",
                "multi_session_reasoning",
                "temporal_reasoning",
                "knowledge_update",
                "abstention",
            ]
        }
        abstention_ct = strategies["abstention"].confidence_threshold
        for qt, s in strategies.items():
            if qt != "abstention":
                assert s.confidence_threshold <= abstention_ct, (
                    f"{qt} confidence_threshold > abstention"
                )

    def test_temporal_reasoning_has_temporal_boost(self) -> None:
        s = classify_question("", "temporal_reasoning")
        assert s.temporal_boost is True

    def test_knowledge_update_has_update_boost(self) -> None:
        s = classify_question("", "knowledge_update")
        assert s.update_boost is True

    def test_weights_are_5_tuple(self) -> None:
        for qt in ["information_extraction", "temporal_reasoning", "knowledge_update"]:
            s = classify_question("", qt)
            assert len(s.weights) == 5, f"{qt} weights not 5-tuple"


class TestDeterminism:
    """Same input always yields same output."""

    def test_same_type_same_strategy(self) -> None:
        a = classify_question("What is the user's job?", "information_extraction")
        b = classify_question("What is the user's job?", "information_extraction")
        assert a == b

    def test_same_question_no_type_same_strategy(self) -> None:
        a = classify_question("When did the user move?")
        b = classify_question("When did the user move?")
        assert a == b

    def test_frozen_strategy(self) -> None:
        s = classify_question("", "information_extraction")
        with pytest.raises(AttributeError):
            s.top_k_final = 999  # type: ignore[misc]


class TestKeywordFallback:
    """When question_type is empty, keywords drive routing."""

    def test_temporal_keyword_when(self) -> None:
        s = classify_question("When did the user start the job?")
        assert s.retrieval_mode == RetrievalMode.HISTORICAL_TRUTH
        assert s.include_superseded is True

    def test_temporal_keyword_before(self) -> None:
        s = classify_question("What happened before the move?")
        assert s.include_superseded is True

    def test_update_keyword_changed(self) -> None:
        s = classify_question("What changed about the user's address?")
        assert s.retrieval_mode == RetrievalMode.CHANGED_TRUTH
        assert s.include_superseded is True

    def test_update_phrase_used_to(self) -> None:
        s = classify_question("The user used to live in Denver, where now?")
        assert s.retrieval_mode == RetrievalMode.CHANGED_TRUTH

    def test_no_keywords_defaults_to_information_extraction(self) -> None:
        s = classify_question("What is the user's favorite color?")
        assert s.retrieval_mode == RetrievalMode.CURRENT_TRUTH
        assert s.include_superseded is False

    def test_unknown_type_uses_fallback(self) -> None:
        s = classify_question("When did it happen?", "bogus_type")
        assert s.retrieval_mode == RetrievalMode.HISTORICAL_TRUTH

    def test_empty_type_uses_fallback(self) -> None:
        s = classify_question("What is the user's name?", "")
        assert s.retrieval_mode == RetrievalMode.CURRENT_TRUTH
