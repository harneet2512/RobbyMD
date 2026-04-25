"""Tests for assistant-turn claim extraction in LongMemEval.

Verifies that claims are extracted from assistant turns (not just user turns),
preserving speaker metadata so downstream logic can distinguish source.
"""
from __future__ import annotations

import pytest

from eval.longmemeval.adapter import LongMemEvalQuestion
from src.substrate.on_new_turn import ExtractedClaim
from tests.e2e.conftest import (
    GoldTracker,
    MockEmbeddingClient,
    make_mock_extractor,
    run_instrumented_pipeline,
)


class TestExtractClaimsFromAssistantTurnNamedEntity:
    """Claims containing named entities from assistant turns must be extracted."""

    def test_manolo_garcia_extracted(self, mock_embedding_client):
        q = LongMemEvalQuestion(
            question_id="asst_name",
            question="Can you remind me of the Spanish-Catalan singer who supports unity?",
            answer="Manolo García",
            question_type="information_extraction",
            haystack_sessions=[
                [
                    {"role": "user", "content": "How has the political climate of Catalonia influenced its music?"},
                    {"role": "assistant", "content": "The Spanish-Catalan singer Manolo García has spoken publicly about his support for unity between Catalonia and Spain"},
                ],
            ],
        )
        extractor = make_mock_extractor([
            ("Manolo", [
                ExtractedClaim(
                    subject="user", predicate="user_fact",
                    value="Manolo García is a Spanish-Catalan singer who supports unity between Catalonia and Spain",
                    confidence=0.9,
                ),
            ]),
        ])

        tracker, trace = run_instrumented_pipeline(
            q, extractor=extractor, embedding_client=mock_embedding_client,
            gold_value="Manolo",
        )

        assert tracker.extracted, "Claim from assistant turn must be extracted"
        assert tracker.in_bundle, f"Gold must reach bundle. Values: {tracker.bundle_values}"


class TestExtractClaimsFromAssistantTurnTitleName:
    """Claims with titles/honorifics from assistant turns must be extracted."""

    def test_dr_prabhakar_extracted(self, mock_embedding_client):
        q = LongMemEvalQuestion(
            question_id="asst_title",
            question="Who is the President's Chief Advisor for Science and Technology?",
            answer="Dr. Arati Prabhakar",
            question_type="information_extraction",
            haystack_sessions=[
                [
                    {"role": "user", "content": "Tell me about the fusion breakthrough at Lawrence Livermore"},
                    {"role": "assistant", "content": "Dr. Arati Prabhakar, the President's Chief Advisor for Science and Technology, commented on the significance of the breakthrough"},
                ],
            ],
        )
        extractor = make_mock_extractor([
            ("Prabhakar", [
                ExtractedClaim(
                    subject="user", predicate="user_fact",
                    value="Dr. Arati Prabhakar is the President's Chief Advisor for Science and Technology",
                    confidence=0.9,
                ),
            ]),
        ])

        tracker, trace = run_instrumented_pipeline(
            q, extractor=extractor, embedding_client=mock_embedding_client,
            gold_value="Prabhakar",
        )

        assert tracker.extracted, "Claim from assistant turn must be extracted"
        assert tracker.in_bundle, f"Gold must reach bundle. Values: {tracker.bundle_values}"


class TestExtractClaimsFromAssistantTurnMetric:
    """Metrics/percentages from assistant turns must be extracted with exact values."""

    def test_20_percent_improvement_extracted(self, mock_embedding_client):
        q = LongMemEvalQuestion(
            question_id="asst_metric",
            question="What was the average improvement in framerate when using the HAMT agent?",
            answer="approximately 20%",
            question_type="information_extraction",
            haystack_sessions=[
                [
                    {"role": "user", "content": "Can you summarize the results of the HAMT paper?"},
                    {"role": "assistant", "content": "The Hardware-Aware Modular Training agent achieved an average improvement in framerate of approximately 20 percent compared to the baseline"},
                ],
            ],
        )
        extractor = make_mock_extractor([
            ("20", [
                ExtractedClaim(
                    subject="user", predicate="user_fact",
                    value="HAMT agent achieved approximately 20% improvement in framerate",
                    confidence=0.9,
                ),
            ]),
        ])

        tracker, trace = run_instrumented_pipeline(
            q, extractor=extractor, embedding_client=mock_embedding_client,
            gold_value="20",
        )

        assert tracker.extracted, "Metric from assistant turn must be extracted"
        assert tracker.in_bundle, f"Gold must reach bundle. Values: {tracker.bundle_values}"


class TestAssistantAndUserClaimsRemainDistinct:
    """User and assistant claims from the same session must preserve source speaker."""

    def test_claims_have_different_source_turns(self, mock_embedding_client):
        q = LongMemEvalQuestion(
            question_id="asst_distinct",
            question="What singer supports unity between Catalonia and Spain?",
            answer="Manolo García",
            question_type="information_extraction",
            haystack_sessions=[
                [
                    {"role": "user", "content": "I am interested in Catalan politics and music"},
                    {"role": "assistant", "content": "The singer Manolo García supports unity between Catalonia and Spain"},
                ],
            ],
        )
        extractor = make_mock_extractor([
            ("interested in Catalan", [
                ExtractedClaim(
                    subject="user", predicate="user_preference",
                    value="interested in Catalan politics and music",
                    confidence=0.9,
                ),
            ]),
            ("Manolo", [
                ExtractedClaim(
                    subject="user", predicate="user_fact",
                    value="Manolo García supports unity between Catalonia and Spain",
                    confidence=0.9,
                ),
            ]),
        ])

        tracker, trace = run_instrumented_pipeline(
            q, extractor=extractor, embedding_client=mock_embedding_client,
            gold_value="Manolo",
        )

        assert tracker.extracted
        assert len(tracker.extracted_values) >= 2, (
            f"Expected claims from both user and assistant turns, got {len(tracker.extracted_values)}"
        )


class TestLongmemevalPipelineIncludesAssistantTurnClaims:
    """Full pipeline test: assistant-turn claims must survive extraction through reader."""

    def test_assistant_claim_reaches_reader(self, mock_embedding_client):
        q = LongMemEvalQuestion(
            question_id="asst_e2e",
            question="What is the name of the singer who supports Catalan unity?",
            answer="Manolo García",
            question_type="information_extraction",
            haystack_sessions=[
                [
                    {"role": "user", "content": "Tell me about Catalan music and politics"},
                    {"role": "assistant", "content": "One notable example is the singer named Manolo García who publicly supports unity between Catalonia and Spain"},
                    {"role": "user", "content": "That is very interesting to know about the political views of musicians"},
                    {"role": "assistant", "content": "Yes there are many different perspectives among Catalan artists"},
                ],
            ],
        )
        extractor = make_mock_extractor([
            ("Manolo", [
                ExtractedClaim(
                    subject="user", predicate="user_fact",
                    value="singer named Manolo García supports unity between Catalonia and Spain",
                    confidence=0.9,
                ),
            ]),
            ("different perspectives", [
                ExtractedClaim(
                    subject="user", predicate="user_fact",
                    value="many different perspectives among Catalan artists on political views",
                    confidence=0.7,
                ),
            ]),
        ])

        tracker, trace = run_instrumented_pipeline(
            q, extractor=extractor, embedding_client=mock_embedding_client,
            gold_value="Manolo",
        )

        assert tracker.extracted, "Claim must be extracted from assistant turn"
        assert tracker.active, "Claim must be active"
        assert tracker.in_candidates, "Claim must be retrieved"
        assert tracker.in_bundle, f"Claim must reach bundle. bundle_values={tracker.bundle_values}"
        assert tracker.reader_output_correct, (
            f"Reader must produce correct answer. Got: '{tracker.reader_output}'"
        )
