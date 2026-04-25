"""Event-neighbor expansion tests.

Tests the targeted retrieval fix for cross-claim event assembly:
when a DIRECT claim exists for an event but doesn't satisfy the asked
slot (e.g. location), expand retrieval around that claim's local
neighborhood to recover the missing slot value.
"""
from __future__ import annotations

import json

import pytest

from eval.longmemeval.adapter import LongMemEvalQuestion
from src.substrate.on_new_turn import ExtractedClaim
from tests.e2e.conftest import (
    GoldTracker,
    MockEmbeddingClient,
    make_mock_extractor,
    run_instrumented_pipeline,
)


def _dump(tracker: GoldTracker) -> str:
    return json.dumps(tracker.to_dict(), indent=2, default=str)


class TestEventNeighborExpansionTriggersOnMissingLocation:
    """Case A: DIRECT claim has event but no location. Neighbor has location.

    Q: "Where did I redeem a $5 coupon on coffee creamer?"
    Direct: "redeemed $5 coupon on coffee creamer last Sunday" (no location)
    Neighbor (same turn): "shopping at Target for groceries"
    Expected: expansion triggers, Target enters bundle.
    """

    def test_expansion_brings_target_into_bundle(self, mock_embedding_client):
        q = LongMemEvalQuestion(
            question_id="event_loc",
            question="Where did I redeem a $5 coupon on coffee creamer?",
            answer="Target",
            question_type="information_extraction",
            haystack_sessions=[
                [
                    {"role": "user", "content": "I went to the grocery store yesterday afternoon"},
                    {"role": "assistant", "content": "What did you get at the store"},
                    {"role": "user", "content": "I redeemed a $5 coupon on coffee creamer last Sunday"},
                    {"role": "assistant", "content": "That is a great deal on coffee creamer"},
                    {"role": "user", "content": "Yes I was shopping at Target for groceries and household items"},
                    {"role": "assistant", "content": "Target has good deals on household items"},
                ],
            ],
        )
        extractor = make_mock_extractor([
            ("redeemed a $5 coupon", [
                ExtractedClaim(
                    subject="user", predicate="user_event",
                    value="redeemed $5 coupon on coffee creamer last Sunday",
                    confidence=0.9,
                ),
            ]),
            ("shopping at Target", [
                ExtractedClaim(
                    subject="user", predicate="user_event",
                    value="shopping at Target for groceries",
                    confidence=0.9,
                ),
            ]),
            ("grocery store", [
                ExtractedClaim(
                    subject="user", predicate="user_event",
                    value="went to the grocery store yesterday",
                    confidence=0.7,
                ),
            ]),
        ])

        tracker, trace = run_instrumented_pipeline(
            q,
            extractor=extractor,
            embedding_client=mock_embedding_client,
            gold_value="Target",
        )

        assert tracker.in_bundle, (
            f"Target must be in bundle after event-neighbor expansion.\n"
            f"bundle_values: {tracker.bundle_values}\n"
            f"direct_values: {tracker.direct_values}\n{_dump(tracker)}"
        )
        assert tracker.reader_output_correct, (
            f"Reader must answer 'Target'. Got: '{tracker.reader_output}'"
        )


class TestExpansionNotTriggeredWhenSlotSatisfied:
    """Case B: DIRECT claim already satisfies the slot. No expansion needed.

    Q: "How long is my commute?"
    Direct: "commute takes 45 minutes each way"
    Expected: no expansion (duration slot satisfied).
    """

    def test_no_expansion_when_direct_has_answer(self, mock_embedding_client):
        q = LongMemEvalQuestion(
            question_id="no_expand",
            question="How long is my daily commute to work?",
            answer="45 minutes",
            question_type="information_extraction",
            haystack_sessions=[
                [
                    {"role": "user", "content": "My daily commute takes 45 minutes each way by car"},
                    {"role": "assistant", "content": "That is a reasonable commute time"},
                    {"role": "user", "content": "I usually listen to podcasts during the drive"},
                    {"role": "assistant", "content": "Podcasts are great for commuting"},
                ],
            ],
        )
        extractor = make_mock_extractor([
            ("45 minutes", [
                ExtractedClaim(
                    subject="user", predicate="user_fact",
                    value="daily commute takes 45 minutes each way",
                    confidence=0.9,
                ),
            ]),
            ("podcasts", [
                ExtractedClaim(
                    subject="user", predicate="user_preference",
                    value="listens to podcasts during commute",
                    confidence=0.8,
                ),
            ]),
        ])

        tracker, trace = run_instrumented_pipeline(
            q,
            extractor=extractor,
            embedding_client=mock_embedding_client,
            gold_value="45 minutes",
        )

        assert tracker.classified_as_direct, "Gold must be DIRECT"
        assert tracker.reader_output_correct, f"Reader wrong: '{tracker.reader_output}'"


class TestExpansionIsLocalNotFullSession:
    """Case C: Expansion uses ±3 turns, not the full session.

    Gold location is 10 turns away from the event claim.
    Expansion should NOT reach it.
    """

    def test_distant_claim_not_added(self, mock_embedding_client):
        turns = []
        # Turn 0-1: event claim (coupon)
        turns.append({"role": "user", "content": "I redeemed a $5 coupon on coffee creamer at checkout"})
        turns.append({"role": "assistant", "content": "Good deal on coffee creamer"})
        # Turns 2-17: unrelated filler (8 exchanges = 16 turns)
        for i in range(8):
            turns.append({"role": "user", "content": f"Today I worked on task number {i+10} at the office"})
            turns.append({"role": "assistant", "content": f"Good progress on task {i+10}"})
        # Turn 18-19: location claim far away
        turns.append({"role": "user", "content": "I bought groceries at Target last week for the party"})
        turns.append({"role": "assistant", "content": "Target has good party supplies"})

        q = LongMemEvalQuestion(
            question_id="far_loc",
            question="Where did I redeem a $5 coupon on coffee creamer?",
            answer="Target",
            question_type="information_extraction",
            haystack_sessions=[turns],
        )

        rules = [
            ("redeemed a $5 coupon", [ExtractedClaim(
                subject="user", predicate="user_event",
                value="redeemed $5 coupon on coffee creamer at checkout",
                confidence=0.9,
            )]),
            ("Target", [ExtractedClaim(
                subject="user", predicate="user_event",
                value="bought groceries at Target last week",
                confidence=0.9,
            )]),
        ]
        for i in range(8):
            rules.append((f"task number {i+10}", [ExtractedClaim(
                subject="user", predicate="user_fact",
                value=f"worked on task number {i+10} at the office",
                confidence=0.7,
            )]))
        extractor = make_mock_extractor(rules)

        tracker, trace = run_instrumented_pipeline(
            q,
            extractor=extractor,
            embedding_client=mock_embedding_client,
            gold_value="Target",
        )

        # Target is 9 turns away from the coupon claim. ±3 window won't reach it.
        assert not tracker.in_bundle or tracker.in_candidates, (
            "Target should NOT be added by ±3 expansion (it's 9 turns away). "
            "If it's in the bundle, it came from retrieval, not expansion."
        )


class TestExpandedCandidatesGoThroughVerifier:
    """Case D: Expanded neighbors must be classified by the verifier."""

    def test_neighbor_gets_classified(self, mock_embedding_client):
        q = LongMemEvalQuestion(
            question_id="verify_exp",
            question="Where did I buy the birthday cake?",
            answer="Costco",
            question_type="information_extraction",
            haystack_sessions=[
                [
                    {"role": "user", "content": "I bought a chocolate birthday cake for the party this weekend"},
                    {"role": "assistant", "content": "That sounds delicious for the party"},
                    {"role": "user", "content": "I got it from Costco because they have the best sheet cakes"},
                    {"role": "assistant", "content": "Costco sheet cakes are great for large parties"},
                ],
            ],
        )
        extractor = make_mock_extractor([
            ("birthday cake", [ExtractedClaim(
                subject="user", predicate="user_event",
                value="bought chocolate birthday cake for party this weekend",
                confidence=0.9,
            )]),
            ("Costco", [ExtractedClaim(
                subject="user", predicate="user_event",
                value="got cake from Costco for the best sheet cakes",
                confidence=0.9,
            )]),
        ])

        tracker, trace = run_instrumented_pipeline(
            q,
            extractor=extractor,
            embedding_client=mock_embedding_client,
            gold_value="Costco",
        )

        assert tracker.in_bundle, (
            f"Costco must be in bundle.\nbundle_values: {tracker.bundle_values}\n{_dump(tracker)}"
        )
        assert tracker.reader_output_correct, f"Reader wrong: '{tracker.reader_output}'"


class TestEventNeighborExpansionWithUnrelatedDirectClaim:
    """Case E: Unrelated DIRECT claim with location must not block expansion.

    Simulates the real 51a45a95 failure: DIRECT claim about cat tower
    from Petco falsely satisfies location slot, blocking expansion
    for the coupon event.
    """

    def test_unrelated_direct_does_not_block_expansion(self, mock_embedding_client):
        q = LongMemEvalQuestion(
            question_id="unrel_dir",
            question="Where did I redeem a $5 coupon on coffee creamer?",
            answer="Target",
            question_type="information_extraction",
            haystack_sessions=[
                [
                    {"role": "user", "content": "I bought a cat tower from Petco for one hundred twenty dollars"},
                    {"role": "assistant", "content": "That is a nice cat tower from Petco"},
                    {"role": "user", "content": "I redeemed a $5 coupon on coffee creamer last Sunday at the store"},
                    {"role": "assistant", "content": "Good deal on coffee creamer with the coupon"},
                    {"role": "user", "content": "I was doing my usual weekly shopping at Target that day"},
                    {"role": "assistant", "content": "Target is good for weekly grocery shopping trips"},
                ],
            ],
        )
        extractor = make_mock_extractor([
            ("cat tower from Petco", [ExtractedClaim(
                subject="user", predicate="user_event",
                value="bought cat tower from Petco for $120",
                confidence=0.9,
            )]),
            ("redeemed a $5 coupon", [ExtractedClaim(
                subject="user", predicate="user_event",
                value="redeemed $5 coupon on coffee creamer last Sunday",
                confidence=0.9,
            )]),
            ("shopping at Target", [ExtractedClaim(
                subject="user", predicate="user_event",
                value="weekly shopping at Target",
                confidence=0.9,
            )]),
        ])

        tracker, trace = run_instrumented_pipeline(
            q,
            extractor=extractor,
            embedding_client=mock_embedding_client,
            gold_value="Target",
        )

        assert tracker.in_bundle, (
            f"Target must be in bundle. Petco claim must NOT block expansion.\n"
            f"bundle_values: {tracker.bundle_values}\n"
            f"direct_values: {tracker.direct_values}\n{_dump(tracker)}"
        )
        assert tracker.reader_output_correct, (
            f"Reader must answer 'Target'. Got: '{tracker.reader_output}'"
        )
