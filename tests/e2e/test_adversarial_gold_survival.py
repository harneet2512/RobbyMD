"""Adversarial end-to-end gold-fact survival tests.

Core invariant: if the gold answer exists in the input, it must be
extracted, remain active, be retrievable, appear in top-3, be classified
DIRECT, survive budget, and the reader must answer correctly.

Every test case injects controlled data through the REAL pipeline
(extraction → supersession → embedding → retrieval → classification →
budget → bundle → reader) using deterministic mocks.

Every failure is labeled at the exact stage it occurs.
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
    make_mock_reader,
    run_instrumented_pipeline,
)


def _dump_tracker(tracker: GoldTracker) -> str:
    """Raw instrumentation output. No summaries."""
    d = tracker.to_dict()
    return json.dumps(d, indent=2, default=str)


# ═══════════════════════════════════════════════════════════════════
# Case A: Clean Extraction (MUST PASS)
#
# One gold fact, no distractors. If this fails, pipeline is
# fundamentally broken.
# ═══════════════════════════════════════════════════════════════════


class TestCleanExtraction:
    def test_gold_survives_all_stages(self, mock_embedding_client):
        q = LongMemEvalQuestion(
            question_id="case_a",
            question="How long is my daily commute to work?",
            answer="45 minutes each way",
            question_type="information_extraction",
            haystack_sessions=[
                [
                    {"role": "user", "content": "My daily commute takes 45 minutes each way by car"},
                    {"role": "assistant", "content": "That is a reasonable commute time for your area"},
                ]
            ],
        )
        extractor = make_mock_extractor([
            ("45 minutes", [
                ExtractedClaim(
                    subject="user",
                    predicate="user_fact",
                    value="daily commute takes 45 minutes each way",
                    confidence=0.9,
                ),
            ]),
        ])

        tracker, trace = run_instrumented_pipeline(
            q,
            extractor=extractor,
            embedding_client=mock_embedding_client,
            gold_value="45 minutes",
        )

        assert tracker.extracted, f"EXTRACTION FAILED.\n{_dump_tracker(tracker)}"
        assert tracker.active, f"SUPERSESSION KILLED GOLD.\n{_dump_tracker(tracker)}"
        assert tracker.in_candidates, f"RETRIEVAL MISSED GOLD.\n{_dump_tracker(tracker)}"
        assert tracker.classified_as_direct, (
            f"CLASSIFICATION WRONG: got '{tracker.classification}'.\n{_dump_tracker(tracker)}"
        )
        assert tracker.in_bundle, f"BUDGET DROPPED GOLD.\n{_dump_tracker(tracker)}"
        assert tracker.reader_input_contains_gold, f"GOLD NOT IN READER INPUT.\n{_dump_tracker(tracker)}"
        assert tracker.reader_output_correct, f"READER WRONG: '{tracker.reader_output}'.\n{_dump_tracker(tracker)}"
        assert tracker.failure_class == "", f"UNEXPECTED FAILURE: {tracker.failure_class}\n{_dump_tracker(tracker)}"


# ═══════════════════════════════════════════════════════════════════
# Case B: Supersession Confusion Attack (EXPECTED FAILURE)
#
# Three competing commute values. Latest supersedes the gold.
# Proves supersession CAN destroy gold facts.
#
# Jaccard calculations (noise words removed):
#   30min tokens: {commute, 30, minutes, each, way}
#   45min tokens: {commute, 45, minutes, each, way}
#   J(30,45) = |{commute,minutes,each,way}| / |{commute,30,45,minutes,each,way}| = 4/6 = 0.67 >= 0.3 → fires
#
#   45min tokens: {commute, 45, minutes, each, way}
#   1hr tokens:   {commute, 1, hour, each, way}
#   J(45,1hr) = |{commute,each,way}| / |{commute,45,minutes,1,hour,each,way}| = 3/7 = 0.43 >= 0.3 → fires
# ═══════════════════════════════════════════════════════════════════


class TestSupersessionConfusionAttack:
    def test_gold_superseded_by_later_value(self, mock_embedding_client):
        q = LongMemEvalQuestion(
            question_id="case_b",
            question="How long is my daily commute to work?",
            answer="45 minutes",
            question_type="information_extraction",
            haystack_sessions=[
                [
                    {"role": "user", "content": "My commute is about 30 minutes each way to work"},
                    {"role": "assistant", "content": "That is not too bad for a daily drive"},
                ],
                [
                    {"role": "user", "content": "Actually my commute is now 45 minutes each way to work"},
                    {"role": "assistant", "content": "Noted the change in your commute time"},
                ],
                [
                    {"role": "user", "content": "My commute is now 1 hour each way to work"},
                    {"role": "assistant", "content": "That is a longer drive than before"},
                ],
            ],
        )
        extractor = make_mock_extractor([
            ("30 minutes", [
                ExtractedClaim(
                    subject="user", predicate="user_fact",
                    value="commute is about 30 minutes each way",
                    confidence=0.9,
                ),
            ]),
            ("45 minutes", [
                ExtractedClaim(
                    subject="user", predicate="user_fact",
                    value="commute is now 45 minutes each way",
                    confidence=0.9,
                ),
            ]),
            ("1 hour", [
                ExtractedClaim(
                    subject="user", predicate="user_fact",
                    value="commute is now 1 hour each way",
                    confidence=0.9,
                ),
            ]),
        ])

        tracker, trace = run_instrumented_pipeline(
            q,
            extractor=extractor,
            embedding_client=mock_embedding_client,
            gold_value="45 minutes",
        )

        assert tracker.extracted, "Gold must be extracted initially"
        assert not tracker.active, (
            f"Gold should be superseded by '1 hour' but is still active.\n{_dump_tracker(tracker)}"
        )
        assert tracker.failure_class == "supersession_failure", (
            f"Expected supersession_failure, got '{tracker.failure_class}'.\n{_dump_tracker(tracker)}"
        )


# ═══════════════════════════════════════════════════════════════════
# Case C: Noisy Supporting Claims (MUST PASS)
#
# Gold buried in 10 turns of noise. Must survive retrieval,
# classification, and budgeting despite irrelevant distractors.
# ═══════════════════════════════════════════════════════════════════


class TestNoisySupportingClaims:
    def test_gold_survives_noise(self, mock_embedding_client):
        turns = []
        topics = [
            "Today I finished reviewing the quarterly budget reports at work",
            "I had a great lunch with Sarah at the Italian place downtown",
            "My evening run was about five miles through the park trail",
            "I started reading a new mystery novel before going to sleep",
            "The weather was really nice today and I walked to work instead",
            "I redeemed the five dollar coupon at Target last weekend",  # GOLD (turn 5)
            "My team meeting ran late because of the project deadline discussion",
            "I picked up groceries on my way home from the gym session",
            "I watched that documentary about deep ocean exploration last night",
            "Tomorrow I need to schedule the dentist appointment for next week",
        ]
        for topic in topics:
            turns.append({"role": "user", "content": topic})
            turns.append({"role": "assistant", "content": "Thanks for sharing that with me"})

        q = LongMemEvalQuestion(
            question_id="case_c",
            question="Where did you redeem the coupon?",
            answer="Target",
            question_type="information_extraction",
            haystack_sessions=[turns],
        )

        rules = []
        for topic in topics:
            if "Target" in topic:
                rules.append(("coupon at Target", [
                    ExtractedClaim(
                        subject="user", predicate="user_event",
                        value="redeemed five dollar coupon at Target",
                        confidence=0.9,
                    ),
                ]))
            else:
                first_words = " ".join(topic.split()[:4])
                rules.append((first_words, [
                    ExtractedClaim(
                        subject="user", predicate="user_fact",
                        value=topic[:60],
                        confidence=0.7,
                    ),
                ]))
        extractor = make_mock_extractor(rules)

        tracker, trace = run_instrumented_pipeline(
            q,
            extractor=extractor,
            embedding_client=mock_embedding_client,
            gold_value="Target",
        )

        assert tracker.extracted, f"EXTRACTION FAILED.\n{_dump_tracker(tracker)}"
        assert tracker.active, f"SUPERSESSION KILLED GOLD.\n{_dump_tracker(tracker)}"
        assert tracker.in_candidates, f"RETRIEVAL MISSED GOLD.\n{_dump_tracker(tracker)}"
        assert tracker.classified_as_direct, (
            f"CLASSIFICATION WRONG: got '{tracker.classification}'.\n{_dump_tracker(tracker)}"
        )
        assert tracker.in_bundle, f"BUDGET DROPPED GOLD.\n{_dump_tracker(tracker)}"
        assert tracker.reader_output_correct, f"READER WRONG: '{tracker.reader_output}'.\n{_dump_tracker(tracker)}"


# ═══════════════════════════════════════════════════════════════════
# Case D: Partial Extraction (EXPECTED FAILURE)
#
# Extractor captures topic but misses the key detail (duration).
# Proves the tracker correctly identifies extraction_precision_failure.
# ═══════════════════════════════════════════════════════════════════


class TestPartialExtraction:
    def test_extraction_precision_failure(self, mock_embedding_client):
        q = LongMemEvalQuestion(
            question_id="case_d",
            question="How long is my daily commute to work?",
            answer="45 minutes",
            question_type="information_extraction",
            haystack_sessions=[
                [
                    {"role": "user", "content": "My daily commute takes 45 minutes each way to work"},
                    {"role": "assistant", "content": "That is a reasonable commute time for the area"},
                ],
            ],
        )
        extractor = make_mock_extractor([
            ("commute", [
                ExtractedClaim(
                    subject="user", predicate="user_fact",
                    value="has a daily commute to work by car",
                    confidence=0.9,
                ),
            ]),
        ])

        tracker, trace = run_instrumented_pipeline(
            q,
            extractor=extractor,
            embedding_client=mock_embedding_client,
            gold_value="45 minutes",
        )

        assert len(tracker.extracted_values) > 0, "Some claim must be extracted"
        assert not tracker.extracted, (
            f"Gold '45 minutes' should NOT be in extracted values: {tracker.extracted_values}"
        )
        assert tracker.failure_class == "extraction_precision_failure", (
            f"Expected extraction_precision_failure, got '{tracker.failure_class}'.\n{_dump_tracker(tracker)}"
        )


# ═══════════════════════════════════════════════════════════════════
# Case E: Conflicting Updates + knowledge_update routing (MUST PASS)
#
# Favorite restaurant changes. Tests supersession + CHANGED_TRUTH mode.
#
# Jaccard:
#   Chili's tokens (no noise): {favorite, restaurant, chili's}
#   Olive Garden tokens:       {favorite, restaurant, olive, garden}
#   J = |{favorite, restaurant}| / |{favorite, restaurant, chili's, olive, garden}| = 2/5 = 0.40 >= 0.3 → fires
#
# Chili's superseded. Olive Garden active. knowledge_update includes superseded.
# ═══════════════════════════════════════════════════════════════════


class TestConflictingUpdates:
    def test_knowledge_update_with_supersession(self, mock_embedding_client):
        q = LongMemEvalQuestion(
            question_id="case_e",
            question="What is your favorite restaurant?",
            answer="Olive Garden",
            question_type="knowledge_update",
            haystack_sessions=[
                [
                    {"role": "user", "content": "My favorite restaurant is Chili's and I go there weekly"},
                    {"role": "assistant", "content": "Chili's has some great appetizer options"},
                ],
                [
                    {"role": "user", "content": "Actually my favorite restaurant is now Olive Garden instead"},
                    {"role": "assistant", "content": "Olive Garden has excellent pasta and breadsticks"},
                ],
            ],
        )
        extractor = make_mock_extractor([
            ("Chili's", [
                ExtractedClaim(
                    subject="user", predicate="user_preference",
                    value="favorite restaurant is Chili's",
                    confidence=0.9,
                ),
            ]),
            ("Olive Garden", [
                ExtractedClaim(
                    subject="user", predicate="user_preference",
                    value="favorite restaurant is now Olive Garden",
                    confidence=0.9,
                ),
            ]),
        ])

        tracker, trace = run_instrumented_pipeline(
            q,
            extractor=extractor,
            embedding_client=mock_embedding_client,
            gold_value="Olive Garden",
        )

        assert trace.retrieval_mode == "changed_truth", (
            f"Expected changed_truth routing, got '{trace.retrieval_mode}'"
        )
        assert trace.include_superseded is True, "knowledge_update must include superseded claims"
        assert tracker.extracted, f"EXTRACTION FAILED.\n{_dump_tracker(tracker)}"
        assert tracker.active, f"GOLD SUPERSEDED (should be active).\n{_dump_tracker(tracker)}"
        assert tracker.in_candidates, f"RETRIEVAL MISSED GOLD.\n{_dump_tracker(tracker)}"
        assert tracker.in_bundle, f"BUDGET DROPPED GOLD.\n{_dump_tracker(tracker)}"
        assert tracker.reader_output_correct, f"READER WRONG: '{tracker.reader_output}'.\n{_dump_tracker(tracker)}"


# ═══════════════════════════════════════════════════════════════════
# Case F: Long Bundle / Budget Pressure (MUST PASS)
#
# 20 diverse claims. Only one is the gold (degree). Tests that
# answer_type_match classifies it as DIRECT and budget never drops
# DIRECT (must_keep).
# ═══════════════════════════════════════════════════════════════════


class TestLongBundleBudgetPressure:
    def test_gold_survives_budget_pressure(self, mock_embedding_client):
        topics = [
            "I enjoy hiking on the mountain trails every weekend morning",
            "My favorite color is blue and I always pick blue clothing",
            "I have two cats named Luna and Milo at home with me",
            "I work in software engineering at a tech startup company",
            "I graduated with a Business Administration degree from college",  # GOLD
            "I live in Portland Oregon near the river district neighborhood",
            "My phone number changed to five five five zero one two three",
            "I prefer morning meetings over afternoon meetings at work",
            "I take vitamin D supplements daily with breakfast every day",
            "My car is a twenty nineteen Honda Civic sedan in silver",
            "I am allergic to shellfish and avoid seafood restaurants completely",
            "I speak Spanish fluently and practice with coworkers at lunch",
            "My birthday is March fifteenth and I celebrate with family dinner",
            "I volunteer at the food bank monthly on Saturday mornings",
            "I have been to Japan twice for vacation in spring season",
            "I prefer audiobooks over reading physical books during commute time",
            "My internet provider is Comcast and the connection is reliable",
            "I drink three cups of coffee daily starting at seven morning",
            "I meditate every morning for fifteen minutes before starting work",
            "I play guitar as a hobby and practice on weekend evenings",
        ]

        turns = []
        for topic in topics:
            turns.append({"role": "user", "content": topic})
            turns.append({"role": "assistant", "content": "I have noted that information for you"})

        q = LongMemEvalQuestion(
            question_id="case_f",
            question="What degree did you earn?",
            answer="Business Administration",
            question_type="information_extraction",
            haystack_sessions=[turns],
        )

        rules = [(topic.split()[1], [
            ExtractedClaim(
                subject="user", predicate="user_fact",
                value=topic,
                confidence=0.8,
            ),
        ]) for topic in topics]
        extractor = make_mock_extractor(rules)

        tracker, trace = run_instrumented_pipeline(
            q,
            extractor=extractor,
            embedding_client=mock_embedding_client,
            gold_value="Business Administration",
        )

        assert tracker.extracted, f"EXTRACTION FAILED.\n{_dump_tracker(tracker)}"
        assert tracker.active, f"SUPERSESSION KILLED GOLD.\n{_dump_tracker(tracker)}"
        assert tracker.classified_as_direct, (
            f"CLASSIFICATION WRONG: got '{tracker.classification}'. "
            f"DIRECT values: {tracker.direct_values}\n{_dump_tracker(tracker)}"
        )
        assert tracker.in_bundle, (
            f"BUDGET DROPPED GOLD. Bundle values: {tracker.bundle_values}\n{_dump_tracker(tracker)}"
        )
        assert tracker.reader_output_correct, f"READER WRONG: '{tracker.reader_output}'.\n{_dump_tracker(tracker)}"


# ═══════════════════════════════════════════════════════════════════
# Case G: Abstention Correctness (MUST PASS)
#
# No mention of pets. Reader must abstain.
# ═══════════════════════════════════════════════════════════════════


class TestAbstentionCorrectness:
    def test_correct_abstention_when_answer_absent(self, mock_embedding_client):
        q = LongMemEvalQuestion(
            question_id="case_g",
            question="What is your pet's name?",
            answer="N/A",
            question_type="abstention",
            haystack_sessions=[
                [
                    {"role": "user", "content": "I enjoy cooking Italian food at home on weekends"},
                    {"role": "assistant", "content": "That sounds delicious and fun to prepare"},
                    {"role": "user", "content": "My favorite hobby is reading science fiction novels"},
                    {"role": "assistant", "content": "Do you have any favorite science fiction authors"},
                ],
            ],
        )
        extractor = make_mock_extractor([
            ("cooking", [
                ExtractedClaim(
                    subject="user", predicate="user_preference",
                    value="enjoys cooking Italian food at home on weekends",
                    confidence=0.8,
                ),
            ]),
            ("reading", [
                ExtractedClaim(
                    subject="user", predicate="user_preference",
                    value="favorite hobby is reading science fiction novels",
                    confidence=0.8,
                ),
            ]),
        ])

        tracker, trace = run_instrumented_pipeline(
            q,
            extractor=extractor,
            embedding_client=mock_embedding_client,
            gold_value="N/A",
        )

        assert trace.verified_direct == 0, (
            f"No claim should be DIRECT for pets question. "
            f"DIRECT values: {tracker.direct_values}"
        )
        assert trace.sufficiency == "insufficient" or trace.sufficiency == "marginal", (
            f"Expected insufficient/marginal, got '{trace.sufficiency}'"
        )
        assert "don't know" in trace.answer.lower(), (
            f"Reader must abstain. Got: '{trace.answer}'"
        )


# ═══════════════════════════════════════════════════════════════════
# TASK 2: INTENTIONAL BREAK TESTS
#
# These create adversarial conditions that stress the system.
# Each test documents the exact failure mode and classifies it.
# ═══════════════════════════════════════════════════════════════════


class TestMultipleSimilarFacts:
    """Confusion attack: 5 similar-topic claims with different values.

    Only one is the gold. Tests whether retrieval+classification can
    distinguish the correct claim from near-duplicates.
    """

    def test_similar_facts_correct_one_survives(self, mock_embedding_client):
        q = LongMemEvalQuestion(
            question_id="break_similar",
            question="How many miles do you run each morning?",
            answer="5 miles",
            question_type="information_extraction",
            haystack_sessions=[
                [
                    {"role": "user", "content": "I started running 2 miles each morning last month"},
                    {"role": "assistant", "content": "That is a good start to build endurance"},
                ],
                [
                    {"role": "user", "content": "I now run about 3 miles each morning before work"},
                    {"role": "assistant", "content": "Nice improvement on your running distance"},
                ],
                [
                    {"role": "user", "content": "I increased my morning run to 5 miles each day"},
                    {"role": "assistant", "content": "Five miles is a solid daily running distance"},
                ],
            ],
        )
        extractor = make_mock_extractor([
            ("2 miles", [ExtractedClaim(
                subject="user", predicate="user_fact",
                value="runs 2 miles each morning",
                confidence=0.9,
            )]),
            ("3 miles", [ExtractedClaim(
                subject="user", predicate="user_fact",
                value="runs about 3 miles each morning",
                confidence=0.9,
            )]),
            ("5 miles", [ExtractedClaim(
                subject="user", predicate="user_fact",
                value="runs 5 miles each morning",
                confidence=0.9,
            )]),
        ])

        tracker, trace = run_instrumented_pipeline(
            q,
            extractor=extractor,
            embedding_client=mock_embedding_client,
            gold_value="5 miles",
        )

        # 5 miles is the LATEST value. Supersession should fire:
        # "runs 2 miles each morning" → "runs about 3 miles each morning" (J >= 0.3)
        # "runs about 3 miles each morning" → "runs 5 miles each morning" (J >= 0.3)
        # So "5 miles" should be the only active claim.
        assert tracker.extracted, f"Gold not extracted.\n{_dump_tracker(tracker)}"
        assert tracker.active, (
            f"Gold '5 miles' should be active (latest).\n{_dump_tracker(tracker)}"
        )
        assert tracker.classified_as_direct, (
            f"Gold must be DIRECT. Got: '{tracker.classification}'.\n{_dump_tracker(tracker)}"
        )
        assert tracker.reader_output_correct, f"Reader wrong: '{tracker.reader_output}'"


class TestLargeDistractorBundle:
    """15+ irrelevant claims, 1 gold. Tests ranking and budget survival."""

    def test_gold_not_buried_by_distractors(self, mock_embedding_client):
        distractors = [
            "I wake up at six thirty every weekday morning",
            "My office is on the fourth floor of the building",
            "I take the bus number forty two to work daily",
            "My lunch break is usually from noon to one pm",
            "I have three monitors on my desk at work",
            "My manager is named Patricia and she is very supportive",
            "I drink green tea instead of coffee at work",
            "My team has standup meetings at nine fifteen am",
            "I use a standing desk for about half the day",
            "My parking spot is in section B lot three",
            "I bring lunch from home three days per week",
            "My commute home takes longer due to evening traffic",
            "I work from home on Fridays during the summer months",
            "My company provides free gym membership at the building",
            "I usually leave the office around five thirty pm",
        ]

        turns = []
        # Gold turn first
        turns.append({"role": "user", "content": "I adopted a golden retriever puppy named Biscuit last weekend"})
        turns.append({"role": "assistant", "content": "Congratulations on your new puppy Biscuit"})
        # Then 15 distractors
        for d in distractors:
            turns.append({"role": "user", "content": d})
            turns.append({"role": "assistant", "content": "I have noted that information for future reference"})

        q = LongMemEvalQuestion(
            question_id="break_distractors",
            question="What is the name of your pet?",
            answer="Biscuit",
            question_type="information_extraction",
            haystack_sessions=[turns],
        )

        rules = [
            ("Biscuit", [ExtractedClaim(
                subject="user", predicate="user_fact",
                value="adopted a golden retriever puppy named Biscuit",
                confidence=0.9,
            )]),
        ]
        for d in distractors:
            first_words = " ".join(d.split()[:3])
            rules.append((first_words, [ExtractedClaim(
                subject="user", predicate="user_fact",
                value=d[:60],
                confidence=0.7,
            )]))
        extractor = make_mock_extractor(rules)

        tracker, trace = run_instrumented_pipeline(
            q,
            extractor=extractor,
            embedding_client=mock_embedding_client,
            gold_value="Biscuit",
        )

        assert tracker.extracted, f"Gold not extracted.\n{_dump_tracker(tracker)}"
        assert tracker.active, f"Gold not active.\n{_dump_tracker(tracker)}"
        assert tracker.in_candidates, f"Gold not in candidates.\n{_dump_tracker(tracker)}"
        assert tracker.in_bundle, f"Gold dropped by budget.\n{_dump_tracker(tracker)}"
        assert tracker.reader_output_correct, f"Reader wrong: '{tracker.reader_output}'"


class TestConflictingClaimsPairDetection:
    """Two claims about the same attribute with different values.

    Must detect the conflict and present both to the reader.
    """

    def test_conflicting_values_both_reach_bundle(self, mock_embedding_client):
        q = LongMemEvalQuestion(
            question_id="break_conflict",
            question="What time do you usually wake up?",
            answer="6:30 AM",
            question_type="information_extraction",
            haystack_sessions=[
                [
                    {"role": "user", "content": "I usually wake up at 6:30 AM on weekday mornings"},
                    {"role": "assistant", "content": "That is an early start to the day"},
                    {"role": "user", "content": "On weekends I wake up at 8:00 AM and take it easy"},
                    {"role": "assistant", "content": "A nice weekend sleep in sounds refreshing"},
                ],
            ],
        )
        extractor = make_mock_extractor([
            ("6:30", [ExtractedClaim(
                subject="user", predicate="user_fact",
                value="usually wakes up at 6:30 AM on weekdays",
                confidence=0.9,
            )]),
            ("8:00", [ExtractedClaim(
                subject="user", predicate="user_fact",
                value="wakes up at 8:00 AM on weekends",
                confidence=0.9,
            )]),
        ])

        tracker, trace = run_instrumented_pipeline(
            q,
            extractor=extractor,
            embedding_client=mock_embedding_client,
            gold_value="6:30",
        )

        assert tracker.extracted, f"Gold not extracted.\n{_dump_tracker(tracker)}"
        assert tracker.in_bundle, f"Gold not in bundle.\n{_dump_tracker(tracker)}"
        assert tracker.reader_output_correct, f"Reader wrong: '{tracker.reader_output}'"


# ═══════════════════════════════════════════════════════════════════
# TASK 4: HARD INVARIANT TESTS
#
# These test system-level invariants that must ALWAYS hold.
# ═══════════════════════════════════════════════════════════════════


class TestInvariantDirectNeverDropped:
    """DIRECT evidence must NEVER be dropped by budget. Period."""

    def test_direct_survives_tiny_budget(self, mock_embedding_client):
        q = LongMemEvalQuestion(
            question_id="inv_budget",
            question="What is your favorite movie?",
            answer="Inception",
            question_type="information_extraction",
            haystack_sessions=[
                [
                    {"role": "user", "content": "My all time favorite movie is Inception by Christopher Nolan"},
                    {"role": "assistant", "content": "Inception is a brilliant film with great visual effects"},
                ],
            ],
        )
        extractor = make_mock_extractor([
            ("Inception", [ExtractedClaim(
                subject="user", predicate="user_preference",
                value="favorite movie is Inception by Christopher Nolan",
                confidence=0.9,
            )]),
        ])

        tracker, trace = run_instrumented_pipeline(
            q,
            extractor=extractor,
            embedding_client=mock_embedding_client,
            gold_value="Inception",
        )

        assert tracker.classified_as_direct, (
            f"Gold must be DIRECT. Got: '{tracker.classification}'"
        )
        assert tracker.in_bundle, (
            "INVARIANT VIOLATED: DIRECT evidence was dropped by budget."
        )


class TestInvariantSufficientWhenDirectExists:
    """If DIRECT evidence exists, sufficiency must be SUFFICIENT."""

    def test_direct_implies_sufficient(self, mock_embedding_client):
        q = LongMemEvalQuestion(
            question_id="inv_sufficient",
            question="What programming language do you prefer?",
            answer="Python",
            question_type="information_extraction",
            haystack_sessions=[
                [
                    {"role": "user", "content": "I strongly prefer Python over other programming languages"},
                    {"role": "assistant", "content": "Python is a great choice for many applications"},
                ],
            ],
        )
        extractor = make_mock_extractor([
            ("Python", [ExtractedClaim(
                subject="user", predicate="user_preference",
                value="strongly prefers Python programming language",
                confidence=0.9,
            )]),
        ])

        tracker, trace = run_instrumented_pipeline(
            q,
            extractor=extractor,
            embedding_client=mock_embedding_client,
            gold_value="Python",
        )

        assert trace.verified_direct > 0, "Must have DIRECT evidence"
        assert trace.sufficiency == "sufficient", (
            f"INVARIANT VIOLATED: DIRECT exists but sufficiency='{trace.sufficiency}'"
        )
        assert not trace.should_abstain, "Must NOT abstain when DIRECT exists"


class TestInvariantAbstainWhenNoEvidence:
    """If no evidence at all, must abstain. Never hallucinate."""

    def test_empty_extraction_forces_abstention(self, mock_embedding_client):
        q = LongMemEvalQuestion(
            question_id="inv_abstain",
            question="What is your social security number?",
            answer="N/A",
            question_type="abstention",
            haystack_sessions=[
                [
                    {"role": "user", "content": "I went to the grocery store and bought some apples"},
                    {"role": "assistant", "content": "Apples are a healthy and delicious snack choice"},
                ],
            ],
        )
        # Extractor returns claims but none about SSN
        extractor = make_mock_extractor([
            ("grocery", [ExtractedClaim(
                subject="user", predicate="user_event",
                value="went to grocery store and bought apples",
                confidence=0.8,
            )]),
        ])

        tracker, trace = run_instrumented_pipeline(
            q,
            extractor=extractor,
            embedding_client=mock_embedding_client,
            gold_value="N/A",
        )

        assert trace.verified_direct == 0, "No claim should be DIRECT for SSN question"
        assert "don't know" in trace.answer.lower(), (
            f"INVARIANT VIOLATED: Must abstain but got: '{trace.answer}'"
        )


class TestInvariantDeterminism:
    """Same input must produce same output. Always."""

    def test_two_runs_identical(self, mock_embedding_client):
        q = LongMemEvalQuestion(
            question_id="inv_determ",
            question="What is your favorite color?",
            answer="blue",
            question_type="information_extraction",
            haystack_sessions=[
                [
                    {"role": "user", "content": "My favorite color has always been blue since childhood"},
                    {"role": "assistant", "content": "Blue is a calming and popular color choice"},
                ],
            ],
        )
        extractor = make_mock_extractor([
            ("blue", [ExtractedClaim(
                subject="user", predicate="user_preference",
                value="favorite color has always been blue",
                confidence=0.9,
            )]),
        ])

        tracker1, trace1 = run_instrumented_pipeline(
            q,
            extractor=extractor,
            embedding_client=mock_embedding_client,
            gold_value="blue",
        )
        tracker2, trace2 = run_instrumented_pipeline(
            q,
            extractor=extractor,
            embedding_client=mock_embedding_client,
            gold_value="blue",
        )

        assert tracker1.failure_class == tracker2.failure_class, (
            f"Non-deterministic failure: run1='{tracker1.failure_class}' run2='{tracker2.failure_class}'"
        )
        assert tracker1.reader_output == tracker2.reader_output, (
            f"Non-deterministic reader: run1='{tracker1.reader_output}' run2='{tracker2.reader_output}'"
        )
        assert tracker1.classified_as_direct == tracker2.classified_as_direct, (
            "Non-deterministic classification"
        )
        assert tracker1.bundle_values == tracker2.bundle_values, (
            "Non-deterministic bundle"
        )
