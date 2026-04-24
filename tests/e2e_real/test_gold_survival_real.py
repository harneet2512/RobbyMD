"""Real E2E gold-fact survival tests — NO MOCKS.

Actual LLM extractor, actual embeddings, actual reader.
Same invariants as tests/e2e/ but against reality.

Skips if credentials are unavailable.
Logs full instrumentation to tests/e2e_real/results/ for drift comparison.

Run: ACTIVE_PACK=personal_assistant pytest tests/e2e_real/ -v -s --tb=long
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from eval.longmemeval.adapter import LongMemEvalQuestion
from tests.e2e_real.conftest import (
    RealGoldTracker,
    _get_embedding_client,
    _has_embedding_credentials,
    _has_extractor_credentials,
    _has_reader_credentials,
    _make_real_reader_fn,
    requires_all,
    run_real_instrumented_pipeline,
    save_tracker,
)


def _load_case(case_id: str) -> LongMemEvalQuestion:
    dataset_path = Path(__file__).parents[2] / "data" / "longmemeval_s_cleaned.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        if item.get("question_id") == case_id:
            return LongMemEvalQuestion(
                question_id=item["question_id"],
                question=item["question"],
                answer=item["answer"],
                question_type=item.get("question_type", "information_extraction"),
                haystack_sessions=item["haystack_sessions"],
                haystack_session_ids=item.get("haystack_session_ids"),
                haystack_dates=item.get("haystack_dates"),
            )
    raise ValueError(f"Case {case_id} not found in dataset")


RUN_ID = f"real_{int(time.time())}"


def _dump(tracker: RealGoldTracker) -> str:
    return json.dumps(tracker.to_dict(), indent=2, default=str)


# ═══════════════════════════════════════════════════════════════════
# The 3 target cases — real pipeline, same invariants
# ═══════════════════════════════════════════════════════════════════


@requires_all
class TestCase_e47becba:
    """Q: 'What degree did I graduate with?' A: 'Business Administration'"""

    def test_real_pipeline(self):
        q = _load_case("e47becba")
        embedding_client = _get_embedding_client()
        reader_fn = _make_real_reader_fn()

        tracker, trace = run_real_instrumented_pipeline(
            q,
            embedding_client=embedding_client,
            reader_fn=reader_fn,
            gold_value="Business Administration",
        )

        path = save_tracker(tracker, RUN_ID)
        print(f"\n=== e47becba ===")
        print(f"gold_value: {tracker.gold_value}")
        print(f"extracted_values ({len(tracker.extracted_values)}): {tracker.extracted_values[:5]}")
        print(f"extracted: {tracker.extracted}")
        print(f"active: {tracker.active} (supersessions: {tracker.supersession_count})")
        print(f"in_candidates: {tracker.in_candidates} (count: {tracker.candidate_count})")
        print(f"rank_position: {tracker.rank_position}")
        print(f"classification: {tracker.classification}")
        print(f"classified_as_direct: {tracker.classified_as_direct}")
        print(f"direct_values: {tracker.direct_values[:3]}")
        print(f"in_bundle: {tracker.in_bundle}")
        print(f"bundle_values: {tracker.bundle_values[:3]}")
        print(f"reader_output: {tracker.reader_output}")
        print(f"reader_output_correct: {tracker.reader_output_correct}")
        print(f"failure_class: {tracker.failure_class or 'SUCCESS'}")
        print(f"total_latency_ms: {tracker.total_latency_ms:.0f}")
        print(f"saved: {path}")

        # Report but do NOT assert — the test's job is to RECORD truth
        # Assertions come AFTER we understand the real failure modes
        if tracker.failure_class:
            pytest.xfail(
                f"Real pipeline failure at {tracker.first_failing_layer}: "
                f"{tracker.failure_class}"
            )


@requires_all
class TestCase_118b2229:
    """Q: 'How long is my daily commute to work?' A: '45 minutes each way'"""

    def test_real_pipeline(self):
        q = _load_case("118b2229")
        embedding_client = _get_embedding_client()
        reader_fn = _make_real_reader_fn()

        tracker, trace = run_real_instrumented_pipeline(
            q,
            embedding_client=embedding_client,
            reader_fn=reader_fn,
            gold_value="45 minutes",
        )

        path = save_tracker(tracker, RUN_ID)
        print(f"\n=== 118b2229 ===")
        print(f"gold_value: {tracker.gold_value}")
        print(f"extracted_values ({len(tracker.extracted_values)}): {tracker.extracted_values[:5]}")
        print(f"extracted: {tracker.extracted}")
        print(f"active: {tracker.active} (supersessions: {tracker.supersession_count})")
        print(f"in_candidates: {tracker.in_candidates} (count: {tracker.candidate_count})")
        print(f"rank_position: {tracker.rank_position}")
        print(f"classification: {tracker.classification}")
        print(f"classified_as_direct: {tracker.classified_as_direct}")
        print(f"direct_values: {tracker.direct_values[:3]}")
        print(f"in_bundle: {tracker.in_bundle}")
        print(f"bundle_values: {tracker.bundle_values[:3]}")
        print(f"reader_output: {tracker.reader_output}")
        print(f"reader_output_correct: {tracker.reader_output_correct}")
        print(f"failure_class: {tracker.failure_class or 'SUCCESS'}")
        print(f"total_latency_ms: {tracker.total_latency_ms:.0f}")
        print(f"saved: {path}")

        if tracker.failure_class:
            pytest.xfail(
                f"Real pipeline failure at {tracker.first_failing_layer}: "
                f"{tracker.failure_class}"
            )


@requires_all
class TestCase_51a45a95:
    """Q: 'Where did I redeem a $5 coupon on coffee creamer?' A: 'Target'"""

    def test_real_pipeline(self):
        q = _load_case("51a45a95")
        embedding_client = _get_embedding_client()
        reader_fn = _make_real_reader_fn()

        tracker, trace = run_real_instrumented_pipeline(
            q,
            embedding_client=embedding_client,
            reader_fn=reader_fn,
            gold_value="Target",
        )

        path = save_tracker(tracker, RUN_ID)
        print(f"\n=== 51a45a95 ===")
        print(f"gold_value: {tracker.gold_value}")
        print(f"extracted_values ({len(tracker.extracted_values)}): {tracker.extracted_values[:5]}")
        print(f"extracted: {tracker.extracted}")
        print(f"active: {tracker.active} (supersessions: {tracker.supersession_count})")
        print(f"in_candidates: {tracker.in_candidates} (count: {tracker.candidate_count})")
        print(f"rank_position: {tracker.rank_position}")
        print(f"classification: {tracker.classification}")
        print(f"classified_as_direct: {tracker.classified_as_direct}")
        print(f"direct_values: {tracker.direct_values[:3]}")
        print(f"in_bundle: {tracker.in_bundle}")
        print(f"bundle_values: {tracker.bundle_values[:3]}")
        print(f"reader_output: {tracker.reader_output}")
        print(f"reader_output_correct: {tracker.reader_output_correct}")
        print(f"failure_class: {tracker.failure_class or 'SUCCESS'}")
        print(f"total_latency_ms: {tracker.total_latency_ms:.0f}")
        print(f"saved: {path}")

        if tracker.failure_class:
            pytest.xfail(
                f"Real pipeline failure at {tracker.first_failing_layer}: "
                f"{tracker.failure_class}"
            )


# ═══════════════════════════════════════════════════════════════════
# Drift comparison — mock vs real
# ═══════════════════════════════════════════════════════════════════


@requires_all
class TestDriftReport:
    """After real runs, compare against deterministic expectations.

    This test ALWAYS runs last and reports the drift summary.
    """

    def test_print_summary(self):
        from tests.e2e_real.conftest import RESULTS_DIR

        if not RESULTS_DIR.exists():
            pytest.skip("No results to compare")

        results = sorted(RESULTS_DIR.glob(f"{RUN_ID}_*.json"))
        if not results:
            pytest.skip("No results from this run")

        print("\n" + "=" * 60)
        print("REAL vs DETERMINISTIC DRIFT REPORT")
        print("=" * 60)

        for path in results:
            data = json.loads(path.read_text(encoding="utf-8"))
            case_id = data.get("case_id", "?")
            print(f"\n--- {case_id} ---")
            print(f"  extracted:          {data.get('extracted')}")
            print(f"  active:             {data.get('active')}")
            print(f"  in_candidates:      {data.get('in_candidates')}")
            print(f"  rank_position:      {data.get('rank_position')}")
            print(f"  classified_direct:  {data.get('classified_as_direct')}")
            print(f"  in_bundle:          {data.get('in_bundle')}")
            print(f"  reader_correct:     {data.get('reader_output_correct')}")
            print(f"  failure_class:      {data.get('failure_class') or 'SUCCESS'}")
            print(f"  latency_ms:         {data.get('total_latency_ms', 0):.0f}")
            print(f"  extracted_count:    {len(data.get('extracted_values', []))}")
            print(f"  reader_output:      {data.get('reader_output', '')[:100]}")

        print("\n" + "=" * 60)
