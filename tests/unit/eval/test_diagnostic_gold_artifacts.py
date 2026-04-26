from eval.longmemeval.diagnostic_slice import _classify_failure_from_trace
from eval.longmemeval.pipeline import CaseTrace


def test_gold_specific_artifacts_are_derived_from_claim_ids() -> None:
    trace = CaseTrace(
        case_id="case-1",
        question="How long did it take?",
        gold_answer="8 days",
        question_type="multi-session",
        answer="It took 3 days.",
        extracted_claim_ids=["c_wrong", "c_gold"],
        extracted_claim_values_by_id={"c_wrong": "3 days", "c_gold": "8 days"},
        active_claim_ids=["c_wrong", "c_gold"],
        active_claim_values_by_id={"c_wrong": "3 days", "c_gold": "8 days"},
        retrieved_claim_ids=["c_wrong", "c_gold"],
        retrieved_claim_values_by_id={"c_wrong": "3 days", "c_gold": "8 days"},
        verified_claim_ids_by_label={"direct": ["c_wrong", "c_gold"]},
        verified_claim_values_by_id={"c_wrong": "3 days", "c_gold": "8 days"},
        final_claim_ids=["c_wrong"],
        final_claim_values=["3 days"],
        final_bundle_text="Question: How long?\n\n== DIRECT_EVIDENCE ==\n- 3 days",
        reader_prompt="Question: How long?\n\n== DIRECT_EVIDENCE ==\n- 3 days",
        reader_output="It took 3 days.",
    )

    case = _classify_failure_from_trace(trace, "8 days")

    assert case.gold_claim_ids_extracted == ["c_gold"]
    assert case.gold_claim_ids_active == ["c_gold"]
    assert case.gold_claim_ids_retrieved == ["c_gold"]
    assert case.gold_claim_ids_verified == ["c_gold"]
    assert case.gold_claim_ids_in_final_bundle == []
    assert case.gold_extracted_exact is True
    assert case.gold_in_bundle_exact is False
    assert case.reader_saw_gold_exact is False
    assert case.failure_class == "bundling_failure"


def test_gold_in_final_bundle_wrong_answer_is_reader_failure() -> None:
    trace = CaseTrace(
        case_id="case-2",
        question="How long did it take?",
        gold_answer="8 days",
        question_type="multi-session",
        answer="It took 3 days.",
        extracted_claim_ids=["c_gold"],
        extracted_claim_values_by_id={"c_gold": "8 days"},
        active_claim_ids=["c_gold"],
        active_claim_values_by_id={"c_gold": "8 days"},
        retrieved_claim_ids=["c_gold"],
        retrieved_claim_values_by_id={"c_gold": "8 days"},
        verified_claim_ids_by_label={"direct": ["c_gold"]},
        verified_claim_values_by_id={"c_gold": "8 days"},
        final_claim_ids=["c_gold"],
        final_claim_values=["8 days"],
        final_bundle_text="Question: How long?\n\n== DIRECT_EVIDENCE ==\n- 8 days",
        reader_prompt="Question: How long?\n\n== DIRECT_EVIDENCE ==\n- 8 days",
        reader_output="It took 3 days.",
    )

    case = _classify_failure_from_trace(trace, "8 days")

    assert case.gold_claim_ids_in_final_bundle == ["c_gold"]
    assert case.reader_saw_gold_exact is True
    assert case.scorer_result is False
    assert case.failure_class == "reader_failure"

