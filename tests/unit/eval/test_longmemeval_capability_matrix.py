from eval.longmemeval.context import _render_extracted_claims_for_prompt
from src.extraction.claim_extractor.prompt import CLAIM_EXTRACTOR_SYSTEM_PROMPT


def test_extractor_system_prompt_preserves_exact_short_facts() -> None:
    prompt = CLAIM_EXTRACTOR_SYSTEM_PROMPT

    assert "Exact fact preservation" in prompt
    for value in (
        "$400,000",
        "8 days",
        "Dr. Arati Prabhakar",
        "the suburbs",
        "25:50",
        "Samsung Galaxy S22",
        "Business Administration",
    ):
        assert value in prompt


def test_longmemeval_turn_prompt_preserves_exact_facts_for_user_turns() -> None:
    rendered = _render_extracted_claims_for_prompt(
        "I was pre-approved for $400,000 and the trip took 8 days.",
        "user",
    )

    assert "EXACT FACT PRESERVATION" in rendered
    assert "$400,000" in rendered
    assert "8 days" in rendered
    assert "Do not round, generalize, or replace values" in rendered


def test_longmemeval_turn_prompt_preserves_exact_facts_for_assistant_turns() -> None:
    rendered = _render_extracted_claims_for_prompt(
        "[assistant] Dr. Arati Prabhakar discussed the result.",
        "assistant",
    )

    assert "IMPORTANT: This is an ASSISTANT turn" in rendered
    assert "amounts, durations, locations, model names" in rendered
    assert "Dr. Arati Prabhakar" in rendered

