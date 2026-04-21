"""Smoke tests for the ASR vocab bias string builder.

Goal: catch accidental explosions of the prompt past faster-whisper's 224-token
safety budget, and guard against the term lists drifting out of their sources.
"""

from __future__ import annotations

from src.extraction.asr.vocab import (
    AHA_ACC_2021_DESCRIPTORS,
    ICD10_R07_DESCRIPTORS,
    MAX_PROMPT_WORDS,
    RXNORM_CHEST_PAIN_DRUGS,
    build_initial_prompt,
)


def test_default_prompt_under_max_words() -> None:
    """Default prompt stays under MAX_PROMPT_WORDS (conservative proxy for 224 tokens)."""
    prompt = build_initial_prompt()
    assert len(prompt.split()) <= MAX_PROMPT_WORDS


def test_prompt_mentions_key_clinical_terms() -> None:
    """Representative chest-pain-relevant strings are present in the bias string."""
    prompt = build_initial_prompt().lower()
    # One canonical drug, one canonical descriptor, one canonical symptom.
    assert "metoprolol" in prompt
    assert "pleuritic" in prompt
    assert "chest pain" in prompt


def test_prompt_truncates_when_inputs_too_long() -> None:
    """An oversized input list still returns a prompt under the word cap."""
    big = tuple(f"drug_{i}" for i in range(500))
    prompt = build_initial_prompt(drugs=big)
    assert len(prompt.split()) <= MAX_PROMPT_WORDS


def test_no_snomed_redistribution() -> None:
    """Sanity check — no SNOMED marker strings leaked into the vocab module.

    Per `research/asr_stack.md` §6 R8 we never redistribute SNOMED CT content.
    This test is cheap belt-and-braces on top of the source-list discipline.
    """
    prompt = build_initial_prompt().lower()
    for marker in ("snomed", "sctid", "concept_id"):
        assert marker not in prompt


def test_term_lists_non_empty() -> None:
    """Source lists are non-trivial — guards against accidental empty-tuple commits."""
    assert len(RXNORM_CHEST_PAIN_DRUGS) >= 20
    assert len(AHA_ACC_2021_DESCRIPTORS) >= 15
    assert len(ICD10_R07_DESCRIPTORS) >= 5
