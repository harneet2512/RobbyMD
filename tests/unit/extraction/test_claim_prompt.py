"""Structural tests for the claim-extractor prompt draft.

Phase 2 is not wired to the substrate; these tests guard the draft's
invariants so it doesn't drift while wt-engine's API is still moving.
"""

from __future__ import annotations

from src.extraction.claim_extractor.prompt import (
    CLAIM_EXTRACTOR_SYSTEM_PROMPT,
    FEW_SHOT_EXAMPLES,
    PREDICATE_FAMILIES,
)


def test_five_or_more_few_shot_examples() -> None:
    """CLAUDE.md §5.2 demands at least 5 few-shot examples."""
    assert len(FEW_SHOT_EXAMPLES) >= 5


def test_required_few_shot_coverage() -> None:
    """The 5 clauses of CLAUDE.md §5.2 each have a named example."""
    names = {e.name for e in FEW_SHOT_EXAMPLES}
    required = {
        "multi_claim_utterance",
        "negative_finding",
        "patient_self_correction_supersession",
        "rare_symptom",
        "ambiguous_phrasing",
    }
    missing = required - names
    assert not missing, f"Missing few-shot examples: {missing}"


def test_prompt_lists_predicate_families() -> None:
    """Every predicate family from Eng_doc.md §4.2 is named in the prompt."""
    for fam in PREDICATE_FAMILIES:
        assert fam in CLAIM_EXTRACTOR_SYSTEM_PROMPT


def test_prompt_rejects_invention() -> None:
    """The prompt explicitly forbids fabricated values (rule 4)."""
    lower = CLAIM_EXTRACTOR_SYSTEM_PROMPT.lower()
    # Use any of several synonyms the drafter might land on.
    assert any(
        phrase in lower
        for phrase in ("never fabricate", "do not fabricate", "not fabricate")
    )


def test_predicate_family_set_matches_active_pack() -> None:
    """Active pack (default: clinical_general) ships 20 predicate families per Eng_doc.md §4.2."""
    # clinical_general was expanded 2026-04-21 to 20 families (added allergy,
    # vital_sign, lab_value, imaging_finding, physical_exam_finding,
    # review_of_systems). Prompt is pack-aware now — this test pins the default
    # pack's count so drift is caught.
    assert len(PREDICATE_FAMILIES) == 20
