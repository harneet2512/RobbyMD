"""Pack-loader tests per Eng_doc.md §4.2 refactor (2026-04-21).

Both `clinical_general` and `personal_assistant` packs must load cleanly from
their per-pack JSON files. Schema invariants (`FewShotExample` count ≥ 5,
closed-vocabulary integrity, sub-slot shape) are locked here so drift is caught.
"""

from __future__ import annotations

from pathlib import Path

from src.substrate.predicate_packs import FewShotExample, PredicatePack, load_pack

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PACKS_DIR = _REPO_ROOT / "predicate_packs"


def test_clinical_general_loads_with_20_families() -> None:
    pack = load_pack(_PACKS_DIR / "clinical_general")
    assert pack.pack_id == "clinical_general"
    assert len(pack.predicate_families) == 20
    # Spot-check the 6 families added in the 2026-04-21 Eng_doc.md §4.2 rewrite.
    for new_family in (
        "allergy",
        "vital_sign",
        "lab_value",
        "imaging_finding",
        "physical_exam_finding",
        "review_of_systems",
    ):
        assert new_family in pack.predicate_families


def test_clinical_general_has_sub_slots_for_structured_predicates() -> None:
    pack = load_pack(_PACKS_DIR / "clinical_general")
    # Eng_doc.md §4.2: predicates with structured values carry sub-slot schemas.
    assert "medication" in pack.sub_slots
    assert {"name", "dose", "route", "frequency"} <= pack.sub_slots["medication"]
    assert "vital_sign" in pack.sub_slots
    assert "lab_value" in pack.sub_slots


def test_clinical_general_has_lr_table_path() -> None:
    pack = load_pack(_PACKS_DIR / "clinical_general")
    assert pack.lr_table_path is not None
    assert pack.lr_table_path.is_file()
    assert pack.lr_table_path.name == "lr_table.json"
    assert "chest_pain" in pack.lr_table_path.as_posix()


def test_clinical_general_has_at_least_5_few_shots() -> None:
    pack = load_pack(_PACKS_DIR / "clinical_general")
    assert len(pack.few_shot_examples) >= 5
    for ex in pack.few_shot_examples:
        assert isinstance(ex, FewShotExample)
        assert ex.name
        assert ex.expected_output


def test_personal_assistant_loads_with_6_families_no_lr_table() -> None:
    pack = load_pack(_PACKS_DIR / "personal_assistant")
    assert pack.pack_id == "personal_assistant"
    expected = {
        "user_fact",
        "user_preference",
        "user_event",
        "user_relationship",
        "user_goal",
        "user_constraint",
    }
    assert pack.predicate_families == frozenset(expected)
    # Personal-assistant pack has no differentials.
    assert pack.lr_table_path is None
    # No sub-slots (natural-language noun phrases).
    assert pack.sub_slots == {}


def test_personal_assistant_has_at_least_5_few_shots_covering_shape_patterns() -> None:
    pack = load_pack(_PACKS_DIR / "personal_assistant")
    assert len(pack.few_shot_examples) >= 5
    names = {ex.name for ex in pack.few_shot_examples}
    # Assert a supersession example is present — mirrors clinical_general's
    # patient_self_correction_supersession pattern.
    assert any("supersession" in name or "correction" in name for name in names)


def test_pack_typing_is_stable() -> None:
    pack = load_pack(_PACKS_DIR / "clinical_general")
    assert isinstance(pack, PredicatePack)
    assert isinstance(pack.predicate_families, frozenset)
    assert isinstance(pack.few_shot_examples, tuple)
