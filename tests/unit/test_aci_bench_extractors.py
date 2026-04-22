"""Unit tests for the 3-tier ConceptExtractor factory + MEDCON F1 maths.

Exercises only the pure-Python surface of `eval/aci_bench/extractors.py`:
- Factory dispatches correctly on `CONCEPT_EXTRACTOR`.
- NullExtractor returns empty set (T2 contract).
- QuickUMLS factory rejects missing `QUICKUMLS_PATH`.
- `compute_medcon_f1` matches the set-intersection definition used by the
  official MEDCON script.

Does NOT load scispaCy or QuickUMLS — those require optional deps that may
not be installed in the licensing-test CI env.
"""
from __future__ import annotations

import pytest

from eval.aci_bench.extractors import (
    MEDCON_SEMANTIC_GROUPS,
    NullExtractor,
    ScispacyExtractor,
    build_extractor,
    compute_medcon_f1,
)


class TestFactoryDispatch:
    def test_default_is_scispacy(self) -> None:
        e = build_extractor(env={})
        assert e.name == "scispacy"
        assert isinstance(e, ScispacyExtractor)

    def test_explicit_scispacy(self) -> None:
        e = build_extractor(env={"CONCEPT_EXTRACTOR": "scispacy"})
        assert e.name == "scispacy"

    def test_explicit_null(self) -> None:
        e = build_extractor(env={"CONCEPT_EXTRACTOR": "null"})
        assert isinstance(e, NullExtractor)
        assert e.name == "null"
        assert e.semantic_groups == frozenset()

    def test_quickumls_without_path_raises(self) -> None:
        with pytest.raises(RuntimeError, match="QUICKUMLS_PATH"):
            build_extractor(env={"CONCEPT_EXTRACTOR": "quickumls"})

    def test_unknown_value_falls_back_to_scispacy(self) -> None:
        e = build_extractor(env={"CONCEPT_EXTRACTOR": "garbage"})
        assert e.name == "scispacy"

    def test_case_insensitive(self) -> None:
        e = build_extractor(env={"CONCEPT_EXTRACTOR": "NULL"})
        assert isinstance(e, NullExtractor)


class TestNullExtractor:
    def test_returns_empty_set(self) -> None:
        e = NullExtractor()
        assert e.extract("Patient has myocardial infarction and diabetes.") == set()

    def test_labels(self) -> None:
        e = NullExtractor()
        assert "MEDCON omitted" in e.label
        assert e.name == "null"


class TestMedconF1:
    def test_perfect_match(self) -> None:
        s = compute_medcon_f1({"C1", "C2"}, {"C1", "C2"})
        assert s["f1"] == 1.0
        assert s["precision"] == 1.0
        assert s["recall"] == 1.0

    def test_disjoint(self) -> None:
        s = compute_medcon_f1({"C1", "C2"}, {"C3", "C4"})
        assert s["f1"] == 0.0

    def test_partial_overlap(self) -> None:
        s = compute_medcon_f1({"C1", "C2", "C3"}, {"C1", "C2", "C4"})
        # tp=2, precision=2/3, recall=2/3, f1=2/3
        assert round(s["f1"], 6) == round(2 / 3, 6)
        assert s["n_gold"] == 3
        assert s["n_pred"] == 3

    def test_both_empty(self) -> None:
        s = compute_medcon_f1(set(), set())
        assert s["f1"] == 0.0
        assert s["n_gold"] == 0
        assert s["n_pred"] == 0

    def test_empty_pred_against_nonempty_gold(self) -> None:
        s = compute_medcon_f1({"C1"}, set())
        assert s["f1"] == 0.0
        assert s["precision"] == 0.0
        assert s["recall"] == 0.0


class TestSemanticGroups:
    def test_seven_groups_defined(self) -> None:
        # Per docs/decisions/2026-04-21_medcon-tiered-fallback.md and the
        # official MEDCON restricted set.
        assert len(MEDCON_SEMANTIC_GROUPS) == 7
        assert "Disorders" in MEDCON_SEMANTIC_GROUPS
        assert "Anatomy" in MEDCON_SEMANTIC_GROUPS
        assert "Procedures" in MEDCON_SEMANTIC_GROUPS
