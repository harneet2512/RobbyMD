"""Unit tests for `src/extraction/claim_extractor/extractor.py`.

Cover:
- `_parse_claims` response-shape handling (bare list, wrapper keys,
  single-key-dict fallback, malformed JSON, non-list top-level).
- `_to_extracted_claims` predicate filtering, confidence validation,
  field-shape validation, char-span resolution.
- `_find_char_span` exact-match, case-insensitive fallback, miss case.
- End-to-end `make_llm_extractor` with a mocked OpenAI client, exercising
  the full factory → closure → LLM-call → parse → filter flow without
  network traffic.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.extraction.claim_extractor.extractor import (
    _find_char_span,
    _parse_claims,
    _to_extracted_claims,
    make_llm_extractor,
)
from src.substrate.on_new_turn import ExtractedClaim
from src.substrate.schema import Speaker, Turn


CLINICAL_PREDICATES = frozenset(
    {
        "onset",
        "character",
        "severity",
        "location",
        "radiation",
        "aggravating_factor",
        "alleviating_factor",
        "associated_symptom",
        "duration",
        "medical_history",
        "medication",
        "allergy",
        "family_history",
        "social_history",
        "risk_factor",
        "vital_sign",
        "lab_value",
        "imaging_finding",
        "physical_exam_finding",
        "review_of_systems",
    }
)


def _turn(text: str, turn_id: str = "t1", session_id: str = "s1") -> Turn:
    return Turn(
        turn_id=turn_id,
        session_id=session_id,
        speaker=Speaker.PATIENT,
        text=text,
        ts=0,
    )


# ── _parse_claims ─────────────────────────────────────────────────────────
class TestParseClaims:
    def test_bare_list(self) -> None:
        raw = '[{"subject": "chest_pain", "predicate": "severity", "value": "7/10", "confidence": 0.9}]'
        result = _parse_claims(raw)
        assert len(result) == 1
        assert result[0]["predicate"] == "severity"

    def test_wrapper_claims_key(self) -> None:
        raw = '{"claims": [{"subject": "x", "predicate": "onset", "value": "2h", "confidence": 0.8}]}'
        assert len(_parse_claims(raw)) == 1

    def test_wrapper_items_key(self) -> None:
        raw = '{"items": [{"subject": "x", "predicate": "onset", "value": "2h", "confidence": 0.8}]}'
        assert len(_parse_claims(raw)) == 1

    def test_wrapper_medical_claims_key(self) -> None:
        raw = '{"medical_claims": [{"subject": "x", "predicate": "onset", "value": "2h", "confidence": 0.8}]}'
        assert len(_parse_claims(raw)) == 1

    def test_wrapper_extracted_claims_key(self) -> None:
        raw = '{"extracted_claims": [{"subject": "x", "predicate": "onset", "value": "2h", "confidence": 0.8}]}'
        assert len(_parse_claims(raw)) == 1

    def test_lenient_single_key_list_fallback(self) -> None:
        # Arbitrary wrapper key with a single list value still gets accepted.
        raw = '{"output": [{"subject": "x", "predicate": "onset", "value": "2h", "confidence": 0.8}]}'
        assert len(_parse_claims(raw)) == 1

    def test_multi_key_dict_with_no_known_wrapper_returns_empty(self) -> None:
        raw = '{"meta": 1, "notes": [{"subject": "x"}]}'
        assert _parse_claims(raw) == []

    def test_invalid_json_returns_empty(self) -> None:
        assert _parse_claims("this is not json") == []

    def test_empty_list(self) -> None:
        assert _parse_claims("[]") == []

    def test_non_dict_items_dropped(self) -> None:
        raw = '[{"subject": "x"}, "string_item", 42, null]'
        result = _parse_claims(raw)
        assert len(result) == 1
        assert result[0] == {"subject": "x"}

    def test_top_level_string_returns_empty(self) -> None:
        assert _parse_claims('"just a string"') == []


# ── _to_extracted_claims ──────────────────────────────────────────────────
class TestToExtractedClaims:
    def test_valid_claim_survives(self) -> None:
        raw = [
            {
                "subject": "chest_pain",
                "predicate": "severity",
                "value": "7/10",
                "confidence": 0.9,
            }
        ]
        turn = _turn("Patient reports chest pain at 7/10 severity.")
        result = _to_extracted_claims(raw, turn, CLINICAL_PREDICATES)
        assert len(result) == 1
        assert isinstance(result[0], ExtractedClaim)
        assert result[0].subject == "chest_pain"
        assert result[0].predicate == "severity"
        assert result[0].value == "7/10"
        assert result[0].confidence == 0.9

    def test_predicate_outside_pack_dropped(self) -> None:
        raw = [
            {
                "subject": "x",
                "predicate": "invented_predicate",
                "value": "v",
                "confidence": 0.8,
            }
        ]
        result = _to_extracted_claims(raw, _turn("text"), CLINICAL_PREDICATES)
        assert result == []

    def test_missing_subject_dropped(self) -> None:
        raw = [{"predicate": "onset", "value": "2h", "confidence": 0.8}]
        result = _to_extracted_claims(raw, _turn("text"), CLINICAL_PREDICATES)
        assert result == []

    def test_empty_subject_dropped(self) -> None:
        raw = [{"subject": "   ", "predicate": "onset", "value": "2h", "confidence": 0.8}]
        result = _to_extracted_claims(raw, _turn("text"), CLINICAL_PREDICATES)
        assert result == []

    def test_confidence_above_one_dropped(self) -> None:
        raw = [{"subject": "x", "predicate": "onset", "value": "v", "confidence": 1.5}]
        result = _to_extracted_claims(raw, _turn("text"), CLINICAL_PREDICATES)
        assert result == []

    def test_confidence_below_zero_dropped(self) -> None:
        raw = [{"subject": "x", "predicate": "onset", "value": "v", "confidence": -0.1}]
        result = _to_extracted_claims(raw, _turn("text"), CLINICAL_PREDICATES)
        assert result == []

    def test_confidence_non_numeric_dropped(self) -> None:
        raw = [{"subject": "x", "predicate": "onset", "value": "v", "confidence": "high"}]
        result = _to_extracted_claims(raw, _turn("text"), CLINICAL_PREDICATES)
        assert result == []

    def test_char_span_resolved_when_value_in_text(self) -> None:
        turn_text = "The patient has chest pain at 7/10 severity."
        raw = [
            {
                "subject": "chest_pain",
                "predicate": "severity",
                "value": "7/10",
                "confidence": 0.9,
            }
        ]
        result = _to_extracted_claims(raw, _turn(turn_text), CLINICAL_PREDICATES)
        assert len(result) == 1
        # "7/10" starts at index 30 in the sample text.
        assert result[0].char_start == turn_text.index("7/10")
        assert result[0].char_end == result[0].char_start + len("7/10")

    def test_char_span_none_when_value_not_in_text(self) -> None:
        raw = [
            {
                "subject": "chest_pain",
                "predicate": "severity",
                "value": "unseen_value_xyz",
                "confidence": 0.9,
            }
        ]
        result = _to_extracted_claims(raw, _turn("ordinary turn"), CLINICAL_PREDICATES)
        assert len(result) == 1
        assert result[0].char_start is None
        assert result[0].char_end is None

    def test_multiple_claims_mixed_validity(self) -> None:
        raw = [
            {"subject": "a", "predicate": "onset", "value": "1h", "confidence": 0.9},
            {"subject": "b", "predicate": "not_in_pack", "value": "v", "confidence": 0.9},
            {"subject": "c", "predicate": "severity", "value": "5/10", "confidence": 0.7},
        ]
        result = _to_extracted_claims(raw, _turn("text"), CLINICAL_PREDICATES)
        assert len(result) == 2
        assert {c.predicate for c in result} == {"onset", "severity"}


# ── _find_char_span ───────────────────────────────────────────────────────
class TestFindCharSpan:
    def test_exact_match(self) -> None:
        start, end = _find_char_span("hello chest pain world", "chest pain")
        assert start == 6
        assert end == 16

    def test_case_insensitive_fallback(self) -> None:
        start, end = _find_char_span("Hello Chest Pain world", "chest pain")
        assert start == 6
        assert end == 16

    def test_miss_returns_none(self) -> None:
        assert _find_char_span("hello world", "dyspnoea") == (None, None)

    def test_empty_inputs(self) -> None:
        assert _find_char_span("", "x") == (None, None)
        assert _find_char_span("x", "") == (None, None)


# ── make_llm_extractor end-to-end (mocked client) ─────────────────────────
class TestMakeLlmExtractor:
    def _fake_response(self, content: str) -> SimpleNamespace:
        message = SimpleNamespace(content=content)
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])

    def test_empty_turn_short_circuits(self) -> None:
        extractor = make_llm_extractor(active_pack_id="clinical_general")
        # Empty text should return [] without calling the LLM (no client init).
        result = extractor(_turn(""))
        assert result == []

    def test_whitespace_only_turn_short_circuits(self) -> None:
        extractor = make_llm_extractor(active_pack_id="clinical_general")
        assert extractor(_turn("   \n  ")) == []

    def test_end_to_end_with_mocked_client(self) -> None:
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **_: self._fake_response(
                        '{"claims": [{"subject": "chest_pain", '
                        '"predicate": "severity", "value": "7/10", '
                        '"confidence": 0.9}]}'
                    )
                )
            )
        )
        with patch(
            "eval._openai_client.make_openai_client",
            return_value=(fake_client, "fake-model"),
        ):
            extractor = make_llm_extractor(active_pack_id="clinical_general")
            result = extractor(
                _turn("Patient reports chest pain at 7/10 severity.")
            )

        assert len(result) == 1
        assert result[0].subject == "chest_pain"
        assert result[0].predicate == "severity"
        assert result[0].value == "7/10"
        assert result[0].confidence == 0.9
        # Char span should resolve because "7/10" appears in the turn text.
        assert result[0].char_start is not None
        assert result[0].char_end is not None

    def test_api_error_returns_empty_list(self) -> None:
        def _raise(**_: object) -> object:
            raise RuntimeError("boom")

        fake_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=_raise)
            )
        )
        with patch(
            "eval._openai_client.make_openai_client",
            return_value=(fake_client, "fake-model"),
        ):
            extractor = make_llm_extractor(active_pack_id="clinical_general")
            result = extractor(_turn("some text"))
        assert result == []

    def test_malformed_response_returns_empty(self) -> None:
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **_: self._fake_response("not json at all")
                )
            )
        )
        with patch(
            "eval._openai_client.make_openai_client",
            return_value=(fake_client, "fake-model"),
        ):
            extractor = make_llm_extractor(active_pack_id="clinical_general")
            result = extractor(_turn("text"))
        assert result == []

    def test_personal_assistant_pack_filters_clinical_predicates(self) -> None:
        # personal_assistant pack has `user_fact`, `user_preference`, etc.
        # A clinical `severity` predicate should be dropped under this pack.
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **_: self._fake_response(
                        '{"claims": ['
                        '{"subject": "user", "predicate": "user_preference", '
                        '"value": "likes coffee", "confidence": 0.9},'
                        '{"subject": "x", "predicate": "severity", '
                        '"value": "7/10", "confidence": 0.9}'
                        ']}'
                    )
                )
            )
        )
        with patch(
            "eval._openai_client.make_openai_client",
            return_value=(fake_client, "fake-model"),
        ):
            extractor = make_llm_extractor(active_pack_id="personal_assistant")
            result = extractor(_turn("I like coffee."))

        assert len(result) == 1
        assert result[0].predicate == "user_preference"
