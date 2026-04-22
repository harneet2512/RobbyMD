"""Unit tests for `eval/aci_bench/llm_medcon.py`.

Hackathon scope (2026-04-22): LLM-MEDCON is the shipped ACI-Bench
concept metric while the UMLS licence is pending. These tests cover the
parse / normalise / empty-input paths that do NOT hit the OpenAI API.
A single mocked end-to-end call exercises the happy path through
`extract()` without network traffic.
"""
from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from eval.aci_bench.llm_medcon import (
    LLM_MEDCON_SEMANTIC_GROUPS,
    LLM_MEDCON_SYSTEM_PROMPT,
    LLMMedconExtractor,
    parse_concepts,
)


# ── parse_concepts ────────────────────────────────────────────────────────
class TestParseConcepts:
    def test_bare_list(self) -> None:
        assert parse_concepts('["chest pain", "dyspnoea"]') == {
            "chest pain",
            "dyspnoea",
        }

    def test_wrapper_concepts_key(self) -> None:
        raw = '{"concepts": ["Hypertension", "Aspirin"]}'
        assert parse_concepts(raw) == {"hypertension", "aspirin"}

    def test_wrapper_items_key(self) -> None:
        raw = '{"items": ["Dyspnoea"]}'
        assert parse_concepts(raw) == {"dyspnoea"}

    def test_wrapper_list_key(self) -> None:
        raw = '{"list": ["Dyspnoea"]}'
        assert parse_concepts(raw) == {"dyspnoea"}

    def test_wrapper_data_key(self) -> None:
        raw = '{"data": ["Dyspnoea"]}'
        assert parse_concepts(raw) == {"dyspnoea"}

    def test_normalises_case_and_whitespace(self) -> None:
        raw = '["  CHEST PAIN  ", "chest pain", "Chest Pain"]'
        assert parse_concepts(raw) == {"chest pain"}

    def test_skips_non_strings(self) -> None:
        raw = '["aspirin", 42, null, "fever"]'
        assert parse_concepts(raw) == {"aspirin", "fever"}

    def test_drops_empty_strings(self) -> None:
        raw = '["   ", "", "aspirin"]'
        assert parse_concepts(raw) == {"aspirin"}

    def test_invalid_json_returns_empty(self) -> None:
        assert parse_concepts("this is not json") == set()

    def test_empty_list(self) -> None:
        assert parse_concepts("[]") == set()

    def test_empty_object(self) -> None:
        assert parse_concepts("{}") == set()

    def test_unexpected_shape_returns_empty(self) -> None:
        assert parse_concepts('{"foo": "bar"}') == set()

    def test_scalar_json_returns_empty(self) -> None:
        assert parse_concepts("42") == set()


# ── LLMMedconExtractor ────────────────────────────────────────────────────
class TestLLMMedconExtractor:
    def test_empty_text_short_circuits(self) -> None:
        """extract('') must not touch the OpenAI client — no API key needed."""
        ext = LLMMedconExtractor()
        assert ext.extract("") == set()
        assert ext.extract("   \n   ") == set()

    def test_metadata_matches_protocol(self) -> None:
        ext = LLMMedconExtractor()
        assert ext.name == "llm_medcon"
        assert "gpt-4o-mini" in ext.label
        assert ext.semantic_groups == LLM_MEDCON_SEMANTIC_GROUPS

    def test_prompt_covers_seven_semantic_groups(self) -> None:
        """The system prompt must name every MEDCON semantic group so
        the extractor is scope-comparable to QuickUMLS-based MEDCON."""
        for group in LLM_MEDCON_SEMANTIC_GROUPS:
            assert group in LLM_MEDCON_SYSTEM_PROMPT, (
                f"semantic group '{group}' missing from prompt"
            )

    def test_missing_api_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        ext = LLMMedconExtractor()
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            ext.extract("Patient complains of chest pain.")

    def test_extract_happy_path_mocked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """End-to-end: mocked client returns a JSON list; extract returns the set."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")

        class FakeClient:
            def __init__(self, api_key: str) -> None:
                self.api_key = api_key
                self.chat = SimpleNamespace(
                    completions=SimpleNamespace(create=self._create)
                )
                self.last_messages: list[dict[str, str]] | None = None

            def _create(self, **kwargs: object):
                self.last_messages = kwargs["messages"]  # type: ignore[assignment]
                msg = SimpleNamespace(
                    content='{"concepts": ["Chest Pain", "Aspirin"]}'
                )
                choice = SimpleNamespace(message=msg)
                return SimpleNamespace(choices=[choice])

        fake = FakeClient(api_key="sk-test-key")
        with patch("openai.OpenAI", return_value=fake):
            ext = LLMMedconExtractor()
            out = ext.extract("Patient complains of chest pain; give aspirin.")

        assert out == {"chest pain", "aspirin"}
        # System prompt must still carry our exact restriction text.
        assert fake.last_messages is not None
        assert fake.last_messages[0]["role"] == "system"
        assert fake.last_messages[0]["content"] == LLM_MEDCON_SYSTEM_PROMPT
        assert fake.last_messages[1]["role"] == "user"


# ── Factory wiring ────────────────────────────────────────────────────────
class TestFactory:
    def test_factory_returns_llm_medcon_when_env_selects_it(self) -> None:
        from eval.aci_bench.extractors import build_extractor

        os.environ.pop("QUICKUMLS_PATH", None)
        extractor = build_extractor(env={"CONCEPT_EXTRACTOR": "llm_medcon"})
        assert isinstance(extractor, LLMMedconExtractor)
        assert extractor.name == "llm_medcon"
