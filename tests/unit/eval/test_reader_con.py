"""Unit tests for the Chain-of-Note reader.

No network. A `SimpleNamespace`-based fake OpenAI client returns canned
responses so we can inspect the two-call structure, parse-error tolerance,
and abstention semantics.
"""
from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from eval.longmemeval.reader_con import (
    ANSWER_SYSTEM,
    NOTE_EXTRACTION_SYSTEM,
    answer_with_con,
)
from src.substrate.claims import insert_claim, insert_turn, new_turn_id, now_ns
from src.substrate.retrieval import RankedClaim
from src.substrate.schema import Claim, ClaimStatus, Speaker, Turn, open_database


# --- fake OpenAI client ------------------------------------------------------


def _msg(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


class _FakeClient:
    """Records the two calls and returns scripted responses."""

    def __init__(self, responses: list[str]) -> None:
        self.calls: list[dict[str, Any]] = []
        self._responses = list(responses)
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    def _create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("unexpected extra call to the fake client")
        return _msg(self._responses.pop(0))


# --- fixtures ----------------------------------------------------------------


def _make_claim(claim_id: str, subject: str, predicate: str, value: str) -> Claim:
    return Claim(
        claim_id=claim_id,
        session_id="s1",
        subject=subject,
        predicate=predicate,
        value=value,
        value_normalised=None,
        confidence=0.9,
        source_turn_id="t1",
        status=ClaimStatus.ACTIVE,
        created_ts=now_ns(),
    )


def _rc(claim: Claim, sim: float) -> RankedClaim:
    return RankedClaim(
        claim=claim, similarity_score=sim, embedding_model_version="bge-m3@test"
    )


# --- tests -------------------------------------------------------------------


class TestTwoCallStructure:
    def test_both_calls_issued_with_correct_systems(self) -> None:
        claim = _make_claim("c1", "user", "meeting_time", "3pm on Monday")
        retrieved = [_rc(claim, 0.92)]
        notes_json = json.dumps(
            {
                "notes": [
                    {
                        "item_id": "item_000",
                        "verbatim_fact": "user / meeting_time = 3pm on Monday",
                        "reason": "Directly answers when the meeting was scheduled.",
                    }
                ]
            }
        )
        client = _FakeClient([notes_json, "The meeting is at 3pm on Monday."])

        answer, prov = answer_with_con(
            "When was the meeting scheduled?",
            retrieved,
            client_pair=(client, "gpt-4o-2024-08-06"),
        )

        assert len(client.calls) == 2
        # Call 1: note extraction, json_object response format, temperature 0.
        c1 = client.calls[0]
        assert c1["messages"][0]["content"] == NOTE_EXTRACTION_SYSTEM
        assert c1["response_format"] == {"type": "json_object"}
        assert c1["temperature"] == 0.0
        # Call 2: answer, no json response format (free-form), temperature 0.
        c2 = client.calls[1]
        assert c2["messages"][0]["content"] == ANSWER_SYSTEM
        assert "response_format" not in c2
        assert c2["temperature"] == 0.0

        assert "Monday" in answer
        assert prov["model"] == "gpt-4o-2024-08-06"
        assert prov["retrieved_claim_ids"] == ["c1"]
        assert len(prov["notes"]) == 1
        assert prov["notes"][0]["item_id"] == "item_000"
        assert prov["total_latency_ms"] >= 0.0


class TestAbstention:
    def test_empty_notes_forces_idk(self) -> None:
        """When the extractor returns zero relevant notes, we must answer 'I don't know'."""
        claim = _make_claim("c1", "user", "food_preference", "sushi")
        retrieved = [_rc(claim, 0.31)]  # low sim, genuinely unrelated
        notes_json = json.dumps({"notes": []})
        # Call 2 is still made but whatever the model returns we coerce to IDK.
        client = _FakeClient([notes_json, "The answer is sushi."])

        answer, prov = answer_with_con(
            "What is the capital of France?",
            retrieved,
            client_pair=(client, "gpt-4o"),
        )

        assert answer == "I don't know"
        assert prov["notes"] == []

    def test_empty_retrieval_still_safe(self) -> None:
        """Zero retrieved claims: extractor sees an empty list; must abstain."""
        notes_json = json.dumps({"notes": []})
        client = _FakeClient([notes_json, ""])

        answer, prov = answer_with_con(
            "When did the user join the gym?",
            [],
            client_pair=(client, "gpt-4o"),
        )
        assert answer == "I don't know"
        assert prov["retrieved_claim_ids"] == []


class TestParseErrorTolerance:
    def test_malformed_json_returns_empty_notes(self) -> None:
        claim = _make_claim("c1", "user", "x", "y")
        client = _FakeClient(["this is not json", ""])  # Call 1 malformed

        answer, prov = answer_with_con(
            "Q?", [_rc(claim, 0.5)], client_pair=(client, "gpt-4o")
        )
        # Empty notes → abstention.
        assert answer == "I don't know"
        assert prov["notes"] == []

    def test_wrapper_key_variants_accepted(self) -> None:
        claim = _make_claim("c1", "user", "role", "engineer")
        notes_json = json.dumps(
            {
                "items": [  # wrapper key other than "notes"
                    {
                        "item_id": "a",
                        "verbatim_fact": "user / role = engineer",
                        "reason": "Directly states role.",
                    }
                ]
            }
        )
        client = _FakeClient([notes_json, "Engineer."])

        answer, prov = answer_with_con(
            "What is the user's role?",
            [_rc(claim, 0.88)],
            client_pair=(client, "gpt-4o"),
        )
        assert "Engineer" in answer
        assert len(prov["notes"]) == 1

    def test_bare_list_accepted(self) -> None:
        claim = _make_claim("c1", "user", "city", "Tokyo")
        notes_json = json.dumps(
            [
                {
                    "item_id": "x",
                    "verbatim_fact": "user / city = Tokyo",
                    "reason": "City stated.",
                }
            ]
        )
        client = _FakeClient([notes_json, "Tokyo."])
        answer, prov = answer_with_con(
            "Which city?", [_rc(claim, 0.8)], client_pair=(client, "gpt-4o")
        )
        assert "Tokyo" in answer
        assert len(prov["notes"]) == 1


class TestClaimLinkBack:
    def test_note_links_back_to_claim_via_value(self) -> None:
        claim = _make_claim("cl_xyz", "user", "preferred_language", "Python")
        notes_json = json.dumps(
            {
                "notes": [
                    {
                        "item_id": "item_000",
                        "verbatim_fact": "user / preferred_language = Python",
                        "reason": "Language explicitly stated.",
                    }
                ]
            }
        )
        client = _FakeClient([notes_json, "Python."])

        answer, prov = answer_with_con(
            "Language?", [_rc(claim, 0.9)], client_pair=(client, "gpt-4o")
        )
        assert prov["notes"][0]["claim_id"] == "cl_xyz"
