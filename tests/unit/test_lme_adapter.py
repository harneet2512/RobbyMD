"""Unit tests for src/adapter/lme_qa.py and src/adapter/lme_reader_prompt.py.

Uses a fresh in-memory substrate per test (no cross-test state). Retrieval
calls are monkey-patched with deterministic fakes so tests exercise the
adapter's routing + chain-walk logic without needing bge-m3 or a real
question corpus.
"""
from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Callable, Iterator
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.adapter import lme_qa
from src.adapter.lme_qa import (
    LMEAdapterInput,
    MAX_CLAIMS_TO_READER,
    LMEAdapterOutput,
    ReaderClaim,
    adapt,
)
from src.adapter.lme_reader_prompt import format_prompt, READER_SYSTEM
from src.substrate import open_database
from src.substrate.claims import insert_claim, insert_turn, new_turn_id, now_ns
from src.substrate.event_tuples import EventTuple
from src.substrate.schema import Claim, Speaker, Turn


SESSION_ID = "sess_lme_test"


@pytest.fixture()
def conn() -> Iterator[sqlite3.Connection]:
    c = open_database(":memory:")
    try:
        yield c
    finally:
        c.close()


@pytest.fixture()
def seed_claim(conn: sqlite3.Connection) -> Callable[..., Claim]:
    """Factory: seed a turn + claim, return the claim."""

    def _seed(
        *,
        subject: str = "user",
        predicate: str = "medical_history",
        value: str,
        text: str | None = None,
    ) -> Claim:
        tid = new_turn_id()
        turn = Turn(
            turn_id=tid,
            session_id=SESSION_ID,
            speaker=Speaker.PATIENT,
            text=text or f"{subject} {predicate} {value}",
            ts=now_ns(),
        )
        insert_turn(conn, turn)
        return insert_claim(
            conn,
            session_id=SESSION_ID,
            subject=subject,
            predicate=predicate,
            value=value,
            confidence=0.9,
            source_turn_id=tid,
        )

    return _seed


@pytest.fixture()
def adapter_input(conn: sqlite3.Connection) -> Callable[[str, str], LMEAdapterInput]:
    def _make(question: str, question_type: str) -> LMEAdapterInput:
        return LMEAdapterInput(
            question=question,
            question_type=question_type,
            session_id=SESSION_ID,
            conn=conn,
            embedding_client=MagicMock(),  # never called — retrieval is mocked
        )

    return _make


# --- helpers -----------------------------------------------------------------


def _add_edge(conn: sqlite3.Connection, old_id: str, new_id: str) -> None:
    conn.execute(
        "INSERT INTO supersession_edges(edge_id, old_claim_id, new_claim_id, "
        "edge_type, identity_score, created_ts) VALUES (?, ?, ?, ?, ?, ?)",
        (str(uuid.uuid4()), old_id, new_id, "patient_correction", None, now_ns()),
    )
    conn.commit()


# --- routing tests -----------------------------------------------------------


class TestRoutingByQuestionType:
    def test_temporal_reasoning_calls_retrieve_event_tuples_with_k15(
        self,
        monkeypatch: pytest.MonkeyPatch,
        adapter_input: Callable[[str, str], LMEAdapterInput],
        seed_claim: Callable[..., Claim],
    ) -> None:
        claim = seed_claim(value="hypertension")
        event = EventTuple(
            subject=claim.subject,
            action=claim.predicate,
            obj=claim.value,
            valid_from_ts=claim.valid_from_ts,
            valid_until_ts=claim.valid_until_ts,
            claim_id=claim.claim_id,
        )
        spy: dict[str, Any] = {}

        def fake_event_tuples(*args: Any, **kwargs: Any) -> list[tuple[EventTuple, float]]:
            spy["top_k"] = kwargs.get("top_k")
            return [(event, 0.72)]

        monkeypatch.setattr(lme_qa, "retrieve_event_tuples", fake_event_tuples)
        out = adapt(adapter_input("when did it start?", "temporal-reasoning"))
        assert spy["top_k"] == 15
        assert len(out.ranked_claims) == 1
        assert out.ranked_claims[0].claim_id == claim.claim_id

    def test_multi_session_calls_retrieve_hybrid_with_k25(
        self,
        monkeypatch: pytest.MonkeyPatch,
        adapter_input: Callable[[str, str], LMEAdapterInput],
        seed_claim: Callable[..., Claim],
    ) -> None:
        claim = seed_claim(value="diabetes")
        spy: dict[str, Any] = {}

        def fake_hybrid(*args: Any, **kwargs: Any) -> list[tuple[Claim, float]]:
            spy["top_k"] = kwargs.get("top_k")
            return [(claim, 0.8)]

        monkeypatch.setattr(lme_qa, "retrieve_hybrid", fake_hybrid)
        out = adapt(adapter_input("what did I say across sessions?", "multi-session"))
        assert spy["top_k"] == 25
        assert len(out.ranked_claims) == 1

    def test_knowledge_update_calls_retrieve_hybrid_with_k15(
        self,
        monkeypatch: pytest.MonkeyPatch,
        adapter_input: Callable[[str, str], LMEAdapterInput],
        seed_claim: Callable[..., Claim],
    ) -> None:
        claim = seed_claim(value="vegan")
        spy: dict[str, Any] = {}

        def fake_hybrid(*args: Any, **kwargs: Any) -> list[tuple[Claim, float]]:
            spy["top_k"] = kwargs.get("top_k")
            return [(claim, 0.7)]

        monkeypatch.setattr(lme_qa, "retrieve_hybrid", fake_hybrid)
        out = adapt(adapter_input("am I vegan?", "knowledge-update"))
        assert spy["top_k"] == 15
        assert len(out.ranked_claims) == 1

    @pytest.mark.parametrize(
        "q_type", ["single-session-user", "single-session-preference"]
    )
    def test_single_session_calls_retrieve_hybrid_with_k10(
        self,
        q_type: str,
        monkeypatch: pytest.MonkeyPatch,
        adapter_input: Callable[[str, str], LMEAdapterInput],
        seed_claim: Callable[..., Claim],
    ) -> None:
        claim = seed_claim(value="likes cats")
        spy: dict[str, Any] = {}

        def fake_hybrid(*args: Any, **kwargs: Any) -> list[tuple[Claim, float]]:
            spy["top_k"] = kwargs.get("top_k")
            return [(claim, 0.6)]

        monkeypatch.setattr(lme_qa, "retrieve_hybrid", fake_hybrid)
        adapt(adapter_input("what pets?", q_type))
        assert spy["top_k"] == 10

    def test_single_session_assistant_includes_source_turn_text(
        self,
        monkeypatch: pytest.MonkeyPatch,
        adapter_input: Callable[[str, str], LMEAdapterInput],
        seed_claim: Callable[..., Claim],
    ) -> None:
        claim = seed_claim(value="yes", text="assistant said YES explicitly")
        spy: dict[str, Any] = {}

        def fake_hybrid(*args: Any, **kwargs: Any) -> list[tuple[Claim, float]]:
            spy["top_k"] = kwargs.get("top_k")
            return [(claim, 0.9)]

        monkeypatch.setattr(lme_qa, "retrieve_hybrid", fake_hybrid)
        out = adapt(adapter_input("what did it say?", "single-session-assistant"))
        assert spy["top_k"] == 10
        assert out.ranked_claims[0].source_turn_text == "assistant said YES explicitly"

    def test_unknown_type_falls_back_to_hybrid_k10(
        self,
        monkeypatch: pytest.MonkeyPatch,
        adapter_input: Callable[[str, str], LMEAdapterInput],
        seed_claim: Callable[..., Claim],
    ) -> None:
        claim = seed_claim(value="something")
        spy: dict[str, Any] = {}

        def fake_hybrid(*args: Any, **kwargs: Any) -> list[tuple[Claim, float]]:
            spy["top_k"] = kwargs.get("top_k")
            return [(claim, 0.5)]

        monkeypatch.setattr(lme_qa, "retrieve_hybrid", fake_hybrid)
        adapt(adapter_input("anything?", "unrecognized-type-xyz"))
        assert spy["top_k"] == 10


# --- supersession chain walk -------------------------------------------------


class TestSupersessionChainWalk:
    def test_non_update_type_returns_only_current_claim(
        self,
        monkeypatch: pytest.MonkeyPatch,
        conn: sqlite3.Connection,
        adapter_input: Callable[[str, str], LMEAdapterInput],
        seed_claim: Callable[..., Claim],
    ) -> None:
        original = seed_claim(value="pescatarian")
        intermediate = seed_claim(value="vegetarian")
        current = seed_claim(value="vegan")
        _add_edge(conn, original.claim_id, intermediate.claim_id)
        _add_edge(conn, intermediate.claim_id, current.claim_id)

        # Retrieval hits the middle link; adapter should surface only the current.
        monkeypatch.setattr(
            lme_qa,
            "retrieve_hybrid",
            lambda *a, **kw: [(intermediate, 0.75)],
        )
        out = adapt(adapter_input("what do I eat?", "single-session-user"))
        assert len(out.ranked_claims) == 1
        assert out.ranked_claims[0].claim_id == current.claim_id
        assert out.ranked_claims[0].claim_text.endswith("= vegan")

    def test_knowledge_update_surfaces_full_chain(
        self,
        monkeypatch: pytest.MonkeyPatch,
        conn: sqlite3.Connection,
        adapter_input: Callable[[str, str], LMEAdapterInput],
        seed_claim: Callable[..., Claim],
    ) -> None:
        original = seed_claim(value="pescatarian")
        intermediate = seed_claim(value="vegetarian")
        current = seed_claim(value="vegan")
        _add_edge(conn, original.claim_id, intermediate.claim_id)
        _add_edge(conn, intermediate.claim_id, current.claim_id)

        monkeypatch.setattr(
            lme_qa,
            "retrieve_hybrid",
            lambda *a, **kw: [(intermediate, 0.75)],
        )
        out = adapt(adapter_input("what do I eat?", "knowledge-update"))
        ids = [rc.claim_id for rc in out.ranked_claims]
        assert ids == [original.claim_id, intermediate.claim_id, current.claim_id]
        # Supersedes/superseded_by links populated correctly.
        assert out.ranked_claims[0].supersedes is None
        assert out.ranked_claims[0].superseded_by == intermediate.claim_id
        assert out.ranked_claims[1].supersedes == original.claim_id
        assert out.ranked_claims[1].superseded_by == current.claim_id
        assert out.ranked_claims[2].supersedes == intermediate.claim_id
        assert out.ranked_claims[2].superseded_by is None


# --- conflict detection ------------------------------------------------------


class TestConflictDetection:
    def test_conflict_formatted_when_chain_has_two_links(
        self,
        monkeypatch: pytest.MonkeyPatch,
        conn: sqlite3.Connection,
        adapter_input: Callable[[str, str], LMEAdapterInput],
        seed_claim: Callable[..., Claim],
    ) -> None:
        orig = seed_claim(value="New York")
        current = seed_claim(value="Boston")
        _add_edge(conn, orig.claim_id, current.claim_id)

        monkeypatch.setattr(
            lme_qa, "retrieve_hybrid", lambda *a, **kw: [(current, 0.8)]
        )
        out = adapt(adapter_input("where do I live?", "single-session-user"))
        assert out.has_conflicts is True
        assert out.conflict_resolution is not None
        assert "'New York'" in out.conflict_resolution
        assert "'Boston'" in out.conflict_resolution
        assert "Current answer: 'Boston'" in out.conflict_resolution

    def test_no_conflict_when_chain_length_one(
        self,
        monkeypatch: pytest.MonkeyPatch,
        adapter_input: Callable[[str, str], LMEAdapterInput],
        seed_claim: Callable[..., Claim],
    ) -> None:
        claim = seed_claim(value="works at Acme")
        monkeypatch.setattr(
            lme_qa, "retrieve_hybrid", lambda *a, **kw: [(claim, 0.8)]
        )
        out = adapt(adapter_input("where do I work?", "single-session-user"))
        assert out.has_conflicts is False
        assert out.conflict_resolution is None


# --- retrieval confidence ----------------------------------------------------


class TestRetrievalConfidence:
    def test_confidence_is_max_similarity_score(
        self,
        monkeypatch: pytest.MonkeyPatch,
        adapter_input: Callable[[str, str], LMEAdapterInput],
        seed_claim: Callable[..., Claim],
    ) -> None:
        c1 = seed_claim(value="a")
        c2 = seed_claim(value="b")
        c3 = seed_claim(value="c")
        monkeypatch.setattr(
            lme_qa,
            "retrieve_hybrid",
            lambda *a, **kw: [(c1, 0.3), (c2, 0.87), (c3, 0.5)],
        )
        out = adapt(adapter_input("?", "single-session-user"))
        assert out.retrieval_confidence == pytest.approx(0.87)

    def test_empty_retrieval_returns_zero_confidence(
        self,
        monkeypatch: pytest.MonkeyPatch,
        adapter_input: Callable[[str, str], LMEAdapterInput],
    ) -> None:
        monkeypatch.setattr(lme_qa, "retrieve_hybrid", lambda *a, **kw: [])
        out = adapt(adapter_input("?", "single-session-user"))
        assert out.ranked_claims == []
        assert out.retrieval_confidence == 0.0
        assert out.has_conflicts is False
        assert out.conflict_resolution is None


# --- capping + ranking -------------------------------------------------------


class TestRankingAndCap:
    def test_returned_claims_sorted_by_similarity_desc(
        self,
        monkeypatch: pytest.MonkeyPatch,
        adapter_input: Callable[[str, str], LMEAdapterInput],
        seed_claim: Callable[..., Claim],
    ) -> None:
        c1 = seed_claim(value="a")
        c2 = seed_claim(value="b")
        c3 = seed_claim(value="c")
        monkeypatch.setattr(
            lme_qa,
            "retrieve_hybrid",
            lambda *a, **kw: [(c1, 0.2), (c2, 0.9), (c3, 0.5)],
        )
        out = adapt(adapter_input("?", "single-session-user"))
        sims = [rc.similarity_score for rc in out.ranked_claims]
        assert sims == sorted(sims, reverse=True)

    def test_returned_claims_capped_at_max(
        self,
        monkeypatch: pytest.MonkeyPatch,
        adapter_input: Callable[[str, str], LMEAdapterInput],
        seed_claim: Callable[..., Claim],
    ) -> None:
        claims = [seed_claim(value=f"val{i}") for i in range(15)]
        monkeypatch.setattr(
            lme_qa,
            "retrieve_hybrid",
            lambda *a, **kw: [(c, 0.5 + i / 100) for i, c in enumerate(claims)],
        )
        out = adapt(adapter_input("?", "multi-session"))
        assert len(out.ranked_claims) == MAX_CLAIMS_TO_READER


# --- reader prompt formatting ------------------------------------------------


class TestReaderPromptFormatting:
    def test_returns_system_and_user_strings(self) -> None:
        output = LMEAdapterOutput(
            ranked_claims=[],
            retrieval_confidence=0.0,
            has_conflicts=False,
            conflict_resolution=None,
        )
        system, user = format_prompt("Q?", "single-session-user", output)
        assert system == READER_SYSTEM
        assert "Q?" in user
        assert "single-session-user" in user
        assert "0 claims" in user
        assert "(none)" in user

    def test_claim_block_includes_similarity_and_validity(self) -> None:
        rc = ReaderClaim(
            claim_text="user / pet = cat",
            claim_id="c1",
            source_session_id="s1",
            source_turn_id="t1",
            confidence=0.92,
            similarity_score=0.87,
            valid_from_ts=1000,
            valid_until_ts=None,
        )
        output = LMEAdapterOutput(
            ranked_claims=[rc],
            retrieval_confidence=0.87,
            has_conflicts=False,
            conflict_resolution=None,
        )
        _, user = format_prompt("pet?", "single-session-user", output)
        assert "similarity: 0.87" in user
        assert "confidence: 0.92" in user
        assert "session: s1" in user
        assert '"user / pet = cat"' in user
        assert "Valid: 1000 → open" in user

    def test_conflict_section_rendered_when_present(self) -> None:
        rc = ReaderClaim(
            claim_text="user / city = Boston",
            claim_id="c2",
            source_session_id="s2",
            source_turn_id="t2",
            confidence=0.9,
            similarity_score=0.8,
        )
        output = LMEAdapterOutput(
            ranked_claims=[rc],
            retrieval_confidence=0.8,
            has_conflicts=True,
            conflict_resolution="User originally said NY. Current: Boston.",
        )
        _, user = format_prompt("city?", "knowledge-update", output)
        assert "CONFLICT RESOLUTION: User originally said NY. Current: Boston." in user
        assert "Use the current value when answering." in user

    def test_supersession_info_rendered(self) -> None:
        rc = ReaderClaim(
            claim_text="user / pet = cat",
            claim_id="c1",
            source_session_id="s1",
            source_turn_id="t1",
            confidence=0.9,
            similarity_score=0.7,
            supersedes="c0",
            superseded_by="c2",
        )
        output = LMEAdapterOutput(
            ranked_claims=[rc],
            retrieval_confidence=0.7,
            has_conflicts=True,
            conflict_resolution="dummy",
        )
        _, user = format_prompt("pet?", "knowledge-update", output)
        assert "supersedes c0" in user
        assert "superseded by c2" in user
