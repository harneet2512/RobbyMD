"""bge-m3 asymmetric query-prefix tests.

bge-m3 expects queries to be prefixed with the task-instruction string
`BGE_M3_QUERY_PREFIX`; documents must be embedded verbatim. The fix adds a
`query_mode` kwarg to `EmbeddingClient.embed` that applies the prefix on
the query side only. These tests verify the prefix is applied (or not)
correctly, and that all three retrieval entry points thread `query_mode=True`
to question text while leaving claim-side embeddings unprefixed.

No torch / sentence-transformers / network calls — backends are spied at
the `_embed_local` level so the base class's prefix logic runs for real.
"""
from __future__ import annotations

import hashlib
import sqlite3
from typing import Any
from collections.abc import Iterator

import pytest

from src.substrate.claims import insert_claim, insert_turn, new_turn_id, now_ns
from src.substrate.retrieval import (
    BGE_M3_QUERY_PREFIX,
    BGE_M3_VERSION_TAG,
    CACHE_DIR_ENV,
    EmbeddingClient,
    backfill_embeddings,
    embed_and_store,
    retrieve_hybrid,
    retrieve_relevant_claims,
)
from src.substrate.schema import Speaker, Turn


@pytest.fixture(autouse=True)
def _isolated_embed_cache(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> Iterator[None]:
    """Pin the embedding cache to a fresh dir per test so embed_and_store
    actually calls the spy backend instead of serving from a prior cache."""
    cache_dir = tmp_path_factory.mktemp("asym_embed_cache")
    monkeypatch.setenv(CACHE_DIR_ENV, str(cache_dir))
    yield


# --- spy embedder ------------------------------------------------------------


class _SpyEmbedder(EmbeddingClient):
    """Records every text that reaches the backend, returns deterministic vectors.

    Overrides `_embed_local` (NOT `embed`) so the base class's query_mode
    prefix logic runs for real. Whatever `_embed_local` receives is exactly
    what was sent to the backend after prefix application.
    """

    def __init__(self) -> None:
        super().__init__(modal_url=None, model_version=BGE_M3_VERSION_TAG)
        self.calls: list[list[str]] = []

    def _embed_local(self, texts: list[str]) -> list[list[float]]:  # type: ignore[override]
        self.calls.append(list(texts))
        out: list[list[float]] = []
        for t in texts:
            digest = hashlib.sha256(t.encode("utf-8")).digest()
            raw = [((b ^ 0x80) - 128) / 128.0 for b in digest[:8]]
            norm = sum(x * x for x in raw) ** 0.5 or 1.0
            out.append([x / norm for x in raw])
        return out


# --- embed-level tests -------------------------------------------------------


class TestEmbedPrefixApplication:
    def test_query_mode_true_prefixes_text(self) -> None:
        spy = _SpyEmbedder()
        spy.embed(["What did I eat yesterday?"], query_mode=True)
        assert len(spy.calls) == 1
        assert spy.calls[0] == [BGE_M3_QUERY_PREFIX + "What did I eat yesterday?"]

    def test_query_mode_false_does_not_prefix(self) -> None:
        spy = _SpyEmbedder()
        spy.embed(["User purchased a couch on 2024-03-15"], query_mode=False)
        assert spy.calls[0] == ["User purchased a couch on 2024-03-15"]

    def test_default_is_query_mode_false(self) -> None:
        """Old call sites (no kwarg) must keep document-mode behaviour."""
        spy = _SpyEmbedder()
        spy.embed(["claim text"])  # no kwarg → doc mode
        assert spy.calls[0] == ["claim text"]
        assert BGE_M3_QUERY_PREFIX not in spy.calls[0][0]

    def test_empty_list_short_circuit(self) -> None:
        """Empty-input branch must short-circuit before the backend is called,
        regardless of query_mode."""
        spy = _SpyEmbedder()
        assert spy.embed([], query_mode=True) == []
        assert spy.embed([], query_mode=False) == []
        assert spy.calls == []  # never reached the backend

    def test_list_semantics_all_prefixed(self) -> None:
        spy = _SpyEmbedder()
        spy.embed(["a", "b", "c"], query_mode=True)
        assert spy.calls[0] == [
            BGE_M3_QUERY_PREFIX + "a",
            BGE_M3_QUERY_PREFIX + "b",
            BGE_M3_QUERY_PREFIX + "c",
        ]

    def test_query_and_doc_vectors_differ_for_same_text(self) -> None:
        """Core asymmetry guarantee: identical input text produces different
        vectors in query vs doc mode — because the backend receives different
        strings."""
        spy = _SpyEmbedder()
        q_vec = spy.embed(["couch"], query_mode=True)[0]
        d_vec = spy.embed(["couch"], query_mode=False)[0]
        assert q_vec != d_vec


# --- retrieval integration ---------------------------------------------------


def _seed_turn(conn: sqlite3.Connection, session_id: str, text: str) -> str:
    tid = new_turn_id()
    turn = Turn(
        turn_id=tid,
        session_id=session_id,
        speaker=Speaker.PATIENT,
        text=text,
        ts=now_ns(),
    )
    insert_turn(conn, turn)
    return tid


def _seed_claim(
    conn: sqlite3.Connection,
    session_id: str,
    *,
    subject: str = "patient",
    predicate: str = "medical_history",
    value: str,
) -> Any:
    """Seed a claim with clinical_general-pack-valid defaults so tests don't
    trip the predicate allowlist in claims.validate_claim."""
    turn_id = _seed_turn(conn, session_id, f"{subject} {predicate} {value}")
    return insert_claim(
        conn,
        session_id=session_id,
        subject=subject,
        predicate=predicate,
        value=value,
        confidence=0.9,
        source_turn_id=turn_id,
    )


class TestRetrievalThreadsQueryMode:
    def test_retrieve_relevant_claims_prefixes_question(
        self, conn: sqlite3.Connection, session_id: str
    ) -> None:
        spy = _SpyEmbedder()
        claim = _seed_claim(conn, session_id, value="patient had chest pain")
        embed_and_store(conn, claim, client=spy)
        spy.calls.clear()  # only want the retrieval-side calls

        retrieve_relevant_claims(
            conn,
            session_id=session_id,
            question="When did the chest pain start?",
            client=spy,
        )

        # The retrieval-side call is the question embed — must be prefixed.
        assert len(spy.calls) >= 1
        question_texts = spy.calls[-1]
        assert question_texts == [
            BGE_M3_QUERY_PREFIX + "When did the chest pain start?"
        ]

    def test_embed_and_store_stays_unprefixed(
        self, conn: sqlite3.Connection, session_id: str
    ) -> None:
        """embed_and_store writes a claim — must call the backend without
        prefix, so the stored vector matches the doc-mode embedding."""
        spy = _SpyEmbedder()
        claim = _seed_claim(conn, session_id, value="diabetes type 2")
        embed_and_store(conn, claim, client=spy)

        # The write-time embed is a single call. Its text must be the
        # claim_retrieval_text output — no query prefix.
        assert len(spy.calls) == 1
        written_text = spy.calls[0][0]
        assert not written_text.startswith(BGE_M3_QUERY_PREFIX)

    def test_retrieve_hybrid_prefixes_query(
        self, conn: sqlite3.Connection, session_id: str
    ) -> None:
        spy = _SpyEmbedder()
        claim = _seed_claim(conn, session_id, value="hypertension")
        embed_and_store(conn, claim, client=spy)
        spy.calls.clear()

        retrieve_hybrid(
            conn,
            session_id=session_id,
            query="Does the patient have high blood pressure?",
            embedding_client=spy,
        )

        query_texts = spy.calls[-1]
        assert query_texts == [
            BGE_M3_QUERY_PREFIX + "Does the patient have high blood pressure?"
        ]


class TestBackfillStaysDocMode:
    def test_backfill_embeddings_does_not_prefix(
        self, conn: sqlite3.Connection, session_id: str
    ) -> None:
        """Backfill re-embeds existing claims — must use doc mode so the
        new vectors are directly comparable to query-mode question vectors
        via cosine similarity."""
        spy = _SpyEmbedder()
        _seed_claim(conn, session_id, value="asthma")
        backfill_embeddings(conn, session_id, client=spy)

        for batch in spy.calls:
            for text in batch:
                assert not text.startswith(BGE_M3_QUERY_PREFIX), (
                    f"backfill used query prefix on claim text: {text!r}"
                )
