"""Unit tests for the retrieval sidecar + query-conditional ranking.

No network. A stub `EmbeddingClient` replaces the real bge-m3 call with a
deterministic hash-based 8-dim vector so tests run identically across
machines and against identical inputs.
"""
from __future__ import annotations

import hashlib
import sqlite3

import pytest

from src.substrate.claims import insert_claim, set_claim_status
from src.substrate.retrieval import (
    BGE_M3_VERSION_TAG,
    EmbeddingClient,
    RankedClaim,
    backfill_embeddings,
    claim_retrieval_text,
    embed_and_store,
    retrieve_relevant_claims,
)
from src.substrate.schema import ClaimStatus, Speaker


# --- stub embedder -----------------------------------------------------------


class _StubEmbedder(EmbeddingClient):
    """Deterministic per-text 8-dim unit vector.

    Vector is the first 8 float32s derived from a sha256 hash of the text,
    then L2-normalised. Same text → same vector across runs, cross-platform.
    """

    def __init__(self, version: str = BGE_M3_VERSION_TAG) -> None:
        super().__init__(modal_url="__unused__", model_version=version)
        # Disable Modal path by clearing the URL after super().__init__ set it.
        self._modal_url = None
        # Prevent the local fallback from ever running.
        self._local_model = object()  # type: ignore[assignment]

    def embed(self, texts: list[str], *, query_mode: bool = False) -> list[list[float]]:  # type: ignore[override]
        # query_mode is ignored in this stub; existing tests verify retrieval
        # plumbing, not the bge-m3 asymmetric-prefix semantics. A dedicated
        # spy-based test in test_asymmetric_embedding.py verifies prefix threading.
        del query_mode
        out: list[list[float]] = []
        for t in texts:
            digest = hashlib.sha256(t.encode("utf-8")).digest()
            # Interpret first 8 bytes each as a signed 8-bit int, scale to [-1, 1].
            raw = [((b ^ 0x80) - 128) / 128.0 for b in digest[:8]]
            norm = sum(x * x for x in raw) ** 0.5 or 1.0
            out.append([x / norm for x in raw])
        return out


# --- helpers -----------------------------------------------------------------


def _seed_turn(conn: sqlite3.Connection, session_id: str, text: str) -> str:
    from src.substrate.claims import insert_turn, new_turn_id, now_ns
    from src.substrate.schema import Turn

    tid = new_turn_id()
    conn_turn = Turn(
        turn_id=tid,
        session_id=session_id,
        speaker=Speaker.PATIENT,
        text=text,
        ts=now_ns(),
    )
    insert_turn(conn, conn_turn)
    return tid


def _seed_claim(
    conn: sqlite3.Connection,
    session_id: str,
    *,
    subject: str,
    predicate: str,
    value: str,
    turn_text: str | None = None,
):
    turn_id = _seed_turn(conn, session_id, turn_text or f"{subject} {predicate} {value}")
    return insert_claim(
        conn,
        session_id=session_id,
        subject=subject,
        predicate=predicate,
        value=value,
        confidence=0.95,
        source_turn_id=turn_id,
    )


@pytest.fixture(autouse=True)
def _isolated_embed_cache(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pin the embedding cache to a fresh dir per test so cache hits don't leak."""
    cache_dir = tmp_path_factory.mktemp("substrate_embed_cache")
    monkeypatch.setenv("SUBSTRATE_EMBED_CACHE_DIR", str(cache_dir))


@pytest.fixture()
def stub_embedder() -> _StubEmbedder:
    return _StubEmbedder()


# --- tests -------------------------------------------------------------------


class TestEmbedDeterminism:
    def test_same_text_same_vector(self, stub_embedder: _StubEmbedder) -> None:
        v1 = stub_embedder.embed(["onset 3 days"])[0]
        v2 = stub_embedder.embed(["onset 3 days"])[0]
        assert v1 == v2

    def test_different_text_different_vector(self, stub_embedder: _StubEmbedder) -> None:
        v1 = stub_embedder.embed(["onset 3 days"])[0]
        v2 = stub_embedder.embed(["radiation to left arm"])[0]
        assert v1 != v2


class TestEmbedAndStore:
    def test_round_trip(
        self, conn: sqlite3.Connection, session_id: str, stub_embedder: _StubEmbedder
    ) -> None:
        claim = _seed_claim(
            conn, session_id, subject="patient", predicate="onset", value="3 days"
        )
        embed_and_store(conn, claim, client=stub_embedder)
        row = conn.execute(
            "SELECT embedding_model_version FROM claim_embeddings WHERE claim_id = ?",
            (claim.claim_id,),
        ).fetchone()
        assert row is not None
        assert row["embedding_model_version"] == BGE_M3_VERSION_TAG

    def test_failure_does_not_raise(
        self, conn: sqlite3.Connection, session_id: str
    ) -> None:
        class _Boom(EmbeddingClient):
            def __init__(self) -> None:
                super().__init__(modal_url=None)
                self._modal_url = None

            def embed(self, texts: list[str], *, query_mode: bool = False) -> list[list[float]]:  # type: ignore[override]
                del query_mode
                raise RuntimeError("upstream unavailable")

        claim = _seed_claim(
            conn, session_id, subject="patient", predicate="onset", value="3 days"
        )
        # Must not propagate — degrade gracefully so the ingest pipeline keeps going.
        embed_and_store(conn, claim, client=_Boom())
        row = conn.execute(
            "SELECT claim_id FROM claim_embeddings WHERE claim_id = ?",
            (claim.claim_id,),
        ).fetchone()
        assert row is None  # claim left un-embedded as documented


class TestRetrievalTopK:
    def test_top_k_ordering(
        self, conn: sqlite3.Connection, session_id: str, stub_embedder: _StubEmbedder
    ) -> None:
        c1 = _seed_claim(
            conn, session_id, subject="patient", predicate="onset", value="3 days"
        )
        c2 = _seed_claim(
            conn, session_id, subject="patient", predicate="radiation", value="left arm"
        )
        c3 = _seed_claim(
            conn,
            session_id,
            subject="patient",
            predicate="character",
            value="pressure",
        )
        for c in (c1, c2, c3):
            embed_and_store(conn, c, client=stub_embedder)

        # Exact-match question text for c1 should rank it first (dot product
        # of a unit vector with itself = 1).
        results = retrieve_relevant_claims(
            conn,
            session_id=session_id,
            question=claim_retrieval_text(c1),
            k=3,
            client=stub_embedder,
        )
        assert results
        assert results[0].claim.claim_id == c1.claim_id
        assert results[0].similarity_score == pytest.approx(1.0, abs=1e-5)
        # Results are sorted descending.
        scores = [r.similarity_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_empty_substrate(
        self, conn: sqlite3.Connection, session_id: str, stub_embedder: _StubEmbedder
    ) -> None:
        out = retrieve_relevant_claims(
            conn,
            session_id=session_id,
            question="any question",
            client=stub_embedder,
        )
        assert out == []

    def test_empty_question_returns_empty(
        self, conn: sqlite3.Connection, session_id: str, stub_embedder: _StubEmbedder
    ) -> None:
        c = _seed_claim(
            conn, session_id, subject="patient", predicate="onset", value="3 days"
        )
        embed_and_store(conn, c, client=stub_embedder)
        assert (
            retrieve_relevant_claims(
                conn, session_id=session_id, question="   ", client=stub_embedder
            )
            == []
        )


class TestSupersessionFilter:
    def test_superseded_claim_never_ranked(
        self, conn: sqlite3.Connection, session_id: str, stub_embedder: _StubEmbedder
    ) -> None:
        c = _seed_claim(
            conn, session_id, subject="patient", predicate="onset", value="3 days"
        )
        embed_and_store(conn, c, client=stub_embedder)
        # Mark superseded — still has its embedding row, but retrieval must skip.
        set_claim_status(conn, c.claim_id, ClaimStatus.SUPERSEDED)
        out = retrieve_relevant_claims(
            conn,
            session_id=session_id,
            question=claim_retrieval_text(c),
            client=stub_embedder,
        )
        assert out == []

    def test_confirmed_claim_still_ranked(
        self, conn: sqlite3.Connection, session_id: str, stub_embedder: _StubEmbedder
    ) -> None:
        c = _seed_claim(
            conn, session_id, subject="patient", predicate="onset", value="3 days"
        )
        embed_and_store(conn, c, client=stub_embedder)
        set_claim_status(conn, c.claim_id, ClaimStatus.CONFIRMED)
        out = retrieve_relevant_claims(
            conn,
            session_id=session_id,
            question=claim_retrieval_text(c),
            client=stub_embedder,
        )
        assert len(out) == 1
        assert out[0].claim.claim_id == c.claim_id


class TestEmbeddingVersionMismatch:
    def test_cross_version_claim_skipped(
        self, conn: sqlite3.Connection, session_id: str
    ) -> None:
        old_version_client = _StubEmbedder(version="BAAI/bge-m3@legacy")
        c = _seed_claim(
            conn, session_id, subject="patient", predicate="onset", value="3 days"
        )
        embed_and_store(conn, c, client=old_version_client)

        new_version_client = _StubEmbedder(version="BAAI/bge-m3@main")
        out = retrieve_relevant_claims(
            conn,
            session_id=session_id,
            question=claim_retrieval_text(c),
            client=new_version_client,
        )
        assert out == []  # version mismatch → skipped

    def test_backfill_reembeds_on_version_change(
        self, conn: sqlite3.Connection, session_id: str
    ) -> None:
        c = _seed_claim(
            conn, session_id, subject="patient", predicate="onset", value="3 days"
        )
        embed_and_store(conn, c, client=_StubEmbedder(version="v1"))
        # New-version backfill re-embeds because the sidecar row is stale.
        new_client = _StubEmbedder(version="v2")
        n = backfill_embeddings(conn, session_id, client=new_client)
        assert n == 1
        row = conn.execute(
            "SELECT embedding_model_version FROM claim_embeddings WHERE claim_id = ?",
            (c.claim_id,),
        ).fetchone()
        assert row["embedding_model_version"] == "v2"


class TestRankedClaimDataclass:
    def test_fields_populated(
        self, conn: sqlite3.Connection, session_id: str, stub_embedder: _StubEmbedder
    ) -> None:
        c = _seed_claim(
            conn, session_id, subject="patient", predicate="onset", value="3 days"
        )
        embed_and_store(conn, c, client=stub_embedder)
        out = retrieve_relevant_claims(
            conn,
            session_id=session_id,
            question=claim_retrieval_text(c),
            client=stub_embedder,
        )
        assert len(out) == 1
        r: RankedClaim = out[0]
        assert r.claim.claim_id == c.claim_id
        assert 0.0 <= r.similarity_score <= 1.01
        assert r.embedding_model_version == BGE_M3_VERSION_TAG
