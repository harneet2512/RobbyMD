"""Property tests for the multi-signal retrieval-fusion head.

Worker 4 — `feature/retrieval-fusion`. Locks down the invariants of:

- `claim_metadata` sidecar population on `insert_claim`
- `temporal_bin` derivation (YYYY-Qn from `valid_from_ts`)
- `retrieve_hybrid()` determinism + signal effects (entity, temporal)

Aligned with Hindsight TEMPR (arXiv:2512.12818); fusion via Reciprocal Rank
Fusion (Bruch et al., ACM TOIS 2023, doi:10.1145/3596512).
"""
from __future__ import annotations

import hashlib
import sqlite3

from src.substrate import open_database
from src.substrate.claims import (
    insert_claim,
    insert_turn,
    new_claim_id,
    new_turn_id,
    now_ns,
)
from src.substrate.retrieval import (
    BGE_M3_VERSION_TAG,
    EmbeddingClient,
    embed_and_store,
    retrieve_hybrid,
)
from src.substrate.schema import Speaker, Turn


# --- stub embedder (matches tests/unit/substrate/test_retrieval.py) ----------


class _StubEmbedder(EmbeddingClient):
    """Deterministic per-text 8-dim unit vector — same scheme as the unit tests.

    Hash-based so identical text → identical vector, cross-platform stable.
    Never reaches the network.
    """

    def __init__(self, version: str = BGE_M3_VERSION_TAG) -> None:
        super().__init__(modal_url="__unused__", model_version=version)
        self._modal_url = None
        self._local_model = object()  # block local fallback

    def embed(self, texts: list[str], *, query_mode: bool = False) -> list[list[float]]:  # type: ignore[override]
        del query_mode
        out: list[list[float]] = []
        for t in texts:
            digest = hashlib.sha256(t.encode("utf-8")).digest()
            raw = [((b ^ 0x80) - 128) / 128.0 for b in digest[:8]]
            norm = sum(x * x for x in raw) ** 0.5 or 1.0
            out.append([x / norm for x in raw])
        return out


def _open() -> sqlite3.Connection:
    return open_database(":memory:")


def _seed_turn(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    text: str,
    speaker: Speaker = Speaker.PATIENT,
) -> str:
    tid = new_turn_id()
    insert_turn(
        conn,
        Turn(
            turn_id=tid,
            session_id=session_id,
            speaker=speaker,
            text=text,
            ts=now_ns(),
        ),
    )
    return tid


# --- claim_metadata population ----------------------------------------------


def test_claim_metadata_populated_on_insert() -> None:
    """After `insert_claim`, the `claim_metadata` row exists with correct fields."""
    conn = _open()
    sid = "sess_meta_01"
    tid = _seed_turn(conn, session_id=sid, text="pain for 3 days")
    c = insert_claim(
        conn,
        session_id=sid,
        subject="patient",
        predicate="onset",
        value="3 days",
        confidence=0.9,
        source_turn_id=tid,
    )
    row = conn.execute(
        "SELECT entity_key, predicate_family, temporal_bin FROM claim_metadata"
        " WHERE claim_id = ?",
        (c.claim_id,),
    ).fetchone()
    assert row is not None
    assert row["entity_key"] == "patient"
    assert row["predicate_family"] == "onset"
    # Default valid_from = now_ns(), so temporal_bin is populated (not NULL).
    assert row["temporal_bin"] is not None


def test_temporal_bin_format_yyyyqn() -> None:
    """A claim with `valid_from_ts` in March 2024 yields temporal_bin == '2024-Q1'."""
    conn = _open()
    sid = "sess_meta_02"
    tid = _seed_turn(conn, session_id=sid, text="pain")
    # 2024-03-15T00:00:00Z = 1710460800 unix seconds.
    march_2024_ns = 1710460800 * 10**9
    c = insert_claim(
        conn,
        session_id=sid,
        subject="patient",
        predicate="onset",
        value="x",
        confidence=0.9,
        source_turn_id=tid,
        valid_from=march_2024_ns,
    )
    row = conn.execute(
        "SELECT temporal_bin FROM claim_metadata WHERE claim_id = ?",
        (c.claim_id,),
    ).fetchone()
    assert row["temporal_bin"] == "2024-Q1"


def test_temporal_bin_null_when_valid_from_null() -> None:
    """Direct-SQL claim insert with NULL valid_from_ts produces NULL temporal_bin.

    `insert_claim` defaults `valid_from = created_ts`, so to test the NULL
    branch we bypass it with a raw INSERT.
    """
    conn = _open()
    sid = "sess_meta_03"
    tid = _seed_turn(conn, session_id=sid, text="pain")
    cid = new_claim_id()
    ts = now_ns()
    conn.execute(
        "INSERT INTO claims (claim_id, session_id, subject, predicate, value,"
        " confidence, source_turn_id, status, created_ts, valid_from_ts)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)",
        (cid, sid, "patient", "onset", "x", 0.9, tid, "active", ts),
    )
    # Mirror the metadata population using the helper directly.
    from src.substrate.claims import _temporal_bin

    conn.execute(
        "INSERT INTO claim_metadata (claim_id, entity_key, predicate_family,"
        " temporal_bin) VALUES (?, ?, ?, ?)",
        (cid, "patient", "onset", _temporal_bin(None)),
    )
    row = conn.execute(
        "SELECT temporal_bin FROM claim_metadata WHERE claim_id = ?", (cid,)
    ).fetchone()
    assert row["temporal_bin"] is None


# --- retrieve_hybrid behaviour ----------------------------------------------


def _seed_claim_with_embedding(
    conn: sqlite3.Connection,
    embedder: _StubEmbedder,
    *,
    session_id: str,
    subject: str,
    predicate: str,
    value: str,
    valid_from: int | None = None,
):
    tid = _seed_turn(conn, session_id=session_id, text=f"{subject} {predicate} {value}")
    c = insert_claim(
        conn,
        session_id=session_id,
        subject=subject,
        predicate=predicate,
        value=value,
        confidence=0.9,
        source_turn_id=tid,
        valid_from=valid_from,
    )
    embed_and_store(conn, c, client=embedder)
    return c


def test_retrieve_hybrid_deterministic() -> None:
    """Same query against the same active set produces an identical ranked list twice."""
    conn = _open()
    sid = "sess_hybrid_det"
    embedder = _StubEmbedder()
    for predicate, value in (
        ("onset", "3 days"),
        ("character", "sharp"),
        ("severity", "8/10"),
        ("location", "chest"),
    ):
        _seed_claim_with_embedding(
            conn, embedder, session_id=sid, subject="patient",
            predicate=predicate, value=value,
        )
    a = retrieve_hybrid(
        conn, session_id=sid, query="chest pain severity",
        embedding_client=embedder,
    )
    b = retrieve_hybrid(
        conn, session_id=sid, query="chest pain severity",
        embedding_client=embedder,
    )
    assert [(c.claim_id, round(s, 12)) for c, s in a] == [
        (c.claim_id, round(s, 12)) for c, s in b
    ]


def test_retrieve_hybrid_entity_hint_promotes_match() -> None:
    """A claim whose entity_key matches `entity_hint` ranks above one that does not.

    Two claims with the same predicate but different subjects; the matching
    one should rank first when the entity weight is non-zero.
    """
    conn = _open()
    sid = "sess_hybrid_ent"
    embedder = _StubEmbedder()
    matching = _seed_claim_with_embedding(
        conn, embedder, session_id=sid, subject="patient",
        predicate="medical_history", value="hypertension",
    )
    non_matching = _seed_claim_with_embedding(
        conn, embedder, session_id=sid, subject="father",
        predicate="medical_history", value="hypertension",
    )
    out = retrieve_hybrid(
        conn,
        session_id=sid,
        query="any history",
        entity_hint="patient",
        embedding_client=embedder,
        weights=(0.0, 1.0, 0.0),  # isolate the entity signal
    )
    ranked_ids = [c.claim_id for c, _ in out]
    assert matching.claim_id in ranked_ids
    assert non_matching.claim_id in ranked_ids
    assert ranked_ids.index(matching.claim_id) < ranked_ids.index(non_matching.claim_id)


def test_retrieve_hybrid_temporal_recency_breaks_tie() -> None:
    """Two claims with identical semantic + entity scores: more-recent ranks first.

    Achieved by giving them the same retrieval text (so the stub embedder
    produces identical vectors) and zero entity weight, so only the temporal
    signal differentiates.
    """
    conn = _open()
    sid = "sess_hybrid_tmp"
    embedder = _StubEmbedder()
    older = _seed_claim_with_embedding(
        conn, embedder, session_id=sid, subject="patient",
        predicate="onset", value="3 days",
        valid_from=1_000_000_000_000_000_000,
    )
    newer = _seed_claim_with_embedding(
        conn, embedder, session_id=sid, subject="patient",
        predicate="onset", value="3 days",
        valid_from=2_000_000_000_000_000_000,
    )
    out = retrieve_hybrid(
        conn,
        session_id=sid,
        query="onset",
        embedding_client=embedder,
        weights=(0.0, 0.0, 1.0),  # isolate the temporal signal
    )
    ranked_ids = [c.claim_id for c, _ in out]
    assert ranked_ids.index(newer.claim_id) < ranked_ids.index(older.claim_id)
