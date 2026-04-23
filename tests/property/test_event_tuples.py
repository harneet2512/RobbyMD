"""Property tests for `EventTuple` projection + temporal-window retrieval.

Locks the four invariants that downstream readers depend on:

1. `claim_to_event` is a lossless projection on the five fields it covers.
2. A claim outside the `valid_at_ts` window is excluded from retrieval
   (i.e. supersession's `valid_until_ts` actually gates retrieval).
3. The same claim *inside* the window is returned.
4. `retrieve_event_tuples` is deterministic — same query + same active set
   produces an identical list of (EventTuple, score) pairs across two calls.

Aligned with Chronos (arXiv:2603.16862) and TEMPR (arXiv:2512.12818):
event-tuple representation + temporal-window filter. RobbyMD's contribution
is that the window comes from deterministic Pass-1 supersession.
"""
from __future__ import annotations

import hashlib
import sqlite3

from src.substrate import open_database
from src.substrate.claims import (
    insert_claim,
    insert_turn,
    new_turn_id,
    now_ns,
)
from src.substrate.event_tuples import EventTuple, claim_to_event
from src.substrate.retrieval import (
    BGE_M3_VERSION_TAG,
    EmbeddingClient,
    embed_and_store,
    retrieve_event_tuples,
)
from src.substrate.schema import Claim, Speaker, Turn


# --- stub embedder -----------------------------------------------------------


class _StubEmbedder(EmbeddingClient):
    """Deterministic per-text 8-dim unit vector — no network, cross-platform.

    Mirrors the stub used in `tests/unit/substrate/test_retrieval.py` so the
    two suites share an identical embedding model and would catch any drift
    in the embedding-version check.
    """

    def __init__(self, version: str = BGE_M3_VERSION_TAG) -> None:
        super().__init__(modal_url="__unused__", model_version=version)
        self._modal_url = None
        self._local_model = object()  # type: ignore[assignment]

    def embed(self, texts: list[str]) -> list[list[float]]:  # type: ignore[override]
        out: list[list[float]] = []
        for t in texts:
            digest = hashlib.sha256(t.encode("utf-8")).digest()
            raw = [((b ^ 0x80) - 128) / 128.0 for b in digest[:8]]
            norm = sum(x * x for x in raw) ** 0.5 or 1.0
            out.append([x / norm for x in raw])
        return out


# --- helpers -----------------------------------------------------------------


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


def _seed_claim(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    source_turn_id: str,
    subject: str = "patient",
    predicate: str = "onset",
    value: str = "3 days",
    valid_from: int | None = None,
    valid_until: int | None = None,
) -> Claim:
    return insert_claim(
        conn,
        session_id=session_id,
        subject=subject,
        predicate=predicate,
        value=value,
        confidence=0.9,
        source_turn_id=source_turn_id,
        valid_from=valid_from,
        valid_until=valid_until,
    )


# --- tests ------------------------------------------------------------------


def test_claim_to_event_preserves_fields() -> None:
    """Round-trip mapping is lossless on the five projected fields."""
    conn = _open()
    sid = "sess_evt_01"
    tid = _seed_turn(conn, session_id=sid, text="pain for 3 days")
    claim = _seed_claim(conn, session_id=sid, source_turn_id=tid)

    et = claim_to_event(claim)
    assert isinstance(et, EventTuple)
    assert et.subject == claim.subject
    assert et.action == claim.predicate
    assert et.obj == claim.value
    assert et.valid_from_ts == claim.valid_from_ts
    assert et.valid_until_ts == claim.valid_until_ts
    assert et.claim_id == claim.claim_id


def test_event_tuple_skipped_when_outside_valid_window() -> None:
    """A claim whose `valid_until_ts` precedes the query timestamp must not
    appear in retrieval. This is the supersession→retrieval guarantee."""
    conn = _open()
    sid = "sess_evt_02"
    tid = _seed_turn(conn, session_id=sid, text="pain for 3 days")

    # Closed window [100, 500). Query at 1000 — strictly after valid_until.
    claim = _seed_claim(
        conn,
        session_id=sid,
        source_turn_id=tid,
        valid_from=100,
        valid_until=500,
    )
    embedder = _StubEmbedder()
    embed_and_store(conn, claim, client=embedder)

    results = retrieve_event_tuples(
        conn,
        session_id=sid,
        query="onset",
        valid_at_ts=1000,
        embedding_client=embedder,
    )
    assert results == []


def test_event_tuple_present_when_inside_valid_window() -> None:
    """The same claim, queried at a timestamp inside its window, *does*
    surface — proves the filter is gated on `valid_at_ts`, not always-on."""
    conn = _open()
    sid = "sess_evt_03"
    tid = _seed_turn(conn, session_id=sid, text="pain for 3 days")

    claim = _seed_claim(
        conn,
        session_id=sid,
        source_turn_id=tid,
        valid_from=100,
        valid_until=500,
    )
    embedder = _StubEmbedder()
    embed_and_store(conn, claim, client=embedder)

    results = retrieve_event_tuples(
        conn,
        session_id=sid,
        query="onset",
        valid_at_ts=300,  # strictly inside [100, 500)
        embedding_client=embedder,
    )
    assert len(results) == 1
    et, score = results[0]
    assert et.claim_id == claim.claim_id
    assert et.action == "onset"
    assert -1.01 <= score <= 1.01

    # Also: querying with no valid_at_ts returns the claim regardless of
    # window (filter is opt-in).
    no_filter = retrieve_event_tuples(
        conn,
        session_id=sid,
        query="onset",
        embedding_client=embedder,
    )
    assert len(no_filter) == 1
    assert no_filter[0][0].claim_id == claim.claim_id


def test_event_tuple_retrieval_deterministic() -> None:
    """Same query + same active set → identical (EventTuple, score) list twice
    in a row. Locks the deterministic-substrate invariant (rules.md §3.6)."""
    conn = _open()
    sid = "sess_evt_04"

    # Three claims with distinct (subject, predicate, value) to avoid Pass-1
    # supersession collapsing them into one survivor.
    embedder = _StubEmbedder()
    for predicate, value in (
        ("onset", "3 days"),
        ("severity", "7 of 10"),
        ("location", "substernal"),
    ):
        tid = _seed_turn(
            conn, session_id=sid, text=f"{predicate} {value}"
        )
        claim = _seed_claim(
            conn,
            session_id=sid,
            source_turn_id=tid,
            predicate=predicate,
            value=value,
        )
        embed_and_store(conn, claim, client=embedder)

    a = retrieve_event_tuples(
        conn,
        session_id=sid,
        query="how bad is the pain",
        top_k=8,
        embedding_client=embedder,
    )
    b = retrieve_event_tuples(
        conn,
        session_id=sid,
        query="how bad is the pain",
        top_k=8,
        embedding_client=embedder,
    )

    # Length, order, claim_ids, and scores must all match.
    assert len(a) == len(b) == 3
    assert [(et.claim_id, round(s, 9)) for et, s in a] == [
        (et.claim_id, round(s, 9)) for et, s in b
    ]
