"""Temporal-validity invariants — property tests for `valid_from_ts` /
`valid_until_ts` fields on claims and the supersession-driven valid_until
update.

Per `advisory/validation/architecture_validation.md` §3 Claim B (deterministic
supersession is stronger than ESAA's hash-level determinism: it works at the
algorithm level). These tests lock the temporal-window invariants the
canonical claim state must satisfy.

Aligned with Zep (arXiv:2501.13956) and Chronos (arXiv:2603.16862) temporal
representations; differs in that supersession algorithmically modifies
`valid_until_ts` on the superseded claim rather than relying on LLM revision.
"""
from __future__ import annotations

import sqlite3

from src.substrate import open_database
from src.substrate.claims import (
    get_claim,
    get_superseded_by,
    insert_claim,
    insert_turn,
    new_turn_id,
    now_ns,
)
from src.substrate.schema import Speaker, Turn
from src.substrate.supersession import detect_pass1


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


# --------------------------------------------------------------- invariants ---


def test_valid_from_defaults_to_created_ts() -> None:
    """A claim inserted without explicit valid_from has valid_from_ts == created_ts."""
    conn = _open()
    sid = "sess_temporal_01"
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
    assert c.valid_from_ts == c.created_ts
    assert c.valid_until_ts is None


def test_valid_until_after_valid_from_invariant() -> None:
    """Schema CHECK enforces valid_until > valid_from when both set."""
    conn = _open()
    sid = "sess_temporal_02"
    tid = _seed_turn(conn, session_id=sid, text="pain for 3 days")
    # Direct SQL insert with bad window must fail the CHECK.
    import pytest

    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO claims (claim_id, session_id, subject, predicate,"
            " value, confidence, source_turn_id, status, created_ts,"
            " valid_from_ts, valid_until_ts)"
            " VALUES ('c_bad','%s','patient','onset','x',0.9,'%s','active',1,"
            " 100, 50)" % (sid, tid)
        )


def test_insert_claim_rejects_inverted_window() -> None:
    """`insert_claim` raises ClaimValidationError if valid_until <= valid_from."""
    import pytest

    from src.substrate.claims import ClaimValidationError

    conn = _open()
    sid = "sess_temporal_03"
    tid = _seed_turn(conn, session_id=sid, text="pain for 3 days")
    with pytest.raises(ClaimValidationError, match="valid_until"):
        insert_claim(
            conn,
            session_id=sid,
            subject="patient",
            predicate="onset",
            value="3 days",
            confidence=0.9,
            source_turn_id=tid,
            valid_from=200,
            valid_until=100,
        )


def test_supersession_sets_valid_until_to_edge_created_ts() -> None:
    """Pass-1 supersession updates the old claim's valid_until_ts to the
    supersession edge's created_ts. No second clock read."""
    conn = _open()
    sid = "sess_temporal_04"
    t1 = _seed_turn(conn, session_id=sid, text="pain 3 days", speaker=Speaker.PATIENT)
    t2 = _seed_turn(
        conn, session_id=sid, text="actually 4 days", speaker=Speaker.PATIENT
    )
    c1 = insert_claim(
        conn,
        session_id=sid,
        subject="patient",
        predicate="onset",
        value="3 days",
        confidence=0.9,
        source_turn_id=t1,
    )
    assert c1.valid_until_ts is None  # unbounded before supersession

    c2 = insert_claim(
        conn,
        session_id=sid,
        subject="patient",
        predicate="onset",
        value="4 days",
        confidence=0.9,
        source_turn_id=t2,
    )
    edge = detect_pass1(conn, c2)
    assert edge is not None

    # Re-fetch c1: valid_until_ts should now equal edge.created_ts.
    refreshed = get_claim(conn, c1.claim_id)
    assert refreshed is not None
    assert refreshed.valid_until_ts == edge.created_ts
    # And valid_from_ts is unchanged.
    assert refreshed.valid_from_ts == c1.valid_from_ts


def test_get_superseded_by_returns_new_claim_id_after_pass1() -> None:
    """The derived `get_superseded_by()` accessor reads from supersession_edges."""
    conn = _open()
    sid = "sess_temporal_05"
    t1 = _seed_turn(conn, session_id=sid, text="x", speaker=Speaker.PATIENT)
    t2 = _seed_turn(conn, session_id=sid, text="y", speaker=Speaker.PATIENT)
    c1 = insert_claim(
        conn,
        session_id=sid,
        subject="patient",
        predicate="onset",
        value="3 days",
        confidence=0.9,
        source_turn_id=t1,
    )
    c2 = insert_claim(
        conn,
        session_id=sid,
        subject="patient",
        predicate="onset",
        value="4 days",
        confidence=0.9,
        source_turn_id=t2,
    )
    detect_pass1(conn, c2)

    assert get_superseded_by(conn, c1.claim_id) == c2.claim_id
    assert get_superseded_by(conn, c2.claim_id) is None  # head of chain


def test_temporal_window_determinism_across_repeated_runs() -> None:
    """Running the same insert sequence twice produces identical temporal windows."""

    def _run_sequence() -> list[tuple[str, int | None, int | None]]:
        conn = _open()
        sid = "sess_det"
        t1 = _seed_turn(conn, session_id=sid, text="pain 3 days", speaker=Speaker.PATIENT)
        t2 = _seed_turn(
            conn, session_id=sid, text="actually 4 days", speaker=Speaker.PATIENT
        )
        c1 = insert_claim(
            conn,
            session_id=sid,
            subject="patient",
            predicate="onset",
            value="3 days",
            confidence=0.9,
            source_turn_id=t1,
        )
        c2 = insert_claim(
            conn,
            session_id=sid,
            subject="patient",
            predicate="onset",
            value="4 days",
            confidence=0.9,
            source_turn_id=t2,
        )
        detect_pass1(conn, c2)
        # Read both back; capture (subject_predicate_value, valid_from_set?, valid_until_set?)
        # Note: the actual ts values vary across runs (clock-driven), but the
        # *structure* of the windows must be identical: superseded claim has
        # valid_until set; head claim has valid_until None.
        out: list[tuple[str, int | None, int | None]] = []
        for cid in (c1.claim_id, c2.claim_id):
            cc = get_claim(conn, cid)
            assert cc is not None
            out.append(
                (
                    f"{cc.subject}|{cc.predicate}|{cc.value}",
                    1 if cc.valid_from_ts is not None else 0,
                    1 if cc.valid_until_ts is not None else 0,
                )
            )
        return out

    a = _run_sequence()
    b = _run_sequence()
    # Window-presence pattern is identical across runs: c1 has valid_from set
    # and valid_until set (superseded); c2 has valid_from set and valid_until None.
    assert a == b
    assert a == [
        ("patient|onset|3 days", 1, 1),
        ("patient|onset|4 days", 1, 0),
    ]
