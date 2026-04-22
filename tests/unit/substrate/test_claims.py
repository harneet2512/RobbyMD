"""Claims CRUD + validation invariants."""
from __future__ import annotations

import sqlite3

import pytest

from src.substrate.claims import (
    ClaimValidationError,
    find_same_identity_claim,
    get_claim,
    insert_claim,
    list_active_claims,
    list_claims_for_turn,
    set_claim_status,
)
from src.substrate.schema import ClaimStatus, Speaker
from tests.unit.substrate.conftest import AddTurnFn


def test_insert_claim_roundtrip(
    conn: sqlite3.Connection, session_id: str, add_turn: AddTurnFn
) -> None:
    tid = add_turn("chest pain for three days")
    claim = insert_claim(
        conn,
        session_id=session_id,
        subject="patient",
        predicate="onset",
        value="3 days",
        confidence=0.9,
        source_turn_id=tid,
        char_start=0,
        char_end=len("chest pain"),
    )
    loaded = get_claim(conn, claim.claim_id)
    assert loaded == claim
    assert loaded is not None
    assert loaded.char_start == 0
    assert loaded.char_end == len("chest pain")


def test_insert_claim_rejects_unknown_predicate(
    conn: sqlite3.Connection, session_id: str, add_turn: AddTurnFn
) -> None:
    tid = add_turn("patient has some new thing")
    with pytest.raises(ClaimValidationError, match="not in closed family"):
        insert_claim(
            conn,
            session_id=session_id,
            subject="patient",
            predicate="vibes",  # not in PREDICATE_FAMILIES
            value="3",
            confidence=0.9,
            source_turn_id=tid,
        )


def test_insert_claim_rejects_bad_confidence(
    conn: sqlite3.Connection, session_id: str, add_turn: AddTurnFn
) -> None:
    tid = add_turn("stabbing chest pain here")
    with pytest.raises(ClaimValidationError, match="confidence"):
        insert_claim(
            conn,
            session_id=session_id,
            subject="patient",
            predicate="onset",
            value="v",
            confidence=1.5,
            source_turn_id=tid,
        )


def test_insert_claim_rejects_missing_source_turn(
    conn: sqlite3.Connection, session_id: str
) -> None:
    with pytest.raises(ClaimValidationError, match="does not reference any turn"):
        insert_claim(
            conn,
            session_id=session_id,
            subject="patient",
            predicate="onset",
            value="3 days",
            confidence=0.9,
            source_turn_id="tu_does_not_exist",
        )


def test_insert_claim_rejects_invalid_span(
    conn: sqlite3.Connection, session_id: str, add_turn: AddTurnFn
) -> None:
    text = "chest pain here"
    tid = add_turn(text)
    # End before start
    with pytest.raises(ClaimValidationError, match="invalid span"):
        insert_claim(
            conn,
            session_id=session_id,
            subject="patient",
            predicate="onset",
            value="x",
            confidence=0.9,
            source_turn_id=tid,
            char_start=5,
            char_end=2,
        )
    # End beyond turn text length
    with pytest.raises(ClaimValidationError, match="exceeds source turn length"):
        insert_claim(
            conn,
            session_id=session_id,
            subject="patient",
            predicate="onset",
            value="x",
            confidence=0.9,
            source_turn_id=tid,
            char_start=0,
            char_end=len(text) + 10,
        )


def test_list_active_claims_excludes_superseded(
    conn: sqlite3.Connection, session_id: str, add_turn: AddTurnFn
) -> None:
    tid = add_turn("something clinical here")
    c1 = insert_claim(
        conn,
        session_id=session_id,
        subject="patient",
        predicate="onset",
        value="3 days",
        confidence=0.9,
        source_turn_id=tid,
    )
    set_claim_status(conn, c1.claim_id, ClaimStatus.SUPERSEDED)
    active = list_active_claims(conn, session_id)
    assert c1.claim_id not in [a.claim_id for a in active]


def test_list_active_claims_includes_confirmed(
    conn: sqlite3.Connection, session_id: str, add_turn: AddTurnFn
) -> None:
    tid = add_turn("something clinical here")
    c1 = insert_claim(
        conn,
        session_id=session_id,
        subject="patient",
        predicate="onset",
        value="3 days",
        confidence=0.9,
        source_turn_id=tid,
        status=ClaimStatus.CONFIRMED,
    )
    active = list_active_claims(conn, session_id)
    assert c1.claim_id in [a.claim_id for a in active]


def test_list_claims_for_turn(
    conn: sqlite3.Connection, session_id: str, add_turn: AddTurnFn
) -> None:
    tid = add_turn("pain three days, sharp character")
    c1 = insert_claim(
        conn,
        session_id=session_id,
        subject="patient",
        predicate="onset",
        value="3 days",
        confidence=0.9,
        source_turn_id=tid,
    )
    c2 = insert_claim(
        conn,
        session_id=session_id,
        subject="patient",
        predicate="character",
        value="sharp",
        confidence=0.9,
        source_turn_id=tid,
    )
    got = list_claims_for_turn(conn, tid)
    assert [c.claim_id for c in got] == [c1.claim_id, c2.claim_id]


def test_find_same_identity_claim_normalises_subject(
    conn: sqlite3.Connection, session_id: str, add_turn: AddTurnFn
) -> None:
    tid = add_turn("patient has some problem now")
    c1 = insert_claim(
        conn,
        session_id=session_id,
        subject="Patient Doe",  # should normalise to patient_doe
        predicate="onset",
        value="3 days",
        confidence=0.9,
        source_turn_id=tid,
    )
    # Search with a differently-cased variant — must still find c1.
    match = find_same_identity_claim(
        conn,
        session_id=session_id,
        subject="patient doe",
        predicate="onset",
    )
    assert match is not None and match.claim_id == c1.claim_id


def test_normalisation_rejects_empty_subject(
    conn: sqlite3.Connection, session_id: str, add_turn: AddTurnFn
) -> None:
    tid = add_turn("something happened today")
    with pytest.raises(ClaimValidationError, match="subject"):
        insert_claim(
            conn,
            session_id=session_id,
            subject="   ",
            predicate="onset",
            value="v",
            confidence=0.9,
            source_turn_id=tid,
        )


def test_set_claim_status_unknown_claim_raises(
    conn: sqlite3.Connection,
) -> None:
    with pytest.raises(ClaimValidationError, match="not found"):
        set_claim_status(conn, "cl_nope", ClaimStatus.DISMISSED)


def test_turn_speaker_persisted_correctly(
    conn: sqlite3.Connection, add_turn: AddTurnFn
) -> None:
    tid = add_turn("physician speaking now here", speaker=Speaker.PHYSICIAN)
    row = conn.execute("SELECT speaker FROM turns WHERE turn_id=?", (tid,)).fetchone()
    assert row["speaker"] == "physician"
