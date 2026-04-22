"""Projections + provenance utilities."""
from __future__ import annotations

import sqlite3

import pytest

from src.substrate.claims import insert_claim
from src.substrate.projections import (
    BRANCH_NAMES,
    claims_grouped_by_subject_predicate,
    per_branch_projection,
    rebuild_active_projection,
)
from src.substrate.provenance import (
    NoteSentenceValidationError,
    claim_ids_for_turn,
    insert_note_sentence,
    note_sentence_ids_for_claim,
    span_for_claim,
    turn_id_for_sentence,
)
from src.substrate.schema import Claim, NoteSection, Speaker
from src.substrate.supersession import detect_pass1
from tests.unit.substrate.conftest import AddTurnFn

# ------------------------------------------------------------ projections --- #


def test_branch_names_is_the_contract() -> None:
    assert BRANCH_NAMES == ("cardiac", "pulmonary", "msk", "gi")


def test_rebuild_active_projection_idempotent(
    conn: sqlite3.Connection, session_id: str, add_turn: AddTurnFn
) -> None:
    t1 = add_turn("pain three days ago today", speaker=Speaker.PATIENT)
    t2 = add_turn("sharp character persists today", speaker=Speaker.PATIENT)
    insert_claim(
        conn,
        session_id=session_id,
        subject="patient",
        predicate="onset",
        value="3 days",
        confidence=0.9,
        source_turn_id=t1,
    )
    insert_claim(
        conn,
        session_id=session_id,
        subject="patient",
        predicate="character",
        value="sharp",
        confidence=0.9,
        source_turn_id=t2,
    )
    first = rebuild_active_projection(conn, session_id)
    second = rebuild_active_projection(conn, session_id)
    assert first == second
    assert {c.predicate for c in first.claims} == {"onset", "character"}


def test_projection_by_predicate_groups_correctly(
    conn: sqlite3.Connection, session_id: str, add_turn: AddTurnFn
) -> None:
    t1 = add_turn("stabbing then dull character", speaker=Speaker.PATIENT)
    t2 = add_turn("followed by burning quality later", speaker=Speaker.PATIENT)
    insert_claim(
        conn,
        session_id=session_id,
        subject="patient",
        predicate="character",
        value="stabbing",
        confidence=0.9,
        source_turn_id=t1,
    )
    insert_claim(
        conn,
        session_id=session_id,
        subject="other",
        predicate="character",
        value="burning",
        confidence=0.9,
        source_turn_id=t2,
    )
    proj = rebuild_active_projection(conn, session_id)
    groups = proj.by_predicate
    assert set(groups.keys()) == {"character"}
    assert len(groups["character"]) == 2


def test_per_branch_projection_rejects_unknown_branch() -> None:
    from src.substrate.projections import ActiveProjection

    proj = ActiveProjection(session_id="s1", claims=())
    with pytest.raises(ValueError, match="unknown branch"):
        per_branch_projection(proj, "not_a_branch", lambda _c: True)


def test_per_branch_projection_filters_claims(
    conn: sqlite3.Connection, session_id: str, add_turn: AddTurnFn
) -> None:
    t1 = add_turn("stabbing left sided pain today", speaker=Speaker.PATIENT)
    insert_claim(
        conn,
        session_id=session_id,
        subject="patient",
        predicate="character",
        value="stabbing",
        confidence=0.9,
        source_turn_id=t1,
    )
    insert_claim(
        conn,
        session_id=session_id,
        subject="patient",
        predicate="location",
        value="left",
        confidence=0.9,
        source_turn_id=t1,
    )
    proj = rebuild_active_projection(conn, session_id)
    # Hypothetical cardiac matcher cares only about character+location pairs.
    cardiac = per_branch_projection(
        proj, "cardiac", lambda c: c.predicate in {"character", "location"}
    )
    assert len(cardiac.claims) == 2


def test_claims_grouped_by_subject_predicate_keeps_newest(
    conn: sqlite3.Connection, session_id: str, add_turn: AddTurnFn
) -> None:
    t1 = add_turn("first statement about onset", speaker=Speaker.PATIENT)
    t2 = add_turn("corrected statement about onset", speaker=Speaker.PATIENT)
    c1 = insert_claim(
        conn,
        session_id=session_id,
        subject="patient",
        predicate="onset",
        value="3 days",
        confidence=0.9,
        source_turn_id=t1,
    )
    c2 = insert_claim(
        conn,
        session_id=session_id,
        subject="patient",
        predicate="onset",
        value="4 days",
        confidence=0.9,
        source_turn_id=t2,
    )
    out = claims_grouped_by_subject_predicate([c1, c2])
    assert out[("patient", "onset")].claim_id == c2.claim_id


# ----------------------------------------------------------- provenance --- #


def _mk_claim_and_turn(
    conn: sqlite3.Connection, session_id: str, add_turn: AddTurnFn, text: str
) -> tuple[str, Claim]:
    tid = add_turn(text)
    claim = insert_claim(
        conn,
        session_id=session_id,
        subject="patient",
        predicate="onset",
        value="3 days",
        confidence=0.9,
        source_turn_id=tid,
        char_start=0,
        char_end=5,
    )
    return tid, claim


def test_forward_and_back_links_roundtrip(
    conn: sqlite3.Connection, session_id: str, add_turn: AddTurnFn
) -> None:
    tid, claim = _mk_claim_and_turn(
        conn, session_id, add_turn, "chest pain three days now"
    )
    # Insert a note sentence grounded on this claim.
    sentence = insert_note_sentence(
        conn,
        session_id=session_id,
        section=NoteSection.SUBJECTIVE,
        ordinal=1,
        text="Patient reports onset of chest pain 3 days prior.",
        source_claim_ids=[claim.claim_id],
    )
    # Forward: claims_for_turn -> [claim_id]
    assert claim_ids_for_turn(conn, tid) == [claim.claim_id]
    # Forward: note_sentence_ids_for_claim -> [sentence_id]
    assert note_sentence_ids_for_claim(conn, claim.claim_id) == [sentence.sentence_id]
    # Back: turn_id_for_sentence -> tid
    assert turn_id_for_sentence(conn, sentence.sentence_id) == tid
    # Span: substring retrieval
    span = span_for_claim(conn, claim.claim_id)
    assert span is not None
    assert (span.char_start, span.char_end) == (0, 5)


def test_note_sentence_without_provenance_is_rejected(
    conn: sqlite3.Connection, session_id: str
) -> None:
    with pytest.raises(NoteSentenceValidationError):
        insert_note_sentence(
            conn,
            session_id=session_id,
            section=NoteSection.SUBJECTIVE,
            ordinal=1,
            text="Orphan sentence.",
            source_claim_ids=[],
        )


def test_note_sentence_unknown_claim_id_is_rejected(
    conn: sqlite3.Connection, session_id: str
) -> None:
    with pytest.raises(NoteSentenceValidationError, match="unknown"):
        insert_note_sentence(
            conn,
            session_id=session_id,
            section=NoteSection.ASSESSMENT,
            ordinal=1,
            text="Assessment sentence.",
            source_claim_ids=["cl_does_not_exist"],
        )


def test_span_for_claim_after_supersession_still_resolves(
    conn: sqlite3.Connection, session_id: str, add_turn: AddTurnFn
) -> None:
    # Supersession does not delete claims; provenance must still work.
    t1 = add_turn("pain three days ago started", speaker=Speaker.PATIENT)
    t2 = add_turn("actually four days now today", speaker=Speaker.PATIENT)
    c1 = insert_claim(
        conn,
        session_id=session_id,
        subject="patient",
        predicate="onset",
        value="3 days",
        confidence=0.9,
        source_turn_id=t1,
        char_start=5,
        char_end=15,
    )
    c2 = insert_claim(
        conn,
        session_id=session_id,
        subject="patient",
        predicate="onset",
        value="4 days",
        confidence=0.9,
        source_turn_id=t2,
    )
    detect_pass1(conn, c2)
    # The superseded claim's span must still be queryable.
    span = span_for_claim(conn, c1.claim_id)
    assert span is not None and span.char_start == 5 and span.char_end == 15
