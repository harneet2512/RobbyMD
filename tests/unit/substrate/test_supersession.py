"""Supersession Pass 1 + Pass 2 invariants."""
from __future__ import annotations

import sqlite3

from src.substrate.claims import get_claim, insert_claim, list_active_claims
from src.substrate.schema import ClaimStatus, EdgeType, Speaker
from src.substrate.supersession import detect_pass1, record_clinician_dismissal
from src.substrate.supersession_semantic import (
    NullEmbedder,
    SemanticSupersession,
    cosine,
    identity_text,
)
from tests.unit.substrate.conftest import AddTurnFn

# -------------------------------------------------------------- Pass 1 --- #


def test_pass1_same_subject_predicate_different_value_creates_edge(
    conn: sqlite3.Connection, session_id: str, add_turn: AddTurnFn
) -> None:
    t1 = add_turn("pain for three days now", speaker=Speaker.PATIENT)
    t2 = add_turn("actually it started four days ago", speaker=Speaker.PATIENT)
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
    edge = detect_pass1(conn, c2)
    assert edge is not None
    assert edge.old_claim_id == c1.claim_id
    assert edge.new_claim_id == c2.claim_id
    # Patient → Patient = PATIENT_CORRECTION.
    assert edge.edge_type is EdgeType.PATIENT_CORRECTION
    # Old claim now marked superseded.
    refreshed = get_claim(conn, c1.claim_id)
    assert refreshed is not None and refreshed.status is ClaimStatus.SUPERSEDED


def test_pass1_physician_speaking_yields_physician_confirm(
    conn: sqlite3.Connection, session_id: str, add_turn: AddTurnFn
) -> None:
    t1 = add_turn("had pain for three days", speaker=Speaker.PATIENT)
    t2 = add_turn("documenting onset as five days", speaker=Speaker.PHYSICIAN)
    insert_claim(
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
        value="5 days",
        confidence=0.95,
        source_turn_id=t2,
    )
    edge = detect_pass1(conn, c2)
    assert edge is not None
    assert edge.edge_type is EdgeType.PHYSICIAN_CONFIRM


def test_pass1_refines_fires_on_strict_subset(
    conn: sqlite3.Connection, session_id: str, add_turn: AddTurnFn
) -> None:
    # GT v2 study §2.3 — new value is narrower (subset) → REFINES.
    t1 = add_turn("radiation to left arm and jaw", speaker=Speaker.PATIENT)
    t2 = add_turn("only the left arm, not the jaw", speaker=Speaker.PATIENT)
    insert_claim(
        conn,
        session_id=session_id,
        subject="patient",
        predicate="radiation",
        value="left arm jaw",
        confidence=0.9,
        source_turn_id=t1,
    )
    c2 = insert_claim(
        conn,
        session_id=session_id,
        subject="patient",
        predicate="radiation",
        value="left arm",
        confidence=0.9,
        source_turn_id=t2,
    )
    edge = detect_pass1(conn, c2)
    assert edge is not None
    assert edge.edge_type is EdgeType.REFINES


def test_pass1_same_value_is_idempotent(
    conn: sqlite3.Connection, session_id: str, add_turn: AddTurnFn
) -> None:
    t1 = add_turn("onset three days ago", speaker=Speaker.PATIENT)
    t2 = add_turn("yes three days", speaker=Speaker.PATIENT)
    insert_claim(
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
        value="3 days",
        confidence=0.9,
        source_turn_id=t2,
    )
    assert detect_pass1(conn, c2) is None


def test_pass1_within_turn_never_fires(
    conn: sqlite3.Connection, session_id: str, add_turn: AddTurnFn
) -> None:
    # Eng_doc.md §5.3: guard — two claims from the same turn are additive.
    t1 = add_turn("sharp pain, dull ache both at once now", speaker=Speaker.PATIENT)
    insert_claim(
        conn,
        session_id=session_id,
        subject="patient",
        predicate="character",
        value="sharp",
        confidence=0.9,
        source_turn_id=t1,
    )
    c2 = insert_claim(
        conn,
        session_id=session_id,
        subject="patient",
        predicate="character",
        value="dull",
        confidence=0.9,
        source_turn_id=t1,
    )
    assert detect_pass1(conn, c2) is None


def test_pass1_different_predicate_no_edge(
    conn: sqlite3.Connection, session_id: str, add_turn: AddTurnFn
) -> None:
    t1 = add_turn("three days ago it began", speaker=Speaker.PATIENT)
    t2 = add_turn("sharp character today", speaker=Speaker.PATIENT)
    insert_claim(
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
        predicate="character",
        value="sharp",
        confidence=0.9,
        source_turn_id=t2,
    )
    assert detect_pass1(conn, c2) is None


def test_clinician_dismissal_records_edge(
    conn: sqlite3.Connection, session_id: str, add_turn: AddTurnFn
) -> None:
    t1 = add_turn("patient mentions something wrong", speaker=Speaker.PATIENT)
    t2 = add_turn("physician clarifies correct finding", speaker=Speaker.PHYSICIAN)
    c1 = insert_claim(
        conn,
        session_id=session_id,
        subject="patient",
        predicate="medication",
        value="tylenol",
        confidence=0.8,
        source_turn_id=t1,
    )
    c2 = insert_claim(
        conn,
        session_id=session_id,
        subject="patient",
        predicate="medication",
        value="metoprolol",
        confidence=0.95,
        source_turn_id=t2,
    )
    edge = record_clinician_dismissal(
        conn,
        dismissed_claim_id=c1.claim_id,
        replacement_claim_id=c2.claim_id,
    )
    assert edge is not None
    assert edge.edge_type is EdgeType.DISMISSED_BY_CLINICIAN
    refreshed = get_claim(conn, c1.claim_id)
    assert refreshed is not None and refreshed.status is ClaimStatus.DISMISSED


# -------------------------------------------------------------- Pass 2 --- #


def test_cosine_identity_of_equal_vectors_is_one() -> None:
    v = [1.0, 2.0, 3.0]
    assert abs(cosine(v, v) - 1.0) < 1e-12


def test_cosine_zero_vectors_returns_zero() -> None:
    assert cosine([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_identity_text_excludes_value_field() -> None:
    """gt_v2_study_notes §3.7 — identity embedding must NOT include value."""
    from src.substrate.schema import Claim, ClaimStatus

    claim = Claim(
        claim_id="c1",
        session_id="s1",
        subject="patient",
        predicate="onset",
        value="3 days",
        value_normalised=None,
        confidence=0.9,
        source_turn_id="t1",
        status=ClaimStatus.ACTIVE,
        created_ts=0,
    )
    text = identity_text(claim, "some surrounding turn context here")
    assert "3 days" not in text, "value must not leak into identity embedding"
    assert "patient" in text and "onset" in text


def test_pass2_null_embedder_is_deterministic(
    conn: sqlite3.Connection, session_id: str, add_turn: AddTurnFn
) -> None:
    # With NullEmbedder, cosine is always 1.0. If Pass 1 has NOT fired (e.g.
    # because the subjects differ) Pass 2 will supersede based on predicate
    # identity alone. This confirms the hook is wired end-to-end.
    t1 = add_turn("pulmonary embolism started two days ago", speaker=Speaker.PATIENT)
    t2 = add_turn("the PE began yesterday actually", speaker=Speaker.PATIENT)
    c1 = insert_claim(
        conn,
        session_id=session_id,
        subject="pulmonary embolism",
        predicate="onset",
        value="2 days",
        confidence=0.9,
        source_turn_id=t1,
    )
    c2 = insert_claim(
        conn,
        session_id=session_id,
        subject="pe",
        predicate="onset",
        value="1 day",
        confidence=0.9,
        source_turn_id=t2,
    )
    # Pass 1 won't match because normalised subjects differ ("pulmonary_embolism" vs "pe").
    assert detect_pass1(conn, c2) is None

    sem = SemanticSupersession(NullEmbedder())
    edge = sem.detect(conn, c2, new_turn_text="the PE began yesterday actually")
    assert edge is not None
    assert edge.edge_type is EdgeType.SEMANTIC_REPLACE
    assert edge.identity_score is not None
    assert abs(edge.identity_score - 1.0) < 1e-6

    refreshed = get_claim(conn, c1.claim_id)
    assert refreshed is not None and refreshed.status is ClaimStatus.SUPERSEDED


def test_pass2_respects_threshold_when_no_match(
    conn: sqlite3.Connection, session_id: str, add_turn: AddTurnFn
) -> None:
    t1 = add_turn("first statement about onset", speaker=Speaker.PATIENT)
    t2 = add_turn("second statement different predicate", speaker=Speaker.PATIENT)
    insert_claim(
        conn,
        session_id=session_id,
        subject="patient",
        predicate="onset",
        value="3 days",
        confidence=0.9,
        source_turn_id=t1,
    )
    # Different predicate → Pass 2 candidate set is empty → None.
    c2 = insert_claim(
        conn,
        session_id=session_id,
        subject="patient",
        predicate="severity",
        value="7 of 10",
        confidence=0.9,
        source_turn_id=t2,
    )
    sem = SemanticSupersession(NullEmbedder())
    assert sem.detect(conn, c2) is None


def test_active_projection_excludes_superseded_after_pass1(
    conn: sqlite3.Connection, session_id: str, add_turn: AddTurnFn
) -> None:
    t1 = add_turn("first statement today", speaker=Speaker.PATIENT)
    t2 = add_turn("corrected statement today", speaker=Speaker.PATIENT)
    insert_claim(
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
    detect_pass1(conn, c2)
    active = list_active_claims(conn, session_id)
    assert [c.claim_id for c in active] == [c2.claim_id]
