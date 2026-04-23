"""Schema-level invariants. These lock the data model contract."""
from __future__ import annotations

import sqlite3

import pytest

from src.substrate import (
    PREDICATE_FAMILIES,
    open_database,
)
from src.substrate.schema import ClaimStatus, EdgeType, NoteSection, Speaker


def test_open_database_creates_all_tables() -> None:
    conn = open_database(":memory:")
    names = {
        row["name"]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert names >= {
        "turns",
        "claims",
        "supersession_edges",
        "decisions",
        "note_sentences",
    }


def test_foreign_keys_are_on(conn: sqlite3.Connection) -> None:
    fk = conn.execute("PRAGMA foreign_keys").fetchone()[0]
    assert fk == 1, "Foreign keys must be ON for referential integrity"


def test_predicate_families_is_closed_14(conn: sqlite3.Connection) -> None:
    # Eng_doc.md §4.2 — closed set, size 14.
    assert len(PREDICATE_FAMILIES) == 14
    assert "onset" in PREDICATE_FAMILIES
    assert "radiation" in PREDICATE_FAMILIES
    assert "custom_family" not in PREDICATE_FAMILIES


def test_speaker_check_constraint_rejects_unknown(conn: sqlite3.Connection) -> None:
    # Direct insert with invalid speaker should fail schema CHECK.
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO turns (turn_id, session_id, speaker, text, ts)"
            " VALUES (?, ?, ?, ?, ?)",
            ("t1", "s1", "nurse", "hi", 1),
        )


def test_edge_type_check_constraint_rejects_unknown(conn: sqlite3.Connection) -> None:
    conn.execute(
        "INSERT INTO turns (turn_id, session_id, speaker, text, ts)"
        " VALUES ('t1', 's1', 'patient', 'hi', 1)"
    )
    conn.execute(
        "INSERT INTO claims (claim_id, session_id, subject, predicate, value,"
        " confidence, source_turn_id, status, created_ts)"
        " VALUES ('c1', 's1', 'pat', 'onset', 'v', 0.9, 't1', 'active', 1)"
    )
    conn.execute(
        "INSERT INTO claims (claim_id, session_id, subject, predicate, value,"
        " confidence, source_turn_id, status, created_ts)"
        " VALUES ('c2', 's1', 'pat', 'onset', 'v2', 0.9, 't1', 'active', 2)"
    )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO supersession_edges"
            " (edge_id, old_claim_id, new_claim_id, edge_type, created_ts)"
            " VALUES ('e1','c1','c2','made_up_type',3)"
        )


def test_enum_values_match_schema_vocabulary() -> None:
    # StrEnum values are what get persisted — keep them aligned with SQL CHECKs.
    assert {s.value for s in Speaker} == {"patient", "physician", "system"}
    assert {s.value for s in ClaimStatus} == {
        "active",
        "superseded",
        "confirmed",
        "dismissed",
        "draft",
        "audited",
    }
    assert {e.value for e in EdgeType} == {
        "patient_correction",
        "physician_confirm",
        "semantic_replace",
        "refines",
        "contradicts",
        "rules_out",
        "dismissed_by_clinician",
    }
    assert {n.value for n in NoteSection} == {"S", "O", "A", "P"}
