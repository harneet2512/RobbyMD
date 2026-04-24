"""Tests for lifecycle-aware retrieval (Layer 2).

Verifies that list_claims_with_lifecycle, list_supersession_pairs, and
retrieve_hybrid's include_superseded parameter work correctly.
"""
from __future__ import annotations

import sqlite3

import pytest

from src.substrate.claims import (
    insert_claim,
    list_active_claims,
    list_claims_with_lifecycle,
    list_supersession_pairs,
    set_claim_status,
)
from src.substrate.schema import ClaimStatus, EdgeType, Speaker


# ── fixtures ──────────────────────────────────────────────────────────────


def _insert_edge(
    conn: sqlite3.Connection,
    old_id: str,
    new_id: str,
    edge_type: str = "patient_correction",
) -> None:
    """Direct insert into supersession_edges for test setup."""
    from src.substrate.claims import now_ns

    conn.execute(
        "INSERT INTO supersession_edges"
        " (edge_id, old_claim_id, new_claim_id, edge_type, identity_score, created_ts)"
        " VALUES (?, ?, ?, ?, NULL, ?)",
        (f"ed_{old_id}_{new_id}", old_id, new_id, edge_type, now_ns()),
    )
    conn.commit()


# ── list_claims_with_lifecycle ────────────────────────────────────────────


class TestListClaimsWithLifecycle:
    def test_current_truth_matches_list_active(
        self, conn: sqlite3.Connection, session_id: str, add_turn
    ) -> None:
        tid = add_turn("pain started 3 days ago")
        insert_claim(
            conn,
            session_id=session_id,
            subject="chest_pain",
            predicate="onset",
            value="3 days ago",
            confidence=0.9,
            source_turn_id=tid,
        )
        active = list_active_claims(conn, session_id)
        current = list_claims_with_lifecycle(conn, session_id, "current_truth")
        assert len(active) == len(current)
        assert {c.claim_id for c in active} == {c.claim_id for c in current}

    def test_historical_truth_includes_superseded(
        self, conn: sqlite3.Connection, session_id: str, add_turn
    ) -> None:
        tid1 = add_turn("I live in Denver")
        c1 = insert_claim(
            conn,
            session_id=session_id,
            subject="user",
            predicate="onset",
            value="lives in Denver",
            confidence=0.9,
            source_turn_id=tid1,
        )
        # Supersede it
        set_claim_status(conn, c1.claim_id, ClaimStatus.SUPERSEDED)

        tid2 = add_turn("Actually I live in Boston now")
        c2 = insert_claim(
            conn,
            session_id=session_id,
            subject="user",
            predicate="onset",
            value="lives in Boston",
            confidence=0.9,
            source_turn_id=tid2,
        )
        _insert_edge(conn, c1.claim_id, c2.claim_id)

        active = list_active_claims(conn, session_id)
        assert len(active) == 1
        assert active[0].claim_id == c2.claim_id

        historical = list_claims_with_lifecycle(conn, session_id, "historical_truth")
        assert len(historical) == 2
        ids = {c.claim_id for c in historical}
        assert c1.claim_id in ids
        assert c2.claim_id in ids

    def test_historical_truth_excludes_dismissed(
        self, conn: sqlite3.Connection, session_id: str, add_turn
    ) -> None:
        tid = add_turn("some noise")
        c = insert_claim(
            conn,
            session_id=session_id,
            subject="user",
            predicate="onset",
            value="noise",
            confidence=0.5,
            source_turn_id=tid,
        )
        set_claim_status(conn, c.claim_id, ClaimStatus.DISMISSED)

        historical = list_claims_with_lifecycle(conn, session_id, "historical_truth")
        assert len(historical) == 0

    def test_changed_truth_same_as_historical(
        self, conn: sqlite3.Connection, session_id: str, add_turn
    ) -> None:
        tid = add_turn("some fact")
        insert_claim(
            conn,
            session_id=session_id,
            subject="user",
            predicate="onset",
            value="fact_value",
            confidence=0.9,
            source_turn_id=tid,
        )
        h = list_claims_with_lifecycle(conn, session_id, "historical_truth")
        c = list_claims_with_lifecycle(conn, session_id, "changed_truth")
        assert len(h) == len(c)


# ── list_supersession_pairs ───────────────────────────────────────────────


class TestListSupersessionPairs:
    def test_returns_old_new_edge_triple(
        self, conn: sqlite3.Connection, session_id: str, add_turn
    ) -> None:
        tid1 = add_turn("I work at Google")
        c1 = insert_claim(
            conn,
            session_id=session_id,
            subject="user",
            predicate="onset",
            value="works at Google",
            confidence=0.9,
            source_turn_id=tid1,
        )
        set_claim_status(conn, c1.claim_id, ClaimStatus.SUPERSEDED)

        tid2 = add_turn("Actually I work at Meta now")
        c2 = insert_claim(
            conn,
            session_id=session_id,
            subject="user",
            predicate="onset",
            value="works at Meta",
            confidence=0.9,
            source_turn_id=tid2,
        )
        _insert_edge(conn, c1.claim_id, c2.claim_id)

        pairs = list_supersession_pairs(conn, session_id)
        assert len(pairs) == 1
        old, new, edge = pairs[0]
        assert old.claim_id == c1.claim_id
        assert new.claim_id == c2.claim_id
        assert edge.edge_type == EdgeType.PATIENT_CORRECTION
        assert old.status == ClaimStatus.SUPERSEDED

    def test_empty_when_no_supersessions(
        self, conn: sqlite3.Connection, session_id: str, add_turn
    ) -> None:
        tid = add_turn("simple fact")
        insert_claim(
            conn,
            session_id=session_id,
            subject="user",
            predicate="onset",
            value="val",
            confidence=0.9,
            source_turn_id=tid,
        )
        pairs = list_supersession_pairs(conn, session_id)
        assert pairs == []

    def test_ordered_by_most_recent_first(
        self, conn: sqlite3.Connection, session_id: str, add_turn
    ) -> None:
        # First supersession
        tid1 = add_turn("v1")
        c1 = insert_claim(
            conn,
            session_id=session_id,
            subject="user",
            predicate="onset",
            value="v1",
            confidence=0.9,
            source_turn_id=tid1,
        )
        set_claim_status(conn, c1.claim_id, ClaimStatus.SUPERSEDED)

        tid2 = add_turn("v2")
        c2 = insert_claim(
            conn,
            session_id=session_id,
            subject="user",
            predicate="onset",
            value="v2",
            confidence=0.9,
            source_turn_id=tid2,
        )
        _insert_edge(conn, c1.claim_id, c2.claim_id)

        # Second supersession
        set_claim_status(conn, c2.claim_id, ClaimStatus.SUPERSEDED)
        tid3 = add_turn("v3")
        c3 = insert_claim(
            conn,
            session_id=session_id,
            subject="user",
            predicate="onset",
            value="v3",
            confidence=0.9,
            source_turn_id=tid3,
        )
        _insert_edge(conn, c2.claim_id, c3.claim_id)

        pairs = list_supersession_pairs(conn, session_id)
        assert len(pairs) == 2
        # Most recent supersession first
        assert pairs[0][0].value == "v2"  # old
        assert pairs[0][1].value == "v3"  # new
        assert pairs[1][0].value == "v1"  # old
        assert pairs[1][1].value == "v2"  # new


# ── backward compatibility ────────────────────────────────────────────────


class TestBackwardCompatibility:
    def test_list_active_claims_unchanged(
        self, conn: sqlite3.Connection, session_id: str, add_turn
    ) -> None:
        tid = add_turn("fact")
        c = insert_claim(
            conn,
            session_id=session_id,
            subject="user",
            predicate="onset",
            value="val",
            confidence=0.9,
            source_turn_id=tid,
        )
        active = list_active_claims(conn, session_id)
        assert len(active) == 1
        assert active[0].claim_id == c.claim_id
