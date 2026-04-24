"""Tests for entity-temporal graph index (Layer 7)."""
from __future__ import annotations

import sqlite3

import pytest

from src.substrate.claims import insert_claim
from src.substrate.entity_graph import EntityGraph, EntityNode
from src.substrate.schema import ClaimStatus, Speaker


class TestEntityGraph:
    def test_single_entity_creates_node(
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
        graph = EntityGraph(conn, session_id)
        assert graph.has_entity("chest_pain")
        node = graph.get_node("chest_pain")
        assert node is not None
        assert len(node.claim_ids) == 1
        assert "onset" in node.predicates

    def test_two_entities_same_turn_are_neighbors(
        self, conn: sqlite3.Connection, session_id: str, add_turn
    ) -> None:
        tid = add_turn("chest pain radiating to left arm")
        insert_claim(
            conn,
            session_id=session_id,
            subject="chest_pain",
            predicate="onset",
            value="acute",
            confidence=0.9,
            source_turn_id=tid,
        )
        insert_claim(
            conn,
            session_id=session_id,
            subject="left_arm",
            predicate="radiation",
            value="radiating",
            confidence=0.9,
            source_turn_id=tid,
        )
        graph = EntityGraph(conn, session_id)
        neighbors = graph.neighbors("chest_pain")
        assert "left_arm" in neighbors

    def test_entities_different_turns_not_neighbors(
        self, conn: sqlite3.Connection, session_id: str, add_turn
    ) -> None:
        tid1 = add_turn("pain started 3 days ago")
        insert_claim(
            conn,
            session_id=session_id,
            subject="chest_pain",
            predicate="onset",
            value="3 days",
            confidence=0.9,
            source_turn_id=tid1,
        )
        tid2 = add_turn("I take aspirin")
        insert_claim(
            conn,
            session_id=session_id,
            subject="aspirin",
            predicate="medication",
            value="aspirin daily",
            confidence=0.9,
            source_turn_id=tid2,
        )
        graph = EntityGraph(conn, session_id)
        neighbors = graph.neighbors("chest_pain")
        assert "aspirin" not in neighbors

    def test_neighbor_claim_ids(
        self, conn: sqlite3.Connection, session_id: str, add_turn
    ) -> None:
        tid = add_turn("pain and nausea")
        c1 = insert_claim(
            conn,
            session_id=session_id,
            subject="pain",
            predicate="onset",
            value="acute",
            confidence=0.9,
            source_turn_id=tid,
        )
        c2 = insert_claim(
            conn,
            session_id=session_id,
            subject="nausea",
            predicate="associated_symptom",
            value="present",
            confidence=0.9,
            source_turn_id=tid,
        )
        graph = EntityGraph(conn, session_id)
        neighbor_ids = graph.neighbor_claim_ids("pain")
        assert c2.claim_id in neighbor_ids
        assert c1.claim_id not in neighbor_ids

    def test_entity_chain_direct_neighbors(
        self, conn: sqlite3.Connection, session_id: str, add_turn
    ) -> None:
        tid = add_turn("pain and nausea")
        insert_claim(
            conn, session_id=session_id, subject="pain",
            predicate="onset", value="v", confidence=0.9, source_turn_id=tid,
        )
        insert_claim(
            conn, session_id=session_id, subject="nausea",
            predicate="associated_symptom", value="v", confidence=0.9, source_turn_id=tid,
        )
        graph = EntityGraph(conn, session_id)
        chain = graph.entity_chain("pain", "nausea")
        assert chain is not None
        assert chain == ["pain", "nausea"]

    def test_entity_chain_no_path(
        self, conn: sqlite3.Connection, session_id: str, add_turn
    ) -> None:
        tid1 = add_turn("pain")
        insert_claim(
            conn, session_id=session_id, subject="pain",
            predicate="onset", value="v", confidence=0.9, source_turn_id=tid1,
        )
        tid2 = add_turn("aspirin")
        insert_claim(
            conn, session_id=session_id, subject="aspirin",
            predicate="medication", value="v", confidence=0.9, source_turn_id=tid2,
        )
        graph = EntityGraph(conn, session_id)
        chain = graph.entity_chain("pain", "aspirin")
        assert chain is None

    def test_empty_session_produces_empty_graph(
        self, conn: sqlite3.Connection, session_id: str
    ) -> None:
        graph = EntityGraph(conn, session_id)
        assert graph.entities == {}
        assert graph.neighbors("anything") == set()

    def test_deterministic_given_same_claims(
        self, conn: sqlite3.Connection, session_id: str, add_turn
    ) -> None:
        tid = add_turn("pain and nausea together")
        insert_claim(
            conn, session_id=session_id, subject="pain",
            predicate="onset", value="acute", confidence=0.9, source_turn_id=tid,
        )
        insert_claim(
            conn, session_id=session_id, subject="nausea",
            predicate="associated_symptom", value="present", confidence=0.9, source_turn_id=tid,
        )
        g1 = EntityGraph(conn, session_id)
        g2 = EntityGraph(conn, session_id)
        assert g1.entities.keys() == g2.entities.keys()
        assert g1.neighbors("pain") == g2.neighbors("pain")

    def test_entity_for_claim(
        self, conn: sqlite3.Connection, session_id: str, add_turn
    ) -> None:
        tid = add_turn("pain started")
        c = insert_claim(
            conn, session_id=session_id, subject="chest_pain",
            predicate="onset", value="3 days", confidence=0.9, source_turn_id=tid,
        )
        graph = EntityGraph(conn, session_id)
        assert graph.entity_for_claim(c.claim_id) == "chest_pain"
        assert graph.entity_for_claim("nonexistent") is None

    def test_multi_hop_neighbors(
        self, conn: sqlite3.Connection, session_id: str, add_turn
    ) -> None:
        # A-B connected via turn1, B-C connected via turn2
        tid1 = add_turn("A and B")
        insert_claim(
            conn, session_id=session_id, subject="entity_a",
            predicate="onset", value="v", confidence=0.9, source_turn_id=tid1,
        )
        insert_claim(
            conn, session_id=session_id, subject="entity_b",
            predicate="onset", value="v", confidence=0.9, source_turn_id=tid1,
        )
        tid2 = add_turn("B and C")
        insert_claim(
            conn, session_id=session_id, subject="entity_b",
            predicate="severity", value="v", confidence=0.9, source_turn_id=tid2,
        )
        insert_claim(
            conn, session_id=session_id, subject="entity_c",
            predicate="severity", value="v", confidence=0.9, source_turn_id=tid2,
        )
        graph = EntityGraph(conn, session_id)
        # 1-hop from A: only B
        assert graph.neighbors("entity_a", max_hops=1) == {"entity_b"}
        # 2-hop from A: B and C
        assert graph.neighbors("entity_a", max_hops=2) == {"entity_b", "entity_c"}
