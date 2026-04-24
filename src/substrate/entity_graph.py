"""Entity-temporal graph index over claims (Layer 7).

An in-memory adjacency index built from ``claim_metadata.entity_key``.
The graph is a VIEW over claims — it does not replace them. Claims
remain the atomic provenance unit; the graph enables graph-neighbor
retrieval for multi-session reasoning.

Entities are connected when they share a session or when they appear in
claims from the same source turn (co-occurrence).
"""
from __future__ import annotations

import sqlite3
from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EntityNode:
    """One entity in the graph with its associated claims."""

    entity_key: str
    claim_ids: frozenset[str]
    predicates: frozenset[str]
    first_seen_ts: int
    last_seen_ts: int


class EntityGraph:
    """Read-only graph index over claims in one session.

    Built from ``claim_metadata`` + ``claims`` tables. Two entities are
    neighbors when they share a source turn (co-occurrence in the same
    utterance) or share the same session (weaker link).
    """

    def __init__(self, conn: sqlite3.Connection, session_id: str) -> None:
        self._nodes: dict[str, EntityNode] = {}
        self._adjacency: dict[str, set[str]] = defaultdict(set)
        self._claim_to_entity: dict[str, str] = {}
        self._build(conn, session_id)

    def _build(self, conn: sqlite3.Connection, session_id: str) -> None:
        rows = conn.execute(
            "SELECT m.claim_id, m.entity_key, m.predicate_family,"
            "       c.source_turn_id, c.created_ts"
            " FROM claim_metadata m"
            " JOIN claims c ON m.claim_id = c.claim_id"
            " WHERE c.session_id = ?"
            "   AND c.status IN ('active','confirmed','audited','superseded')"
            " ORDER BY c.created_ts ASC",
            (session_id,),
        ).fetchall()

        # Group claims by entity
        entity_claims: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            entity_claims[r["entity_key"]].append({
                "claim_id": r["claim_id"],
                "predicate": r["predicate_family"],
                "turn_id": r["source_turn_id"],
                "ts": r["created_ts"],
            })
            self._claim_to_entity[r["claim_id"]] = r["entity_key"]

        # Build nodes
        for ek, claims in entity_claims.items():
            self._nodes[ek] = EntityNode(
                entity_key=ek,
                claim_ids=frozenset(c["claim_id"] for c in claims),
                predicates=frozenset(c["predicate"] for c in claims),
                first_seen_ts=min(c["ts"] for c in claims),
                last_seen_ts=max(c["ts"] for c in claims),
            )

        # Build adjacency: entities sharing a source turn are neighbors
        turn_entities: dict[str, set[str]] = defaultdict(set)
        for r in rows:
            turn_entities[r["source_turn_id"]].add(r["entity_key"])

        for entities in turn_entities.values():
            entity_list = list(entities)
            for i, e1 in enumerate(entity_list):
                for e2 in entity_list[i + 1:]:
                    self._adjacency[e1].add(e2)
                    self._adjacency[e2].add(e1)

    @property
    def entities(self) -> dict[str, EntityNode]:
        return dict(self._nodes)

    def has_entity(self, entity_key: str) -> bool:
        return entity_key in self._nodes

    def get_node(self, entity_key: str) -> EntityNode | None:
        return self._nodes.get(entity_key)

    def neighbors(self, entity_key: str, max_hops: int = 1) -> set[str]:
        """Return entity_keys reachable within *max_hops* from *entity_key*."""
        if entity_key not in self._nodes:
            return set()

        visited: set[str] = set()
        frontier: set[str] = {entity_key}

        for _ in range(max_hops):
            next_frontier: set[str] = set()
            for ek in frontier:
                for neighbor in self._adjacency.get(ek, set()):
                    if neighbor not in visited and neighbor != entity_key:
                        next_frontier.add(neighbor)
            visited.update(next_frontier)
            frontier = next_frontier
            if not frontier:
                break

        return visited

    def neighbor_claim_ids(self, entity_key: str, max_hops: int = 1) -> set[str]:
        """Return claim_ids from all neighbor entities."""
        neighbor_entities = self.neighbors(entity_key, max_hops)
        claim_ids: set[str] = set()
        for ek in neighbor_entities:
            node = self._nodes.get(ek)
            if node:
                claim_ids.update(node.claim_ids)
        return claim_ids

    def entity_for_claim(self, claim_id: str) -> str | None:
        """Return the entity_key for a given claim_id."""
        return self._claim_to_entity.get(claim_id)

    def entity_chain(self, entity_a: str, entity_b: str, max_depth: int = 5) -> list[str] | None:
        """BFS shortest path between two entities. Returns entity_key list or None."""
        if entity_a not in self._nodes or entity_b not in self._nodes:
            return None
        if entity_a == entity_b:
            return [entity_a]

        visited: set[str] = {entity_a}
        queue: list[tuple[str, list[str]]] = [(entity_a, [entity_a])]

        while queue:
            current, path = queue.pop(0)
            if len(path) > max_depth:
                continue
            for neighbor in self._adjacency.get(current, set()):
                if neighbor == entity_b:
                    return path + [neighbor]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return None


__all__ = [
    "EntityGraph",
    "EntityNode",
]
