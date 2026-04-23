"""LongMemEval QA adapter: substrate state → reader-ready evidence.

The adapter routes retrieval by LongMemEval question_type, walks supersession
chains so the reader sees either the current-valid claim (most cases) or the
entire history (knowledge-update), attaches temporal-validity windows to every
claim, and pre-computes a human-readable conflict resolution when a chain has
been rewritten.

Contract with the reader is a flat `LMEAdapterOutput` that the sibling
`lme_reader_prompt` module renders into a single-call prompt. This replaces
the CoN two-call reader (`eval/longmemeval/reader_con.py`) and its empty-notes
IDK short-circuit.

No LLM calls in this module. Pure substrate reads + supersession chain walks.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from src.substrate.event_tuples import EventTuple
from src.substrate.retrieval import (
    EmbeddingClient,
    retrieve_event_tuples,
    retrieve_hybrid,
)
from src.substrate.schema import Claim


# LME canonical question_type values (from the dataset).
QUESTION_TYPES: frozenset[str] = frozenset({
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "multi-session",
    "temporal-reasoning",
    "knowledge-update",
})


MAX_CLAIMS_TO_READER = 10  # cap after routing; prevents reader context bloat.


@dataclass(frozen=True, slots=True)
class ReaderClaim:
    """One claim rendered for the reader, with enough provenance for citation."""

    claim_text: str
    claim_id: str
    source_session_id: str
    source_turn_id: str
    confidence: float
    similarity_score: float
    valid_from_ts: int | None = None
    valid_until_ts: int | None = None
    superseded_by: str | None = None
    supersedes: str | None = None
    source_turn_text: str | None = None  # populated for single-session-assistant


@dataclass(frozen=True, slots=True)
class LMEAdapterInput:
    question: str
    question_type: str
    session_id: str
    conn: sqlite3.Connection
    embedding_client: EmbeddingClient


@dataclass(frozen=True, slots=True)
class LMEAdapterOutput:
    ranked_claims: list[ReaderClaim]
    retrieval_confidence: float
    has_conflicts: bool
    conflict_resolution: str | None


# --- helpers -----------------------------------------------------------------


def _load_claim_row(conn: sqlite3.Connection, claim_id: str) -> Claim | None:
    """Fetch a single claim by id. Returns None if the row is missing."""
    row = conn.execute(
        "SELECT claim_id, session_id, subject, predicate, value, value_normalised, "
        "confidence, source_turn_id, status, created_ts, char_start, char_end, "
        "valid_from_ts, valid_until_ts FROM claims WHERE claim_id = ?",
        (claim_id,),
    ).fetchone()
    if row is None:
        return None
    from src.substrate.schema import ClaimStatus
    return Claim(
        claim_id=row["claim_id"],
        session_id=row["session_id"],
        subject=row["subject"],
        predicate=row["predicate"],
        value=row["value"],
        value_normalised=row["value_normalised"],
        confidence=row["confidence"],
        source_turn_id=row["source_turn_id"],
        status=ClaimStatus(row["status"]),
        created_ts=row["created_ts"],
        char_start=row["char_start"],
        char_end=row["char_end"],
        valid_from_ts=row["valid_from_ts"],
        valid_until_ts=row["valid_until_ts"],
    )


def _walk_chain(conn: sqlite3.Connection, claim_id: str) -> list[str]:
    """Walk the supersession chain starting at the claim's earliest ancestor.

    Returns ordered list [ancestor, ..., current]. If the claim has no edges
    the chain is [claim_id] alone.
    """
    # Walk backward to the earliest ancestor.
    cursor = claim_id
    seen: set[str] = {cursor}
    while True:
        row = conn.execute(
            "SELECT old_claim_id FROM supersession_edges WHERE new_claim_id = ? LIMIT 1",
            (cursor,),
        ).fetchone()
        if row is None:
            break
        prev = row["old_claim_id"]
        if prev in seen:  # cycle guard (shouldn't happen)
            break
        seen.add(prev)
        cursor = prev
    # Walk forward from the earliest ancestor to the terminal current.
    chain = [cursor]
    forward_seen: set[str] = {cursor}
    while True:
        row = conn.execute(
            "SELECT new_claim_id FROM supersession_edges WHERE old_claim_id = ? LIMIT 1",
            (cursor,),
        ).fetchone()
        if row is None:
            break
        nxt = row["new_claim_id"]
        if nxt in forward_seen:
            break
        chain.append(nxt)
        forward_seen.add(nxt)
        cursor = nxt
    return chain


def _load_turn_text(conn: sqlite3.Connection, turn_id: str) -> str | None:
    row = conn.execute("SELECT text FROM turns WHERE turn_id = ?", (turn_id,)).fetchone()
    return row["text"] if row is not None else None


def _render_claim_text(c: Claim) -> str:
    return f"{c.subject} / {c.predicate} = {c.value}"


def _make_reader_claim(
    c: Claim,
    similarity_score: float,
    *,
    superseded_by: str | None = None,
    supersedes: str | None = None,
    source_turn_text: str | None = None,
) -> ReaderClaim:
    return ReaderClaim(
        claim_text=_render_claim_text(c),
        claim_id=c.claim_id,
        source_session_id=c.session_id,
        source_turn_id=c.source_turn_id,
        confidence=c.confidence,
        similarity_score=similarity_score,
        valid_from_ts=c.valid_from_ts,
        valid_until_ts=c.valid_until_ts,
        superseded_by=superseded_by,
        supersedes=supersedes,
        source_turn_text=source_turn_text,
    )


def _retrieve_for_type(inp: LMEAdapterInput) -> list[tuple[Claim, float]]:
    """Fan out to the right retrieval function and normalize to (Claim, score)."""
    q_type = inp.question_type
    if q_type == "temporal-reasoning":
        event_hits: list[tuple[EventTuple, float]] = retrieve_event_tuples(
            inp.conn,
            session_id=inp.session_id,
            query=inp.question,
            top_k=15,
            embedding_client=inp.embedding_client,
        )
        out: list[tuple[Claim, float]] = []
        for event, score in event_hits:
            claim = _load_claim_row(inp.conn, event.claim_id)
            if claim is not None:
                out.append((claim, score))
        return out
    if q_type == "multi-session":
        return retrieve_hybrid(
            inp.conn,
            session_id=inp.session_id,
            query=inp.question,
            top_k=25,
            embedding_client=inp.embedding_client,
        )
    if q_type == "knowledge-update":
        return retrieve_hybrid(
            inp.conn,
            session_id=inp.session_id,
            query=inp.question,
            top_k=15,
            embedding_client=inp.embedding_client,
        )
    # single-session-user / single-session-preference / single-session-assistant
    # plus graceful default for any unrecognised type.
    return retrieve_hybrid(
        inp.conn,
        session_id=inp.session_id,
        query=inp.question,
        top_k=10,
        embedding_client=inp.embedding_client,
    )


def _format_conflict(chain_claims: list[Claim]) -> str:
    """Render a 2+-link chain into a human-readable conflict resolution string."""
    first, last = chain_claims[0], chain_claims[-1]
    return (
        f"User originally said '{first.value}' "
        f"(session {first.session_id}, turn {first.source_turn_id}). "
        f"Later updated to '{last.value}' "
        f"(session {last.session_id}, turn {last.source_turn_id}). "
        f"Current answer: '{last.value}'."
    )


# --- entry point -------------------------------------------------------------


def adapt(inp: LMEAdapterInput) -> LMEAdapterOutput:
    """Map substrate state to reader-ready evidence for one LME question."""
    retrieved = _retrieve_for_type(inp)

    is_knowledge_update = inp.question_type == "knowledge-update"
    is_assistant = inp.question_type == "single-session-assistant"

    # Walk supersession chains and build ReaderClaims.
    ranked: list[ReaderClaim] = []
    any_conflict_chain: list[Claim] | None = None
    seen_claim_ids: set[str] = set()

    for claim, score in retrieved:
        chain_ids = _walk_chain(inp.conn, claim.claim_id)
        chain_claims = [_load_claim_row(inp.conn, cid) for cid in chain_ids]
        chain_claims = [c for c in chain_claims if c is not None]
        if not chain_claims:
            continue

        if len(chain_claims) > 1 and any_conflict_chain is None:
            any_conflict_chain = chain_claims

        if is_knowledge_update:
            # Surface the entire chain so the reader sees the update history.
            for idx, c in enumerate(chain_claims):
                if c.claim_id in seen_claim_ids:
                    continue
                seen_claim_ids.add(c.claim_id)
                superseded_by = (
                    chain_claims[idx + 1].claim_id
                    if idx + 1 < len(chain_claims)
                    else None
                )
                supersedes = chain_claims[idx - 1].claim_id if idx > 0 else None
                # Historical links inherit the retrieval score of the hit so they
                # rank alongside the current claim.
                ranked.append(
                    _make_reader_claim(
                        c, score, superseded_by=superseded_by, supersedes=supersedes
                    )
                )
        else:
            # Surface only the terminal-current claim, drop superseded silently.
            current = chain_claims[-1]
            if current.claim_id in seen_claim_ids:
                continue
            seen_claim_ids.add(current.claim_id)
            source_turn_text = (
                _load_turn_text(inp.conn, current.source_turn_id)
                if is_assistant
                else None
            )
            ranked.append(
                _make_reader_claim(current, score, source_turn_text=source_turn_text)
            )

    ranked.sort(key=lambda rc: rc.similarity_score, reverse=True)
    ranked = ranked[:MAX_CLAIMS_TO_READER]

    retrieval_confidence = max((rc.similarity_score for rc in ranked), default=0.0)
    has_conflicts = any_conflict_chain is not None
    conflict_resolution = (
        _format_conflict(any_conflict_chain) if any_conflict_chain is not None else None
    )

    return LMEAdapterOutput(
        ranked_claims=ranked,
        retrieval_confidence=retrieval_confidence,
        has_conflicts=has_conflicts,
        conflict_resolution=conflict_resolution,
    )
