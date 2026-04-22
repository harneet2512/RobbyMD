"""End-to-end orchestrator tests — `on_new_turn`."""
from __future__ import annotations

import sqlite3

from src.substrate.claims import list_active_claims
from src.substrate.event_bus import (
    CLAIM_CREATED,
    CLAIM_SUPERSEDED,
    PROJECTION_UPDATED,
    TURN_ADDED,
    EventBus,
)
from src.substrate.on_new_turn import ExtractedClaim, ExtractorFn, on_new_turn
from src.substrate.schema import EdgeType, Speaker, Turn
from src.substrate.supersession_semantic import NullEmbedder, SemanticSupersession


def _make_extractor(
    claims_per_turn: dict[str, list[ExtractedClaim]],
) -> ExtractorFn:
    """Extractor stub keyed by turn text (exact match)."""

    def extractor(turn: Turn) -> list[ExtractedClaim]:
        # Match exact turn text — simple and deterministic for tests.
        return claims_per_turn.get(turn.text, [])

    return extractor


def test_on_new_turn_rejects_short_text(
    conn: sqlite3.Connection, session_id: str
) -> None:
    result = on_new_turn(
        conn,
        session_id=session_id,
        speaker=Speaker.PATIENT,
        text="uh",
        extractor=_make_extractor({}),
    )
    assert not result.admitted
    assert result.turn is None
    assert result.admission_reason.startswith("only_")


def test_on_new_turn_happy_path_creates_claims_and_events(
    conn: sqlite3.Connection, session_id: str
) -> None:
    bus = EventBus()
    events: list[tuple[str, dict[str, object]]] = []
    for name in (TURN_ADDED, CLAIM_CREATED, PROJECTION_UPDATED):
        bus.subscribe(name, lambda p, n=name: events.append((n, p)))

    text = "chest pain started three days ago"
    extractor = _make_extractor(
        {
            text: [
                ExtractedClaim(
                    subject="patient",
                    predicate="onset",
                    value="3 days",
                    confidence=0.9,
                    char_start=0,
                    char_end=10,
                ),
                ExtractedClaim(
                    subject="patient",
                    predicate="character",
                    value="stabbing",
                    confidence=0.85,
                ),
            ],
        }
    )
    result = on_new_turn(
        conn,
        session_id=session_id,
        speaker=Speaker.PATIENT,
        text=text,
        extractor=extractor,
        bus=bus,
    )
    assert result.admitted
    assert len(result.created_claims) == 2
    assert result.dropped_claims == ()
    # Exactly one TURN_ADDED, one PROJECTION_UPDATED, two CLAIM_CREATED.
    event_names = [n for (n, _) in events]
    assert event_names.count(TURN_ADDED) == 1
    assert event_names.count(CLAIM_CREATED) == 2
    assert event_names.count(PROJECTION_UPDATED) == 1
    # Active-claim list reflects insertions.
    assert {c.predicate for c in list_active_claims(conn, session_id)} == {
        "onset",
        "character",
    }


def test_on_new_turn_invalid_claim_dropped_not_raised(
    conn: sqlite3.Connection, session_id: str
) -> None:
    text = "patient reports something unusual now"
    extractor = _make_extractor(
        {
            text: [
                ExtractedClaim(
                    subject="patient",
                    predicate="onset",
                    value="3 days",
                    confidence=0.9,
                ),
                ExtractedClaim(
                    subject="patient",
                    predicate="nonsense_predicate",  # invalid
                    value="x",
                    confidence=0.9,
                ),
                ExtractedClaim(
                    subject="patient",
                    predicate="severity",
                    value="7",
                    confidence=2.0,  # invalid
                ),
            ],
        }
    )
    result = on_new_turn(
        conn,
        session_id=session_id,
        speaker=Speaker.PATIENT,
        text=text,
        extractor=extractor,
    )
    assert len(result.created_claims) == 1
    # Both invalid claims surfaced in dropped_claims with reason strings.
    assert len(result.dropped_claims) == 2
    reasons = [r for (_, r) in result.dropped_claims]
    assert any("closed family" in r for r in reasons)
    assert any("confidence" in r for r in reasons)


def test_on_new_turn_supersession_fires_across_turns(
    conn: sqlite3.Connection, session_id: str
) -> None:
    bus = EventBus()
    sups: list[dict[str, object]] = []
    bus.subscribe(CLAIM_SUPERSEDED, lambda p: sups.append(p))

    t1_text = "onset was three days ago today"
    t2_text = "actually it was four days ago"
    ex = _make_extractor(
        {
            t1_text: [
                ExtractedClaim(
                    subject="patient",
                    predicate="onset",
                    value="3 days",
                    confidence=0.9,
                ),
            ],
            t2_text: [
                ExtractedClaim(
                    subject="patient",
                    predicate="onset",
                    value="4 days",
                    confidence=0.9,
                ),
            ],
        }
    )
    on_new_turn(
        conn,
        session_id=session_id,
        speaker=Speaker.PATIENT,
        text=t1_text,
        extractor=ex,
        bus=bus,
    )
    on_new_turn(
        conn,
        session_id=session_id,
        speaker=Speaker.PATIENT,
        text=t2_text,
        extractor=ex,
        bus=bus,
    )
    assert len(sups) == 1
    assert sups[0]["edge_type"] == EdgeType.PATIENT_CORRECTION.value

    # Active claims now = 1 (the newer one).
    active = list_active_claims(conn, session_id)
    assert len(active) == 1
    assert active[0].value == "4 days"


def test_on_new_turn_pass2_skipped_when_pass1_caught(
    conn: sqlite3.Connection, session_id: str
) -> None:
    """If Pass 1 already superseded, Pass 2 must not create a second edge."""
    bus = EventBus()
    sups: list[dict[str, object]] = []
    bus.subscribe(CLAIM_SUPERSEDED, lambda p: sups.append(p))

    sem = SemanticSupersession(NullEmbedder())
    t1_text = "onset was three days ago today"
    t2_text = "actually it was four days ago"
    ex = _make_extractor(
        {
            t1_text: [
                ExtractedClaim(
                    subject="patient",
                    predicate="onset",
                    value="3 days",
                    confidence=0.9,
                ),
            ],
            t2_text: [
                ExtractedClaim(
                    subject="patient",
                    predicate="onset",
                    value="4 days",
                    confidence=0.9,
                ),
            ],
        }
    )
    on_new_turn(
        conn,
        session_id=session_id,
        speaker=Speaker.PATIENT,
        text=t1_text,
        extractor=ex,
        bus=bus,
        semantic=sem,
    )
    on_new_turn(
        conn,
        session_id=session_id,
        speaker=Speaker.PATIENT,
        text=t2_text,
        extractor=ex,
        bus=bus,
        semantic=sem,
    )
    # Exactly one supersession event — Pass 2 must not double-fire.
    assert len(sups) == 1
