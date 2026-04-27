"""Scripted Mr. Torres encounter replay for demo mode.

Feeds pre-scripted turns into the substrate with a mock claim extractor.
Each turn produces deterministic claims without any API dependency.
Includes supersession (patient correction) and SOAP note generation.

Run via server.py with DEMO_MODE=1 (default).
"""
from __future__ import annotations

import sqlite3
import time
import uuid
from dataclasses import dataclass
from typing import Any

from src.substrate.event_bus import EventBus
from src.substrate.schema import Speaker, ClaimStatus, NoteSection

ENCOUNTER_START_NS = int(time.time() * 1_000_000_000)


@dataclass
class ScriptedTurn:
    delay_s: float
    speaker: Speaker
    text: str
    claims: list[dict[str, Any]]


TORRES_SCRIPT: list[ScriptedTurn] = [
    ScriptedTurn(
        delay_s=2.0,
        speaker=Speaker.PATIENT,
        text="I've been having this chest pain, doc. It's kind of a burning feeling behind my breastbone, worse after eating.",
        claims=[
            {"predicate": "character", "value": "burning", "confidence": 0.92},
            {"predicate": "location", "value": "substernal", "confidence": 0.88},
            {"predicate": "aggravating_factor", "value": "post-prandial", "confidence": 0.85},
        ],
    ),
    ScriptedTurn(
        delay_s=4.0,
        speaker=Speaker.PHYSICIAN,
        text="Does the pain get worse with exertion, like walking upstairs?",
        claims=[],
    ),
    ScriptedTurn(
        delay_s=3.0,
        speaker=Speaker.PATIENT,
        text="No, not really. It's more about when I eat. Exercise doesn't seem to make it worse.",
        claims=[
            {"predicate": "aggravating_factor", "value": "negated:exertion", "confidence": 0.90},
        ],
    ),
    ScriptedTurn(
        delay_s=3.5,
        speaker=Speaker.PHYSICIAN,
        text="When did this start?",
        claims=[],
    ),
    ScriptedTurn(
        delay_s=2.5,
        speaker=Speaker.PATIENT,
        text="It started Monday night... well, actually, no, it was Sunday evening. I remember because I was watching the game.",
        claims=[
            {"predicate": "onset", "value": "Mon night", "confidence": 0.60, "_superseded_by": "onset_correction"},
            {"predicate": "onset", "value": "Sun evening", "confidence": 0.82, "_supersession_tag": "onset_correction"},
        ],
    ),
    ScriptedTurn(
        delay_s=4.0,
        speaker=Speaker.PHYSICIAN,
        text="Any history of acid reflux or GERD?",
        claims=[],
    ),
    ScriptedTurn(
        delay_s=3.0,
        speaker=Speaker.PATIENT,
        text="Yeah, I was on omeprazole for about two years, but I stopped taking it maybe two weeks ago. Ran out and never refilled it.",
        claims=[
            {"predicate": "medical_history", "value": "GERD", "confidence": 0.95},
            {"predicate": "medication", "value": "omeprazole · discontinued 2w", "confidence": 0.90},
        ],
    ),
    ScriptedTurn(
        delay_s=5.0,
        speaker=Speaker.PHYSICIAN,
        text="Any leg swelling, recent surgery, or long periods of immobility?",
        claims=[],
    ),
    ScriptedTurn(
        delay_s=2.0,
        speaker=Speaker.PATIENT,
        text="No, nothing like that.",
        claims=[
            {"predicate": "risk_factor", "value": "negated:immobilization", "confidence": 0.88},
            {"predicate": "associated_symptom", "value": "negated:leg_swelling", "confidence": 0.85},
        ],
    ),
]

SOAP_SENTENCES = [
    {"section": "S", "ordinal": 0,
     "text": "52-year-old male presents with substernal burning chest pain, worse after eating, onset Sunday evening.",
     "predicates": ["character", "location", "aggravating_factor", "onset"]},
    {"section": "S", "ordinal": 1,
     "text": "Pain is not exertional. Patient has a history of GERD, previously treated with omeprazole discontinued two weeks ago.",
     "predicates": ["aggravating_factor", "medical_history", "medication"]},
    {"section": "S", "ordinal": 2,
     "text": "Denies leg swelling, recent surgery, or prolonged immobility.",
     "predicates": ["associated_symptom", "risk_factor"]},
    {"section": "A", "ordinal": 0,
     "text": "Substernal burning chest pain in the setting of GERD with recently discontinued PPI, most consistent with acid reflux exacerbation.",
     "predicates": ["character", "location", "medical_history", "medication"]},
    {"section": "P", "ordinal": 0,
     "text": "Restart omeprazole. Consider EKG if symptoms persist or change in character to rule out cardiac etiology.",
     "predicates": ["medication"]},
]


def _make_turn_id() -> str:
    return f"tu_{uuid.uuid4().hex[:12]}"


def _make_claim_id() -> str:
    return f"cl_{uuid.uuid4().hex[:12]}"


def _make_sentence_id() -> str:
    return f"ns_{uuid.uuid4().hex[:12]}"


def _insert_turn(conn: sqlite3.Connection, session_id: str, turn: ScriptedTurn, ts_ns: int) -> str:
    turn_id = _make_turn_id()
    conn.execute(
        "INSERT INTO turns (turn_id, session_id, speaker, text, ts, asr_confidence) VALUES (?, ?, ?, ?, ?, ?)",
        (turn_id, session_id, turn.speaker.value, turn.text, ts_ns, 0.95),
    )
    conn.commit()
    return turn_id


def _insert_claim(
    conn: sqlite3.Connection,
    session_id: str,
    turn_id: str,
    claim_data: dict[str, Any],
    ts_ns: int,
) -> dict[str, Any]:
    claim_id = _make_claim_id()
    conn.execute(
        """INSERT INTO claims
           (claim_id, session_id, subject, predicate, value, value_normalised,
            confidence, source_turn_id, status, created_ts, char_start, char_end,
            valid_from_ts, valid_until_ts)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            claim_id, session_id, "chest_pain",
            claim_data["predicate"], claim_data["value"], claim_data["value"],
            claim_data["confidence"], turn_id,
            ClaimStatus.ACTIVE.value, ts_ns,
            None, None, ts_ns, None,
        ),
    )
    conn.commit()
    return {
        "claim_id": claim_id,
        "session_id": session_id,
        "subject": "chest_pain",
        "predicate": claim_data["predicate"],
        "value": claim_data["value"],
        "confidence": claim_data["confidence"],
        "source_turn_id": turn_id,
        "status": "active",
        "created_ts_ns": ts_ns,
    }


def _insert_supersession_edge(
    conn: sqlite3.Connection,
    old_claim_id: str,
    new_claim_id: str,
    ts_ns: int,
) -> None:
    edge_id = f"edge_{uuid.uuid4().hex[:12]}"
    conn.execute(
        """INSERT INTO supersession_edges
           (edge_id, old_claim_id, new_claim_id, edge_type, identity_score, created_ts)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (edge_id, old_claim_id, new_claim_id, "patient_correction", 0.85, ts_ns),
    )
    conn.execute(
        "UPDATE claims SET status = 'superseded' WHERE claim_id = ?",
        (old_claim_id,),
    )
    conn.commit()


def start_demo(conn: sqlite3.Connection, bus: EventBus, session_id: str) -> None:
    """Run the scripted encounter. Called from a daemon thread."""
    time.sleep(2.0)

    all_claims: dict[str, str] = {}
    supersession_pending: dict[str, str] = {}

    for scripted in TORRES_SCRIPT:
        time.sleep(scripted.delay_s)

        ts_ns = int(time.time() * 1_000_000_000)
        turn_id = _insert_turn(conn, session_id, scripted, ts_ns)

        bus.publish("turn.added", {
            "turn_id": turn_id,
            "session_id": session_id,
        })

        for claim_data in scripted.claims:
            claim_ts = int(time.time() * 1_000_000_000)
            claim_info = _insert_claim(conn, session_id, turn_id, claim_data, claim_ts)

            claim_info["source_turn_text"] = scripted.text
            claim_info["source_turn_speaker"] = scripted.speaker.value
            claim_info["char_start"] = None
            claim_info["char_end"] = None

            bus.publish("claim.created", claim_info)

            tag = claim_data.get("_supersession_tag")
            if tag:
                supersession_pending[tag] = claim_info["claim_id"]

            superseded_by = claim_data.get("_superseded_by")
            if superseded_by:
                all_claims[superseded_by] = claim_info["claim_id"]

            time.sleep(0.3)

        for tag, old_claim_id in list(all_claims.items()):
            new_claim_id = supersession_pending.get(tag)
            if new_claim_id:
                edge_ts = int(time.time() * 1_000_000_000)
                _insert_supersession_edge(conn, old_claim_id, new_claim_id, edge_ts)
                bus.publish("claim.superseded", {
                    "old_claim_id": old_claim_id,
                    "new_claim_id": new_claim_id,
                    "edge_type": "patient_correction",
                    "identity_score": 0.85,
                    "session_id": session_id,
                })
                del all_claims[tag]
                del supersession_pending[tag]

        bus.publish("projection.updated", {
            "session_id": session_id,
            "active_count": -1,
        })

    time.sleep(2.0)
    _generate_soap_sentences(conn, bus, session_id)


def _generate_soap_sentences(
    conn: sqlite3.Connection,
    bus: EventBus,
    session_id: str,
) -> None:
    """Insert pre-scripted SOAP sentences with provenance links to actual claims."""
    active_claims = conn.execute(
        "SELECT claim_id, predicate FROM claims WHERE session_id = ? AND status = 'active'",
        (session_id,),
    ).fetchall()
    predicate_to_claim: dict[str, str] = {}
    for row in active_claims:
        predicate_to_claim[row[1]] = row[0]

    for soap in SOAP_SENTENCES:
        sentence_id = _make_sentence_id()
        source_claim_ids = [
            predicate_to_claim[p] for p in soap["predicates"]
            if p in predicate_to_claim
        ]

        conn.execute(
            """INSERT INTO note_sentences
               (sentence_id, session_id, section, ordinal, text, source_claim_ids)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                sentence_id, session_id,
                soap["section"], soap["ordinal"],
                soap["text"],
                ",".join(source_claim_ids),
            ),
        )
        conn.commit()

        bus.publish("note_sentence.added", {
            "sentence_id": sentence_id,
            "session_id": session_id,
            "section": soap["section"],
            "ordinal": soap["ordinal"],
            "text": soap["text"],
            "source_claim_ids": source_claim_ids,
        })

        time.sleep(0.5)
