"""Decision log CRUD — physician decisions with evidence snapshots.

The ``decisions`` table exists in schema.py but has no API functions.
This module provides them, following claims.py patterns.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from enum import StrEnum
from time import time_ns

import structlog

log = structlog.get_logger(__name__)


class DecisionKind(StrEnum):
    CONFIRM_CLAIM = "confirm_claim"
    DISMISS_CLAIM = "dismiss_claim"
    DOWNGRADE_BRANCH = "downgrade_branch"
    UPGRADE_BRANCH = "upgrade_branch"
    REQUEST_TEST = "request_test"
    CLOSE_ENCOUNTER = "close_encounter"
    APPROVE_AFTERCARE = "approve_aftercare"


class TargetType(StrEnum):
    CLAIM = "claim"
    BRANCH = "branch"
    ENCOUNTER = "encounter"
    AFTERCARE_PACKAGE = "aftercare_package"


@dataclass(frozen=True, slots=True)
class Decision:
    decision_id: str
    session_id: str
    kind: DecisionKind
    target_type: TargetType
    target_id: str
    claim_state_snapshot: dict
    ts: int


def new_decision_id() -> str:
    return f"dc_{uuid.uuid4().hex[:12]}"


def record_decision(
    conn: sqlite3.Connection,
    session_id: str,
    kind: DecisionKind,
    target_type: TargetType,
    target_id: str,
    claim_state_snapshot: dict,
) -> Decision:
    decision_id = new_decision_id()
    ts = time_ns()
    conn.execute(
        "INSERT INTO decisions (decision_id, session_id, kind, target_type,"
        " target_id, claim_state_snapshot, ts) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            decision_id,
            session_id,
            kind.value,
            target_type.value,
            target_id,
            json.dumps(claim_state_snapshot),
            ts,
        ),
    )
    conn.commit()
    decision = Decision(
        decision_id=decision_id,
        session_id=session_id,
        kind=kind,
        target_type=target_type,
        target_id=target_id,
        claim_state_snapshot=claim_state_snapshot,
        ts=ts,
    )
    log.info(
        "decision.recorded",
        decision_id=decision_id,
        session_id=session_id,
        kind=kind.value,
    )
    return decision


def get_decisions(
    conn: sqlite3.Connection, session_id: str
) -> tuple[Decision, ...]:
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM decisions WHERE session_id = ? ORDER BY ts",
        (session_id,),
    ).fetchall()
    return tuple(_row_to_decision(r) for r in rows)


def get_decision(
    conn: sqlite3.Connection, decision_id: str
) -> Decision | None:
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM decisions WHERE decision_id = ?",
        (decision_id,),
    ).fetchone()
    return _row_to_decision(row) if row else None


def _row_to_decision(row: sqlite3.Row) -> Decision:
    return Decision(
        decision_id=row["decision_id"],
        session_id=row["session_id"],
        kind=DecisionKind(row["kind"]),
        target_type=TargetType(row["target_type"]),
        target_id=row["target_id"],
        claim_state_snapshot=json.loads(row["claim_state_snapshot"]),
        ts=row["ts"],
    )
