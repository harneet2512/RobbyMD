"""Shift Handoff Agent — public interface.

Helps incoming physicians understand the clinical reasoning state from
a previous physician's encounter. Read-only access to encounter snapshots.
"""
from __future__ import annotations

import sqlite3
from typing import Any, Iterator

from src.agents.orchestrator import create_orchestrator

import structlog

log = structlog.get_logger(__name__)


class HandoffAgent:
    """Convenience wrapper for the Shift Handoff Agent."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        encounter_id: str,
        use_managed: bool = True,
    ) -> None:
        self.encounter_id = encounter_id
        self._orchestrator = create_orchestrator(conn, use_managed=use_managed)
        self._session = self._orchestrator.create_handoff_session(encounter_id)
        log.info(
            "handoff_agent.created",
            encounter_id=encounter_id,
            mode="managed" if use_managed else "raw",
        )

    def ask(self, question: str) -> str:
        parts: list[str] = []
        for event in self._session.send_message(question):
            if event["type"] == "text":
                parts.append(event["text"])
        return "".join(parts)

    def ask_stream(self, question: str) -> Iterator[dict[str, Any]]:
        yield from self._session.send_message(question)
