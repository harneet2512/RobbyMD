"""Patient Aftercare Agent — public interface.

The Patient Agent has access ONLY to physician-approved aftercare
information. Four allowed actions: answer from approved info, explain
terms, check red flags, escalate.

Uses Claude Managed Agents API with 7 custom tools.
"""
from __future__ import annotations

import sqlite3
from typing import Any, Iterator

from src.agents.orchestrator import create_orchestrator

import structlog

log = structlog.get_logger(__name__)


class PatientAgent:
    """Convenience wrapper for the Patient Aftercare Agent."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        encounter_id: str,
        use_managed: bool = True,
    ) -> None:
        self.encounter_id = encounter_id
        self._orchestrator = create_orchestrator(conn, use_managed=use_managed)
        self._session = self._orchestrator.create_patient_session(encounter_id)
        log.info(
            "patient_agent.created",
            encounter_id=encounter_id,
            mode="managed" if use_managed else "raw",
        )

    def ask(self, question: str) -> str:
        """Send a question and return the full text response."""
        parts: list[str] = []
        for event in self._session.send_message(question):
            if event["type"] == "text":
                parts.append(event["text"])
        return "".join(parts)

    def ask_stream(self, question: str) -> Iterator[dict[str, Any]]:
        """Send a question and yield streaming events."""
        yield from self._session.send_message(question)
