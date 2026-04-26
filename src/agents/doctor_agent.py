"""Doctor Aftercare Agent — public interface.

The Doctor Agent has full access to the reasoning substrate. It helps
physicians review clinical reasoning, handle late evidence, process
patient escalations, and support covering physician handoff.

Uses Claude Managed Agents API with 8 custom tools.
"""
from __future__ import annotations

import sqlite3
from typing import Any, Iterator

from src.agents.orchestrator import create_orchestrator

import structlog

log = structlog.get_logger(__name__)


class DoctorAgent:
    """Convenience wrapper for the Doctor Aftercare Agent."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        encounter_id: str,
        use_managed: bool = True,
    ) -> None:
        self.encounter_id = encounter_id
        self._orchestrator = create_orchestrator(conn, use_managed=use_managed)
        self._session = self._orchestrator.create_doctor_session(encounter_id)
        log.info(
            "doctor_agent.created",
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

    def review_escalation(self, escalation_id: str) -> str:
        """Convenience: ask the agent to review a specific escalation."""
        return self.ask(
            f"A patient escalation has arrived (ID: {escalation_id}). "
            "Please review it with full reasoning context and recommend "
            "an urgency level."
        )
