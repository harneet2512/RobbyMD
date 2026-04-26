"""Clinical Note Co-Author Agent — public interface.

Collaborative SOAP note editing where physician edits feed back into
the substrate. Detects conflicts, proposes substrate actions, requires
physician approval for mutations.
"""
from __future__ import annotations

import sqlite3
from typing import Any, Iterator

from src.agents.orchestrator import create_orchestrator

import structlog

log = structlog.get_logger(__name__)


class NoteCoauthorAgent:
    """Convenience wrapper for the Clinical Note Co-Author Agent."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        encounter_id: str,
        use_managed: bool = True,
    ) -> None:
        self.encounter_id = encounter_id
        self._orchestrator = create_orchestrator(conn, use_managed=use_managed)
        self._session = self._orchestrator.create_note_coauthor_session(encounter_id)
        log.info(
            "note_coauthor_agent.created",
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

    def submit_edit(self, edited_text: str) -> str:
        """Submit an edited note and get conflict analysis."""
        return self.ask(
            f"The physician has submitted an edited note. Please analyze "
            f"it for conflicts with active claims:\n\n{edited_text}"
        )
