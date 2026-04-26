"""Diagnostic Bias Monitor Agent — public interface.

Detects systematic patterns across multiple encounters: anchoring bias,
premature closure, confirmation bias. Operates on accumulated encounter
history, not individual sessions.
"""
from __future__ import annotations

import sqlite3
from typing import Any, Iterator

from src.agents.orchestrator import create_orchestrator

import structlog

log = structlog.get_logger(__name__)


class BiasMonitorAgent:
    """Convenience wrapper for the Diagnostic Bias Monitor Agent."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        use_managed: bool = True,
    ) -> None:
        self._orchestrator = create_orchestrator(conn, use_managed=use_managed)
        self._session = self._orchestrator.create_bias_monitor_session()
        log.info(
            "bias_monitor_agent.created",
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

    def run_full_analysis(self) -> str:
        """Run all three bias detections and return a combined report."""
        return self.ask(
            "Run a complete bias analysis: check for anchoring bias, "
            "premature closure, and confirmation bias across all available "
            "encounters. Present the findings with evidence."
        )
