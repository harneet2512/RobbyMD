"""Managed Agent orchestrator — creates and manages Doctor + Patient agents.

Primary path: Claude Managed Agents API (client.beta.agents) with custom
tools. Each agent is a cloud-hosted Claude instance; custom tools execute
locally against our substrate and results are sent back.

Fallback path: raw client.messages.create() with tool_use loop. Works
offline and when the managed agents beta is unavailable.

Architecture:
    Two agent definitions  ──►  different tool manifests
    Shared environment      ──►  same substrate backend
    Per-conversation sessions ──► isolated chat histories
    Custom tool dispatch    ──►  local Python against SQLite
"""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Iterator

import structlog

log = structlog.get_logger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"
MODEL_ID = "claude-opus-4-7-20250415"


def _load_prompt(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8").strip()


# ═══════════════════════════════════════════════════════════════════════════
# PRIMARY: Managed Agents API
# ═══════════════════════════════════════════════════════════════════════════


class ManagedAgentOrchestrator:
    """Orchestrator using Claude Managed Agents (client.beta.agents)."""

    def __init__(
        self, conn: sqlite3.Connection, api_key: str | None = None
    ) -> None:
        from anthropic import Anthropic

        self.client = Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )
        self.conn = conn
        self._doctor_agent_id: str | None = None
        self._patient_agent_id: str | None = None
        self._environment_id: str | None = None

    def initialize(self) -> None:
        """Create agent definitions and environment. Call once at startup."""
        from src.agents.tools.doctor_tools import doctor_tool_manifest
        from src.agents.tools.patient_tools import patient_tool_manifest

        env = self.client.beta.environments.create(
            name="robbymd-aftercare",
            config={"type": "cloud", "networking": {"type": "unrestricted"}},
        )
        self._environment_id = env.id
        log.info("managed_agents.env_created", env_id=env.id)

        doctor = self.client.beta.agents.create(
            name="RobbyMD Doctor Aftercare Agent",
            model="claude-opus-4-7",
            system=_load_prompt("doctor_system.txt"),
            tools=doctor_tool_manifest(),
        )
        self._doctor_agent_id = doctor.id
        log.info("managed_agents.doctor_created", agent_id=doctor.id)

        patient = self.client.beta.agents.create(
            name="RobbyMD Patient Aftercare Agent",
            model="claude-opus-4-7",
            system=_load_prompt("patient_system.txt"),
            tools=patient_tool_manifest(),
        )
        self._patient_agent_id = patient.id
        log.info("managed_agents.patient_created", agent_id=patient.id)

    def create_doctor_session(
        self, encounter_id: str
    ) -> "ManagedAgentSession":
        from src.agents.tools.doctor_tools import DoctorToolDispatcher

        session = self.client.beta.sessions.create(
            agent=self._doctor_agent_id,
            environment_id=self._environment_id,
            title=f"Doctor review: {encounter_id}",
        )
        return ManagedAgentSession(
            self.client,
            session.id,
            DoctorToolDispatcher(self.conn),
            encounter_id,
        )

    def create_patient_session(
        self, encounter_id: str
    ) -> "ManagedAgentSession":
        from src.agents.tools.patient_tools import PatientToolDispatcher

        session = self.client.beta.sessions.create(
            agent=self._patient_agent_id,
            environment_id=self._environment_id,
            title=f"Patient aftercare: {encounter_id}",
        )
        return ManagedAgentSession(
            self.client,
            session.id,
            PatientToolDispatcher(self.conn),
            encounter_id,
        )


class ManagedAgentSession:
    """Live session with a managed agent. Handles the tool dispatch loop."""

    def __init__(
        self,
        client: Any,
        session_id: str,
        dispatcher: Any,
        encounter_id: str,
    ) -> None:
        self.client = client
        self.session_id = session_id
        self.dispatcher = dispatcher
        self.encounter_id = encounter_id

    def send_message(self, text: str) -> Iterator[dict[str, Any]]:
        """Send a user message and yield response events."""
        events_by_id: dict[str, Any] = {}

        with self.client.beta.sessions.events.stream(self.session_id) as stream:
            self.client.beta.sessions.events.send(
                self.session_id,
                events=[
                    {
                        "type": "user.message",
                        "content": [{"type": "text", "text": text}],
                    }
                ],
            )

            for event in stream:
                events_by_id[event.id] = event

                if event.type == "agent.message":
                    for block in event.content:
                        if hasattr(block, "text"):
                            yield {"type": "text", "text": block.text}

                elif event.type == "agent.custom_tool_use":
                    yield {
                        "type": "tool_call",
                        "name": event.name,
                        "input": event.input,
                    }

                elif event.type == "session.status_idle":
                    if (
                        hasattr(event, "stop_reason")
                        and event.stop_reason
                    ):
                        if event.stop_reason.type == "requires_action":
                            for event_id in event.stop_reason.event_ids:
                                tool_event = events_by_id[event_id]
                                result = self.dispatcher.dispatch(
                                    tool_event.name, tool_event.input
                                )
                                self.client.beta.sessions.events.send(
                                    self.session_id,
                                    events=[
                                        {
                                            "type": "user.custom_tool_result",
                                            "custom_tool_use_id": event_id,
                                            "content": [
                                                {
                                                    "type": "text",
                                                    "text": json.dumps(result),
                                                }
                                            ],
                                        }
                                    ],
                                )
                                yield {
                                    "type": "tool_result",
                                    "name": tool_event.name,
                                    "result": result,
                                }
                        elif event.stop_reason.type == "end_turn":
                            return

                elif event.type == "session.error":
                    msg = (
                        event.error.message
                        if hasattr(event, "error") and event.error
                        else "unknown"
                    )
                    yield {"type": "error", "message": msg}
                    return


# ═══════════════════════════════════════════════════════════════════════════
# FALLBACK: Raw messages.create() with tool_use loop
# ═══════════════════════════════════════════════════════════════════════════


class RawMessageOrchestrator:
    """Fallback using client.messages.create() with tool_use.

    For testing and offline usage when the managed agents beta API
    is unavailable.
    """

    def __init__(
        self, conn: sqlite3.Connection, api_key: str | None = None
    ) -> None:
        from anthropic import Anthropic

        self.client = Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )
        self.conn = conn

    def create_doctor_session(self, encounter_id: str) -> "RawAgentSession":
        from src.agents.tools.doctor_tools import (
            DoctorToolDispatcher,
            doctor_tool_manifest,
        )

        return RawAgentSession(
            self.client,
            _load_prompt("doctor_system.txt"),
            _to_messages_format(doctor_tool_manifest()),
            DoctorToolDispatcher(self.conn),
            encounter_id,
        )

    def create_patient_session(self, encounter_id: str) -> "RawAgentSession":
        from src.agents.tools.patient_tools import (
            PatientToolDispatcher,
            patient_tool_manifest,
        )

        return RawAgentSession(
            self.client,
            _load_prompt("patient_system.txt"),
            _to_messages_format(patient_tool_manifest()),
            PatientToolDispatcher(self.conn),
            encounter_id,
        )


class RawAgentSession:
    """Agentic tool_use loop via client.messages.create()."""

    def __init__(
        self,
        client: Any,
        system: str,
        tools: list[dict[str, Any]],
        dispatcher: Any,
        encounter_id: str,
    ) -> None:
        self.client = client
        self.system = system
        self.tools = tools
        self.dispatcher = dispatcher
        self.encounter_id = encounter_id
        self.messages: list[dict[str, Any]] = []

    def send_message(self, text: str) -> Iterator[dict[str, Any]]:
        """Send a user message, run tool loop, yield events."""
        self.messages.append({"role": "user", "content": text})

        while True:
            response = self.client.messages.create(
                model=MODEL_ID,
                system=self.system,
                tools=self.tools,
                messages=self.messages,
                max_tokens=4096,
            )

            assistant_content: list[dict[str, Any]] = []
            tool_calls: list[Any] = []

            for block in response.content:
                if block.type == "text":
                    assistant_content.append(
                        {"type": "text", "text": block.text}
                    )
                    yield {"type": "text", "text": block.text}
                elif block.type == "tool_use":
                    assistant_content.append(
                        {
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        }
                    )
                    tool_calls.append(block)
                    yield {
                        "type": "tool_call",
                        "name": block.name,
                        "input": block.input,
                    }

            self.messages.append(
                {"role": "assistant", "content": assistant_content}
            )

            if response.stop_reason == "tool_use" and tool_calls:
                tool_results: list[dict[str, Any]] = []
                for tc in tool_calls:
                    result = self.dispatcher.dispatch(tc.name, tc.input)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tc.id,
                            "content": json.dumps(result),
                        }
                    )
                    yield {
                        "type": "tool_result",
                        "name": tc.name,
                        "result": result,
                    }
                self.messages.append({"role": "user", "content": tool_results})
            else:
                return


# ═══════════════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════════════


def _to_messages_format(
    managed_tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert managed-agent tool format to messages API format."""
    return [
        {
            "name": t["name"],
            "description": t["description"],
            "input_schema": t["input_schema"],
        }
        for t in managed_tools
    ]


def create_orchestrator(
    conn: sqlite3.Connection,
    api_key: str | None = None,
    use_managed: bool = True,
) -> ManagedAgentOrchestrator | RawMessageOrchestrator:
    """Factory: prefer managed agents, fall back to raw messages."""
    if use_managed:
        try:
            orch = ManagedAgentOrchestrator(conn, api_key)
            orch.initialize()
            log.info("orchestrator.using_managed_agents")
            return orch
        except Exception as exc:
            log.warning("managed_agents.fallback", error=str(exc))

    log.info("orchestrator.using_raw_messages")
    return RawMessageOrchestrator(conn, api_key)
