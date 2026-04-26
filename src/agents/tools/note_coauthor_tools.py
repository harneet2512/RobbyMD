"""Clinical Note Co-Author tools — bidirectional SOAP note editing.

Seven tools enabling collaborative note editing where physician edits
feed back into the substrate. Substrate-modifying tools use always_ask
permission — physician must approve before mutations execute.
"""
from __future__ import annotations

import difflib
import sqlite3
from typing import Any

import structlog

from src.persistence.note_versions import (
    get_latest_version,
    list_versions,
    save_note_version,
)
from src.substrate.claims import list_active_claims, set_claim_status
from src.substrate.schema import ClaimStatus

log = structlog.get_logger(__name__)


def note_coauthor_tool_manifest() -> list[dict[str, Any]]:
    return [
        {
            "type": "custom",
            "name": "generate_note_draft",
            "description": (
                "Generates initial SOAP note from the encounter's active claims "
                "and dialogue. Saves as version 1."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "encounter_id": {"type": "string"},
                    "dialogue_text": {"type": "string", "description": "Full encounter dialogue"},
                },
                "required": ["encounter_id"],
            },
        },
        {
            "type": "custom",
            "name": "get_current_note",
            "description": "Returns the latest note version with diff from previous.",
            "input_schema": {
                "type": "object",
                "properties": {"encounter_id": {"type": "string"}},
                "required": ["encounter_id"],
            },
        },
        {
            "type": "custom",
            "name": "submit_physician_edit",
            "description": (
                "Accepts edited note text. Detects conflicts with active claims "
                "and returns a list of conflicts for review."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "encounter_id": {"type": "string"},
                    "edited_note_text": {"type": "string"},
                },
                "required": ["encounter_id", "edited_note_text"],
            },
        },
        {
            "type": "custom",
            "name": "get_note_conflicts",
            "description": "Returns unresolved conflicts from the latest note version.",
            "input_schema": {
                "type": "object",
                "properties": {"encounter_id": {"type": "string"}},
                "required": ["encounter_id"],
            },
        },
        {
            "type": "custom",
            "name": "propose_claim_dismissal",
            "description": (
                "Proposes dismissing a claim that conflicts with physician edits. "
                "Requires physician approval before executing."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "encounter_id": {"type": "string"},
                    "claim_id": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["encounter_id", "claim_id", "reason"],
            },
        },
        {
            "type": "custom",
            "name": "propose_claim_confirmation",
            "description": (
                "Proposes confirming a claim reinforced by physician edits. "
                "Requires physician approval before executing."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "encounter_id": {"type": "string"},
                    "claim_id": {"type": "string"},
                },
                "required": ["encounter_id", "claim_id"],
            },
        },
        {
            "type": "custom",
            "name": "propose_supersession",
            "description": (
                "Proposes creating a supersession edge when physician edits "
                "refine a claim's value. Requires physician approval."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "encounter_id": {"type": "string"},
                    "old_claim_id": {"type": "string"},
                    "new_value": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["encounter_id", "old_claim_id", "new_value", "reason"],
            },
        },
    ]


ALWAYS_ASK_TOOLS = frozenset({
    "propose_claim_dismissal",
    "propose_claim_confirmation",
    "propose_supersession",
})


class NoteCoauthorToolDispatcher:
    """Dispatches note co-authoring tools. Substrate-modifying tools return
    requires_approval=True instead of executing directly."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def dispatch(self, tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
        handler = getattr(self, f"_handle_{tool_name}", None)
        if handler is None:
            return {"error": f"Unknown tool: {tool_name}"}
        try:
            return handler(tool_input)
        except Exception as exc:
            log.error("note_coauthor_tool.error", tool=tool_name, error=str(exc))
            return {"error": str(exc)}

    def execute_approved_action(
        self, action_type: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute a previously proposed action after physician approval."""
        if action_type == "dismiss_claim":
            set_claim_status(self.conn, params["claim_id"], ClaimStatus.DISMISSED)
            from src.substrate.decisions import record_decision, DecisionKind, TargetType
            record_decision(
                self.conn,
                session_id=params["encounter_id"],
                kind=DecisionKind.DISMISS_CLAIM,
                target_type=TargetType.CLAIM,
                target_id=params["claim_id"],
                claim_state_snapshot={"reason": params.get("reason", "physician_edit"), "source": "note_coauthor"},
            )
            return {"executed": True, "action": "dismiss_claim", "claim_id": params["claim_id"]}

        if action_type == "confirm_claim":
            set_claim_status(self.conn, params["claim_id"], ClaimStatus.CONFIRMED)
            from src.substrate.decisions import record_decision, DecisionKind, TargetType
            record_decision(
                self.conn,
                session_id=params["encounter_id"],
                kind=DecisionKind.CONFIRM_CLAIM,
                target_type=TargetType.CLAIM,
                target_id=params["claim_id"],
                claim_state_snapshot={"source": "note_coauthor"},
            )
            return {"executed": True, "action": "confirm_claim", "claim_id": params["claim_id"]}

        if action_type == "supersede_claim":
            from src.substrate.claims import get_claim, insert_claim
            old = get_claim(self.conn, params["old_claim_id"])
            if not old:
                return {"error": f"Claim {params['old_claim_id']} not found"}
            set_claim_status(self.conn, params["old_claim_id"], ClaimStatus.SUPERSEDED)
            new_claim = insert_claim(
                self.conn,
                session_id=old.session_id,
                subject=old.subject,
                predicate=old.predicate,
                value=params["new_value"],
                confidence=old.confidence,
                source_turn_id=old.source_turn_id,
            )
            return {
                "executed": True,
                "action": "supersede_claim",
                "old_claim_id": params["old_claim_id"],
                "new_claim_id": new_claim.claim_id,
            }

        return {"error": f"Unknown action type: {action_type}"}

    def _handle_generate_note_draft(self, inp: dict[str, Any]) -> dict[str, Any]:
        encounter_id = inp["encounter_id"]
        claims = list_active_claims(self.conn, encounter_id)

        sections: dict[str, list[str]] = {"S": [], "O": [], "A": [], "P": []}
        for c in claims:
            if c.predicate in ("onset", "character", "severity", "location", "radiation",
                               "aggravating_factor", "alleviating_factor", "associated_symptom",
                               "duration", "social_history", "family_history"):
                sections["S"].append(f"Patient reports {c.predicate.replace('_', ' ')}: {c.value}.")
            elif c.predicate == "medical_history":
                sections["O"].append(f"Medical history: {c.value}.")
            elif c.predicate == "medication":
                sections["O"].append(f"Current medication: {c.value}.")
            elif c.predicate == "risk_factor":
                sections["A"].append(f"Risk factor identified: {c.value}.")

        if not sections["P"]:
            sections["P"].append("Follow up with patient as clinically indicated.")

        note_parts = []
        for sec_key, sec_name in [("S", "Subjective"), ("O", "Objective"), ("A", "Assessment"), ("P", "Plan")]:
            note_parts.append(f"## {sec_name}")
            if sections[sec_key]:
                note_parts.extend(sections[sec_key])
            else:
                note_parts.append(f"No {sec_name.lower()} findings documented.")
            note_parts.append("")

        note_text = "\n".join(note_parts)
        version = save_note_version(encounter_id, note_text, source="generated")

        return {
            "encounter_id": encounter_id,
            "note_text": note_text,
            "version": version["version"],
            "active_claims_used": len(claims),
        }

    def _handle_get_current_note(self, inp: dict[str, Any]) -> dict[str, Any]:
        latest = get_latest_version(inp["encounter_id"])
        if not latest:
            return {"error": "No note versions found. Generate a draft first."}
        return {
            "encounter_id": inp["encounter_id"],
            "note_text": latest["note_text"],
            "version": latest["version"],
            "source": latest["source"],
            "diff_from_previous": latest.get("diff_from_previous"),
        }

    def _handle_submit_physician_edit(self, inp: dict[str, Any]) -> dict[str, Any]:
        encounter_id = inp["encounter_id"]
        edited_text = inp["edited_note_text"]

        latest = get_latest_version(encounter_id)
        if not latest:
            version = save_note_version(encounter_id, edited_text, source="physician_edit")
            return {"version": version["version"], "conflicts": []}

        prev_lines = set(latest["note_text"].splitlines())
        new_lines = set(edited_text.splitlines())
        removed_lines = prev_lines - new_lines

        claims = list_active_claims(self.conn, encounter_id)
        conflicts: list[dict[str, Any]] = []

        for claim in claims:
            claim_mentions = [claim.value.lower(), claim.predicate.replace("_", " ")]
            for removed in removed_lines:
                if any(mention in removed.lower() for mention in claim_mentions):
                    conflicts.append({
                        "conflict_type": "contradicts_active_claim",
                        "removed_text": removed.strip(),
                        "claim_id": claim.claim_id,
                        "claim_predicate": claim.predicate,
                        "claim_value": claim.value,
                        "claim_status": claim.status.value,
                        "claim_confidence": claim.confidence,
                    })

        version = save_note_version(
            encounter_id, edited_text, source="physician_edit", conflicts=conflicts,
        )

        return {
            "encounter_id": encounter_id,
            "version": version["version"],
            "conflicts": conflicts,
            "conflict_count": len(conflicts),
        }

    def _handle_get_note_conflicts(self, inp: dict[str, Any]) -> dict[str, Any]:
        latest = get_latest_version(inp["encounter_id"])
        if not latest:
            return {"conflicts": []}
        return {
            "encounter_id": inp["encounter_id"],
            "version": latest["version"],
            "conflicts": latest.get("conflicts", []),
        }

    def _handle_propose_claim_dismissal(self, inp: dict[str, Any]) -> dict[str, Any]:
        from src.substrate.claims import get_claim
        claim = get_claim(self.conn, inp["claim_id"])
        if not claim:
            return {"error": f"Claim {inp['claim_id']} not found"}
        return {
            "requires_approval": True,
            "action": "dismiss_claim",
            "claim_id": inp["claim_id"],
            "claim_predicate": claim.predicate,
            "claim_value": claim.value,
            "current_status": claim.status.value,
            "reason": inp["reason"],
            "encounter_id": inp["encounter_id"],
        }

    def _handle_propose_claim_confirmation(self, inp: dict[str, Any]) -> dict[str, Any]:
        from src.substrate.claims import get_claim
        claim = get_claim(self.conn, inp["claim_id"])
        if not claim:
            return {"error": f"Claim {inp['claim_id']} not found"}
        return {
            "requires_approval": True,
            "action": "confirm_claim",
            "claim_id": inp["claim_id"],
            "claim_predicate": claim.predicate,
            "claim_value": claim.value,
            "current_status": claim.status.value,
            "encounter_id": inp["encounter_id"],
        }

    def _handle_propose_supersession(self, inp: dict[str, Any]) -> dict[str, Any]:
        from src.substrate.claims import get_claim
        claim = get_claim(self.conn, inp["old_claim_id"])
        if not claim:
            return {"error": f"Claim {inp['old_claim_id']} not found"}
        return {
            "requires_approval": True,
            "action": "supersede_claim",
            "old_claim_id": inp["old_claim_id"],
            "old_value": claim.value,
            "new_value": inp["new_value"],
            "predicate": claim.predicate,
            "reason": inp["reason"],
            "encounter_id": inp["encounter_id"],
        }
