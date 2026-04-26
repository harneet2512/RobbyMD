"""Shift Handoff Agent tools — read-only access to prior encounter snapshots.

Six tools for incoming physicians to query the full reasoning state
from a prior physician's encounter. No substrate modification.
"""
from __future__ import annotations

from typing import Any

import structlog

from src.persistence.snapshots import (
    list_encounter_snapshots,
    read_encounter_snapshot,
)

log = structlog.get_logger(__name__)


def handoff_tool_manifest() -> list[dict[str, Any]]:
    return [
        {
            "type": "custom",
            "name": "get_prior_encounter_state",
            "description": (
                "Returns the full reasoning snapshot for a closed encounter: "
                "differential ranking, all claims, decisions, pending items."
            ),
            "input_schema": {
                "type": "object",
                "properties": {"encounter_id": {"type": "string"}},
                "required": ["encounter_id"],
            },
        },
        {
            "type": "custom",
            "name": "get_prior_differential_ranking",
            "description": "Returns the final hypothesis ranking from a prior encounter.",
            "input_schema": {
                "type": "object",
                "properties": {"encounter_id": {"type": "string"}},
                "required": ["encounter_id"],
            },
        },
        {
            "type": "custom",
            "name": "get_prior_decisions",
            "description": (
                "Returns all physician decisions with evidence snapshots "
                "from a prior encounter."
            ),
            "input_schema": {
                "type": "object",
                "properties": {"encounter_id": {"type": "string"}},
                "required": ["encounter_id"],
            },
        },
        {
            "type": "custom",
            "name": "get_prior_unresolved_items",
            "description": (
                "Returns unresolved claims, pending follow-ups, and the "
                "dissent log from a prior encounter."
            ),
            "input_schema": {
                "type": "object",
                "properties": {"encounter_id": {"type": "string"}},
                "required": ["encounter_id"],
            },
        },
        {
            "type": "custom",
            "name": "list_available_encounters",
            "description": "Lists all encounter snapshots available for handoff review.",
            "input_schema": {
                "type": "object",
                "properties": {},
            },
        },
        {
            "type": "custom",
            "name": "compare_encounters",
            "description": (
                "Compares differential rankings between two encounters — "
                "useful when the same patient returns."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "encounter_id_a": {"type": "string"},
                    "encounter_id_b": {"type": "string"},
                },
                "required": ["encounter_id_a", "encounter_id_b"],
            },
        },
    ]


class HandoffToolDispatcher:
    """Read-only dispatcher — queries persisted encounter snapshots."""

    def dispatch(self, tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
        handler = getattr(self, f"_handle_{tool_name}", None)
        if handler is None:
            return {"error": f"Unknown tool: {tool_name}"}
        try:
            return handler(tool_input)
        except Exception as exc:
            log.error("handoff_tool.error", tool=tool_name, error=str(exc))
            return {"error": str(exc)}

    def _handle_get_prior_encounter_state(self, inp: dict[str, Any]) -> dict[str, Any]:
        snap = read_encounter_snapshot(inp["encounter_id"])
        if not snap:
            return {"error": f"No snapshot found for {inp['encounter_id']}"}
        return snap

    def _handle_get_prior_differential_ranking(self, inp: dict[str, Any]) -> dict[str, Any]:
        snap = read_encounter_snapshot(inp["encounter_id"])
        if not snap:
            return {"error": f"No snapshot found for {inp['encounter_id']}"}
        return {
            "encounter_id": inp["encounter_id"],
            "differential_ranking": snap.get("differential_ranking", []),
            "closed_at": snap.get("closed_at"),
        }

    def _handle_get_prior_decisions(self, inp: dict[str, Any]) -> dict[str, Any]:
        snap = read_encounter_snapshot(inp["encounter_id"])
        if not snap:
            return {"error": f"No snapshot found for {inp['encounter_id']}"}
        return {
            "encounter_id": inp["encounter_id"],
            "decisions": snap.get("decisions", []),
            "total": len(snap.get("decisions", [])),
        }

    def _handle_get_prior_unresolved_items(self, inp: dict[str, Any]) -> dict[str, Any]:
        snap = read_encounter_snapshot(inp["encounter_id"])
        if not snap:
            return {"error": f"No snapshot found for {inp['encounter_id']}"}
        return {
            "encounter_id": inp["encounter_id"],
            "unresolved_claims": snap.get("unresolved_claims", []),
            "pending_follow_ups": snap.get("pending_follow_ups", []),
            "dissent_log": snap.get("dissent_log", []),
            "aftercare_status": snap.get("aftercare_status", {}),
        }

    def _handle_list_available_encounters(self, _inp: dict[str, Any]) -> dict[str, Any]:
        snapshots = list_encounter_snapshots()
        return {"encounters": snapshots, "total": len(snapshots)}

    def _handle_compare_encounters(self, inp: dict[str, Any]) -> dict[str, Any]:
        snap_a = read_encounter_snapshot(inp["encounter_id_a"])
        snap_b = read_encounter_snapshot(inp["encounter_id_b"])
        if not snap_a:
            return {"error": f"No snapshot for {inp['encounter_id_a']}"}
        if not snap_b:
            return {"error": f"No snapshot for {inp['encounter_id_b']}"}

        rank_a = {r["branch"]: r["posterior"] for r in snap_a.get("differential_ranking", [])}
        rank_b = {r["branch"]: r["posterior"] for r in snap_b.get("differential_ranking", [])}
        all_branches = sorted(set(rank_a) | set(rank_b))

        comparison = []
        for branch in all_branches:
            pa = rank_a.get(branch, 0.0)
            pb = rank_b.get(branch, 0.0)
            comparison.append({
                "branch": branch,
                "encounter_a_posterior": pa,
                "encounter_b_posterior": pb,
                "delta": round(pb - pa, 4),
            })

        decisions_a = {d["kind"] for d in snap_a.get("decisions", [])}
        decisions_b = {d["kind"] for d in snap_b.get("decisions", [])}

        return {
            "encounter_a": inp["encounter_id_a"],
            "encounter_b": inp["encounter_id_b"],
            "ranking_comparison": comparison,
            "decisions_a_count": len(snap_a.get("decisions", [])),
            "decisions_b_count": len(snap_b.get("decisions", [])),
            "new_decision_types_in_b": sorted(decisions_b - decisions_a),
        }
