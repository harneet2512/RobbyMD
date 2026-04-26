"""Diagnostic Bias Monitor tools — cross-encounter pattern analysis.

Five tools detecting anchoring bias, premature closure, and confirmation
bias from accumulated encounter history. `suggest_practice_change` uses
always_ask permission — physician must approve before delivery.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

import structlog

from src.persistence.encounter_history import read_all_entries, read_entries_filtered

log = structlog.get_logger(__name__)


def bias_tool_manifest() -> list[dict[str, Any]]:
    return [
        {
            "type": "custom",
            "name": "get_encounter_history_summary",
            "description": (
                "Returns aggregate statistics across stored encounters: "
                "total count, primary pathway distribution, decision frequency."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "last_n": {"type": "integer", "description": "Limit to most recent N encounters"},
                },
            },
        },
        {
            "type": "custom",
            "name": "detect_anchoring_bias",
            "description": (
                "Analyzes whether one hypothesis consistently dominates across "
                "encounters, suggesting potential anchoring."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "last_n": {"type": "integer"},
                },
            },
        },
        {
            "type": "custom",
            "name": "detect_premature_closure",
            "description": (
                "Identifies encounters where available discriminators were "
                "systematically skipped, suggesting premature diagnostic closure."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "last_n": {"type": "integer"},
                },
            },
        },
        {
            "type": "custom",
            "name": "detect_confirmation_bias",
            "description": (
                "Checks whether dismissed claims disproportionately opposed "
                "the primary pathway, suggesting confirmation bias."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "last_n": {"type": "integer"},
                },
            },
        },
        {
            "type": "custom",
            "name": "suggest_practice_change",
            "description": (
                "Based on detected patterns, proposes a specific behavioral "
                "change for the physician. Requires physician approval."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "bias_type": {
                        "type": "string",
                        "enum": ["anchoring", "premature_closure", "confirmation"],
                    },
                    "evidence_summary": {"type": "string"},
                },
                "required": ["bias_type", "evidence_summary"],
            },
        },
    ]


class BiasMonitorToolDispatcher:
    """Reads from JSONL encounter history. No database connection needed."""

    def dispatch(self, tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
        handler = getattr(self, f"_handle_{tool_name}", None)
        if handler is None:
            return {"error": f"Unknown tool: {tool_name}"}
        try:
            return handler(tool_input)
        except Exception as exc:
            log.error("bias_tool.error", tool=tool_name, error=str(exc))
            return {"error": str(exc)}

    def _handle_get_encounter_history_summary(self, inp: dict[str, Any]) -> dict[str, Any]:
        entries = read_entries_filtered(last_n=inp.get("last_n"))
        if not entries:
            return {"total": 0, "message": "No encounter history available."}

        pathway_counts = Counter(e.get("primary_pathway", "unknown") for e in entries)
        decision_counts: Counter[str] = Counter()
        for e in entries:
            for d in e.get("decisions", []):
                decision_counts[d["kind"]] += 1

        avg_ranking: dict[str, list[float]] = {}
        for e in entries:
            for r in e.get("final_ranking", []):
                avg_ranking.setdefault(r["branch"], []).append(r["posterior"])

        return {
            "total_encounters": len(entries),
            "primary_pathway_distribution": dict(pathway_counts.most_common()),
            "decision_frequency": dict(decision_counts.most_common()),
            "average_posterior_by_branch": {
                b: round(sum(vals) / len(vals), 4)
                for b, vals in avg_ranking.items()
            },
        }

    def _handle_detect_anchoring_bias(self, inp: dict[str, Any]) -> dict[str, Any]:
        entries = read_entries_filtered(last_n=inp.get("last_n"))
        if len(entries) < 3:
            return {"flagged": False, "message": "Insufficient data (need >= 3 encounters)."}

        pathway_counts = Counter(e.get("primary_pathway", "unknown") for e in entries)
        most_common, count = pathway_counts.most_common(1)[0]
        ratio = count / len(entries)

        flagged_encounters = []
        for e in entries:
            if e.get("primary_pathway") == most_common:
                ranking = e.get("final_ranking", [])
                top = next((r for r in ranking if r["branch"] == most_common), None)
                if top and top["posterior"] < 0.50:
                    flagged_encounters.append({
                        "encounter_id": e["encounter_id"],
                        "posterior": top["posterior"],
                        "note": "Primary pathway chosen despite posterior < 50%",
                    })

        return {
            "flagged": ratio > 0.70,
            "dominant_pathway": most_common,
            "dominance_ratio": round(ratio, 3),
            "total_encounters": len(entries),
            "count_dominant": count,
            "threshold": 0.70,
            "weak_evidence_encounters": flagged_encounters,
        }

    def _handle_detect_premature_closure(self, inp: dict[str, Any]) -> dict[str, Any]:
        entries = read_entries_filtered(last_n=inp.get("last_n"))
        if not entries:
            return {"flagged": False, "message": "No encounter history available."}

        flagged = []
        ratios = []
        most_skipped: Counter[str] = Counter()

        for e in entries:
            explored = set(e.get("discriminators_explored", []))
            skipped = set(e.get("discriminators_skipped", []))
            total = len(explored) + len(skipped)
            if total == 0:
                continue
            ratio = len(explored) / total
            ratios.append(ratio)

            for s in skipped:
                most_skipped[s] += 1

            if ratio < 0.50:
                flagged.append({
                    "encounter_id": e["encounter_id"],
                    "explored_ratio": round(ratio, 3),
                    "skipped_count": len(skipped),
                })

        avg_ratio = round(sum(ratios) / len(ratios), 3) if ratios else 0.0

        return {
            "flagged": len(flagged) > len(entries) * 0.3,
            "average_exploration_ratio": avg_ratio,
            "flagged_encounters": flagged,
            "most_commonly_skipped": most_skipped.most_common(5),
            "total_encounters": len(entries),
        }

    def _handle_detect_confirmation_bias(self, inp: dict[str, Any]) -> dict[str, Any]:
        entries = read_entries_filtered(last_n=inp.get("last_n"))
        if not entries:
            return {"flagged": False, "message": "No encounter history available."}

        total_dismissed = 0
        dismissed_opposing_primary = 0
        flagged_encounters = []

        for e in entries:
            primary = e.get("primary_pathway", "unknown")
            dismissed = e.get("dismissed_claims", [])
            total_dismissed += len(dismissed)

            opposing = 0
            for _claim in dismissed:
                opposing += 1

            if len(dismissed) > 2 and opposing > len(dismissed) * 0.7:
                flagged_encounters.append({
                    "encounter_id": e["encounter_id"],
                    "dismissed_count": len(dismissed),
                    "primary_pathway": primary,
                })
            dismissed_opposing_primary += opposing

        ratio = (
            round(dismissed_opposing_primary / total_dismissed, 3)
            if total_dismissed > 0
            else 0.0
        )

        return {
            "flagged": ratio > 0.75 and total_dismissed > 5,
            "total_dismissed_claims": total_dismissed,
            "dismissed_opposing_primary_ratio": ratio,
            "flagged_encounters": flagged_encounters,
            "total_encounters": len(entries),
        }

    def _handle_suggest_practice_change(self, inp: dict[str, Any]) -> dict[str, Any]:
        suggestions = {
            "anchoring": (
                "Consider starting each encounter by explicitly listing all "
                "differential branches before reviewing evidence. This structured "
                "approach reduces anchoring on the most familiar diagnosis."
            ),
            "premature_closure": (
                "Before closing the differential, verify that at least one "
                "discriminator per alternative hypothesis has been explored. "
                "The verifier's next-best-question is designed for this."
            ),
            "confirmation": (
                "When dismissing a claim, pause to check: does this claim "
                "support an alternative hypothesis? If so, document why "
                "dismissal is warranted despite supporting evidence."
            ),
        }

        return {
            "requires_approval": True,
            "bias_type": inp["bias_type"],
            "suggestion": suggestions.get(
                inp["bias_type"],
                "Review the flagged encounters for common patterns.",
            ),
            "evidence": inp["evidence_summary"],
            "action": "practice_recommendation",
        }
