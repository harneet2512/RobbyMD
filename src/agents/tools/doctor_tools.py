"""Doctor Agent tools — full reasoning substrate access.

Eight tools giving the Doctor Agent complete access to the encounter's
reasoning graph, evidence, decisions, and escalations.
"""
from __future__ import annotations

import sqlite3
from typing import Any

import structlog

from src.aftercare.escalation import get_escalation_store
from src.aftercare.package import get_cached_package
from src.differential.engine import rank_branches
from src.differential.lr_table import load_lr_table
from src.differential.types import ActiveClaim
from src.substrate.claims import list_active_claims, list_claims_with_lifecycle
from src.substrate.decisions import get_decisions
from src.substrate.predicate_packs import active_pack
from src.substrate.projections import rebuild_active_projection

log = structlog.get_logger(__name__)


def doctor_tool_manifest() -> list[dict[str, Any]]:
    """Tool definitions for the Doctor managed agent."""
    return [
        {
            "type": "custom",
            "name": "get_encounter_summary",
            "description": (
                "Returns the physician-facing summary: primary pathway, "
                "deprioritized hypotheses, pending workups, key decisions."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "encounter_id": {"type": "string"},
                },
                "required": ["encounter_id"],
            },
        },
        {
            "type": "custom",
            "name": "get_reasoning_review",
            "description": (
                "Returns the full reasoning graph: all claims, hypothesis "
                "branches with weights, supersession edges, decisions."
            ),
            "input_schema": {
                "type": "object",
                "properties": {"encounter_id": {"type": "string"}},
                "required": ["encounter_id"],
            },
        },
        {
            "type": "custom",
            "name": "query_evidence_for_against",
            "description": (
                "Returns evidence supporting and contradicting a specific "
                "hypothesis at the current reasoning state."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "encounter_id": {"type": "string"},
                    "hypothesis": {
                        "type": "string",
                        "description": "Branch: cardiac, pulmonary, msk, or gi",
                    },
                },
                "required": ["encounter_id", "hypothesis"],
            },
        },
        {
            "type": "custom",
            "name": "rerank_with_new_evidence",
            "description": (
                "Re-ranks hypotheses with new evidence (late labs, imaging). "
                "Does NOT mutate the locked encounter."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "encounter_id": {"type": "string"},
                    "new_evidence": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "predicate": {"type": "string"},
                                "value": {"type": "string"},
                            },
                        },
                    },
                },
                "required": ["encounter_id", "new_evidence"],
            },
        },
        {
            "type": "custom",
            "name": "review_patient_escalation",
            "description": "Reviews a patient escalation with full reasoning context.",
            "input_schema": {
                "type": "object",
                "properties": {"escalation_id": {"type": "string"}},
                "required": ["escalation_id"],
            },
        },
        {
            "type": "custom",
            "name": "get_dissent_log",
            "description": (
                "Returns every point where the system and the physician "
                "disagreed during the encounter."
            ),
            "input_schema": {
                "type": "object",
                "properties": {"encounter_id": {"type": "string"}},
                "required": ["encounter_id"],
            },
        },
        {
            "type": "custom",
            "name": "get_note",
            "description": "Returns the finalized SOAP note with provenance chain.",
            "input_schema": {
                "type": "object",
                "properties": {"encounter_id": {"type": "string"}},
                "required": ["encounter_id"],
            },
        },
        {
            "type": "custom",
            "name": "list_escalations",
            "description": "Lists patient escalations, optionally unreviewed only.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "encounter_id": {"type": "string"},
                    "unreviewed_only": {"type": "boolean", "default": True},
                },
                "required": ["encounter_id"],
            },
        },
    ]


class DoctorToolDispatcher:
    """Executes doctor tools against the substrate."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def dispatch(self, tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
        handler = getattr(self, f"_handle_{tool_name}", None)
        if handler is None:
            return {"error": f"Unknown tool: {tool_name}"}
        try:
            return handler(tool_input)
        except Exception as exc:
            log.error("doctor_tool.error", tool=tool_name, error=str(exc))
            return {"error": str(exc)}

    def _handle_get_encounter_summary(self, inp: dict[str, Any]) -> dict[str, Any]:
        sid = inp["encounter_id"]
        claims = list_active_claims(self.conn, sid)
        decisions = get_decisions(self.conn, sid)
        ranking = _build_ranking(self.conn, sid, claims)

        return {
            "encounter_id": sid,
            "primary_pathway": ranking[0] if ranking else None,
            "deprioritized": ranking[1:] if len(ranking) > 1 else [],
            "active_claims_count": len(claims),
            "decisions": [
                {"kind": d.kind.value, "target": d.target_id, "ts": d.ts}
                for d in decisions
            ],
        }

    def _handle_get_reasoning_review(self, inp: dict[str, Any]) -> dict[str, Any]:
        sid = inp["encounter_id"]
        all_claims = list_claims_with_lifecycle(self.conn, sid, mode="all")
        decisions = get_decisions(self.conn, sid)
        ranking = _build_ranking(
            self.conn,
            sid,
            [c for c in all_claims if c.status.value in ("active", "confirmed")],
        )

        return {
            "encounter_id": sid,
            "claims": {
                "active": [
                    _claim_dict(c)
                    for c in all_claims
                    if c.status.value in ("active", "confirmed")
                ],
                "superseded": [
                    _claim_dict(c)
                    for c in all_claims
                    if c.status.value == "superseded"
                ],
                "dismissed": [
                    _claim_dict(c)
                    for c in all_claims
                    if c.status.value == "dismissed"
                ],
            },
            "hypothesis_ranking": ranking,
            "decisions": [
                {
                    "decision_id": d.decision_id,
                    "kind": d.kind.value,
                    "target": d.target_id,
                    "snapshot": d.claim_state_snapshot,
                }
                for d in decisions
            ],
        }

    def _handle_query_evidence_for_against(
        self, inp: dict[str, Any]
    ) -> dict[str, Any]:
        sid = inp["encounter_id"]
        hypothesis = inp["hypothesis"].lower()
        claims = list_active_claims(self.conn, sid)

        pack = active_pack()
        lr_table = load_lr_table(pack.lr_table_path) if pack.lr_table_path else None

        evidence_for: list[dict[str, Any]] = []
        evidence_against: list[dict[str, Any]] = []
        missing: list[dict[str, Any]] = []

        if lr_table:
            claim_paths = {
                f"{c.predicate}={c.value_normalised or c.value}": c for c in claims
            }
            for row in lr_table.rows_on_branch(hypothesis):
                matched = claim_paths.get(row.predicate_path)
                if matched:
                    entry = {
                        "feature": row.feature,
                        "claim": _claim_dict(matched),
                        "lr_plus": row.lr_plus,
                        "lr_minus": row.lr_minus,
                    }
                    if row.lr_plus and row.lr_plus > 1.0:
                        evidence_for.append(entry)
                    else:
                        evidence_against.append(entry)
                else:
                    missing.append(
                        {"feature": row.feature, "lr_plus": row.lr_plus}
                    )

        return {
            "hypothesis": hypothesis,
            "for": evidence_for,
            "against": evidence_against,
            "missing": missing,
        }

    def _handle_rerank_with_new_evidence(
        self, inp: dict[str, Any]
    ) -> dict[str, Any]:
        sid = inp["encounter_id"]
        new_evidence = inp["new_evidence"]

        claims = list_active_claims(self.conn, sid)
        pack = active_pack()
        lr_table = load_lr_table(pack.lr_table_path) if pack.lr_table_path else None

        if not lr_table:
            return {"error": "No LR table available"}

        active = _claims_to_active(claims)
        for ev in new_evidence:
            path = f"{ev['predicate']}={ev['value']}"
            active.append(
                ActiveClaim(
                    claim_id=f"new_{ev['predicate']}",
                    predicate_path=path,
                    polarity=not ev["value"].startswith("negated:"),
                )
            )

        new_ranking = rank_branches(tuple(active), lr_table)
        return {
            "reranked_hypotheses": [
                {
                    "branch": r.branch,
                    "posterior": round(r.posterior, 4),
                    "applied_lrs": len(r.applied),
                }
                for r in new_ranking.scores
            ],
            "new_evidence_added": [
                f"{e['predicate']}={e['value']}" for e in new_evidence
            ],
            "note": "Post-visit addendum. Original encounter is preserved.",
        }

    def _handle_review_patient_escalation(
        self, inp: dict[str, Any]
    ) -> dict[str, Any]:
        store = get_escalation_store()
        esc = store.get(inp["escalation_id"])
        if not esc:
            return {"error": f"Escalation {inp['escalation_id']} not found"}

        reasoning_context: dict[str, Any] = {}
        if esc.matched_red_flag:
            for branch in ("cardiac", "pulmonary", "msk", "gi"):
                ctx = self._handle_query_evidence_for_against(
                    {"encounter_id": esc.encounter_id, "hypothesis": branch}
                )
                if ctx.get("for") or ctx.get("missing"):
                    reasoning_context[branch] = ctx

        return {
            "escalation": esc.to_dict(),
            "reasoning_context": reasoning_context,
        }

    def _handle_get_dissent_log(self, inp: dict[str, Any]) -> dict[str, Any]:
        sid = inp["encounter_id"]
        all_claims = list_claims_with_lifecycle(self.conn, sid, mode="all")
        dissents = [
            {
                "claim": _claim_dict(c),
                "type": "physician_override",
                "note": f"Physician dismissed: {c.predicate}={c.value}",
            }
            for c in all_claims
            if c.status.value == "dismissed"
        ]
        return {"encounter_id": sid, "dissents": dissents}

    def _handle_get_note(self, inp: dict[str, Any]) -> dict[str, Any]:
        sid = inp["encounter_id"]
        pkg = get_cached_package(sid)
        if pkg:
            return {
                "encounter_id": sid,
                "note": pkg.summary,
                "source": "aftercare_package",
            }

        rows = self.conn.execute(
            "SELECT section, ordinal, text, source_claim_ids "
            "FROM note_sentences WHERE session_id = ? "
            "ORDER BY section, ordinal",
            (sid,),
        ).fetchall()

        if rows:
            sections: dict[str, list[str]] = {}
            for r in rows:
                sections.setdefault(r[0], []).append(r[2])
            return {
                "encounter_id": sid,
                "note": {sec: "\n".join(texts) for sec, texts in sections.items()},
                "source": "note_sentences",
            }

        return {"encounter_id": sid, "note": None, "source": "none"}

    def _handle_list_escalations(self, inp: dict[str, Any]) -> dict[str, Any]:
        store = get_escalation_store()
        escalations = store.list_for_encounter(
            inp["encounter_id"],
            unreviewed_only=inp.get("unreviewed_only", True),
        )
        return {
            "encounter_id": inp["encounter_id"],
            "escalations": [e.to_dict() for e in escalations],
        }


def _build_ranking(
    conn: sqlite3.Connection,
    session_id: str,
    claims: list[Any],
) -> list[dict[str, Any]]:
    pack = active_pack()
    lr_table = load_lr_table(pack.lr_table_path) if pack.lr_table_path else None

    if not lr_table or not claims:
        return [
            {"branch": b, "posterior": 0.25}
            for b in ("cardiac", "pulmonary", "msk", "gi")
        ]

    active = _claims_to_active(claims)
    ranking = rank_branches(tuple(active), lr_table)
    return [
        {
            "branch": r.branch,
            "posterior": round(r.posterior, 4),
            "applied_count": len(r.applied),
        }
        for r in ranking.scores
    ]


def _claims_to_active(claims: list[Any]) -> list[ActiveClaim]:
    active: list[ActiveClaim] = []
    for c in claims:
        val = c.value_normalised or c.value
        path = f"{c.predicate}={val}"
        active.append(
            ActiveClaim(
                claim_id=c.claim_id,
                predicate_path=path,
                polarity=not val.startswith("negated:"),
            )
        )
    return active


def _claim_dict(c: Any) -> dict[str, Any]:
    return {
        "claim_id": c.claim_id,
        "predicate": c.predicate,
        "value": c.value,
        "status": c.status.value if hasattr(c.status, "value") else str(c.status),
        "source_turn_id": c.source_turn_id,
    }
