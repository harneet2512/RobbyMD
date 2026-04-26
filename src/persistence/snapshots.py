"""Encounter snapshot persistence — JSON files capturing full reasoning state at encounter close."""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SNAPSHOTS_DIR = _REPO_ROOT / "data" / "snapshots"
_lock = threading.Lock()


def write_encounter_snapshot(
    conn: sqlite3.Connection,
    encounter_id: str,
    physician_id: str | None = None,
    chief_complaint: str = "chest pain",
) -> Path:
    """Capture the full reasoning state for an encounter and persist as JSON."""
    from src.substrate.claims import list_active_claims, list_claims_with_lifecycle
    from src.substrate.decisions import get_decisions
    from src.differential.engine import rank_branches
    from src.differential.lr_table import load_lr_table
    from src.differential.types import ActiveClaim
    from src.substrate.predicate_packs import active_pack
    from src.aftercare.package import get_cached_package

    all_claims = list_claims_with_lifecycle(conn, encounter_id, mode="all")
    active_claims = [c for c in all_claims if c.status.value in ("active", "confirmed")]
    decisions = get_decisions(conn, encounter_id)

    pack = active_pack()
    lr_table = load_lr_table(pack.lr_table_path) if pack.lr_table_path else None

    ranking_data: list[dict[str, Any]] = []
    if lr_table and active_claims:
        active = [
            ActiveClaim(
                claim_id=c.claim_id,
                predicate_path=f"{c.predicate}={c.value_normalised or c.value}",
                polarity=not (c.value_normalised or c.value).startswith("negated:"),
            )
            for c in active_claims
        ]
        ranking = rank_branches(tuple(active), lr_table)
        ranking_data = [
            {"branch": s.branch, "posterior": round(s.posterior, 4), "applied_count": len(s.applied)}
            for s in ranking.scores
        ]

    pkg = get_cached_package(encounter_id)

    snapshot: dict[str, Any] = {
        "encounter_id": encounter_id,
        "closed_at": datetime.now(UTC).isoformat(),
        "physician_id": physician_id,
        "chief_complaint": chief_complaint,
        "differential_ranking": ranking_data,
        "decisions": [
            {
                "decision_id": d.decision_id,
                "kind": d.kind.value,
                "target_type": d.target_type.value,
                "target_id": d.target_id,
                "claim_state_snapshot": d.claim_state_snapshot,
                "ts": d.ts,
            }
            for d in decisions
        ],
        "active_claims": [_claim_to_dict(c) for c in active_claims],
        "superseded_claims": [
            _claim_to_dict(c) for c in all_claims if c.status.value == "superseded"
        ],
        "dismissed_claims": [
            _claim_to_dict(c) for c in all_claims if c.status.value == "dismissed"
        ],
        "unresolved_claims": [
            _claim_to_dict(c) for c in active_claims if c.confidence < 0.7
        ],
        "pending_follow_ups": [],
        "aftercare_status": {
            "generated": pkg is not None,
            "approved": pkg.approved if pkg else False,
        },
        "dissent_log": [
            {"claim_id": c.claim_id, "predicate": c.predicate, "value": c.value}
            for c in all_claims
            if c.status.value == "dismissed"
        ],
    }

    if pkg:
        snapshot["pending_follow_ups"] = [
            {"action": f.action, "timeframe": f.timeframe}
            for f in pkg.follow_up_plan
        ]

    _SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = _SNAPSHOTS_DIR / f"{encounter_id}.json"
    with _lock:
        path.write_text(json.dumps(snapshot, indent=2, default=str), encoding="utf-8")

    log.info("snapshot.written", encounter_id=encounter_id, path=str(path))
    return path


def read_encounter_snapshot(encounter_id: str) -> dict[str, Any] | None:
    path = _SNAPSHOTS_DIR / f"{encounter_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def list_encounter_snapshots() -> list[dict[str, str]]:
    if not _SNAPSHOTS_DIR.exists():
        return []
    results = []
    for p in sorted(_SNAPSHOTS_DIR.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            results.append({
                "encounter_id": data["encounter_id"],
                "closed_at": data.get("closed_at", ""),
                "chief_complaint": data.get("chief_complaint", ""),
                "physician_id": data.get("physician_id", ""),
            })
        except (json.JSONDecodeError, KeyError):
            continue
    return results


def _claim_to_dict(c: Any) -> dict[str, Any]:
    return {
        "claim_id": c.claim_id,
        "predicate": c.predicate,
        "value": c.value,
        "confidence": c.confidence,
        "status": c.status.value if hasattr(c.status, "value") else str(c.status),
        "source_turn_id": c.source_turn_id,
    }
