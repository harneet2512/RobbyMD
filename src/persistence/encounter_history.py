"""Append-only encounter history for diagnostic bias monitoring."""
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
_HISTORY_PATH = _REPO_ROOT / "data" / "encounter_history.jsonl"
_lock = threading.Lock()


def append_encounter_entry(
    conn: sqlite3.Connection,
    encounter_id: str,
    chief_complaint: str = "chest pain",
) -> dict[str, Any]:
    """Build and append one encounter's summary to the history store."""
    from src.substrate.claims import list_active_claims, list_claims_with_lifecycle
    from src.substrate.decisions import get_decisions
    from src.differential.engine import rank_branches
    from src.differential.lr_table import load_lr_table
    from src.differential.types import ActiveClaim
    from src.substrate.predicate_packs import active_pack
    from src.verifier.verifier import select_discriminator, _default_client

    all_claims = list_claims_with_lifecycle(conn, encounter_id, mode="all")
    active_claims = [c for c in all_claims if c.status.value in ("active", "confirmed")]
    decisions = get_decisions(conn, encounter_id)

    pack = active_pack()
    lr_table = load_lr_table(pack.lr_table_path) if pack.lr_table_path else None

    ranking_data: list[dict[str, Any]] = []
    primary_pathway = "unknown"
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
            {"branch": s.branch, "posterior": round(s.posterior, 4)}
            for s in ranking.scores
        ]
        if ranking.scores:
            primary_pathway = ranking.scores[0].branch

    all_discriminators = set()
    explored_discriminators = set()
    if lr_table:
        for branch in lr_table.branches:
            for row in lr_table.rows_on_branch(branch):
                all_discriminators.add(row.feature)
        claim_paths = {f"{c.predicate}={c.value_normalised or c.value}" for c in active_claims}
        for row_set in lr_table.by_predicate_path.values():
            for row in row_set:
                if row.predicate_path in claim_paths:
                    explored_discriminators.add(row.feature)

    entry: dict[str, Any] = {
        "encounter_id": encounter_id,
        "closed_at": datetime.now(UTC).isoformat(),
        "chief_complaint": chief_complaint,
        "final_ranking": ranking_data,
        "primary_pathway": primary_pathway,
        "decisions": [{"kind": d.kind.value, "target_id": d.target_id} for d in decisions],
        "dismissed_hypotheses": [
            r["branch"] for r in ranking_data if r["posterior"] < 0.10
        ],
        "dismissed_claims": [
            {"predicate": c.predicate, "value": c.value}
            for c in all_claims if c.status.value == "dismissed"
        ],
        "confirmed_claims": [
            {"predicate": c.predicate, "value": c.value}
            for c in all_claims if c.status.value == "confirmed"
        ],
        "discriminators_explored": sorted(explored_discriminators),
        "discriminators_skipped": sorted(all_discriminators - explored_discriminators),
    }

    _HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        with open(_HISTORY_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    log.info("encounter_history.appended", encounter_id=encounter_id)
    return entry


def read_all_entries() -> list[dict[str, Any]]:
    if not _HISTORY_PATH.exists():
        return []
    entries = []
    for line in _HISTORY_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def read_entries_filtered(
    chief_complaint: str | None = None,
    last_n: int | None = None,
) -> list[dict[str, Any]]:
    entries = read_all_entries()
    if chief_complaint:
        entries = [e for e in entries if e.get("chief_complaint") == chief_complaint]
    if last_n:
        entries = entries[-last_n:]
    return entries
