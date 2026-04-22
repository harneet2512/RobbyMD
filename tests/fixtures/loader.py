"""Shared fixture loaders for wt-trees tests."""

from __future__ import annotations

import json
from pathlib import Path

from src.differential.types import ActiveClaim

_REPO_ROOT = Path(__file__).resolve().parents[2]
LR_TABLE_PATH = (
    _REPO_ROOT
    / "predicate_packs"
    / "clinical_general"
    / "differentials"
    / "chest_pain"
    / "lr_table.json"
)
MID_CASE_PATH = _REPO_ROOT / "tests" / "fixtures" / "chest_pain_mid_case.json"


def load_mid_case_claims(path: Path | str | None = None) -> list[ActiveClaim]:
    """Load the synthetic mid-case active-claim set (tests/fixtures/chest_pain_mid_case.json)."""
    p = Path(path) if path else MID_CASE_PATH
    with p.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    claims = []
    for c in raw["active_claims"]:
        claims.append(
            ActiveClaim(
                claim_id=str(c["claim_id"]),
                predicate_path=str(c["predicate_path"]),
                polarity=bool(c.get("polarity", True)),
                confidence=float(c.get("confidence", 1.0)),
                source_turn_id=str(c.get("source_turn_id", "")),
            )
        )
    return claims
