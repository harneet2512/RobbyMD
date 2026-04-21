"""LR-table loader + validated row model.

Per Eng_doc.md §4.4 and rules.md §4.4. Every row must carry a citation (non-empty
`source`). Approximations carry `approximation=True`. Invented numbers are rejected.

Load once, cache by `predicate_path` for O(1) lookup at engine-scoring time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.differential.types import ActiveClaim  # re-export for consumers  # noqa: F401

# Closed set from Eng_doc.md §4.2. Any predicate outside this list is rejected.
PREDICATE_FAMILIES: frozenset[str] = frozenset(
    {
        "onset",
        "character",
        "severity",
        "location",
        "radiation",
        "aggravating_factor",
        "alleviating_factor",
        "associated_symptom",
        "duration",
        "medical_history",
        "medication",
        "family_history",
        "social_history",
        "risk_factor",
    }
)

BRANCHES: frozenset[str] = frozenset({"cardiac", "pulmonary", "msk", "gi"})


@dataclass(frozen=True, slots=True)
class LRRow:
    """One row of the LR table — a single clinical feature with its pooled LRs.

    `lr_plus` / `lr_minus` may each be None when the source only publishes one
    direction. `approximation=True` flags any row whose single-feature LR was
    extrapolated from an adjacent published estimate (rules.md §4.4).
    """

    branch: str
    feature: str
    predicate_path: str  # e.g. "aggravating_factor=exertion"
    lr_plus: float | None
    lr_minus: float | None
    source: str
    source_url: str
    approximation: bool
    notes: str = ""

    @property
    def predicate(self) -> str:
        """The left-hand side of predicate_path (Eng_doc.md §4.2 family)."""
        return self.predicate_path.split("=", 1)[0]

    @property
    def value(self) -> str:
        """The right-hand side of predicate_path."""
        parts = self.predicate_path.split("=", 1)
        return parts[1] if len(parts) == 2 else ""


@dataclass(frozen=True, slots=True)
class LRTable:
    """Parsed LR table with deterministic indexing by predicate_path → list[LRRow].

    Multiple rows may share a `predicate_path` (e.g. `aggravating_factor=inspiration`
    appears across cardiac rule-out, pulmonary pleuritic pain, and MSK costochondritis
    branches). The engine iterates all rows for each claim.
    """

    version: str
    chief_complaint: str
    rows: tuple[LRRow, ...]
    by_predicate_path: dict[str, tuple[LRRow, ...]] = field(default_factory=dict)
    by_branch: dict[str, tuple[LRRow, ...]] = field(default_factory=dict)

    def rows_for(self, predicate_path: str) -> tuple[LRRow, ...]:
        """O(1) lookup — returns empty tuple when no rows match."""
        return self.by_predicate_path.get(predicate_path, ())

    def rows_on_branch(self, branch: str) -> tuple[LRRow, ...]:
        return self.by_branch.get(branch, ())


def _validate_row(raw: dict[str, Any], idx: int) -> LRRow:
    """Strict schema validation — rules.md §4.4: no invented values, every row sourced."""
    for required in ("branch", "feature", "predicate_path", "source", "source_url"):
        if required not in raw or not raw[required]:
            raise ValueError(f"lr_table row {idx}: missing or empty '{required}'")

    branch = str(raw["branch"]).strip().lower()
    if branch not in BRANCHES:
        raise ValueError(
            f"lr_table row {idx}: unknown branch '{branch}' (expected {sorted(BRANCHES)})"
        )

    predicate_path = str(raw["predicate_path"])
    if "=" not in predicate_path:
        raise ValueError(
            f"lr_table row {idx}: predicate_path '{predicate_path}' must be 'family=value'"
        )
    family = predicate_path.split("=", 1)[0]
    if family not in PREDICATE_FAMILIES:
        raise ValueError(
            f"lr_table row {idx}: predicate family '{family}' not in Eng_doc.md §4.2 closed set"
        )

    lr_plus = raw.get("lr_plus")
    lr_minus = raw.get("lr_minus")
    if lr_plus is None and lr_minus is None:
        raise ValueError(f"lr_table row {idx}: at least one of lr_plus / lr_minus required")
    if lr_plus is not None and (not isinstance(lr_plus, (int, float)) or lr_plus <= 0):
        raise ValueError(f"lr_table row {idx}: lr_plus must be positive number")
    if lr_minus is not None and (not isinstance(lr_minus, (int, float)) or lr_minus <= 0):
        raise ValueError(f"lr_table row {idx}: lr_minus must be positive number")

    return LRRow(
        branch=branch,
        feature=str(raw["feature"]),
        predicate_path=predicate_path,
        lr_plus=float(lr_plus) if lr_plus is not None else None,
        lr_minus=float(lr_minus) if lr_minus is not None else None,
        source=str(raw["source"]),
        source_url=str(raw["source_url"]),
        approximation=bool(raw.get("approximation", False)),
        notes=str(raw.get("notes", "")),
    )


def load_lr_table(path: str | Path) -> LRTable:
    """Load and validate an LR-table JSON file. Raises on any schema violation.

    Consumers treat the returned object as read-only. Deterministic: same file →
    same object contents (Python ≥3.7 dict insertion order is stable).
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)

    entries = raw.get("entries", [])
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{p}: 'entries' must be a non-empty list")

    rows = tuple(_validate_row(row, i) for i, row in enumerate(entries))

    by_predicate_path: dict[str, list[LRRow]] = {}
    by_branch: dict[str, list[LRRow]] = {}
    for r in rows:
        by_predicate_path.setdefault(r.predicate_path, []).append(r)
        by_branch.setdefault(r.branch, []).append(r)

    return LRTable(
        version=str(raw.get("version", "0.0.0")),
        chief_complaint=str(raw.get("chief_complaint", "")),
        rows=rows,
        by_predicate_path={k: tuple(v) for k, v in by_predicate_path.items()},
        by_branch={k: tuple(v) for k, v in by_branch.items()},
    )
