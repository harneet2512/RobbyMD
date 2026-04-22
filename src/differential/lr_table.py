"""LR-table loader + validated row model.

Per `Eng_doc.md §4.4` and `rules.md §4.4`. Every row must carry a citation
(non-empty `source`). Approximations carry `approximation=True`. Invented
numbers are rejected.

**Pack-aware (2026-04-21)**: `PREDICATE_FAMILIES` is derived from the active
`PredicatePack` (`src/substrate/predicate_packs.py::active_pack`), not a
hardcoded module constant. This lets a non-clinical pack (e.g.
`personal_assistant` for LongMemEval-S) use a different closed vocabulary
without engine changes. Addresses audit finding #1 from commit `767d3e8`.

**No hardcoded `BRANCHES`**: branches are derived from the loaded LR table
(`LRTable.branches`). Empty LR table → empty `branches` → differential engine
no-ops gracefully — required for packs with no differentials.

Load once, cache by `predicate_path` for O(1) lookup at engine-scoring time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.differential.types import ActiveClaim  # re-export for consumers  # noqa: F401
from src.substrate.predicate_packs import active_pack


def _active_predicate_families() -> frozenset[str]:
    """Return the active pack's closed predicate vocabulary (re-read each call).

    We don't cache at module import because tests may mutate `ACTIVE_PACK` and
    call `active_pack.cache_clear()`; this helper follows the active pack state.
    """
    return active_pack().predicate_families


# Backward-compatible module symbol. Resolved lazily against the active pack —
# value is correct for the default `clinical_general` pack on first import.
# Re-reading the active pack in `_validate_row` keeps the source of truth live.
PREDICATE_FAMILIES: frozenset[str] = _active_predicate_families()


@dataclass(frozen=True, slots=True)
class LRRow:
    """One row of the LR table — a single clinical feature with its pooled LRs.

    `lr_plus` / `lr_minus` may each be None when the source only publishes one
    direction. `approximation=True` flags any row whose single-feature LR was
    extrapolated from an adjacent published estimate (`rules.md §4.4`).
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
        """The left-hand side of predicate_path (active pack's family)."""
        return self.predicate_path.split("=", 1)[0]

    @property
    def value(self) -> str:
        """The right-hand side of predicate_path."""
        parts = self.predicate_path.split("=", 1)
        return parts[1] if len(parts) == 2 else ""


@dataclass(frozen=True, slots=True)
class LRTable:
    """Parsed LR table with deterministic indexing by predicate_path → list[LRRow].

    `branches` is derived from observed rows at load time — empty frozenset for
    tables with no rows (e.g. a non-clinical pack). The differential engine
    no-ops gracefully on an empty branch set.

    Multiple rows may share a `predicate_path` (e.g. `aggravating_factor=inspiration`
    appears across cardiac rule-out, pulmonary pleuritic pain, and MSK costochondritis
    branches). The engine iterates all rows for each claim.
    """

    version: str
    chief_complaint: str
    rows: tuple[LRRow, ...]
    branches: frozenset[str] = field(default_factory=frozenset)
    by_predicate_path: dict[str, tuple[LRRow, ...]] = field(default_factory=dict)
    by_branch: dict[str, tuple[LRRow, ...]] = field(default_factory=dict)

    def rows_for(self, predicate_path: str) -> tuple[LRRow, ...]:
        """O(1) lookup — returns empty tuple when no rows match."""
        return self.by_predicate_path.get(predicate_path, ())

    def rows_on_branch(self, branch: str) -> tuple[LRRow, ...]:
        return self.by_branch.get(branch, ())

    @classmethod
    def empty(cls, chief_complaint: str = "") -> LRTable:
        """Construct an empty LR table — for packs with no differentials.

        The differential engine treats an empty table as a no-op: `rank_branches`
        returns `BranchRanking(scores=())`. Required for `personal_assistant`
        (no differential hypotheses) and any future data-only pack.
        """
        return cls(
            version="0.0.0",
            chief_complaint=chief_complaint,
            rows=(),
            branches=frozenset(),
            by_predicate_path={},
            by_branch={},
        )


def _validate_row(raw: dict[str, Any], idx: int) -> LRRow:
    """Strict schema validation — `rules.md §4.4`: no invented values, every row sourced.

    Branch name is not checked against a hardcoded allowlist — per the 2026-04-21
    refactor, branch vocabulary is per-pack / per-complaint. We validate that the
    branch name is a non-empty single token (no whitespace); semantic meaning is
    the pack author's responsibility.

    Predicate family IS checked against the active pack's closed vocabulary.
    """
    for required in ("branch", "feature", "predicate_path", "source", "source_url"):
        if required not in raw or not raw[required]:
            raise ValueError(f"lr_table row {idx}: missing or empty '{required}'")

    branch = str(raw["branch"]).strip().lower()
    if not branch or any(ch.isspace() for ch in branch):
        raise ValueError(
            f"lr_table row {idx}: branch '{branch!r}' must be a non-empty single token"
        )

    predicate_path = str(raw["predicate_path"])
    if "=" not in predicate_path:
        raise ValueError(
            f"lr_table row {idx}: predicate_path '{predicate_path}' must be 'family=value'"
        )
    family = predicate_path.split("=", 1)[0]
    allowed = _active_predicate_families()
    if family not in allowed:
        raise ValueError(
            f"lr_table row {idx}: predicate family '{family}' not in active pack's closed set"
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

    Empty `entries` is permitted — returns `LRTable.empty(chief_complaint=...)`
    so packs with no differentials (e.g. `personal_assistant`) can still round-trip
    through the loader without special-casing.

    Consumers treat the returned object as read-only. Deterministic: same file →
    same object contents (Python ≥3.7 dict insertion order is stable).
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)

    entries = raw.get("entries", [])
    if not isinstance(entries, list):
        raise ValueError(f"{p}: 'entries' must be a list (got {type(entries).__name__})")

    chief_complaint = str(raw.get("chief_complaint", ""))

    if not entries:
        return LRTable.empty(chief_complaint=chief_complaint)

    rows = tuple(_validate_row(row, i) for i, row in enumerate(entries))

    by_predicate_path: dict[str, list[LRRow]] = {}
    by_branch: dict[str, list[LRRow]] = {}
    for r in rows:
        by_predicate_path.setdefault(r.predicate_path, []).append(r)
        by_branch.setdefault(r.branch, []).append(r)

    branches = frozenset(r.branch for r in rows)

    return LRTable(
        version=str(raw.get("version", "0.0.0")),
        chief_complaint=chief_complaint,
        rows=rows,
        branches=branches,
        by_predicate_path={k: tuple(v) for k, v in by_predicate_path.items()},
        by_branch={k: tuple(v) for k, v in by_branch.items()},
    )
