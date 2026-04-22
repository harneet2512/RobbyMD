"""Projections — materialised views over active claims.

Per `Eng_doc.md` §4.3 + `docs/gt_v2_study_notes.md` §2.5:

Two projection kinds:

1.  **Active-claims projection** — the session's current active / confirmed
    claim set, one row per (subject, predicate). Used by any UI panel
    that needs "the current state of what we know."

2.  **Per-branch projection** — filter of the active-claims projection
    keyed by the four differential branches (Cardiac / Pulmonary / MSK /
    GI). `wt-trees` owns the LR table; here we expose a deterministic
    filter hook so the differential engine can request "what active
    claims are relevant to branch X?"

Pure Python, no LLM, no randomness. Recompute ≤50 ms target per
`Eng_doc.md` §4.3.
"""
from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Final

import structlog

from src.substrate.claims import list_active_claims
from src.substrate.schema import Claim

log = structlog.get_logger(__name__)


# Differential branch names used by the UI + differential engine. Kept
# here as the single source of truth so wt-trees can `from src.substrate.projections
# import BRANCH_NAMES` rather than repeating string constants.
BRANCH_NAMES: Final[tuple[str, ...]] = ("cardiac", "pulmonary", "msk", "gi")


@dataclass(frozen=True, slots=True)
class ActiveProjection:
    """The authoritative active claim set for a session at a point in time."""

    session_id: str
    claims: tuple[Claim, ...]

    @property
    def by_predicate(self) -> dict[str, tuple[Claim, ...]]:
        """Claims grouped by predicate (stable order preserved)."""
        groups: dict[str, list[Claim]] = {}
        for c in self.claims:
            groups.setdefault(c.predicate, []).append(c)
        return {k: tuple(v) for k, v in groups.items()}

    def for_branch(self, predicate_matcher: Callable[[Claim], bool]) -> BranchProjection:
        filtered = tuple(c for c in self.claims if predicate_matcher(c))
        return BranchProjection(session_id=self.session_id, claims=filtered)


@dataclass(frozen=True, slots=True)
class BranchProjection:
    """Active claims filtered down to those relevant to one differential branch."""

    session_id: str
    claims: tuple[Claim, ...]


# ---------------------------------------------------------- rebuild helpers ---


def rebuild_active_projection(
    conn: sqlite3.Connection, session_id: str
) -> ActiveProjection:
    """Rebuild the active-claims projection for a session.

    Implementation note: there is no caching. `list_active_claims` is a cheap
    indexed SELECT; "rebuild" here is just idiomatic naming that mirrors the
    GT v2 projections/briefing pattern. If caching ever matters we add it
    here without touching callers.
    """
    claims = tuple(list_active_claims(conn, session_id))
    log.debug(
        "substrate.active_projection_rebuilt",
        session_id=session_id,
        active_count=len(claims),
    )
    return ActiveProjection(session_id=session_id, claims=claims)


def per_branch_projection(
    active: ActiveProjection,
    branch: str,
    predicate_matcher: Callable[[Claim], bool],
) -> BranchProjection:
    """Project active claims to the subset relevant to one branch.

    `predicate_matcher` is supplied by `wt-trees` (the LR table owner) so
    `src/substrate/` stays decoupled from clinical content. The differential
    engine will typically construct a matcher like:

        def matcher(claim: Claim) -> bool:
            return (claim.predicate, claim.value_normalised or claim.value.lower()) in ROWS

    Called once per (branch, claim-state-change) → four calls per update.
    """
    if branch not in BRANCH_NAMES:
        raise ValueError(f"unknown branch {branch!r}; allowed={BRANCH_NAMES}")
    return active.for_branch(predicate_matcher)


def claims_grouped_by_subject_predicate(
    claims: Iterable[Claim],
) -> dict[tuple[str, str], Claim]:
    """Keep only the newest active claim per (subject, predicate).

    A safety net for downstream consumers — if two active claims ever land
    for the same identity (shouldn't happen post-supersession, but Pass 2
    with `NullEmbedder` could leave dupes), return the most recent.
    """
    out: dict[tuple[str, str], Claim] = {}
    for c in claims:
        key = (c.subject, c.predicate)
        prev = out.get(key)
        if prev is None or c.created_ts > prev.created_ts:
            out[key] = c
    return out
