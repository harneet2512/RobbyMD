"""Shared claim shape for the differential engine.

Decoupled from `src/substrate/claims.py` (owned by wt-engine). Once wt-engine
ships its `ActiveClaimSet`, we accept either a `list[ActiveClaim]` or any
iterable of duck-typed objects exposing the attributes on `ActiveClaim`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ActiveClaim:
    """A deterministic projection of a claim suitable for LR scoring.

    Mirrors the `claims` table in Eng_doc.md §4.1 with the fields the engine
    needs. `polarity` indicates whether the feature is *present* (True) or
    *absent / explicitly negated* (False). The engine uses LR+ for present
    claims and LR- for absent/negated claims.

    `claim_id` and `source_turn_id` are carried for provenance UI (rules.md §4.1);
    the engine itself does not consume them.
    """

    claim_id: str
    predicate_path: str  # e.g. "aggravating_factor=exertion"
    polarity: bool = True  # True=present, False=absent/negated
    confidence: float = 1.0
    source_turn_id: str = ""


@dataclass(frozen=True, slots=True)
class PhysicianOverride:
    """A physician's explicit steering of a differential branch.

    `direction` is "upgrade" or "downgrade". `weight` is the log-odds
    adjustment (default 2.0 ≈ LR of 7.4). Enters the same audit trail
    as LR-based evidence so the full reasoning trace — machine AND human
    — is reconstructible.
    """

    decision_id: str
    branch: str
    direction: str  # "upgrade" | "downgrade"
    weight: float = 2.0
    physician_id: str = ""
    rationale: str = ""
