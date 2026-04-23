"""Event-tuple projection over Claims.

A claim ⟨subject, predicate, value⟩ with the Wave-A temporal-validity window
⟨valid_from_ts, valid_until_ts⟩ projects naturally onto an *event tuple* —
the canonical representation used by Chronos (arXiv:2603.16862, ⟨subject, verb,
object, start_datetime, end_datetime⟩) and by the temporal KG family more
broadly (TEMPR — arXiv:2512.12818; Zep — arXiv:2501.13956). RobbyMD's
contribution: the projection is derived from *deterministically superseded*
claims (Pass-1 supersession sets `valid_until_ts` algorithmically) rather
than from LLM-generated revisions, so the temporal window for each tuple is
reproducible across runs.

This module is a pure read-side projection — it never writes back to the
substrate. The mapping `Claim → EventTuple` keeps the source `claim_id` so
downstream callers can recover full provenance (rules.md §3.4).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from src.substrate.schema import Claim


@dataclass(frozen=True, slots=True)
class EventTuple:
    """(subject, action, object, valid_from, valid_until) projection of a Claim.

    Aligned with Chronos (arXiv:2603.16862) ⟨subject, verb, object⟩ +
    start_datetime/end_datetime tuples. RobbyMD's contribution: derived from
    deterministically superseded claims, not LLM-generated revisions.

    `action` is the equivalent of Chronos's `verb` and the substrate's
    `predicate`; `obj` mirrors the substrate's `value`. We keep the substrate
    naming on the source side and use the temporal-KG vocabulary on the
    projection side so call-sites can read either way without renaming.
    """

    subject: str
    action: str
    obj: str
    valid_from_ts: int | None
    valid_until_ts: int | None
    claim_id: str


def claim_to_event(claim: Claim) -> EventTuple:
    """Map a single `Claim` to its `EventTuple` projection.

    Lossless on the five fields the projection covers — round-trip preserves
    `subject`, `predicate→action`, `value→obj`, both temporal-window
    boundaries, and `claim_id` for provenance.
    """
    return EventTuple(
        subject=claim.subject,
        action=claim.predicate,
        obj=claim.value,
        valid_from_ts=claim.valid_from_ts,
        valid_until_ts=claim.valid_until_ts,
        claim_id=claim.claim_id,
    )


def claims_to_events(claims: Iterable[Claim]) -> list[EventTuple]:
    """Map an iterable of `Claim`s to a list of `EventTuple`s.

    Order-preserving — useful when callers want to keep the substrate's
    `created_ts ASC` ordering (e.g. from `list_active_claims`).
    """
    return [claim_to_event(c) for c in claims]


__all__ = [
    "EventTuple",
    "claim_to_event",
    "claims_to_events",
]
