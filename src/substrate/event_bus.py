"""Simple synchronous in-memory pub/sub for UI subscribers.

Per `docs/gt_v2_study_notes.md` §2.8. The UI (and tests) subscribe to
named events; the orchestrator (`on_new_turn`) publishes.

Event names and payload shapes are the substrate ↔ UI contract; bump
the payload shape only after coordinating with `wt-ui`:

- `turn.added`                     {turn_id, session_id}
- `claim.created`                  {claim_id, session_id, predicate, status}
- `claim.superseded`               {old_claim_id, new_claim_id, edge_type,
                                    identity_score?}
- `claim.status_changed`           {claim_id, status}
- `projection.updated`             {session_id, active_count}
- `note_sentence.added`            {sentence_id, session_id, section,
                                    source_claim_ids}

Design choices:

- Synchronous. Callers get "event delivered" semantics before the publish
  call returns; no queue, no asyncio — the whole substrate pipeline is
  synchronous for the hackathon.
- Per-event callback lists, not a global stream; UI can subscribe to
  only the events it needs.
- **Subscriber exceptions do not crash the publisher.** We log at
  ERROR and move on. This is *not* `except Exception: pass` because
  we (a) log and (b) only shield sibling subscribers from one faulty
  one — the original request continues. Rules.md §8 style.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import structlog

log = structlog.get_logger(__name__)


# Event-name constants — preferred over raw strings at call sites so typos
# surface as attribute errors, not silent no-ops on an unknown event.
TURN_ADDED = "turn.added"
CLAIM_CREATED = "claim.created"
CLAIM_SUPERSEDED = "claim.superseded"
CLAIM_STATUS_CHANGED = "claim.status_changed"
PROJECTION_UPDATED = "projection.updated"
NOTE_SENTENCE_ADDED = "note_sentence.added"


Payload = dict[str, Any]
Callback = Callable[[Payload], None]


class EventBus:
    """Tiny synchronous pub/sub."""

    def __init__(self) -> None:
        self._subs: dict[str, list[Callback]] = {}

    def subscribe(self, event: str, callback: Callback) -> None:
        """Register a callback for one event."""
        self._subs.setdefault(event, []).append(callback)
        log.debug("substrate.event_subscribe", event_name=event)

    def unsubscribe(self, event: str, callback: Callback) -> None:
        """Remove a previously-registered callback. No-op if unknown."""
        if event in self._subs and callback in self._subs[event]:
            self._subs[event].remove(callback)

    def publish(self, event: str, payload: Payload) -> None:
        """Deliver `payload` to every subscriber synchronously.

        If a subscriber raises, we log and continue (shielding siblings).
        The publisher does **not** re-raise — UI callbacks failing should
        not corrupt substrate state.
        """
        subs = self._subs.get(event, ())
        for cb in subs:
            try:
                cb(payload)
            except Exception:
                log.exception(
                    "substrate.event_subscriber_failed",
                    event_name=event,
                    callback=getattr(cb, "__qualname__", repr(cb)),
                )

    def subscriber_count(self, event: str) -> int:
        return len(self._subs.get(event, ()))
