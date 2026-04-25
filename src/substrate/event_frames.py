"""Event-frame assembly layer above claims.

Groups co-referent claims into multi-slot event records. Claims remain
the atomic provenance unit; event frames are a compositional view that
enables slot-fill questions ("Where did I redeem the coupon?") to be
answered from fragments spread across turns.

Assembled after ingestion + supersession, stored in ``event_frames``
and ``event_frame_claims`` tables (schema in ``schema.py``).
"""
from __future__ import annotations

import json
import re
import sqlite3
import uuid
from collections import defaultdict
from dataclasses import dataclass

import structlog

from src.substrate.claims import list_active_claims
from src.substrate.schema import Claim

log = structlog.get_logger(__name__)


EVENT_TYPES: frozenset[str] = frozenset({
    "purchase_redemption",
    "travel_commute",
    "education_milestone",
    "named_artifact",
    "appointment",
    "location_linked_action",
})

_EVENT_SIGNALS: dict[str, list[str]] = {
    "purchase_redemption": [
        "bought", "purchased", "redeemed", "coupon", "paid", "price",
        "$", "checkout", "receipt", "sale",
    ],
    "travel_commute": [
        "commute", "drive", "drove", "flew", "traveled", "trip",
        "flight", "commuting", "transit",
    ],
    "education_milestone": [
        "degree", "graduated", "enrolled", "studied", "university",
        "college", "school", "diploma", "major",
    ],
    "named_artifact": [
        "playlist", "recipe", "book", "movie", "song", "show",
        "app", "podcast", "album",
    ],
    "appointment": [
        "appointment", "meeting", "scheduled", "booked", "reservation",
        "visit", "check-up",
    ],
    "location_linked_action": [
        "went to", "visited", "shopping at", "stopped by",
        "picked up from", "dropped off at",
    ],
}

_DURATION_RE = re.compile(r"\d+\s*(?:minute|hour|day|week|month|year|min|hr)", re.I)
_TIME_SIGNALS = re.compile(
    r"\d{1,2}:\d{2}|\d{1,2}\s*(?:am|pm)"
    r"|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)",
    re.I,
)
_DAY_MONTH_NAMES = re.compile(
    r"\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday"
    r"|January|February|March|April|May|June|July|August|September"
    r"|October|November|December)\b", re.I,
)
_LOCATION_RE = re.compile(r"[A-Z][a-z]{2,}")
_AMOUNT_RE = re.compile(r"\$\s*[\d,.]+|\d+\s*(?:dollars?|cents?)", re.I)
_AT_FROM_IN = re.compile(r"\b(?:at|from|in)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)")

_KNOWN_STORES = frozenset({
    "target", "walmart", "costco", "petco", "ikea", "trader",
    "safeway", "kroger", "publix", "aldi", "whole foods",
})


@dataclass(frozen=True, slots=True)
class EventFrame:
    """A multi-slot event record assembled from co-referent claims."""

    event_id: str
    event_type: str
    participants: tuple[str, ...]
    item: str | None
    location: str | None
    time_expr: str | None
    amount: str | None
    supporting_claim_ids: tuple[str, ...]
    source_turn_ids: tuple[str, ...]
    session_id: str
    confidence: float
    missing_slots: tuple[str, ...]

    def frame_text(self) -> str:
        """Textual representation for embedding."""
        parts = [self.event_type.replace("_", " ")]
        if self.item:
            parts.append(self.item)
        if self.location:
            parts.append(f"at {self.location}")
        if self.time_expr:
            parts.append(f"on {self.time_expr}")
        if self.amount:
            parts.append(f"for {self.amount}")
        return ": ".join(parts[:1]) + " " + " ".join(parts[1:])


def classify_event_type(claim_values: list[str]) -> str | None:
    """Classify event type from claim values. Returns None if no match."""
    combined = " ".join(claim_values).lower()
    best_type: str | None = None
    best_count = 0
    for etype, signals in _EVENT_SIGNALS.items():
        count = sum(1 for s in signals if s in combined)
        if count > best_count:
            best_count = count
            best_type = etype
    return best_type if best_count >= 1 else None


def _extract_location(values: list[str]) -> str | None:
    """Extract location from claim values."""
    for val in values:
        m = _AT_FROM_IN.search(val)
        if m:
            loc = m.group(1)
            if not _DAY_MONTH_NAMES.match(loc):
                return loc
        for word in val.lower().split():
            if word in _KNOWN_STORES:
                for m2 in _LOCATION_RE.finditer(val):
                    if m2.group().lower() == word or m2.group().lower().startswith(word):
                        return m2.group()
                return word.title()
    for val in values:
        for m in _LOCATION_RE.finditer(val):
            if not _DAY_MONTH_NAMES.match(m.group()):
                return m.group()
    return None


def _extract_time(values: list[str]) -> str | None:
    """Extract time expression from claim values."""
    for val in values:
        m = _TIME_SIGNALS.search(val)
        if m:
            return m.group()
        m2 = _DURATION_RE.search(val)
        if m2:
            return m2.group()
    for val in values:
        m = _DAY_MONTH_NAMES.search(val)
        if m:
            return m.group()
    return None


def _extract_amount(values: list[str]) -> str | None:
    """Extract amount/price from claim values."""
    for val in values:
        m = _AMOUNT_RE.search(val)
        if m:
            return m.group()
    return None


def _extract_item(values: list[str], event_type: str) -> str | None:
    """Extract the primary item/object from claim values."""
    if event_type == "purchase_redemption":
        for val in values:
            lower = val.lower()
            for trigger in ("coupon on ", "bought ", "purchased ", "redeemed "):
                idx = lower.find(trigger)
                if idx >= 0:
                    rest = val[idx + len(trigger):].strip()
                    rest = re.sub(r"\s*(at|from|for|last|on)\s.*$", "", rest, flags=re.I).strip()
                    if rest:
                        return rest
    elif event_type == "education_milestone":
        for val in values:
            lower = val.lower()
            for trigger in ("degree in ", "studied ", "major in ", "enrolled in "):
                idx = lower.find(trigger)
                if idx >= 0:
                    rest = val[idx + len(trigger):].strip()
                    rest = re.sub(r"\s*(at|from|for)\s.*$", "", rest, flags=re.I).strip()
                    if rest:
                        return rest
    elif event_type == "named_artifact":
        for val in values:
            m = re.search(r'"([^"]+)"', val)
            if m:
                return m.group(1)
            m2 = re.search(r"(?:named|called|titled)\s+(.+?)(?:\s+(?:by|from|on)\b|$)", val, re.I)
            if m2:
                return m2.group(1).strip()
    return None


def _get_turn_index(conn: sqlite3.Connection, turn_id: str) -> int | None:
    """Get the ordinal position of a turn within its session."""
    row = conn.execute(
        "SELECT session_id, ts FROM turns WHERE turn_id = ?", (turn_id,)
    ).fetchone()
    if row is None:
        return None
    all_turns = conn.execute(
        "SELECT turn_id FROM turns WHERE session_id = ? ORDER BY ts ASC",
        (row["session_id"],)
    ).fetchall()
    for i, r in enumerate(all_turns):
        if r["turn_id"] == turn_id:
            return i
    return None


def _content_tokens(value: str) -> set[str]:
    """Extract content tokens from a claim value (lowercase, no stopwords)."""
    _stops = frozenset({"a", "an", "the", "i", "my", "for", "on", "at", "in", "to", "of", "and", "was", "is"})
    return set(re.findall(r"[a-z0-9]+", value.lower())) - _stops


_SLOT_PROVIDER_TYPES = frozenset({"location_linked_action"})


def _turns_corefer(
    claims_a: list[Claim], claims_b: list[Claim],
    entity_keys: dict[str, str],
) -> bool:
    """Check if claims from two turns likely refer to the same event.

    Entity-key overlap alone is insufficient (personal_assistant pack uses
    "user" for everything). Merge when:
    1. Non-trivial entity_key overlap (keys other than common subjects), OR
    2. Content token overlap >= 2 between any pair of claims, OR
    3. Complementary types: one turn is a "slot provider" (location_linked_action)
       and the other is a core event type. This handles "shopping at Target"
       merging with "redeemed coupon on coffee creamer".

    Does NOT merge two turns that both independently classify as the same
    core event type without content overlap (e.g. "bought cat tower from Petco"
    vs "redeemed coupon" are both purchase_redemption but are different purchases).
    """
    trivial_keys = frozenset({"user", "patient", "i", ""})

    entities_a = {entity_keys.get(c.claim_id, "") for c in claims_a} - trivial_keys
    entities_b = {entity_keys.get(c.claim_id, "") for c in claims_b} - trivial_keys
    if entities_a and entities_b and (entities_a & entities_b):
        return True

    for ca in claims_a:
        tokens_a = _content_tokens(ca.value)
        for cb in claims_b:
            tokens_b = _content_tokens(cb.value)
            if len(tokens_a & tokens_b) >= 2:
                return True

    return False


def assemble_event_frames(
    conn: sqlite3.Connection, session_id: str,
) -> list[EventFrame]:
    """Group co-referent claims into event frames.

    Called after ingestion + supersession. Each claim participates in
    at most one event frame.
    """
    claims = list_active_claims(conn, session_id)
    if not claims:
        return []

    turn_to_claims: dict[str, list[Claim]] = defaultdict(list)
    for claim in claims:
        turn_to_claims[claim.source_turn_id].append(claim)

    turn_index_cache: dict[str, int] = {}
    for tid in turn_to_claims:
        idx = _get_turn_index(conn, tid)
        if idx is not None:
            turn_index_cache[tid] = idx

    entity_keys: dict[str, str] = {}
    rows = conn.execute(
        "SELECT claim_id, entity_key FROM claim_metadata WHERE claim_id IN "
        f"({','.join('?' for _ in claims)})",
        [c.claim_id for c in claims],
    ).fetchall()
    for r in rows:
        entity_keys[r["claim_id"]] = r["entity_key"]

    sorted_tids = sorted(turn_to_claims.keys(), key=lambda t: turn_index_cache.get(t, 0))

    # Pass 1: group turns with content overlap (strong co-reference).
    groups: list[set[str]] = []
    turn_id_to_group: dict[str, int] = {}
    for tid in sorted_tids:
        turn_id_to_group[tid] = len(groups)
        groups.append({tid})

    for i, tid_a in enumerate(sorted_tids):
        for j, tid_b in enumerate(sorted_tids):
            if i >= j:
                continue
            idx_a = turn_index_cache.get(tid_a, 0)
            idx_b = turn_index_cache.get(tid_b, 0)
            if abs(idx_a - idx_b) > 5:
                continue
            if not _turns_corefer(turn_to_claims[tid_a], turn_to_claims[tid_b], entity_keys):
                continue
            gi = turn_id_to_group[tid_a]
            gj = turn_id_to_group[tid_b]
            if gi != gj:
                groups[gi] = groups[gi] | groups[gj]
                for tid_in_gj in groups[gj]:
                    turn_id_to_group[tid_in_gj] = gi
                groups[gj] = set()

    # Pass 2: attach unmerged slot-provider turns to the nearest core event group.
    for tid in sorted_tids:
        gi = turn_id_to_group[tid]
        if len(groups[gi]) > 1:
            continue
        turn_type = classify_event_type([c.value for c in turn_to_claims[tid]])
        if turn_type not in _SLOT_PROVIDER_TYPES:
            continue
        tid_idx = turn_index_cache.get(tid, 0)
        best_gi: int | None = None
        best_dist = 999
        for other_tid in sorted_tids:
            if other_tid == tid:
                continue
            other_idx = turn_index_cache.get(other_tid, 0)
            dist = abs(tid_idx - other_idx)
            if dist > 3 or dist >= best_dist:
                continue
            other_type = classify_event_type([c.value for c in turn_to_claims[other_tid]])
            if other_type and other_type not in _SLOT_PROVIDER_TYPES:
                best_dist = dist
                best_gi = turn_id_to_group[other_tid]
        if best_gi is not None and best_gi != gi:
            groups[best_gi] = groups[best_gi] | groups[gi]
            for merge_tid in groups[gi]:
                turn_id_to_group[merge_tid] = best_gi
            groups[gi] = set()

    used_claim_ids: set[str] = set()
    frames: list[EventFrame] = []

    seen_group_ids: set[int] = set()
    for gi, group in enumerate(groups):
        if not group or gi in seen_group_ids:
            continue
        seen_group_ids.add(gi)

        group_claims: list[Claim] = []
        for tid in group:
            for c in turn_to_claims[tid]:
                if c.claim_id not in used_claim_ids:
                    group_claims.append(c)

        if len(group_claims) < 1:
            continue

        values = [c.value for c in group_claims]
        event_type = classify_event_type(values)
        if event_type is None:
            continue

        location = _extract_location(values)
        time_expr = _extract_time(values)
        amount = _extract_amount(values)
        item = _extract_item(values, event_type)

        missing: list[str] = []
        if location is None:
            missing.append("location")
        if time_expr is None:
            missing.append("time")
        if amount is None:
            missing.append("amount")
        if item is None:
            missing.append("item")

        claim_ids = tuple(c.claim_id for c in group_claims)
        source_turns = tuple(sorted({c.source_turn_id for c in group_claims}))
        participants = tuple(sorted({entity_keys.get(c.claim_id, c.subject) for c in group_claims}))
        confidence = min(c.confidence for c in group_claims)

        event_id = f"ev_{uuid.uuid4().hex[:12]}"
        frame = EventFrame(
            event_id=event_id,
            event_type=event_type,
            participants=participants,
            item=item,
            location=location,
            time_expr=time_expr,
            amount=amount,
            supporting_claim_ids=claim_ids,
            source_turn_ids=source_turns,
            session_id=session_id,
            confidence=confidence,
            missing_slots=tuple(missing),
        )
        frames.append(frame)
        used_claim_ids.update(claim_ids)

    _persist_frames(conn, frames)
    log.info(
        "event_frames.assembled",
        session_id=session_id,
        frame_count=len(frames),
        claim_count=len(used_claim_ids),
    )
    return frames


def _persist_frames(conn: sqlite3.Connection, frames: list[EventFrame]) -> None:
    """Write event frames to SQLite."""
    for frame in frames:
        conn.execute(
            "INSERT OR REPLACE INTO event_frames"
            " (event_id, event_type, participants, item, location,"
            "  time_expr, amount, supporting_claim_ids, source_turn_ids,"
            "  session_id, confidence, missing_slots)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                frame.event_id,
                frame.event_type,
                json.dumps(frame.participants),
                frame.item,
                frame.location,
                frame.time_expr,
                frame.amount,
                json.dumps(frame.supporting_claim_ids),
                json.dumps(frame.source_turn_ids),
                frame.session_id,
                frame.confidence,
                json.dumps(frame.missing_slots),
            ),
        )
        for claim_id in frame.supporting_claim_ids:
            slot_role = _infer_slot_role(conn, claim_id, frame)
            conn.execute(
                "INSERT OR REPLACE INTO event_frame_claims"
                " (event_id, claim_id, slot_role)"
                " VALUES (?, ?, ?)",
                (frame.event_id, claim_id, slot_role),
            )


def _infer_slot_role(
    conn: sqlite3.Connection, claim_id: str, frame: EventFrame,
) -> str:
    """Determine what slot role a claim fills within a frame."""
    row = conn.execute(
        "SELECT value FROM claims WHERE claim_id = ?", (claim_id,)
    ).fetchone()
    if row is None:
        return "context"
    val = row["value"]
    if frame.location and frame.location.lower() in val.lower():
        return "location"
    if frame.amount and frame.amount in val:
        return "amount"
    if frame.time_expr and frame.time_expr.lower() in val.lower():
        return "time"
    if frame.item and frame.item.lower() in val.lower():
        return "item"
    return "context"


def list_event_frames(
    conn: sqlite3.Connection, session_id: str,
) -> list[EventFrame]:
    """Load all event frames for a session."""
    rows = conn.execute(
        "SELECT * FROM event_frames WHERE session_id = ?", (session_id,)
    ).fetchall()
    return [_row_to_frame(r) for r in rows]


def _row_to_frame(row: sqlite3.Row) -> EventFrame:
    return EventFrame(
        event_id=row["event_id"],
        event_type=row["event_type"],
        participants=tuple(json.loads(row["participants"])),
        item=row["item"],
        location=row["location"],
        time_expr=row["time_expr"],
        amount=row["amount"],
        supporting_claim_ids=tuple(json.loads(row["supporting_claim_ids"])),
        source_turn_ids=tuple(json.loads(row["source_turn_ids"])),
        session_id=row["session_id"],
        confidence=row["confidence"],
        missing_slots=tuple(json.loads(row["missing_slots"])),
    )
