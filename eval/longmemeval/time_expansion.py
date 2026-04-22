"""Temporal anchor extraction for LongMemEval questions.

A LongMemEval `temporal-reasoning` question often requires filtering the
haystack to a specific time window — "what did I decide last month" means
the retrieval head should preferentially surface claims whose `created_ts`
falls roughly one month ago.

We use `dateparser` (BSD-3-Clause) because it handles relative references
("last week", "two months ago") robustly — the `arrow` library only
handles absolute timestamps well. Decision logged in `reasons.md`
("dateparser over arrow").

Soft boundaries (logged in `reasons.md`):
- **±7 days** for relative references (dateparser's output is approximate).
- **±1 day** for explicit calendar dates (accommodates timezone drift).

API is intentionally conservative: parse failures return `None` so the
caller treats it as "no filter" — we'd rather over-retrieve and let the
reader filter than silently drop facts.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import structlog

from src.substrate.retrieval import DateRange

log = structlog.get_logger(__name__)


# Soft boundary defaults (see module docstring / reasons.md).
RELATIVE_WINDOW_DAYS = 7
EXPLICIT_WINDOW_DAYS = 1


# Keywords that strongly suggest a temporal anchor is present. Not exhaustive,
# but enough to trigger time expansion outside of the `temporal-reasoning`
# question type. Kept lowercase — we match against the lowered question text.
_RELATIVE_HINTS = (
    "yesterday",
    "today",
    "tomorrow",
    "last week",
    "last month",
    "last year",
    "next week",
    "next month",
    "next year",
    "ago",
    "recently",
    "this morning",
    "this week",
    "this month",
    "this year",
)

_EXPLICIT_HINTS = (
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
)


def _has_temporal_hint(question: str) -> bool:
    q = question.lower()
    for h in _RELATIVE_HINTS:
        if h in q:
            return True
    for h in _EXPLICIT_HINTS:
        if h in q:
            return True
    # Heuristic: a 4-digit year or a pattern like "3/15" or "2024-01-15" is a strong hint.
    # We don't need perfect recall here — parse failures fall back to None anyway.
    if any(tok.isdigit() and len(tok) == 4 and tok.startswith("20") for tok in q.split()):
        return True
    return False


def _is_relative_phrase(question: str) -> bool:
    q = question.lower()
    return any(h in q for h in _RELATIVE_HINTS)


def _dateparser_parse(text: str) -> datetime | None:
    """Extract the first date-like span from `text` via dateparser.search_dates.

    `dateparser.parse` requires the whole string to be a date; LongMemEval
    questions are full English sentences so we use `search_dates`, which
    returns a list of `(matched_phrase, datetime)` tuples. Returns the
    first match's datetime normalised to UTC, or `None` on no match / any
    exception (dateparser occasionally throws on odd input).
    """
    try:
        from dateparser.search import search_dates  # type: ignore[import-not-found]
    except ImportError:
        log.warning("time_expansion.dateparser_not_installed")
        return None

    settings: dict[str, Any] = {
        "RETURN_AS_TIMEZONE_AWARE": True,
        "TIMEZONE": "UTC",
        "TO_TIMEZONE": "UTC",
        # LongMemEval haystacks are historical transcripts — "the meeting"
        # refers to a past event, never a future one.
        "PREFER_DATES_FROM": "past",
    }
    try:
        found = search_dates(text, settings=settings)
    except Exception as exc:  # noqa: BLE001 — search_dates can throw on odd input
        log.warning("time_expansion.parse_error", text=text[:80], error=repr(exc)[:80])
        found = None

    parsed: datetime | None = None
    if found:
        parsed = found[0][1]
    else:
        # Fallback: search_dates sometimes fails on full English sentences
        # ("What did I decide last week?") even when a hint is clearly present.
        # Try each RELATIVE_HINT we find in the text directly via dateparser.parse.
        try:
            import dateparser  # type: ignore[import-not-found]
        except ImportError:
            return None
        lowered = text.lower()
        for hint in _RELATIVE_HINTS:
            if hint in lowered:
                try:
                    parsed = dateparser.parse(hint, settings=settings)
                except Exception:  # noqa: BLE001
                    parsed = None
                if parsed is not None:
                    break

    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def extract_time_window(
    question: str,
    question_type: str | None = None,
) -> DateRange | None:
    """Return a soft-bounded `DateRange` anchored at the question's temporal reference.

    Returns `None` when:
    - the question carries no obvious temporal hint AND `question_type` is not
      `"temporal-reasoning"`, or
    - `dateparser` can't parse any anchor from the question text.

    Soft boundary width follows the module constants `RELATIVE_WINDOW_DAYS`
    (±7) and `EXPLICIT_WINDOW_DAYS` (±1). Same-side windows are half-open
    (see `DateRange.contains`).
    """
    if not question:
        return None

    is_temporal_type = (question_type or "").lower() == "temporal-reasoning"
    has_hint = _has_temporal_hint(question)
    if not is_temporal_type and not has_hint:
        return None

    anchor = _dateparser_parse(question)
    if anchor is None:
        log.debug("time_expansion.no_anchor", question=question[:80])
        return None

    relative = _is_relative_phrase(question)
    width_days = RELATIVE_WINDOW_DAYS if relative else EXPLICIT_WINDOW_DAYS
    delta = timedelta(days=width_days)
    return DateRange(start=anchor - delta, end=anchor + delta)


__all__ = [
    "EXPLICIT_WINDOW_DAYS",
    "RELATIVE_WINDOW_DAYS",
    "extract_time_window",
]
