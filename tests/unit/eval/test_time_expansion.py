"""Unit tests for LongMemEval time-expansion.

No external services. Depends on `dateparser` (runtime dep, BSD-3-Clause).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from eval.longmemeval.time_expansion import (
    EXPLICIT_WINDOW_DAYS,
    RELATIVE_WINDOW_DAYS,
    extract_time_window,
)


class TestTemporalReasoningType:
    def test_relative_last_week_returns_window(self) -> None:
        out = extract_time_window("What did I decide last week?", "temporal-reasoning")
        assert out is not None
        assert out.start is not None
        assert out.end is not None
        # Width should be ~ 2 * RELATIVE_WINDOW_DAYS.
        width = (out.end - out.start).total_seconds() / 86400
        assert width == pytest.approx(2 * RELATIVE_WINDOW_DAYS, abs=0.01)

    def test_relative_two_months_ago(self) -> None:
        out = extract_time_window(
            "What was I working on two months ago?", "temporal-reasoning"
        )
        assert out is not None
        # Anchor should be roughly ~60 days in the past.
        mid = out.start + (out.end - out.start) / 2  # type: ignore[operator]
        now = datetime.now(tz=timezone.utc)
        days_ago = (now - mid).total_seconds() / 86400
        assert 30 <= days_ago <= 120  # loose — dateparser's approximation


class TestExplicitDate:
    def test_explicit_date_narrower_window(self) -> None:
        out = extract_time_window(
            "What did I say on January 15, 2024?", "temporal-reasoning"
        )
        assert out is not None
        width = (out.end - out.start).total_seconds() / 86400  # type: ignore[operator]
        assert width == pytest.approx(2 * EXPLICIT_WINDOW_DAYS, abs=0.01)

    def test_explicit_date_anchor(self) -> None:
        out = extract_time_window(
            "What did I say on March 10, 2024?", "temporal-reasoning"
        )
        assert out is not None
        mid = out.start + (out.end - out.start) / 2  # type: ignore[operator]
        # Should be anchored near 2024-03-10 within ±30 days.
        target = datetime(2024, 3, 10, tzinfo=timezone.utc)
        diff_days = abs((mid - target).total_seconds()) / 86400
        assert diff_days < 30


class TestNoHintNoWindow:
    def test_non_temporal_non_hinted_returns_none(self) -> None:
        out = extract_time_window(
            "What is the user's favourite colour?", "single-session-preference"
        )
        assert out is None

    def test_temporal_type_still_parses_without_hint(self) -> None:
        # Even without hint-words, temporal-reasoning type triggers parse attempt.
        # dateparser on a non-date sentence returns None → our function returns None.
        out = extract_time_window(
            "What is the user's favourite colour?", "temporal-reasoning"
        )
        # dateparser returns None on bare prose → None overall.
        assert out is None


class TestParseFailureReturnsNone:
    def test_nonsense_returns_none(self) -> None:
        out = extract_time_window("blerg blerg blerg", "temporal-reasoning")
        assert out is None

    def test_empty_question_returns_none(self) -> None:
        assert extract_time_window("", "temporal-reasoning") is None


class TestOrdinalHandling:
    def test_ordinal_meeting_hint(self) -> None:
        # Ordinals like "third meeting" are not parseable as dates — return None
        # per the docstring: better to over-retrieve than silently filter.
        out = extract_time_window(
            "What happened in the third meeting?", "temporal-reasoning"
        )
        # dateparser cannot resolve "third meeting" to a date → None.
        assert out is None


class TestHintWithoutType:
    def test_hint_alone_triggers_parse(self) -> None:
        # No question_type passed, but a strong temporal hint: should still parse.
        out = extract_time_window("What happened yesterday?")
        assert out is not None
        assert out.start is not None
        assert out.end is not None
