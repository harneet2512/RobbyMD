"""Admission filter + event bus unit tests."""
from __future__ import annotations

import pytest

from src.substrate.admission import MIN_CONTENT_WORDS, admit
from src.substrate.event_bus import CLAIM_CREATED, TURN_ADDED, EventBus

# ----------------------------------------------------------- admission --- #


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "[silence]",
        "[noise]",
        "uh",
        "um, uh, right",
        "ok ok ok",
    ],
)
def test_admission_rejects_noise(text: str) -> None:
    result = admit(text)
    assert not result.admitted, f"expected {text!r} rejected"
    assert result.reason, "reason must be populated for telemetry"


@pytest.mark.parametrize(
    "text",
    [
        "chest pain started three days ago",
        "patient reports radiation to the left arm",
        "sharp stabbing quality",
    ],
)
def test_admission_accepts_clinical_text(text: str) -> None:
    result = admit(text)
    assert result.admitted, f"expected {text!r} admitted"
    assert result.reason == "admitted"


def test_admission_short_but_content_admits() -> None:
    # MIN_CONTENT_WORDS = 3; at exactly 3 we admit.
    assert MIN_CONTENT_WORDS == 3
    assert admit("stabbing chest pain").admitted


def test_admission_two_content_words_rejects() -> None:
    assert not admit("chest pain").admitted


# ----------------------------------------------------------- event bus --- #


def test_event_bus_publish_to_subscribers() -> None:
    bus = EventBus()
    got: list[dict[str, object]] = []

    def handler(payload: dict[str, object]) -> None:
        got.append(payload)

    bus.subscribe(TURN_ADDED, handler)
    bus.publish(TURN_ADDED, {"turn_id": "t1"})
    bus.publish(TURN_ADDED, {"turn_id": "t2"})
    assert got == [{"turn_id": "t1"}, {"turn_id": "t2"}]


def test_event_bus_no_subscribers_is_noop() -> None:
    bus = EventBus()
    # No subscribers — must not raise.
    bus.publish(CLAIM_CREATED, {"claim_id": "c1"})


def test_event_bus_unsubscribe() -> None:
    bus = EventBus()
    got: list[int] = []

    def handler(_payload: dict[str, object]) -> None:
        got.append(1)

    bus.subscribe(TURN_ADDED, handler)
    bus.publish(TURN_ADDED, {})
    bus.unsubscribe(TURN_ADDED, handler)
    bus.publish(TURN_ADDED, {})
    assert got == [1]


def test_event_bus_shields_failing_subscriber() -> None:
    """A raising subscriber must not prevent siblings from receiving the event."""
    bus = EventBus()

    def bad(_payload: dict[str, object]) -> None:
        raise RuntimeError("boom")

    got: list[int] = []

    def good(_payload: dict[str, object]) -> None:
        got.append(1)

    bus.subscribe(TURN_ADDED, bad)
    bus.subscribe(TURN_ADDED, good)
    # Publish must not raise; good subscriber must still run.
    bus.publish(TURN_ADDED, {})
    assert got == [1]


def test_event_bus_subscriber_count() -> None:
    bus = EventBus()

    def a(_p: dict[str, object]) -> None: ...
    def b(_p: dict[str, object]) -> None: ...

    bus.subscribe(TURN_ADDED, a)
    bus.subscribe(TURN_ADDED, b)
    assert bus.subscriber_count(TURN_ADDED) == 2
    assert bus.subscriber_count(CLAIM_CREATED) == 0
