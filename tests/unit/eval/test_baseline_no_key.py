"""Test that the ACI-Bench and LongMemEval baselines refuse to silently
return gold-as-prediction when ANTHROPIC_API_KEY is unset.

Prior behaviour: `predict_note` / `predict_answer` returned `enc.gold_note` /
`q.answer` verbatim with a `[STUB]` flag. That meant any unattended run
(CI, sweep, scratch) without the key would score the baseline at ~1.0 and
publish meaningless numbers. Worker 1 (fix/benchmark-integrity) replaced
the stub with a `RuntimeError` so the failure is loud and immediate.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest


@dataclass(frozen=True)
class _StubACIEncounter:
    encounter_id: str = "D2N001"
    dialogue: tuple[object, ...] = ()
    gold_note: str = "GOLD NOTE — must NOT leak into prediction"


@dataclass(frozen=True)
class _StubLMEQuestion:
    question_id: str = "q_001"
    question_type: str = "single-session-user"
    question: str = "what is X?"
    answer: str = "GOLD ANSWER — must NOT leak into prediction"
    haystack_sessions: tuple[object, ...] = ()
    haystack_dates: tuple[object, ...] = ()


def test_aci_baseline_raises_runtime_error_without_anthropic_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from eval.aci_bench.baseline import predict_note

    enc = _StubACIEncounter()
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        predict_note(enc)  # type: ignore[arg-type]


def test_lme_baseline_raises_runtime_error_without_anthropic_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from eval.longmemeval.baseline import predict_answer

    q = _StubLMEQuestion()
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        predict_answer(q)  # type: ignore[arg-type]


def test_aci_baseline_message_explains_why(monkeypatch: pytest.MonkeyPatch) -> None:
    """Error message should explain that returning gold inflates the score —
    so an operator who hits this knows it's intentional, not a bug."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from eval.aci_bench.baseline import predict_note

    with pytest.raises(RuntimeError, match="inflate score"):
        predict_note(_StubACIEncounter())  # type: ignore[arg-type]


def test_lme_baseline_message_explains_why(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from eval.longmemeval.baseline import predict_answer

    with pytest.raises(RuntimeError, match="inflate score"):
        predict_answer(_StubLMEQuestion())  # type: ignore[arg-type]
