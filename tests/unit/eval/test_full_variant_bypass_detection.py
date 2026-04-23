"""Confirm that `--variant full` raises AssertionError today, until the
real substrate is wired.

The orchestrator's CRITICAL CONSTRAINT: "Do not allow benchmark runs where
full benchmark path == baseline pipeline. If detected, stop execution and
report." Today, both `eval.aci_bench.full.FullRunner.predict_note` and
`eval.longmemeval.full.FullRunner.predict_answer` create a `SubstrateStub`,
write turns to it, and then fall back to `baseline_predict` — the substrate
is bypassed. Worker 1 added an `assert "[SUBSTRATE STUB]" not in raw_response`
guard so this bypass is loud, not silent.

When Worker 3 lands the real substrate path (event tuples + shared backend
in `eval/_substrate_backend.py`), the `[SUBSTRATE STUB]` sentinel goes away
and the assertion passes silently. At that point these tests should be
inverted (assert the assertion does NOT fire) — that change rides with
Worker 3's PR.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pytest


@dataclass(frozen=True)
class _StubACIEncounter:
    encounter_id: str = "D2N001"
    dialogue: tuple[object, ...] = ()
    gold_note: str = "gold"


@dataclass(frozen=True)
class _StubLMEQuestion:
    question_id: str = "q_001"
    question_type: str = "single-session-user"
    question: str = "?"
    answer: str = "a"
    haystack_sessions: tuple[object, ...] = field(default_factory=tuple)
    haystack_dates: tuple[object, ...] = field(default_factory=tuple)


def test_aci_bench_full_variant_assertion_fires_on_bypass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bypass-detection assertion fires today because the substrate is
    not wired into eval/aci_bench/full.py. Worker 3 will fix this."""
    # The assertion fires AFTER baseline_predict is called. We need that call
    # to succeed (and be visible — we monkeypatch it to return a valid prediction).
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-actually-called")
    from eval.aci_bench import baseline as aci_baseline
    from eval.aci_bench.full import FullRunner

    monkeypatch.setattr(
        aci_baseline,
        "predict_note",
        lambda enc: aci_baseline.ACINotePrediction(
            encounter_id=enc.encounter_id,
            predicted_note="dummy SOAP note",
            raw_response="dummy response",
        ),
    )
    # The full module imported predict_note at module load — patch its alias too.
    from eval.aci_bench import full as aci_full

    monkeypatch.setattr(aci_full, "baseline_predict", aci_baseline.predict_note)

    runner = FullRunner()
    with pytest.raises(AssertionError, match="Bypass detected"):
        runner.predict_note(_StubACIEncounter())  # type: ignore[arg-type]


def test_lme_full_variant_assertion_fires_on_bypass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bypass-detection assertion fires today because the substrate is
    not wired into eval/longmemeval/full.py. Worker 3 will fix this."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-actually-called")
    from eval.longmemeval import baseline as lme_baseline
    from eval.longmemeval.full import FullRunner

    monkeypatch.setattr(
        lme_baseline,
        "predict_answer",
        lambda q: lme_baseline.LongMemEvalPrediction(
            question_id=q.question_id,
            question_type=q.question_type,
            predicted_answer="dummy answer",
            raw_response="dummy response",
        ),
    )
    from eval.longmemeval import full as lme_full

    monkeypatch.setattr(lme_full, "baseline_predict", lme_baseline.predict_answer)

    runner = FullRunner()
    with pytest.raises(AssertionError, match="Bypass detected"):
        runner.predict_answer(_StubLMEQuestion())  # type: ignore[arg-type]
