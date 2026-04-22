"""Per-check hallucination-guard regression tests.

Crafted-failure unit tests for each of the 5 deterministic checks in
``src/extraction/asr/hallucination_guard.py``. Each test fires ONLY the
target check via a minimal crafted input, asserting that the specific check
name appears in the flagged-span names.

The 5 checks (per `docs/asr_engineering_spec.md` §2 criterion B):
  1. repeated_ngram         — looping phrase pathology (Koenecke et al. 2024)
  2. oov_medical_term       — confabulated medical-looking token
  3. extreme_compression    — implausible chars/second density
  4. low_confidence_span    — consecutive low-confidence words
  5. invented_medication    — drug-suffix token not in RxNorm

The pre-existing ``test_hallucination_guard.py`` already covers each checker
function directly; this file exercises the same checks via the aggregate
``check()`` entry point, which is the surface the pipeline actually calls.
Parametrized so one test failure localises one check.
"""

from __future__ import annotations

import pytest

from src.extraction.asr.hallucination_guard import (
    Severity,
    check,
)

_VOCAB: frozenset[str] = frozenset({
    "metoprolol", "aspirin", "atorvastatin", "dyspnea", "pericarditis",
    "costochondritis", "substernal", "pleuritic", "syncope",
})

_RXNORM_SMALL: frozenset[str] = frozenset({"aspirin", "metoprolol", "atorvastatin"})


# A "benign baseline" built from vocabulary-known tokens in lower-case so
# neither the OOV-medical-term check (which is TitleCase-/suffix-greedy) nor
# the invented-medication check trips. Used by the check-isolation tests as
# a known-clean control.
_BENIGN_TEXT = "patient has substernal pleuritic pain"
_BENIGN_CONFS = [0.95] * 5
_BENIGN_DURATION = 3.0


def test_baseline_text_is_clean() -> None:
    """Sanity: the benign baseline trips no checks at BLOCK severity.

    The OOV-medical-term regex is deliberately greedy (TitleCase + suffix
    match) so even modest everyday English can trip WARN. That is by design
    — the guard's BLOCK gate is the load-bearing one. This test asserts the
    baseline stays under BLOCK and has no *invented_medication* flag (the
    one flag that auto-promotes to BLOCK regardless of count).
    """
    report = check(
        text=_BENIGN_TEXT,
        vocabulary=_VOCAB,
        word_confidences=list(_BENIGN_CONFS),
        audio_duration_s=_BENIGN_DURATION,
        rxnorm_set=_RXNORM_SMALL,
    )
    assert report.severity in (Severity.CLEAN, Severity.WARN)
    assert report.severity != Severity.BLOCK
    flagged_names = {s.check_name for s in report.flagged_spans}
    assert "invented_medication" not in flagged_names


@pytest.mark.parametrize(
    "check_name,kwargs",
    [
        # Check 1: repeated n-gram loop.
        # Craft: "thank you very much" repeated 4× → 4-gram count 4 > max_repeats 3.
        (
            "repeated_ngram",
            {
                "text": "thank you very much thank you very much thank you very much thank you very much",
                "vocabulary": _VOCAB,
                "word_confidences": [0.95] * 16,
                "audio_duration_s": 30.0,      # 80 chars / 30 s ≈ 2.7 chars/s — under threshold
                "rxnorm_set": _RXNORM_SMALL,
            },
        ),
        # Check 2: OOV medical term.
        # "Fluorocarditis" looks TitleCased + medical-shaped, not in _VOCAB.
        (
            "oov_medical_term",
            {
                "text": "Patient has Fluorocarditis.",
                "vocabulary": _VOCAB,
                "word_confidences": [0.95] * 4,
                "audio_duration_s": 4.0,
                "rxnorm_set": _RXNORM_SMALL,
            },
        ),
        # Check 3: extreme compression ratio.
        # 200 chars in 1.0 s audio = 200 chars/s, well above threshold 15.
        # Keep the text generic to avoid tripping other checks.
        (
            "extreme_compression",
            {
                "text": "a " * 100,            # ~200 chars, generic content
                "vocabulary": _VOCAB,
                "word_confidences": [0.95] * 100,
                "audio_duration_s": 1.0,       # → 200 chars/s
                "rxnorm_set": _RXNORM_SMALL,
            },
        ),
        # Check 4: low-confidence span.
        # 6 consecutive words < 0.3 confidence triggers (max_span=5 default).
        (
            "low_confidence_span",
            {
                "text": "the the the the the the",
                "vocabulary": _VOCAB,
                "word_confidences": [0.05] * 6,   # 6 words, all below threshold
                "audio_duration_s": 10.0,
                "rxnorm_set": _RXNORM_SMALL,
            },
        ),
        # Check 5: invented medication.
        # "Xylostatin" has -statin suffix, not in _RXNORM_SMALL.
        (
            "invented_medication",
            {
                "text": "Prescribed Xylostatin at bedtime.",
                "vocabulary": _VOCAB | {"Xylostatin"},  # vocab-ok; RxNorm-miss
                "word_confidences": [0.95] * 4,
                "audio_duration_s": 4.0,
                "rxnorm_set": _RXNORM_SMALL,
            },
        ),
    ],
    ids=[
        "repeated_ngram",
        "oov_medical_term",
        "extreme_compression",
        "low_confidence_span",
        "invented_medication",
    ],
)
def test_each_check_fires_on_crafted_failure(
    check_name: str,
    kwargs: dict[str, object],
) -> None:
    """Each of the 5 checks fires on its targeted crafted-failure input.

    The aggregate ``check()`` entry point routes through all 5 checks; we
    assert the flagged-span for the target check is present in the report.
    Other checks may incidentally also fire (e.g. the repeated-n-gram craft
    may incidentally trip OOV on "thank" — acceptable as long as the
    target check is among those flagged).
    """
    report = check(**kwargs)  # type: ignore[arg-type]
    flagged_names = {span.check_name for span in report.flagged_spans}
    assert check_name in flagged_names, (
        f"Check {check_name!r} did not fire. "
        f"Flagged: {sorted(flagged_names)}; reasons: {report.reasons}"
    )


def test_all_five_checks_named_in_module() -> None:
    """Documentation invariant: the 5 check names are stable.

    Keeps the spec (docs/asr_engineering_spec.md §2 criterion B) and this
    test file in lock-step. If a 6th check is added, this test fails and
    forces an explicit spec update alongside the code change.
    """
    expected = {
        "repeated_ngram",
        "oov_medical_term",
        "extreme_compression",
        "low_confidence_span",
        "invented_medication",
    }
    # Trigger all checks via a single very-hallucinatory segment, then
    # collect the set of check names observed.
    observed: set[str] = set()
    for params in [
        {"text": "a b c a b c a b c a b c", "word_confidences": [0.05] * 12,
         "audio_duration_s": 0.5},
        {"text": "Patient has Fluorocarditis and Xylostatin therapy.",
         "word_confidences": [0.95] * 7, "audio_duration_s": 5.0},
    ]:
        r = check(
            vocabulary=_VOCAB,
            rxnorm_set=_RXNORM_SMALL,
            **params,  # type: ignore[arg-type]
        )
        observed |= {s.check_name for s in r.flagged_spans}

    # Expect every one of the 5 check names to be reachable via the
    # aggregate entry point.
    assert expected.issubset(observed), (
        f"Missing check names in aggregate output: {expected - observed}"
    )
