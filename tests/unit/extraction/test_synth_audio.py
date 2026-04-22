"""Script-level tests for the synthetic chest-pain dialogue.

The actual WAV render requires a TTS backend (SAPI5 / eSpeak) and is exercised
in the benchmark harness, not unit tests. Here we just assert the script is
well-formed and gold-transcript derivation is deterministic.
"""

from __future__ import annotations

from src.extraction.asr.synth_audio import DEMO_SCRIPT, script_as_gold_transcript


def test_demo_script_non_empty_and_has_both_speakers() -> None:
    assert len(DEMO_SCRIPT) >= 5
    speakers = {u.speaker for u in DEMO_SCRIPT}
    assert speakers == {"patient", "physician"}


def test_demo_script_contains_benchmark_anchors() -> None:
    """Anchors the benchmark uses: a drug name, a descriptor, a supersession moment."""
    joined = script_as_gold_transcript().lower()
    assert "metoprolol" in joined  # drug — tests vocab bias
    assert "radiates" in joined  # AHA/ACC descriptor
    assert "actually" in joined  # supersession marker — Pass 1 / Pass 2 fixture


def test_gold_transcript_is_deterministic() -> None:
    first = script_as_gold_transcript()
    second = script_as_gold_transcript()
    assert first == second
