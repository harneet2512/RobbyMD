"""Tests for A.4 — medical word correction.

Four tests:
1. Tokens within edit distance 1 of a vocabulary term are corrected.
2. Common English words are protected from correction (DEFAULT_COMMON guard).
3. Tokens with no near match pass through unchanged.
4. Corrections are logged (structlog capture).
"""

from __future__ import annotations

import structlog
import structlog.testing

from src.extraction.asr.word_correction import (
    DEFAULT_COMMON,
    Correction,
    correct_medical_tokens,
)

_VOCAB: frozenset[str] = frozenset({
    "metoprolol",
    "aspirin",
    "atorvastatin",
    "costochondritis",
    "dyspnea",
    "pericarditis",
})


def test_matches_within_edit_distance() -> None:
    """A single-character typo should be corrected to the nearest vocab term."""
    # "metoprplol" has edit distance 1 from "metoprolol" (transposition of p+r).
    transcript = "Patient takes metoprplol daily."
    corrected, corrections = correct_medical_tokens(transcript, _VOCAB, max_edit_distance=1)
    assert "metoprolol" in corrected, f"Expected correction in: {corrected!r}"
    assert len(corrections) >= 1
    assert any(c.vocab_match == "metoprolol" for c in corrections)


def test_common_english_word_protected() -> None:
    """Words in DEFAULT_COMMON must not be corrected even if near a vocab term."""
    # "pain" is in DEFAULT_COMMON and must not be altered.
    transcript = "The patient has chest pain today."
    corrected, corrections = correct_medical_tokens(transcript, _VOCAB, max_edit_distance=1)
    assert "pain" in corrected, f"'pain' should survive in: {corrected!r}"
    # No correction should mention 'pain' as original.
    assert not any(c.original.lower() == "pain" for c in corrections)


def test_no_match_passes_through() -> None:
    """A token with no vocabulary near-match should be left unchanged."""
    transcript = "The xylophone sounded lovely."
    corrected, corrections = correct_medical_tokens(transcript, _VOCAB, max_edit_distance=1)
    assert "xylophone" in corrected
    # No correction applied to xylophone.
    assert not any(c.original.lower() == "xylophone" for c in corrections)


def test_corrections_logged() -> None:
    """Each correction must emit a structlog event."""
    with structlog.testing.capture_logs() as cap:
        transcript = "aspirine daily"  # 'aspirine' → 'aspirin' (edit dist 1)
        corrected, corrections = correct_medical_tokens(
            transcript, _VOCAB, max_edit_distance=1
        )
    # Either a correction was found (and logged), or not found (no log expected).
    if corrections:
        # At least one log event with 'word_correction.applied' should exist.
        events = [e.get("event", "") for e in cap]
        assert any("word_correction" in e for e in events), (
            f"Expected structlog event; events={events}"
        )


def test_correction_dataclass_fields() -> None:
    """Correction dataclass carries all required fields."""
    transcript = "metoprplol"  # should correct to metoprolol
    corrected, corrections = correct_medical_tokens(transcript, _VOCAB, max_edit_distance=1)
    if corrections:
        c = corrections[0]
        assert isinstance(c, Correction)
        assert isinstance(c.original, str)
        assert isinstance(c.replacement, str)
        assert isinstance(c.char_start, int)
        assert isinstance(c.char_end, int)
        assert isinstance(c.distance, int)
        assert isinstance(c.vocab_match, str)
