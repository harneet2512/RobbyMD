"""Tests for Part B — FreeFlow-pattern two-speaker medical transcript cleanup.

Six tests using a mock OpenAI client (no real API calls):
1. Doctor segment: filler words are removed.
2. Doctor segment: medical term correction (met oh pro lol → metoprolol).
3. Patient segment: lay-language tagging ([likely: ...] tags preserved).
4. Backtracking resolution in a doctor segment.
5. Provenance: original_text is preserved on all paths.
6. "EMPTY" response from LLM → empty CleanedSegment, original_text still kept.

All tests patch openai.OpenAI AND the OPENAI_API_KEY env var so no real calls
are made and the key-presence guard in _call_llm passes.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from src.extraction.asr.transcript_cleanup import (
    CleanedSegment,
    DiarisedSegment,
    TranscriptCleaner,
)

_VOCAB: set[str] = {"metoprolol", "aspirin", "atorvastatin", "dyspnea", "costochondritis"}


def _make_mock_openai_client(response_text: str) -> MagicMock:
    """Return a mock openai.OpenAI client that returns `response_text`."""
    mock_message = MagicMock()
    mock_message.content = response_text
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


def _make_cleaner(response_text: str) -> tuple[TranscriptCleaner, MagicMock]:
    cleaner = TranscriptCleaner(
        medical_vocabulary=_VOCAB,
        cleanup_model="gpt-4o-mini",
    )
    mock_client = _make_mock_openai_client(response_text)
    return cleaner, mock_client


def _patches(mock_client: MagicMock) -> list:
    """Return context managers that mock both openai.OpenAI and the API key env var."""
    return [
        patch("openai.OpenAI", return_value=mock_client),
        patch.dict(os.environ, {"OPENAI_API_KEY": "test-key-mock"}, clear=False),
    ]


def test_doctor_filler_removal() -> None:
    """Doctor segment: 'um' and 'uh' fillers should be removed in cleaned output."""
    cleaned_response = "Can you describe the chest pain?"
    cleaner, mock_client = _make_cleaner(cleaned_response)

    seg = DiarisedSegment(
        speaker_role="doctor",
        raw_text="Um, can you uh describe the chest pain?",
    )

    with patch("openai.OpenAI", return_value=mock_client), \
         patch.dict(os.environ, {"OPENAI_API_KEY": "test-key-mock"}, clear=False):
        result = cleaner.clean(seg)

    assert isinstance(result, CleanedSegment)
    assert result.cleaned_text == cleaned_response
    assert result.speaker_role == "doctor"


def test_doctor_medical_term_correction() -> None:
    """Doctor segment: 'met oh pro lol' should be corrected to 'metoprolol'."""
    cleaned_response = "The patient is on metoprolol 50 mg daily."
    cleaner, mock_client = _make_cleaner(cleaned_response)

    seg = DiarisedSegment(
        speaker_role="doctor",
        raw_text="The patient is on met oh pro lol 50 mg daily.",
    )

    with patch("openai.OpenAI", return_value=mock_client), \
         patch.dict(os.environ, {"OPENAI_API_KEY": "test-key-mock"}, clear=False):
        result = cleaner.clean(seg)

    assert "metoprolol" in result.cleaned_text


def test_patient_lay_language_tagging() -> None:
    """Patient segment: lay terms should receive [likely: ...] tags."""
    cleaned_response = (
        "It feels like someone's sitting on my chest [likely: pressure-type chest pain]."
    )
    cleaner, mock_client = _make_cleaner(cleaned_response)

    seg = DiarisedSegment(
        speaker_role="patient",
        raw_text="It feels like someone's sitting on my chest.",
    )

    with patch("openai.OpenAI", return_value=mock_client), \
         patch.dict(os.environ, {"OPENAI_API_KEY": "test-key-mock"}, clear=False):
        result = cleaner.clean(seg)

    assert "[likely:" in result.cleaned_text


def test_backtracking_resolution() -> None:
    """Doctor segment: self-correction should be resolved to the final statement."""
    cleaned_response = "The pain started about 30 minutes ago."
    cleaner, mock_client = _make_cleaner(cleaned_response)

    seg = DiarisedSegment(
        speaker_role="doctor",
        raw_text="The pain started — actually it was about 30 minutes ago, not an hour.",
    )

    with patch("openai.OpenAI", return_value=mock_client), \
         patch.dict(os.environ, {"OPENAI_API_KEY": "test-key-mock"}, clear=False):
        result = cleaner.clean(seg)

    # Cleaned text should NOT contain the backtrack dash.
    assert "—" not in result.cleaned_text or "actually" not in result.cleaned_text


def test_original_text_preserved_on_all_paths() -> None:
    """original_text must equal the raw segment text regardless of LLM output."""
    raw = "Um, the patient uh takes aspirine daily."
    cleaned_response = "The patient takes aspirin daily."
    cleaner, mock_client = _make_cleaner(cleaned_response)

    seg = DiarisedSegment(speaker_role="doctor", raw_text=raw)

    with patch("openai.OpenAI", return_value=mock_client), \
         patch.dict(os.environ, {"OPENAI_API_KEY": "test-key-mock"}, clear=False):
        result = cleaner.clean(seg)

    assert result.original_text == raw, (
        f"original_text must match raw input; got {result.original_text!r}"
    )


def test_empty_response_handling() -> None:
    """LLM returning 'EMPTY' → empty cleaned_text, original_text still preserved."""
    raw = "um, uh, er"
    cleaner, mock_client = _make_cleaner("EMPTY")

    seg = DiarisedSegment(speaker_role="doctor", raw_text=raw)

    with patch("openai.OpenAI", return_value=mock_client), \
         patch.dict(os.environ, {"OPENAI_API_KEY": "test-key-mock"}, clear=False):
        result = cleaner.clean(seg)

    assert result.cleaned_text == "", (
        f"EMPTY response should produce empty cleaned_text; got {result.cleaned_text!r}"
    )
    assert result.original_text == raw
