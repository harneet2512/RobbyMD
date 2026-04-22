"""Tests for A.1 — hardened faster-whisper decoder parameters.

Six tests, one assertion per parameter. Uses a mock WhisperModel to capture
the kwargs passed to `model.transcribe` without loading any GPU weights.

Per the spec: each test verifies that the pipeline forwards the exact value
defined in PipelineConfig to the underlying transcriber.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.extraction.asr.pipeline import (
    AsrPipeline,
    PipelineConfig,
    _guess_speaker_role,  # pyright: ignore[reportPrivateUsage]
)


# ---------------------------------------------------------------------------
# Mock transcriber that captures all kwargs
# ---------------------------------------------------------------------------

@dataclass
class _CapturingTranscriber:
    """Captures the last set of kwargs passed to transcribe()."""

    canned: list[dict[str, Any]] = field(
        default_factory=lambda: [
            {"start": 0.0, "end": 2.5, "text": "hello", "avg_logprob": -0.3},
        ]
    )
    last_kwargs: dict[str, Any] = field(default_factory=dict)

    def transcribe(
        self,
        wav_path: Path,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        del wav_path
        self.last_kwargs = kwargs
        return self.canned


@dataclass
class _FakeVad:
    def speech_segments(self, wav_path: Path, sample_rate: int) -> list[tuple[float, float]]:
        del wav_path, sample_rate
        return [(0.0, 5.0)]


@dataclass
class _FakeAlignerDiarizer:
    canned: list[dict[str, Any]] = field(
        default_factory=lambda: [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 2.5,
             "text": "hello", "avg_logprob": -0.3},
        ]
    )

    def align_and_diarise(self, wav_path: Path, segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
        del wav_path, segments
        return self.canned


def _make_pipeline(cfg: PipelineConfig | None = None) -> tuple[AsrPipeline, _CapturingTranscriber]:
    cfg = cfg or PipelineConfig(hf_token="test", enable_cleanup=False, enable_preprocessing=False)
    trans = _CapturingTranscriber()
    pipe = AsrPipeline(
        config=cfg,
        vad=_FakeVad(),
        transcriber=trans,
        aligner_diarizer=_FakeAlignerDiarizer(),
        initial_prompt="BIAS",
    )
    return pipe, trans


# ---------------------------------------------------------------------------
# One test per decoder parameter
# ---------------------------------------------------------------------------

def test_condition_on_previous_text_forwarded() -> None:
    cfg = PipelineConfig(
        hf_token="t",
        enable_cleanup=False,
        enable_preprocessing=False,
        condition_on_previous_text=False,
    )
    pipe, trans = _make_pipeline(cfg)
    list(pipe.transcribe(Path("fake.wav")))
    assert trans.last_kwargs["condition_on_previous_text"] is False


def test_temperature_forwarded() -> None:
    cfg = PipelineConfig(
        hf_token="t",
        enable_cleanup=False,
        enable_preprocessing=False,
        temperature=0.0,
    )
    pipe, trans = _make_pipeline(cfg)
    list(pipe.transcribe(Path("fake.wav")))
    assert trans.last_kwargs["temperature"] == 0.0


def test_beam_size_forwarded() -> None:
    cfg = PipelineConfig(
        hf_token="t",
        enable_cleanup=False,
        enable_preprocessing=False,
        beam_size=5,
    )
    pipe, trans = _make_pipeline(cfg)
    list(pipe.transcribe(Path("fake.wav")))
    assert trans.last_kwargs["beam_size"] == 5


def test_compression_ratio_threshold_forwarded() -> None:
    cfg = PipelineConfig(
        hf_token="t",
        enable_cleanup=False,
        enable_preprocessing=False,
        compression_ratio_threshold=2.4,
    )
    pipe, trans = _make_pipeline(cfg)
    list(pipe.transcribe(Path("fake.wav")))
    assert trans.last_kwargs["compression_ratio_threshold"] == 2.4


def test_logprob_threshold_forwarded() -> None:
    cfg = PipelineConfig(
        hf_token="t",
        enable_cleanup=False,
        enable_preprocessing=False,
        logprob_threshold=-1.0,
    )
    pipe, trans = _make_pipeline(cfg)
    list(pipe.transcribe(Path("fake.wav")))
    assert trans.last_kwargs["logprob_threshold"] == -1.0


def test_no_speech_threshold_forwarded() -> None:
    cfg = PipelineConfig(
        hf_token="t",
        enable_cleanup=False,
        enable_preprocessing=False,
        no_speech_threshold=0.6,
    )
    pipe, trans = _make_pipeline(cfg)
    list(pipe.transcribe(Path("fake.wav")))
    assert trans.last_kwargs["no_speech_threshold"] == 0.6
