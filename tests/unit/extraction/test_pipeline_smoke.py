"""Pipeline smoke tests — exercise the composition logic without GPU deps.

These tests inject lightweight stubs into `AsrPipeline` that obey the runtime
`_VadGate` / `_Transcriber` / `_AlignerDiarizer` protocols. They prove:
1. The module imports cleanly with no heavy deps present.
2. The VAD → Whisper → diariser orchestration forwards arguments correctly.
3. `Turn` normalisation handles missing `avg_logprob` gracefully.
4. `_collapse_turns` merges adjacent same-speaker WhisperX segments.

Full end-to-end (GPU + HF_TOKEN) integration is separately gated and lives
under `tests/integration/` once the extraction dev env is primed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.extraction.asr.pipeline import (
    AsrPipeline,
    PipelineConfig,
    Turn,
    _collapse_turns,  # pyright: ignore[reportPrivateUsage]  # internal helper tested directly
    _to_turn,  # pyright: ignore[reportPrivateUsage]  # internal helper tested directly
)


@dataclass
class _FakeVad:
    """Reports one speech segment per test — enough to trip the pipeline on."""

    segments: list[tuple[float, float]] = field(
        default_factory=lambda: [(0.0, 5.0)]
    )

    def speech_segments(
        self, wav_path: Path, sample_rate: int
    ) -> list[tuple[float, float]]:
        del wav_path, sample_rate
        return self.segments


@dataclass
class _FakeTranscriber:
    """Records the call args so tests can assert prompt threading."""

    canned: list[dict[str, Any]] = field(
        default_factory=lambda: [
            {"start": 0.0, "end": 2.5, "text": "Chest pain for one hour.", "avg_logprob": -0.3},
            {"start": 2.5, "end": 5.0, "text": "Takes metoprolol daily.", "avg_logprob": -0.2},
        ]
    )
    last_prompt: str | None = None
    last_beam: int | None = None

    def transcribe(
        self,
        wav_path: Path,
        *,
        initial_prompt: str | None,
        beam_size: int,
        vad_filter: bool,
    ) -> list[dict[str, Any]]:
        del wav_path, vad_filter
        self.last_prompt = initial_prompt
        self.last_beam = beam_size
        return self.canned


@dataclass
class _FakeAlignerDiarizer:
    """Returns canned WhisperX-shaped output, speaker labels interleaved."""

    canned: list[dict[str, Any]] = field(
        default_factory=lambda: [
            {
                "speaker": "SPEAKER_00",
                "start": 0.0,
                "end": 2.5,
                "text": "Chest pain for one hour.",
                "avg_logprob": -0.3,
            },
            {
                "speaker": "SPEAKER_01",
                "start": 2.5,
                "end": 5.0,
                "text": "Takes metoprolol daily.",
                "avg_logprob": -0.2,
            },
        ]
    )

    def align_and_diarise(
        self,
        wav_path: Path,
        segments: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        del wav_path, segments
        return self.canned


def _make_pipeline(config: PipelineConfig | None = None) -> tuple[
    AsrPipeline, _FakeVad, _FakeTranscriber, _FakeAlignerDiarizer
]:
    vad = _FakeVad()
    transcriber = _FakeTranscriber()
    aligner = _FakeAlignerDiarizer()
    pipe = AsrPipeline(
        config=config or PipelineConfig(hf_token="test-token"),
        vad=vad,
        transcriber=transcriber,
        aligner_diarizer=aligner,
        initial_prompt="CHEST-PAIN-BIAS",
    )
    return pipe, vad, transcriber, aligner


def test_pipeline_emits_turns_in_order() -> None:
    pipe, _, _, _ = _make_pipeline()
    turns = list(pipe.transcribe(Path("fake.wav")))
    assert len(turns) == 2
    assert turns[0].speaker == "SPEAKER_00"
    assert turns[1].speaker == "SPEAKER_01"
    assert turns[0].t_start == 0.0
    assert turns[1].t_end == 5.0
    assert "metoprolol" in turns[1].text


def test_pipeline_threads_initial_prompt_when_enabled() -> None:
    pipe, _, transcriber, _ = _make_pipeline()
    list(pipe.transcribe(Path("fake.wav")))
    assert transcriber.last_prompt == "CHEST-PAIN-BIAS"
    assert transcriber.last_beam == 5


def test_pipeline_skips_initial_prompt_when_disabled() -> None:
    cfg = PipelineConfig(hf_token="t", use_initial_prompt=False)
    pipe, _, transcriber, _ = _make_pipeline(config=cfg)
    list(pipe.transcribe(Path("fake.wav")))
    assert transcriber.last_prompt is None


def test_pipeline_handles_no_speech() -> None:
    """VAD reporting zero segments → empty iterator, no Whisper / diariser call."""
    pipe, vad, transcriber, _ = _make_pipeline()
    vad.segments = []
    turns = list(pipe.transcribe(Path("fake.wav")))
    assert turns == []
    assert transcriber.last_prompt is None  # never called


def test_to_turn_handles_missing_logprob() -> None:
    turn = _to_turn({"speaker": "S", "text": "hi", "start": 0.0, "end": 1.0})
    assert isinstance(turn, Turn)
    # Missing logprob → very low confidence, but non-negative.
    assert 0.0 <= turn.asr_confidence <= 1.0


def test_to_turn_confidence_in_range() -> None:
    # avg_logprob = 0.0 → exp(0) = 1.0 (best case).
    best = _to_turn({"speaker": "S", "text": "x", "start": 0.0, "end": 1.0, "avg_logprob": 0.0})
    assert best.asr_confidence == 1.0


def test_collapse_turns_merges_same_speaker() -> None:
    segs = [
        {"speaker": "A", "start": 0.0, "end": 1.0, "text": "hi", "avg_logprob": -0.1},
        {"speaker": "A", "start": 1.0, "end": 2.0, "text": "there", "avg_logprob": -0.2},
        {"speaker": "B", "start": 2.0, "end": 3.0, "text": "bye", "avg_logprob": -0.1},
    ]
    out = _collapse_turns(segs)
    assert len(out) == 2
    assert out[0]["text"] == "hi there"
    assert out[0]["end"] == 2.0
    assert out[1]["speaker"] == "B"


def test_collapse_turns_empty_input() -> None:
    assert _collapse_turns([]) == []
