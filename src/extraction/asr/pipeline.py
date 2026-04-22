"""ASR pipeline — VAD → Whisper → WhisperX alignment → pyannote diarisation.

Wired per `research/asr_stack.md` §2.1. Emits `Turn` dataclasses the substrate's
`src/substrate/admission.py` consumes (wt-engine owns that interface).

Model stack (all declared in MODEL_ATTRIBUTIONS.md):
- silero-vad v5 (MIT) — voice activity gate.
- faster-whisper + CTranslate2 with openai/whisper-large-v3 weights (Apache-2.0).
  Distil-Whisper large-v3 (MIT) is loaded too for the ≤1.5 s latency fallback.
- WhisperX (BSD-2-Clause) — wav2vec2 forced-alignment for word timestamps.
- pyannote/speaker-diarization-community-1 (CC-BY-4.0) — speaker labels.

Heavy imports (`faster_whisper`, `whisperx`, `pyannote.audio`, `silero_vad`,
`torch`) are pushed down into `build_pipeline` so the rest of the repo (and
lightweight unit tests) can import this module without pulling a GPU stack.
Tests that exercise the model graph are gated on a `HF_TOKEN` + GPU and
marked `@pytest.mark.slow`.

Contract with downstream:
- One `Turn` per diarised utterance, in timestamp order.
- `t_start`, `t_end` are seconds from the audio clip's t=0.
- `asr_confidence` is faster-whisper's `avg_logprob`, mapped to [0, 1]
  via `exp(logprob)` (Whisper's published convention).

Per CLAUDE.md §8: structlog JSON, session_id / claim_id on every line.
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast

import structlog

from src.extraction.asr.vocab import build_initial_prompt

logger = structlog.get_logger(__name__)


# Model identifiers — these strings must exactly match rows in MODEL_ATTRIBUTIONS.md.
# `tests/licensing/test_model_attributions.py` scans for these at every commit.
WHISPER_LARGE_V3 = "openai/whisper-large-v3"
DISTIL_LARGE_V3 = "distil-whisper/distil-large-v3"
PYANNOTE_DIARIZER = "pyannote/speaker-diarization-community-1"


@dataclass(frozen=True, slots=True)
class Turn:
    """One diarised utterance emitted to the substrate.

    Mirrors the `turns` table schema in Eng_doc.md §4.1 modulo substrate-assigned
    ids (`turn_id`, `session_id`). Those are stamped by wt-engine's admission
    filter, not here — we're upstream of the write API.
    """

    speaker: str  # diariser label, e.g. "SPEAKER_00"; substrate maps to patient/physician
    text: str
    t_start: float  # seconds from audio t=0
    t_end: float
    asr_confidence: float  # [0, 1]; exp(Whisper avg_logprob)


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Runtime knobs for the ASR pipeline.

    Defaults match `Eng_doc.md` §3.1 targets (RTF <= 0.7 on 16-24 GB GPU).
    """

    whisper_model_id: str = WHISPER_LARGE_V3
    distil_model_id: str = DISTIL_LARGE_V3
    diarizer_model_id: str = PYANNOTE_DIARIZER
    # faster-whisper knobs — documented in research/asr_stack.md §2.1.
    compute_type: str = "int8_float16"  # CTranslate2 quantisation
    beam_size: int = 5
    vad_filter: bool = False  # silero handles this upstream
    use_initial_prompt: bool = True
    # Fallback switch thresholds (research/asr_stack.md §2.1 "fallback switch").
    # Demo-video path leaves these disabled so every shot runs large-v3 (see §7 Q4).
    enable_distil_fallback: bool = False
    device: str = "cuda"  # "cpu" is supported; WER parity not guaranteed
    hf_token: str | None = None  # required for pyannote community-1 download


class _VadGate(Protocol):
    """Runtime-dispatched silero-vad wrapper; typed loosely to avoid import pull."""

    def speech_segments(self, wav_path: Path, sample_rate: int) -> list[tuple[float, float]]:
        ...


class _Transcriber(Protocol):
    """Runtime-dispatched faster-whisper wrapper."""

    def transcribe(
        self,
        wav_path: Path,
        *,
        initial_prompt: str | None,
        beam_size: int,
        vad_filter: bool,
    ) -> list[dict[str, Any]]:
        ...


class _AlignerDiarizer(Protocol):
    """Runtime-dispatched WhisperX + pyannote wrapper."""

    def align_and_diarise(
        self,
        wav_path: Path,
        segments: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        ...


@dataclass
class AsrPipeline:
    """Composed VAD + ASR + alignment + diariser.

    Construct via `build_pipeline` in production. Callers inject typed stubs
    directly in tests (see `tests/unit/extraction/test_pipeline_smoke.py`).
    """

    config: PipelineConfig
    vad: _VadGate
    transcriber: _Transcriber
    aligner_diarizer: _AlignerDiarizer
    initial_prompt: str = field(default_factory=build_initial_prompt)

    def transcribe(self, wav_path: Path) -> Iterator[Turn]:
        """Transcribe one WAV file, yielding diarised `Turn`s in order.

        Per research/asr_stack.md §2.1 the diariser runs post-utterance on the
        full file, not per-window. This file is tolerant of 0 speech segments
        (empty iterator), which happens for silent or sub-threshold-VAD input.
        """
        t0 = time.perf_counter()
        segments = self.vad.speech_segments(wav_path, sample_rate=16_000)
        if not segments:
            logger.info(
                "asr_pipeline.vad.no_speech",
                wav=str(wav_path),
                elapsed_s=round(time.perf_counter() - t0, 3),
            )
            return

        prompt = self.initial_prompt if self.config.use_initial_prompt else None
        raw = self.transcriber.transcribe(
            wav_path,
            initial_prompt=prompt,
            beam_size=self.config.beam_size,
            vad_filter=self.config.vad_filter,
        )
        logger.info(
            "asr_pipeline.whisper.done",
            segments=len(raw),
            prompt_used=bool(prompt),
            elapsed_s=round(time.perf_counter() - t0, 3),
        )

        diarised = self.aligner_diarizer.align_and_diarise(wav_path, raw)
        logger.info(
            "asr_pipeline.diarised.done",
            turns=len(diarised),
            elapsed_s=round(time.perf_counter() - t0, 3),
        )

        for turn_dict in diarised:
            yield _to_turn(turn_dict)


def _to_turn(d: dict[str, Any]) -> Turn:
    """Normalise the aligner/diariser's dict-of-lists output to a Turn.

    Expected keys (per WhisperX documented schema): `speaker`, `text`, `start`,
    `end`, `avg_logprob`. Missing `avg_logprob` → confidence 0.0 (worst) so the
    admission filter can down-weight it.
    """
    logprob = float(d.get("avg_logprob", math.log(1e-6)))
    # Whisper avg_logprob is in (-∞, 0]; exp maps to (0, 1]. Clamp defensively.
    confidence = max(0.0, min(1.0, math.exp(logprob)))
    return Turn(
        speaker=str(d.get("speaker", "SPEAKER_UNKNOWN")),
        text=str(d.get("text", "")).strip(),
        t_start=float(d.get("start", 0.0)),
        t_end=float(d.get("end", 0.0)),
        asr_confidence=confidence,
    )


def build_pipeline(config: PipelineConfig | None = None) -> AsrPipeline:
    """Construct the live pipeline — loads the model graph.

    Side effects: downloads/loads ~3 GB of weights on first call. Callers needing
    a lightweight import (unit-test smoke, docs) inject stubs into `AsrPipeline`
    directly instead of calling this. See `tests/unit/extraction/test_pipeline_smoke.py`.

    Raises `RuntimeError` if HF_TOKEN is missing (required for pyannote download).
    """
    cfg = config or PipelineConfig()

    if cfg.hf_token is None:
        raise RuntimeError(
            "build_pipeline: HF_TOKEN required for pyannote/speaker-diarization-community-1 "
            "download. Set HF_TOKEN env var or pass PipelineConfig(hf_token=...). "
            "See research/asr_stack.md §2.1."
        )

    # Lazy imports so the module is importable without the GPU stack present.
    # Each is resolved via runtime adapter classes below.
    vad = _build_silero_vad()
    transcriber = _build_faster_whisper(cfg)
    aligner_diarizer = _build_whisperx_pyannote(cfg)

    logger.info(
        "asr_pipeline.built",
        whisper=cfg.whisper_model_id,
        distil=cfg.distil_model_id if cfg.enable_distil_fallback else None,
        diarizer=cfg.diarizer_model_id,
        device=cfg.device,
    )
    return AsrPipeline(
        config=cfg,
        vad=vad,
        transcriber=transcriber,
        aligner_diarizer=aligner_diarizer,
    )


def _build_silero_vad() -> _VadGate:
    """Load silero-vad v5 and wrap it to the `_VadGate` protocol."""
    # Heavy dep without type stubs — cast to Any so pyright strict doesn't
    # leak `Unknown` types through the rest of the file.
    import silero_vad as silero  # type: ignore[import-not-found]

    silero_any = cast(Any, silero)
    model = silero_any.load_silero_vad()

    class _SileroGate:
        def speech_segments(
            self, wav_path: Path, sample_rate: int
        ) -> list[tuple[float, float]]:
            wav = silero_any.read_audio(str(wav_path), sampling_rate=sample_rate)
            raw = silero_any.get_speech_timestamps(
                wav, model, sampling_rate=sample_rate
            )
            return [(s["start"] / sample_rate, s["end"] / sample_rate) for s in raw]

    return _SileroGate()


def _build_faster_whisper(cfg: PipelineConfig) -> _Transcriber:
    """Load faster-whisper with large-v3 weights and wrap it to `_Transcriber`."""
    import faster_whisper  # type: ignore[import-not-found]

    fw_any = cast(Any, faster_whisper)
    # NOTE: WhisperModel("large-v3") is aliased to openai/whisper-large-v3 in
    # tests/licensing/test_model_attributions.py — both forms map to the same
    # attribution row in MODEL_ATTRIBUTIONS.md. The alias table in that test is
    # the authoritative canonicalisation.
    model = fw_any.WhisperModel(
        cfg.whisper_model_id,
        device=cfg.device,
        compute_type=cfg.compute_type,
    )

    class _FasterWhisper:
        def transcribe(
            self,
            wav_path: Path,
            *,
            initial_prompt: str | None,
            beam_size: int,
            vad_filter: bool,
        ) -> list[dict[str, Any]]:
            segments_iter, _info = model.transcribe(
                str(wav_path),
                initial_prompt=initial_prompt,
                beam_size=beam_size,
                vad_filter=vad_filter,
            )
            out: list[dict[str, Any]] = []
            for s in segments_iter:
                out.append(
                    {
                        "start": float(s.start),
                        "end": float(s.end),
                        "text": str(s.text),
                        "avg_logprob": float(s.avg_logprob),
                    }
                )
            return out

    return _FasterWhisper()


def _build_whisperx_pyannote(cfg: PipelineConfig) -> _AlignerDiarizer:
    """Load WhisperX alignment model + pyannote diariser, return `_AlignerDiarizer`."""
    import whisperx  # type: ignore[import-not-found]

    wx_any = cast(Any, whisperx)
    align_model, align_meta = wx_any.load_align_model(
        language_code="en", device=cfg.device
    )
    # HF token presence is guaranteed by build_pipeline's guard, but the extra
    # check gives pyright strict a narrower type for the arg below.
    if cfg.hf_token is None:
        raise RuntimeError("hf_token missing — build_pipeline should have caught this")
    diarize_model = wx_any.DiarizationPipeline(
        model_name=cfg.diarizer_model_id,
        use_auth_token=cfg.hf_token,
        device=cfg.device,
    )

    class _WhisperXDiar:
        def align_and_diarise(
            self,
            wav_path: Path,
            segments: list[dict[str, Any]],
        ) -> list[dict[str, Any]]:
            audio = wx_any.load_audio(str(wav_path))
            aligned = wx_any.align(
                segments,
                align_model,
                align_meta,
                audio,
                cfg.device,
                return_char_alignments=False,
            )
            diarisation = diarize_model(audio)
            merged = wx_any.assign_word_speakers(diarisation, aligned)
            # WhisperX returns `segments` with per-segment `speaker`. Collapse
            # consecutive same-speaker segments so one utterance = one Turn.
            raw_segs: list[dict[str, Any]] = list(merged.get("segments", []))
            return _collapse_turns(raw_segs)

    return _WhisperXDiar()


def _collapse_turns(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge consecutive same-speaker WhisperX segments into one turn each."""
    if not segments:
        return []
    out: list[dict[str, Any]] = []
    cur = dict(segments[0])
    cur.setdefault("speaker", "SPEAKER_UNKNOWN")
    for seg in segments[1:]:
        speaker = seg.get("speaker", "SPEAKER_UNKNOWN")
        if speaker == cur["speaker"]:
            cur["end"] = seg["end"]
            cur["text"] = (cur["text"] + " " + seg["text"]).strip()
            # avg_logprob: take the length-weighted mean so long segments dominate.
            if "avg_logprob" in seg and "avg_logprob" in cur:
                dur_cur = cur.get("end", 0.0) - cur.get("start", 0.0)
                dur_seg = seg.get("end", 0.0) - seg.get("start", 0.0)
                total = max(1e-6, dur_cur + dur_seg)
                cur["avg_logprob"] = (
                    cur["avg_logprob"] * dur_cur + seg["avg_logprob"] * dur_seg
                ) / total
        else:
            out.append(cur)
            cur = dict(seg)
    out.append(cur)
    return out
