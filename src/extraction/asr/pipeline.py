"""ASR pipeline — preprocess → VAD → Whisper → WhisperX → diarise → cleanup → guard → correct.

Wired per `research/asr_stack.md` §2.1 and ASR hardening spec (Parts A-C).
Emits `CleanedDiarisedTurn` dataclasses the substrate's `src/substrate/admission.py`
consumes (wt-engine owns that interface).

Pipeline order (per Part C spec, order matters):
  audio
    → preprocess.normalize_audio        (16 kHz mono PCM, loudnorm)
    → preprocess.trim_silence           (strip boundary dead-air / hallucination trigger)
    → faster-whisper.transcribe         (6 hardened decoder params + initial_prompt)
    → whisperx align
    → pyannote diarise
    → TranscriptCleaner.clean           (per segment, cheap LLM)
    → hallucination_guard.check         (per segment, 5 deterministic checks)
    → word_correction.correct_medical_tokens  (per segment, Levenshtein)
    → list[CleanedDiarisedTurn]

Downstream (on_new_turn):
  cleaned_text    → claim extractor (demo-path Opus 4.7, untouched)
  original_text   → provenance payload  (rules.md §4)

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

Decoder parameters (A.1 hardening, explicit values prevent faster-whisper
default drift across versions):
- condition_on_previous_text=False  → prevents error cascades between segments
- temperature=0.0                   → deterministic decoding
- beam_size=5                       → matches PipelineConfig default
- compression_ratio_threshold=2.4   → recommended by faster-whisper docs
- logprob_threshold=-1.0            → suppress very-low-confidence hypotheses
- no_speech_threshold=0.6           → suppress silent-frame hallucinations

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

from src.extraction.asr import hallucination_guard, word_correction
from src.extraction.asr.hallucination_guard import HallucinationReport
from src.extraction.asr.transcript_cleanup import (
    CleanedSegment,
    CleanupUnavailable,
    DiarisedSegment,
    TranscriptCleaner,
)
from src.extraction.asr.vocab import build_initial_prompt

logger = structlog.get_logger(__name__)


# Model identifiers — these strings must exactly match rows in MODEL_ATTRIBUTIONS.md.
# `tests/licensing/test_model_attributions.py` scans for these at every commit.
WHISPER_LARGE_V3 = "openai/whisper-large-v3"
DISTIL_LARGE_V3 = "distil-whisper/distil-large-v3"
PYANNOTE_DIARIZER = "pyannote/speaker-diarization-community-1"


@dataclass(frozen=True, slots=True)
class Turn:
    """One diarised utterance emitted to the substrate (legacy; pre-cleanup).

    Mirrors the `turns` table schema in Eng_doc.md §4.1 modulo substrate-assigned
    ids (`turn_id`, `session_id`). Those are stamped by wt-engine's admission
    filter, not here — we're upstream of the write API.

    Retained for compatibility with existing tests. New code should prefer
    `CleanedDiarisedTurn` which carries provenance and guard results.
    """

    speaker: str  # diariser label, e.g. "SPEAKER_00"; substrate maps to patient/physician
    text: str
    t_start: float  # seconds from audio t=0
    t_end: float
    asr_confidence: float  # [0, 1]; exp(Whisper avg_logprob)


@dataclass(frozen=True, slots=True)
class CleanedDiarisedTurn:
    """One processed utterance after the full 8-stage pipeline.

    This is the type the substrate's `on_new_turn` handler receives.
    `original_text` is ALWAYS preserved for provenance (rules.md §4).
    `cleaned_text` is what the claim extractor (Opus 4.7 on demo path) sees.
    """

    speaker_role: str               # "doctor", "patient", or "unknown"
    cleaned_text: str               # post-cleanup + post-correction text
    original_text: str              # raw ASR output (provenance anchor)
    t_start: float
    t_end: float
    word_confidences: tuple[float, ...]
    corrections_applied: tuple[Any, ...]   # CleanupCorrection + word_correction.Correction
    hallucination_report: HallucinationReport


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Runtime knobs for the ASR pipeline.

    Defaults match `Eng_doc.md` §3.1 targets (RTF <= 0.7 on 16-24 GB GPU).

    Decoder parameters (A.1 hardening) are explicit here so they are visible
    to tests and cannot drift with faster-whisper version changes.
    """

    whisper_model_id: str = WHISPER_LARGE_V3
    distil_model_id: str = DISTIL_LARGE_V3
    diarizer_model_id: str = PYANNOTE_DIARIZER
    # faster-whisper knobs — documented in research/asr_stack.md §2.1.
    compute_type: str = "int8_float16"  # CTranslate2 quantisation
    beam_size: int = 5
    vad_filter: bool = False  # silero handles this upstream
    use_initial_prompt: bool = True
    # A.1 hardened decoder params — explicit values prevent version-drift surprises.
    condition_on_previous_text: bool = False   # prevents error cascades between segments
    temperature: float = 0.0                   # deterministic decoding
    compression_ratio_threshold: float = 2.4   # suppress repetitive-loop segments
    logprob_threshold: float = -1.0            # suppress very-low-confidence hypotheses
    no_speech_threshold: float = 0.6           # suppress silent-frame hallucinations
    # Fallback switch thresholds (research/asr_stack.md §2.1 "fallback switch").
    # Demo-video path leaves these disabled so every shot runs large-v3 (see §7 Q4).
    enable_distil_fallback: bool = False
    device: str = "cuda"  # "cpu" is supported; WER parity not guaranteed
    hf_token: str | None = None  # required for pyannote community-1 download
    # Cleanup model — passed through to TranscriptCleaner.
    # NEVER "claude-opus-*" (config.py guards this).
    cleanup_model: str = "gpt-4o-mini"
    enable_cleanup: bool = True      # set False in tests to skip LLM cleanup
    enable_preprocessing: bool = True  # set False to skip ffmpeg preprocessing
    # Text-input dormancy flag (docs/asr_engineering_spec.md §7).
    # When True (default), the pipeline skips TranscriptCleaner even if
    # ``enable_cleanup`` would otherwise have invoked it. This is the
    # defence-in-depth guarantee that ACI-Bench / LongMemEval text paths
    # never stack LLM cleanup on top of already-clean transcripts. Flip to
    # False ONLY for raw-audio ingest paths where cleanup provides value.
    bypass_cleanup_for_text_input: bool = True


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
        condition_on_previous_text: bool,
        temperature: float,
        compression_ratio_threshold: float,
        logprob_threshold: float,
        no_speech_threshold: float,
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
    """Composed VAD + ASR + alignment + diariser + cleanup + guard + correction.

    Construct via `build_pipeline` in production. Callers inject typed stubs
    directly in tests (see `tests/unit/extraction/test_pipeline_smoke.py`).

    Pipeline order (Part C spec):
      preprocess → vad → transcribe → align+diarise → cleanup → guard → correct
    """

    config: PipelineConfig
    vad: _VadGate
    transcriber: _Transcriber
    aligner_diarizer: _AlignerDiarizer
    initial_prompt: str = field(default_factory=build_initial_prompt)
    # Vocabulary for guard + correction (set at construction; refreshed per pack).
    medical_vocabulary: set[str] = field(default_factory=set)

    def transcribe(self, wav_path: Path) -> Iterator[Turn]:
        """Transcribe one WAV file, yielding legacy diarised `Turn`s.

        This method preserves backward compatibility with existing tests and
        callers. New code should use `transcribe_full` which returns
        `CleanedDiarisedTurn`s with provenance.
        """
        for cleaned_turn in self.transcribe_full(wav_path):
            # Synthesise a legacy Turn from the cleaned output.
            yield Turn(
                speaker=cleaned_turn.speaker_role,
                text=cleaned_turn.cleaned_text,
                t_start=cleaned_turn.t_start,
                t_end=cleaned_turn.t_end,
                asr_confidence=1.0 if not cleaned_turn.hallucination_report.flagged_spans else 0.0,
            )

    def transcribe_full(self, wav_path: Path) -> Iterator[CleanedDiarisedTurn]:
        """Full 8-stage pipeline — preprocess → … → CleanedDiarisedTurn.

        Per research/asr_stack.md §2.1 the diariser runs post-utterance on the
        full file, not per-window. This file is tolerant of 0 speech segments
        (empty iterator), which happens for silent or sub-threshold-VAD input.
        """
        from src.extraction.asr import preprocess, telemetry

        t0 = time.perf_counter()

        # Stage 1+2: preprocess (normalize + trim).
        if self.config.enable_preprocessing and preprocess.ffmpeg_available():
            import tempfile
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                norm_path = tmp_path / "norm.wav"
                trim_path = tmp_path / "trim.wav"

                @telemetry.measure("normalize")
                def _normalize() -> Path:
                    return preprocess.normalize_audio(wav_path, norm_path)

                @telemetry.measure("trim")
                def _trim() -> Path:
                    return preprocess.trim_silence(norm_path, trim_path)

                _normalize()
                _trim()
                effective_wav = trim_path
                yield from self._transcribe_inner(effective_wav, t0)
        else:
            yield from self._transcribe_inner(wav_path, t0)

    def _transcribe_inner(self, wav_path: Path, t0: float) -> Iterator[CleanedDiarisedTurn]:
        """Inner pipeline: VAD → transcribe → align+diarise → cleanup → guard → correct."""
        from src.extraction.asr import telemetry

        # Stage 3: VAD gate.
        segments = self.vad.speech_segments(wav_path, sample_rate=16_000)
        if not segments:
            logger.info(
                "asr_pipeline.vad.no_speech",
                wav=str(wav_path),
                elapsed_s=round(time.perf_counter() - t0, 3),
            )
            return

        # Stage 4: transcribe with all 6 hardened decoder params (A.1).
        prompt = self.initial_prompt if self.config.use_initial_prompt else None

        @telemetry.measure("transcribe")
        def _transcribe() -> list[dict[str, Any]]:
            return self.transcriber.transcribe(
                wav_path,
                initial_prompt=prompt,
                beam_size=self.config.beam_size,
                vad_filter=self.config.vad_filter,
                condition_on_previous_text=self.config.condition_on_previous_text,
                temperature=self.config.temperature,
                compression_ratio_threshold=self.config.compression_ratio_threshold,
                logprob_threshold=self.config.logprob_threshold,
                no_speech_threshold=self.config.no_speech_threshold,
            )

        raw = _transcribe()
        logger.info(
            "asr_pipeline.whisper.done",
            segments=len(raw),
            prompt_used=bool(prompt),
            elapsed_s=round(time.perf_counter() - t0, 3),
        )

        # Stage 5: align + diarise.
        @telemetry.measure("diarise")
        def _diarise() -> list[dict[str, Any]]:
            return self.aligner_diarizer.align_and_diarise(wav_path, raw)

        diarised = _diarise()
        logger.info(
            "asr_pipeline.diarised.done",
            turns=len(diarised),
            elapsed_s=round(time.perf_counter() - t0, 3),
        )

        # Build cleanup helper (shared context across turns in this clip).
        # Text-input dormancy (docs/asr_engineering_spec.md §7): when the caller
        # has set bypass_cleanup_for_text_input=True, we short-circuit before
        # ever instantiating the cleaner — a belt-and-braces guarantee that no
        # LLM cleanup runs on text-input eval paths even if enable_cleanup
        # gets flipped somewhere downstream.
        cleaner: TranscriptCleaner | None = None
        if self.config.bypass_cleanup_for_text_input:
            logger.info(
                "asr_pipeline.cleanup_bypassed_for_text_input",
                reason="bypass_cleanup_for_text_input=True",
            )
        elif self.config.enable_cleanup:
            try:
                cleaner = TranscriptCleaner(
                    medical_vocabulary=self.medical_vocabulary,
                    cleanup_model=self.config.cleanup_model,
                )
            except Exception as exc:
                logger.warning("asr_pipeline.cleaner_init_failed", error=str(exc))

        for turn_dict in diarised:
            turn = _to_turn(turn_dict)
            speaker_role = _guess_speaker_role(turn.speaker)
            raw_text = turn.text

            # Stage 6: LLM cleanup (per segment).
            cleaned_text = raw_text
            cleanup_corrections: tuple[Any, ...] = ()
            if cleaner is not None:
                try:
                    @telemetry.measure("cleanup")
                    def _cleanup() -> CleanedSegment:
                        assert cleaner is not None  # noqa: S101 — for pyright
                        seg = DiarisedSegment(
                            speaker_role=speaker_role,
                            raw_text=raw_text,
                            t_start=turn.t_start,
                            t_end=turn.t_end,
                        )
                        return cleaner.clean(seg)

                    cleaned_seg = _cleanup()
                    cleaned_text = cleaned_seg.cleaned_text or raw_text
                    cleanup_corrections = cleaned_seg.corrections_applied
                except CleanupUnavailable as exc:
                    logger.warning("asr_pipeline.cleanup_unavailable", error=str(exc))
                except Exception as exc:
                    logger.warning("asr_pipeline.cleanup_error", error=str(exc))

            # Stage 7: hallucination guard (on cleaned text).
            word_confs = list(turn_dict.get("word_confidences", []))

            @telemetry.measure("guard")
            def _guard() -> HallucinationReport:
                return hallucination_guard.check(
                    text=cleaned_text,
                    vocabulary=self.medical_vocabulary,
                    word_confidences=word_confs,
                    audio_duration_s=max(0.0, turn.t_end - turn.t_start),
                )

            report = _guard()

            # Stage 8: word correction (on cleaned text, after guard).
            final_text = cleaned_text
            word_corrections: list[word_correction.Correction] = []
            if self.medical_vocabulary:
                try:
                    @telemetry.measure("correct")
                    def _correct() -> tuple[str, list[word_correction.Correction]]:
                        return word_correction.correct_medical_tokens(
                            transcript=cleaned_text,
                            vocabulary=self.medical_vocabulary,
                        )

                    final_text, word_corrections = _correct()
                except Exception as exc:
                    logger.warning("asr_pipeline.word_correction_error", error=str(exc))

            all_corrections: tuple[Any, ...] = cleanup_corrections + tuple(word_corrections)

            yield CleanedDiarisedTurn(
                speaker_role=speaker_role,
                cleaned_text=final_text,
                original_text=raw_text,
                t_start=turn.t_start,
                t_end=turn.t_end,
                word_confidences=tuple(word_confs),
                corrections_applied=all_corrections,
                hallucination_report=report,
            )


def _guess_speaker_role(speaker_label: str) -> str:
    """Map a diariser speaker label to a role string.

    The pyannote diariser emits labels like "SPEAKER_00" / "SPEAKER_01". We
    adopt the convention that SPEAKER_00 is the physician (who typically
    speaks first) and SPEAKER_01 is the patient. Unknown / higher-numbered
    labels → "unknown".

    This mapping is a demo heuristic. Production would use explicit role
    assignment from the session setup UI.
    """
    label = speaker_label.upper()
    if "00" in label:
        return "doctor"
    if "01" in label:
        return "patient"
    return "unknown"


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
            condition_on_previous_text: bool = False,
            temperature: float = 0.0,
            compression_ratio_threshold: float = 2.4,
            logprob_threshold: float = -1.0,
            no_speech_threshold: float = 0.6,
        ) -> list[dict[str, Any]]:
            # All 6 A.1 hardened decoder params passed explicitly.
            segments_iter, _info = model.transcribe(
                str(wav_path),
                initial_prompt=initial_prompt,
                beam_size=beam_size,
                vad_filter=vad_filter,
                condition_on_previous_text=condition_on_previous_text,
                temperature=temperature,
                compression_ratio_threshold=compression_ratio_threshold,
                logprob_threshold=logprob_threshold,
                no_speech_threshold=no_speech_threshold,
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
