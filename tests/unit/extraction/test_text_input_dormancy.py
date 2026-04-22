"""Text-input cleanup dormancy regression test.

Guarantees (docs/asr_engineering_spec.md §7):
    When ``PipelineConfig.bypass_cleanup_for_text_input=True`` (the default),
    the ASR pipeline MUST NOT invoke ``TranscriptCleaner.clean`` for any turn
    — even when ``enable_cleanup=True`` and a valid ``cleanup_model`` is set.

Rationale: ACI-Bench and LongMemEval feed the substrate already-clean text.
Running cleanup on text-input eval paths would paraphrase or re-wrap content,
invalidating apples-to-apples comparison against published baselines. This
test is the load-bearing regression guard against that silent failure mode.

See also:
- ``src/extraction/asr/pipeline.py::AsrPipeline._transcribe_inner``
  (cleanup gate; branches on ``self.config.bypass_cleanup_for_text_input``).
- ``src/extraction/asr/config.py`` (module docstring records the invariant).
- ``reasons.md`` entry 2026-04-22
  "bypass_cleanup_for_text_input default True — defence-in-depth".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from src.extraction.asr.pipeline import (
    AsrPipeline,
    PipelineConfig,
)


# ---------------------------------------------------------------------------
# Stubs — same shape as tests/unit/extraction/test_pipeline_smoke.py, kept
# self-contained so this file can be read as a standalone regression guard.
# ---------------------------------------------------------------------------


@dataclass
class _FakeVad:
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
    canned: list[dict[str, Any]] = field(
        default_factory=lambda: [
            {"start": 0.0, "end": 2.5, "text": "already clean text", "avg_logprob": -0.2},
        ]
    )

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
        del wav_path, initial_prompt, beam_size, vad_filter
        del condition_on_previous_text, temperature, compression_ratio_threshold
        del logprob_threshold, no_speech_threshold
        return self.canned


@dataclass
class _FakeAlignerDiarizer:
    canned: list[dict[str, Any]] = field(
        default_factory=lambda: [
            {
                "speaker": "SPEAKER_00",
                "start": 0.0,
                "end": 2.5,
                "text": "already clean text",
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


def _make_pipeline(
    *,
    bypass: bool,
    enable_cleanup: bool = True,
) -> AsrPipeline:
    cfg = PipelineConfig(
        hf_token="test-token",
        enable_cleanup=enable_cleanup,
        enable_preprocessing=False,   # skip ffmpeg; not what we are testing
        bypass_cleanup_for_text_input=bypass,
    )
    return AsrPipeline(
        config=cfg,
        vad=_FakeVad(),
        transcriber=_FakeTranscriber(),
        aligner_diarizer=_FakeAlignerDiarizer(),
        initial_prompt="",
    )


# ---------------------------------------------------------------------------
# Regression tests
# ---------------------------------------------------------------------------


def test_bypass_true_does_not_invoke_transcript_cleaner() -> None:
    """When bypass_cleanup_for_text_input=True, cleanup is never invoked.

    Patches ``TranscriptCleaner.clean`` with a MagicMock, runs a full pipeline
    pass, asserts the mock's ``call_count`` is zero.
    """
    pipe = _make_pipeline(bypass=True, enable_cleanup=True)

    with patch(
        "src.extraction.asr.pipeline.TranscriptCleaner",
    ) as cleaner_cls:
        instance = MagicMock()
        cleaner_cls.return_value = instance

        turns = list(pipe.transcribe_full(Path("fake.wav")))

        # Cleaner was never constructed because the pipeline short-circuited
        # before instantiation. Asserting on the class itself is the strongest
        # invariant — no cleanup LLM was even primed.
        assert cleaner_cls.call_count == 0, (
            "TranscriptCleaner must not be constructed on text-input paths "
            f"(got {cleaner_cls.call_count} constructions)."
        )
        # And of course .clean() was never called.
        assert instance.clean.call_count == 0, (
            "TranscriptCleaner.clean must not be invoked on text-input paths "
            f"(got {instance.clean.call_count} invocations)."
        )

    # Pipeline still emits the turn, with cleaned_text == original_text
    # (no cleanup transformation applied).
    assert len(turns) == 1
    assert turns[0].cleaned_text == turns[0].original_text == "already clean text"


def test_bypass_false_does_invoke_transcript_cleaner() -> None:
    """Control: when bypass=False AND enable_cleanup=True, cleanup IS invoked.

    This is the negative control that proves the regression test above is
    measuring the flag, not an unrelated short-circuit.
    """
    pipe = _make_pipeline(bypass=False, enable_cleanup=True)

    with patch(
        "src.extraction.asr.pipeline.TranscriptCleaner",
    ) as cleaner_cls:
        instance = MagicMock()
        # Make the cleaner pass the text through unchanged so the test does
        # not depend on any real cleanup transform.
        from src.extraction.asr.transcript_cleanup import CleanedSegment

        instance.clean.return_value = CleanedSegment(
            speaker_role="doctor",
            cleaned_text="already clean text",
            original_text="already clean text",
            corrections_applied=(),
            confidence=None,
            t_start=0.0,
            t_end=2.5,
        )
        cleaner_cls.return_value = instance

        list(pipe.transcribe_full(Path("fake.wav")))

        assert cleaner_cls.call_count == 1, (
            "With bypass=False and enable_cleanup=True, TranscriptCleaner "
            "must be constructed exactly once per clip."
        )
        assert instance.clean.call_count == 1, (
            "With bypass=False and enable_cleanup=True, clean() must be "
            "invoked once per segment (1 segment in this fixture)."
        )


def test_bypass_default_is_true() -> None:
    """The default value of bypass_cleanup_for_text_input is True.

    Safer default: the only path that opts in is the raw-audio demo path,
    which must flip the flag deliberately. Any new eval harness written in
    the future inherits the dormant-cleanup guarantee for free.
    """
    cfg = PipelineConfig(hf_token="t")
    assert cfg.bypass_cleanup_for_text_input is True
