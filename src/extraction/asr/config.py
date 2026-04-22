"""ASR and transcript-cleanup configuration dataclasses.

Two environments, two config pairs:
- **Demo** — optimised for lowest latency in a live consultation. Small batch,
  short chunks, gpt-4o-mini cleanup, 500 ms cleanup budget.
- **Eval** — optimised for throughput on the benchmark harness. Larger batch,
  longer chunks, local Qwen2.5-7B cleanup, 2 s cleanup budget.

Hard safety rule (Eng_doc.md §3.5 "Tank for the war" policy):
    Opus 4.7 is NEVER a cleanup model. The cleanup step is bulk / throughput,
    not demo-path. Using Opus 4.7 here would both burn the API budget and
    muddy benchmark comparisons. If any caller accidentally passes a
    ``cleanup_model`` that starts with ``"claude-opus"``, the constructors raise
    ``ValueError`` at construction time.

Text-input eval dormancy (2026-04-22):
    ACI-Bench and LongMemEval feed the substrate already-clean text, not raw
    audio. Running ``TranscriptCleaner`` on top of that text is defence-in-depth
    that becomes a liability: the cleanup LLM can paraphrase, re-wrap, or
    "helpfully" normalise medical phrasing, which invalidates apples-to-apples
    comparison against the published baselines. ``PipelineConfig`` carries a
    ``bypass_cleanup_for_text_input`` flag (default ``True``) that the pipeline
    orchestration consults before invoking the cleanup stage. See
    ``docs/asr_engineering_spec.md`` §7 for the full dormancy guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DemoCleanupConfig:
    """Transcript-cleanup settings for the live consultation demo path.

    Model: gpt-4o-mini (cheapest OpenAI model; acceptable quality for single-
    turn cleanup in a streaming context).
    Latency budget: 500 ms P95 (matches the overall 5 s utterance-to-claim
    P50 target in docs/asr_performance_spec.md).
    """

    cleanup_model: str = "gpt-4o-mini"
    max_latency_ms: int = 500
    context_window: int = 5  # number of previous cleaned turns to inject

    def __post_init__(self) -> None:
        _guard_no_opus(self.cleanup_model)


@dataclass
class EvalCleanupConfig:
    """Transcript-cleanup settings for the eval benchmark harness.

    Model: qwen2.5-7b-local (Apache-2.0, self-hosted via vLLM on GCP L4 spot).
    Latency budget: 2000 ms P95 (throughput run, not interactive).
    Context window: 10 (longer context helps on multi-turn ACI-Bench encounters).
    """

    cleanup_model: str = "qwen2.5-7b-local"
    max_latency_ms: int = 2000
    context_window: int = 10

    def __post_init__(self) -> None:
        _guard_no_opus(self.cleanup_model)


@dataclass
class DemoASRConfig:
    """ASR pipeline batch / chunking settings for the live demo.

    Small batch and short chunks keep per-utterance latency low.
    """

    batch_size: int = 4
    chunk_size_s: int = 5
    overlap_s: int = 2


@dataclass
class EvalASRConfig:
    """ASR pipeline batch / chunking settings for the eval harness.

    Large batch and long chunks maximise GPU utilisation during bulk runs.
    No overlap needed for non-streaming batch transcription.
    """

    batch_size: int = 16
    chunk_size_s: int = 30
    overlap_s: int = 0


def _guard_no_opus(cleanup_model: str) -> None:
    """Raise ValueError if the cleanup model is Opus 4.7.

    Opus 4.7 is reserved for demo-path claim extraction, verifier phrasing,
    and SOAP note composition (Eng_doc.md §3.5). Cleanup is a bulk / throughput
    step that must use a cheap model (gpt-4o-mini / qwen2.5-7b-local).
    """
    if cleanup_model.startswith("claude-opus"):
        raise ValueError(
            f"cleanup_model={cleanup_model!r} is not allowed. "
            "Opus 4.7 is reserved for demo-path calls only (Eng_doc.md §3.5). "
            "Use 'gpt-4o-mini' (DemoCleanupConfig) or 'qwen2.5-7b-local' "
            "(EvalCleanupConfig) for transcript cleanup."
        )
