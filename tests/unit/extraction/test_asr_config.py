"""Tests for Part B config — DemoCleanupConfig, EvalCleanupConfig.

Two tests:
1. Demo vs eval configs have distinct defaults (different cleanup models, latencies, context windows).
2. Passing an Opus model string to any cleanup config raises ValueError.
"""

from __future__ import annotations

import pytest

from src.extraction.asr.config import DemoCleanupConfig, EvalASRConfig, EvalCleanupConfig, DemoASRConfig


def test_demo_vs_eval_configs_are_distinct() -> None:
    """DemoCleanupConfig and EvalCleanupConfig should differ in key parameters."""
    demo = DemoCleanupConfig()
    eval_ = EvalCleanupConfig()

    assert demo.cleanup_model != eval_.cleanup_model, (
        "Demo and eval cleanup models should differ"
    )
    assert demo.max_latency_ms < eval_.max_latency_ms, (
        "Demo latency budget should be tighter than eval"
    )
    assert demo.context_window < eval_.context_window, (
        "Eval should use a larger context window"
    )

    demo_asr = DemoASRConfig()
    eval_asr = EvalASRConfig()
    assert demo_asr.batch_size < eval_asr.batch_size, (
        "Eval ASR should use larger batch"
    )
    assert demo_asr.chunk_size_s < eval_asr.chunk_size_s, (
        "Eval ASR should use longer chunks"
    )


def test_opus_model_rejected_at_construction() -> None:
    """Passing any claude-opus model as cleanup_model must raise ValueError immediately."""
    with pytest.raises(ValueError, match="claude-opus"):
        DemoCleanupConfig(cleanup_model="claude-opus-4-7")

    with pytest.raises(ValueError, match="claude-opus"):
        EvalCleanupConfig(cleanup_model="claude-opus-4-5")
