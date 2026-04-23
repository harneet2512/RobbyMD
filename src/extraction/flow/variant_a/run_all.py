"""Run the Variant A measurement harness across all 6 clips.

Reads eval/synthetic_clips/ground_truth.jsonl, runs the pipeline, emits
per-clip metrics + aggregate results.json into
eval/flow_results/variant_a/<UTC-timestamp>/.

Run from the repo root on the L4:
    python -m src.extraction.flow.variant_a.run_all
"""

from __future__ import annotations

import datetime
import json
import os
import statistics
import traceback
from pathlib import Path

from src.extraction.flow.variant_a.measure import measure_one
from src.extraction.flow.variant_a.pipeline import VariantAPipeline

# Student's t two-tailed critical values at alpha=0.05 for df = n-1.
# Used for 95% CIs on small n. For df>=30 we fall back to the normal
# approximation 1.96.
_T_95 = {
    1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45,
    7: 2.36, 8: 2.31, 9: 2.26, 10: 2.23, 11: 2.20, 12: 2.18,
    13: 2.16, 14: 2.14, 15: 2.13, 16: 2.12, 17: 2.11, 18: 2.10,
    19: 2.09, 20: 2.09, 25: 2.06, 30: 2.04,
}


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    vs = sorted(values)
    idx = min(int(p * len(vs)), len(vs) - 1)
    return vs[idx]


def _stat_block(values: list[float]) -> dict | None:
    """Return {mean, stdev, n, ci95_low, ci95_high, min, max} for a list.

    Drops None values. Returns None if nothing left.
    """
    xs = [v for v in values if v is not None]
    n = len(xs)
    if n == 0:
        return None
    m = statistics.mean(xs)
    s = statistics.stdev(xs) if n > 1 else 0.0
    tcrit = _T_95.get(n - 1) or (2.04 if n - 1 >= 30 else 1.96)
    margin = tcrit * s / (n ** 0.5) if n > 1 else 0.0
    return {
        "mean": m,
        "stdev": s,
        "n": n,
        "ci95_low": m - margin,
        "ci95_high": m + margin,
        "min": min(xs),
        "max": max(xs),
    }


def _paired_diff_block(a_values: list[float], b_values: list[float]) -> dict | None:
    """Stat block for the paired difference b - a (e.g. cleaned - raw)."""
    diffs = [
        b - a
        for a, b in zip(a_values, b_values)
        if a is not None and b is not None
    ]
    return _stat_block(diffs)


def main() -> None:
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(f"eval/flow_results/variant_a/{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN environment variable is required")

    with open("eval/synthetic_clips/ground_truth.jsonl", encoding="utf-8") as f:
        clips = [json.loads(line) for line in f if line.strip()]

    pipeline = VariantAPipeline(hf_token=hf_token)

    per_clip_path = out_dir / "per_clip_metrics.jsonl"
    all_metrics: list[dict] = []
    for clip in clips:
        print(f"Measuring {clip['scenario']}...", flush=True)
        try:
            m = measure_one(clip, pipeline)
        except Exception as e:
            m = {
                "scenario": clip["scenario"],
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
            print(f"  FAILED: {e}", flush=True)
        all_metrics.append(m)
        with per_clip_path.open("a", encoding="utf-8") as o:
            o.write(json.dumps(m) + "\n")

    ok = [m for m in all_metrics if "error" not in m]
    diarisation_enabled = bool(ok and ok[0].get("diarisation_enabled"))

    agg: dict = {
        "variant": "A",
        "stack": (
            "Whisper-large-v3-turbo (MIT) + "
            "pyannote speaker-diarization-community-1 (MIT code + "
            "CC-BY-4.0 weights, pyannote.audio 4.x) + "
            "BioMistral-7B-DARE-AWQ cleanup (Apache-2.0) + "
            "Kokoro-82M TTS (Apache-2.0)."
        ) if diarisation_enabled else (
            "Whisper-large-v3-turbo (MIT) + "
            "BioMistral-7B-DARE-AWQ cleanup (Apache-2.0) + "
            "Kokoro-82M TTS (Apache-2.0). "
            "Diarisation disabled — speaker labels via alternating-turn "
            "heuristic; DER is N/A."
        ),
        "all_licenses_osi_or_open_data": True,
        "hardware": "NVIDIA L4 24GB on GCP",
        "n_clips": len(all_metrics),
        "n_ok": len(ok),
        "n_failed": len(all_metrics) - len(ok),
        "diarisation_enabled": diarisation_enabled,
        "calibration_note": (
            "Audio is Kokoro-rendered synthetic TTS (studio-clean, no room "
            "noise, no overlap, invariant mic distance). Real clinical audio "
            "typically gives 2-3× higher WER. These numbers are a best-case "
            "upper-bound on Whisper's performance, not a real-deployment "
            "estimate."
        ),
        "methodology_notes": {
            "first_segment_after_ingest_ms": (
                "Time from the transcribe() call to the first yielded "
                "Whisper segment. faster-whisper processes the first 30s "
                "chunk before yielding, so this is first-segment-after-"
                "ingest latency, NOT streaming first-word latency. Wispr "
                "Flow's 700 ms p99 is streaming-first-word; do not "
                "compare directly."
            ),
            "streaming_first_word_ms": (
                "A true streaming first-word measurement requires pre-"
                "chunking audio and measuring time to first non-empty word "
                "emission. Deferred to a follow-up run — reported as null "
                "until then."
            ),
            "medical_term_wer": (
                "Alignment-based error rate restricted to reference "
                "positions whose token is in the MEDICAL_TERMS set. Uses "
                "jiwer.process_words on the full strings then checks "
                "per-position match status — not a WER on subsetted token "
                "lists."
            ),
            "cleanup_llm_calls": (
                "Number of segments that actually hit the cleanup LLM. "
                "Segments with no filler token detected by the regex "
                "fast-path are returned verbatim and never sent to vLLM."
            ),
        },
    }

    if ok:
        der_values = [m["der"] for m in ok]
        agg["metrics"] = {
            "wer_raw": _stat_block([m["wer_raw"] for m in ok]),
            "wer_cleaned": _stat_block([m["wer_cleaned"] for m in ok]),
            "wer_cleaned_minus_raw_paired": _paired_diff_block(
                [m["wer_raw"] for m in ok], [m["wer_cleaned"] for m in ok]
            ),
            "medical_term_wer": _stat_block([m["medical_term_wer"] for m in ok]),
            "der": _stat_block(der_values),
            "asr_ms": _stat_block([m["asr_ms"] for m in ok]),
            "diar_ms": _stat_block([m["diar_ms"] for m in ok]),
            "cleanup_ms": _stat_block([m["cleanup_ms"] for m in ok]),
            "e2e_ms": _stat_block([m["e2e_ms"] for m in ok]),
            "first_segment_after_ingest_ms": _stat_block(
                [m["first_segment_after_ingest_ms"] for m in ok]
            ),
        }
        # Keep flat percentiles too — useful for competitive-table citation.
        agg["latency_percentiles_ms"] = {
            "first_segment_after_ingest_p50": _percentile(
                [m["first_segment_after_ingest_ms"] for m in ok], 0.5
            ),
            "first_segment_after_ingest_p90": _percentile(
                [m["first_segment_after_ingest_ms"] for m in ok], 0.9
            ),
            "e2e_p50": _percentile([m["e2e_ms"] for m in ok], 0.5),
            "e2e_p90": _percentile([m["e2e_ms"] for m in ok], 0.9),
            "e2e_p99": _percentile([m["e2e_ms"] for m in ok], 0.99),
        }
        agg["cleanup_llm_call_rate"] = {
            "total_calls": sum(m.get("cleanup_llm_calls", 0) for m in ok),
            "total_segments": sum(m.get("cleanup_total_segments", 0) for m in ok),
        }
        agg["vram_peak_mb_max"] = max(m["vram_peak_mb"] for m in ok)

    agg["competitive_context"] = {
        "wispr_flow_p99_target_ms": 700,
        "wispr_flow_metric": "streaming_first_word",
        "wispr_flow_medical_specialized": False,
        "nuance_dax_latency": "2-3 min async",
        "abridge_latency": "<2 min async",
        "nabla_latency": "<20 sec",
        "our_metric_note": (
            "Our first_segment_after_ingest_ms is NOT directly comparable "
            "to Wispr Flow's streaming_first_word p99. We sit above them "
            "for first-segment; streaming first-word from our stack is "
            "unmeasured."
        ),
    }
    agg["per_clip"] = all_metrics

    with (out_dir / "results.json").open("w", encoding="utf-8") as o:
        json.dump(agg, o, indent=2)

    summary = {k: v for k, v in agg.items() if k != "per_clip"}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
