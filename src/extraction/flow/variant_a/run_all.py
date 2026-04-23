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
import traceback
from pathlib import Path

from src.extraction.flow.variant_a.measure import measure_one
from src.extraction.flow.variant_a.pipeline import VariantAPipeline


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    vs = sorted(values)
    idx = min(int(p * len(vs)), len(vs) - 1)
    return vs[idx]


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
    agg = {
        "variant": "A",
        "stack": (
            "Whisper-large-v3-turbo (MIT) + WhisperX alignment (BSD-2) + "
            "pyannote speaker-diarization-community-1 (MIT code + CC-BY-4.0 weights) + "
            "BioMistral-7B-DARE cleanup (Apache-2.0) + Kokoro-82M TTS (Apache-2.0)"
        ),
        "all_licenses_osi_or_open_data": True,
        "hardware": "NVIDIA L4 24GB on GCP (Aravind project)",
        "n_clips": len(all_metrics),
        "n_ok": len(ok),
        "n_failed": len(all_metrics) - len(ok),
    }
    if ok:
        agg.update(
            {
                "wer_raw_mean": sum(m["wer_raw"] for m in ok) / len(ok),
                "wer_cleaned_mean": sum(m["wer_cleaned"] for m in ok) / len(ok),
                "medical_term_wer_mean": sum(m["medical_term_wer"] for m in ok) / len(ok),
                "der_mean": sum(m["der"] for m in ok if m["der"] >= 0) / max(
                    1, len([m for m in ok if m["der"] >= 0])
                ),
                "first_token_ms_p50": _percentile([m["first_token_ms"] for m in ok], 0.5),
                "first_token_ms_p90": _percentile([m["first_token_ms"] for m in ok], 0.9),
                "asr_ms_mean": sum(m["asr_ms"] for m in ok) / len(ok),
                "diar_ms_mean": sum(m["diar_ms"] for m in ok) / len(ok),
                "cleanup_ms_mean": sum(m["cleanup_ms"] for m in ok) / len(ok),
                "e2e_ms_p50": _percentile([m["e2e_ms"] for m in ok], 0.5),
                "e2e_ms_p90": _percentile([m["e2e_ms"] for m in ok], 0.9),
                "e2e_ms_p99": _percentile([m["e2e_ms"] for m in ok], 0.99),
                "vram_peak_mb_max": max(m["vram_peak_mb"] for m in ok),
            }
        )
    agg["competitive_context"] = {
        "wispr_flow_p99_target_ms": 700,
        "wispr_flow_medical_specialized": False,
        "nuance_dax_latency": "2-3 min async",
        "abridge_latency": "<2 min async",
        "nabla_latency": "<20 sec",
        "our_positioning": (
            "Real-time two-speaker medical with visible reasoning — "
            "Wispr Flow's speed target applied to medical dialogue with "
            "claim extraction and differential updates."
        ),
    }
    agg["per_clip"] = all_metrics

    with (out_dir / "results.json").open("w", encoding="utf-8") as o:
        json.dump(agg, o, indent=2)

    # Log everything except per_clip for readability.
    summary = {k: v for k, v in agg.items() if k != "per_clip"}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
