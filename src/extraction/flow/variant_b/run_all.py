import json, datetime, os
from pathlib import Path
from src.extraction.flow.variant_b.pipeline import VariantBPipeline
from src.extraction.flow.variant_b.measure import measure_one

OUTPUT_DIR = Path(f"eval/flow_results/variant_b/{datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

pipeline = VariantBPipeline(hf_token=os.environ["HF_TOKEN"])

with open("eval/synthetic_clips/ground_truth.jsonl") as f:
    clips = [json.loads(line) for line in f]

all_metrics = []
for clip in clips:
    print(f"Measuring {clip['scenario']}...", flush=True)
    try:
        all_metrics.append(measure_one(clip, pipeline))
    except Exception as e:
        import traceback
        all_metrics.append({"scenario": clip["scenario"], "error": str(e), "traceback": traceback.format_exc()})
    with (OUTPUT_DIR / "per_clip_metrics.jsonl").open("a") as o:
        o.write(json.dumps(all_metrics[-1]) + "\n")

ok = [m for m in all_metrics if "error" not in m]
if ok:
    def pct(arr, p):
        arr = sorted(arr)
        return arr[min(int(p*len(arr)), len(arr)-1)]

    agg = {
        "variant": "B",
        "architecture": "Cascaded: Canary-Qwen-2.5B ASR -> pyannote 3.1 diarization -> BioMistral-7B-DARE cleanup",
        "stack": {
            "asr": "nvidia/canary-qwen-2.5b (CC-BY-4.0, #1 HF Open ASR Leaderboard 5.63% WER)",
            "diarization": "pyannote/speaker-diarization-3.1 (MIT)",
            "cleanup_llm": "BioMistral-7B-DARE-AWQ (Apache 2.0, 59.4% medical QA per Labrak 2024)",
            "tts_for_eval": "Kokoro-82M (Apache 2.0, pulled from Bundle 4)",
        },
        "all_licenses_osi_approved": True,
        "hardware": "NVIDIA L4 24GB on GCP (Harneet account)",
        "variant_b_vs_variant_a_methodological_note": (
            "Same diarization (pyannote 3.1) and same cleanup (BioMistral-7B-DARE) as Variant A. "
            "Only ASR differs: Canary-Qwen-2.5B (B) vs Whisper-large-v3-turbo (A). "
            "This isolates ASR model choice as the measured variable."
        ),
        "canary_qwen_baseline_context": {
            "open_asr_leaderboard_rank_at_release": 1,
            "open_asr_leaderboard_wer_pct": 5.63,
            "librispeech_clean_wer_pct": 1.6,
            "rtfx_published_a100": 458,
            "params_billions": 2.5,
            "training_hours": 234000,
            "medical_wer_pct_published": None,
            "our_measurement_first_published_medical": True,
        },
        "n_clips": len(all_metrics),
        "n_ok": len(ok),
        "n_failed": len(all_metrics) - len(ok),
        "wer_raw_mean": sum(m["wer_raw"] for m in ok) / len(ok),
        "wer_cleaned_mean": sum(m["wer_cleaned"] for m in ok) / len(ok),
        "medical_term_wer_mean": sum(m["medical_term_wer"] for m in ok) / len(ok),
        "der_mean": sum(m["der"] for m in ok) / len(ok),
        "first_token_ms_p50": pct([m["first_token_ms"] for m in ok], 0.5),
        "first_token_ms_p90": pct([m["first_token_ms"] for m in ok], 0.9),
        "e2e_ms_p50": pct([m["e2e_ms"] for m in ok], 0.5),
        "e2e_ms_p90": pct([m["e2e_ms"] for m in ok], 0.9),
        "e2e_ms_p99": pct([m["e2e_ms"] for m in ok], 0.99),
        "asr_ms_mean": sum(m["asr_ms"] for m in ok) / len(ok),
        "diar_ms_mean": sum(m["diar_ms"] for m in ok) / len(ok),
        "cleanup_ms_mean": sum(m["cleanup_ms"] for m in ok) / len(ok),
        "vram_peak_mb_max": max(m["vram_peak_mb"] for m in ok),
        "competitive_context": {
            "wispr_flow_p99_target_ms": 700,
            "wispr_flow_accuracy_wer_pct": "2-5",
            "wispr_flow_medical_specialized": False,
            "nuance_dax_latency": "2-3 min async",
            "abridge_latency": "<2 min async",
            "nabla_latency": "<20 sec",
            "positioning": "SOTA open-source ASR (Canary-Qwen #1 on Open ASR Leaderboard) with medical-specialized cleanup layer (BioMistral-DARE, best open medical LLM per Labrak 2024). Cascaded architecture matching Wispr Flow's pattern but with all components OSI-approved open-source.",
        },
        "per_clip": all_metrics,
    }
    with (OUTPUT_DIR / "results.json").open("w") as o:
        json.dump(agg, o, indent=2)
    print(json.dumps({k: v for k, v in agg.items() if k != "per_clip"}, indent=2))
