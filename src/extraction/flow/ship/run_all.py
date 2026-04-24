"""
Run the ship pipeline across 7 clips (6 original + unseen pediatric),
aggregate, write results.json + per_clip_metrics.jsonl to
eval/flow_results/ship/<UTC-timestamp>/.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import statistics
from pathlib import Path

from src.extraction.flow.ship.measure import measure_one
from src.extraction.flow.ship.pipeline import ShipPipeline


def _pct(arr, p: float):
    if not arr:
        return None
    arr = sorted(arr)
    return arr[min(int(p * len(arr)), len(arr) - 1)]


def _safe_mean(arr):
    return sum(arr) / len(arr) if arr else None


def _safe_stdev(arr):
    return statistics.stdev(arr) if len(arr) >= 2 else 0.0


def main() -> None:
    ts = _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(f"eval/flow_results/ship/{ts}")
    output_dir.mkdir(parents=True, exist_ok=True)

    hf_token = os.environ.get("HF_TOKEN")
    pipeline = ShipPipeline(hf_token=hf_token, enable_diarization=True)

    gt_path = Path("eval/synthetic_clips/ground_truth_ship.jsonl")
    with gt_path.open() as f:
        clips = [json.loads(line) for line in f if line.strip()]

    all_metrics: list[dict] = []
    per_clip_out = output_dir / "per_clip_metrics.jsonl"

    for clip in clips:
        print(f"\nmeasuring {clip['scenario']}...", flush=True)
        try:
            m = measure_one(clip, pipeline)
            all_metrics.append(m)
            print(
                f"  WER raw={m['wer_raw']:.3f} corrected={m['wer_corrected']:.3f} "
                f"med_raw={m['medical_term_wer_raw']:.3f} med_corr={m['medical_term_wer_corrected']:.3f} "
                f"e2e={m['e2e_ms']:.0f}ms corrections={m['num_corrections']}",
                flush=True,
            )
        except Exception as exc:
            import traceback
            err = {
                "scenario": clip["scenario"],
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            all_metrics.append(err)
            print(f"  FAILED: {exc}", flush=True)

        with per_clip_out.open("a") as o:
            o.write(json.dumps(all_metrics[-1], default=str) + "\n")

    ok = [m for m in all_metrics if "error" not in m]
    if not ok:
        print("\nALL CLIPS FAILED — no aggregate computed", flush=True)
        return

    original = [m for m in ok if m["scenario"] != "pediatric_fever_rash"]
    unseen = [m for m in ok if m["scenario"] == "pediatric_fever_rash"]

    agg = {
        "pipeline": (
            "Whisper-large-v3-turbo + medical hotwords + fuzzy medical "
            "correction (rapidfuzz, 200-term vocab) + pyannote community-1"
        ),
        "changes_from_variant_a": [
            "ADDED: hotwords parameter (medical terms biased at beam-search decoding)",
            "ADDED: fuzzy medical correction (rapidfuzz edit-distance, zero VRAM)",
            "REMOVED: BioMistral-7B-DARE cleanup (+1.6pp WER regression, rejected)",
            "CHANGED: VAD min_silence_duration_ms=500 (reduces turn-deletion)",
            "CHANGED: pyannote community-1 (replaces blocked pyannote 4.0)",
        ],
        "n_clips_total": len(all_metrics),
        "n_clips_ok": len(ok),
        "n_clips_failed": len(all_metrics) - len(ok),
        "n_clips_original": len(original),
        "n_clips_unseen": len(unseen),
        "original_6": {},
        "unseen_pediatric": {},
        "vs_variant_a": {
            # Default jiwer (case+punct sensitive) numbers, as variant_a reported.
            "variant_a_wer_raw_mean": 0.1229,
            "variant_a_wer_cleaned_mean": 0.1390,
            "variant_a_cleaned_minus_raw_pp": 0.0161,
            # Normalized (lowercase + strip punct) — the fair apples-to-apples
            # baseline. Computed via scripts/renormalize_wer.py on
            # eval/flow_results/variant_a/20260423T194716Z/per_clip_metrics.jsonl.
            "variant_a_wer_raw_normalized_mean": 0.0356,
            "variant_a_wer_cleaned_normalized_mean": 0.0495,
            "variant_a_cleaned_minus_raw_pp_normalized": 0.0139,
            "variant_a_medical_term_wer_mean": 0.1861,
            "variant_a_e2e_p50_ms": 6459,
            "variant_a_vram_peak_mb": 17052,
            "variant_a_cleanup_method": "BioMistral-7B-DARE-AWQ (REJECTED: +1.6pp)",
            "variant_a_diarization": "disabled",
        },
        "per_clip": all_metrics,
    }

    if original:
        wer_raw = [m["wer_raw"] for m in original]
        wer_corr = [m["wer_corrected"] for m in original]
        wer_raw_n = [m.get("wer_raw_normalized", m["wer_raw"]) for m in original]
        wer_corr_n = [m.get("wer_corrected_normalized", m["wer_corrected"]) for m in original]
        mwer_raw = [m["medical_term_wer_raw"] for m in original]
        mwer_corr = [m["medical_term_wer_corrected"] for m in original]
        e2e = [m["e2e_ms"] for m in original]
        asr = [m["asr_ms"] for m in original]
        corr_ms = [m["correction_ms"] for m in original]
        diar_ms = [m["diar_ms"] for m in original]
        vram = [m["vram_peak_mb"] for m in original]
        ders = [m["der"] for m in original if m["der"] is not None]

        agg["original_6"] = {
            "wer_raw_mean": _safe_mean(wer_raw),
            "wer_raw_stdev": _safe_stdev(wer_raw),
            "wer_corrected_mean": _safe_mean(wer_corr),
            "wer_corrected_stdev": _safe_stdev(wer_corr),
            "correction_delta_pp": _safe_mean([c - r for c, r in zip(wer_corr, wer_raw)]),
            # Normalized WER (lowercase + strip punct) — the fair
            # apples-to-apples comparison vs variant_a, since faster-whisper
            # emits case-inconsistent output under hotwords+VAD on some clips.
            "wer_raw_normalized_mean": _safe_mean(wer_raw_n),
            "wer_raw_normalized_stdev": _safe_stdev(wer_raw_n),
            "wer_corrected_normalized_mean": _safe_mean(wer_corr_n),
            "wer_corrected_normalized_stdev": _safe_stdev(wer_corr_n),
            "correction_delta_pp_normalized": _safe_mean(
                [c - r for c, r in zip(wer_corr_n, wer_raw_n)]
            ),
            "medical_term_wer_raw_mean": _safe_mean(mwer_raw),
            "medical_term_wer_corrected_mean": _safe_mean(mwer_corr),
            "medical_term_correction_delta_pp": _safe_mean(
                [c - r for c, r in zip(mwer_corr, mwer_raw)]
            ),
            "still_inverted": (
                _safe_mean(mwer_corr) > _safe_mean(wer_corr)
                if mwer_corr and wer_corr
                else None
            ),
            "e2e_ms_p50": _pct(e2e, 0.5),
            "e2e_ms_p90": _pct(e2e, 0.9),
            "asr_ms_mean": _safe_mean(asr),
            "correction_ms_mean": _safe_mean(corr_ms),
            "diar_ms_mean": _safe_mean(diar_ms),
            "vram_peak_mb_max": max(vram) if vram else None,
            "total_corrections_made": sum(m["num_corrections"] for m in original),
            "der_mean": _safe_mean(ders) if ders else None,
            "der_n": len(ders),
        }

    if unseen:
        u = unseen[0]
        agg["unseen_pediatric"] = {
            "wer_raw": u["wer_raw"],
            "wer_corrected": u["wer_corrected"],
            "medical_term_wer_raw": u["medical_term_wer_raw"],
            "medical_term_wer_corrected": u["medical_term_wer_corrected"],
            "e2e_ms": u["e2e_ms"],
            "vram_peak_mb": u["vram_peak_mb"],
            "der": u["der"],
            "num_corrections": u["num_corrections"],
            "delta_wer_corrected_vs_original": (
                u["wer_corrected"] - agg["original_6"]["wer_corrected_mean"]
                if agg["original_6"]
                else None
            ),
        }
    else:
        agg["unseen_pediatric"] = {"status": "not run or failed"}

    with (output_dir / "results.json").open("w") as o:
        json.dump(agg, o, indent=2, default=str)

    print("\n" + "=" * 60, flush=True)
    print("SHIP PIPELINE RESULTS", flush=True)
    print("=" * 60, flush=True)
    if agg["original_6"]:
        o6 = agg["original_6"]
        print(f"Original 6 clips:", flush=True)
        print(f"  WER raw (default):      {o6['wer_raw_mean']:.1%} [case+punct sensitive]", flush=True)
        print(f"  WER raw (normalized):   {o6['wer_raw_normalized_mean']:.1%} [vs variant_a 3.56%]", flush=True)
        print(f"  WER corrected (norm):   {o6['wer_corrected_normalized_mean']:.1%} [vs variant_a 4.95%]", flush=True)
        print(f"  Correction delta norm:  {o6['correction_delta_pp_normalized']:+.2%}pp [vs variant_a +1.39pp]", flush=True)
        print(f"  Med-term WER raw:       {o6['medical_term_wer_raw_mean']:.1%} [vs variant_a 18.6%]", flush=True)
        print(f"  Med-term corrected:     {o6['medical_term_wer_corrected_mean']:.1%}", flush=True)
        print(f"  Still inverted:         {o6['still_inverted']}", flush=True)
        print(f"  E2E p50:                {o6['e2e_ms_p50']:.0f}ms [vs variant_a 6459ms]", flush=True)
        print(f"  VRAM peak:              {o6['vram_peak_mb_max']}MB [vs variant_a 17052MB]", flush=True)
        print(f"  DER mean:               {o6['der_mean']} (n={o6['der_n']})", flush=True)
        print(f"  Total corrections:      {o6['total_corrections_made']}", flush=True)
    if unseen:
        u = agg["unseen_pediatric"]
        print(f"\nUnseen pediatric:", flush=True)
        print(f"  WER corrected:      {u['wer_corrected']:.1%}", flush=True)
        print(f"  Med-term corrected: {u['medical_term_wer_corrected']:.1%}", flush=True)
        if u["delta_wer_corrected_vs_original"] is not None:
            print(
                f"  Delta vs original:  {u['delta_wer_corrected_vs_original']:+.1%}pp",
                flush=True,
            )
    print("=" * 60, flush=True)
    print(f"results: {output_dir / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
