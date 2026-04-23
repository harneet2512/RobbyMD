"""Per-clip measurement: latency, WER, DER, medical-term WER, VRAM peak."""

from __future__ import annotations

import subprocess
import threading
import time
from typing import Any

import jiwer
import soundfile as sf
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate

from src.extraction.flow.variant_a.medical_terms import MEDICAL_TERMS
from src.extraction.flow.variant_a.pipeline import VariantAPipeline


class VRAMSampler:
    def __init__(self, interval_sec: float = 0.1) -> None:
        self.interval = interval_sec
        self.max_mb = 0
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._stop:
            try:
                out = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=2,
                )
                mb = int(out.stdout.strip().splitlines()[0])
                if mb > self.max_mb:
                    self.max_mb = mb
            except Exception:
                pass
            time.sleep(self.interval)

    def __enter__(self) -> "VRAMSampler":
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._stop = True
        self._thread.join(timeout=2)


def _ground_truth_annotation(turns: list[dict], audio_duration: float) -> Annotation:
    ann = Annotation()
    n = len(turns)
    if n == 0:
        return ann
    seg_dur = audio_duration / n
    for i, t in enumerate(turns):
        ann[Segment(i * seg_dur, (i + 1) * seg_dur)] = t["speaker"]
    return ann


def _hyp_annotation(segments: list[dict]) -> Annotation:
    ann = Annotation()
    for i, seg in enumerate(segments):
        if seg["end"] > seg["start"]:
            ann[Segment(seg["start"], seg["end"]), f"seg{i}"] = seg["speaker"]
    return ann


_PUNCT = ".,?!;:\"'()"


def medical_term_wer(reference: str, hypothesis: str) -> float:
    ref_tokens = [t.lower().strip(_PUNCT) for t in reference.split()]
    hyp_tokens = [t.lower().strip(_PUNCT) for t in hypothesis.split()]
    ref_medical = [t for t in ref_tokens if t in MEDICAL_TERMS]
    if not ref_medical:
        return 0.0
    hyp_medical = [t for t in hyp_tokens if t in MEDICAL_TERMS]
    if not hyp_medical:
        return 1.0
    return float(jiwer.wer(" ".join(ref_medical), " ".join(hyp_medical)))


def measure_one(clip: dict, pipeline: VariantAPipeline) -> dict[str, Any]:
    info = sf.info(clip["audio_path"])
    audio_duration = info.frames / info.samplerate

    with VRAMSampler() as vram:
        result = pipeline.run(clip["audio_path"])

    hyp_raw = " ".join(s["raw_text"] for s in result["segments"])
    hyp_cleaned = " ".join(s["cleaned_text"] for s in result["segments"])
    ref = clip["full_text_reference"]

    der_metric = DiarizationErrorRate()
    gt_ann = _ground_truth_annotation(clip["turns"], audio_duration)
    hyp_ann = _hyp_annotation(result["segments"])
    try:
        der = float(der_metric(gt_ann, hyp_ann))
    except Exception as e:
        der = -1.0
        print(f"DER computation failed for {clip['scenario']}: {e}")

    return {
        "scenario": clip["scenario"],
        "audio_duration_sec": audio_duration,
        "num_speakers_detected": len({s["speaker"] for s in result["segments"]}),
        "first_token_ms": result["timings"]["first_token_ms"],
        "asr_ms": result["timings"]["asr_ms"],
        "diar_ms": result["timings"]["diar_ms"],
        "cleanup_ms": result["timings"]["cleanup_ms"],
        "e2e_ms": result["timings"]["e2e_ms"],
        "vram_peak_mb": vram.max_mb,
        "wer_raw": float(jiwer.wer(ref, hyp_raw)) if hyp_raw else 1.0,
        "wer_cleaned": float(jiwer.wer(ref, hyp_cleaned)) if hyp_cleaned else 1.0,
        "medical_term_wer": medical_term_wer(ref, hyp_cleaned),
        "der": der,
        "hypothesis_raw": hyp_raw,
        "hypothesis_cleaned": hyp_cleaned,
        "reference": ref,
    }
