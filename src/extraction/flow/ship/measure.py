"""
Per-clip measurement harness for the ship pipeline.

Computes WER (raw + corrected, both default and normalized), medical-term
WER (raw + corrected), DER (when diarisation enabled), VRAM peak, and
timing breakdown.

Normalized WER lowercases and strips punctuation — Whisper sometimes emits
lowercase/no-punct output, and variant_a's reported 12.3% WER was a
different inflection of this same jiwer call, so we report both.
"""
from __future__ import annotations

import re
import string
import subprocess
import threading
import time
from typing import Optional

import jiwer

from src.extraction.flow.ship.medical_correction import MEDICAL_VOCABULARY

_PUNCT_RE = re.compile(r"[" + re.escape(string.punctuation) + "]")
_WS_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    t = text.lower()
    t = _PUNCT_RE.sub(" ", t)
    t = _WS_RE.sub(" ", t).strip()
    return t

MEDICAL_TERMS_SET: set[str] = set()
for term in MEDICAL_VOCABULARY:
    for word in term.lower().split():
        if len(word) >= 4:
            MEDICAL_TERMS_SET.add(word)


class VRAMSampler:
    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.max_mb = 0
        self.stop = False
        self.t = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self.stop:
            out = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
            )
            try:
                self.max_mb = max(self.max_mb, int(out.stdout.strip().split("\n")[0]))
            except Exception:
                pass
            time.sleep(self.interval)

    def __enter__(self) -> "VRAMSampler":
        self.t.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop = True
        self.t.join(timeout=1)


def medical_term_wer(reference: str, hypothesis: str) -> float:
    """Set-membership medical-term WER.

    Filters both reference and hypothesis to tokens that appear in
    MEDICAL_TERMS_SET, then computes jiwer WER on the filtered strings.
    Returns 0.0 if the reference has no medical terms, 1.0 if the reference
    has medical terms but the hypothesis has none.
    """
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    ref_medical = [t.strip(".,?!;:") for t in ref_tokens if t.strip(".,?!;:") in MEDICAL_TERMS_SET]
    hyp_medical = [t.strip(".,?!;:") for t in hyp_tokens if t.strip(".,?!;:") in MEDICAL_TERMS_SET]
    if not ref_medical:
        return 0.0
    if not hyp_medical:
        return 1.0
    return jiwer.wer(" ".join(ref_medical), " ".join(hyp_medical))


def compute_der(diarization_raw, ground_truth_turns: list, audio_duration: float) -> Optional[float]:
    """Compute DER from pyannote diarization against ground-truth turn list.

    Ground-truth turns are assigned equal-duration segments spanning the
    audio (we don't have per-turn timestamps in the ground truth).
    """
    if diarization_raw is None:
        return None
    try:
        from pyannote.metrics.diarization import DiarizationErrorRate
        from pyannote.core import Annotation, Segment

        gt = Annotation()
        n = len(ground_truth_turns)
        if n == 0:
            return None
        seg_dur = audio_duration / n
        for i, turn in enumerate(ground_truth_turns):
            gt[Segment(i * seg_dur, (i + 1) * seg_dur)] = turn["speaker"]

        hyp = Annotation()
        for turn, _, spk in diarization_raw.itertracks(yield_label=True):
            hyp[Segment(turn.start, turn.end)] = spk

        metric = DiarizationErrorRate()
        return float(metric(gt, hyp))
    except Exception:
        return None


def measure_one(clip: dict, pipeline) -> dict:
    with VRAMSampler() as vram:
        result = pipeline.run(clip["audio_path"])

    hyp_corrected = " ".join(s["text"] for s in result["segments"])
    hyp_raw = " ".join(s.get("raw_text", s["text"]) for s in result["segments"])
    ref = clip["full_text_reference"]

    der = compute_der(
        result["diarization_raw"],
        clip["turns"],
        result["audio_duration_sec"],
    )

    return {
        "scenario": clip["scenario"],
        "audio_duration_sec": result["audio_duration_sec"],
        "num_segments": len(result["segments"]),
        "num_corrections": len(result["corrections"]),
        "corrections": result["corrections"],
        "diarization_enabled": result["diarization_enabled"],
        "wer_raw": jiwer.wer(ref, hyp_raw) if hyp_raw else 1.0,
        "wer_corrected": jiwer.wer(ref, hyp_corrected) if hyp_corrected else 1.0,
        "wer_raw_normalized": jiwer.wer(_normalize(ref), _normalize(hyp_raw)) if hyp_raw else 1.0,
        "wer_corrected_normalized": jiwer.wer(_normalize(ref), _normalize(hyp_corrected)) if hyp_corrected else 1.0,
        "medical_term_wer_raw": medical_term_wer(ref, hyp_raw),
        "medical_term_wer_corrected": medical_term_wer(ref, hyp_corrected),
        "asr_ms": result["timings"]["asr_ms"],
        "diar_ms": result["timings"]["diar_ms"],
        "correction_ms": result["timings"]["correction_ms"],
        "e2e_ms": result["timings"]["e2e_ms"],
        "vram_peak_mb": vram.max_mb,
        "hypothesis_raw": hyp_raw,
        "hypothesis_corrected": hyp_corrected,
        "reference": ref,
        "der": der,
    }
