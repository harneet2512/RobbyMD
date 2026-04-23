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


def _strip(tok: str) -> str:
    return tok.lower().strip(_PUNCT)


def medical_term_wer(reference: str, hypothesis: str) -> float:
    """Error rate on reference tokens that are in MEDICAL_TERMS.

    Uses jiwer's word-level alignment on the FULL reference and hypothesis
    (not a pre-subsetted string), then counts errors restricted to
    reference positions whose normalised token is in MEDICAL_TERMS.

    This replaces an earlier implementation that subsetted both sides to
    only medical tokens and called `jiwer.wer`. That approach lost
    positional information: reorders counted as errors, non-medical
    substitutions silently deleted adjacent medical positions, and
    inserted medical tokens could falsely inflate the count.

    Returns a float in [0.0, 1.0]. Returns 0.0 if the reference has no
    medical tokens.
    """
    ref_tokens_raw = reference.split()
    hyp_tokens_raw = hypothesis.split()
    if not ref_tokens_raw:
        return 0.0

    ref_norm = [_strip(t) for t in ref_tokens_raw]
    hyp_norm = [_strip(t) for t in hyp_tokens_raw]

    medical_positions = [i for i, t in enumerate(ref_norm) if t in MEDICAL_TERMS]
    if not medical_positions:
        return 0.0

    # Align. jiwer.process_words accepts list-of-lists for ref/hyp to skip
    # its own tokenisation.
    try:
        out = jiwer.process_words([ref_norm], [hyp_norm])
    except Exception:
        # Fallback if jiwer API differs: treat all medical positions as errors.
        return 1.0

    # Walk the alignment chunks and mark which ref positions got a direct
    # equal match. Substitute / delete / insert leave the position unmatched
    # (for "insert" it's a hyp-only chunk and doesn't cover any ref position).
    n_ref = len(ref_norm)
    matched = [False] * n_ref
    try:
        chunks = out.alignments[0]
    except (AttributeError, IndexError):
        return 1.0
    for ch in chunks:
        ch_type = getattr(ch, "type", None)
        if ch_type != "equal":
            continue
        ref_start = getattr(ch, "ref_start_idx", None)
        ref_end = getattr(ch, "ref_end_idx", None)
        if ref_start is None or ref_end is None:
            continue
        for i in range(ref_start, min(ref_end, n_ref)):
            matched[i] = True

    errors = sum(1 for p in medical_positions if not matched[p])
    return errors / len(medical_positions)


def measure_one(clip: dict, pipeline: VariantAPipeline) -> dict[str, Any]:
    info = sf.info(clip["audio_path"])
    audio_duration = info.frames / info.samplerate

    with VRAMSampler() as vram:
        result = pipeline.run(clip["audio_path"])

    hyp_raw = " ".join(s["raw_text"] for s in result["segments"])
    hyp_cleaned = " ".join(s["cleaned_text"] for s in result["segments"])
    ref = clip["full_text_reference"]

    if result.get("diarisation_enabled"):
        der_metric = DiarizationErrorRate()
        gt_ann = _ground_truth_annotation(clip["turns"], audio_duration)
        hyp_ann = _hyp_annotation(result["segments"])
        try:
            der = float(der_metric(gt_ann, hyp_ann))
        except Exception as e:
            der = None
            print(f"DER computation failed for {clip['scenario']}: {e}")
    else:
        der = None

    return {
        "scenario": clip["scenario"],
        "audio_duration_sec": audio_duration,
        "num_speakers_detected": len({s["speaker"] for s in result["segments"]}),
        "first_segment_after_ingest_ms": result["timings"]["first_segment_after_ingest_ms"],
        "streaming_first_word_ms": result["timings"]["streaming_first_word_ms"],
        "asr_ms": result["timings"]["asr_ms"],
        "diar_ms": result["timings"]["diar_ms"],
        "cleanup_ms": result["timings"]["cleanup_ms"],
        "e2e_ms": result["timings"]["e2e_ms"],
        "vram_peak_mb": vram.max_mb,
        "wer_raw": float(jiwer.wer(ref, hyp_raw)) if hyp_raw else 1.0,
        "wer_cleaned": float(jiwer.wer(ref, hyp_cleaned)) if hyp_cleaned else 1.0,
        "medical_term_wer": medical_term_wer(ref, hyp_cleaned),
        "der": der,
        "diarisation_enabled": result.get("diarisation_enabled", False),
        "cleanup_llm_calls": result.get("cleanup_llm_calls", 0),
        "cleanup_total_segments": result.get("cleanup_total_segments", 0),
        "hypothesis_raw": hyp_raw,
        "hypothesis_cleaned": hyp_cleaned,
        "reference": ref,
    }
