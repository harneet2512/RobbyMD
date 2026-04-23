import json, os, time, threading, subprocess
import jiwer
from pathlib import Path
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.core import Segment, Annotation
from src.extraction.flow.variant_b.pipeline import VariantBPipeline
from src.extraction.flow.variant_b.medical_terms import MEDICAL_TERMS

class VRAMSampler:
    def __init__(self, interval=0.1):
        self.interval = interval; self.max_mb = 0; self.stop = False
        self.t = threading.Thread(target=self._loop, daemon=True)
    def _loop(self):
        while not self.stop:
            out = subprocess.run(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"],
                                 capture_output=True, text=True)
            try: self.max_mb = max(self.max_mb, int(out.stdout.strip().split("\n")[0]))
            except Exception: pass
            time.sleep(self.interval)
    def __enter__(self): self.t.start(); return self
    def __exit__(self, *a): self.stop = True; self.t.join(timeout=1)

def ground_truth_to_annotation(turns, audio_duration):
    ann = Annotation()
    n = len(turns); seg_dur = audio_duration / n
    for i, t in enumerate(turns):
        ann[Segment(i*seg_dur, (i+1)*seg_dur)] = t["speaker"]
    return ann

def pipeline_output_to_annotation(segments):
    ann = Annotation()
    for seg in segments:
        ann[Segment(seg["start"], seg["end"])] = seg["speaker"]
    return ann

def medical_term_wer(reference: str, hypothesis: str) -> float:
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    ref_medical_positions = [i for i, t in enumerate(ref_tokens) if t.strip(".,?!") in MEDICAL_TERMS]
    if not ref_medical_positions: return 0.0
    ref_medical = " ".join(ref_tokens[i] for i in ref_medical_positions)
    hyp_medical = " ".join(t for t in hyp_tokens if t.strip(".,?!") in MEDICAL_TERMS)
    if not hyp_medical.strip(): return 1.0
    return jiwer.wer(ref_medical, hyp_medical)

def measure_one(clip, pipeline):
    import soundfile as sf
    info = sf.info(clip["audio_path"])
    audio_duration = info.frames / info.samplerate

    with VRAMSampler() as vram:
        result = pipeline.run(clip["audio_path"])

    hyp_raw = result["raw_transcript"]
    hyp_cleaned = " ".join(s["cleaned_text"] for s in result["segments"])
    ref = clip["full_text_reference"]

    der_metric = DiarizationErrorRate()
    gt_ann = ground_truth_to_annotation(clip["turns"], audio_duration)
    hyp_ann = pipeline_output_to_annotation(result["segments"])
    der = der_metric(gt_ann, hyp_ann)

    return {
        "scenario": clip["scenario"],
        "audio_duration_sec": audio_duration,
        "num_speakers_detected": len(set(s["speaker"] for s in result["segments"])),
        "first_segment_ms": result["timings"]["first_segment_ms"],
        "asr_ms": result["timings"]["asr_ms"],
        "diar_ms": result["timings"]["diar_ms"],
        "cleanup_ms": result["timings"]["cleanup_ms"],
        "e2e_ms": result["timings"]["e2e_ms"],
        "vram_peak_mb": vram.max_mb,
        "wer_raw": jiwer.wer(ref, hyp_raw),
        "wer_cleaned": jiwer.wer(ref, hyp_cleaned),
        "medical_term_wer": medical_term_wer(ref, hyp_cleaned),
        "der": der,
        "hypothesis_raw": hyp_raw,
        "hypothesis_cleaned": hyp_cleaned,
        "reference": ref,
    }
