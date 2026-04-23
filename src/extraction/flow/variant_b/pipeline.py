import time, os, requests
from pathlib import Path
from pyannote.audio import Pipeline as DiarPipeline
import nemo.collections.asr as nemo_asr

CLEANUP_SYSTEM = (
    "You are a medical transcript cleaner. Given a raw ASR segment, remove filler "
    "words and disfluencies, normalize medical terminology to standard forms, fix "
    "obvious transcription errors in drug names and anatomical terms, and preserve "
    "every clinically relevant claim. Return ONLY the cleaned text, no explanation."
)

class VariantBPipeline:
    def __init__(self, hf_token: str, cleanup_url: str = "http://127.0.0.1:8001/v1"):
        # Canary-Qwen-2.5B ASR via NeMo
        self.asr = nemo_asr.models.EncDecMultiTaskModel.from_pretrained("nvidia/canary-qwen-2.5b")
        self.asr = self.asr.eval()
        # Canary-Qwen uses ASR mode by default (LLM mode is separate)
        decode_cfg = self.asr.cfg.decoding
        decode_cfg.beam.beam_size = 1  # greedy for speed per NVIDIA's reference eval
        self.asr.change_decoding_strategy(decode_cfg)
        import torch
        self.asr = self.asr.to(torch.device("cuda"))

        # pyannote 3.1 diarization
        self.diar = DiarPipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )
        self.diar.to("cuda")

        self.cleanup_url = cleanup_url

    def transcribe(self, audio_path: str):
        t = time.perf_counter()
        # Canary-Qwen transcribe API: returns list of Hypothesis objects
        preds = self.asr.transcribe([audio_path], batch_size=1, task="asr", source_lang="en", target_lang="en", pnc=True)
        dt = (time.perf_counter() - t) * 1000
        # preds is list of strings or Hypothesis objects depending on NeMo version
        if hasattr(preds[0], "text"):
            text = preds[0].text
        else:
            text = str(preds[0])
        return text, dt

    def diarize(self, audio_path: str):
        return self.diar(audio_path, num_speakers=2)

    def cleanup_segment(self, raw_text: str) -> str:
        payload = {
            "model": "biomistral-7b-dare",
            "messages": [
                {"role": "system", "content": CLEANUP_SYSTEM},
                {"role": "user", "content": raw_text},
            ],
            "temperature": 0.0,
            "max_tokens": max(64, len(raw_text.split()) * 3),
        }
        r = requests.post(f"{self.cleanup_url}/chat/completions", json=payload, timeout=30)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()

    def align_speakers_from_raw(self, raw_transcript: str, diarization, audio_duration: float):
        """
        Canary-Qwen returns a full transcript without word-level timestamps by default.
        We approximate alignment by splitting the transcript proportionally to pyannote's
        speaker-turn durations — same approach as Variant A's fallback path, so the two
        pipelines handle alignment identically.
        """
        turns = [(turn.start, turn.end, spk)
                 for turn, _, spk in diarization.itertracks(yield_label=True)]
        if not turns:
            return [{"speaker": "UNKNOWN", "start": 0, "end": audio_duration, "raw_text": raw_transcript}]

        words = raw_transcript.split()
        total_speech = sum(e - s for s, e, _ in turns)
        segments = []
        word_idx = 0
        for s, e, spk in turns:
            share = (e - s) / total_speech if total_speech > 0 else 1.0 / len(turns)
            n_words = max(1, int(round(len(words) * share)))
            chunk = " ".join(words[word_idx : word_idx + n_words])
            word_idx += n_words
            if chunk.strip():
                segments.append({"speaker": spk, "start": s, "end": e, "raw_text": chunk})
        if word_idx < len(words) and segments:
            segments[-1]["raw_text"] += " " + " ".join(words[word_idx:])
        return segments

    def run(self, audio_path: str) -> dict:
        import soundfile as sf
        info = sf.info(audio_path)
        audio_duration = info.frames / info.samplerate

        timings = {}
        t0 = time.perf_counter()

        # Stage 1: Canary-Qwen ASR
        raw_transcript, asr_ms = self.transcribe(audio_path)
        timings["asr_ms"] = asr_ms
        timings["first_token_ms"] = asr_ms  # Canary returns full transcript in one call

        # Stage 2: pyannote diarization (runs in parallel on paper, sequential here)
        t_d = time.perf_counter()
        diarization = self.diarize(audio_path)
        timings["diar_ms"] = (time.perf_counter() - t_d) * 1000

        # Stage 3: align speakers to transcript
        aligned = self.align_speakers_from_raw(raw_transcript, diarization, audio_duration)

        # Stage 4: BioMistral-7B-DARE cleanup per segment
        t_c = time.perf_counter()
        for seg in aligned:
            seg["cleaned_text"] = self.cleanup_segment(seg["raw_text"])
        timings["cleanup_ms"] = (time.perf_counter() - t_c) * 1000

        timings["e2e_ms"] = (time.perf_counter() - t0) * 1000
        return {"raw_transcript": raw_transcript, "segments": aligned, "timings": timings, "diarization": diarization}
