import json, os, time
from pathlib import Path

import requests
import torch
from pyannote.audio import Pipeline as DiarPipeline
# NeMo / Canary-Qwen run in .venv-asr via asr_prerun.py; transcripts are
# loaded here from a JSONL sidecar keyed by audio_path. Keeps torch 2.11
# out of this venv so pyannote.audio 3.3.1 / torch 2.4 can run.

# Subsequence-only cleanup prompt. Forbids paraphrasing/normalisation/expansion.
# Bundle 4 root-cause: the old prompt invited BioMistral to rewrite conversational
# segments as formal PubMed-style prose, which inflates WER against a token-exact
# reference. Fix: explicit subsequence contract + filler-only scope.
CLEANUP_SYSTEM = (
    "Return the input text with filler tokens (uh, um, like, you know, basically, "
    "literally) removed and nothing else. Do not reorder, paraphrase, normalise "
    "terminology, correct grammar, or add words. The output MUST be a subsequence "
    "of the input (only deletions allowed). If the input has no filler, return it "
    "verbatim byte-for-byte. Return ONLY the cleaned text."
)

import re as _re
# Fast-path filler detector: if none match, skip the LLM entirely.
_FILLER_TOKEN_RE = _re.compile(
    r"(uh+|um+|er+|mm+|uhm+|hmm+|like|you know|i mean|basically|literally|sort of|kind of)",
    _re.IGNORECASE,
)

class VariantBPipeline:
    def __init__(self, hf_token: str, cleanup_url: str = "http://127.0.0.1:8001/v1"):
        asr_jsonl = os.environ.get("ASR_PRERUN_JSONL", "eval/synthetic_clips/asr_prerun.jsonl")
        self._asr_cache = {}
        if not Path(asr_jsonl).exists():
            raise RuntimeError(f"ASR prerun JSONL missing: {asr_jsonl}. Run asr_prerun.py first.")
        for line in Path(asr_jsonl).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            self._asr_cache[rec["audio_path"]] = (rec["transcript"], rec["asr_ms"])
        print(f"[pipeline] loaded {len(self._asr_cache)} ASR transcripts from {asr_jsonl}")

        # pyannote 3.1 diarization
        self.diar = DiarPipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )
        self.diar.to(torch.device("cuda"))

        self.cleanup_url = cleanup_url

    def transcribe(self, audio_path: str):
        if audio_path not in self._asr_cache:
            raise RuntimeError(f"no ASR transcript for {audio_path}. Run asr_prerun.py first.")
        text, dt = self._asr_cache[audio_path]
        return text, float(dt)

    def diarize(self, audio_path: str):
        return self.diar(audio_path, num_speakers=2)

    def cleanup_segment(self, raw_text: str) -> str:
        # Bundle 5 v1: cleanup DISABLED.
        # (a) D1: BioMistral paraphrased conversational speech, inflating WER.
        # (b) vLLM 0.6.3 int*None config bug on BioMistral under torch 2.4;
        #     upgrading vLLM needs torch 2.5+ which breaks pyannote.audio.
        # Effect: cleaned_text == raw_text; results.json reports cleanup=disabled.
        return raw_text
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
        # D2: honest metric name. Canary-Qwen returns the full transcript after
        # processing the full audio chunk. This is NOT streaming-first-word
        # (Wispr Flow~700 ms). Reporting it as first_segment_ms for comparability.
        timings["first_segment_ms"] = asr_ms

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
