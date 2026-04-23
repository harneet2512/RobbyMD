"""Variant A cascaded ASR pipeline.

Whisper-large-v3-turbo (MIT, via faster-whisper) → pyannote
speaker-diarization-community-1 (MIT code + CC-BY-4.0 weights,
pyannote.audio >=4.0) → midpoint-based speaker alignment →
BioMistral-7B-DARE cleanup (Apache-2.0, via a local vLLM OpenAI-compatible
endpoint).

All component licences are OSI-approved or open-data per
docs/decisions/licensing_clarifications.md §Q2.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

import requests
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline as DiarPipeline

MEDICAL_INITIAL_PROMPT = (
    "Clinical conversation. Common terms: chest pain, dyspnea, palpitations, "
    "myocardial infarction, angina, pulmonary embolism, pneumonia, cholecystitis, "
    "appendicitis, migraine, subarachnoid hemorrhage, vasovagal, orthostatic, BPPV. "
    "Drugs: aspirin, nitroglycerin, ibuprofen, metformin, amlodipine, atorvastatin, "
    "lisinopril, metoprolol, clopidogrel. "
    "Decision rules: HEART score, TIMI risk, Wells criteria, PERC rule."
)

# Subsequence-only prompt: rewritten 2026-04-23 after the first run showed
# the cleanup LLM compressing patient-voice narrative into clinical-note
# shorthand (e.g. "I've been having this pressure in my chest" →
# "Pressure in chest started two hours ago") and doubling WER on
# medically-dense clips. The new prompt disallows paraphrase, reorder, and
# terminology normalisation; the filler-regex fast-path below skips the
# LLM entirely when no disfluency is present.
CLEANUP_SYSTEM_PROMPT = (
    "You are a transcript cleaner. Your output must be a subsequence of "
    "the input: you may only delete filler tokens (uh, um, er, ah, like, "
    "you know) and obvious duplicate words. You must NOT paraphrase, NOT "
    "reorder, NOT normalise terminology, NOT expand abbreviations, NOT "
    "add clarifications, NOT add punctuation that was absent. If the "
    "input has no filler, return it byte-for-byte unchanged. Output only "
    "the cleaned text, nothing else."
)

# Regex fast-path. If the segment has no filler, we skip the LLM call and
# return raw_text unchanged. Kokoro-rendered TTS has no spontaneous
# disfluencies, so this eliminates most of the cleanup-stage cost on this
# test set. Real clinical audio will hit the LLM on a larger fraction.
_FILLER_RE = re.compile(
    r"\b(?:uh+|um+|er+|ah+|hmm+|mmm+|like|you\s+know)\b",
    re.IGNORECASE,
)

DIARISER_MODEL = "pyannote/speaker-diarization-community-1"


@dataclass
class AlignedSegment:
    speaker: str
    start: float
    end: float
    raw_text: str
    cleaned_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "speaker": self.speaker,
            "start": self.start,
            "end": self.end,
            "raw_text": self.raw_text,
            "cleaned_text": self.cleaned_text,
        }


class VariantAPipeline:
    def __init__(
        self,
        hf_token: str,
        cleanup_url: str = "http://127.0.0.1:8000/v1",
        cleanup_model: str = "biomistral-7b-dare",
        enable_diarisation: bool = True,
    ) -> None:
        # Diarisation is re-enabled by default now that vLLM 0.8+ carries
        # torch 2.8, which pyannote.audio 4.0 (community-1 diariser) needs.
        # If the caller is still on the old torch-2.4 / vLLM-0.6 stack, they
        # can pass enable_diarisation=False to fall back to the alternating
        # heuristic.
        self.whisper = WhisperModel(
            "large-v3-turbo",
            device="cuda",
            compute_type="float16",
        )
        self.enable_diarisation = enable_diarisation
        self.diar = None
        if enable_diarisation:
            self.diar = DiarPipeline.from_pretrained(
                DIARISER_MODEL,
                use_auth_token=hf_token,
            )
            import torch

            self.diar.to(torch.device("cuda"))
        self.cleanup_url = cleanup_url.rstrip("/")
        self.cleanup_model = cleanup_model

    def transcribe(
        self, audio_path: str, word_timestamps: bool = False
    ) -> tuple[list, float]:
        """Return (segments_list, first_segment_after_ingest_ms).

        faster-whisper processes the whole file before yielding anything: the
        first generator yield corresponds to the first 30-second chunk being
        fully transcribed, not to a streaming first-word event. For a real
        streaming-first-word latency (comparable to Wispr Flow's ~700 ms p99)
        the audio would need to be pre-chunked and fed in sequentially — that
        measurement is not done here and is reported as `None` upstream.

        word_timestamps=False by default (previous default was True, which
        adds 5-10 % latency via a second alignment pass and nothing consumes
        the word-level offsets downstream).
        """
        t0 = time.perf_counter()
        generator, _info = self.whisper.transcribe(
            audio_path,
            beam_size=5,
            initial_prompt=MEDICAL_INITIAL_PROMPT,
            word_timestamps=word_timestamps,
            language="en",
        )
        segments: list = []
        first_segment_ms: float | None = None
        for seg in generator:
            if first_segment_ms is None:
                first_segment_ms = (time.perf_counter() - t0) * 1000.0
            segments.append(seg)
        if first_segment_ms is None:
            first_segment_ms = (time.perf_counter() - t0) * 1000.0
        return segments, first_segment_ms

    def diarize(self, audio_path: str):
        return self.diar(audio_path, num_speakers=2)

    @staticmethod
    def align_speakers(whisper_segments, diarization) -> list[AlignedSegment]:
        turns = [
            (turn.start, turn.end, spk)
            for turn, _, spk in diarization.itertracks(yield_label=True)
        ]
        aligned: list[AlignedSegment] = []
        for seg in whisper_segments:
            mid = (seg.start + seg.end) / 2.0
            speaker = "UNKNOWN"
            for s, e, spk in turns:
                if s <= mid <= e:
                    speaker = spk
                    break
            aligned.append(
                AlignedSegment(
                    speaker=speaker,
                    start=seg.start,
                    end=seg.end,
                    raw_text=seg.text.strip(),
                )
            )
        return aligned

    def cleanup_segment(self, raw_text: str) -> str:
        """Cleanup a single segment. Skips the LLM when there is no filler.

        Returns the raw_text unchanged if `_FILLER_RE` finds no match —
        previous run's damage was concentrated on clean segments being
        rewritten into medical-note shorthand, so a regex gate eliminates
        that failure mode entirely. When filler IS present, the subsequence-
        only system prompt (merged into the user message because Mistral's
        chat template rejects the system role) bounds rewriting.

        max_tokens is capped at word-count + 4 (the subsequence-only
        constraint means valid output is never longer than the input).
        """
        if not raw_text.strip():
            return ""
        if not _FILLER_RE.search(raw_text):
            return raw_text
        payload = {
            "model": self.cleanup_model,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        CLEANUP_SYSTEM_PROMPT
                        + "\n\nTranscript segment to clean:\n"
                        + raw_text
                    ),
                },
            ],
            "temperature": 0.0,
            "max_tokens": max(16, len(raw_text.split()) + 4),
        }
        r = requests.post(
            f"{self.cleanup_url}/chat/completions", json=payload, timeout=60
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()

    def run(self, audio_path: str) -> dict[str, Any]:
        timings: dict[str, float | None] = {}
        t_total = time.perf_counter()

        t = time.perf_counter()
        whisper_segs, first_segment_ms = self.transcribe(audio_path)
        timings["asr_ms"] = (time.perf_counter() - t) * 1000.0
        # Renamed from `first_token_ms` to truthfully describe the metric.
        # faster-whisper yields after the first 30s audio chunk is processed,
        # so this is first-segment latency after full file ingest — not a
        # streaming first-word metric. Kept alongside as `first_token_ms`
        # (value = None) to flag the intended streaming measurement is
        # deferred.
        timings["first_segment_after_ingest_ms"] = first_segment_ms
        timings["streaming_first_word_ms"] = None  # deferred to follow-up

        diarization = None
        if self.enable_diarisation:
            t = time.perf_counter()
            diarization = self.diarize(audio_path)
            timings["diar_ms"] = (time.perf_counter() - t) * 1000.0
            aligned = self.align_speakers(whisper_segs, diarization)
        else:
            timings["diar_ms"] = 0.0
            aligned = [
                AlignedSegment(
                    speaker="DOCTOR" if i % 2 == 0 else "PATIENT",
                    start=seg.start,
                    end=seg.end,
                    raw_text=seg.text.strip(),
                )
                for i, seg in enumerate(whisper_segs)
            ]

        t = time.perf_counter()
        cleanup_llm_calls = 0
        for seg in aligned:
            if not seg.raw_text:
                seg.cleaned_text = ""
                continue
            had_filler = bool(_FILLER_RE.search(seg.raw_text))
            seg.cleaned_text = self.cleanup_segment(seg.raw_text)
            if had_filler:
                cleanup_llm_calls += 1
        timings["cleanup_ms"] = (time.perf_counter() - t) * 1000.0

        timings["e2e_ms"] = (time.perf_counter() - t_total) * 1000.0

        return {
            "segments": [s.to_dict() for s in aligned],
            "timings": timings,
            "diarization": diarization,
            "diarisation_enabled": self.enable_diarisation,
            "cleanup_llm_calls": cleanup_llm_calls,
            "cleanup_total_segments": len(aligned),
        }
