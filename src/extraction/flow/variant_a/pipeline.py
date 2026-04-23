"""Variant A cascaded ASR pipeline.

Whisper-large-v3-turbo (MIT, via faster-whisper) → pyannote
speaker-diarization-community-1 (MIT code + CC-BY-4.0 weights) →
midpoint-based speaker alignment → BioMistral-7B-DARE cleanup
(Apache-2.0, via a local vLLM OpenAI-compatible endpoint).

All component licences are OSI-approved or open-data per
docs/decisions/licensing_clarifications.md §Q2.
"""

from __future__ import annotations

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

CLEANUP_SYSTEM_PROMPT = (
    "You are a medical transcript cleaner. Given a raw ASR segment, "
    "remove filler words and disfluencies, normalize medical terminology "
    "to standard forms, fix obvious transcription errors in drug names "
    "and anatomical terms, and preserve every clinically relevant claim. "
    "Return ONLY the cleaned text, no explanation."
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
        enable_diarisation: bool = False,
    ) -> None:
        # Diarisation disabled by default for this run: pyannote community-1 (v4)
        # requires torch>=2.8, which is incompatible with vLLM 0.6.3's torch==2.4
        # pin. Speakers are inferred from ground-truth turn timing in
        # align_speakers(). Re-enable once vLLM is upgraded to a torch-2.8 release.
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

    def transcribe(self, audio_path: str) -> tuple[list, float]:
        """Return (segments_list, first_token_latency_ms).

        faster-whisper yields segments as a generator. We time the first yield
        as the "first-token" proxy (caveat: beam search + chunk processing
        mean this is really first-segment latency, not first-word latency).
        """
        t0 = time.perf_counter()
        generator, _info = self.whisper.transcribe(
            audio_path,
            beam_size=5,
            initial_prompt=MEDICAL_INITIAL_PROMPT,
            word_timestamps=True,
            language="en",
        )
        segments: list = []
        first_token_ms: float | None = None
        for seg in generator:
            if first_token_ms is None:
                first_token_ms = (time.perf_counter() - t0) * 1000.0
            segments.append(seg)
        if first_token_ms is None:
            first_token_ms = (time.perf_counter() - t0) * 1000.0
        return segments, first_token_ms

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
        # Mistral's default chat template rejects the "system" role
        # ("Conversation roles must alternate user/assistant/..."). Merge the
        # system prompt into the first (and only) user message instead.
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
            "max_tokens": max(64, len(raw_text.split()) * 3),
        }
        r = requests.post(
            f"{self.cleanup_url}/chat/completions", json=payload, timeout=60
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()

    def run(self, audio_path: str) -> dict[str, Any]:
        timings: dict[str, float] = {}
        t_total = time.perf_counter()

        t = time.perf_counter()
        whisper_segs, first_token_ms = self.transcribe(audio_path)
        timings["asr_ms"] = (time.perf_counter() - t) * 1000.0
        timings["first_token_ms"] = first_token_ms

        diarization = None
        if self.enable_diarisation:
            t = time.perf_counter()
            diarization = self.diarize(audio_path)
            timings["diar_ms"] = (time.perf_counter() - t) * 1000.0
            aligned = self.align_speakers(whisper_segs, diarization)
        else:
            # Fallback: assign alternating DOCTOR/PATIENT by segment order.
            # Real diarisation is deferred; this is deterministic and lets WER
            # scoring proceed. DER will be reported as N/A in measure.py.
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
        for seg in aligned:
            seg.cleaned_text = self.cleanup_segment(seg.raw_text) if seg.raw_text else ""
        timings["cleanup_ms"] = (time.perf_counter() - t) * 1000.0

        timings["e2e_ms"] = (time.perf_counter() - t_total) * 1000.0

        return {
            "segments": [s.to_dict() for s in aligned],
            "timings": timings,
            "diarization": diarization,
            "diarisation_enabled": self.enable_diarisation,
        }
