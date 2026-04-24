"""
Ship pipeline: Whisper-large-v3-turbo + hotwords + pyannote community-1 +
fuzzy medical correction.

Replaces variant_a's BioMistral cleanup (which regressed WER +1.6pp) with
medical-term hotwords biasing at the Whisper decoder plus a zero-VRAM
post-hoc fuzzy corrector.
"""
from __future__ import annotations

import time
from typing import Optional

from faster_whisper import WhisperModel

from src.extraction.flow.ship.medical_correction import correct_medical_terms

MEDICAL_HOTWORDS = (
    "chest pain dyspnea troponin aspirin nitroglycerin "
    "myocardial infarction angina pulmonary embolism "
    "cholecystitis appendicitis diverticulitis pancreatitis "
    "migraine subarachnoid hemorrhage vasovagal orthostatic "
    "BPPV palpitations tachycardia bradycardia arrhythmia "
    "HEART score TIMI Wells PERC Kawasaki pneumonia "
    "pneumothorax pleuritic hemoptysis SpO2 "
    "amlodipine atorvastatin metformin lisinopril losartan "
    "acetaminophen ibuprofen immunoglobulin IVIG "
    "strawberry tongue desquamation febrile "
    "echocardiogram electrocardiogram D-dimer"
)


class ShipPipeline:
    def __init__(self, hf_token: Optional[str] = None, enable_diarization: bool = True):
        self.whisper = WhisperModel(
            "large-v3-turbo",
            device="cuda",
            compute_type="float16",
        )

        self.diar = None
        self.diar_enabled = False
        if enable_diarization and hf_token:
            try:
                # pyannote 4.x telemetry calls torchcodec's AudioDecoder on
                # every pipeline apply, and AudioDecoder isn't importable on
                # this DLVM image. Disable telemetry before loading to avoid
                # the NameError during inference.
                try:
                    from pyannote.audio.telemetry import set_telemetry_metrics
                    set_telemetry_metrics(False)
                except Exception:
                    pass
                from pyannote.audio import Pipeline as DiarPipeline
                try:
                    self.diar = DiarPipeline.from_pretrained(
                        "pyannote/speaker-diarization-community-1",
                        token=hf_token,
                    )
                except TypeError:
                    self.diar = DiarPipeline.from_pretrained(
                        "pyannote/speaker-diarization-community-1",
                        use_auth_token=hf_token,
                    )
                import torch as _torch
                self.diar.to(_torch.device("cuda"))
                self.diar_enabled = True
                print("pyannote community-1 loaded")
            except Exception as exc:
                print(f"pyannote failed to load: {exc}")
                print("falling back to alternating-turn heuristic")

    def transcribe(self, audio_path: str):
        segments, info = self.whisper.transcribe(
            audio_path,
            beam_size=5,
            hotwords=MEDICAL_HOTWORDS,
            word_timestamps=True,
            language="en",
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": 500,
                "speech_pad_ms": 200,
            },
            condition_on_previous_text=False,
        )
        return list(segments), info

    def diarize(self, audio_path: str):
        if not self.diar_enabled:
            return None
        # Preload via torchaudio so pyannote 4.x skips torchcodec's
        # AudioDecoder path (not importable on this DLVM image). Same
        # workaround variant_a uses.
        import torchaudio
        waveform, sample_rate = torchaudio.load(audio_path)
        return self.diar(
            {"waveform": waveform, "sample_rate": sample_rate},
            num_speakers=2,
        )

    def assign_speakers(self, whisper_segments, diarization):
        if diarization is None:
            segments = []
            for i, seg in enumerate(whisper_segments):
                segments.append({
                    "speaker": "DOCTOR" if i % 2 == 0 else "PATIENT",
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text.strip(),
                })
            return segments

        speaker_turns = [
            (turn.start, turn.end, spk)
            for turn, _, spk in diarization.itertracks(yield_label=True)
        ]
        label_map: dict[str, str] = {}
        for _, _, spk in speaker_turns:
            if spk not in label_map:
                label_map[spk] = "DOCTOR" if len(label_map) == 0 else "PATIENT"

        segments = []
        for seg in whisper_segments:
            mid = (seg.start + seg.end) / 2
            speaker = "UNKNOWN"
            for s, e, spk in speaker_turns:
                if s <= mid <= e:
                    speaker = label_map.get(spk, "UNKNOWN")
                    break
            segments.append({
                "speaker": speaker,
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip(),
            })
        return segments

    def correct(self, segments):
        all_corrections = []
        for seg in segments:
            corrected_text, corrections = correct_medical_terms(seg["text"])
            seg["raw_text"] = seg["text"]
            seg["text"] = corrected_text
            seg["corrections"] = corrections
            all_corrections.extend(corrections)
        return segments, all_corrections

    def run(self, audio_path: str) -> dict:
        import soundfile as sf
        info = sf.info(audio_path)
        audio_duration = info.frames / info.samplerate

        timings: dict[str, float] = {}
        t0 = time.perf_counter()

        t_asr = time.perf_counter()
        whisper_segments, whisper_info = self.transcribe(audio_path)
        timings["asr_ms"] = (time.perf_counter() - t_asr) * 1000

        t_diar = time.perf_counter()
        diarization = self.diarize(audio_path)
        timings["diar_ms"] = (time.perf_counter() - t_diar) * 1000

        segments = self.assign_speakers(whisper_segments, diarization)

        t_correct = time.perf_counter()
        segments, corrections = self.correct(segments)
        timings["correction_ms"] = (time.perf_counter() - t_correct) * 1000

        timings["e2e_ms"] = (time.perf_counter() - t0) * 1000

        return {
            "audio_path": audio_path,
            "audio_duration_sec": audio_duration,
            "segments": segments,
            "corrections": corrections,
            "diarization_enabled": self.diar_enabled,
            "diarization_raw": diarization,
            "timings": timings,
            "whisper_language": whisper_info.language,
            "whisper_language_probability": whisper_info.language_probability,
        }
