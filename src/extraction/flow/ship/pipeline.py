"""
Ship pipeline: Whisper-large-v3-turbo + hotwords + pyannote community-1 +
fuzzy medical correction.

Hotwords and correction vocabulary loaded from predicate packs (default:
predicate_packs/clinical_general/). Domain-agnostic substrate with pluggable
clinical vocabulary.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from faster_whisper import WhisperModel

from src.extraction.flow.ship.medical_correction import MedicalCorrector


def load_predicate_pack(pack_name: str = "clinical_general") -> dict:
    """Load hotwords, correction vocab, and speaker config from a predicate pack."""
    pack_dir = Path(__file__).resolve().parents[4] / "predicate_packs" / pack_name
    config = json.loads((pack_dir / "config.json").read_text(encoding="utf-8"))

    hotwords_path = pack_dir / config["hotwords_file"]
    hotwords = " ".join(
        line.strip()
        for line in hotwords_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    )

    correction_vocab_path = pack_dir / config["correction_vocab_file"]
    correction_vocab = [
        line.strip()
        for line in correction_vocab_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

    return {
        "hotwords": hotwords,
        "correction_vocab": correction_vocab,
        "min_speakers": config.get("min_speakers", 2),
        "max_speakers": config.get("max_speakers", 5),
        "speaker_roles": config.get("speaker_roles", ["DOCTOR", "PATIENT"]),
    }


class ShipPipeline:
    def __init__(
        self,
        hf_token: Optional[str] = None,
        enable_diarization: bool = True,
        pack_name: str = "clinical_general",
    ):
        pack = load_predicate_pack(pack_name)
        self.hotwords = pack["hotwords"]
        self.min_speakers = pack["min_speakers"]
        self.max_speakers = pack["max_speakers"]
        self.speaker_roles = pack["speaker_roles"]
        self.corrector = MedicalCorrector(vocabulary=pack["correction_vocab"])

        self.whisper = WhisperModel(
            "large-v3-turbo",
            device="cuda",
            compute_type="float16",
        )

        self.diar = None
        self.diar_enabled = False
        if enable_diarization and hf_token:
            try:
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
                # Force CPU. NVRTC 13 smoke-test passes on short tensors
                # but fails on real audio — different JIT code paths for
                # different lengths. CUDA pyannote on this DLVM (torch
                # 2.11+cu130 vs CUDA 12.9) is unfixable without a toolkit
                # upgrade. CPU adds ~2min/clip but produces DER 0.280.
                self.diar.to(_torch.device("cpu"))
                self.diar_enabled = True
                print("pyannote community-1 loaded on CPU")
            except Exception as exc:
                print(f"pyannote failed to load: {exc}")
                print("falling back to alternating-turn heuristic")

    def transcribe(self, audio_path: str):
        segments, info = self.whisper.transcribe(
            audio_path,
            beam_size=5,
            hotwords=self.hotwords,
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

    def diarize(self, audio_path: str, force_num_speakers: Optional[int] = None):
        if not self.diar_enabled:
            return None
        try:
            import torchaudio
            waveform, sample_rate = torchaudio.load(audio_path)
            if force_num_speakers is not None:
                out = self.diar(
                    {"waveform": waveform, "sample_rate": sample_rate},
                    num_speakers=force_num_speakers,
                )
            else:
                out = self.diar(
                    {"waveform": waveform, "sample_rate": sample_rate},
                    min_speakers=self.min_speakers,
                    max_speakers=self.max_speakers,
                )
            if hasattr(out, "speaker_diarization"):
                return out
            return out
        except Exception as exc:
            print(f"  diarize error on {audio_path}: {type(exc).__name__}: {exc}")
            return None

    def _get_speaker_turns(self, diarization):
        """Extract speaker turns, preferring exclusive mode (pyannote 4.x)."""
        try:
            return [
                (seg.start, seg.end, spk)
                for seg, spk in diarization.exclusive_speaker_diarization
            ]
        except AttributeError:
            pass
        try:
            return [
                (turn.start, turn.end, spk)
                for turn, _, spk in diarization.itertracks(yield_label=True)
            ]
        except AttributeError:
            pass
        if hasattr(diarization, "speaker_diarization"):
            ann = diarization.speaker_diarization
            return [
                (turn.start, turn.end, spk)
                for turn, _, spk in ann.itertracks(yield_label=True)
            ]
        return []

    def assign_speakers(self, whisper_segments, diarization):
        if diarization is None:
            segments = []
            for i, seg in enumerate(whisper_segments):
                segments.append({
                    "speaker": self.speaker_roles[i % len(self.speaker_roles)],
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text.strip(),
                })
            return segments

        speaker_turns = self._get_speaker_turns(diarization)

        label_map: dict[str, str] = {}
        for _, _, spk in speaker_turns:
            if spk not in label_map:
                idx = len(label_map)
                label_map[spk] = (
                    self.speaker_roles[idx]
                    if idx < len(self.speaker_roles)
                    else f"SPEAKER_{idx}"
                )

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
            corrected_text, corrections = self.corrector.correct(seg["text"])
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
