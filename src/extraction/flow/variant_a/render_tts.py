"""Render the 6 synthetic clinical dialogues to WAV via Kokoro-82M (Apache-2.0).

Reads eval/synthetic_clips/<scenario>_script.txt, accepts both
DOCTOR/PATIENT and DR/PT labels (existing Stream D scripts use DR/PT),
concatenates Kokoro turns with 300ms inter-turn silence, writes to
eval/synthetic_clips/audio/<scenario>.wav at 24 kHz mono.

Deterministic via torch.manual_seed(42). Run once from the repo root on the L4:
    python -m src.extraction.flow.variant_a.render_tts
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from kokoro import KPipeline
from pydub import AudioSegment

SCENARIOS = [
    "chest_pain",
    "abdominal_pain",
    "dyspnea",
    "headache",
    "fatigue_weight_loss",
    "dizziness_syncope",
]

DOCTOR_VOICE = "am_michael"
PATIENT_VOICE = "af_bella"
SAMPLE_RATE = 24_000
INTER_TURN_SILENCE_MS = 300

SCRIPTS_DIR = Path("eval/synthetic_clips")
AUDIO_DIR = SCRIPTS_DIR / "audio"

# Accept both DOCTOR/PATIENT (brief) and DR/PT (Stream D scripts on disk).
SPEAKER_RE = re.compile(r"^(?P<role>DOCTOR|PATIENT|DR|PT):\s*(?P<text>.+)$")


def _normalise_role(raw: str) -> str:
    return "DOCTOR" if raw in ("DOCTOR", "DR") else "PATIENT"


def _parse_turns(script_path: Path) -> list[tuple[str, str]]:
    turns: list[tuple[str, str]] = []
    for line in script_path.read_text(encoding="utf-8").splitlines():
        m = SPEAKER_RE.match(line.strip())
        if m:
            turns.append((_normalise_role(m.group("role")), m.group("text")))
    return turns


def render_all() -> None:
    torch.manual_seed(42)
    np.random.seed(42)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    # Force CPU: Kokoro's CUDA init hits a cuBLAS symbol-load issue against
    # the L4's torch+cu121 stack. 82M-param model runs in a few seconds per
    # clip on CPU and frees VRAM for Whisper + pyannote + vLLM downstream.
    pipeline = KPipeline(lang_code="a", device="cpu")

    for scenario in SCENARIOS:
        script_path = SCRIPTS_DIR / f"{scenario}_script.txt"
        if not script_path.exists():
            raise FileNotFoundError(f"Missing script: {script_path}")

        turns = _parse_turns(script_path)
        if not turns:
            raise RuntimeError(
                f"No DR/PT or DOCTOR/PATIENT lines parsed from {script_path}; "
                f"check SPEAKER_RE against script format."
            )

        tmp_paths: list[Path] = []
        for i, (role, text) in enumerate(turns):
            voice = DOCTOR_VOICE if role == "DOCTOR" else PATIENT_VOICE
            tmp = AUDIO_DIR / f"_tmp_{scenario}_{i:03d}.wav"
            generator = pipeline(text, voice=voice, speed=1.0)
            chunks = []
            for _, _, audio in generator:
                arr = audio.cpu().numpy() if torch.is_tensor(audio) else audio
                chunks.append(np.asarray(arr).astype(np.float32))
            if not chunks:
                raise RuntimeError(f"Kokoro returned zero chunks for turn {i}")
            sf.write(str(tmp), np.concatenate(chunks), SAMPLE_RATE)
            tmp_paths.append(tmp)

        silence = AudioSegment.silent(duration=INTER_TURN_SILENCE_MS)
        combined = AudioSegment.empty()
        for tmp in tmp_paths:
            combined += AudioSegment.from_wav(tmp) + silence

        out_path = AUDIO_DIR / f"{scenario}.wav"
        combined.export(out_path, format="wav")
        for tmp in tmp_paths:
            tmp.unlink()

        print(f"Rendered {scenario}: {out_path} ({len(combined) / 1000:.1f}s, {len(turns)} turns)")


if __name__ == "__main__":
    render_all()
