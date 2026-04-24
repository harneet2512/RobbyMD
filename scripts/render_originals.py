"""
Render the 6 original synthetic clips via Kokoro — same voices, seed,
inter-turn silence as scripts/render_pediatric.py.

Standalone so the ship pipeline can rebuild the audio corpus on a fresh
VM without needing variant_a code in the branch history.

Run from repo root with .venv-ship active:
    python scripts/render_originals.py
"""
from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from kokoro import KPipeline
from pydub import AudioSegment

SCRIPTS_DIR = Path("eval/synthetic_clips")
AUDIO_DIR = SCRIPTS_DIR / "audio"

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

SPEAKER_RE = re.compile(r"^(?P<role>DOCTOR|PATIENT|DR|PT):\s*(?P<text>.+)$")


def _role(raw: str) -> str:
    return "DOCTOR" if raw in ("DOCTOR", "DR") else "PATIENT"


def _render_one(pipe: KPipeline, scenario: str) -> Path:
    script_path = SCRIPTS_DIR / f"{scenario}_script.txt"
    if not script_path.exists():
        raise FileNotFoundError(f"{script_path} does not exist")

    turns: list[tuple[str, str]] = []
    for line in script_path.read_text(encoding="utf-8").splitlines():
        m = SPEAKER_RE.match(line.strip())
        if m:
            turns.append((_role(m.group("role")), m.group("text").strip()))
    if not turns:
        raise RuntimeError(f"no DR/PT lines in {script_path}")

    tmp_paths: list[Path] = []
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        for i, (role, text) in enumerate(turns):
            voice = DOCTOR_VOICE if role == "DOCTOR" else PATIENT_VOICE
            audio_chunks = []
            for _, _, audio in pipe(text, voice=voice, speed=1.0):
                audio_chunks.append(audio)
            if not audio_chunks:
                raise RuntimeError(f"Kokoro returned zero chunks for {scenario} turn {i}")
            combined = np.concatenate(audio_chunks)
            tmp = td_path / f"{scenario}_{i:03d}.wav"
            sf.write(tmp, combined, SAMPLE_RATE)
            tmp_paths.append(tmp)

        silence = AudioSegment.silent(duration=INTER_TURN_SILENCE_MS, frame_rate=SAMPLE_RATE)
        combined_audio = AudioSegment.empty()
        for idx, tmp in enumerate(tmp_paths):
            seg = AudioSegment.from_wav(str(tmp))
            if idx > 0:
                combined_audio += silence
            combined_audio += seg

        out_path = AUDIO_DIR / f"{scenario}.wav"
        combined_audio.export(out_path, format="wav")
        print(f"rendered {scenario}: {out_path} ({len(combined_audio) / 1000:.1f}s, {len(turns)} turns)")
        return out_path


def main() -> int:
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)
    pipe = KPipeline(lang_code="a", device="cpu")

    for scenario in SCENARIOS:
        try:
            _render_one(pipe, scenario)
        except Exception as exc:
            print(f"FAILED {scenario}: {exc}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
