"""
Render pediatric_fever_rash.wav via Kokoro with the same voices, seed, and
inter-turn silence as src/extraction/flow/variant_a/render_tts.py.

Standalone so the ship pipeline can produce its 7th clip without modifying
variant_a (SCENARIOS list is frozen there).

Run from repo root with the .venv-ship virtualenv active:
    python scripts/render_pediatric.py
"""
from __future__ import annotations

import json
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
SCENARIO = "pediatric_fever_rash"

DOCTOR_VOICE = "am_michael"
PATIENT_VOICE = "af_bella"
SAMPLE_RATE = 24_000
INTER_TURN_SILENCE_MS = 300

SPEAKER_RE = re.compile(r"^(?P<role>DOCTOR|PATIENT|DR|PT):\s*(?P<text>.+)$")


def _role(raw: str) -> str:
    return "DOCTOR" if raw in ("DOCTOR", "DR") else "PATIENT"


def main() -> int:
    script_path = SCRIPTS_DIR / f"{SCENARIO}_script.txt"
    if not script_path.exists():
        print(f"ERROR: {script_path} does not exist", file=sys.stderr)
        return 1

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    turns: list[tuple[str, str]] = []
    for line in script_path.read_text(encoding="utf-8").splitlines():
        m = SPEAKER_RE.match(line.strip())
        if m:
            turns.append((_role(m.group("role")), m.group("text").strip()))

    if not turns:
        print(f"ERROR: no DR/PT or DOCTOR/PATIENT lines in {script_path}", file=sys.stderr)
        return 1

    print(f"parsed {len(turns)} turns from {script_path}")

    torch.manual_seed(42)
    np.random.seed(42)

    # Kokoro on CPU — frees GPU for Whisper + pyannote downstream.
    pipe = KPipeline(lang_code="a", device="cpu")

    tmp_paths: list[Path] = []
    turn_durations_s: list[float] = []
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        for i, (role, text) in enumerate(turns):
            voice = DOCTOR_VOICE if role == "DOCTOR" else PATIENT_VOICE
            generator = pipe(text, voice=voice, speed=1.0)
            tmp = td_path / f"{SCENARIO}_{i:03d}.wav"
            audio_chunks = []
            for _, _, audio in generator:
                audio_chunks.append(audio)
            if not audio_chunks:
                raise RuntimeError(f"Kokoro returned zero chunks for turn {i}")
            combined = np.concatenate(audio_chunks)
            sf.write(tmp, combined, SAMPLE_RATE)
            tmp_paths.append(tmp)
            turn_durations_s.append(len(combined) / SAMPLE_RATE)

        silence = AudioSegment.silent(duration=INTER_TURN_SILENCE_MS, frame_rate=SAMPLE_RATE)
        combined_audio = AudioSegment.empty()
        for idx, tmp in enumerate(tmp_paths):
            seg = AudioSegment.from_wav(str(tmp))
            if idx > 0:
                combined_audio += silence
            combined_audio += seg

        out_path = AUDIO_DIR / f"{SCENARIO}.wav"
        combined_audio.export(out_path, format="wav")
        print(f"rendered {SCENARIO}: {out_path} ({len(combined_audio) / 1000:.1f}s, {len(turns)} turns)")

    # Emit per-turn timing sidecar so measure.compute_der can build a real
    # ground-truth Annotation instead of the equal-duration-slot proxy.
    inter_silence_s = INTER_TURN_SILENCE_MS / 1000.0
    timeline: list[dict] = []
    t = 0.0
    for (role, text), dur in zip(turns, turn_durations_s):
        timeline.append({
            "speaker": role,
            "text": text,
            "start_s": round(t, 3),
            "end_s": round(t + dur, 3),
        })
        t = t + dur + inter_silence_s
    sidecar = out_path.with_suffix(".turns.json")
    sidecar.write_text(json.dumps(timeline, indent=2), encoding="utf-8")
    print(f"  sidecar: {sidecar.name} ({len(timeline)} turns)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
