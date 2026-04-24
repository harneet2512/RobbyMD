"""
Render the 6 original synthetic clips via Kokoro — same voices, seed,
inter-turn silence as scripts/render_pediatric.py.

Standalone so the ship pipeline can rebuild the audio corpus on a fresh
VM without needing variant_a code in the branch history.

Run from repo root with .venv-ship active:
    python scripts/render_originals.py
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
    turn_durations_s: list[float] = []
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
            turn_durations_s.append(len(combined) / SAMPLE_RATE)

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

    # Emit per-turn timing sidecar so measure.compute_der can build a real
    # ground-truth Annotation instead of the equal-duration-slot proxy.
    _write_turn_sidecar(out_path, turns, turn_durations_s)
    return out_path


def _write_turn_sidecar(
    out_path: Path,
    turns: list[tuple[str, str]],
    turn_durations_s: list[float],
) -> None:
    """Write {scenario}.turns.json alongside the wav with real per-turn times."""
    inter_silence_s = INTER_TURN_SILENCE_MS / 1000.0
    timeline: list[dict] = []
    t = 0.0
    for (role, text), dur in zip(turns, turn_durations_s):
        start_s = t
        end_s = t + dur
        timeline.append({
            "speaker": role,  # DOCTOR or PATIENT (normalized in _role)
            "text": text,
            "start_s": round(start_s, 3),
            "end_s": round(end_s, 3),
        })
        t = end_s + inter_silence_s
    sidecar = out_path.with_suffix(".turns.json")
    sidecar.write_text(json.dumps(timeline, indent=2), encoding="utf-8")
    print(f"  sidecar: {sidecar.name} ({len(timeline)} turns, total {timeline[-1]['end_s']:.1f}s)")


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
