"""Emit eval/synthetic_clips/ground_truth.jsonl from the per-scenario scripts.

One JSON object per scenario with {scenario, audio_path, turns, full_text_reference}.
Accepts both DOCTOR/PATIENT and DR/PT labels to match render_tts.py.

Run from the repo root on the L4:
    python -m src.extraction.flow.variant_a.build_ground_truth
"""

from __future__ import annotations

import json
import re
from pathlib import Path

SCENARIOS = [
    "chest_pain",
    "abdominal_pain",
    "dyspnea",
    "headache",
    "fatigue_weight_loss",
    "dizziness_syncope",
]

SCRIPTS_DIR = Path("eval/synthetic_clips")
OUT_PATH = SCRIPTS_DIR / "ground_truth.jsonl"

SPEAKER_RE = re.compile(r"^(?P<role>DOCTOR|PATIENT|DR|PT):\s*(?P<text>.+)$")


def _normalise_role(raw: str) -> str:
    return "DOCTOR" if raw in ("DOCTOR", "DR") else "PATIENT"


def build() -> None:
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for scenario in SCENARIOS:
            script_path = SCRIPTS_DIR / f"{scenario}_script.txt"
            if not script_path.exists():
                raise FileNotFoundError(f"Missing script: {script_path}")

            turns: list[dict] = []
            for line in script_path.read_text(encoding="utf-8").splitlines():
                m = SPEAKER_RE.match(line.strip())
                if m:
                    turns.append(
                        {"speaker": _normalise_role(m.group("role")), "text": m.group("text")}
                    )

            if not turns:
                raise RuntimeError(f"No turns parsed from {script_path}")

            record = {
                "scenario": scenario,
                "audio_path": f"eval/synthetic_clips/audio/{scenario}.wav",
                "turns": turns,
                "full_text_reference": " ".join(t["text"] for t in turns),
            }
            f.write(json.dumps(record) + "\n")
            print(f"{scenario}: {len(turns)} turns")

    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    build()
