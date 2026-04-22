"""Synthetic chest-pain dialogue audio — no real-human voice input.

Per `rules.md` §2.1 / §2.5 the demo pipeline is evaluated only on synthetic or
public-benchmark audio. This module generates a scripted short clip via
`pyttsx3` (MIT, SAPI5 / NSSpeechSynthesizer / eSpeak backend), which offers
two distinct system voices we alternate between to fake a two-speaker
consultation without any real recording.

The script below is authored from scratch for the hackathon (rules.md §1.1).
Every phrase is generic chest-pain dialogue chosen to exercise:
- One drug-name utterance ("metoprolol", "aspirin") so the vocab-bias benchmark
  has something to score.
- One supersession moment ("— actually, about 30 minutes ago") so downstream
  components have a canonical test fixture.
- Pleuritic / exertional descriptors to light up the AHA/ACC bias vocabulary.

The WAV and a pinned transcript JSON are consumed by `docs/asr_benchmark.md`.
`SYNTHETIC_DATA.md` declares this artefact.
"""

from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast


@dataclass(frozen=True, slots=True)
class ScriptedUtterance:
    """One synthetic dialogue line."""

    speaker: str  # "patient" or "physician"
    text: str


# Canonical chest-pain demo script. Kept short (~45 s at normal TTS rate) so
# the benchmark harness loops quickly. If this script is edited, regenerate the
# WAV via `synthesise_clip` and re-run the benchmark (both tracked in
# `docs/asr_benchmark.md` reproducibility section).
DEMO_SCRIPT: tuple[ScriptedUtterance, ...] = (
    ScriptedUtterance("physician", "Good morning. Can you tell me about the chest pain?"),
    ScriptedUtterance("patient", "It started about an hour ago. It feels like pressure in the centre of my chest."),
    ScriptedUtterance("physician", "Does the pain go anywhere else, like your arm or jaw?"),
    ScriptedUtterance("patient", "Yes, it radiates to my left arm. And I feel short of breath."),
    ScriptedUtterance("physician", "Is it worse when you breathe in, or when you move?"),
    ScriptedUtterance("patient", "No, not really. But it got worse when I climbed the stairs."),
    ScriptedUtterance("physician", "Any medications you take regularly?"),
    ScriptedUtterance("patient", "I take metoprolol and aspirin. And atorvastatin in the evening."),
    ScriptedUtterance("physician", "You said it started an hour ago — can you be more precise?"),
    ScriptedUtterance("patient", "Actually, thinking again, it was more like thirty minutes ago. Not a full hour."),
)


def script_as_gold_transcript() -> str:
    """Concatenate the script as a single reference string for WER.

    One token-stream is a defensible choice on such a short clip because
    jiwer and whisper WER tools all operate on whole-corpus comparison. See
    `docs/asr_benchmark.md` methodology.
    """
    return " ".join(u.text for u in DEMO_SCRIPT)


def _pick_voice_ids(engine: Any) -> tuple[str, str]:
    """Pick two distinct `pyttsx3` voice ids for patient / physician.

    Falls back to the default voice twice if only one is available (e.g.
    minimal Linux + eSpeak install). Diarisation still works — pyannote
    will label both tracks as one speaker, which the benchmark records.
    """
    # `engine` is typed `Any` so this module imports without pyttsx3 stubs.
    voices = engine.getProperty("voices")
    ids: list[str] = [str(v.id) for v in voices]
    if len(ids) >= 2:
        return ids[0], ids[1]
    fallback = ids[0] if ids else ""
    return fallback, fallback


def synthesise_clip(out_wav: Path, out_script_json: Path | None = None) -> Path:
    """Render `DEMO_SCRIPT` to a single-channel WAV at `out_wav`.

    Writes `out_script_json` (if supplied) alongside with the script + picked
    voice ids, so a benchmark can re-derive the gold transcript deterministically.

    Returns the path of the written WAV.

    Requires `pyttsx3` installed. On Windows this uses SAPI5 (system voices);
    on Linux it requires eSpeak. On macOS it uses NSSpeechSynthesizer.
    """
    # Lazy import — pyttsx3 is an optional extraction-dev dep.
    import pyttsx3  # type: ignore[import-not-found]

    pyttsx3_any = cast(Any, pyttsx3)
    engine: Any = pyttsx3_any.init()
    voice_patient, voice_physician = _pick_voice_ids(engine)
    engine.setProperty("rate", 175)

    out_wav.parent.mkdir(parents=True, exist_ok=True)

    # pyttsx3 cannot concat clips into one WAV natively; render each utterance
    # to a tmp file and concatenate with the `wave` stdlib module to avoid
    # pulling ffmpeg as an extra dep.
    import wave

    tmp_dir = out_wav.parent / ".synth_tmp"
    tmp_dir.mkdir(exist_ok=True)
    part_paths: list[Path] = []

    try:
        for i, utt in enumerate(DEMO_SCRIPT):
            voice = voice_patient if utt.speaker == "patient" else voice_physician
            engine.setProperty("voice", voice)
            part = tmp_dir / f"utt_{i:02d}.wav"
            engine.save_to_file(utt.text, str(part))
            engine.runAndWait()
            part_paths.append(part)

        with wave.open(str(out_wav), "wb") as writer:
            first = True
            for part in part_paths:
                with wave.open(str(part), "rb") as reader:
                    if first:
                        writer.setparams(reader.getparams())
                        first = False
                    writer.writeframes(reader.readframes(reader.getnframes()))
    finally:
        for part in part_paths:
            part.unlink(missing_ok=True)
        if tmp_dir.exists():
            # Leave on filesystem on error; next run cleans up.
            with contextlib.suppress(OSError):
                tmp_dir.rmdir()

    if out_script_json is not None:
        out_script_json.write_text(
            json.dumps(
                {
                    "script": [
                        {"speaker": u.speaker, "text": u.text} for u in DEMO_SCRIPT
                    ],
                    "voices": {"patient": voice_patient, "physician": voice_physician},
                    "gold_transcript": script_as_gold_transcript(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return out_wav
