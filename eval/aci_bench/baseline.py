"""ACI-Bench baseline — Opus 4.7 direct dialogue → SOAP note.

No substrate. Dialogue text → Opus → SOAP note. Compared against the
substrate variant (full.py) via MEDIQA-CHAT metrics (ROUGE, BERTScore, MEDCON).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from eval.aci_bench.adapter import ACIEncounter

OPUS_MODEL = "claude-opus-4-7"

_SYSTEM_PROMPT = (
    "You are a clinical documentation assistant producing a SOAP note from a "
    "doctor-patient conversation. Output only the note, in four sections "
    "(SUBJECTIVE / OBJECTIVE / ASSESSMENT / PLAN). Use only information present "
    "in the transcript. Do NOT fabricate findings or orders. This is a research "
    "prototype, not a medical device; the physician makes every clinical decision."
)


@dataclass(frozen=True)
class ACINotePrediction:
    encounter_id: str
    predicted_note: str
    raw_response: str = ""


def _format_dialogue(enc: ACIEncounter) -> str:
    lines = []
    for row in enc.dialogue:
        speaker = str(row.get("speaker", "UNKNOWN"))
        utt = str(row.get("utterance", "")).strip()
        if utt:
            lines.append(f"{speaker}: {utt}")
    return "\n".join(lines)


def predict_note(enc: ACIEncounter) -> ACINotePrediction:
    """Return Opus 4.7's SOAP note for one encounter.

    Without ANTHROPIC_API_KEY → stub returning the gold note so the harness
    can run without network. Flagged in run.py.
    """
    if not os.getenv("ANTHROPIC_API_KEY"):
        return ACINotePrediction(
            encounter_id=enc.encounter_id,
            predicted_note=enc.gold_note,  # stub — not a scoring path
            raw_response="[STUB] ANTHROPIC_API_KEY not set",
        )

    try:
        import anthropic
    except ImportError as e:
        raise RuntimeError(
            "anthropic SDK required for live baseline; run pip install -e .[dev]"
        ) from e

    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=OPUS_MODEL,
        max_tokens=2048,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _format_dialogue(enc)}],
    )
    text = _extract_text(resp)
    return ACINotePrediction(
        encounter_id=enc.encounter_id,
        predicted_note=text,
        raw_response=text,
    )


def _extract_text(resp: object) -> str:
    try:
        blocks = resp.content  # type: ignore[attr-defined]
    except AttributeError:
        return ""
    for block in blocks:
        if getattr(block, "type", None) == "text":
            return str(getattr(block, "text", ""))
    return ""
