"""DDXPlus → substrate turn-stream adapter.

Per Eng_doc.md §10.1: each DDXPlus record

```json
{
  "AGE": 42, "SEX": "F",
  "PATHOLOGY": "...",
  "EVIDENCES": ["E_123_@_V_1", "E_456", ...],
  "DIFFERENTIAL_DIAGNOSIS": [["Pathology A", 0.55], ...]
}
```

is converted to a natural-dialogue turn stream (physician prompt + patient
answer, repeated per evidence) that the substrate can ingest. Conversion is
deterministic — same record in, same turn list out.

Evidence IDs (`E_XXX_@_V_Y`) are resolved via `release_evidences.json`
from the DDXPlus repo, which maps IDs to human-readable questions + answers.

The adapter does NOT call the substrate; it returns a list of `Turn` objects.
`full.py` is the place that writes those turns via the stub (and later the
real `wt-engine` write API).
"""
from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eval._common import Turn

# Upstream DDXPlus uses this format for evidence IDs: `E_<code>_@_V_<value>`.
# When `_@_V_` is absent, the evidence is a boolean (present/absent).
EVIDENCE_SEP = "_@_V_"


@dataclass(frozen=True)
class DDXPlusCase:
    """One DDXPlus patient record, post-JSON-parse."""

    patient_id: str
    age: int
    sex: str
    pathology: str
    evidences: list[str]
    differential: list[tuple[str, float]]  # [(pathology, probability), ...]
    initial_evidence: str | None = None


def load_evidence_dictionary(release_evidences_path: Path) -> Mapping[str, Mapping[str, Any]]:
    """Load `release_evidences.json` → {evidence_id: {question_en, value_meaning, ...}}.

    The DDXPlus repo ships this file at `release_evidences.json`; it maps evidence
    IDs to their English question text and value enumerations. Required to render
    turns as natural dialogue.
    """
    return json.loads(release_evidences_path.read_text(encoding="utf-8"))


def _resolve_evidence(
    ev_id: str, evidence_dict: Mapping[str, Mapping[str, Any]]
) -> tuple[str, str] | None:
    """Resolve a single evidence ID to (question_text, answer_text).

    Returns None if the ID isn't in the dictionary — adapter logs + skips.
    """
    base_id, _, value = ev_id.partition(EVIDENCE_SEP)
    entry = evidence_dict.get(base_id)
    if entry is None:
        return None
    question = str(entry.get("question_en") or entry.get("question_fr") or base_id)
    if value:
        # Multi-value evidence — look up the value's English rendering.
        value_meanings = entry.get("value_meaning") or {}
        if isinstance(value_meanings, dict):
            v = value_meanings.get(value, {})
            answer = str(v.get("en", value)) if isinstance(v, dict) else str(v)
        else:
            answer = value
    else:
        # Boolean evidence — presence means "yes".
        answer = "Yes."
    return question, answer


def record_to_turns(
    case: DDXPlusCase, evidence_dict: Mapping[str, Mapping[str, Any]]
) -> list[Turn]:
    """Convert one DDXPlus record into a deterministic turn sequence.

    Turn 0: physician opener ("What brings you in today?").
    Turn 1: patient initial complaint (derived from INITIAL_EVIDENCE if
            present, else a canned "I've been feeling unwell" opener).
    Turns 2+: alternating physician question / patient answer per evidence.

    `turn_id` is derived from the patient_id + ordinal so it's stable across runs.
    """
    turns: list[Turn] = []
    pid = case.patient_id
    ts = 0

    def _tid(i: int) -> str:
        return f"{pid}::turn::{i:04d}"

    turns.append(
        Turn(
            turn_id=_tid(0),
            speaker="physician",
            text=f"I have a {case.age}-year-old {case.sex.lower()} patient. "
            "What brings you in today?",
            ts=ts,
        )
    )
    ts += 1000

    initial = case.initial_evidence
    if initial:
        resolved = _resolve_evidence(initial, evidence_dict)
        if resolved:
            q, a = resolved
            turns.append(Turn(turn_id=_tid(1), speaker="patient", text=a, ts=ts))
            ts += 1000

    idx = len(turns)
    for ev_id in case.evidences:
        resolved = _resolve_evidence(ev_id, evidence_dict)
        if not resolved:
            continue
        q, a = resolved
        turns.append(Turn(turn_id=_tid(idx), speaker="physician", text=q, ts=ts))
        ts += 1000
        idx += 1
        turns.append(Turn(turn_id=_tid(idx), speaker="patient", text=a, ts=ts))
        ts += 1000
        idx += 1

    return turns


def iter_cases(records_path: Path) -> Iterable[DDXPlusCase]:
    """Stream DDXPlus cases from a JSONL or JSON-array file.

    DDXPlus ships splits as CSVs under `release_validate_patients.csv` etc. The
    H-DDx 730-case stratified subset is derived from those; consumers feed us
    either the raw CSV-converted JSON or a pre-stratified subset file.

    For now this accepts a JSONL-per-line format (one record per line);
    `run.py` owns the CSV→JSONL conversion (TODO below).
    """
    # TODO(wt-eval): add CSV reader once fetch.py lands the DDXPlus release CSVs.
    with records_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            yield DDXPlusCase(
                patient_id=str(obj.get("patient_id") or obj.get("PATIENT_ID") or ""),
                age=int(obj["AGE"]),
                sex=str(obj["SEX"]),
                pathology=str(obj["PATHOLOGY"]),
                evidences=list(obj.get("EVIDENCES", [])),
                differential=[
                    (str(p), float(prob))
                    for p, prob in obj.get("DIFFERENTIAL_DIAGNOSIS", [])
                ],
                initial_evidence=obj.get("INITIAL_EVIDENCE"),
            )
