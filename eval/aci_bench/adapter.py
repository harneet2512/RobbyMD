"""ACI-Bench → substrate turn-stream adapter.

ACI-Bench ships encounters as:

- `{split}/{id}.dialogue.json` — turns as [{"speaker": "PATIENT", "utterance": "..."}]
- `{split}/{id}.note.txt`      — gold SOAP note text

The adapter converts one encounter into a `Turn` list (matching Eng_doc.md
§4.1) so the substrate can ingest it, plus keeps the gold note for the
eval comparison pass.
"""
from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from eval._common import Turn

# Mapping of ACI-Bench speaker labels → substrate speaker channel.
_SPEAKER_MAP = {
    "PATIENT": "patient",
    "DOCTOR": "physician",
    "PROVIDER": "physician",
    "CLINICIAN": "physician",
    "NURSE": "physician",
    "GUEST_FAMILY": "patient",   # family members map to patient side
}


@dataclass(frozen=True)
class ACIEncounter:
    """One ACI-Bench encounter."""

    encounter_id: str
    split: str          # "aci" or "virtscribe"
    subsplit: str       # "test1" | "test2" | "test3" | "test"
    dialogue: list[dict[str, str]]  # raw [{speaker, utterance}, ...]
    gold_note: str


def encounter_to_turns(enc: ACIEncounter) -> list[Turn]:
    """Convert one encounter's dialogue into a Turn list.

    Unknown speaker labels fall back to `system` so nothing is dropped;
    downstream retrieval can still see the content. Turn IDs embed encounter
    + ordinal so they're stable across runs.
    """
    turns: list[Turn] = []
    for i, row in enumerate(enc.dialogue):
        speaker_raw = str(row.get("speaker", "")).upper().strip()
        speaker = _SPEAKER_MAP.get(speaker_raw, "system")
        text = str(row.get("utterance", ""))
        if not text:
            continue
        turns.append(
            Turn(
                turn_id=f"{enc.encounter_id}::turn::{i:04d}",
                speaker=speaker,  # type: ignore[arg-type] — Literal narrowing
                text=text,
                ts=i * 1000,
            )
        )
    return turns


def iter_encounters(
    data_root: Path, split: str, subsplits: Iterable[str] | None = None
) -> Iterator[ACIEncounter]:
    """Stream encounters from ACI-Bench's directory layout.

    data_root: path to the cloned `aci-bench-repo/src/data/` (or equivalent).
    split:     "aci" or "virtscribe".
    subsplits: e.g. ("test1", "test2", "test3") for aci; ("test",) for virtscribe.

    Accepts both `*.dialogue.json` + `*.note.txt` layouts and (older) CSV
    layouts. TODO(wt-eval): confirm and harden against actual repo structure
    on first fetch. For now, we scan for paired `*.json` + `*.txt` files.
    """
    if subsplits is None:
        subsplits = ("test",) if split == "virtscribe" else ("test1", "test2", "test3")

    for sub in subsplits:
        base = data_root / split / sub
        if not base.exists():
            continue
        for dlg_path in sorted(base.glob("*.dialogue.json")):
            eid = dlg_path.name.removesuffix(".dialogue.json")
            note_path = base / f"{eid}.note.txt"
            if not note_path.exists():
                print(f"[adapter] WARN missing note for {dlg_path}; skipping")
                continue
            dialogue_raw = json.loads(dlg_path.read_text(encoding="utf-8"))
            gold_note = note_path.read_text(encoding="utf-8")
            if isinstance(dialogue_raw, dict):
                dialogue_raw = dialogue_raw.get("dialogue", [])
            yield ACIEncounter(
                encounter_id=eid,
                split=split,
                subsplit=sub,
                dialogue=list(dialogue_raw),
                gold_note=gold_note,
            )


def iter_all_test_encounters(data_root: Path) -> Iterator[ACIEncounter]:
    """Stream the FULL 90-encounter test set (aci test1-3 + virtscribe test).

    Preferred entry point for eval runs — keeps the "no slicing" invariant
    visible in one call site.
    """
    yield from iter_encounters(data_root, "aci", ("test1", "test2", "test3"))
    yield from iter_encounters(data_root, "virtscribe", ("test",))
