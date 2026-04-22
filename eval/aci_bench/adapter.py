"""ACI-Bench → substrate turn-stream adapter.

The shipped ACI-Bench repo layout (wyim/aci-bench, b909b2b) is a flat set
of JSON files under `data/challenge_data_json/`:

    clinicalnlp_taskB_test1.json   — 40 rows (test1, clinicalnlp shared task)
    clinicalnlp_taskC_test2.json   — 20 rows (test2, clinicalnlp shared task)
    clef_taskC_test3.json          — 40 rows (test3, CLEF shared task)

Each JSON has shape `{"data": [<row>, ...]}`; each row has:

    src  — concatenated dialogue, `[doctor] text\\n[patient] text\\n...`
    tgt  — gold SOAP note (prose)
    file — encounter id (e.g. `D2N088-virtassist`)

The adapter parses the `src` bracket-prefixed format into
`[{"speaker": ..., "utterance": ...}, ...]`, keeps the `tgt` as
`gold_note`, and exposes one encounter per row as an `ACIEncounter`.
"""
from __future__ import annotations

import json
import re
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


# Mapping of test-split id → challenge-data JSON filename, in adapter scope
# so downstream callers can target a single file when needed.
_CHALLENGE_DATA_FILES: dict[str, str] = {
    "test1": "clinicalnlp_taskB_test1.json",
    "test2": "clinicalnlp_taskC_test2.json",
    "test3": "clef_taskC_test3.json",
}


# Match lines like `[doctor] utterance text` (possibly multi-line between
# successive `[label]` openers). The same row's `src` string inlines every
# turn back-to-back with `\n[...]` separators.
_TURN_PATTERN = re.compile(r"\[([a-zA-Z_][\w ]*)\]\s*(.*?)(?=\n\[|\Z)", re.DOTALL)


def _parse_src_dialogue(src: str) -> list[dict[str, str]]:
    """Parse ACI-Bench's bracket-prefixed `src` string into dialogue dicts.

    Example input:  `[doctor] hi, andrew .\\n[patient] hey , good to see you .`
    Example output: [{"speaker": "doctor", "utterance": "hi, andrew ."},
                     {"speaker": "patient", "utterance": "hey , good to see you ."}]
    """
    out: list[dict[str, str]] = []
    for m in _TURN_PATTERN.finditer(src):
        speaker = m.group(1).strip()
        utterance = m.group(2).strip()
        if utterance:
            out.append({"speaker": speaker.upper(), "utterance": utterance})
    return out


def _resolve_challenge_data_root(data_root: Path) -> Path:
    """Return the `challenge_data_json` directory, accepting either the
    clone's root or the already-nested JSON subdir as input.

    - If `data_root` already points at `.../challenge_data_json`, return as-is.
    - Otherwise look for `data_root/challenge_data_json` and return that.
    - Raises `FileNotFoundError` otherwise so callers see the mismatch fast.
    """
    if data_root.name == "challenge_data_json" and data_root.exists():
        return data_root
    candidate = data_root / "challenge_data_json"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"ACI-Bench challenge_data_json not found under {data_root}. "
        f"Expected one of:\n  {data_root}/challenge_data_json\n  {data_root} "
        f"(if it is itself the challenge_data_json dir)"
    )


def iter_encounters(
    data_root: Path, split: str, subsplits: Iterable[str] | None = None
) -> Iterator[ACIEncounter]:
    """Stream encounters from ACI-Bench's shipped JSON layout.

    data_root: path to `challenge_data_json/` or its parent (`data/`).
    split:     retained for signature stability; always treated as "aci" —
               the shipped challenge_data JSONs cover the full 100 encounters
               including what older docs called the "virtscribe" split.
    subsplits: subset of {"test1", "test2", "test3"}; `None` means all three.
    """
    del split  # kept for API compatibility; no longer drives file selection
    root = _resolve_challenge_data_root(data_root)

    if subsplits is None:
        keys = list(_CHALLENGE_DATA_FILES.keys())
    else:
        keys = [k for k in subsplits if k in _CHALLENGE_DATA_FILES]

    for sub in keys:
        fname = _CHALLENGE_DATA_FILES[sub]
        path = root / fname
        if not path.exists():
            print(f"[adapter] WARN missing challenge file at {path}; skipping {sub}")
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("data") if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            print(f"[adapter] WARN unexpected shape in {path}; skipping {sub}")
            continue
        for row in rows:
            encounter_id = str(row.get("file", "")).strip()
            src = str(row.get("src", ""))
            gold_note = str(row.get("tgt", ""))
            if not encounter_id or not src:
                continue
            yield ACIEncounter(
                encounter_id=encounter_id,
                split="aci",
                subsplit=sub,
                dialogue=_parse_src_dialogue(src),
                gold_note=gold_note,
            )


def iter_all_test_encounters(data_root: Path) -> Iterator[ACIEncounter]:
    """Stream the full ACI-Bench test set (test1 + test2 + test3).

    Preferred entry point for eval runs — keeps the "no slicing" invariant
    visible in one call site.
    """
    yield from iter_encounters(data_root, "aci", ("test1", "test2", "test3"))
