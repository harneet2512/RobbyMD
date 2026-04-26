"""Note version persistence — tracks SOAP note edits with diffs."""
from __future__ import annotations

import difflib
import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_VERSIONS_DIR = _REPO_ROOT / "data" / "note_versions"
_lock = threading.Lock()


def save_note_version(
    encounter_id: str,
    note_text: str,
    source: str,
    conflicts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Save a new note version, computing diff from previous."""
    enc_dir = _VERSIONS_DIR / encounter_id
    enc_dir.mkdir(parents=True, exist_ok=True)

    existing = list_versions(encounter_id)
    version = len(existing) + 1

    diff_text: str | None = None
    if existing:
        prev = get_version(encounter_id, version - 1)
        if prev:
            diff_text = "\n".join(difflib.unified_diff(
                prev["note_text"].splitlines(),
                note_text.splitlines(),
                fromfile=f"v{version - 1}",
                tofile=f"v{version}",
                lineterm="",
            ))

    record: dict[str, Any] = {
        "encounter_id": encounter_id,
        "version": version,
        "created_at": datetime.now(UTC).isoformat(),
        "note_text": note_text,
        "source": source,
        "diff_from_previous": diff_text,
        "conflicts": conflicts or [],
    }

    path = enc_dir / f"v{version}.json"
    with _lock:
        path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")

    log.info("note_version.saved", encounter_id=encounter_id, version=version)
    return record


def get_latest_version(encounter_id: str) -> dict[str, Any] | None:
    versions = list_versions(encounter_id)
    if not versions:
        return None
    return versions[-1]


def get_version(encounter_id: str, version: int) -> dict[str, Any] | None:
    path = _VERSIONS_DIR / encounter_id / f"v{version}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def list_versions(encounter_id: str) -> list[dict[str, Any]]:
    enc_dir = _VERSIONS_DIR / encounter_id
    if not enc_dir.exists():
        return []
    results = []
    for p in sorted(enc_dir.glob("v*.json")):
        try:
            results.append(json.loads(p.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return results


def compute_diff(encounter_id: str, v_from: int, v_to: int) -> str | None:
    a = get_version(encounter_id, v_from)
    b = get_version(encounter_id, v_to)
    if not a or not b:
        return None
    return "\n".join(difflib.unified_diff(
        a["note_text"].splitlines(),
        b["note_text"].splitlines(),
        fromfile=f"v{v_from}",
        tofile=f"v{v_to}",
        lineterm="",
    ))
