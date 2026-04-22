"""Pluggable domain packs per `Eng_doc.md §4.2`.

A pack declares the closed predicate vocabulary, structured sub-slot schemas,
few-shot examples, and (optionally for clinical packs) an LR-table path. Packs
live under `predicate_packs/<pack_id>/` and are loaded once at startup by
whichever module needs the active pack.

Controlled by env var `ACTIVE_PACK` (default: `clinical_general`). Cached via
`@lru_cache`; set the env var BEFORE importing modules that call `active_pack()`,
or call `active_pack.cache_clear()` in tests that need to switch.

Addresses audit finding #2 from commit `8f0d9db`: few-shot examples move from
a hardcoded Python module constant into per-pack JSON, so loading a different
pack swaps in its own examples.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PACKS_DIR = _REPO_ROOT / "predicate_packs"


@dataclass(frozen=True, slots=True)
class FewShotExample:
    """One canonical prompt-response pair for in-context learning."""

    name: str
    scenario: str
    prior_turns: tuple[tuple[str, str], ...]  # (speaker, text)
    current_turn: tuple[str, str]
    active_claims_summary: str
    expected_output: str


@dataclass(frozen=True, slots=True)
class PredicatePack:
    """A domain pack registered with the extractor.

    Clinical packs set `lr_table_path` and carry a `differentials/` subdir with
    branches + LR tables. Non-clinical packs (e.g. `personal_assistant`) leave
    `lr_table_path` as `None` — the differential engine no-ops on empty LR.
    """

    pack_id: str
    predicate_families: frozenset[str]
    sub_slots: dict[str, frozenset[str]] = field(default_factory=dict)
    lr_table_path: Path | None = None
    few_shot_examples: tuple[FewShotExample, ...] = ()
    description: str = ""


def _parse_few_shot(raw: dict) -> FewShotExample:
    prior = tuple(
        (str(t.get("speaker", "")), str(t.get("text", "")))
        for t in raw.get("prior_turns", [])
    )
    cur = raw.get("current_turn", {})
    return FewShotExample(
        name=str(raw.get("name", "")),
        scenario=str(raw.get("scenario", "")),
        prior_turns=prior,
        current_turn=(str(cur.get("speaker", "")), str(cur.get("text", ""))),
        active_claims_summary=str(raw.get("active_claims_summary", "")),
        expected_output=str(raw.get("expected_output", "")),
    )


def load_pack(pack_dir: str | Path) -> PredicatePack:
    """Load a pack from `predicate_packs/<pack_id>/`.

    Expected files:
    - `predicates.json` — pack_id, predicate_families, optional sub_slots, optional description.
    - `few_shot_examples.json` — either `{"examples": [...]}` or a bare list.

    Clinical packs additionally contain `differentials/<complaint>/lr_table.json`;
    the first such path found (sorted) is stored as `lr_table_path`.
    """
    p = Path(pack_dir)
    if not p.is_absolute():
        p = _PACKS_DIR / pack_dir
    if not p.is_dir():
        raise FileNotFoundError(f"Pack directory not found: {p}")

    predicates_raw = json.loads((p / "predicates.json").read_text(encoding="utf-8"))
    examples_raw = json.loads((p / "few_shot_examples.json").read_text(encoding="utf-8"))

    pack_id = str(predicates_raw.get("pack_id", p.name))
    families = frozenset(predicates_raw["predicate_families"])
    sub_slots_raw = predicates_raw.get("sub_slots", {})
    sub_slots = {str(k): frozenset(v) for k, v in sub_slots_raw.items()}

    # Resolve `differentials/<complaint>/lr_table.json` for clinical packs.
    differentials_dir = p / "differentials"
    lr_table_path: Path | None = None
    if differentials_dir.is_dir():
        for complaint_dir in sorted(differentials_dir.iterdir()):
            if not complaint_dir.is_dir():
                continue
            cand = complaint_dir / "lr_table.json"
            if cand.is_file():
                lr_table_path = cand
                break

    example_entries = examples_raw.get("examples", examples_raw) if isinstance(examples_raw, dict) else examples_raw
    examples = tuple(_parse_few_shot(ex) for ex in example_entries)

    return PredicatePack(
        pack_id=pack_id,
        predicate_families=families,
        sub_slots=sub_slots,
        lr_table_path=lr_table_path,
        few_shot_examples=examples,
        description=str(predicates_raw.get("description", "")),
    )


@lru_cache(maxsize=None)
def active_pack() -> PredicatePack:
    """Return the active pack.

    Controlled by env var `ACTIVE_PACK` (default: `clinical_general`). Cached —
    tests that need to switch should call `active_pack.cache_clear()` after
    mutating the env var.
    """
    pack_id = os.environ.get("ACTIVE_PACK", "clinical_general")
    return load_pack(pack_id)
