"""Shared types + substrate-write stub used by all three benchmark adapters.

Per CLAUDE.md §5.5, `wt-eval` scaffolds adapters only — it does NOT implement
the substrate. The substrate write API ships from `wt-engine`; until then, the
adapters here call `SubstrateStub` which mirrors the Eng_doc.md §4.1 data model
shape enough to validate adapter outputs.

When wt-engine publishes its write API:
1. Replace `SubstrateStub` with the real substrate import.
2. Each adapter's TODO marker (search for `TODO(wt-engine)`) flags the call site
   to swap.
3. No changes needed to `run.py` / `baseline.py`.

This stub is deliberately minimal — it records calls to an in-memory list so
tests can assert adapter behaviour without a live substrate.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal

Speaker = Literal["patient", "physician", "system"]


@dataclass(frozen=True)
class Turn:
    """A single conversation turn as ingested by the substrate.

    Mirrors the `turns` table in Eng_doc.md §4.1 (minus session_id /
    asr_confidence, which the substrate assigns at write time).
    """

    turn_id: str
    speaker: Speaker
    text: str
    ts: int  # ms since epoch OR monotonic session time; substrate doesn't care


@dataclass
class SubstrateStub:
    """In-memory stand-in for the substrate write API.

    `write_turn` / `write_turns` are the only methods the adapters call. When
    wt-engine ships the real substrate, this class is deleted and the real one
    is imported in its place (adapters reference `SubstrateStub` by name;
    run.py injects the instance).

    TODO(wt-engine): delete this stub once `src/substrate/` exposes a
    `ClaimStore` (or equivalent) with a `write_turn(turn: Turn)` method.
    """

    session_id: str
    turns: list[Turn] = field(default_factory=list)

    def write_turn(self, turn: Turn) -> None:
        self.turns.append(turn)

    def write_turns(self, turns: Iterable[Turn]) -> None:
        for t in turns:
            self.write_turn(t)

    def reset(self) -> None:
        self.turns.clear()


@dataclass(frozen=True)
class EvalCase:
    """Generic envelope for one benchmark case.

    `payload` is benchmark-specific (see per-benchmark adapter for shape).
    `case_id` is the benchmark's own ID (DDXPlus `patient_id`, LongMemEval
    `question_id`, ACI-Bench `encounter_id`).
    """

    case_id: str
    payload: object
