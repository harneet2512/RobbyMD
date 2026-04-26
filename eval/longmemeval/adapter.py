"""LongMemEval-S → substrate turn-stream adapter.

LongMemEval ships questions as records with `haystack_sessions` — a list of
session transcripts (each a list of `{role, content}` messages). The adapter
converts each session into a substrate write, mapping `user` → `patient`-like
role and `assistant` → `physician`-like role.

Caveat: LongMemEval is general-purpose chat memory, not clinical. We preserve
original `role` strings in turn metadata but use the substrate's `system`
speaker channel for writes so we don't conflate domains.
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from eval._common import Turn

# Official LongMemEval-S question_type labels in the released JSON.
QUESTION_CATEGORIES: frozenset[str] = frozenset(
    {
        "single-session-user",
        "single-session-assistant",
        "single-session-preference",
        "multi-session",
        "temporal-reasoning",
        "knowledge-update",
    }
)


@dataclass(frozen=True)
class LongMemEvalQuestion:
    """One LongMemEval-S question + its haystack."""

    question_id: str
    question: str
    answer: str
    question_type: str
    haystack_sessions: list[list[dict[str, str]]]
    haystack_session_ids: list[str] | None = None
    haystack_dates: list[str] | None = None


def _session_id_for(q: LongMemEvalQuestion, idx: int) -> str:
    if q.haystack_session_ids and idx < len(q.haystack_session_ids):
        return q.haystack_session_ids[idx]
    return f"{q.question_id}::session::{idx:03d}"


def session_to_turns(
    q: LongMemEvalQuestion, session_idx: int
) -> list[Turn]:
    """Convert one haystack session to a deterministic Turn list.

    Turn IDs embed the question ID + session index + message index so they're
    stable across runs and unique across sessions.
    """
    session = q.haystack_sessions[session_idx]
    session_id = _session_id_for(q, session_idx)
    turns: list[Turn] = []
    ts = 0
    for i, msg in enumerate(session):
        role = str(msg.get("role", "system")).lower()
        content = str(msg.get("content", ""))
        if not content:
            continue
        # LongMemEval is not clinical; map both roles to `system` so the
        # substrate's speaker channel isn't abused. Downstream retrieval still
        # sees the content unambiguously.
        speaker_kw = "system"
        turns.append(
            Turn(
                turn_id=f"{session_id}::msg::{i:04d}",
                speaker=speaker_kw,
                text=f"[{role}] {content}",
                ts=ts,
            )
        )
        ts += 1000
    return turns


def iter_questions(questions_path: Path) -> Iterable[LongMemEvalQuestion]:
    """Stream LongMemEval-S questions from a JSON file.

    LongMemEval ships `longmemeval_s.json` as a single top-level array. We
    stream via `json.load` (the full file fits comfortably — 500 records).
    """
    data = json.loads(questions_path.read_text(encoding="utf-8"))
    for obj in data:
        qt = str(obj.get("question_type", ""))
        if qt not in QUESTION_CATEGORIES:
            # Skip unknown categories loudly — keep counting toward the "did
            # the authors add a new category in the cleanup?" detector.
            print(f"[adapter] WARN unknown question_type: {qt!r} (question {obj.get('question_id')})")
        yield LongMemEvalQuestion(
            question_id=str(obj["question_id"]),
            question=str(obj["question"]),
            answer=str(obj["answer"]),
            question_type=qt,
            haystack_sessions=list(obj.get("haystack_sessions", [])),
            haystack_session_ids=obj.get("haystack_session_ids"),
            haystack_dates=obj.get("haystack_dates"),
        )
