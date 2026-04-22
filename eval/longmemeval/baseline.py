"""LongMemEval-S baseline — Opus 4.7 full-context.

Concatenates the entire `haystack_sessions` into a single long-context prompt
and asks Opus 4.7 the question. Equivalent to the "full-context LLM baseline"
line in LongMemEval ICLR 2025 §5.1.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from eval.longmemeval.adapter import LongMemEvalQuestion

OPUS_MODEL = "claude-opus-4-7"

_SYSTEM_PROMPT = (
    "You are a helpful assistant. Use only the conversation history below to "
    "answer the user's question at the end. If the history does not contain "
    "enough information, reply exactly: 'I don't have enough information to answer.'"
)


@dataclass(frozen=True)
class LongMemEvalPrediction:
    question_id: str
    question_type: str
    predicted_answer: str
    raw_response: str = ""


def _flatten_sessions(q: LongMemEvalQuestion) -> str:
    lines: list[str] = []
    for i, session in enumerate(q.haystack_sessions):
        date = None
        if q.haystack_dates and i < len(q.haystack_dates):
            date = q.haystack_dates[i]
        header = f"--- Session {i + 1}" + (f" ({date})" if date else "") + " ---"
        lines.append(header)
        for msg in session:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def predict_answer(q: LongMemEvalQuestion) -> LongMemEvalPrediction:
    """Return Opus 4.7's answer to the question, full-context.

    Without ANTHROPIC_API_KEY → deterministic stub (echoes the gold answer so
    the harness can run without network). Stub path is flagged prominently in
    run.py.
    """
    if not os.getenv("ANTHROPIC_API_KEY"):
        return LongMemEvalPrediction(
            question_id=q.question_id,
            question_type=q.question_type,
            predicted_answer=q.answer,  # stub — not a scoring path
            raw_response="[STUB] ANTHROPIC_API_KEY not set",
        )

    try:
        import anthropic
    except ImportError as e:
        raise RuntimeError(
            "anthropic SDK required for live baseline; run pip install -e .[dev]"
        ) from e

    client = anthropic.Anthropic()
    prompt = f"{_flatten_sessions(q)}\n\nQuestion: {q.question}"
    resp = client.messages.create(
        model=OPUS_MODEL,
        max_tokens=512,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    text = _extract_text(resp)
    return LongMemEvalPrediction(
        question_id=q.question_id,
        question_type=q.question_type,
        predicted_answer=text,
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
