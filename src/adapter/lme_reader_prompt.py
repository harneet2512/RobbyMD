"""Single-call reader prompt for LongMemEval.

Renders an `LMEAdapterOutput` into (system, user) strings for one OpenAI
chat.completions call. Replaces the Chain-of-Note two-call pattern in
`eval/longmemeval/reader_con.py` — the adapter has already walked
supersession chains, attached temporal metadata, and pre-computed any
conflict resolution, so the reader just needs to answer.
"""
from __future__ import annotations

from src.adapter.lme_qa import LMEAdapterOutput, ReaderClaim


READER_SYSTEM = (
    "You are a precise question-answering assistant. You answer questions about "
    "a user's conversation history based on retrieved evidence.\n\n"
    "RULES:\n"
    "- Answer based ONLY on the provided evidence claims.\n"
    "- If evidence is sufficient, give a direct, concise answer.\n"
    "- If evidence is partially relevant (similarity scores are low), give your "
    "best approximation and note uncertainty.\n"
    "- Say \"I don't know\" ONLY if zero claims were retrieved or all claims are "
    "completely irrelevant to the question.\n"
    "- For knowledge-update questions: if a conflict resolution is provided, use "
    "the CURRENT value as the answer.\n"
    "- For temporal-reasoning questions: use the timestamps on claims to reason "
    "about when events occurred."
)


READER_USER_TEMPLATE = (
    "QUESTION: {question}\n"
    "QUESTION TYPE: {question_type}\n\n"
    "RETRIEVED EVIDENCE ({n_claims} claims, retrieval confidence: "
    "{retrieval_confidence:.2f}):\n"
    "{formatted_claims}\n"
    "{conflict_section}"
    "Answer the question. Be concise. If you're uncertain, say so but still give "
    "your best answer."
)


def _format_ts(ts: int | None) -> str:
    return "open" if ts is None else str(ts)


def _format_claim(i: int, rc: ReaderClaim) -> str:
    lines = [
        f"[Claim {i}] (similarity: {rc.similarity_score:.2f}, "
        f"session: {rc.source_session_id}, confidence: {rc.confidence:.2f})",
        f'  "{rc.claim_text}"',
        f"  Valid: {_format_ts(rc.valid_from_ts)} → {_format_ts(rc.valid_until_ts)}",
    ]
    supersession_parts: list[str] = []
    if rc.supersedes is not None:
        supersession_parts.append(f"supersedes {rc.supersedes}")
    if rc.superseded_by is not None:
        supersession_parts.append(f"superseded by {rc.superseded_by}")
    if supersession_parts:
        lines.append(f"  ({'; '.join(supersession_parts)})")
    if rc.source_turn_text is not None:
        # Trim long turn text to keep the prompt bounded.
        snippet = rc.source_turn_text.strip().replace("\n", " ")
        if len(snippet) > 400:
            snippet = snippet[:400] + "…"
        lines.append(f'  Source turn: "{snippet}"')
    return "\n".join(lines)


def _format_conflict_section(resolution: str | None) -> str:
    if resolution is None:
        return ""
    return (
        f"CONFLICT RESOLUTION: {resolution}\n"
        "Use the current value when answering.\n\n"
    )


def format_prompt(
    question: str, question_type: str, output: LMEAdapterOutput
) -> tuple[str, str]:
    """Return `(system_prompt, user_prompt)` for a single reader call."""
    if not output.ranked_claims:
        formatted = "(none)"
    else:
        formatted = "\n".join(
            _format_claim(i, rc) for i, rc in enumerate(output.ranked_claims)
        )
    user = READER_USER_TEMPLATE.format(
        question=question,
        question_type=question_type,
        n_claims=len(output.ranked_claims),
        retrieval_confidence=output.retrieval_confidence,
        formatted_claims=formatted,
        conflict_section=_format_conflict_section(output.conflict_resolution),
    )
    return READER_SYSTEM, user
