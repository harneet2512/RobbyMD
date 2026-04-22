"""Chain-of-Note reader for the LongMemEval substrate-variant smoke.

Paper-faithful CoN (Yu et al. 2023, arXiv 2311.09210) — two LLM calls per
question over a retrieved evidence bundle:

1. **Note extraction**: pull the relevant facts from the retrieved claims
   into structured notes. Output is JSON with `item_id`, `verbatim_fact`,
   and a one-sentence `reason` per retained claim.
2. **Answer composition**: answer using ONLY the extracted notes. Abstain
   with "I don't know" when the notes don't contain the answer.

Why paper-faithful instead of substrate-aware? The plan (Stream A,
`reasons.md` entry "CoN paper-faithful over substrate-aware") locked this
choice to avoid contaminating the retrieval-head signal with a novel
prompt. Substrate-aware variants (e.g. feeding supersession edges into
the reader) can be explored in a follow-up once the base retrieval
numbers are known.

Both calls route via `eval._openai_client.make_openai_client(purpose=...)`
so the Azure / OpenAI routing is shared with every other eval caller. The
default `purpose="longmemeval_reader"` picks up gpt-4o-2024-08-06 when
the operator has deployed it, or falls back to gpt-4.1 with a WARN.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import structlog

from src.substrate.retrieval import RankedClaim

log = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class CoNNote:
    """One extracted note — the unit of evidence the reader uses."""

    item_id: str
    verbatim_fact: str
    reason: str
    claim_id: str | None = None


NOTE_EXTRACTION_SYSTEM = (
    "You are a careful note-taker. Given a question and a list of retrieved "
    "facts from prior conversation, extract ONLY the facts that are directly "
    "relevant to answering the question. Do not speculate; do not add new "
    "information. Return a JSON object with key \"notes\", a list of objects "
    "each having {\"item_id\": str, \"verbatim_fact\": str, \"reason\": str}. "
    "`verbatim_fact` must be copied verbatim from the provided fact. `reason` "
    "is one sentence explaining relevance. If no fact is relevant, return "
    "{\"notes\": []}."
)

ANSWER_SYSTEM = (
    "You are a careful assistant. Answer the user's question using ONLY the "
    "extracted notes provided. Do not use any outside knowledge. If the notes "
    "do not contain the answer, reply exactly: \"I don't know\". Give a "
    "concise answer, one to two sentences."
)


def _format_retrieved_claims_for_extraction(
    retrieved_claims: list[RankedClaim],
) -> str:
    """Render retrieved claims as a numbered list for the note-extraction call."""
    lines: list[str] = []
    for i, rc in enumerate(retrieved_claims):
        claim = rc.claim
        lines.append(
            f"[item_{i:03d}] (claim_id={claim.claim_id}, sim={rc.similarity_score:.2f}) "
            f"{claim.subject} / {claim.predicate} = {claim.value}"
        )
    return "\n".join(lines)


def _parse_notes(raw: str) -> list[dict[str, Any]]:
    """Parse the note-extraction response into a list of note dicts.

    Mirrors the error-tolerant shape handling in `eval.aci_bench.llm_medcon.parse_concepts`.
    Bare list, wrapper-keyed list (`notes`, `items`, `list`, `data`), or
    single-key dict whose only value is a list all accepted. Anything else
    logs a WARN and returns an empty list — one malformed response can't
    abort the whole pipeline.
    """
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        log.warning("con.notes_parse_failed", raw=raw[:200])
        return []

    items: list[Any] | None = None
    if isinstance(parsed, list):
        items = parsed
    elif isinstance(parsed, dict):
        for key in ("notes", "items", "list", "data", "extracted"):
            value = parsed.get(key)
            if isinstance(value, list):
                items = value
                break
        if items is None and len(parsed) == 1:
            only_value = next(iter(parsed.values()))
            if isinstance(only_value, list):
                items = only_value
        if items is None and len(parsed) == 0:
            # Empty object == "no notes relevant". Don't warn.
            return []
        if items is None:
            log.warning("con.notes_unexpected_shape", keys=list(parsed.keys())[:10])
            return []
    else:
        log.warning("con.notes_not_list_or_dict", type=type(parsed).__name__)
        return []

    out: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict):
            out.append(item)
    return out


def _notes_to_prompt_block(notes: list[CoNNote]) -> str:
    if not notes:
        return "(no notes were extracted from the retrieved facts.)"
    lines = ["Extracted notes:"]
    for n in notes:
        lines.append(f"- [{n.item_id}] {n.verbatim_fact}  (reason: {n.reason})")
    return "\n".join(lines)


def answer_with_con(
    question: str,
    retrieved_claims: list[RankedClaim],
    reader_purpose: str = "longmemeval_reader",
    *,
    client_pair: tuple[Any, str] | None = None,
    env: dict[str, str] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Two-call Chain-of-Note over a retrieved claim bundle.

    Parameters
    ----------
    question:
        User's question as asked in LongMemEval.
    retrieved_claims:
        Top-k `RankedClaim` objects from `retrieve_relevant_claims`.
    reader_purpose:
        `make_openai_client` purpose string. Defaults to `"longmemeval_reader"`
        (gpt-4o-2024-08-06 paper-faithful; gpt-4.1 under the documented
        fallback — see `eval/_openai_client.py`).
    client_pair:
        Pre-built `(client, model)` for tests. Production callers pass `None`
        and the factory resolves it from env.
    env:
        Injectable environment (passed through to `make_openai_client`).

    Returns
    -------
    (answer, provenance) where `provenance` carries:
        - `notes`: list of `CoNNote` as dicts
        - `retrieved_claim_ids`: ordered claim_ids fed to the extractor
        - `note_extraction_latency_ms`, `answer_latency_ms`, `total_latency_ms`
        - `reader_purpose`, `model`
    """
    if client_pair is None:
        from eval._openai_client import make_openai_client

        client, model = make_openai_client(reader_purpose, env=env)  # type: ignore[arg-type]
    else:
        client, model = client_pair

    retrieved_claim_ids = [rc.claim.claim_id for rc in retrieved_claims]

    # --- Call 1: note extraction ---
    extraction_user = (
        f"Question: {question}\n\n"
        f"Retrieved facts:\n{_format_retrieved_claims_for_extraction(retrieved_claims)}\n"
    )
    t_extract_start = time.monotonic()
    try:
        resp1 = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": NOTE_EXTRACTION_SYSTEM},
                {"role": "user", "content": extraction_user},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        raw_notes = resp1.choices[0].message.content or "{}"
    except Exception as exc:  # noqa: BLE001 — log + propagate up to caller
        log.warning("con.extraction_api_error", error=repr(exc)[:200])
        raise
    extract_latency_ms = (time.monotonic() - t_extract_start) * 1000

    note_dicts = _parse_notes(raw_notes)
    # Build typed note list, mapping verbatim facts back to the underlying claim.
    id_to_claim = {rc.claim.claim_id: rc.claim for rc in retrieved_claims}
    notes: list[CoNNote] = []
    for i, n in enumerate(note_dicts):
        item_id = str(n.get("item_id") or f"item_{i:03d}")
        verbatim = str(n.get("verbatim_fact") or "").strip()
        reason = str(n.get("reason") or "").strip()
        if not verbatim:
            continue
        # Best-effort: link back to a claim whose value appears in the verbatim fact.
        matched_claim_id: str | None = None
        for cid, claim in id_to_claim.items():
            if claim.value and claim.value in verbatim:
                matched_claim_id = cid
                break
        notes.append(
            CoNNote(
                item_id=item_id,
                verbatim_fact=verbatim,
                reason=reason,
                claim_id=matched_claim_id,
            )
        )

    # --- Call 2: answer composition ---
    answer_user = (
        f"{_notes_to_prompt_block(notes)}\n\n"
        f"Question: {question}\n"
    )
    t_answer_start = time.monotonic()
    try:
        resp2 = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": ANSWER_SYSTEM},
                {"role": "user", "content": answer_user},
            ],
            temperature=0.0,
            max_tokens=256,
        )
        answer = (resp2.choices[0].message.content or "").strip()
    except Exception as exc:  # noqa: BLE001
        log.warning("con.answer_api_error", error=repr(exc)[:200])
        raise
    answer_latency_ms = (time.monotonic() - t_answer_start) * 1000

    # Abstention path: empty notes → "I don't know" is correct behaviour; the
    # Call-2 system prompt already instructs this, but we also coerce here so
    # downstream judges never score a hallucinated answer against an empty
    # retrieval bundle.
    if not notes:
        answer = "I don't know"

    provenance: dict[str, Any] = {
        "notes": [
            {
                "item_id": n.item_id,
                "verbatim_fact": n.verbatim_fact,
                "reason": n.reason,
                "claim_id": n.claim_id,
            }
            for n in notes
        ],
        "retrieved_claim_ids": retrieved_claim_ids,
        "note_extraction_latency_ms": extract_latency_ms,
        "answer_latency_ms": answer_latency_ms,
        "total_latency_ms": extract_latency_ms + answer_latency_ms,
        "reader_purpose": reader_purpose,
        "model": model,
    }
    return answer, provenance


__all__ = [
    "ANSWER_SYSTEM",
    "CoNNote",
    "NOTE_EXTRACTION_SYSTEM",
    "answer_with_con",
]
