"""Protected token budget for evidence bundles (Layer 6).

Prevents token bloat without losing critical evidence. DIRECT and CONFLICT
evidence is never dropped. SUPPORTING evidence can be compressed.
BACKGROUND evidence is dropped first when over budget.

Compression is deterministic (first-sentence truncation, no LLM).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from eval.longmemeval.evidence_verifier import ClassifiedEvidence, EvidenceType


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return max(1, len(text) // 4)


@dataclass(frozen=True, slots=True)
class BudgetAllocation:
    must_keep: tuple[ClassifiedEvidence, ...]
    compressible: tuple[ClassifiedEvidence, ...]
    droppable: tuple[ClassifiedEvidence, ...]
    total_tokens_estimate: int
    budget_tokens: int
    overflow: bool


@dataclass(frozen=True, slots=True)
class BudgetedEvidence:
    evidence: tuple[ClassifiedEvidence, ...]
    compressed_texts: dict[str, str]  # claim_id -> compressed text
    dropped_claim_ids: tuple[str, ...]
    final_token_estimate: int
    budget_tokens: int
    retried: bool


def _evidence_text(ev: ClassifiedEvidence) -> str:
    c = ev.claim
    return f"{c.subject} {c.predicate} {c.value}"


def _compress_text(text: str, max_chars: int = 80) -> str:
    """Truncate to first sentence or max_chars, whichever is shorter."""
    # First sentence boundary
    for sep in (". ", "! ", "? "):
        idx = text.find(sep)
        if idx != -1 and idx < max_chars:
            return text[: idx + 1]
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def allocate_budget(
    classified: list[ClassifiedEvidence],
    budget_tokens: int = 2000,
) -> BudgetAllocation:
    """Partition classified evidence into must_keep, compressible, droppable."""
    must_keep: list[ClassifiedEvidence] = []
    compressible: list[ClassifiedEvidence] = []
    droppable: list[ClassifiedEvidence] = []

    for ev in classified:
        if ev.evidence_type in (EvidenceType.DIRECT, EvidenceType.CONFLICT):
            must_keep.append(ev)
        elif ev.evidence_type == EvidenceType.SUPPORTING:
            compressible.append(ev)
        elif ev.evidence_type == EvidenceType.BACKGROUND:
            droppable.append(ev)
        # IRRELEVANT should already be filtered out

    total = sum(estimate_tokens(_evidence_text(ev)) for ev in classified)
    must_keep_tokens = sum(estimate_tokens(_evidence_text(ev)) for ev in must_keep)

    return BudgetAllocation(
        must_keep=tuple(must_keep),
        compressible=tuple(compressible),
        droppable=tuple(droppable),
        total_tokens_estimate=total,
        budget_tokens=budget_tokens,
        overflow=must_keep_tokens > budget_tokens,
    )


def apply_budget(
    allocation: BudgetAllocation,
    *,
    max_retries: int = 1,
    expansion_factor: float = 1.5,
) -> BudgetedEvidence:
    """Apply the budget: keep must_keep, compress supporting, drop background."""
    budget = allocation.budget_tokens
    retried = False

    # If must_keep alone exceeds budget, expand
    must_keep_tokens = sum(
        estimate_tokens(_evidence_text(ev)) for ev in allocation.must_keep
    )
    if must_keep_tokens > budget and max_retries > 0:
        budget = int(budget * expansion_factor)
        retried = True

    used = must_keep_tokens
    kept: list[ClassifiedEvidence] = list(allocation.must_keep)
    compressed_texts: dict[str, str] = {}
    dropped: list[str] = []

    # Add compressible items
    for ev in allocation.compressible:
        raw_text = _evidence_text(ev)
        raw_tokens = estimate_tokens(raw_text)
        compressed = _compress_text(raw_text)
        comp_tokens = estimate_tokens(compressed)

        if used + comp_tokens <= budget:
            kept.append(ev)
            if len(compressed) < len(raw_text):
                compressed_texts[ev.claim.claim_id] = compressed
            used += comp_tokens
        else:
            dropped.append(ev.claim.claim_id)

    # Add droppable items if room
    for ev in allocation.droppable:
        text_tokens = estimate_tokens(_evidence_text(ev))
        if used + text_tokens <= budget:
            kept.append(ev)
            used += text_tokens
        else:
            dropped.append(ev.claim.claim_id)

    return BudgetedEvidence(
        evidence=tuple(kept),
        compressed_texts=compressed_texts,
        dropped_claim_ids=tuple(dropped),
        final_token_estimate=used,
        budget_tokens=budget,
        retried=retried,
    )


__all__ = [
    "BudgetAllocation",
    "BudgetedEvidence",
    "allocate_budget",
    "apply_budget",
    "estimate_tokens",
]
