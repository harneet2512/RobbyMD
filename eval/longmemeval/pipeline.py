"""Canonical LongMemEval context layer pipeline.

Single entry point. Each step has one function and typed input/output.
No duplicate paths, no conditional format selection, no dead code.

Usage:
    from eval.longmemeval.pipeline import run_substrate_case
    result = run_substrate_case(question, embedding_client, reader_fn)
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from eval.longmemeval.adapter import LongMemEvalQuestion
from eval.longmemeval.context import (
    ContextBundle,
    RetrievedEvidence,
    build_query_variants,
    format_structured_bundle,
    ingest_longmemeval_case,
    _claim_coverage_score,
    _collapse_ws,
    _extract_entity_hint,
    _preview,
    _summarise_conflicts,
    _tokenize,
    _clamp,
)
from eval.longmemeval.decision_snapshot import build_snapshot
from eval.longmemeval.evidence_verifier import (
    EvidenceSufficiency,
    classify_evidence,
    filter_evidence,
)
from eval.longmemeval.question_router import classify_question
from eval.longmemeval.token_budget import allocate_budget, apply_budget
from src.substrate.claims import (
    list_active_claims,
    list_claims_with_lifecycle,
    list_supersession_pairs,
)
from src.substrate.retrieval import EmbeddingClient, backfill_embeddings, retrieve_hybrid


@dataclass
class CaseTrace:
    """Full diagnostic trace of one case through the pipeline."""

    case_id: str
    question: str
    gold_answer: str
    question_type: str

    # Extraction
    claims_written: int = 0
    active_claims: int = 0
    superseded_claims: int = 0
    supersession_pairs: int = 0

    # Router
    retrieval_mode: str = ""
    include_superseded: bool = False

    # Retrieval
    retrieved_candidates: int = 0

    # Verifier
    verified_direct: int = 0
    verified_supporting: int = 0
    verified_conflict: int = 0
    verified_background: int = 0
    verified_irrelevant: int = 0

    # Sufficiency
    sufficiency: str = ""
    should_abstain: bool = False

    # Budget
    must_keep: int = 0
    compressible: int = 0
    droppable: int = 0
    dropped: int = 0
    bundle_tokens: int = 0
    budget_retried: bool = False

    # Bundle
    final_evidence_count: int = 0
    final_claim_ids: list[str] = field(default_factory=list)
    final_claim_values: list[str] = field(default_factory=list)
    final_claim_types: list[str] = field(default_factory=list)
    reader_input: str = ""

    # Session-neighbor expansion
    slot_type: str = ""
    direct_slot_satisfied: bool = False
    session_neighbor_triggered: bool = False
    neighbor_claims_added: int = 0
    neighbor_claims_kept: int = 0

    # Answer
    answer: str = ""
    failure_class: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


ReaderFn = Callable[[str, str], str]

# ── Slot completeness helpers ──────────────────────────────────────

import re as _re

_SLOT_PATTERNS: list[tuple[str, list[str]]] = [
    ("location", ["where", "what place", "which store", "what store", "which location"]),
    ("person", ["who", "what person", "whose"]),
    ("time", ["when", "what date", "what time", "what day"]),
    ("duration", ["how long", "how many minutes", "how many hours", "how much time"]),
    ("degree", ["what degree", "what did you study", "what major"]),
    ("name", ["what is the name", "what was the name", "what play", "what playlist", "what song"]),
]

_DURATION_RE = _re.compile(r"\d+\s*(?:minute|hour|day|week|month|year|min|hr)", _re.I)
_LOCATION_SIGNALS = _re.compile(r"[A-Z][a-z]{2,}|target|walmart|costco|store|city|town|at\s+\w+", _re.I)
_TIME_SIGNALS = _re.compile(r"\d{1,2}:\d{2}|\d{1,2}\s*(?:am|pm)|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", _re.I)


def infer_question_slot(question: str) -> str:
    ql = question.lower()
    for slot, patterns in _SLOT_PATTERNS:
        if any(p in ql for p in patterns):
            return slot
    return ""


_DAY_MONTH_NAMES = _re.compile(
    r"\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday"
    r"|January|February|March|April|May|June|July|August|September"
    r"|October|November|December)\b", _re.I,
)


def direct_claims_satisfy_slot(direct_claims: list, slot: str) -> bool:
    """Check if any DIRECT claim's value contains the answer type for the slot."""
    for ev in direct_claims:
        val = ev.claim.value
        if slot == "location":
            # Look for proper nouns that are NOT day/month names
            for m in _re.finditer(r"[A-Z][a-z]{2,}", val):
                if not _DAY_MONTH_NAMES.match(m.group()):
                    return True
            if any(w in val.lower() for w in ("target", "walmart", "costco", "store", "city")):
                return True
            continue
        if slot == "duration" and _DURATION_RE.search(val):
            return True
        if slot == "time" and _TIME_SIGNALS.search(val):
            return True
        if slot == "person" and _re.search(r"[A-Z][a-z]+\s+[A-Z][a-z]+", val):
            return True
        if slot == "degree" and any(w in val.lower() for w in ("degree", "bachelor", "master", "administration", "engineering")):
            return True
        if slot == "name" and _re.search(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+", val):
            return True
    return False


def run_substrate_case(
    q: LongMemEvalQuestion,
    *,
    embedding_client: EmbeddingClient | None = None,
    reader_fn: ReaderFn | None = None,
) -> tuple[str, CaseTrace]:
    """Run one LongMemEval case through the canonical pipeline.

    Returns (answer, trace) where trace captures every layer's output
    for diagnosis.
    """
    trace = CaseTrace(
        case_id=q.question_id,
        question=q.question,
        gold_answer=q.answer,
        question_type=q.question_type,
    )

    # ── Step 1: Extract claims ──────────────────────────────────────
    conn, stats = ingest_longmemeval_case(q)
    trace.claims_written = stats.claims_written_count
    trace.active_claims = stats.active_claim_count
    trace.superseded_claims = stats.supersessions_fired_count

    # ── Step 2: Route question ──────────────────────────────────────
    strategy = classify_question(q.question, q.question_type)
    trace.retrieval_mode = strategy.retrieval_mode.value
    trace.include_superseded = strategy.include_superseded

    # ── Step 3: Build context state ─────────────────────────────────
    effective_client = embedding_client or EmbeddingClient()
    backfill_embeddings(conn, q.question_id, client=effective_client)

    all_claims = list_claims_with_lifecycle(
        conn, q.question_id, strategy.retrieval_mode.value
    )
    active = list_active_claims(conn, q.question_id)
    pairs = list_supersession_pairs(conn, q.question_id)
    trace.supersession_pairs = len(pairs)

    pair_claim_ids = frozenset(
        cid for old, new, edge in pairs for cid in (old.claim_id, new.claim_id)
    )
    supersession_info = {old.claim_id: new.claim_id for old, new, edge in pairs}

    # ── Step 4: Retrieve evidence ───────────────────────────────────
    query_variants = build_query_variants(q.question)
    if strategy.temporal_boost:
        focus = " ".join(_tokenize(q.question)[:10])
        query_variants = query_variants + (_collapse_ws(f"temporal sequence and timing of {focus}"),)
    if strategy.update_boost:
        focus = " ".join(_tokenize(q.question)[:10])
        query_variants = query_variants + (_collapse_ws(f"what changed or was updated about {focus}"),)

    question_tokens = set(_tokenize(q.question))
    entity_hint = _extract_entity_hint(q.question)
    selected: dict[str, dict[str, Any]] = {}

    for query in query_variants:
        ranked = retrieve_hybrid(
            conn, session_id=q.question_id, query=query,
            entity_hint=entity_hint, top_k=max(strategy.top_k_candidates, 8),
            weights=strategy.weights, embedding_client=effective_client,
            include_superseded=strategy.include_superseded,
        )
        for rank_idx, (claim, fused_score) in enumerate(ranked, start=1):
            coverage = _claim_coverage_score(question_tokens, claim)
            key = claim.claim_id
            existing = selected.get(key)
            if existing is None:
                selected[key] = {
                    "claim": claim, "fused_score": fused_score,
                    "query_hits": 1, "best_query_variant": query,
                    "coverage": coverage, "best_rank": rank_idx,
                }
            else:
                existing["query_hits"] += 1
                existing["best_rank"] = min(int(existing["best_rank"]), rank_idx)
                if fused_score > float(existing["fused_score"]):
                    existing["fused_score"] = fused_score
                    existing["best_query_variant"] = query
                existing["coverage"] = max(float(existing["coverage"]), coverage)

    ordered = sorted(
        selected.values(),
        key=lambda item: (
            -(float(item["fused_score"]) * 0.62 + float(item["coverage"]) * 0.12
              + min(int(item["query_hits"]), max(1, len(query_variants))) / max(1, len(query_variants)) * 0.21
              + 1.0 / (1.0 + int(item["best_rank"])) * 0.05),
            str(item["claim"].claim_id),
        ),
    )
    candidates = [(item["claim"], float(item["fused_score"])) for item in ordered[:strategy.top_k_candidates]]
    trace.retrieved_candidates = len(candidates)

    # ── Step 5: Verify evidence ─────────────────────────────────────
    classified = classify_evidence(
        q.question, q.question_type, candidates,
        supersession_pair_claim_ids=pair_claim_ids,
    )
    kept = filter_evidence(classified)

    for e in classified:
        t = e.evidence_type.value
        if t == "direct": trace.verified_direct += 1
        elif t == "supporting": trace.verified_supporting += 1
        elif t == "conflict": trace.verified_conflict += 1
        elif t == "background": trace.verified_background += 1
        elif t == "irrelevant": trace.verified_irrelevant += 1

    # ── Step 5b: Session-neighbor expansion if DIRECT is slot-incomplete ──
    slot = infer_question_slot(q.question)
    trace.slot_type = slot
    direct_claims = [e for e in classified if e.evidence_type.value == "direct"]

    if slot and direct_claims:
        trace.direct_slot_satisfied = direct_claims_satisfy_slot(direct_claims, slot)
    elif not slot:
        trace.direct_slot_satisfied = True  # no slot to check

    if slot and direct_claims and not trace.direct_slot_satisfied:
        # Expand: find claims from the same session near the DIRECT claim's source turn
        direct_turn_ids = {e.claim.source_turn_id for e in direct_claims}
        # Get turn timestamps to find neighbors
        neighbor_claims: list[tuple[Any, float]] = []
        existing_ids = {c.claim_id for c, _ in candidates}

        for dtid in direct_turn_ids:
            turn_row = conn.execute(
                "SELECT ts FROM turns WHERE turn_id = ?", (dtid,)
            ).fetchone()
            if turn_row is None:
                continue
            direct_ts = turn_row["ts"]
            # Find turns within ±5 positions (by ordering) in the same session
            nearby_turns = conn.execute(
                "SELECT turn_id FROM turns WHERE session_id = ?"
                " ORDER BY ts ASC", (q.question_id,)
            ).fetchall()
            turn_ids_ordered = [r["turn_id"] for r in nearby_turns]
            try:
                idx = turn_ids_ordered.index(dtid)
            except ValueError:
                continue
            window_start = max(0, idx - 5)
            window_end = min(len(turn_ids_ordered), idx + 6)
            neighbor_turn_ids = set(turn_ids_ordered[window_start:window_end])

            # Find active claims from those turns not already in candidates
            for claim in list_active_claims(conn, q.question_id):
                if claim.claim_id in existing_ids:
                    continue
                if claim.source_turn_id in neighbor_turn_ids:
                    neighbor_claims.append((claim, 0.01))
                    existing_ids.add(claim.claim_id)

        if neighbor_claims:
            trace.session_neighbor_triggered = True
            trace.neighbor_claims_added = len(neighbor_claims)

            # Re-verify the expanded set
            expanded_candidates = candidates + neighbor_claims
            classified = classify_evidence(
                q.question, q.question_type, expanded_candidates,
                supersession_pair_claim_ids=pair_claim_ids,
            )
            kept = filter_evidence(classified)

            # Recount verifier labels
            trace.verified_direct = sum(1 for e in classified if e.evidence_type.value == "direct")
            trace.verified_supporting = sum(1 for e in classified if e.evidence_type.value == "supporting")
            trace.verified_conflict = sum(1 for e in classified if e.evidence_type.value == "conflict")
            trace.verified_background = sum(1 for e in classified if e.evidence_type.value == "background")
            trace.verified_irrelevant = sum(1 for e in classified if e.evidence_type.value == "irrelevant")
            trace.neighbor_claims_kept = sum(
                1 for e in kept
                if any(e.claim.claim_id == nc.claim_id for nc, _ in neighbor_claims)
            )

    # ── Step 6: Assess sufficiency ──────────────────────────────────
    trace.sufficiency = EvidenceSufficiency.assess(classified)
    trace.should_abstain = EvidenceSufficiency.should_abstain(trace.sufficiency)

    # ── Step 7: Apply token budget ──────────────────────────────────
    alloc = allocate_budget(kept)
    budgeted = apply_budget(alloc)
    trace.must_keep = len(alloc.must_keep)
    trace.compressible = len(alloc.compressible)
    trace.droppable = len(alloc.droppable)
    trace.dropped = len(budgeted.dropped_claim_ids)
    trace.bundle_tokens = budgeted.final_token_estimate
    trace.budget_retried = budgeted.retried

    # ── Step 8: Build structured bundle ─────────────────────────────
    evidence_list: list[RetrievedEvidence] = []
    for ev in budgeted.evidence:
        claim = ev.claim
        item = selected.get(claim.claim_id, {})
        evidence_list.append(RetrievedEvidence(
            claim_id=claim.claim_id, session_id=claim.session_id,
            source_turn_id=claim.source_turn_id, subject=claim.subject,
            predicate=claim.predicate, value=claim.value,
            fused_score=ev.fused_score, rerank_score=0.0,
            query_hits=int(item.get("query_hits", 1)),
            best_query_variant=str(item.get("best_query_variant", "")),
            similarity_score=ev.fused_score,
            value_preview=_preview(claim.value),
            valid_from_ts=claim.valid_from_ts, valid_until_ts=claim.valid_until_ts,
        ))

    bundle = ContextBundle(
        question_id=q.question_id, question=q.question,
        question_type=q.question_type, query_variants=query_variants,
        evidence=tuple(evidence_list),
        conflict_notes=_summarise_conflicts([ev.claim for ev in budgeted.evidence]),
        retrieval_confidence=0.0, provenance={"retrieval_mode": trace.retrieval_mode},
    )

    trace.final_evidence_count = len(evidence_list)
    trace.final_claim_ids = [e.claim_id for e in evidence_list]
    trace.final_claim_values = [ev.claim.value for ev in budgeted.evidence]
    trace.final_claim_types = [ev.evidence_type.value for ev in budgeted.evidence]

    structured_text = format_structured_bundle(bundle, list(budgeted.evidence), supersession_info)
    trace.reader_input = structured_text

    # ── Step 9: Reader ──────────────────────────────────────────────
    if trace.should_abstain:
        trace.answer = "I don't know"
        trace.failure_class = "sufficiency_failure" if trace.retrieved_candidates > 0 else "extraction_failure"
    elif reader_fn is not None:
        system = (
            "You answer LongMemEval questions using ONLY the structured evidence "
            "bundle below. Use the DIRECT_EVIDENCE section first. If the evidence "
            'does not contain the answer, reply exactly: "I don\'t know".'
        )
        trace.answer = reader_fn(system, f"{structured_text}\n\nQuestion: {q.question}")
    else:
        trace.answer = f"[NO READER] bundle has {trace.final_evidence_count} claims"

    conn.close()
    return trace.answer, trace
