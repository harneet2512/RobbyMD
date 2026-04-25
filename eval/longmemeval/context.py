"""LongMemEval benchmark-owned context repair layer.

This module keeps the benchmark-specific repair logic out of
`src/substrate/*`:

- retry-safe claim extraction with bounded exponential backoff
- deterministic query expansion
- multi-signal retrieval fusion and reranking
- evidence compression + stable formatting
- ingestion helpers shared by `full.py` and the smoke harness
"""
from __future__ import annotations

import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import structlog

from eval.longmemeval.adapter import LongMemEvalQuestion, session_to_turns
from src.extraction.claim_extractor.prompt import CLAIM_EXTRACTOR_SYSTEM_PROMPT
from src.substrate.claims import list_active_claims
from src.substrate.on_new_turn import ExtractedClaim, ExtractorFn, on_new_turn
from src.substrate.predicate_packs import active_pack
from src.substrate.retrieval import EmbeddingClient, backfill_embeddings, retrieve_hybrid
from src.substrate.schema import Claim, Speaker, open_database

log = structlog.get_logger(__name__)

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "do",
        "does",
        "for",
        "from",
        "get",
        "has",
        "have",
        "how",
        "i",
        "in",
        "is",
        "it",
        "just",
        "me",
        "my",
        "of",
        "on",
        "or",
        "please",
        "tell",
        "that",
        "the",
        "their",
        "there",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
        "you",
    }
)


@dataclass(frozen=True, slots=True)
class IngestionStats:
    claims_written_count: int = 0
    supersessions_fired_count: int = 0
    projection_nonempty: bool = False
    active_pack: str = ""
    active_claim_count: int = 0
    admitted_turn_count: int = 0
    empty_extraction_turn_count: int = 0
    event_frames_assembled: int = 0


@dataclass(frozen=True, slots=True)
class RetrievedEvidence:
    claim_id: str
    session_id: str
    source_turn_id: str
    subject: str
    predicate: str
    value: str
    fused_score: float
    rerank_score: float
    query_hits: int
    best_query_variant: str
    similarity_score: float
    value_preview: str
    valid_from_ts: int | None
    valid_until_ts: int | None


@dataclass(frozen=True, slots=True)
class ContextBundle:
    question_id: str
    question: str
    question_type: str
    query_variants: tuple[str, ...]
    evidence: tuple[RetrievedEvidence, ...]
    conflict_notes: tuple[str, ...]
    retrieval_confidence: float
    provenance: dict[str, Any]

    def format(self) -> str:
        return format_context_bundle(self)


def _collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9']+", text.lower())
    return [t for t in tokens if t and t not in _STOPWORDS]


def build_query_variants(question: str) -> tuple[str, ...]:
    """Deterministic query expansion for LongMemEval retrieval."""
    base = _collapse_ws(question)
    tokens = _tokenize(question)
    focus = " ".join(tokens[:10]) if tokens else base
    memory_focus = f"prior conversation facts about {focus}" if focus else base

    variants: list[str] = []
    for item in (base, focus, memory_focus):
        item = _collapse_ws(item)
        if item and item not in variants:
            variants.append(item)

    lowered = question.lower()
    if any(word in lowered for word in ("when", "date", "time", "before", "after")):
        temporal = _collapse_ws(f"temporal memory facts about {focus or base}")
        if temporal not in variants:
            variants.append(temporal)

    if any(word in lowered for word in ("who", "what", "which", "where")):
        entity = _collapse_ws(f"entity and fact memory about {focus or base}")
        if entity not in variants:
            variants.append(entity)

    return tuple(variants)


def _extract_entity_hint(question: str) -> str | None:
    quoted = re.findall(r'"([^"]+)"', question)
    if quoted:
        return _collapse_ws(quoted[0])
    # Prefer short title-cased spans as a light-weight entity hint.
    spans = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b", question)
    if spans:
        return _collapse_ws(spans[0])
    return None


def _is_transient_llm_error(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if status in (429, 503):
        return True
    resp = getattr(exc, "response", None)
    if resp is not None and getattr(resp, "status_code", None) in (429, 503):
        return True
    text = f"{type(exc).__name__} {exc}".lower()
    return "rate" in text and "limit" in text or "too many requests" in text or "503" in text or "service unavailable" in text


def _render_extracted_claims_for_prompt(
    turn_text: str,
    speaker_label: str,
    active_claims_summary: str = "(none provided; fresh extraction)",
) -> str:
    return (
        f"Current turn:\n    {speaker_label}: {turn_text}\n\n"
        f"Active claims: {active_claims_summary}\n"
    )


def make_retrying_longmemeval_extractor(
    *,
    env: Mapping[str, str] | None = None,
    purpose: str = "claim_extractor_gpt4omini",
    temperature: float = 0.0,
    max_attempts: int = 4,
    base_delay_s: float = 0.5,
    sleep_fn: Callable[[float], None] = time.sleep,
    active_claims_fn: Callable[[], str] | None = None,
) -> ExtractorFn:
    """Return a LongMemEval extractor with bounded retry/backoff.

    We do not use `src.extraction.claim_extractor.extractor.make_llm_extractor`
    because that factory collapses API failures into empty extractions, which
    makes rate limits look like retrieval failures. This wrapper preserves
    empty-turn semantics while retrying 429 / 503 responses explicitly.
    """

    cache: dict[str, Any] = {"client": None, "model": None}

    def _get_client() -> tuple[Any, str]:
        if cache["client"] is None:
            from eval._openai_client import make_openai_client

            client, model = make_openai_client(purpose, dict(env) if env is not None else None)
            cache["client"] = client
            cache["model"] = model
        return cache["client"], cache["model"]

    def _extract(turn: Any) -> list[ExtractedClaim]:
        text = str(getattr(turn, "text", "") or "")
        if not text.strip():
            log.info("longmemeval.extractor.empty_turn", turn_id=getattr(turn, "turn_id", None))
            return []

        speaker = getattr(turn, "speaker", "system")
        speaker_label = speaker.value if hasattr(speaker, "value") else str(speaker)
        claims_summary = "(none provided; fresh extraction)"
        if active_claims_fn is not None:
            try:
                claims_summary = active_claims_fn() or claims_summary
            except Exception:
                pass
        user_content = _render_extracted_claims_for_prompt(text, speaker_label, active_claims_summary=claims_summary)

        client, model = _get_client()
        last_error: Exception | None = None
        raw_content: str = "{}"
        for attempt in range(1, max_attempts + 1):
            try:
                response = client.chat.completions.create(  # type: ignore[attr-defined]
                    model=model,
                    messages=[
                        {"role": "system", "content": CLAIM_EXTRACTOR_SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=temperature,
                    response_format={"type": "json_object"},
                    timeout=30.0,
                )
                raw_content = response.choices[0].message.content or "{}"
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt < max_attempts and _is_transient_llm_error(exc):
                    delay = base_delay_s * (2 ** (attempt - 1))
                    log.warning(
                        "longmemeval.extractor.retry",
                        turn_id=getattr(turn, "turn_id", None),
                        attempt=attempt,
                        delay_s=delay,
                        error=repr(exc)[:200],
                    )
                    sleep_fn(delay)
                    continue
                log.warning(
                    "longmemeval.extractor.api_error",
                    turn_id=getattr(turn, "turn_id", None),
                    attempt=attempt,
                    error=repr(exc)[:200],
                )
                return []

        from src.extraction.claim_extractor.extractor import _parse_claims, _to_extracted_claims

        raw_claims = _parse_claims(raw_content)
        allowed_predicates = active_pack().predicate_families
        claims = _to_extracted_claims(raw_claims, turn, allowed_predicates)
        if not claims:
            if last_error is not None:
                log.warning(
                    "longmemeval.extractor.empty_after_retry",
                    turn_id=getattr(turn, "turn_id", None),
                    error=repr(last_error)[:200],
                )
            else:
                log.info("longmemeval.extractor.empty_extraction", turn_id=getattr(turn, "turn_id", None))
        return claims

    return _extract


def ingest_longmemeval_case(
    q: LongMemEvalQuestion,
    extractor: ExtractorFn | None = None,
) -> tuple[Any, IngestionStats]:
    """Ingest one LongMemEval question into an ephemeral substrate DB."""

    conn = open_database(":memory:")

    if extractor is None:
        def _active_claims_summary() -> str:
            active = list_active_claims(conn, q.question_id)
            if not active:
                return "(no active claims yet)"
            lines = [f"{c.claim_id}: {c.subject}.{c.predicate} = {c.value!r} (confidence {c.confidence})" for c in active[-5:]]
            summary = "; ".join(lines)
            if len(active) > 5:
                summary += f" ... ({len(active) - 5} more)"
            return summary

        extractor = make_retrying_longmemeval_extractor(active_claims_fn=_active_claims_summary)
    claims_written = 0
    supersessions_fired = 0
    admitted_turns = 0
    empty_extractions = 0

    for sidx in range(len(q.haystack_sessions)):
        for t in session_to_turns(q, sidx):
            result = on_new_turn(
                conn,
                session_id=q.question_id,
                speaker=Speaker(t.speaker),
                text=t.text,
                extractor=extractor,
            )
            if result.admitted:
                admitted_turns += 1
                claims_written += len(result.created_claims)
                supersessions_fired += len(result.supersession_edges)
                if not result.created_claims:
                    empty_extractions += 1

    active = list_active_claims(conn, q.question_id)

    from src.substrate.event_frames import assemble_event_frames
    event_frames = assemble_event_frames(conn, q.question_id)

    stats = IngestionStats(
        claims_written_count=claims_written,
        supersessions_fired_count=supersessions_fired,
        projection_nonempty=admitted_turns > 0,
        active_pack=os.environ.get("ACTIVE_PACK", ""),
        active_claim_count=len(active),
        admitted_turn_count=admitted_turns,
        empty_extraction_turn_count=empty_extractions,
        event_frames_assembled=len(event_frames),
    )
    return conn, stats


def _claim_text(claim: Claim) -> str:
    return f"{claim.subject} / {claim.predicate} = {claim.value}"


def _claim_coverage_score(question_tokens: set[str], claim: Claim) -> float:
    claim_tokens = set(_tokenize(_claim_text(claim)))
    if not question_tokens or not claim_tokens:
        return 0.0
    overlap = len(question_tokens & claim_tokens)
    return overlap / max(1, len(question_tokens))


def _summarise_conflicts(claims: list[Claim]) -> tuple[str, ...]:
    grouped: dict[tuple[str, str], set[str]] = defaultdict(set)
    for claim in claims:
        grouped[(claim.subject, claim.predicate)].add(claim.value.strip())
    notes: list[str] = []
    for (subject, predicate), values in sorted(grouped.items()):
        if len(values) > 1:
            rendered = " | ".join(sorted(values))
            notes.append(f"{subject} / {predicate}: conflicting active values -> {rendered}")
    return tuple(notes)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


# STALE: Superseded by eval/longmemeval/pipeline.py::run_substrate_case()
# which adds verifier, budget, structured bundle, expansion, and sufficiency layers.
def build_longmemeval_context(
    q: LongMemEvalQuestion,
    conn: Any,
    *,
    top_k: int = 8,
    embedding_client: EmbeddingClient | None = None,
) -> ContextBundle:
    """Return one deterministic evidence bundle for the reader."""

    query_variants = build_query_variants(q.question)
    question_tokens = set(_tokenize(q.question))
    entity_hint = _extract_entity_hint(q.question)
    effective_client = embedding_client or EmbeddingClient()
    backfill_embeddings(conn, q.question_id, client=effective_client)
    active_claims = list_active_claims(conn, q.question_id)
    selected: dict[str, dict[str, Any]] = {}

    for variant_idx, query in enumerate(query_variants):
        ranked = retrieve_hybrid(
            conn,
            session_id=q.question_id,
            query=query,
            entity_hint=entity_hint,
            top_k=max(top_k * 2, 8),
            embedding_client=effective_client,
        )
        for rank_idx, (claim, fused_score) in enumerate(ranked, start=1):
            coverage = _claim_coverage_score(question_tokens, claim)
            recency = float(claim.valid_from_ts or claim.created_ts)
            key = claim.claim_id
            existing = selected.get(key)
            if existing is None:
                selected[key] = {
                    "claim": claim,
                    "fused_score": fused_score,
                    "query_hits": 1,
                    "best_query_variant": query,
                    "coverage": coverage,
                    "recency": recency,
                    "best_rank": rank_idx,
                }
            else:
                existing["query_hits"] += 1
                existing["best_rank"] = min(int(existing["best_rank"]), rank_idx)
                if fused_score > float(existing["fused_score"]):
                    existing["fused_score"] = fused_score
                    existing["best_query_variant"] = query
                existing["coverage"] = max(float(existing["coverage"]), coverage)

    def _rerank_score(item: dict[str, Any]) -> float:
        query_hits = min(int(item["query_hits"]), max(1, len(query_variants)))
        hit_bonus = query_hits / max(1, len(query_variants))
        rank_bonus = 1.0 / (1.0 + int(item["best_rank"]))
        coverage = float(item["coverage"])
        fused = float(item["fused_score"])
        return (fused * 0.62) + (coverage * 0.12) + (hit_bonus * 0.21) + (rank_bonus * 0.05)

    ordered_items = sorted(
        selected.values(),
        key=lambda item: (
            -_rerank_score(item),
            -float(item["fused_score"]),
            -int(item["query_hits"]),
            -float(item["coverage"]),
            str(item["claim"].claim_id),
        ),
    )
    evidence: list[RetrievedEvidence] = []
    for item in ordered_items[:top_k]:
        claim = item["claim"]
        evidence.append(
            RetrievedEvidence(
                claim_id=claim.claim_id,
                session_id=claim.session_id,
                source_turn_id=claim.source_turn_id,
                subject=claim.subject,
                predicate=claim.predicate,
                value=claim.value,
                fused_score=float(item["fused_score"]),
                rerank_score=float(_rerank_score(item)),
                query_hits=int(item["query_hits"]),
                best_query_variant=str(item["best_query_variant"]),
                similarity_score=float(item["fused_score"]),
                value_preview=_preview(claim.value),
                valid_from_ts=claim.valid_from_ts,
                valid_until_ts=claim.valid_until_ts,
            )
        )

    selected_claims = [item["claim"] for item in ordered_items[:top_k]]
    conflict_notes = _summarise_conflicts(selected_claims)

    top_scores = [_rerank_score(item) for item in ordered_items[:top_k]]
    best = max(top_scores) if top_scores else 0.0
    avg = sum(top_scores) / len(top_scores) if top_scores else 0.0
    variant_mass = len({item["best_query_variant"] for item in ordered_items[:top_k]}) / max(1, len(query_variants))
    confidence = _clamp((best * 0.58) + (avg * 0.22) + (variant_mass * 0.20))

    provenance = {
        "question_id": q.question_id,
        "question_type": q.question_type,
        "session_id": q.question_id,
        "active_pack": os.environ.get("ACTIVE_PACK", ""),
        "question_tokens": len(question_tokens),
        "query_strategy": "hybrid_rrf_rerank_v1",
        "query_variants": list(query_variants),
        "top_k_requested": top_k,
        "top_k_retrieved": len(evidence),
        "top_k_similarity_mean": (sum(e.similarity_score for e in evidence) / len(evidence)) if evidence else 0.0,
        "top_k_similarity_min": min((e.similarity_score for e in evidence), default=0.0),
        "selected_claim_ids": [e.claim_id for e in evidence],
        "active_claim_count": len(active_claims),
        "entity_hint": entity_hint,
        "conflict_count": len(conflict_notes),
    }
    return ContextBundle(
        question_id=q.question_id,
        question=q.question,
        question_type=q.question_type,
        query_variants=query_variants,
        evidence=tuple(evidence),
        conflict_notes=conflict_notes,
        retrieval_confidence=confidence,
        provenance=provenance,
    )


def build_longmemeval_context_v2(
    q: LongMemEvalQuestion,
    conn: Any,
    *,
    embedding_client: EmbeddingClient | None = None,
    strategy: Any | None = None,
) -> tuple[ContextBundle, dict[str, Any]]:
    """Lifecycle-aware context builder integrating all upgraded layers.

    Returns ``(bundle, layer_outputs)`` where ``layer_outputs`` carries
    intermediate results for diagnostics (classified evidence, budget
    allocation, supersession pairs, decision snapshot).
    """
    from eval.longmemeval.question_router import classify_question
    from eval.longmemeval.evidence_verifier import classify_evidence, filter_evidence
    from eval.longmemeval.token_budget import allocate_budget, apply_budget, estimate_tokens
    from eval.longmemeval.decision_snapshot import build_snapshot

    # Layer 1: question-type router
    if strategy is None:
        strategy = classify_question(q.question, q.question_type)

    top_k = strategy.top_k_final
    top_k_candidates = strategy.top_k_candidates

    # Embedding backfill
    effective_client = embedding_client or EmbeddingClient()
    backfill_embeddings(conn, q.question_id, client=effective_client)

    # Layer 2: lifecycle-aware claim listing
    from src.substrate.claims import list_claims_with_lifecycle, list_supersession_pairs

    all_claims = list_claims_with_lifecycle(
        conn, q.question_id, strategy.retrieval_mode.value
    )
    active_claims = list_active_claims(conn, q.question_id)
    superseded_count = len(all_claims) - len(active_claims)

    # Supersession pairs for knowledge_update
    pairs = list_supersession_pairs(conn, q.question_id)
    pair_claim_ids = frozenset(
        cid
        for old, new, edge in pairs
        for cid in (old.claim_id, new.claim_id)
    )
    supersession_info: dict[str, str] = {}
    for old, new, edge in pairs:
        supersession_info[old.claim_id] = new.claim_id

    # Query expansion with strategy-driven variants
    query_variants = build_query_variants(q.question)
    if strategy.temporal_boost:
        focus = " ".join(_tokenize(q.question)[:10])
        temporal_v = _collapse_ws(f"temporal sequence and timing of {focus}")
        if temporal_v not in query_variants:
            query_variants = query_variants + (temporal_v,)
    if strategy.update_boost:
        focus = " ".join(_tokenize(q.question)[:10])
        update_v = _collapse_ws(f"what changed or was updated about {focus}")
        if update_v not in query_variants:
            query_variants = query_variants + (update_v,)

    # Layer 3: hybrid retrieval with BM25 (4-tuple weights) + lifecycle
    question_tokens = set(_tokenize(q.question))
    entity_hint = _extract_entity_hint(q.question)
    selected: dict[str, dict[str, Any]] = {}

    for variant_idx, query in enumerate(query_variants):
        ranked = retrieve_hybrid(
            conn,
            session_id=q.question_id,
            query=query,
            entity_hint=entity_hint,
            top_k=max(top_k_candidates, 8),
            weights=strategy.weights,
            embedding_client=effective_client,
            include_superseded=strategy.include_superseded,
        )
        for rank_idx, (claim, fused_score) in enumerate(ranked, start=1):
            coverage = _claim_coverage_score(question_tokens, claim)
            key = claim.claim_id
            existing = selected.get(key)
            if existing is None:
                selected[key] = {
                    "claim": claim,
                    "fused_score": fused_score,
                    "query_hits": 1,
                    "best_query_variant": query,
                    "coverage": coverage,
                    "best_rank": rank_idx,
                }
            else:
                existing["query_hits"] += 1
                existing["best_rank"] = min(int(existing["best_rank"]), rank_idx)
                if fused_score > float(existing["fused_score"]):
                    existing["fused_score"] = fused_score
                    existing["best_query_variant"] = query
                existing["coverage"] = max(float(existing["coverage"]), coverage)

    # Rerank
    def _rerank(item: dict[str, Any]) -> float:
        qh = min(int(item["query_hits"]), max(1, len(query_variants)))
        hit_bonus = qh / max(1, len(query_variants))
        rank_bonus = 1.0 / (1.0 + int(item["best_rank"]))
        return (
            float(item["fused_score"]) * 0.62
            + float(item["coverage"]) * 0.12
            + hit_bonus * 0.21
            + rank_bonus * 0.05
        )

    ordered = sorted(
        selected.values(),
        key=lambda item: (-_rerank(item), -float(item["fused_score"]), str(item["claim"].claim_id)),
    )

    # Layer 4: evidence verification
    candidates_for_verify = [
        (item["claim"], float(item["fused_score"]))
        for item in ordered[:top_k_candidates]
    ]
    classified = classify_evidence(
        q.question,
        q.question_type,
        candidates_for_verify,
        supersession_pair_claim_ids=pair_claim_ids,
    )
    kept = filter_evidence(classified)

    # Layer 6: token budget
    alloc = allocate_budget(kept, budget_tokens=2000)
    budgeted = apply_budget(alloc)

    # Build evidence list from budgeted evidence
    evidence: list[RetrievedEvidence] = []
    for ev in budgeted.evidence:
        claim = ev.claim
        item = selected.get(claim.claim_id, {})
        evidence.append(
            RetrievedEvidence(
                claim_id=claim.claim_id,
                session_id=claim.session_id,
                source_turn_id=claim.source_turn_id,
                subject=claim.subject,
                predicate=claim.predicate,
                value=claim.value,
                fused_score=ev.fused_score,
                rerank_score=_rerank(item) if item else 0.0,
                query_hits=int(item.get("query_hits", 1)),
                best_query_variant=str(item.get("best_query_variant", "")),
                similarity_score=ev.fused_score,
                value_preview=_preview(claim.value),
                valid_from_ts=claim.valid_from_ts,
                valid_until_ts=claim.valid_until_ts,
            )
        )

    selected_claims = [ev.claim for ev in budgeted.evidence]
    conflict_notes = _summarise_conflicts(selected_claims)

    top_scores = [ev.fused_score for ev in budgeted.evidence]
    best = max(top_scores) if top_scores else 0.0
    avg = sum(top_scores) / len(top_scores) if top_scores else 0.0
    variant_mass = len({
        selected.get(ev.claim.claim_id, {}).get("best_query_variant", "")
        for ev in budgeted.evidence
    }) / max(1, len(query_variants))
    confidence = _clamp((best * 0.58) + (avg * 0.22) + (variant_mass * 0.20))

    provenance = {
        "question_id": q.question_id,
        "question_type": q.question_type,
        "session_id": q.question_id,
        "active_pack": os.environ.get("ACTIVE_PACK", ""),
        "question_tokens": len(question_tokens),
        "query_strategy": "hybrid_rrf_bm25_lifecycle_v2",
        "retrieval_mode": strategy.retrieval_mode.value,
        "query_variants": list(query_variants),
        "top_k_requested": top_k,
        "top_k_candidates": top_k_candidates,
        "top_k_retrieved": len(evidence),
        "top_k_similarity_mean": (sum(e.similarity_score for e in evidence) / len(evidence)) if evidence else 0.0,
        "top_k_similarity_min": min((e.similarity_score for e in evidence), default=0.0),
        "selected_claim_ids": [e.claim_id for e in evidence],
        "active_claim_count": len(active_claims),
        "superseded_claim_count": superseded_count,
        "entity_hint": entity_hint,
        "conflict_count": len(conflict_notes),
        "evidence_types": {
            t.value: sum(1 for ev in budgeted.evidence if ev.evidence_type == t)
            for t in set(ev.evidence_type for ev in budgeted.evidence)
        } if budgeted.evidence else {},
        "dropped_claim_ids": list(budgeted.dropped_claim_ids),
        "bundle_tokens": budgeted.final_token_estimate,
        "budget_retried": budgeted.retried,
        "include_superseded": strategy.include_superseded,
    }

    bundle = ContextBundle(
        question_id=q.question_id,
        question=q.question,
        question_type=q.question_type,
        query_variants=query_variants,
        evidence=tuple(evidence),
        conflict_notes=conflict_notes,
        retrieval_confidence=confidence,
        provenance=provenance,
    )

    # Layer 8: decision snapshot
    snapshot = build_snapshot(
        question_id=q.question_id,
        question_type=q.question_type,
        retrieval_mode=strategy.retrieval_mode.value,
        active_claim_count=len(active_claims),
        superseded_claim_count=superseded_count,
        retrieved_candidate_count=len(candidates_for_verify),
        classified_evidence=list(budgeted.evidence),
        bundle_tokens=budgeted.final_token_estimate,
        answer="",  # filled by caller after reader
        depended_on_claim_ids=[e.claim_id for e in evidence],
        excluded_superseded_ids=list(budgeted.dropped_claim_ids),
        unresolved_conflicts=list(conflict_notes),
        retrieval_confidence=confidence,
    )

    layer_outputs = {
        "strategy": strategy,
        "classified_evidence": classified,
        "budgeted_evidence": budgeted,
        "supersession_pairs": pairs,
        "supersession_info": supersession_info,
        "snapshot": snapshot,
        "diagnostics": {
            "question_type": q.question_type,
            "retrieval_mode": strategy.retrieval_mode.value,
            "claims_written": len(all_claims),
            "active_claims": len(active_claims),
            "superseded_claims": superseded_count,
            "retrieved_candidates": len(candidates_for_verify),
            "verified_direct": sum(1 for e in classified if e.evidence_type.value == "direct"),
            "verified_supporting": sum(1 for e in classified if e.evidence_type.value == "supporting"),
            "verified_conflict": sum(1 for e in classified if e.evidence_type.value == "conflict"),
            "verified_background": sum(1 for e in classified if e.evidence_type.value == "background"),
            "verified_irrelevant": sum(1 for e in classified if e.evidence_type.value == "irrelevant"),
            "bundle_evidence_count": len(evidence),
            "bundle_tokens": budgeted.final_token_estimate,
            "dropped_count": len(budgeted.dropped_claim_ids),
            "budget_retried": budgeted.retried,
            "include_superseded": strategy.include_superseded,
            "entity_hint": entity_hint,
            "query_variant_count": len(query_variants),
            "supersession_pair_count": len(pairs),
        },
    }

    return bundle, layer_outputs


def _preview(text: str, limit: int = 120) -> str:
    cleaned = _collapse_ws(text)
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + "..."


def format_context_bundle(bundle: ContextBundle) -> str:
    """Render the bundle in a stable, reader-friendly format."""
    lines: list[str] = []
    lines.append("LONGMEMEVAL EVIDENCE BUNDLE")
    lines.append(f"Question ID: {bundle.question_id}")
    lines.append(f"Question Type: {bundle.question_type}")
    lines.append(f"Question: {_collapse_ws(bundle.question)}")
    lines.append(f"Retrieval confidence: {bundle.retrieval_confidence:.3f}")
    lines.append("Query variants:")
    for idx, variant in enumerate(bundle.query_variants, start=1):
        lines.append(f"  {idx}. {variant}")
    lines.append("Evidence:")
    if bundle.evidence:
        for idx, ev in enumerate(bundle.evidence, start=1):
            lines.append(
                f"  {idx}. [{ev.claim_id}] score={ev.rerank_score:.3f} "
                f"hits={ev.query_hits} fused={ev.fused_score:.3f}"
            )
            lines.append(f"     {ev.subject} / {ev.predicate} = {ev.value}")
            lines.append(
                f"     turn={ev.source_turn_id} valid_from={ev.valid_from_ts} valid_until={ev.valid_until_ts}"
            )
    else:
        lines.append("  (no evidence retrieved)")
    lines.append("Conflict notes:")
    if bundle.conflict_notes:
        for note in bundle.conflict_notes:
            lines.append(f"  - {note}")
    else:
        lines.append("  - none")
    lines.append("Provenance:")
    for key in (
        "question_id",
        "question_type",
        "session_id",
        "active_pack",
        "query_strategy",
        "top_k_requested",
        "top_k_retrieved",
        "top_k_similarity_mean",
        "top_k_similarity_min",
        "active_claim_count",
        "entity_hint",
        "conflict_count",
    ):
        lines.append(f"  {key}={bundle.provenance.get(key)}")
    return "\n".join(lines)


def format_structured_bundle(
    bundle: ContextBundle,
    classified: list[Any] | None = None,
    supersession_info: dict[str, str] | None = None,
) -> str:
    """Render evidence bundle with structured sections.

    Sections: DIRECT_EVIDENCE, SUPPORTING_CONTEXT, CONFLICTS_OR_UPDATES,
    TIMELINE, EXCLUDED_SUPERSEDED_CONTEXT, METADATA.
    """
    lines: list[str] = []
    lines.append(f"Question: {_collapse_ws(bundle.question)}")

    if not classified:
        lines.append("\n== EVIDENCE ==")
        for ev in bundle.evidence:
            lines.append(f"- [{ev.claim_id}] {ev.subject}/{ev.predicate} = {ev.value}")
        return "\n".join(lines)

    ss = supersession_info or {}

    direct = [e for e in classified if e.evidence_type.value == "direct"]
    supporting = [e for e in classified if e.evidence_type.value == "supporting"]
    conflicts = [e for e in classified if e.evidence_type.value == "conflict"]
    background = [e for e in classified if e.evidence_type.value == "background"]

    if direct:
        lines.append("\n== DIRECT_EVIDENCE ==")
        for e in direct:
            c = e.claim
            status = c.status.value if hasattr(c.status, "value") else str(c.status)
            replaced = ss.get(c.claim_id, "")
            suffix = f" [replaced by {replaced}]" if replaced else ""
            lines.append(f"- [{c.claim_id}] (status={status}{suffix}) {c.subject}/{c.predicate} = {c.value}")

    if supporting:
        lines.append("\n== SUPPORTING_CONTEXT ==")
        for e in supporting:
            c = e.claim
            lines.append(f"- [{c.claim_id}] {c.subject}/{c.predicate} = {c.value}")

    if conflicts:
        lines.append("\n== CONFLICTS_OR_UPDATES ==")
        for e in conflicts:
            c = e.claim
            lines.append(f"- [{c.claim_id}] CONFLICTS WITH {e.conflict_with}: {c.subject}/{c.predicate} = {c.value}")

    if bundle.evidence:
        lines.append("\n== TIMELINE ==")
        sorted_ev = sorted(bundle.evidence, key=lambda x: x.valid_from_ts or 0)
        for ev in sorted_ev[:5]:
            lines.append(f"- [{ev.claim_id}] from={ev.valid_from_ts} until={ev.valid_until_ts} {ev.value[:60]}")

    if background:
        lines.append("\n== BACKGROUND ==")
        for e in background[:3]:
            c = e.claim
            lines.append(f"- [{c.claim_id}] {c.value[:80]}")

    lines.append("\n== METADATA ==")
    lines.append(f"- question_type: {bundle.question_type}")
    lines.append(f"- retrieval_mode: {bundle.provenance.get('retrieval_mode', 'unknown')}")
    lines.append(f"- evidence_count: {len(bundle.evidence)}")
    lines.append(f"- direct_count: {len(direct)}")
    lines.append(f"- conflict_count: {len(conflicts)}")

    return "\n".join(lines)


def evidence_prompt(question: str, bundle: ContextBundle) -> tuple[str, str]:
    """Return `(system_prompt, user_prompt)` for the one-shot reader call."""
    system = (
        "You answer LongMemEval questions using ONLY the evidence bundle and "
        "the question below. If the bundle does not contain enough information, "
        'reply exactly: "I don\'t know". Do not mention the bundle or your reasoning.'
    )
    user = (
        f"{bundle.format()}\n\n"
        f"Task: answer the question using only the bundle.\n"
        f"Question: {_collapse_ws(question)}"
    )
    return system, user


def apply_reader_prompt_guard(answer: str) -> str:
    answer = _collapse_ws(answer)
    if not answer:
        return "I don't know"
    return answer


def answer_with_bundle_anthropic(
    bundle: ContextBundle,
    *,
    model: str = "claude-opus-4-7",
) -> tuple[str, dict[str, Any]]:
    """Single evidence-grounded Anthropic reader call for `full.py`."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY required for LongMemEval full runner; refusing to swap models."
        )
    try:
        import anthropic
    except ImportError as exc:
        raise RuntimeError("anthropic SDK required for LongMemEval full runner") from exc

    client = anthropic.Anthropic()
    system, user = evidence_prompt(bundle.question, bundle)
    t0 = time.monotonic()
    resp = client.messages.create(
        model=model,
        max_tokens=256,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    latency_ms = (time.monotonic() - t0) * 1000
    text = _extract_anthropic_text(resp)
    answer = apply_reader_prompt_guard(text)
    provenance = {
        "reader_model": model,
        "reader_latency_ms": latency_ms,
        "bundle": bundle.provenance,
    }
    return answer, provenance


def _extract_anthropic_text(resp: object) -> str:
    try:
        blocks = resp.content  # type: ignore[attr-defined]
    except AttributeError:
        return ""
    for block in blocks:
        if getattr(block, "type", None) == "text":
            return str(getattr(block, "text", ""))
    return ""
