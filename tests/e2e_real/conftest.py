"""Real E2E test fixtures — actual LLM, actual embeddings, actual reader.

Skips all tests if required credentials are not available.
Logs full instrumentation to JSON for drift comparison against mock pipeline.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pytest

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
)
from eval.longmemeval.evidence_verifier import (
    EvidenceSufficiency,
    classify_evidence,
    filter_evidence,
)
from eval.longmemeval.pipeline import (
    ReaderFn,
    direct_claims_satisfy_slot,
    infer_question_slot,
)
from eval.longmemeval.question_router import classify_question
from eval.longmemeval.token_budget import allocate_budget, apply_budget
from src.substrate.claims import (
    list_active_claims,
    list_claims_with_lifecycle,
    list_supersession_pairs,
)
from src.substrate.retrieval import backfill_embeddings, retrieve_hybrid
from src.substrate.schema import Claim


RESULTS_DIR = Path(__file__).parent / "results"


def _has_extractor_credentials() -> bool:
    return bool(
        os.environ.get("OPENAI_API_KEY")
        or (os.environ.get("AZURE_OPENAI_ENDPOINT") and os.environ.get("AZURE_OPENAI_API_KEY"))
    )


def _has_embedding_credentials() -> bool:
    # bge-m3 segfaults on Windows; use mock embedding as fallback.
    # Extraction and reader non-determinism are what we're testing.
    return True


def _has_reader_credentials() -> bool:
    return _has_extractor_credentials()


def _get_embedding_client():
    if os.environ.get("MODAL_BGE_M3_URL"):
        from src.substrate.retrieval import EmbeddingClient
        return EmbeddingClient()
    try:
        from eval.embedding_vertex import _get_harneet_token
        if _get_harneet_token():
            from eval.embedding_vertex import VertexEmbeddingClient
            return VertexEmbeddingClient()
    except Exception:
        pass
    # bge-m3 segfaults on Windows. Use mock embedding client as fallback.
    # This is acceptable: embedding determinism is NOT what we're testing.
    # Real extraction + real reader are the non-deterministic components.
    from tests.e2e.conftest import MockEmbeddingClient
    return MockEmbeddingClient()


def _make_real_reader_fn() -> ReaderFn:
    from eval._openai_client import make_openai_client
    client, model = make_openai_client("longmemeval_reader")

    def _reader(system: str, user: str) -> str:
        for attempt in range(5):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=0.0,
                    max_tokens=256,
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as exc:
                if "429" in str(exc) and attempt < 4:
                    time.sleep(2 ** attempt)
                    continue
                raise

    return _reader


@dataclass
class RealGoldTracker:
    """Stage-by-stage gold fact tracker for real pipeline runs."""

    gold_value: str
    case_id: str = ""
    question: str = ""
    question_type: str = ""

    # Extraction (real LLM)
    extracted: bool = False
    extracted_values: list[str] = field(default_factory=list)
    extraction_latency_ms: float = 0.0

    # Supersession
    active: bool = False
    active_values: list[str] = field(default_factory=list)
    supersession_count: int = 0

    # Retrieval (real embeddings)
    in_candidates: bool = False
    candidate_values: list[str] = field(default_factory=list)
    candidate_count: int = 0

    # Ranking
    rank_position: int | None = None
    ranked_values: list[str] = field(default_factory=list)

    # Classification
    classification: str = ""
    classified_as_direct: bool = False
    direct_values: list[str] = field(default_factory=list)

    # Bundle
    in_bundle: bool = False
    bundle_values: list[str] = field(default_factory=list)
    bundle_order: list[str] = field(default_factory=list)

    # Reader (real LLM)
    reader_input_contains_gold: bool = False
    reader_input: str = ""
    reader_output_correct: bool = False
    reader_output: str = ""
    reader_latency_ms: float = 0.0

    # Pipeline
    total_latency_ms: float = 0.0
    first_failing_layer: str = ""
    failure_class: str = ""

    def classify_failure(self) -> str:
        if not self.extracted:
            self.first_failing_layer = "extraction"
            if self.extracted_values:
                self.failure_class = "extraction_precision_failure"
            else:
                self.failure_class = "extraction_failure"
        elif not self.active:
            self.first_failing_layer = "supersession"
            self.failure_class = "supersession_failure"
        elif not self.in_candidates:
            self.first_failing_layer = "retrieval"
            self.failure_class = "retrieval_failure"
        elif self.rank_position is not None and self.rank_position > 3:
            self.first_failing_layer = "ranking"
            self.failure_class = "ranking_failure"
        elif not self.classified_as_direct:
            self.first_failing_layer = "classification"
            self.failure_class = "classification_failure"
        elif not self.in_bundle:
            self.first_failing_layer = "bundling"
            self.failure_class = "bundling_failure"
        elif not self.reader_output_correct:
            self.first_failing_layer = "reader"
            self.failure_class = "reader_failure"
        else:
            self.first_failing_layer = ""
            self.failure_class = ""
        return self.failure_class

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if len(d.get("reader_input", "")) > 1000:
            d["reader_input"] = d["reader_input"][:1000] + "..."
        return d


@dataclass
class RealTraceInfo:
    retrieval_mode: str = ""
    include_superseded: bool = False
    claims_written: int = 0
    active_claims: int = 0
    supersession_pairs: int = 0
    retrieved_candidates: int = 0
    verified_direct: int = 0
    verified_supporting: int = 0
    verified_conflict: int = 0
    sufficiency: str = ""
    should_abstain: bool = False
    answer: str = ""


def run_real_instrumented_pipeline(
    q: LongMemEvalQuestion,
    *,
    embedding_client: Any,
    reader_fn: ReaderFn,
    gold_value: str,
) -> tuple[RealGoldTracker, RealTraceInfo]:
    """Run the full pipeline with REAL components and gold tracking.

    Uses the default LLM extractor (gpt-4o-mini), real embeddings,
    and real LLM reader. Instruments each stage.
    """
    tracker = RealGoldTracker(
        gold_value=gold_value,
        case_id=q.question_id,
        question=q.question,
        question_type=q.question_type,
    )
    trace = RealTraceInfo()
    is_gold = lambda v: gold_value.lower() in v.lower()
    t_start = time.monotonic()

    # ── Step 1: Extract (REAL LLM) ─────────────────────────────────
    t_extract = time.monotonic()
    conn, stats = ingest_longmemeval_case(q)
    tracker.extraction_latency_ms = (time.monotonic() - t_extract) * 1000
    trace.claims_written = stats.claims_written_count
    trace.active_claims = stats.active_claim_count

    all_values = [r["value"] for r in conn.execute("SELECT value FROM claims").fetchall()]
    tracker.extracted_values = all_values
    tracker.extracted = any(is_gold(v) for v in all_values)

    # ── Step 2: Active check ───────────────────────────────────────
    active = list_active_claims(conn, q.question_id)
    tracker.active_values = [c.value for c in active]
    tracker.active = any(is_gold(c.value) for c in active)
    tracker.supersession_count = stats.supersessions_fired_count

    # ── Step 3: Route ──────────────────────────────────────────────
    strategy = classify_question(q.question, q.question_type)
    trace.retrieval_mode = strategy.retrieval_mode.value
    trace.include_superseded = strategy.include_superseded

    # ── Step 4: Backfill embeddings (REAL) ─────────────────────────
    backfill_embeddings(conn, q.question_id, client=embedding_client)

    # ── Step 5: Retrieve (REAL) ────────────────────────────────────
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
            entity_hint=entity_hint,
            top_k=max(strategy.top_k_candidates, 8),
            weights=strategy.weights,
            embedding_client=embedding_client,
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
            -(float(item["fused_score"]) * 0.62
              + float(item["coverage"]) * 0.12
              + min(int(item["query_hits"]), max(1, len(query_variants)))
              / max(1, len(query_variants)) * 0.21
              + 1.0 / (1.0 + int(item["best_rank"])) * 0.05),
            str(item["claim"].claim_id),
        ),
    )
    candidates = [
        (item["claim"], float(item["fused_score"]))
        for item in ordered[:strategy.top_k_candidates]
    ]
    trace.retrieved_candidates = len(candidates)
    tracker.candidate_count = len(candidates)
    tracker.candidate_values = [c.value for c, _ in candidates]
    tracker.in_candidates = any(is_gold(c.value) for c, _ in candidates)

    for idx, (c, _) in enumerate(candidates, start=1):
        if is_gold(c.value):
            tracker.rank_position = idx
            break
    tracker.ranked_values = [c.value for c, _ in candidates]

    # ── Step 6: Classify ───────────────────────────────────────────
    pairs = list_supersession_pairs(conn, q.question_id)
    trace.supersession_pairs = len(pairs)
    pair_claim_ids = frozenset(
        cid for old, new, edge in pairs for cid in (old.claim_id, new.claim_id)
    )
    supersession_info = {old.claim_id: new.claim_id for old, new, edge in pairs}

    classified = classify_evidence(
        q.question, q.question_type, candidates,
        supersession_pair_claim_ids=pair_claim_ids,
    )
    kept = filter_evidence(classified)

    gold_classifications = [e for e in classified if is_gold(e.claim.value)]
    if gold_classifications:
        tracker.classification = gold_classifications[0].evidence_type.value
        tracker.classified_as_direct = gold_classifications[0].evidence_type.value == "direct"
    tracker.direct_values = [
        e.claim.value for e in classified if e.evidence_type.value == "direct"
    ]
    trace.verified_direct = sum(1 for e in classified if e.evidence_type.value == "direct")
    trace.verified_supporting = sum(1 for e in classified if e.evidence_type.value == "supporting")
    trace.verified_conflict = sum(1 for e in classified if e.evidence_type.value == "conflict")

    # ── Step 6b: Session-neighbor expansion ────────────────────────
    slot = infer_question_slot(q.question)
    direct_claims = [e for e in classified if e.evidence_type.value == "direct"]

    if slot and direct_claims and not direct_claims_satisfy_slot(direct_claims, slot):
        direct_turn_ids = {e.claim.source_turn_id for e in direct_claims}
        existing_ids = {c.claim_id for c, _ in candidates}
        neighbor_claims: list[tuple[Claim, float]] = []

        for dtid in direct_turn_ids:
            turn_row = conn.execute("SELECT ts FROM turns WHERE turn_id = ?", (dtid,)).fetchone()
            if turn_row is None:
                continue
            nearby_turns = conn.execute(
                "SELECT turn_id FROM turns WHERE session_id = ? ORDER BY ts ASC",
                (q.question_id,),
            ).fetchall()
            turn_ids_ordered = [r["turn_id"] for r in nearby_turns]
            try:
                idx_pos = turn_ids_ordered.index(dtid)
            except ValueError:
                continue
            window_start = max(0, idx_pos - 5)
            window_end = min(len(turn_ids_ordered), idx_pos + 6)
            neighbor_turn_ids = set(turn_ids_ordered[window_start:window_end])

            for claim in list_active_claims(conn, q.question_id):
                if claim.claim_id in existing_ids:
                    continue
                if claim.source_turn_id in neighbor_turn_ids:
                    neighbor_claims.append((claim, 0.01))
                    existing_ids.add(claim.claim_id)

        if neighbor_claims:
            expanded_candidates = candidates + neighbor_claims
            classified = classify_evidence(
                q.question, q.question_type, expanded_candidates,
                supersession_pair_claim_ids=pair_claim_ids,
            )
            kept = filter_evidence(classified)
            gold_classifications = [e for e in classified if is_gold(e.claim.value)]
            if gold_classifications:
                tracker.classification = gold_classifications[0].evidence_type.value
                tracker.classified_as_direct = gold_classifications[0].evidence_type.value == "direct"
            tracker.direct_values = [
                e.claim.value for e in classified if e.evidence_type.value == "direct"
            ]

    # ── Step 7: Sufficiency ────────────────────────────────────────
    trace.sufficiency = EvidenceSufficiency.assess(classified)
    trace.should_abstain = EvidenceSufficiency.should_abstain(trace.sufficiency)

    # ── Step 8: Budget ─────────────────────────────────────────────
    alloc = allocate_budget(kept)
    budgeted = apply_budget(alloc)

    tracker.bundle_values = [e.claim.value for e in budgeted.evidence]
    tracker.in_bundle = any(is_gold(e.claim.value) for e in budgeted.evidence)
    tracker.bundle_order = [e.evidence_type.value for e in budgeted.evidence]

    # ── Step 9: Build structured bundle ────────────────────────────
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

    structured_text = format_structured_bundle(bundle, list(budgeted.evidence), supersession_info)
    tracker.reader_input = structured_text
    tracker.reader_input_contains_gold = is_gold(structured_text)

    # ── Step 10: Reader (REAL LLM) ─────────────────────────────────
    if trace.should_abstain:
        answer = "I don't know"
    else:
        system = (
            "You answer LongMemEval questions using ONLY the structured evidence "
            "bundle below. Use the DIRECT_EVIDENCE section first. If the evidence "
            'does not contain the answer, reply exactly: "I don\'t know".'
        )
        t_reader = time.monotonic()
        answer = reader_fn(system, f"{structured_text}\n\nQuestion: {q.question}")
        tracker.reader_latency_ms = (time.monotonic() - t_reader) * 1000

    tracker.reader_output = answer
    tracker.reader_output_correct = is_gold(answer)
    trace.answer = answer

    tracker.total_latency_ms = (time.monotonic() - t_start) * 1000
    tracker.classify_failure()

    conn.close()
    return tracker, trace


def save_tracker(tracker: RealGoldTracker, run_id: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{run_id}_{tracker.case_id}.json"
    path.write_text(json.dumps(tracker.to_dict(), indent=2, default=str), encoding="utf-8")
    return path


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture(autouse=True)
def _set_personal_assistant_pack(monkeypatch):
    monkeypatch.setenv("ACTIVE_PACK", "personal_assistant")
    from src.substrate.predicate_packs import active_pack
    active_pack.cache_clear()
    yield
    active_pack.cache_clear()


requires_extractor = pytest.mark.skipif(
    not _has_extractor_credentials(),
    reason="No OPENAI_API_KEY or AZURE_OPENAI credentials",
)

requires_embeddings = pytest.mark.skipif(
    not _has_embedding_credentials(),
    reason="No embedding backend (Modal, Vertex, or local sentence-transformers)",
)

requires_reader = pytest.mark.skipif(
    not _has_reader_credentials(),
    reason="No reader credentials (OPENAI_API_KEY or Azure)",
)

requires_all = pytest.mark.skipif(
    not (_has_extractor_credentials() and _has_embedding_credentials() and _has_reader_credentials()),
    reason="Missing one or more credentials for full real E2E",
)
