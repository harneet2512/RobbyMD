"""Adversarial E2E test fixtures: deterministic mocks + gold-fact tracker.

Zero external dependencies. Every mock is deterministic across runs.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

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
    EvidenceType,
    classify_evidence,
    filter_evidence,
)
from eval.longmemeval.pipeline import (
    CaseTrace,
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
from src.substrate.on_new_turn import ExtractedClaim, ExtractorFn
from src.substrate.retrieval import backfill_embeddings, retrieve_hybrid
from src.substrate.schema import Claim, Turn


# ═══════════════════════════════════════════════════════════════════
# Mock Embedding Client
# ═══════════════════════════════════════════════════════════════════


class MockEmbeddingClient:
    """Deterministic bag-of-words embedding. Cosine = token overlap."""

    def __init__(self, dim: int = 128):
        self._dim = dim
        self._model_version = "mock-bow-v1"

    @property
    def model_version(self) -> str:
        return self._model_version

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        vec = [0.0] * self._dim
        for token in tokens:
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            idx = h % self._dim
            vec[idx] += 1.0
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec


# ═══════════════════════════════════════════════════════════════════
# Mock Extractor
# ═══════════════════════════════════════════════════════════════════


def make_mock_extractor(
    rules: list[tuple[str, list[ExtractedClaim]]],
) -> ExtractorFn:
    """Extractor matching turn.text against substring rules. First match wins."""

    def _extract(turn: Turn) -> list[ExtractedClaim]:
        text = turn.text
        for substring, claims in rules:
            if substring.lower() in text.lower():
                return list(claims)
        return []

    return _extract


# ═══════════════════════════════════════════════════════════════════
# Mock Reader
# ═══════════════════════════════════════════════════════════════════


def make_mock_reader(gold_answer: str) -> ReaderFn:
    """Returns gold_answer if it appears in the bundle text, else abstains."""

    def _read(system: str, user: str) -> str:
        if gold_answer.lower() in user.lower():
            return gold_answer
        return "I don't know"

    return _read


# ═══════════════════════════════════════════════════════════════════
# Gold Tracker
# ═══════════════════════════════════════════════════════════════════


@dataclass
class GoldTracker:
    """Stage-by-stage record of whether the gold fact survived."""

    gold_value: str

    # Extraction
    extracted: bool = False
    extracted_values: list[str] = field(default_factory=list)

    # Supersession
    active: bool = False
    active_values: list[str] = field(default_factory=list)

    # Retrieval
    in_candidates: bool = False
    candidate_values: list[str] = field(default_factory=list)

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

    # Reader
    reader_input_contains_gold: bool = False
    reader_input: str = ""
    reader_output_correct: bool = False
    reader_output: str = ""

    # Failure
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
        if len(d.get("reader_input", "")) > 500:
            d["reader_input"] = d["reader_input"][:500] + "..."
        return d


# ═══════════════════════════════════════════════════════════════════
# Instrumented Pipeline Runner
# ═══════════════════════════════════════════════════════════════════


@dataclass
class TraceInfo:
    """Lightweight trace for test assertions."""

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


def run_instrumented_pipeline(
    q: LongMemEvalQuestion,
    *,
    extractor: ExtractorFn,
    embedding_client: MockEmbeddingClient,
    gold_value: str,
    reader_fn: ReaderFn | None = None,
) -> tuple[GoldTracker, TraceInfo]:
    """Run the full pipeline with stage-by-stage gold tracking.

    Mirrors pipeline.py:run_substrate_case but keeps conn open and
    instruments each stage for the gold fact.
    """
    tracker = GoldTracker(gold_value=gold_value)
    trace = TraceInfo()
    is_gold = lambda v: gold_value.lower() in v.lower()

    # ── Step 1: Extract claims ─────────────────────────────────────
    conn, stats = ingest_longmemeval_case(q, extractor=extractor)
    trace.claims_written = stats.claims_written_count
    trace.active_claims = stats.active_claim_count

    all_values = [r["value"] for r in conn.execute("SELECT value FROM claims").fetchall()]
    tracker.extracted_values = all_values
    tracker.extracted = any(is_gold(v) for v in all_values)

    # ── Step 2: Active check ───────────────────────────────────────
    active = list_active_claims(conn, q.question_id)
    tracker.active_values = [c.value for c in active]
    tracker.active = any(is_gold(c.value) for c in active)

    # ── Step 3: Route question ─────────────────────────────────────
    strategy = classify_question(q.question, q.question_type)
    trace.retrieval_mode = strategy.retrieval_mode.value
    trace.include_superseded = strategy.include_superseded

    # ── Step 4: Backfill embeddings ────────────────────────────────
    backfill_embeddings(conn, q.question_id, client=embedding_client)

    # ── Step 5: Retrieve evidence ──────────────────────────────────
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
            conn,
            session_id=q.question_id,
            query=query,
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

    ordered = sorted(
        selected.values(),
        key=lambda item: (
            -(
                float(item["fused_score"]) * 0.62
                + float(item["coverage"]) * 0.12
                + min(int(item["query_hits"]), max(1, len(query_variants)))
                / max(1, len(query_variants))
                * 0.21
                + 1.0 / (1.0 + int(item["best_rank"])) * 0.05
            ),
            str(item["claim"].claim_id),
        ),
    )
    candidates = [
        (item["claim"], float(item["fused_score"]))
        for item in ordered[: strategy.top_k_candidates]
    ]
    trace.retrieved_candidates = len(candidates)

    tracker.candidate_values = [c.value for c, _ in candidates]
    tracker.in_candidates = any(is_gold(c.value) for c, _ in candidates)

    for idx, (c, _) in enumerate(candidates, start=1):
        if is_gold(c.value):
            tracker.rank_position = idx
            break
    tracker.ranked_values = [c.value for c, _ in candidates]

    # ── Step 6: Classify evidence ──────────────────────────────────
    pairs = list_supersession_pairs(conn, q.question_id)
    trace.supersession_pairs = len(pairs)
    pair_claim_ids = frozenset(
        cid for old, new, edge in pairs for cid in (old.claim_id, new.claim_id)
    )
    supersession_info = {old.claim_id: new.claim_id for old, new, edge in pairs}

    classified = classify_evidence(
        q.question,
        q.question_type,
        candidates,
        supersession_pair_claim_ids=pair_claim_ids,
    )
    kept = filter_evidence(classified)

    gold_classifications = [e for e in classified if is_gold(e.claim.value)]
    if gold_classifications:
        tracker.classification = gold_classifications[0].evidence_type.value
        tracker.classified_as_direct = (
            gold_classifications[0].evidence_type.value == "direct"
        )
    tracker.direct_values = [
        e.claim.value for e in classified if e.evidence_type.value == "direct"
    ]

    trace.verified_direct = sum(
        1 for e in classified if e.evidence_type.value == "direct"
    )
    trace.verified_supporting = sum(
        1 for e in classified if e.evidence_type.value == "supporting"
    )
    trace.verified_conflict = sum(
        1 for e in classified if e.evidence_type.value == "conflict"
    )

    # ── Step 6b: Session-neighbor expansion ────────────────────────
    slot = infer_question_slot(q.question)
    direct_claims = [e for e in classified if e.evidence_type.value == "direct"]

    if slot and direct_claims and not direct_claims_satisfy_slot(direct_claims, slot):
        direct_turn_ids = {e.claim.source_turn_id for e in direct_claims}
        existing_ids = {c.claim_id for c, _ in candidates}
        neighbor_claims: list[tuple[Claim, float]] = []

        for dtid in direct_turn_ids:
            turn_row = conn.execute(
                "SELECT ts FROM turns WHERE turn_id = ?", (dtid,)
            ).fetchone()
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
                q.question,
                q.question_type,
                expanded_candidates,
                supersession_pair_claim_ids=pair_claim_ids,
            )
            kept = filter_evidence(classified)
            # Re-check gold classification after expansion
            gold_classifications = [e for e in classified if is_gold(e.claim.value)]
            if gold_classifications:
                tracker.classification = gold_classifications[0].evidence_type.value
                tracker.classified_as_direct = (
                    gold_classifications[0].evidence_type.value == "direct"
                )
            tracker.direct_values = [
                e.claim.value
                for e in classified
                if e.evidence_type.value == "direct"
            ]

    # ── Step 7: Assess sufficiency ─────────────────────────────────
    trace.sufficiency = EvidenceSufficiency.assess(classified)
    trace.should_abstain = EvidenceSufficiency.should_abstain(trace.sufficiency)

    # ── Step 8: Apply token budget ─────────────────────────────────
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
        evidence_list.append(
            RetrievedEvidence(
                claim_id=claim.claim_id,
                session_id=claim.session_id,
                source_turn_id=claim.source_turn_id,
                subject=claim.subject,
                predicate=claim.predicate,
                value=claim.value,
                fused_score=ev.fused_score,
                rerank_score=0.0,
                query_hits=int(item.get("query_hits", 1)),
                best_query_variant=str(item.get("best_query_variant", "")),
                similarity_score=ev.fused_score,
                value_preview=_preview(claim.value),
                valid_from_ts=claim.valid_from_ts,
                valid_until_ts=claim.valid_until_ts,
            )
        )

    bundle = ContextBundle(
        question_id=q.question_id,
        question=q.question,
        question_type=q.question_type,
        query_variants=query_variants,
        evidence=tuple(evidence_list),
        conflict_notes=_summarise_conflicts([ev.claim for ev in budgeted.evidence]),
        retrieval_confidence=0.0,
        provenance={"retrieval_mode": trace.retrieval_mode},
    )

    structured_text = format_structured_bundle(
        bundle, list(budgeted.evidence), supersession_info
    )
    tracker.reader_input = structured_text
    tracker.reader_input_contains_gold = is_gold(structured_text)

    # ── Step 10: Reader ────────────────────────────────────────────
    if trace.should_abstain:
        answer = "I don't know"
    elif reader_fn is not None:
        system = (
            "You answer LongMemEval questions using ONLY the structured evidence "
            "bundle below. Use the DIRECT_EVIDENCE section first. If the evidence "
            'does not contain the answer, reply exactly: "I don\'t know".'
        )
        answer = reader_fn(system, f"{structured_text}\n\nQuestion: {q.question}")
    else:
        reader = make_mock_reader(gold_value)
        system = "Answer using ONLY the evidence."
        answer = reader(system, f"{structured_text}\n\nQuestion: {q.question}")

    tracker.reader_output = answer
    if gold_value == "N/A":
        tracker.reader_output_correct = "don't know" in answer.lower()
    else:
        tracker.reader_output_correct = is_gold(answer)

    trace.answer = answer
    tracker.classify_failure()

    conn.close()
    return tracker, trace


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture(autouse=True)
def _set_personal_assistant_pack(monkeypatch):
    """Every e2e test uses the personal_assistant predicate pack."""
    monkeypatch.setenv("ACTIVE_PACK", "personal_assistant")
    from src.substrate.predicate_packs import active_pack

    active_pack.cache_clear()
    yield
    active_pack.cache_clear()


@pytest.fixture
def mock_embedding_client() -> MockEmbeddingClient:
    return MockEmbeddingClient()
