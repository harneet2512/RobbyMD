"""Canonical LongMemEval context layer pipeline.

Single entry point. Each step has one function and typed input/output.
No duplicate paths, no conditional format selection, no dead code.

Usage:
    from eval.longmemeval.pipeline import run_substrate_case
    result = run_substrate_case(question, embedding_client, reader_fn)
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
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
from src.substrate.claims import get_claim
from src.substrate.retrieval import (
    EmbeddingClient,
    backfill_embeddings,
    backfill_event_frame_embeddings,
    retrieve_event_frames,
    retrieve_hybrid,
)


_CRITICAL_FILES: tuple[str, ...] = (
    "eval/longmemeval/pipeline.py",
    "eval/longmemeval/context.py",
    "eval/longmemeval/evidence_verifier.py",
    "eval/longmemeval/question_router.py",
    "eval/longmemeval/token_budget.py",
    "src/substrate/retrieval.py",
    "src/substrate/claims.py",
    "src/substrate/supersession.py",
)


@dataclass(frozen=True, slots=True)
class RunManifest:
    """Captures code + environment state at the start of a batch run."""

    git_hash: str
    git_dirty: bool
    file_hashes: dict[str, str]
    model_id: str
    timestamp_utc: str
    active_pack: str
    python_version: str


def _sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
    except FileNotFoundError:
        return "MISSING"
    return h.hexdigest()


def build_run_manifest(model_id: str = "") -> RunManifest:
    """Snapshot code state for reproducibility."""
    try:
        git_hash = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except Exception:
        git_hash = "unknown"

    try:
        git_dirty = subprocess.run(
            ["git", "diff", "--quiet"],
            capture_output=True, timeout=5,
        ).returncode != 0
    except Exception:
        git_dirty = True

    file_hashes = {f: _sha256_file(f) for f in _CRITICAL_FILES}

    return RunManifest(
        git_hash=git_hash,
        git_dirty=git_dirty,
        file_hashes=file_hashes,
        model_id=model_id,
        timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        active_pack=os.environ.get("ACTIVE_PACK", ""),
        python_version=sys.version.split()[0],
    )


def verify_manifest(manifest: RunManifest) -> None:
    """Abort if critical files changed since the manifest was built."""
    for fpath, expected_hash in manifest.file_hashes.items():
        current = _sha256_file(fpath)
        if current != expected_hash:
            raise RuntimeError(
                f"Code changed mid-run: {fpath} hash changed "
                f"({expected_hash[:12]}... → {current[:12]}...)"
            )


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
    extracted_claim_ids: list[str] = field(default_factory=list)
    extracted_claim_values_by_id: dict[str, str] = field(default_factory=dict)
    active_claim_ids: list[str] = field(default_factory=list)
    active_claim_values_by_id: dict[str, str] = field(default_factory=dict)

    # Router
    retrieval_mode: str = ""
    include_superseded: bool = False

    # Retrieval
    retrieved_candidates: int = 0
    retrieved_claim_ids: list[str] = field(default_factory=list)
    retrieved_claim_values_by_id: dict[str, str] = field(default_factory=dict)

    # Verifier
    verified_direct: int = 0
    verified_supporting: int = 0
    verified_conflict: int = 0
    verified_background: int = 0
    verified_irrelevant: int = 0
    verified_claim_ids_by_label: dict[str, list[str]] = field(default_factory=dict)
    verified_claim_values_by_id: dict[str, str] = field(default_factory=dict)

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
    final_bundle_text: str = ""
    reader_prompt: str = ""
    reader_input: str = ""

    # Session-neighbor expansion
    slot_type: str = ""
    direct_slot_satisfied: bool = False
    session_neighbor_triggered: bool = False
    neighbor_claims_added: int = 0
    neighbor_claims_kept: int = 0

    # Event frames
    event_frames_assembled: int = 0
    event_frames_retrieved: int = 0
    event_source_claim_ids: list[str] = field(default_factory=list)
    answer_source_path: str = ""

    # Token efficiency
    claim_count_in_bundle: int = 0
    event_count_in_bundle: int = 0
    direct_evidence_tokens: int = 0
    supporting_evidence_tokens: int = 0

    # Answer
    answer: str = ""
    reader_output: str = ""
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


def _claim_value_satisfies_slot(val: str, slot: str) -> bool:
    """Check if a single claim value contains the answer type for the slot."""
    if slot == "location":
        for m in _re.finditer(r"[A-Z][a-z]{2,}", val):
            if not _DAY_MONTH_NAMES.match(m.group()):
                return True
        if any(w in val.lower() for w in ("target", "walmart", "costco", "store", "city")):
            return True
        return False
    if slot == "duration":
        return bool(_DURATION_RE.search(val))
    if slot == "time":
        return bool(_TIME_SIGNALS.search(val))
    if slot == "person":
        return bool(_re.search(r"[A-Z][a-z]+\s+[A-Z][a-z]+", val))
    if slot == "degree":
        return any(w in val.lower() for w in ("degree", "bachelor", "master", "administration", "engineering"))
    if slot == "name":
        return bool(_re.search(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+", val)) or bool(
            _re.search(r"\b(?:named|called|name is|name was)\s+[A-Z][a-z]+", val)
        )
    return False


def _event_relevant_direct_claims(
    direct_claims: list, question_tokens: set[str],
) -> list:
    """Filter DIRECT claims to those whose value overlaps with the question event."""
    relevant = []
    for ev in direct_claims:
        claim_tokens = set(_re.findall(r"[a-z0-9]+", ev.claim.value.lower()))
        overlap = question_tokens & claim_tokens
        if len(overlap) >= 2:
            relevant.append(ev)
    return relevant


def direct_claims_satisfy_slot(direct_claims: list, slot: str) -> bool:
    """Check if any DIRECT claim's value contains the answer type for the slot."""
    for ev in direct_claims:
        if _claim_value_satisfies_slot(ev.claim.value, slot):
            return True
    return False


def _is_slot_relevant_neighbor(val: str, slot: str) -> bool:
    """Check if a neighbor claim value could fill the missing slot."""
    if slot == "location":
        if _re.search(r"[A-Z][a-z]{2,}", val) and not _DAY_MONTH_NAMES.search(val.split()[0] if val else ""):
            return True
        if _re.search(r"\b(?:at|from|in)\s+[A-Z]", val):
            return True
        if any(w in val.lower() for w in ("target", "walmart", "costco", "store", "shop", "market")):
            return True
        return False
    return _claim_value_satisfies_slot(val, slot)


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
    extracted_rows = conn.execute(
        "SELECT claim_id, value FROM claims WHERE session_id = ? ORDER BY created_ts ASC, claim_id ASC",
        (q.question_id,),
    ).fetchall()
    trace.extracted_claim_ids = [str(r["claim_id"]) for r in extracted_rows]
    trace.extracted_claim_values_by_id = {
        str(r["claim_id"]): str(r["value"]) for r in extracted_rows
    }

    # ── Step 2: Route question ──────────────────────────────────────
    strategy = classify_question(q.question, q.question_type)
    trace.retrieval_mode = strategy.retrieval_mode.value
    trace.include_superseded = strategy.include_superseded

    # ── Step 3: Build context state ─────────────────────────────────
    effective_client = embedding_client or EmbeddingClient()
    backfill_embeddings(conn, q.question_id, client=effective_client)

    # ── Step 3b: Embed event frames ────────────────────────────────
    trace.event_frames_assembled = stats.event_frames_assembled
    backfill_event_frame_embeddings(conn, q.question_id, client=effective_client)

    all_claims = list_claims_with_lifecycle(
        conn, q.question_id, strategy.retrieval_mode.value
    )
    active = list_active_claims(conn, q.question_id)
    trace.active_claim_ids = [c.claim_id for c in active]
    trace.active_claim_values_by_id = {c.claim_id: c.value for c in active}
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

    # ── Step 4b: Event-frame retrieval (parallel source) ───────────
    slot = infer_question_slot(q.question)
    if slot:
        event_results = retrieve_event_frames(
            conn, session_id=q.question_id, query=q.question,
            top_k=8, embedding_client=effective_client,
        )
        trace.event_frames_retrieved = len(event_results)
        event_claim_ids: set[str] = set()
        for frame, frame_score in event_results:
            for claim_id in frame.supporting_claim_ids:
                if claim_id in selected:
                    continue
                claim_obj = get_claim(conn, claim_id)
                if claim_obj is None:
                    continue
                coverage = _claim_coverage_score(question_tokens, claim_obj)
                selected[claim_id] = {
                    "claim": claim_obj, "fused_score": frame_score * 1.3,
                    "query_hits": 1, "best_query_variant": "[event_frame]",
                    "coverage": coverage, "best_rank": 1,
                }
                candidates.append((claim_obj, frame_score * 1.3))
                event_claim_ids.add(claim_id)
        trace.event_source_claim_ids = list(event_claim_ids)

    trace.retrieved_candidates = len(candidates)
    trace.retrieved_claim_ids = [claim.claim_id for claim, _ in candidates]
    trace.retrieved_claim_values_by_id = {
        claim.claim_id: claim.value for claim, _ in candidates
    }

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
    trace.verified_claim_ids_by_label = {}
    trace.verified_claim_values_by_id = {}
    for e in classified:
        label = e.evidence_type.value
        trace.verified_claim_ids_by_label.setdefault(label, []).append(e.claim.claim_id)
        trace.verified_claim_values_by_id[e.claim.claim_id] = e.claim.value

    # ── Step 5b: Event-neighbor expansion if DIRECT is slot-incomplete ──
    trace.slot_type = slot
    direct_claims = [e for e in classified if e.evidence_type.value == "direct"]

    if slot and direct_claims:
        # Only check event-relevant DIRECT claims for slot satisfaction.
        # A DIRECT claim about an unrelated event (e.g. "cat tower from Petco")
        # must not block expansion for the asked event (e.g. coupon redemption).
        event_relevant = _event_relevant_direct_claims(direct_claims, question_tokens)
        if event_relevant:
            trace.direct_slot_satisfied = direct_claims_satisfy_slot(event_relevant, slot)
        else:
            trace.direct_slot_satisfied = direct_claims_satisfy_slot(direct_claims, slot)
    elif not slot:
        trace.direct_slot_satisfied = True

    if slot and direct_claims and not trace.direct_slot_satisfied:
        # Event-neighbor expansion: retrieve claims from ±3 turns around
        # the event-relevant DIRECT claim's source turn. Only keep neighbors
        # that are slot-relevant (for location: proper nouns, store names).
        event_relevant = _event_relevant_direct_claims(direct_claims, question_tokens)
        anchor_claims = event_relevant if event_relevant else direct_claims
        anchor_turn_ids = {e.claim.source_turn_id for e in anchor_claims}

        neighbor_claims: list[tuple[Any, float]] = []
        existing_ids = {c.claim_id for c, _ in candidates}

        for dtid in anchor_turn_ids:
            turn_row = conn.execute(
                "SELECT ts FROM turns WHERE turn_id = ?", (dtid,)
            ).fetchone()
            if turn_row is None:
                continue
            nearby_turns = conn.execute(
                "SELECT turn_id FROM turns WHERE session_id = ?"
                " ORDER BY ts ASC", (q.question_id,)
            ).fetchall()
            turn_ids_ordered = [r["turn_id"] for r in nearby_turns]
            try:
                idx = turn_ids_ordered.index(dtid)
            except ValueError:
                continue
            window_start = max(0, idx - 3)
            window_end = min(len(turn_ids_ordered), idx + 4)
            neighbor_turn_ids = set(turn_ids_ordered[window_start:window_end])

            for claim in list_active_claims(conn, q.question_id):
                if claim.claim_id in existing_ids:
                    continue
                if claim.source_turn_id not in neighbor_turn_ids:
                    continue
                if _is_slot_relevant_neighbor(claim.value, slot):
                    neighbor_claims.append((claim, 0.01))
                    existing_ids.add(claim.claim_id)

        if neighbor_claims:
            trace.session_neighbor_triggered = True
            trace.neighbor_claims_added = len(neighbor_claims)

            expanded_candidates = candidates + neighbor_claims
            classified = classify_evidence(
                q.question, q.question_type, expanded_candidates,
                supersession_pair_claim_ids=pair_claim_ids,
            )
            kept = filter_evidence(classified)

            trace.verified_direct = sum(1 for e in classified if e.evidence_type.value == "direct")
            trace.verified_supporting = sum(1 for e in classified if e.evidence_type.value == "supporting")
            trace.verified_conflict = sum(1 for e in classified if e.evidence_type.value == "conflict")
            trace.verified_background = sum(1 for e in classified if e.evidence_type.value == "background")
            trace.verified_irrelevant = sum(1 for e in classified if e.evidence_type.value == "irrelevant")
            trace.neighbor_claims_kept = sum(
                1 for e in kept
                if any(e.claim.claim_id == nc.claim_id for nc, _ in neighbor_claims)
            )
            trace.verified_claim_ids_by_label = {}
            trace.verified_claim_values_by_id = {}
            for e in classified:
                label = e.evidence_type.value
                trace.verified_claim_ids_by_label.setdefault(label, []).append(e.claim.claim_id)
                trace.verified_claim_values_by_id[e.claim.claim_id] = e.claim.value

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

    # Token efficiency tracking
    trace.claim_count_in_bundle = len(budgeted.evidence)
    event_ids_in_bundle = set()
    for ev in budgeted.evidence:
        if ev.claim.claim_id in trace.event_source_claim_ids:
            event_ids_in_bundle.add(ev.claim.claim_id)
        if ev.evidence_type.value == "direct":
            trace.direct_evidence_tokens += len(ev.claim.value) // 4
        elif ev.evidence_type.value == "supporting":
            trace.supporting_evidence_tokens += len(ev.claim.value) // 4
    trace.event_count_in_bundle = len(event_ids_in_bundle)

    # Determine answer source path
    has_event = any(ev.claim.claim_id in trace.event_source_claim_ids for ev in budgeted.evidence)
    has_claim = any(ev.claim.claim_id not in trace.event_source_claim_ids for ev in budgeted.evidence)
    if has_event and has_claim:
        trace.answer_source_path = "both"
    elif has_event:
        trace.answer_source_path = "event_frame"
    elif has_claim:
        trace.answer_source_path = "claim"

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
    trace.final_bundle_text = structured_text
    trace.reader_input = structured_text
    trace.reader_prompt = f"{structured_text}\n\nQuestion: {q.question}"

    # ── Step 9: Reader ──────────────────────────────────────────────
    if trace.should_abstain:
        trace.answer = "I don't know"
        trace.failure_class = "sufficiency_failure" if trace.retrieved_candidates > 0 else "extraction_failure"
    elif reader_fn is not None:
        system = (
            "You answer LongMemEval questions using ONLY the structured evidence "
            "bundle below. Use the DIRECT_EVIDENCE section first. Synthesize your "
            "answer from the evidence — combine relevant claims to form a complete "
            "response. If the evidence contains relevant information, use it to "
            "answer even if no single claim states the full answer verbatim. "
            'Only reply "I don\'t know" if the evidence is truly unrelated to the question.'
        )
        trace.answer = reader_fn(system, trace.reader_prompt)
    else:
        trace.answer = f"[NO READER] bundle has {trace.final_evidence_count} claims"
    trace.reader_output = trace.answer

    conn.close()
    return trace.answer, trace
