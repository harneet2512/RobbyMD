"""20-case diagnostic slice for category-based failure attribution.

Selects a stratified sample of LongMemEval questions, runs them through
the canonical pipeline, and reports per-category accuracy and
first-failing-layer breakdown.

Usage:
    python -m eval.longmemeval.diagnostic_slice --data data/longmemeval_oracle.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import structlog

from eval.longmemeval.adapter import LongMemEvalQuestion, iter_questions
from eval.longmemeval.pipeline import (
    CaseTrace,
    RunManifest,
    ReaderFn,
    build_run_manifest,
    infer_question_slot,
    run_substrate_case,
    verify_manifest,
)
from src.substrate.retrieval import EmbeddingClient

log = structlog.get_logger(__name__)

# Category mapping from oracle type names to diagnostic categories.
_CATEGORY_MAP: dict[str, str] = {
    "single-session-user": "information_extraction",
    "single-session-assistant": "information_extraction",
    "single-session-preference": "information_extraction",
    "multi-session": "multi_session_reasoning",
    "temporal-reasoning": "temporal_reasoning",
    "knowledge-update": "knowledge_update",
    "information_extraction": "information_extraction",
    "multi_session_reasoning": "multi_session_reasoning",
    "temporal_reasoning": "temporal_reasoning",
    "knowledge_update": "knowledge_update",
    "abstention": "abstention",
}

_SLOT_PATTERNS = re.compile(r"\b(?:where|what place|which store|when|what time|who|whose)\b", re.I)

DIAGNOSTIC_CATEGORIES = (
    "information_extraction",
    "event_slot_fill",
    "temporal_reasoning",
    "knowledge_update",
    "multi_session_reasoning",
)

# Enhanced failure taxonomy
FAILURE_TYPES = (
    "extraction_failure",
    "extraction_precision_failure",
    "supersession_failure",
    "indexing_failure",
    "event_assembly_failure",
    "retrieval_failure",
    "event_retrieval_failure",
    "ranking_failure",
    "classification_failure",
    "bundling_failure",
    "reader_failure",
    "scoring_failure",
    "unknown_failure",
)


@dataclass
class DiagnosticCase:
    """Result of one case through the diagnostic pipeline."""

    case_id: str
    category: str
    gold: str
    predicted: str
    score: float
    gold_answer: str = ""
    gold_normalized: str = ""
    predicted_answer: str = ""
    predicted_normalized: str = ""
    extracted_claim_ids: list[str] = field(default_factory=list)
    active_claim_ids: list[str] = field(default_factory=list)
    retrieved_claim_ids: list[str] = field(default_factory=list)
    verified_claim_ids_by_label: dict[str, list[str]] = field(default_factory=dict)
    final_bundle_claim_ids: list[str] = field(default_factory=list)
    final_bundle_claim_values: list[str] = field(default_factory=list)
    final_bundle_text: str = ""
    reader_prompt: str = ""
    reader_output: str = ""
    scorer_inputs: dict[str, str] = field(default_factory=dict)
    scorer_result: bool = False
    extracted_claim_values_by_id: dict[str, str] = field(default_factory=dict)
    active_claim_values_by_id: dict[str, str] = field(default_factory=dict)
    retrieved_claim_values_by_id: dict[str, str] = field(default_factory=dict)
    verified_claim_values_by_id: dict[str, str] = field(default_factory=dict)
    gold_claim_ids_extracted: list[str] = field(default_factory=list)
    gold_claim_ids_active: list[str] = field(default_factory=list)
    gold_claim_ids_retrieved: list[str] = field(default_factory=list)
    gold_claim_ids_verified: list[str] = field(default_factory=list)
    gold_claim_ids_in_final_bundle: list[str] = field(default_factory=list)
    gold_extracted_exact: bool = False
    gold_active_exact: bool = False
    gold_retrieved_exact: bool = False
    gold_verified_exact: bool = False
    gold_in_bundle_exact: bool = False
    reader_saw_gold_exact: bool = False
    gold_extracted: bool = False
    gold_active: bool = False
    gold_retrieved: bool = False
    gold_verified: bool = False
    gold_in_bundle: bool = False
    source_path: str = ""
    first_failing_layer: str = ""
    failure_class: str = ""
    event_frames_assembled: int = 0
    event_frames_contributed: bool = False
    bundle_tokens: int = 0
    claim_count_in_bundle: int = 0
    direct_evidence_tokens: int = 0
    supporting_evidence_tokens: int = 0
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _classify_diagnostic_category(q: LongMemEvalQuestion) -> str:
    """Map a question to its diagnostic category."""
    base = _CATEGORY_MAP.get(q.question_type, "information_extraction")
    if base == "information_extraction" and _SLOT_PATTERNS.search(q.question):
        return "event_slot_fill"
    return base


def _normalize_short_answer_text(text: str) -> str:
    """Normalize short answers while preserving numeric identity."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = re.sub(r"(?<=\d)[,_](?=\d)", "", normalized)
    chars: list[str] = []
    for ch in normalized:
        if unicodedata.category(ch) == "Sc":
            continue
        chars.append(ch if ch.isalnum() else " ")
    return re.sub(r"\s+", " ", "".join(chars)).strip()


def _short_answer_token_count(normalized: str) -> int:
    return len(re.findall(r"[a-z0-9]+", normalized))


def _contains_boundary_match(gold_norm: str, text_norm: str) -> bool:
    pattern = rf"(?<![a-z0-9]){re.escape(gold_norm)}(?![a-z0-9])"
    return re.search(pattern, text_norm) is not None


def _gold_in_text(gold: str, text: str) -> bool:
    """Check if the gold answer appears in the text, with strict short answers."""
    gold_norm = _normalize_short_answer_text(gold)
    text_norm = _normalize_short_answer_text(text)
    if _short_answer_token_count(gold_norm) <= 3:
        if not gold_norm or not text_norm:
            return False
        return gold_norm == text_norm or _contains_boundary_match(gold_norm, text_norm)

    gl = gold.lower().strip()
    tl = text.lower().strip()
    if gl in tl:
        return True
    _stops = {"a", "an", "the", "and", "or", "of", "in", "to", "for", "is", "was",
              "it", "that", "this", "with", "on", "at", "by", "from", "as", "be",
              "would", "not", "they", "their", "user", "prefer", "may", "also",
              "suggestions", "responses", "those", "related", "can", "have"}
    gold_tokens = set(re.findall(r"[a-z0-9]+", gl)) - _stops
    text_tokens = set(re.findall(r"[a-z0-9]+", tl)) - _stops
    if not gold_tokens:
        return False
    overlap = len(gold_tokens & text_tokens) / len(gold_tokens)
    return overlap >= 0.45


def _claim_matches_gold(gold: str, claim_value: str) -> bool:
    """Identify claim IDs whose values directly or partially express the gold."""
    return _gold_in_text(gold, claim_value) or _gold_in_text(claim_value, gold)


def _matching_claim_ids(gold: str, ids: list[str], values_by_id: dict[str, str]) -> list[str]:
    return [cid for cid in ids if _claim_matches_gold(gold, values_by_id.get(cid, ""))]


def _verified_ids_by_label(verified: dict[str, list[str]]) -> list[str]:
    ids: list[str] = []
    for label_ids in verified.values():
        ids.extend(label_ids)
    return ids


def _classify_exact_failure(dc: DiagnosticCase) -> None:
    if dc.scorer_result:
        dc.first_failing_layer = ""
        dc.failure_class = ""
    elif not dc.gold_claim_ids_extracted:
        dc.first_failing_layer = "extraction"
        dc.failure_class = "extraction_failure"
    elif not dc.gold_claim_ids_active:
        dc.first_failing_layer = "supersession"
        dc.failure_class = "supersession_failure"
    elif not dc.gold_claim_ids_retrieved:
        dc.first_failing_layer = "retrieval"
        dc.failure_class = "retrieval_failure"
    elif not dc.gold_claim_ids_verified:
        dc.first_failing_layer = "verifier"
        dc.failure_class = "verifier_failure"
    elif not dc.gold_claim_ids_in_final_bundle:
        dc.first_failing_layer = "bundling"
        dc.failure_class = "bundling_failure"
    elif not dc.scorer_result:
        dc.first_failing_layer = "reader"
        dc.failure_class = "reader_failure"


def _classify_failure_from_trace(trace: CaseTrace, gold: str) -> DiagnosticCase:
    """Build a DiagnosticCase from a CaseTrace, classifying the first failing layer."""
    scorer_result = _gold_in_text(gold, trace.answer)
    dc = DiagnosticCase(
        case_id=trace.case_id,
        category="",
        gold=gold,
        predicted=trace.answer,
        score=1.0 if scorer_result else 0.0,
        gold_answer=gold,
        gold_normalized=_normalize_short_answer_text(gold),
        predicted_answer=trace.answer,
        predicted_normalized=_normalize_short_answer_text(trace.answer),
        extracted_claim_ids=trace.extracted_claim_ids,
        active_claim_ids=trace.active_claim_ids,
        retrieved_claim_ids=trace.retrieved_claim_ids,
        verified_claim_ids_by_label=trace.verified_claim_ids_by_label,
        final_bundle_claim_ids=trace.final_claim_ids,
        final_bundle_claim_values=trace.final_claim_values,
        final_bundle_text=trace.final_bundle_text,
        reader_prompt=trace.reader_prompt,
        reader_output=trace.reader_output or trace.answer,
        scorer_inputs={"gold": gold, "predicted": trace.answer},
        scorer_result=scorer_result,
        extracted_claim_values_by_id=trace.extracted_claim_values_by_id,
        active_claim_values_by_id=trace.active_claim_values_by_id,
        retrieved_claim_values_by_id=trace.retrieved_claim_values_by_id,
        verified_claim_values_by_id=trace.verified_claim_values_by_id,
        event_frames_assembled=trace.event_frames_assembled,
        event_frames_contributed=len(trace.event_source_claim_ids) > 0,
        bundle_tokens=trace.bundle_tokens,
        claim_count_in_bundle=trace.claim_count_in_bundle,
        direct_evidence_tokens=trace.direct_evidence_tokens,
        supporting_evidence_tokens=trace.supporting_evidence_tokens,
        source_path=trace.answer_source_path,
    )
    dc.gold_claim_ids_extracted = _matching_claim_ids(
        gold, trace.extracted_claim_ids, trace.extracted_claim_values_by_id
    )
    dc.gold_claim_ids_active = _matching_claim_ids(
        gold, trace.active_claim_ids, trace.active_claim_values_by_id
    )
    dc.gold_claim_ids_retrieved = _matching_claim_ids(
        gold, trace.retrieved_claim_ids, trace.retrieved_claim_values_by_id
    )
    verified_ids = _verified_ids_by_label(trace.verified_claim_ids_by_label)
    dc.gold_claim_ids_verified = _matching_claim_ids(
        gold, verified_ids, trace.verified_claim_values_by_id
    )
    final_values_by_id = dict(zip(trace.final_claim_ids, trace.final_claim_values, strict=False))
    dc.gold_claim_ids_in_final_bundle = _matching_claim_ids(gold, trace.final_claim_ids, final_values_by_id)
    dc.gold_extracted_exact = bool(dc.gold_claim_ids_extracted)
    dc.gold_active_exact = bool(dc.gold_claim_ids_active)
    dc.gold_retrieved_exact = bool(dc.gold_claim_ids_retrieved)
    dc.gold_verified_exact = bool(dc.gold_claim_ids_verified)
    dc.gold_in_bundle_exact = bool(dc.gold_claim_ids_in_final_bundle)
    dc.gold_extracted = dc.gold_extracted_exact
    dc.gold_active = dc.gold_active_exact
    dc.gold_retrieved = dc.gold_retrieved_exact
    dc.gold_verified = dc.gold_verified_exact
    dc.gold_in_bundle = dc.gold_in_bundle_exact
    dc.reader_saw_gold_exact = _gold_in_text(gold, trace.reader_prompt or trace.reader_input)
    _classify_exact_failure(dc)
    return dc

def select_diagnostic_slice(
    questions_path: Path,
    n_per_category: int = 4,
) -> list[LongMemEvalQuestion]:
    """Select a stratified diagnostic slice from the oracle."""
    by_category: dict[str, list[LongMemEvalQuestion]] = defaultdict(list)

    for q in iter_questions(questions_path):
        cat = _classify_diagnostic_category(q)
        by_category[cat].append(q)

    # Ensure information_extraction has non-slot questions
    # by moving slot-pattern matches to event_slot_fill first
    if "information_extraction" in by_category:
        ie_pool = by_category["information_extraction"]
        non_slot = [q for q in ie_pool if not _SLOT_PATTERNS.search(q.question)]
        slot = [q for q in ie_pool if _SLOT_PATTERNS.search(q.question)]
        by_category["information_extraction"] = non_slot
        by_category["event_slot_fill"] = slot + by_category.get("event_slot_fill", [])

    selected: list[LongMemEvalQuestion] = []
    for cat in DIAGNOSTIC_CATEGORIES:
        pool = by_category.get(cat, [])
        if not pool:
            log.warning("diagnostic_slice.empty_category", category=cat)
            continue
        selected.extend(pool[:n_per_category])

    log.info(
        "diagnostic_slice.selected",
        total=len(selected),
        by_category={cat: len([q for q in selected if _classify_diagnostic_category(q) == cat])
                     for cat in DIAGNOSTIC_CATEGORIES},
    )
    return selected


def run_diagnostic_slice(
    cases: list[LongMemEvalQuestion],
    *,
    embedding_client: EmbeddingClient | None = None,
    reader_fn: ReaderFn | None = None,
    manifest: RunManifest | None = None,
) -> list[DiagnosticCase]:
    """Run each case through the canonical pipeline and classify failures."""
    results: list[DiagnosticCase] = []

    for i, q in enumerate(cases):
        if manifest:
            verify_manifest(manifest)

        t0 = time.monotonic()
        answer, trace = run_substrate_case(
            q, embedding_client=embedding_client, reader_fn=reader_fn,
        )
        elapsed_ms = (time.monotonic() - t0) * 1000

        dc = _classify_failure_from_trace(trace, q.answer)
        dc.category = _classify_diagnostic_category(q)
        dc.latency_ms = elapsed_ms
        results.append(dc)

        status = "PASS" if dc.score > 0 else f"FAIL ({dc.failure_class})"
        log.info(
            "diagnostic_slice.case_done",
            i=i + 1,
            total=len(cases),
            case_id=q.question_id,
            category=dc.category,
            status=status,
            latency_ms=round(elapsed_ms),
        )

    return results


def build_diagnostic_report(results: list[DiagnosticCase]) -> dict[str, Any]:
    """Build a structured report from diagnostic results."""
    per_category_accuracy: dict[str, float] = {}
    per_category_failures: dict[str, dict[str, int]] = {}
    failure_counter: Counter[str] = Counter()
    failure_examples: dict[str, list[str]] = defaultdict(list)
    source_paths: dict[str, Counter[str]] = defaultdict(Counter)

    for cat in DIAGNOSTIC_CATEGORIES:
        cat_results = [r for r in results if r.category == cat]
        if not cat_results:
            continue
        correct = sum(1 for r in cat_results if r.score > 0)
        per_category_accuracy[cat] = correct / len(cat_results)

        failures: dict[str, int] = defaultdict(int)
        for r in cat_results:
            if r.failure_class:
                failures[r.failure_class] += 1
                failure_counter[r.failure_class] += 1
                failure_examples[r.failure_class].append(r.case_id)
            if r.source_path:
                source_paths[cat][r.source_path] += 1
        per_category_failures[cat] = dict(failures)

    top_3 = [
        {"failure_type": ft, "count": count, "example_case_ids": failure_examples[ft][:3]}
        for ft, count in failure_counter.most_common(3)
    ]

    overall_correct = sum(1 for r in results if r.score > 0)
    overall_accuracy = overall_correct / len(results) if results else 0.0

    mean_tokens = sum(r.bundle_tokens for r in results) / len(results) if results else 0
    correct_results = [r for r in results if r.score > 0]
    tokens_per_correct = sum(r.bundle_tokens for r in correct_results) / len(correct_results) if correct_results else 0

    return {
        "n_cases": len(results),
        "overall_accuracy": overall_accuracy,
        "per_category_accuracy": per_category_accuracy,
        "per_category_failure_breakdown": per_category_failures,
        "top_3_bottlenecks": top_3,
        "claim_vs_event_source": {cat: dict(source_paths[cat]) for cat in DIAGNOSTIC_CATEGORIES if cat in source_paths},
        "token_efficiency": {
            "mean_bundle_tokens": mean_tokens,
            "tokens_per_correct_answer": tokens_per_correct,
        },
    }


def compare_token_efficiency(
    claim_only: list[DiagnosticCase],
    claim_event: list[DiagnosticCase],
    baseline: list[DiagnosticCase] | None = None,
) -> dict[str, Any]:
    """Compare token efficiency across retrieval modes."""
    def _stats(results: list[DiagnosticCase]) -> dict[str, float]:
        if not results:
            return {"mean_tokens": 0, "accuracy": 0, "tokens_per_correct": 0}
        correct = [r for r in results if r.score > 0]
        return {
            "mean_tokens": sum(r.bundle_tokens for r in results) / len(results),
            "accuracy": len(correct) / len(results),
            "tokens_per_correct": sum(r.bundle_tokens for r in correct) / len(correct) if correct else 0,
        }

    result = {
        "claim_only": _stats(claim_only),
        "claim_event": _stats(claim_event),
    }
    if baseline:
        result["full_context"] = _stats(baseline)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LongMemEval diagnostic slice")
    parser.add_argument("--data", default="data/longmemeval_oracle.json", help="Path to oracle JSON")
    parser.add_argument("--n-per-category", type=int, default=4, help="Cases per category")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    questions_path = Path(args.data)
    if not questions_path.exists():
        print(f"Data file not found: {questions_path}")
        sys.exit(1)

    manifest = build_run_manifest(model_id="diagnostic")

    cases = select_diagnostic_slice(questions_path, n_per_category=args.n_per_category)
    print(f"Selected {len(cases)} cases across {len(DIAGNOSTIC_CATEGORIES)} categories")

    results = run_diagnostic_slice(cases, manifest=manifest)
    report = build_diagnostic_report(results)

    print("\n" + "=" * 60)
    print("DIAGNOSTIC REPORT")
    print("=" * 60)
    print(f"\nOverall accuracy: {report['overall_accuracy']:.1%} ({len(results)} cases)")
    print("\nPer-category accuracy:")
    for cat, acc in report["per_category_accuracy"].items():
        print(f"  {cat}: {acc:.1%}")
    print("\nPer-category failure breakdown:")
    for cat, failures in report["per_category_failure_breakdown"].items():
        if failures:
            print(f"  {cat}: {failures}")
    print("\nTop 3 bottlenecks:")
    for b in report["top_3_bottlenecks"]:
        print(f"  {b['failure_type']}: {b['count']} ({b['example_case_ids']})")
    print(f"\nToken efficiency:")
    te = report["token_efficiency"]
    print(f"  Mean bundle tokens: {te['mean_bundle_tokens']:.0f}")
    print(f"  Tokens per correct answer: {te['tokens_per_correct_answer']:.0f}")

    if args.output:
        out = Path(args.output)
        out.write_text(json.dumps(report, indent=2, default=str))
        print(f"\nReport saved to {out}")

    # Also save per-case results
    per_case_out = Path(args.output or "eval/longmemeval/diagnostic_results.json").with_suffix(".cases.json")
    per_case_out.write_text(json.dumps([r.to_dict() for r in results], indent=2, default=str))
    print(f"Per-case results saved to {per_case_out}")


if __name__ == "__main__":
    main()
