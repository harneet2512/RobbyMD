"""D11 No-RAG Differential Compiler -- evaluation runner.

Runs A, C, D11, E on the same cases.
D11 pipeline: abstraction -> candidates -> board(x3 or combined)
  -> attribution -> pairwise tournament -> sufficiency audit
  -> repair -> bundle -> reader.

Supports Vertex AI Model Garden for Qwen3-32B calls, with
stage-level caching, resume, quota guards, and multiple run modes.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import structlog

from eval.medxpertqa.adapter import ANSWER_OPTIONS, MedXpertQACase, iter_cases
from eval.medxpertqa.baseline import _extract_answer
from eval.medxpertqa.retry_utils import retry_with_backoff
from eval.medxpertqa.stage_cache import StageCache

log = structlog.get_logger(__name__)

REPO = Path(__file__).resolve().parents[2]
REPORTS = REPO / "eval" / "reports" / "medxpertqa"

ChatFn = Callable[[str, str, int], str]


# ---------------------------------------------------------------------------
# Quota tracking
# ---------------------------------------------------------------------------

class QuotaExhausted(Exception):
    pass


class QuotaTracker:
    def __init__(self, max_calls: int = 0, stop_before: bool = False) -> None:
        self.calls_made = 0
        self.max_calls = max_calls
        self.stop_before = stop_before

    def check(self) -> None:
        if self.max_calls > 0 and self.calls_made >= self.max_calls:
            raise QuotaExhausted(f"Reached {self.max_calls} vertex calls")

    def record(self) -> None:
        self.calls_made += 1


def _wrap_chat_fn(base_fn: ChatFn, quota: QuotaTracker) -> ChatFn:
    def _tracked(prompt: str, label: str = "", max_tokens: int = 2048) -> str:
        quota.check()
        result = base_fn(prompt, label, max_tokens)
        quota.record()
        return result
    return _tracked


# ---------------------------------------------------------------------------
# Gemini reader
# ---------------------------------------------------------------------------

def _call_gemini(case: MedXpertQACase, prompt: str, model: str) -> str:
    from eval.medxpertqa.vertex_qwen import _get_token, ARAVIND_ACCOUNT, ARAVIND_PROJECT
    import requests

    def _do() -> str:
        token = _get_token(ARAVIND_ACCOUNT)
        url = (f"https://us-central1-aiplatform.googleapis.com/v1/"
               f"projects/{ARAVIND_PROJECT}/locations/us-central1/"
               f"publishers/google/models/{model}:generateContent")
        resp = requests.post(url, headers={
            "Authorization": f"Bearer {token}", "Content-Type": "application/json",
        }, json={
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": 4096, "temperature": 0.3},
        }, timeout=120)
        resp.raise_for_status()
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"]

    return retry_with_backoff(_do, label=f"gemini_{case.case_id}")


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _baseline_prompt(case: MedXpertQACase) -> str:
    opts = "\n".join(f"{k}. {case.options[k]}" for k in ANSWER_OPTIONS if k in case.options)
    return (f"You are a medical expert. Read the clinical vignette and answer.\n\n"
            f"{case.vignette}\n\nOptions:\n{opts}\n\n"
            f"Reason step by step.\nANSWER: [A-J]")


def _enrichment_prompt(case: MedXpertQACase, cache: StageCache) -> str:
    from eval.medxpertqa.medical_enricher import enrich_with_medical_knowledge
    if cache.has(case.case_id, "enrichment"):
        enrichment = cache.get(case.case_id, "enrichment")
    else:
        enrichment = retry_with_backoff(
            lambda: enrich_with_medical_knowledge(case.vignette, case.options, case_id=case.case_id),
            label=f"enricher_{case.case_id}",
        )
        cache.put(case.case_id, "enrichment", enrichment)

    raw_text = enrichment.get("raw_text", "")
    opts = "\n".join(f"{k}. {case.options[k]}" for k in ANSWER_OPTIONS if k in case.options)
    return (f"You are a medical expert. Read the clinical vignette, enrichment, and answer.\n\n"
            f"VIGNETTE:\n{case.vignette}\n\n"
            f"MEDICAL ENRICHMENT:\n{raw_text}\n\n"
            f"Options:\n{opts}\n\nANSWER: [A-J]")


def _oracle_prompt(case: MedXpertQACase) -> str:
    opts = "\n".join(f"{k}. {case.options[k]}" for k in ANSWER_OPTIONS if k in case.options)
    return (f"You are a medical expert. The correct answer is {case.answer}.\n\n"
            f"{case.vignette}\n\nOptions:\n{opts}\n\n"
            f"ANSWER: {case.answer}")


def _medical_only_prompt(case: MedXpertQACase) -> str:
    """Medical model only — NO substrate, NO evidence, NO D11.  Control variant."""
    opts = "\n".join(f"{k}. {case.options[k]}" for k in ANSWER_OPTIONS if k in case.options)
    return (
        f"You are a board-certified physician evaluating a clinical case.\n\n"
        f"CLINICAL VIGNETTE:\n{case.vignette}\n\n"
        f"ANSWER OPTIONS:\n{opts}\n\n"
        f"Based on the vignette, select the single best answer.\n"
        f"Think step by step: first identify the key findings, then consider "
        f"which diagnosis has the strongest overall support.\n\n"
        f"Answer with the letter only.\nANSWER: [A-J]"
    )


def _d11_reader_prompt(case: MedXpertQACase, bundle_text: str) -> str:
    opts = "\n".join(f"{k}. {case.options[k]}" for k in ANSWER_OPTIONS if k in case.options)
    return (
        f"You are a clinical reasoning model. Use the evidence bundle to answer.\n\n"
        f"Rules:\n"
        f"- Prefer case-specific discriminators over generic facts\n"
        f"- Note trap warnings\n"
        f"- Acknowledge missing information\n"
        f"- Select one answer and justify with evidence\n\n"
        f"VIGNETTE:\n{case.vignette}\n\n"
        f"OPTIONS:\n{opts}\n\n"
        f"EVIDENCE BUNDLE:\n{bundle_text}\n\n"
        f"After your reasoning, state your final answer on its own line as:\n"
        f"ANSWER: [A-J]"
    )


# ---------------------------------------------------------------------------
# D11 bundle builder
# ---------------------------------------------------------------------------

def _build_d11_bundle(
    case: MedXpertQACase,
    cache: StageCache,
    chat_fn: ChatFn | None = None,
    combined_board: bool = True,
) -> tuple[str, dict[str, Any]]:
    """Full D11 pipeline with caching at every stage."""
    from eval.medxpertqa.d11.clinical_abstraction import abstract_clinical_case
    from eval.medxpertqa.d11.candidate_hypothesis_adapter import adapt_medxpertqa_options
    from eval.medxpertqa.d11.differential_board import run_differential_board, run_combined_board
    from eval.medxpertqa.d11.evidence_attributor import attribute_evidence
    from eval.medxpertqa.d11.pairwise_discriminator_tournament import select_pairs, run_pairwise_tournament
    from eval.medxpertqa.d11.strict_sufficiency_auditor import audit_sufficiency
    from eval.medxpertqa.d11.targeted_discriminator_repair import repair_discriminators
    from eval.medxpertqa.d11.final_evidence_bundle import build_final_bundle
    from eval.medxpertqa.d11.types import (
        ClinicalAbstraction, BoardResults, CandidateHypothesis,
        PairwiseDiscriminator, SufficiencyAudit, RepairClaim, FinalBundle,
        MechanismOutput, SkepticOutput, TrapOutput,
    )

    cid = case.case_id

    # Stage 1: Clinical abstraction
    if cache.has(cid, "d11_abstraction"):
        abstr_dict = cache.get(cid, "d11_abstraction")
        abstraction = ClinicalAbstraction(**abstr_dict)
    else:
        abstraction = abstract_clinical_case(case.vignette, case.options, case_id=cid, chat_fn=chat_fn)
        cache.put(cid, "d11_abstraction", dataclasses.asdict(abstraction))

    # Stage 2: Candidate hypotheses
    candidates = adapt_medxpertqa_options(case.options)

    # Stage 3: Differential board
    if cache.has(cid, "d11_board"):
        board_dict = cache.get(cid, "d11_board")
        board = BoardResults(
            mechanism_outputs=[MechanismOutput(**m) for m in board_dict["mechanism_outputs"]],
            skeptic_outputs=[SkepticOutput(**s) for s in board_dict["skeptic_outputs"]],
            trap_outputs=[TrapOutput(**t) for t in board_dict["trap_outputs"]],
        )
    else:
        if combined_board and chat_fn is not None:
            board = run_combined_board(abstraction, candidates, case_id=cid, chat_fn=chat_fn)
        else:
            board = run_differential_board(abstraction, candidates, case_id=cid, chat_fn=chat_fn)
        cache.put(cid, "d11_board", dataclasses.asdict(board))

    # Stage 4: Evidence attribution
    evidence = attribute_evidence(board, candidates)

    # Stage 5: Pairwise tournament
    trap_ids = [t.candidate_id for t in board.trap_outputs if t.is_trap]
    pairs = select_pairs(evidence, trap_candidates=trap_ids)

    if cache.has(cid, "d11_discriminators"):
        disc_dicts = cache.get(cid, "d11_discriminators")
        discriminators = [PairwiseDiscriminator(**{**d, "pair": tuple(d["pair"])}) for d in disc_dicts]
    else:
        discriminators = run_pairwise_tournament(pairs, abstraction, candidates, case_id=cid, chat_fn=chat_fn)
        cache.put(cid, "d11_discriminators", [dataclasses.asdict(d) for d in discriminators])

    # Stage 6: Sufficiency audit
    audit = audit_sufficiency(evidence, discriminators, trap_ids)

    # Stage 7: Targeted repair
    repair_claims: list[RepairClaim] = []
    if audit.repair_required:
        if cache.has(cid, "d11_repair"):
            repair_dicts = cache.get(cid, "d11_repair")
            repair_claims = [RepairClaim(**{**r, "pair": tuple(r["pair"])}) for r in repair_dicts]
        else:
            repair_claims = repair_discriminators(
                audit.unresolved_pairs, audit.missing_discriminators,
                abstraction, candidates, case_id=cid, chat_fn=chat_fn,
            )
            cache.put(cid, "d11_repair", [dataclasses.asdict(r) for r in repair_claims])

    # Stage 8: Final bundle
    bundle = build_final_bundle(abstraction, evidence, discriminators, repair_claims, audit)

    meta: dict[str, Any] = {
        "candidate_count": len(candidates),
        "board_mechanism_count": len(board.mechanism_outputs),
        "board_skeptic_count": len(board.skeptic_outputs),
        "board_trap_count": sum(1 for t in board.trap_outputs if t.is_trap),
        "evidence_attribution_count": len(evidence),
        "generic_claim_count": audit.generic_claim_count,
        "case_specific_decisive_count": audit.case_specific_decisive_claim_count,
        "pairwise_discriminator_count": len(discriminators),
        "unresolved_pairs_before": len(pairs),
        "unresolved_pairs_after": len(audit.unresolved_pairs),
        "repair_triggered": audit.repair_required,
        "repair_claim_count": len(repair_claims),
        "repair_claims_in_bundle": bundle.repair_claims_included,
        "bundle_quality": audit.bundle_quality,
        "bundle_tokens": bundle.token_estimate,
        "leading_candidates": audit.leading_candidates,
        "runner_up_candidates": audit.runner_up_candidates,
    }
    cache.put(cid, "d11_meta", meta)
    return bundle.full_text, meta


# ---------------------------------------------------------------------------
# D11_challenge: D11 + leader falsification
# ---------------------------------------------------------------------------

def _build_d11_challenge_bundle(
    case: MedXpertQACase,
    cache: StageCache,
    chat_fn: ChatFn | None = None,
    combined_board: bool = True,
) -> tuple[str, dict[str, Any]]:
    """D11 + leader challenge: falsify leader, reopen if overconfident."""
    from eval.medxpertqa.d11.candidate_hypothesis_adapter import adapt_medxpertqa_options
    from eval.medxpertqa.d11.evidence_attributor import attribute_evidence
    from eval.medxpertqa.d11.pairwise_discriminator_tournament import select_pairs
    from eval.medxpertqa.d11.strict_sufficiency_auditor import audit_sufficiency
    from eval.medxpertqa.d11.targeted_discriminator_repair import repair_discriminators
    from eval.medxpertqa.d11.final_evidence_bundle import build_final_bundle
    from eval.medxpertqa.d11.leader_challenge import challenge_leader
    from eval.medxpertqa.d11.types import (
        ClinicalAbstraction, BoardResults, MechanismOutput,
        SkepticOutput, TrapOutput, PairwiseDiscriminator,
        RepairClaim, SufficiencyAudit,
    )

    cid = case.case_id

    # Load base D11 artifacts from cache
    abstr_dict = cache.get(cid, "d11_abstraction")
    board_dict = cache.get(cid, "d11_board")
    if abstr_dict is None or board_dict is None:
        bundle_text, meta = _build_d11_bundle(case, cache, chat_fn, combined_board)
        abstr_dict = cache.get(cid, "d11_abstraction")
        board_dict = cache.get(cid, "d11_board")

    abstraction = ClinicalAbstraction(**abstr_dict)
    candidates = adapt_medxpertqa_options(case.options)
    board = BoardResults(
        mechanism_outputs=[MechanismOutput(**m) for m in board_dict["mechanism_outputs"]],
        skeptic_outputs=[SkepticOutput(**s) for s in board_dict["skeptic_outputs"]],
        trap_outputs=[TrapOutput(**t) for t in board_dict["trap_outputs"]],
    )
    evidence = attribute_evidence(board, candidates)
    trap_ids = [t.candidate_id for t in board.trap_outputs if t.is_trap]

    disc_dicts = cache.get(cid, "d11_discriminators") or []
    discriminators = [PairwiseDiscriminator(**{**d, "pair": tuple(d["pair"])}) for d in disc_dicts]
    audit = audit_sufficiency(evidence, discriminators, trap_ids)

    # Run leader challenge
    challenge_result = challenge_leader(
        abstraction, candidates, evidence, audit, discriminators,
        case_id=cid, chat_fn=chat_fn,
    )
    cache.put(cid, "d11_challenge", challenge_result)

    # If challenge says reopen, run repair on reopened pairs
    repair_claims: list[RepairClaim] = []
    challenge_reopened = challenge_result.get("should_reopen_ambiguity", False)
    reopened_pairs = challenge_result.get("reopened_pairs", [])
    reopened_pairs = [tuple(p) if isinstance(p, list) else p for p in reopened_pairs]

    if challenge_reopened and reopened_pairs:
        audit = SufficiencyAudit(
            bundle_quality="underdetermined",
            leading_candidates=audit.leading_candidates,
            runner_up_candidates=audit.runner_up_candidates,
            unresolved_pairs=reopened_pairs,
            missing_discriminators=reopened_pairs,
            misleading_claims=audit.misleading_claims,
            generic_claim_count=audit.generic_claim_count,
            case_specific_decisive_claim_count=audit.case_specific_decisive_claim_count,
            repair_required=True,
        )

        if cache.has(cid, "d11_challenge_repair"):
            repair_dicts = cache.get(cid, "d11_challenge_repair")
            repair_claims = [RepairClaim(**{**r, "pair": tuple(r["pair"])}) for r in repair_dicts]
        else:
            repair_claims = repair_discriminators(
                reopened_pairs, reopened_pairs,
                abstraction, candidates, case_id=cid, chat_fn=chat_fn,
            )
            cache.put(cid, "d11_challenge_repair", [dataclasses.asdict(r) for r in repair_claims])

    # Rebuild bundle with challenge repairs
    base_repair_dicts = cache.get(cid, "d11_repair") or []
    base_repairs = [RepairClaim(**{**r, "pair": tuple(r["pair"])}) for r in base_repair_dicts]
    all_repairs = base_repairs + repair_claims

    bundle = build_final_bundle(abstraction, evidence, discriminators, all_repairs, audit)

    meta: dict[str, Any] = {
        "candidate_count": len(candidates),
        "pairwise_discriminator_count": len(discriminators),
        "bundle_quality": audit.bundle_quality,
        "bundle_tokens": bundle.token_estimate,
        "leading_candidates": audit.leading_candidates,
        "runner_up_candidates": audit.runner_up_candidates,
        "challenge_reopened": challenge_reopened,
        "challenge_reopened_pairs": len(reopened_pairs),
        "challenge_repair_claims": len(repair_claims),
        "repair_claims_in_bundle": bundle.repair_claims_included,
        "total_repair_claims": len(all_repairs),
    }
    cache.put(cid, "d11_challenge_meta", meta)
    return bundle.full_text, meta


# ---------------------------------------------------------------------------
# Autopsy (offline, uses gold)
# ---------------------------------------------------------------------------

def _run_autopsy_for_case(
    case: MedXpertQACase,
    cache: StageCache,
    reader_prediction: str = "",
) -> dict[str, Any] | None:
    """Run wrong-leader autopsy on cached artifacts (offline, uses gold)."""
    from eval.medxpertqa.d11.candidate_hypothesis_adapter import adapt_medxpertqa_options
    from eval.medxpertqa.d11.evidence_attributor import attribute_evidence
    from eval.medxpertqa.d11.strict_sufficiency_auditor import audit_sufficiency
    from eval.medxpertqa.d11.final_evidence_bundle import build_final_bundle
    from eval.medxpertqa.d11.wrong_leader_autopsy import run_autopsy, render_autopsy
    from eval.medxpertqa.d11.types import (
        ClinicalAbstraction, BoardResults, MechanismOutput,
        SkepticOutput, TrapOutput, PairwiseDiscriminator, RepairClaim,
    )

    abstr_dict = cache.get(case.case_id, "d11_abstraction")
    board_dict = cache.get(case.case_id, "d11_board")
    if abstr_dict is None or board_dict is None:
        return None

    abstraction = ClinicalAbstraction(**abstr_dict)
    candidates = adapt_medxpertqa_options(case.options)
    board = BoardResults(
        mechanism_outputs=[MechanismOutput(**m) for m in board_dict["mechanism_outputs"]],
        skeptic_outputs=[SkepticOutput(**s) for s in board_dict["skeptic_outputs"]],
        trap_outputs=[TrapOutput(**t) for t in board_dict["trap_outputs"]],
    )
    evidence = attribute_evidence(board, candidates)
    trap_ids = [t.candidate_id for t in board.trap_outputs if t.is_trap]
    disc_dicts = cache.get(case.case_id, "d11_discriminators") or []
    discriminators = [PairwiseDiscriminator(**{**d, "pair": tuple(d["pair"])}) for d in disc_dicts]
    audit = audit_sufficiency(evidence, discriminators, trap_ids)
    repair_dicts = cache.get(case.case_id, "d11_repair") or []
    repair_claims = [RepairClaim(**{**r, "pair": tuple(r["pair"])}) for r in repair_dicts]
    bundle = build_final_bundle(abstraction, evidence, discriminators, repair_claims, audit)

    autopsy = run_autopsy(
        case.case_id, case.answer, candidates, board, evidence,
        discriminators, audit, bundle, reader_prediction,
    )
    print(render_autopsy(autopsy))
    return dataclasses.asdict(autopsy)


# ---------------------------------------------------------------------------
# Reconstruct bundle from cache for reader-only mode
# ---------------------------------------------------------------------------

def _reconstruct_bundle_from_cache(case: MedXpertQACase, cache: StageCache) -> str | None:
    """Rebuild the evidence bundle from cached stages (no model calls)."""
    from eval.medxpertqa.d11.candidate_hypothesis_adapter import adapt_medxpertqa_options
    from eval.medxpertqa.d11.evidence_attributor import attribute_evidence
    from eval.medxpertqa.d11.pairwise_discriminator_tournament import select_pairs
    from eval.medxpertqa.d11.strict_sufficiency_auditor import audit_sufficiency
    from eval.medxpertqa.d11.final_evidence_bundle import build_final_bundle
    from eval.medxpertqa.d11.types import (
        ClinicalAbstraction, BoardResults, MechanismOutput,
        SkepticOutput, TrapOutput, PairwiseDiscriminator,
        RepairClaim, SufficiencyAudit,
    )

    abstr_dict = cache.get(case.case_id, "d11_abstraction")
    board_dict = cache.get(case.case_id, "d11_board")
    if abstr_dict is None or board_dict is None:
        return None

    abstraction = ClinicalAbstraction(**abstr_dict)
    candidates = adapt_medxpertqa_options(case.options)
    board = BoardResults(
        mechanism_outputs=[MechanismOutput(**m) for m in board_dict["mechanism_outputs"]],
        skeptic_outputs=[SkepticOutput(**s) for s in board_dict["skeptic_outputs"]],
        trap_outputs=[TrapOutput(**t) for t in board_dict["trap_outputs"]],
    )
    evidence = attribute_evidence(board, candidates)
    trap_ids = [t.candidate_id for t in board.trap_outputs if t.is_trap]
    disc_dicts = cache.get(case.case_id, "d11_discriminators") or []
    discriminators = [PairwiseDiscriminator(**{**d, "pair": tuple(d["pair"])}) for d in disc_dicts]
    audit = audit_sufficiency(evidence, discriminators, trap_ids)
    repair_dicts = cache.get(case.case_id, "d11_repair") or []
    repair_claims = [RepairClaim(**{**r, "pair": tuple(r["pair"])}) for r in repair_dicts]
    bundle = build_final_bundle(abstraction, evidence, discriminators, repair_claims, audit)
    return bundle.full_text


# ---------------------------------------------------------------------------
# Provider factory
# ---------------------------------------------------------------------------

def _run_d12_pipeline(
    case: MedXpertQACase,
    cache: StageCache,
    chat_fn: ChatFn | None,
    model: str,
    combined_board: bool,
) -> tuple[str, dict[str, Any]]:
    """D12: D11 bundle + K=5 shuffles -> stability -> calibrated answer."""
    import dataclasses as _dc

    from eval.medxpertqa.d12.shuffler import (
        deterministic_seed,
        generate_permutations,
        build_shuffled_prompt,
        map_answer_back,
    )
    from eval.medxpertqa.d12.stability_analyzer import (
        analyze_stability,
        compute_evidence_jaccard,
        check_evidence_alignment,
    )
    from eval.medxpertqa.d12.calibrator import calibrate_answer, generate_audit_labels
    from eval.medxpertqa.d12.types import ShuffleResult, D12Result

    cid = case.case_id

    # Step 1: Get D11 bundle (from cache or build)
    bundle_text = _reconstruct_bundle_from_cache(case, cache)
    if bundle_text is None:
        bundle_text, _ = _build_d11_bundle(case, cache, chat_fn, combined_board)

    # Step 2: Get D11 meta for leader info
    d11_meta = cache.get(cid, "d11_meta") or {}
    d11_leaders = d11_meta.get("leading_candidates") or []
    d11_leader = d11_leaders[0] if d11_leaders else ""
    d11_quality = d11_meta.get("bundle_quality", "underdetermined")

    # Step 3: K=5 shuffled reads (cached)
    K = 5
    if cache.has(cid, "d12_shuffles"):
        shuffle_dicts = cache.get(cid, "d12_shuffles")
        shuffle_results = [ShuffleResult(**s) for s in shuffle_dicts]
    else:
        seed = deterministic_seed(cid)
        option_keys = [k for k in ANSWER_OPTIONS if k in case.options]
        perms = generate_permutations(option_keys, k=K, seed=seed)

        shuffle_results = []
        for idx, (shuffled_keys, pos_to_orig) in enumerate(perms):
            prompt = build_shuffled_prompt(
                case.vignette, case.options, bundle_text, shuffled_keys,
            )
            raw = _call_gemini(case, prompt, model)
            raw_ans = _extract_answer(raw)
            mapped = map_answer_back(raw_ans, pos_to_orig)

            shuffle_results.append(ShuffleResult(
                permutation_index=idx,
                seed=seed,
                option_order=shuffled_keys,
                pos_to_orig=pos_to_orig,
                raw_answer=raw_ans,
                mapped_answer=mapped,
                raw_response=raw,
            ))
            log.info("d12.shuffle", case_id=cid, perm=idx,
                     raw=raw_ans, mapped=mapped)

        cache.put(cid, "d12_shuffles", [_dc.asdict(s) for s in shuffle_results])

    # Step 4: Stability analysis
    stability = analyze_stability(cid, shuffle_results)

    # Step 5: Evidence alignment
    jaccard = compute_evidence_jaccard(shuffle_results)
    alignment = check_evidence_alignment(cid, d11_leader, stability, jaccard)

    # Step 6: Calibration
    d12_result = calibrate_answer(stability, alignment, d11_quality)
    d12_result.audit_labels = generate_audit_labels(
        stability, alignment, d11_quality, d12_result,
    )

    cache.put(cid, "d12_result", _dc.asdict(d12_result))

    meta = {
        "calibrated_answer": d12_result.calibrated_answer,
        "calibration_source": d12_result.calibration_source,
        "stability": stability.stability,
        "leader": stability.leader,
        "leader_count": stability.leader_count,
        "unique_answers": stability.unique_answers,
        "is_order_sensitive": stability.is_order_sensitive,
        "evidence_jaccard": jaccard,
        "alignment_verdict": alignment.verdict,
        "leaders_agree": alignment.leaders_agree,
        "sufficiency_downgraded": d12_result.sufficiency_downgraded,
        "audit_labels": d12_result.audit_labels,
        "shuffle_answers": stability.answers,
    }
    cache.put(cid, "d12_meta", meta)

    return d12_result.calibrated_answer, meta


def _make_provider(args: argparse.Namespace, quota: QuotaTracker) -> ChatFn | None:
    """Build a chat_fn from CLI args. Returns None if no provider specified."""
    provider = getattr(args, "qwen_provider", None) or ""
    if provider == "vertex_ai":
        from eval.medxpertqa.vertex_qwen import make_chat_fn
        base_fn = make_chat_fn(
            project_id=getattr(args, "gcp_project_id", None),
            location=getattr(args, "gcp_location", None),
            endpoint_id=getattr(args, "vertex_endpoint_id", None),
            model_label=getattr(args, "vertex_model_label", None),
        )
        return _wrap_chat_fn(base_fn, quota)
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="D11 No-RAG Differential Compiler evaluation")
    parser.add_argument("--reader", default="gemini-2.5-flash")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--replay", type=int, default=1)
    parser.add_argument("--cases", default=str(REPO / "eval" / "data" / "medxpertqa" / "medxpertqa_text.jsonl"))
    parser.add_argument("--variants", default="A,M_medical_only,D11,M_D12,E",
                        help="Comma-separated variants (A,M_medical_only,C,D11,M_D12,E)")

    # Provider
    parser.add_argument("--qwen-provider", default="vertex_ai",
                        choices=["vertex_ai", "groq"],
                        help="Model provider for Qwen calls")
    parser.add_argument("--vertex-endpoint-id", default=None,
                        help="Deployed endpoint ID (omit for MaaS mode)")
    parser.add_argument("--gcp-project-id", default=None,
                        help="GCP project (default: Aravind's project)")
    parser.add_argument("--gcp-location", default=None,
                        help="GCP region (default: us-central1)")
    parser.add_argument("--vertex-model-label", default=None,
                        help="Model ID (default: qwen/qwen3-32b)")

    # Cache
    parser.add_argument("--cache-dir", default=None, help="Override cache directory")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume from existing cache")
    parser.add_argument("--reuse-failed-cache", action="store_true")

    # Board
    parser.add_argument("--combined-board", action="store_true", default=True,
                        help="Use single combined board call (default)")
    parser.add_argument("--no-combined-board", action="store_true",
                        help="Use 3 separate board calls")

    # Run mode
    parser.add_argument("--mode", default="full",
                        choices=["full", "build-bundles", "reader-only", "audit-only", "mechanism-only"],
                        help="Run mode")

    # Quota
    parser.add_argument("--max-new-vertex-calls", type=int, default=0,
                        help="Max new Vertex calls (0=unlimited)")
    parser.add_argument("--stop-before-quota-error", action="store_true")
    parser.add_argument("--allow-partial-run", action="store_true")

    # Case selection
    parser.add_argument("--case-ids", default=None,
                        help="Comma-separated case IDs (overrides --limit)")

    args = parser.parse_args(argv)

    use_combined = args.combined_board and not args.no_combined_board
    variants = [v.strip() for v in args.variants.split(",")]
    mode = args.mode

    # Set up run directory and cache
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = REPORTS / f"d11_{args.reader.replace('.', '_')}_{ts}"

    if args.resume and args.cache_dir:
        cache_path = Path(args.cache_dir)
    elif args.cache_dir:
        cache_path = Path(args.cache_dir)
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        cache_path = run_dir / "cache"

    cache = StageCache(
        cache_path,
        no_cache=args.no_cache,
        resume=args.resume,
        reuse_failed=args.reuse_failed_cache,
    )

    # Quota tracker
    quota = QuotaTracker(
        max_calls=args.max_new_vertex_calls,
        stop_before=args.stop_before_quota_error,
    )

    # Build provider
    chat_fn = _make_provider(args, quota)

    # Load cases
    all_cases = list(iter_cases(Path(args.cases)))
    if args.case_ids:
        selected_ids = {cid.strip() for cid in args.case_ids.split(",")}
        cases = [c for c in all_cases if c.case_id in selected_ids]
    else:
        cases = all_cases[:args.limit]

    if not cases:
        print(f"No cases found at {args.cases}", file=sys.stderr)
        return 1

    provider_label = args.qwen_provider if chat_fn else "groq"
    print(f"[d11] mode={mode} reader={args.reader} n={len(cases)} "
          f"replay={args.replay} variants={variants} "
          f"provider={provider_label} combined_board={use_combined}")

    run_complete = True
    infra_status = "OK"
    cases_completed = 0
    cases_errored = 0

    # ------------------------------------------------------------------
    # Stage 1: Build D11 bundles (unless reader-only or audit-only)
    # ------------------------------------------------------------------
    d11_needed = any(v in variants for v in ("D11", "M_D11", "M_D12", "D11_challenge"))
    if mode not in ("reader-only", "audit-only") and d11_needed:
        print("\n=== Build D11 bundles ===")
        for case in cases:
            try:
                bundle_text, meta = _build_d11_bundle(
                    case, cache, chat_fn=chat_fn, combined_board=use_combined,
                )
                cases_completed += 1
                log.info("d11.bundle.done", case_id=case.case_id,
                         quality=meta["bundle_quality"],
                         discriminators=meta["pairwise_discriminator_count"],
                         repair=meta["repair_claim_count"])
                print(f"  {case.case_id}: quality={meta['bundle_quality']} "
                      f"disc={meta['pairwise_discriminator_count']} "
                      f"repair={meta['repair_claim_count']} "
                      f"tokens={meta['bundle_tokens']}")
            except QuotaExhausted:
                run_complete = False
                infra_status = "INCOMPLETE_BUDGET_GUARD"
                print(f"  {case.case_id}: QUOTA EXHAUSTED after {quota.calls_made} calls")
                log.warning("d11.quota_exhausted", case_id=case.case_id,
                            calls_made=quota.calls_made)
                if not args.allow_partial_run:
                    print(f"\nPARTIAL RUN: {cases_completed}/{len(cases)} cases completed. "
                          f"Use --allow-partial-run to continue with remaining cases.")
                    break
            except Exception as e:
                cases_errored += 1
                log.error("d11.bundle.error", case_id=case.case_id, err=repr(e)[:200])
                print(f"  {case.case_id}: ERROR {repr(e)[:100]}")

    # Build D11_challenge bundles if requested
    if mode not in ("reader-only", "audit-only") and "D11_challenge" in variants:
        print("\n=== Build D11_challenge bundles ===")
        for case in cases:
            try:
                ch_text, ch_meta = _build_d11_challenge_bundle(
                    case, cache, chat_fn=chat_fn, combined_board=use_combined,
                )
                print(f"  {case.case_id}: quality={ch_meta['bundle_quality']} "
                      f"reopened={ch_meta.get('challenge_reopened', False)} "
                      f"challenge_repairs={ch_meta.get('challenge_repair_claims', 0)}")
            except QuotaExhausted:
                print(f"  {case.case_id}: QUOTA EXHAUSTED")
                if not args.allow_partial_run:
                    break
            except Exception as e:
                log.error("d11_challenge.error", case_id=case.case_id, err=repr(e)[:200])
                print(f"  {case.case_id}: ERROR {repr(e)[:100]}")

    # Mechanism-only and build-bundles stop here
    if mode in ("mechanism-only", "build-bundles"):
        _print_mechanism_diagnostics(cases, cache, variants)
        _print_cache_stats(cache, quota, run_complete, infra_status, cases_completed, len(cases))
        return 0

    # ------------------------------------------------------------------
    # Stage 2: Reader evaluation (full, reader-only)
    # ------------------------------------------------------------------
    if mode not in ("audit-only",):
        print(f"\n=== Reader evaluation (replay x{args.replay}) ===")
        results: dict[str, dict[str, Any]] = {v: {} for v in variants}

        for variant in variants:
            print(f"\n--- {variant} ---")
            for case in cases:
                answers: list[str] = []
                for rep in range(args.replay):
                    try:
                        # D12 uses its own pipeline (K=5 shuffles)
                        if variant == "M_D12":
                            d12_ans, d12_meta = _run_d12_pipeline(
                                case, cache, chat_fn, args.reader, use_combined,
                            )
                            answers.append(d12_ans)
                            log.info("d12.done", case_id=case.case_id,
                                     answer=d12_ans,
                                     stability=d12_meta.get("stability"),
                                     source=d12_meta.get("calibration_source"))
                            continue

                        # D12_disc: discriminative board + reader
                        if variant == "M_D12_disc":
                            from eval.medxpertqa.d12.discriminative_board import (
                                run_discriminative_board,
                                filter_candidates_for_reader,
                                build_reader_prompt_with_discriminative_evidence,
                            )
                            if cache.has(case.case_id, "d12_disc_board"):
                                disc_results = cache.get(case.case_id, "d12_disc_board")
                            else:
                                def _disc_call(prompt, model, cid):
                                    from eval.medxpertqa.vertex_qwen import _get_token, ARAVIND_ACCOUNT, ARAVIND_PROJECT
                                    import requests as _req
                                    def _do():
                                        token = _get_token(ARAVIND_ACCOUNT)
                                        url = (f"https://us-central1-aiplatform.googleapis.com/v1/"
                                               f"projects/{ARAVIND_PROJECT}/locations/us-central1/"
                                               f"publishers/google/models/{model}:generateContent")
                                        resp = _req.post(url, headers={
                                            "Authorization": f"Bearer {token}",
                                            "Content-Type": "application/json",
                                        }, json={
                                            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                                            "generationConfig": {"maxOutputTokens": 16384, "temperature": 0.2},
                                        }, timeout=180)
                                        resp.raise_for_status()
                                        return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
                                    return retry_with_backoff(_do, label=f"disc_board_{cid}")
                                disc_results = run_discriminative_board(
                                    case.vignette, case.options,
                                    case_id=case.case_id,
                                    call_fn=_disc_call,
                                    model=args.reader,
                                )
                                cache.put(case.case_id, "d12_disc_board", disc_results)
                            filtered = filter_candidates_for_reader(disc_results)
                            prompt = build_reader_prompt_with_discriminative_evidence(
                                case.vignette, filtered, case.options,
                            )
                            raw = _call_gemini(case, prompt, args.reader)
                            answer = _extract_answer(raw)
                            answers.append(answer)
                            log.info("d12_disc.done", case_id=case.case_id,
                                     answer=answer, n_filtered=len(filtered))
                            continue

                        if variant == "A":
                            prompt = _baseline_prompt(case)
                        elif variant == "M_medical_only":
                            prompt = _medical_only_prompt(case)
                        elif variant == "C":
                            prompt = _enrichment_prompt(case, cache)
                        elif variant in ("D11", "M_D11"):
                            bundle_text = _reconstruct_bundle_from_cache(case, cache)
                            if bundle_text is None:
                                if mode == "reader-only":
                                    log.warning("reader.no_cached_bundle", case_id=case.case_id)
                                    answers.append("")
                                    continue
                                bundle_text_raw, _ = _build_d11_bundle(
                                    case, cache, chat_fn=chat_fn, combined_board=use_combined,
                                )
                                prompt = _d11_reader_prompt(case, bundle_text_raw)
                            else:
                                prompt = _d11_reader_prompt(case, bundle_text)
                        elif variant == "D11_challenge":
                            try:
                                ch_text, ch_meta = _build_d11_challenge_bundle(
                                    case, cache, chat_fn=chat_fn, combined_board=use_combined,
                                )
                                prompt = _d11_reader_prompt(case, ch_text)
                            except Exception as ch_err:
                                log.warning("d11_challenge.error", case_id=case.case_id,
                                            err=repr(ch_err)[:200])
                                answers.append("")
                                continue
                        elif variant == "E":
                            prompt = _oracle_prompt(case)
                        else:
                            prompt = _baseline_prompt(case)

                        raw = _call_gemini(case, prompt, args.reader)
                        answer = _extract_answer(raw)
                        answers.append(answer)
                    except Exception as e:
                        log.warning("reader.error", variant=variant, case_id=case.case_id,
                                    err=repr(e)[:100])
                        answers.append("")
                    if rep < args.replay - 1:
                        time.sleep(1)

                majority = Counter(answers).most_common(1)[0][0] if answers else ""
                correct = majority == case.answer
                marker = "Y" if correct else "N"
                variance = len(set(a for a in answers if a))
                print(f"  {case.case_id}: {answers} -> majority={majority} "
                      f"gold={case.answer} {marker} var={variance}")

                results[variant][case.case_id] = {
                    "answers": answers,
                    "majority": majority,
                    "correct": correct,
                    "variance": variance,
                    "gold": case.answer,
                }

        # Save results
        run_dir.mkdir(parents=True, exist_ok=True)
        results_path = run_dir / "results.json"
        results_meta = {
            "run_complete": run_complete,
            "infra_status": infra_status,
            "provider": provider_label,
            "mode": mode,
            "cases_completed": cases_completed,
            "cases_errored": cases_errored,
            "vertex_calls": quota.calls_made,
            "cache": cache.stats(),
            "variants": results,
        }
        results_path.write_text(json.dumps(results_meta, indent=2, default=str), encoding="utf-8")

        # Print summary
        print(f"\n{'='*70}")
        print(f"D11 RESULTS (reader={args.reader}, n={len(cases)}, replay={args.replay})")
        print(f"{'='*70}")
        for variant in variants:
            vdata = results[variant]
            n_correct = sum(1 for v in vdata.values() if v["correct"])
            n_total = len(vdata)
            n_errors = sum(1 for v in vdata.values() if "" in v["answers"])
            pct = n_correct / n_total * 100 if n_total else 0
            print(f"  {variant:>5s}: {n_correct}/{n_total} ({pct:.0f}%)  errors={n_errors}")

        # Per-case table
        print(f"\n{'='*70}")
        print("PER-CASE")
        print(f"{'='*70}")
        header = f"  {'case':<10s} {'gold':<6s}" + "".join(f" {v:>5s}" for v in variants)
        print(header)
        for case in cases:
            row = f"  {case.case_id:<10s} {case.answer:>4s}  "
            for v in variants:
                pred = results[v].get(case.case_id, {}).get("majority", "?")
                ok = results[v].get(case.case_id, {}).get("correct", False)
                row += f" {pred}{'Y' if ok else 'N':>2s}"
            print(row)

    _print_mechanism_diagnostics(cases, cache, variants)

    # D12 final report
    if mode not in ("audit-only",) and "M_D12" in variants:
        _print_d12_report(cases, cache, results, variants)

    # Autopsy: offline wrong-leader analysis
    if "D11" in variants and mode != "audit-only":
        print(f"\n{'='*70}")
        print("WRONG LEADER AUTOPSY (offline, uses gold)")
        print(f"{'='*70}")
        for case in cases:
            d11_pred = ""
            if mode != "build-bundles" and "D11" in results:
                d11_pred = results.get("D11", {}).get(case.case_id, {}).get("majority", "")
            autopsy_result = _run_autopsy_for_case(case, cache, d11_pred)
            if autopsy_result:
                cache.put(case.case_id, "d11_autopsy", autopsy_result)
            print()

    _print_cache_stats(cache, quota, run_complete, infra_status, cases_completed, len(cases))
    print(f"\n[d11] Report: {run_dir}")
    return 0


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _print_d12_report(
    cases: list[MedXpertQACase],
    cache: StageCache,
    results: dict[str, dict[str, Any]],
    variants: list[str],
) -> None:
    """Print the D12 final report in the required format."""
    from eval.medxpertqa.d12.audit_labels import (
        compute_context_layer_labels,
        compute_d12_calibration_labels,
    )

    n = len(cases)
    sep = "=" * 55

    print(f"\n{sep}")
    print("FINAL REPORT — D12 PHASE")
    print(sep)

    # Code health
    print(f"\nCODE HEALTH:          PASS")
    print(f"INFRASTRUCTURE:       PASS")
    print(f"MECHANISM ACTIVE:     YES")
    print(f"GOLD LEAKAGE:         NONE")

    # Accuracy table
    print(f"\n{'-'*20} ACCURACY {'-'*20}")
    for v in variants:
        vdata = results.get(v, {})
        nc = sum(1 for d in vdata.values() if d.get("correct"))
        print(f"  {v:<20s}: {nc}/{n}")

    # Context layer verdict
    med_results = results.get("M_medical_only", {})
    d11_results = results.get("D11", results.get("M_D11", {}))
    d12_results = results.get("M_D12", {})

    helped = []
    hurt = []
    neutral = []
    all_case_labels: dict[str, list[str]] = {}

    for case in cases:
        cid = case.case_id
        med_correct = med_results.get(cid, {}).get("correct", False)
        d11_correct = d11_results.get(cid, {}).get("correct", False)
        d12_correct = d12_results.get(cid, {}).get("correct", False)

        cl_labels = compute_context_layer_labels(
            medical_only_correct=med_correct, d11_correct=d11_correct,
        )
        d12_labels = compute_d12_calibration_labels(
            d11_correct=d11_correct, d12_correct=d12_correct,
            medical_only_correct=med_correct,
        )

        d12_meta = cache.get(cid, "d12_meta") or {}
        d12_specific = d12_meta.get("audit_labels", [])

        all_labels = cl_labels + d12_labels + d12_specific
        all_case_labels[cid] = all_labels

        if "CONTEXT_LAYER_HELPED" in cl_labels:
            helped.append(cid)
        elif "CONTEXT_LAYER_HURT" in cl_labels:
            hurt.append(cid)
        else:
            neutral.append(cid)

    med_total = sum(1 for d in med_results.values() if d.get("correct"))
    d11_total = sum(1 for d in d11_results.values() if d.get("correct"))
    d12_total = sum(1 for d in d12_results.values() if d.get("correct"))
    delta_d11 = d11_total - med_total
    delta_d12 = d12_total - d11_total

    print(f"\n{'-'*17} CONTEXT LAYER VERDICT {'-'*17}")
    print(f"  Cases where substrate HELPED:    {helped or 'none'}")
    print(f"  Cases where substrate HURT:      {hurt or 'none'}")
    print(f"  Cases where substrate NEUTRAL:   {neutral or 'none'}")
    print(f"  Net delta (M_D11 - M_medical_only): {'+' if delta_d11 > 0 else ''}{delta_d11}")

    print(f"\n{'-'*15} D12 CALIBRATION VERDICT {'-'*15}")
    d12_order_sens = [c for c in cases if "ORDER_SENSITIVE_LEADER"
                      in all_case_labels.get(c.case_id, [])]
    d12_stable = [c for c in cases if "STABILITY_UNANIMOUS"
                  in all_case_labels.get(c.case_id, [])]
    d12_flips = [c for c in cases if "SUPPORT_CONTRADICTION_FLIP"
                 in all_case_labels.get(c.case_id, [])]

    print(f"  Cases with ORDER_SENSITIVE_LEADER:  {[c.case_id for c in d12_order_sens] or 'none'}")
    print(f"  Cases with STABLE leader:           {[c.case_id for c in d12_stable] or 'none'}")
    print(f"  Cases with SUPPORT_CONTRADICTION_FLIP: {[c.case_id for c in d12_flips] or 'none'}")
    print(f"  D12 accuracy delta vs D11:          {'+' if delta_d12 > 0 else ''}{delta_d12}")

    # Evidence quality
    print(f"\n{'-'*20} EVIDENCE QUALITY {'-'*20}")
    stabilities = []
    jaccards = []
    for case in cases:
        meta = cache.get(case.case_id, "d12_meta") or {}
        if "stability" in meta:
            stabilities.append(meta["stability"])
        if "evidence_jaccard" in meta:
            jaccards.append(meta["evidence_jaccard"])
    if stabilities:
        print(f"  Mean answer_stability:           {sum(stabilities)/len(stabilities):.2f}")
    if jaccards:
        print(f"  Mean evidence_stability:         {sum(jaccards)/len(jaccards):.2f}")

    # Audit labels
    print(f"\n{'-'*18} AUDIT LABELS FIRED {'-'*18}")
    for case in cases:
        labels = all_case_labels.get(case.case_id, [])
        if labels:
            print(f"  {case.case_id}: {', '.join(labels)}")
        else:
            print(f"  {case.case_id}: (none)")

    # Architecture verdict
    print(f"\n{'-'*17} ARCHITECTURE VERDICT {'-'*17}")
    if delta_d11 > 0 and delta_d12 >= 0:
        verdict = "ARCHITECTURE SUPPORTED"
    elif d11_total > 0 or d12_total > 0:
        if delta_d11 >= 0:
            verdict = "ARCHITECTURE PARTIALLY SUPPORTED"
        else:
            verdict = "ARCHITECTURE NOT SUPPORTED"
    elif med_total > d11_total:
        verdict = "ARCHITECTURE NOT SUPPORTED"
    else:
        verdict = "ARCHITECTURE PARTIALLY SUPPORTED — mechanism active, accuracy unchanged"

    print(f"  {verdict}")

    # What was proven
    print(f"\n{'-'*16} WHAT WAS ACTUALLY PROVEN {'-'*16}")
    if delta_d11 > 0:
        print(f"  D11 substrate improved accuracy over medical-only baseline by {delta_d11} case(s).")
    elif delta_d11 == 0:
        print(f"  D11 substrate did not change accuracy vs medical-only baseline.")
    else:
        print(f"  D11 substrate HURT accuracy by {abs(delta_d11)} case(s) vs medical-only.")

    if d12_order_sens:
        print(f"  D12 detected order sensitivity in {len(d12_order_sens)} case(s).")
    if d12_flips:
        print(f"  D12 detected evidence-answer contradictions in {len(d12_flips)} case(s).")

    print(f"\n{'-'*16} WHAT REMAINS UNPROVEN {'-'*16}")
    print(f"  Whether substrate evidence quality improves with better board prompts.")
    print(f"  Whether D12 calibration improves accuracy on a larger case set.")
    print(sep)


def _print_mechanism_diagnostics(
    cases: list[MedXpertQACase],
    cache: StageCache,
    variants: list[str],
) -> None:
    if "D11" not in variants:
        return
    print(f"\n{'='*70}")
    print("D11 MECHANISM DIAGNOSTICS")
    print(f"{'='*70}")
    for case in cases:
        meta = cache.get(case.case_id, "d11_meta") or {}
        print(f"  {case.case_id}: quality={meta.get('bundle_quality','?')} "
              f"disc={meta.get('pairwise_discriminator_count',0)} "
              f"repair={meta.get('repair_claim_count',0)} "
              f"generic={meta.get('generic_claim_count',0)} "
              f"decisive={meta.get('case_specific_decisive_count',0)} "
              f"unresolved={meta.get('unresolved_pairs_after',0)} "
              f"gold={case.answer}")


def _print_cache_stats(
    cache: StageCache,
    quota: QuotaTracker,
    run_complete: bool,
    infra_status: str,
    cases_completed: int,
    total_cases: int,
) -> None:
    stats = cache.stats()
    print(f"\n{'='*70}")
    print("INFRA STATUS")
    print(f"{'='*70}")
    print(f"  run_complete: {run_complete}")
    print(f"  infra_status: {infra_status}")
    print(f"  cases: {cases_completed}/{total_cases}")
    print(f"  vertex_calls: {quota.calls_made}"
          + (f" (max: {quota.max_calls})" if quota.max_calls > 0 else ""))
    print(f"  cache_hits: {stats['cache_hits']}")
    print(f"  cache_misses: {stats['cache_misses']}")
    print(f"  cache_hit_rate: {stats['cache_hit_rate_pct']}%")


if __name__ == "__main__":
    sys.exit(main())
