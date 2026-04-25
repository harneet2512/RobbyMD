"""D11 No-RAG Differential Compiler — evaluation runner.

Runs A, C, D, D10, D11, E on the same cases.
D11 pipeline: abstraction -> candidates -> board(×3) -> attribution
  -> pairwise tournament -> sufficiency audit -> repair -> bundle -> reader.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import structlog

from eval.medxpertqa.adapter import ANSWER_OPTIONS, MedXpertQACase, iter_cases
from eval.medxpertqa.baseline import _extract_answer
from eval.medxpertqa.pipeline_cache import PipelineCache
from eval.medxpertqa.retry_utils import retry_with_backoff

log = structlog.get_logger(__name__)

REPO = Path(__file__).resolve().parents[2]
REPORTS = REPO / "eval" / "reports" / "medxpertqa"


def _call_gemini(case: MedXpertQACase, prompt: str, model: str) -> str:
    from eval.medxpertqa.gemini_reader import _get_token, HARNEET_PROJECT
    import requests

    def _do():
        token = _get_token()
        url = (f"https://us-central1-aiplatform.googleapis.com/v1/"
               f"projects/{HARNEET_PROJECT}/locations/us-central1/"
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


def _baseline_prompt(case: MedXpertQACase) -> str:
    opts = "\n".join(f"{k}. {case.options[k]}" for k in ANSWER_OPTIONS if k in case.options)
    return (f"You are a medical expert. Read the clinical vignette and answer.\n\n"
            f"{case.vignette}\n\nOptions:\n{opts}\n\n"
            f"Reason step by step.\nANSWER: [A-J]")


def _build_d11_bundle(case: MedXpertQACase, cache: PipelineCache) -> tuple[str, dict[str, Any]]:
    """Full D11 pipeline with caching at every stage."""
    from eval.medxpertqa.d11.clinical_abstraction import abstract_clinical_case
    from eval.medxpertqa.d11.candidate_hypothesis_adapter import adapt_medxpertqa_options
    from eval.medxpertqa.d11.differential_board import run_differential_board
    from eval.medxpertqa.d11.evidence_attributor import attribute_evidence
    from eval.medxpertqa.d11.pairwise_discriminator_tournament import select_pairs, run_pairwise_tournament
    from eval.medxpertqa.d11.strict_sufficiency_auditor import audit_sufficiency
    from eval.medxpertqa.d11.targeted_discriminator_repair import repair_discriminators
    from eval.medxpertqa.d11.final_evidence_bundle import build_final_bundle
    from eval.medxpertqa.d11.types import (
        ClinicalAbstraction, BoardResults, CandidateHypothesis,
        PairwiseDiscriminator, SufficiencyAudit, RepairClaim, FinalBundle,
    )
    import dataclasses

    cid = case.case_id

    # Stage 1: Clinical abstraction
    if cache.has(cid, "d11_abstraction"):
        abstr_dict = cache.get(cid, "d11_abstraction")
        abstraction = ClinicalAbstraction(**abstr_dict)
    else:
        abstraction = abstract_clinical_case(case.vignette, case.options, case_id=cid)
        cache.put(cid, "d11_abstraction", dataclasses.asdict(abstraction))

    # Stage 2: Candidate hypotheses
    candidates = adapt_medxpertqa_options(case.options)

    # Stage 3: Differential board (3 parallel roles)
    if cache.has(cid, "d11_board"):
        board_dict = cache.get(cid, "d11_board")
        from eval.medxpertqa.d11.types import MechanismOutput, SkepticOutput, TrapOutput
        board = BoardResults(
            mechanism_outputs=[MechanismOutput(**m) for m in board_dict["mechanism_outputs"]],
            skeptic_outputs=[SkepticOutput(**s) for s in board_dict["skeptic_outputs"]],
            trap_outputs=[TrapOutput(**t) for t in board_dict["trap_outputs"]],
        )
    else:
        board = run_differential_board(abstraction, candidates, case_id=cid)
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
        discriminators = run_pairwise_tournament(pairs, abstraction, candidates, case_id=cid)
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
                abstraction, candidates, case_id=cid,
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


def _enrichment_prompt(case: MedXpertQACase, cache: PipelineCache) -> str:
    """Variant C: raw enrichment text as prompt context."""
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="D11 No-RAG Differential Compiler evaluation")
    parser.add_argument("--reader", default="gemini-2.5-flash")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--replay", type=int, default=1)
    parser.add_argument("--cases", default=str(REPO / "eval" / "data" / "medxpertqa" / "medxpertqa_text.jsonl"))
    parser.add_argument("--variants", default="A,C,D11,E",
                        help="Comma-separated variants to run (A,C,D,D10,D11,E)")
    args = parser.parse_args(argv)

    variants = [v.strip() for v in args.variants.split(",")]
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = REPORTS / f"d11_{args.reader.replace('.', '_')}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cache = PipelineCache(run_dir / "cache")

    all_cases = list(iter_cases(Path(args.cases)))
    cases = all_cases[:args.limit]
    if not cases:
        print(f"No cases found at {args.cases}", file=sys.stderr)
        return 1

    print(f"[d11] reader={args.reader} n={len(cases)} replay={args.replay} variants={variants}")

    # Stage 1: Build D11 bundles
    if "D11" in variants:
        print("\n=== Stage 1: Build D11 bundles ===")
        for case in cases:
            try:
                bundle_text, meta = _build_d11_bundle(case, cache)
                log.info("d11.bundle.done", case_id=case.case_id,
                         quality=meta["bundle_quality"],
                         discriminators=meta["pairwise_discriminator_count"],
                         repair=meta["repair_claim_count"])
                print(f"  {case.case_id}: quality={meta['bundle_quality']} "
                      f"disc={meta['pairwise_discriminator_count']} "
                      f"repair={meta['repair_claim_count']} "
                      f"tokens={meta['bundle_tokens']}")
            except Exception as e:
                log.error("d11.bundle.error", case_id=case.case_id, err=repr(e)[:200])
                print(f"  {case.case_id}: ERROR {repr(e)[:100]}")

    # Stage 2: Reader evaluation
    print(f"\n=== Stage 2: Reader evaluation (replay x{args.replay}) ===")
    results: dict[str, dict[str, Any]] = {v: {} for v in variants}

    for variant in variants:
        print(f"\n--- {variant} ---")
        for case in cases:
            answers: list[str] = []
            for rep in range(args.replay):
                try:
                    if variant == "A":
                        prompt = _baseline_prompt(case)
                    elif variant == "C":
                        prompt = _enrichment_prompt(case, cache)
                    elif variant == "D11":
                        if cache.has(case.case_id, "d11_meta"):
                            bundle_text = cache.get(case.case_id, "d11_bundle_text")
                            if bundle_text is None:
                                _, _ = _build_d11_bundle(case, cache)
                                # Re-read from the built bundle
                                from eval.medxpertqa.d11.types import FinalBundle
                                meta = cache.get(case.case_id, "d11_meta")
                            # Build reader prompt from cached bundle
                            # We need the full_text — reconstruct from stages
                            import dataclasses
                            from eval.medxpertqa.d11.clinical_abstraction import abstract_clinical_case
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
                            abstraction = ClinicalAbstraction(**abstr_dict)
                            candidates = adapt_medxpertqa_options(case.options)
                            board_dict = cache.get(case.case_id, "d11_board")
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
                            prompt = _d11_reader_prompt(case, bundle.full_text)
                        else:
                            bundle_text_raw, _ = _build_d11_bundle(case, cache)
                            prompt = _d11_reader_prompt(case, bundle_text_raw)
                    elif variant == "E":
                        prompt = _oracle_prompt(case)
                    else:
                        prompt = _baseline_prompt(case)

                    raw = _call_gemini(case, prompt, args.reader)
                    answer = _extract_answer(raw)
                    answers.append(answer)
                except Exception as e:
                    log.warning("reader.error", variant=variant, case_id=case.case_id, err=repr(e)[:100])
                    answers.append("")
                if rep < args.replay - 1:
                    time.sleep(1)

            majority = Counter(answers).most_common(1)[0][0] if answers else ""
            correct = majority == case.answer
            marker = "Y" if correct else "N"
            variance = len(set(a for a in answers if a))
            print(f"  {case.case_id}: {answers} -> majority={majority} gold={case.answer} {marker} var={variance}")

            results[variant][case.case_id] = {
                "answers": answers,
                "majority": majority,
                "correct": correct,
                "variance": variance,
                "gold": case.answer,
            }

    # Save results
    results_path = run_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

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

    # D11 diagnostics
    if "D11" in variants:
        print(f"\n{'='*70}")
        print("D11 MECHANISM DIAGNOSTICS")
        print(f"{'='*70}")
        for case in cases:
            meta = cache.get(case.case_id, "d11_meta") or {}
            d11_pred = results.get("D11", {}).get(case.case_id, {}).get("majority", "?")
            print(f"  {case.case_id}: quality={meta.get('bundle_quality','?')} "
                  f"disc={meta.get('pairwise_discriminator_count',0)} "
                  f"repair={meta.get('repair_claim_count',0)} "
                  f"generic={meta.get('generic_claim_count',0)} "
                  f"decisive={meta.get('case_specific_decisive_count',0)} "
                  f"unresolved={meta.get('unresolved_pairs_after',0)} "
                  f"pred={d11_pred} gold={case.answer}")

    print(f"\n[d11] Report: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
