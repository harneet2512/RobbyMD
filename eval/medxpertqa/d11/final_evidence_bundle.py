"""Assemble the final evidence bundle for the reader.

Concatenates clinical abstraction, candidate evidence, pairwise
discriminators, repair claims, trap warnings, and missing information
into a single structured text block.  Pure Python — no API calls.
"""
from __future__ import annotations

import structlog

from eval.medxpertqa.d11.types import (
    CandidateEvidence,
    ClinicalAbstraction,
    FinalBundle,
    PairwiseDiscriminator,
    RepairClaim,
    SufficiencyAudit,
)

log = structlog.get_logger(__name__)


def _render_clinical_abstraction(abstraction: ClinicalAbstraction) -> str:
    """Render section 1 — clinical abstraction."""
    findings = ", ".join(abstraction.key_findings) if abstraction.key_findings else "(none)"
    return (
        f"Problem: {abstraction.clinical_problem}\n"
        f"Key findings: {findings}\n"
        f"Temporal: {abstraction.temporal_pattern}\n"
        f"Body system: {abstraction.body_system}"
    )


def _render_candidate_evidence_table(evidence: list[CandidateEvidence]) -> str:
    """Render section 2 — per-candidate evidence table.

    Skips candidates with net_status == 'insufficient'.
    """
    lines: list[str] = []
    for ev in evidence:
        if ev.net_status == "insufficient":
            continue

        lines.append(f"{ev.candidate_id} ({ev.candidate_label}): {ev.net_status}")

        if ev.supporting_claims:
            support_texts = ", ".join(c.claim for c in ev.supporting_claims)
            lines.append(f"  Supporting: {support_texts}")

        if ev.contradicting_claims:
            contra_texts = ", ".join(c.claim for c in ev.contradicting_claims)
            lines.append(f"  Against: {contra_texts}")

        if ev.missing_required_clues:
            missing_texts = ", ".join(ev.missing_required_clues)
            lines.append(f"  Missing: {missing_texts}")

    return "\n".join(lines) if lines else "(no candidate evidence)"


def _render_pairwise_discriminators(
    discriminators: list[PairwiseDiscriminator],
) -> str:
    """Render section 3 — pairwise discriminators."""
    if not discriminators:
        return "(no pairwise discriminators)"

    lines: list[str] = []
    for d in discriminators:
        pair_str = f"{d.pair[0]} vs {d.pair[1]}"
        lines.append(
            f"[{pair_str}] {d.discriminator}\n"
            f"  Case clue: {d.case_clue}\n"
            f"  Supports: {d.supports} | Rules out: {d.rules_out}\n"
            f"  Confidence: {d.confidence} — {d.why_decisive}"
        )
    return "\n".join(lines)


def _render_repair_claims(repair_claims: list[RepairClaim]) -> str:
    """Render section 4 — repair claims.  MUST include all; never drop."""
    if not repair_claims:
        return "(no repair claims)"

    lines: list[str] = []
    for rc in repair_claims:
        pair_str = f"{rc.pair[0]} vs {rc.pair[1]}"
        lines.append(
            f"[REPAIR {pair_str}] {rc.claim}\n"
            f"  Case clue: {rc.case_clue}\n"
            f"  Supports: {rc.supports} | Rules out: {rc.rules_out}\n"
            f"  Confidence: {rc.confidence}"
        )
    return "\n".join(lines)


def _render_trap_warnings(evidence: list[CandidateEvidence]) -> str:
    """Render section 5 — candidates flagged as traps."""
    traps = [ev for ev in evidence if ev.net_status == "trap"]
    if not traps:
        return "(no trap warnings)"

    lines: list[str] = []
    for ev in traps:
        exposing = ", ".join(c.claim for c in ev.contradicting_claims) if ev.contradicting_claims else "(no exposing clue)"
        lines.append(
            f"TRAP: {ev.candidate_id} ({ev.candidate_label})\n"
            f"  Exposing clue: {exposing}"
        )
    return "\n".join(lines)


def _render_missing_information(
    abstraction: ClinicalAbstraction,
    audit: SufficiencyAudit,
) -> str:
    """Render section 6 — missing context and unresolved pairs."""
    lines: list[str] = []

    if abstraction.missing_context:
        for ctx in abstraction.missing_context:
            lines.append(f"- Missing context: {ctx}")

    if audit.unresolved_pairs:
        for pair in audit.unresolved_pairs:
            lines.append(f"- Unresolved pair: {pair[0]} vs {pair[1]}")

    return "\n".join(lines) if lines else "(no missing information)"


def _render_decision_evidence(audit: SufficiencyAudit) -> str:
    """Render a concise decision-evidence summary from the audit."""
    lines: list[str] = [
        f"Bundle quality: {audit.bundle_quality}",
        f"Leading candidates: {', '.join(audit.leading_candidates) if audit.leading_candidates else '(none)'}",
        f"Runner-ups: {', '.join(audit.runner_up_candidates) if audit.runner_up_candidates else '(none)'}",
        f"Case-specific decisive claims: {audit.case_specific_decisive_claim_count}",
        f"Generic claims: {audit.generic_claim_count}",
        f"Repair required: {audit.repair_required}",
    ]
    return "\n".join(lines)


def _count_total_claims(evidence: list[CandidateEvidence]) -> int:
    """Total claims across all candidates (supporting + contradicting + generic)."""
    total = 0
    for ev in evidence:
        total += len(ev.supporting_claims)
        total += len(ev.contradicting_claims)
        total += len(ev.generic_claims)
    return total


def build_final_bundle(
    abstraction: ClinicalAbstraction,
    evidence: list[CandidateEvidence],
    discriminators: list[PairwiseDiscriminator],
    repair_claims: list[RepairClaim],
    audit: SufficiencyAudit,
) -> FinalBundle:
    """Assemble the final evidence bundle for the reader.

    Args:
        abstraction: Structured clinical abstraction of the case.
        evidence: Per-candidate evidence from the evidence attributor.
        discriminators: Pairwise discriminators from the board.
        repair_claims: Targeted repair claims for unresolved pairs.
        audit: Sufficiency audit results.

    Returns:
        A :class:`FinalBundle` with all sections rendered and concatenated.
    """
    section_abstraction = _render_clinical_abstraction(abstraction)
    section_evidence = _render_candidate_evidence_table(evidence)
    section_disc = _render_pairwise_discriminators(discriminators)
    section_repair = _render_repair_claims(repair_claims)
    section_traps = _render_trap_warnings(evidence)
    section_missing = _render_missing_information(abstraction, audit)
    section_decision = _render_decision_evidence(audit)

    # Never drop repair claims — count must equal input length.
    repair_count = len(repair_claims)

    full_text = "\n\n".join([
        "=== CLINICAL ABSTRACTION ===",
        section_abstraction,
        "=== CANDIDATE EVIDENCE ===",
        section_evidence,
        "=== PAIRWISE DISCRIMINATORS ===",
        section_disc,
        "=== REPAIR CLAIMS ===",
        section_repair,
        "=== TRAP WARNINGS ===",
        section_traps,
        "=== MISSING INFORMATION ===",
        section_missing,
        "=== DECISION EVIDENCE ===",
        section_decision,
    ])

    token_estimate = int(len(full_text.split()) * 1.3)
    total_claims = _count_total_claims(evidence) + repair_count

    bundle = FinalBundle(
        clinical_abstraction_text=section_abstraction,
        candidate_evidence_table=section_evidence,
        pairwise_discriminators_text=section_disc,
        trap_warnings=section_traps,
        missing_information=section_missing,
        decision_evidence=section_decision,
        full_text=full_text,
        token_estimate=token_estimate,
        repair_claims_included=repair_count,
        total_claims=total_claims,
    )

    log.info(
        "final_bundle.done",
        token_estimate=token_estimate,
        total_claims=total_claims,
        repair_claims=repair_count,
        quality=audit.bundle_quality,
    )
    return bundle
