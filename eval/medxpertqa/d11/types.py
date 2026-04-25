"""Shared dataclasses for the D11 differential compiler.

All modules in the d11 package import from here to ensure
consistent type definitions across pipeline stages.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ClinicalAbstraction:
    clinical_problem: str
    key_findings: list[str]
    temporal_pattern: str
    body_system: str
    specialty_hint: str
    task_type: str  # diagnosis|mechanism|management|anatomy|pathology|adverse_effect|next_step|other
    missing_context: list[str]


@dataclass
class CandidateHypothesis:
    candidate_id: str      # e.g. "cand_A" for MedXpertQA, "cand_1" for product
    candidate_label: str   # the actual text
    candidate_type: str    # diagnosis|mechanism|treatment|anatomy|complication|other


@dataclass
class Claim:
    claim: str
    case_specific: bool
    source_role: str       # mechanism_specialist|option_skeptic|trap_detector
    strength: str          # weak|medium|strong
    related_candidate_id: str = ""


@dataclass
class CandidateEvidence:
    candidate_id: str
    candidate_label: str
    supporting_claims: list[Claim]
    contradicting_claims: list[Claim]
    missing_required_clues: list[str]
    generic_claims: list[Claim]
    net_status: str  # supported|contradicted|unresolved|trap|insufficient


@dataclass
class MechanismOutput:
    candidate_id: str
    supporting_clues: list[str]
    contradicting_clues: list[str]


@dataclass
class SkepticOutput:
    candidate_id: str
    missing_required_clues: list[str]
    contradictions: list[str]


@dataclass
class TrapOutput:
    candidate_id: str
    is_trap: bool
    why_tempting: str
    exposing_clue: str


@dataclass
class BoardResults:
    mechanism_outputs: list[MechanismOutput]
    skeptic_outputs: list[SkepticOutput]
    trap_outputs: list[TrapOutput]


@dataclass
class PairwiseDiscriminator:
    pair: tuple[str, str]  # (candidate_id_a, candidate_id_b)
    discriminator: str
    case_clue: str
    supports: str          # candidate_id or "neither" or "unclear"
    rules_out: str         # candidate_id or "neither" or "unclear"
    confidence: str        # low|medium|high
    why_decisive: str


@dataclass
class SufficiencyAudit:
    bundle_quality: str    # strong|underdetermined|conflicting|misleading|generic|insufficient
    leading_candidates: list[str]
    runner_up_candidates: list[str]
    unresolved_pairs: list[tuple[str, str]]
    missing_discriminators: list[tuple[str, str]]
    misleading_claims: list[Claim]
    generic_claim_count: int
    case_specific_decisive_claim_count: int
    repair_required: bool


@dataclass
class RepairClaim:
    pair: tuple[str, str]
    claim: str
    supports: str
    rules_out: str
    case_clue: str
    confidence: str


@dataclass
class FinalBundle:
    clinical_abstraction_text: str
    candidate_evidence_table: str
    pairwise_discriminators_text: str
    trap_warnings: str
    missing_information: str
    decision_evidence: str
    full_text: str  # all sections concatenated
    token_estimate: int
    repair_claims_included: int
    total_claims: int
