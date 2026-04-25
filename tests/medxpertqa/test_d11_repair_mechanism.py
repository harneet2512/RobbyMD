"""Adversarial tests for the D11 repair mechanism.

Tests that repair claims appear in the final bundle, that repair theater
(repairs that resolve nothing) is detectable, and that repair does not
trigger when sufficiency says it is not required.
"""
from __future__ import annotations

import pytest

from eval.medxpertqa.d11.final_evidence_bundle import build_final_bundle
from eval.medxpertqa.d11.strict_sufficiency_auditor import audit_sufficiency
from eval.medxpertqa.d11.types import (
    CandidateEvidence,
    Claim,
    ClinicalAbstraction,
    FinalBundle,
    PairwiseDiscriminator,
    RepairClaim,
    SufficiencyAudit,
)


class TestRepairMechanism:
    def test_repair_claim_enters_final_bundle(self):
        """Every repair claim must appear in the final bundle."""
        abstraction = ClinicalAbstraction(
            clinical_problem="Test problem",
            key_findings=["finding1"],
            temporal_pattern="acute",
            body_system="cardiovascular",
            specialty_hint="cardiology",
            task_type="diagnosis",
            missing_context=[],
        )
        evidence = [
            CandidateEvidence("cand_A", "A", [Claim("Supports A", True, "mechanism", "medium")], [], [], [], "supported"),
            CandidateEvidence("cand_B", "B", [Claim("Supports B", True, "mechanism", "medium")], [], [], [], "supported"),
        ]
        repair = RepairClaim(
            pair=("cand_A", "cand_B"),
            claim="Specific discriminator between A and B based on case finding X",
            supports="cand_A",
            rules_out="cand_B",
            case_clue="Finding X",
            confidence="high",
        )
        audit = SufficiencyAudit(
            bundle_quality="underdetermined",
            leading_candidates=["cand_A"],
            runner_up_candidates=["cand_B"],
            unresolved_pairs=[("cand_A", "cand_B")],
            missing_discriminators=[("cand_A", "cand_B")],
            misleading_claims=[],
            generic_claim_count=0,
            case_specific_decisive_claim_count=1,
            repair_required=True,
        )
        bundle = build_final_bundle(abstraction, evidence, [], [repair], audit)
        assert repair.claim in bundle.full_text
        assert bundle.repair_claims_included == 1

    def test_repair_theater_detected(self):
        """Repair fires but unresolved pairs unchanged -> audit detects theater."""
        # Build pre-repair audit with 2 unresolved pairs
        evidence = [
            CandidateEvidence("cand_A", "A", [Claim("A", True, "mechanism", "medium")], [], [], [], "supported"),
            CandidateEvidence("cand_B", "B", [Claim("B", True, "mechanism", "medium")], [], [], [], "supported"),
            CandidateEvidence("cand_C", "C", [Claim("C", True, "mechanism", "medium")], [], [], [], "supported"),
        ]
        # Repair generates claims but they don't resolve anything (supports/rules_out are "unclear")
        repairs = [
            RepairClaim(("cand_A", "cand_B"), "Vague claim", "unclear", "unclear", "some clue", "low"),
            RepairClaim(("cand_A", "cand_C"), "Another vague", "unclear", "unclear", "some clue", "low"),
        ]
        # Re-audit after repair -- unresolved pairs should still exist
        # The audit should detect this as theater
        post_repair_discs = [
            PairwiseDiscriminator(("cand_A", "cand_B"), "Vague", "some clue", "unclear", "unclear", "low", ""),
            PairwiseDiscriminator(("cand_A", "cand_C"), "Vague", "some clue", "unclear", "unclear", "low", ""),
        ]
        audit = audit_sufficiency(evidence, post_repair_discs, [])
        assert len(audit.unresolved_pairs) >= 2  # unresolved pairs NOT decreased
        assert audit.repair_required  # still needs repair

    def test_repair_only_triggers_when_needed(self):
        """Repair must NOT fire when sufficiency audit says repair_required=False."""
        evidence = [
            CandidateEvidence(
                "cand_A",
                "A",
                [Claim("Decisive", True, "mechanism", "strong")],
                [],
                [],
                [],
                "supported",
            ),
            CandidateEvidence(
                "cand_B",
                "B",
                [],
                [Claim("Ruled out", True, "skeptic", "strong")],
                [],
                [],
                "contradicted",
            ),
        ]
        disc = PairwiseDiscriminator(
            ("cand_A", "cand_B"), "Key finding", "clue", "cand_A", "cand_B", "high", "pathognomonic"
        )
        audit = audit_sufficiency(evidence, [disc], [])
        assert not audit.repair_required
