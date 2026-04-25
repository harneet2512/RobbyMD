"""Adversarial tests for the D11 strict sufficiency auditor.

Tests that generic-only evidence does not produce 'strong' quality,
that multiple supported candidates without discriminators are not 'strong',
that leading candidates without runner-up ruleouts fail, that confident
wrong evidence is flagged, and that all criteria met yields 'strong'.
"""
from __future__ import annotations

import pytest

from eval.medxpertqa.d11.strict_sufficiency_auditor import audit_sufficiency
from eval.medxpertqa.d11.types import (
    CandidateEvidence,
    Claim,
    PairwiseDiscriminator,
    SufficiencyAudit,
)


class TestSufficiencyAudit:
    def test_generic_facts_do_not_create_sufficiency(self):
        """15 generic claims, no case-specific -> quality=generic or insufficient, decisive=0."""
        evidence = [
            CandidateEvidence(
                "cand_A",
                "A",
                supporting_claims=[Claim(f"Generic fact {i}", False, "mechanism", "weak") for i in range(15)],
                contradicting_claims=[],
                missing_required_clues=[],
                generic_claims=[],
                net_status="supported",
            ),
        ]
        audit = audit_sufficiency(evidence, discriminators=[], trap_candidates=[])
        assert audit.bundle_quality in ("generic", "insufficient")
        assert audit.case_specific_decisive_claim_count == 0
        assert audit.repair_required

    def test_multiple_supported_not_strong(self):
        """A and B both supported, no discriminator -> not strong."""
        evidence = [
            CandidateEvidence(
                "cand_A",
                "A",
                [Claim("Case-specific support for A", True, "mechanism", "strong")],
                [],
                [],
                [],
                "supported",
            ),
            CandidateEvidence(
                "cand_B",
                "B",
                [Claim("Case-specific support for B", True, "mechanism", "strong")],
                [],
                [],
                [],
                "supported",
            ),
        ]
        audit = audit_sufficiency(evidence, discriminators=[], trap_candidates=[])
        assert audit.bundle_quality != "strong"
        pair_sets = [set(p) for p in audit.unresolved_pairs]
        assert {"cand_A", "cand_B"} in pair_sets

    def test_leading_requires_runner_up_ruleout(self):
        """A has more support than B, but B not ruled out -> not strong."""
        evidence = [
            CandidateEvidence(
                "cand_A",
                "A",
                [Claim("Strong A1", True, "mechanism", "strong"), Claim("Strong A2", True, "mechanism", "strong")],
                [],
                [],
                [],
                "supported",
            ),
            CandidateEvidence(
                "cand_B",
                "B",
                [Claim("Some B support", True, "mechanism", "medium")],
                [],
                [],
                [],
                "supported",
            ),
        ]
        # No discriminator ruling out B
        audit = audit_sufficiency(evidence, discriminators=[], trap_candidates=[])
        assert audit.bundle_quality != "strong"
        assert audit.repair_required

    def test_confident_wrong_evidence_marked_misleading(self):
        """Strong support for H but E also plausible, no discriminator -> misleading or conflicting."""
        evidence = [
            CandidateEvidence(
                "cand_H",
                "H",
                [
                    Claim("Strongly supports H", True, "mechanism", "strong"),
                    Claim("Another strong H", True, "mechanism", "strong"),
                ],
                [],
                [],
                [],
                "supported",
            ),
            CandidateEvidence(
                "cand_E",
                "E",
                [Claim("E also fits", True, "mechanism", "medium")],
                [],
                [],
                [],
                "supported",
            ),
        ]
        audit = audit_sufficiency(evidence, discriminators=[], trap_candidates=[])
        assert audit.bundle_quality in ("underdetermined", "conflicting", "misleading")

    def test_all_criteria_met_is_strong(self):
        """When all 6 criteria pass, quality=strong."""
        evidence = [
            CandidateEvidence(
                "cand_A",
                "A",
                [Claim("Decisive for A", True, "mechanism", "strong")],
                [],
                [],
                [],
                "supported",
            ),
            CandidateEvidence(
                "cand_B",
                "B",
                [],
                [Claim("Rules out B", True, "skeptic", "strong")],
                [],
                [],
                "contradicted",
            ),
        ]
        disc = PairwiseDiscriminator(
            pair=("cand_A", "cand_B"),
            discriminator="Key finding X",
            case_clue="Finding X present",
            supports="cand_A",
            rules_out="cand_B",
            confidence="high",
            why_decisive="X is pathognomonic for A",
        )
        audit = audit_sufficiency(evidence, discriminators=[disc], trap_candidates=[])
        assert audit.bundle_quality == "strong"
        assert not audit.repair_required
