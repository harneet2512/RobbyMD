"""Adversarial tests for the D11 evidence attributor.

Tests that case-specific vs generic claims are correctly tagged,
contradictions are preserved, trap candidates get net_status='trap',
and missing clues from the skeptic role propagate correctly.
"""
from __future__ import annotations

import pytest

from eval.medxpertqa.d11.evidence_attributor import attribute_evidence
from eval.medxpertqa.d11.types import (
    BoardResults,
    CandidateEvidence,
    CandidateHypothesis,
    Claim,
    MechanismOutput,
    SkepticOutput,
    TrapOutput,
)


class TestEvidenceAttribution:
    def test_generic_facts_marked_not_case_specific(self):
        """Generic medical facts must be case_specific=False."""
        board = BoardResults(
            mechanism_outputs=[
                MechanismOutput("cand_A", ["Hypertension can cause headaches"], []),
                MechanismOutput("cand_B", ["Diabetes may lead to neuropathy"], []),
            ],
            skeptic_outputs=[],
            trap_outputs=[],
        )
        candidates = [
            CandidateHypothesis("cand_A", "Hypertension", "diagnosis"),
            CandidateHypothesis("cand_B", "Diabetes", "diagnosis"),
        ]
        evidence = attribute_evidence(board, candidates)
        # Generic claims (no case reference) must be tagged case_specific=False
        for ev in evidence:
            for claim in ev.supporting_claims + ev.generic_claims:
                if "can cause" in claim.claim or "may lead to" in claim.claim:
                    assert not claim.case_specific, f"Generic claim marked case_specific: {claim.claim}"

    def test_case_specific_claims_tagged_correctly(self):
        """Claims referencing case findings must be case_specific=True."""
        board = BoardResults(
            mechanism_outputs=[
                MechanismOutput("cand_A", ["The patient's elevated troponin supports MI"], []),
            ],
            skeptic_outputs=[],
            trap_outputs=[],
        )
        candidates = [CandidateHypothesis("cand_A", "MI", "diagnosis")]
        evidence = attribute_evidence(board, candidates)
        all_claims = evidence[0].supporting_claims + evidence[0].generic_claims
        patient_claims = [c for c in all_claims if "patient" in c.claim.lower()]
        assert len(patient_claims) > 0, "Patient-referencing claim should exist"
        for claim in patient_claims:
            assert claim.case_specific

    def test_contradictions_preserved(self):
        """Contradicting claims must not be dropped or hidden."""
        board = BoardResults(
            mechanism_outputs=[
                MechanismOutput("cand_A", ["Supports A"], ["Age pattern contradicts A"]),
            ],
            skeptic_outputs=[
                SkepticOutput("cand_A", [], ["No fever argues against A"]),
            ],
            trap_outputs=[],
        )
        candidates = [CandidateHypothesis("cand_A", "Candidate A", "diagnosis")]
        evidence = attribute_evidence(board, candidates)
        assert len(evidence[0].contradicting_claims) >= 2

    def test_trap_candidate_status(self):
        """Candidates flagged as traps must have net_status='trap'."""
        board = BoardResults(
            mechanism_outputs=[MechanismOutput("cand_A", ["Looks like A"], [])],
            skeptic_outputs=[],
            trap_outputs=[TrapOutput("cand_A", True, "Looks correct superficially", "But age is wrong")],
        )
        candidates = [CandidateHypothesis("cand_A", "Trap candidate", "diagnosis")]
        evidence = attribute_evidence(board, candidates)
        assert evidence[0].net_status == "trap"

    def test_missing_clues_from_skeptic(self):
        """Missing required clues from skeptic must appear in evidence."""
        board = BoardResults(
            mechanism_outputs=[MechanismOutput("cand_A", ["Some support"], [])],
            skeptic_outputs=[SkepticOutput("cand_A", ["Expected fever", "Expected rash"], [])],
            trap_outputs=[],
        )
        candidates = [CandidateHypothesis("cand_A", "Candidate A", "diagnosis")]
        evidence = attribute_evidence(board, candidates)
        assert len(evidence[0].missing_required_clues) == 2

    def test_missing_clue_does_not_hard_veto_strong_support(self):
        """3 strong case-specific support + 1 nonfatal missing clue -> supported, not blocked."""
        board = BoardResults(
            mechanism_outputs=[
                MechanismOutput("cand_A", [
                    "The patient's troponin elevation supports MI",
                    "The patient's ST-segment changes are consistent with MI",
                    "The patient's chest pain radiation pattern supports MI",
                ], []),
            ],
            skeptic_outputs=[SkepticOutput("cand_A", ["No echocardiogram mentioned"], [])],
            trap_outputs=[],
        )
        candidates = [CandidateHypothesis("cand_A", "Myocardial infarction", "diagnosis")]
        evidence = attribute_evidence(board, candidates)
        assert evidence[0].net_status in ("supported", "unresolved"), (
            f"3 strong case-specific claims + 1 missing clue should not be insufficient, "
            f"got {evidence[0].net_status}"
        )
        assert evidence[0].net_status == "supported", (
            f"Expected supported with strong case-specific support, got {evidence[0].net_status}"
        )

    def test_generic_support_does_not_create_supported(self):
        """5 generic support claims, no case-specific -> not supported."""
        board = BoardResults(
            mechanism_outputs=[
                MechanismOutput("cand_A", [
                    "MI can cause chest pain",
                    "MI is associated with elevated troponin",
                    "MI typically presents with substernal discomfort",
                    "MI is a common cause of cardiac arrest",
                    "MI may lead to cardiogenic shock",
                ], []),
            ],
            skeptic_outputs=[],
            trap_outputs=[],
        )
        candidates = [CandidateHypothesis("cand_A", "MI", "diagnosis")]
        evidence = attribute_evidence(board, candidates)
        assert evidence[0].net_status != "supported", (
            "Generic-only support must not produce 'supported' status"
        )

    def test_fatal_contradiction_blocks_supported(self):
        """Strong case-specific contradiction blocks supported even with support."""
        board = BoardResults(
            mechanism_outputs=[
                MechanismOutput("cand_A",
                    ["The patient's troponin elevation supports MI"],
                    ["The patient's normal ECG and negative stress test rule out MI"]),
            ],
            skeptic_outputs=[
                SkepticOutput("cand_A", [], ["The patient's age of 22 is atypical for MI"]),
            ],
            trap_outputs=[],
        )
        candidates = [CandidateHypothesis("cand_A", "MI", "diagnosis")]
        evidence = attribute_evidence(board, candidates)
        assert evidence[0].net_status == "contradicted"

    def test_balanced_evidence_becomes_unresolved(self):
        """Meaningful support and meaningful contradiction -> unresolved."""
        board = BoardResults(
            mechanism_outputs=[
                MechanismOutput("cand_A",
                    ["The patient's tachycardia supports PE",
                     "The patient's hypoxia is consistent with PE"],
                    ["The patient's D-dimer is borderline"]),
            ],
            skeptic_outputs=[SkepticOutput("cand_A", ["No CT angiogram performed"], [])],
            trap_outputs=[],
        )
        candidates = [CandidateHypothesis("cand_A", "PE", "diagnosis")]
        evidence = attribute_evidence(board, candidates)
        assert evidence[0].net_status == "unresolved"

    def test_attributor_produces_leader_when_evidence_supports(self):
        """When one candidate has clear case-specific support, leaders should not be empty."""
        from eval.medxpertqa.d11.strict_sufficiency_auditor import audit_sufficiency
        board = BoardResults(
            mechanism_outputs=[
                MechanismOutput("cand_A", [
                    "The patient's troponin elevation strongly supports MI",
                    "The patient's ST changes are diagnostic for MI",
                    "The patient's risk factors point to MI",
                ], []),
                MechanismOutput("cand_B", [
                    "PE can cause chest pain",
                ], ["The patient's D-dimer is normal, arguing against PE"]),
                MechanismOutput("cand_C", [
                    "Costochondritis is a common cause of chest pain",
                ], []),
            ],
            skeptic_outputs=[
                SkepticOutput("cand_A", ["No echocardiogram"], []),
                SkepticOutput("cand_B", ["Expected tachycardia"], ["Normal D-dimer"]),
                SkepticOutput("cand_C", ["Expected reproducible tenderness"], []),
            ],
            trap_outputs=[],
        )
        candidates = [
            CandidateHypothesis("cand_A", "MI", "diagnosis"),
            CandidateHypothesis("cand_B", "PE", "diagnosis"),
            CandidateHypothesis("cand_C", "Costochondritis", "diagnosis"),
        ]
        evidence = attribute_evidence(board, candidates)
        supported = [e for e in evidence if e.net_status == "supported"]
        assert len(supported) >= 1, (
            f"Expected at least one supported candidate, got statuses: "
            f"{[e.net_status for e in evidence]}"
        )
        audit = audit_sufficiency(evidence, [], [])
        assert len(audit.leading_candidates) >= 1, (
            f"Expected at least one leader, got {audit.leading_candidates}"
        )
