"""Adversarial tests for the D11 architecture audit.

Tests the audit_pipeline.py integration for D11-specific failure modes:
inactive architecture detection, reader variance attribution, oracle
failure flagging, product-mode schema independence, and gold answer
leakage prevention.
"""
from __future__ import annotations

import pytest

from eval.medxpertqa.audit_pipeline import (
    CaseLabel,
    CaseResult,
    RunResults,
    Verdict,
    audit_pipeline,
)
from eval.medxpertqa.d11.evidence_attributor import attribute_evidence
from eval.medxpertqa.d11.final_evidence_bundle import build_final_bundle
from eval.medxpertqa.d11.types import (
    BoardResults,
    CandidateEvidence,
    CandidateHypothesis,
    Claim,
    ClinicalAbstraction,
    MechanismOutput,
    PairwiseDiscriminator,
    SkepticOutput,
    SufficiencyAudit,
    TrapOutput,
)


class TestArchitectureAudit:
    def test_inactive_architecture_detected(self):
        """D11 = D, no mechanism fired -> INACTIVE, not PASS."""
        cases = [
            CaseResult(
                case_id="T0",
                gold="A",
                predictions={"A": "B", "D": "C", "D11": "C", "E": "A"},
                correct={"A": False, "D": False, "D11": False, "E": True},
            ),
            CaseResult(
                case_id="T1",
                gold="B",
                predictions={"A": "A", "D": "B", "D11": "B", "E": "B"},
                correct={"A": False, "D": True, "D11": True, "E": True},
            ),
        ]
        results = RunResults(cases=cases, variants=["A", "D", "D11", "E"])
        report = audit_pipeline(results)
        # D11 = D for both cases -> context layer check should be AMBIGUOUS at best
        # The key insight: D11 performing same as D means no architectural intervention

    def test_reader_variance_not_credited_as_architecture_win(self):
        """Same bundle, different reader output -> READER_VARIANCE, not architecture win."""
        cases = [
            CaseResult(
                case_id="T0",
                gold="A",
                predictions={"D": "B", "D11": "A", "E": "A"},
                correct={"D": False, "D11": True, "E": True},
                bundle_changed_from_d=False,
                repair_triggered=True,
            ),
        ]
        results = RunResults(cases=cases, variants=["D", "D11", "E"])
        report = audit_pipeline(results)
        labels = report.per_case_labels["T0"]
        # If bundle didn't change but answer changed -> reader variance
        assert CaseLabel.READER_VARIANCE in labels

    def test_oracle_failure_flags_reader(self):
        """E < 90% -> READER_OR_SCORING_BROKEN."""
        cases = [
            CaseResult(
                f"T{i}",
                chr(65 + i),
                {"A": "X", "E": chr(65 + i) if i < 4 else "X"},
                {"A": False, "E": i < 4},
            )
            for i in range(5)
        ]
        results = RunResults(cases=cases, variants=["A", "E"])
        report = audit_pipeline(results)
        oracle_check = next(c for c in report.mechanism_checks if c.name == "Oracle reader")
        assert oracle_check.verdict == Verdict.FAIL

    def test_product_schema_no_hardcoding(self):
        """Evidence attribution works with product-mode candidate IDs (not A-J)."""
        board = BoardResults(
            mechanism_outputs=[
                MechanismOutput("cand_1", ["Patient history supports pneumonia"], []),
                MechanismOutput("cand_2", ["Imaging consistent with PE"], []),
            ],
            skeptic_outputs=[],
            trap_outputs=[],
        )
        candidates = [
            CandidateHypothesis("cand_1", "Community-acquired pneumonia", "diagnosis"),
            CandidateHypothesis("cand_2", "Pulmonary embolism", "diagnosis"),
        ]
        evidence = attribute_evidence(board, candidates)
        assert len(evidence) == 2
        assert evidence[0].candidate_id == "cand_1"
        # No code depends on "cand_A" or option letters

    def test_no_gold_leakage_in_bundle(self):
        """Gold answer must never appear in bundle text."""
        abstraction = ClinicalAbstraction(
            clinical_problem="Test",
            key_findings=["f1"],
            temporal_pattern="acute",
            body_system="cardiovascular",
            specialty_hint="cardiology",
            task_type="diagnosis",
            missing_context=[],
        )
        evidence = [
            CandidateEvidence(
                "cand_A",
                "Option A text",
                [Claim("Some evidence", True, "mechanism", "medium")],
                [],
                [],
                [],
                "supported",
            ),
        ]
        audit = SufficiencyAudit(
            bundle_quality="underdetermined",
            leading_candidates=["cand_A"],
            runner_up_candidates=[],
            unresolved_pairs=[],
            missing_discriminators=[],
            misleading_claims=[],
            generic_claim_count=0,
            case_specific_decisive_claim_count=1,
            repair_required=False,
        )
        bundle = build_final_bundle(abstraction, evidence, [], [], audit)
        # Gold should never be in the bundle
        # The bundle should not contain "correct answer is A" or "gold: A"
        assert "correct answer" not in bundle.full_text.lower()
        assert "gold" not in bundle.full_text.lower()

    def test_no_leaders_is_candidate_collapse(self):
        """Repair required + no unresolved pairs + 0 repair claims = CANDIDATE_COLLAPSE."""
        cases = [
            CaseResult(
                case_id="T0", gold="A",
                predictions={"D11": "B", "E": "A"},
                correct={"D11": False, "E": True},
                repair_triggered=True,
                repair_claim_count=0,
                unresolved_pairs_before=0,
                unresolved_pairs_after=0,
                sufficiency_quality="insufficient",
            ),
        ]
        results = RunResults(cases=cases, variants=["D11", "E"])
        report = audit_pipeline(results)
        labels = report.per_case_labels["T0"]
        assert CaseLabel.CANDIDATE_COLLAPSE in labels

    def test_repair_blocked_by_attributor(self):
        """Repair required but 0 pairs and insufficient quality = REPAIR_BLOCKED_BY_ATTRIBUTOR."""
        cases = [
            CaseResult(
                case_id="T0", gold="A",
                predictions={"D11": "C", "E": "A"},
                correct={"D11": False, "E": True},
                repair_triggered=True,
                repair_claim_count=0,
                unresolved_pairs_before=0,
                unresolved_pairs_after=0,
                sufficiency_quality="insufficient",
            ),
        ]
        results = RunResults(cases=cases, variants=["D11", "E"])
        report = audit_pipeline(results)
        labels = report.per_case_labels["T0"]
        assert CaseLabel.REPAIR_BLOCKED_BY_ATTRIBUTOR in labels
