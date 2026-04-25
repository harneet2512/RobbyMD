"""Behavioral tests for the audit pipeline.

Each test creates a deterministic fixture that represents a specific
architectural failure mode. The tests assert that the audit harness
correctly identifies the failure — not that the code merely runs.

If a mechanism is inactive, the report must not say PASS.
If a result is ambiguous, the report must say AMBIGUOUS.
If evidence abundance hides wrong evidence, the report must say SUFFICIENCY_OVERCONFIDENT.
"""
from __future__ import annotations

import pytest

from eval.medxpertqa.audit_pipeline import (
    AuditReport,
    CaseLabel,
    CaseResult,
    MechanismCheck,
    RunResults,
    Verdict,
    audit_pipeline,
)


# ── Fixture helpers ───────────────────────────────────────────────────


def _case(
    case_id: str,
    gold: str,
    predictions: dict[str, str],
    *,
    repair_triggered: bool = False,
    repair_claim_count: int = 0,
    unresolved_pairs_before: int = 0,
    unresolved_pairs_after: int = 0,
    sufficiency_quality: str = "",
    bundle_changed_from_d: bool = False,
    evidence_selected_option: str = "",
) -> CaseResult:
    correct = {v: (p == gold) for v, p in predictions.items()}
    return CaseResult(
        case_id=case_id,
        gold=gold,
        predictions=predictions,
        correct=correct,
        repair_triggered=repair_triggered,
        repair_claim_count=repair_claim_count,
        unresolved_pairs_before=unresolved_pairs_before,
        unresolved_pairs_after=unresolved_pairs_after,
        sufficiency_quality=sufficiency_quality,
        bundle_changed_from_d=bundle_changed_from_d,
        evidence_selected_option=evidence_selected_option,
    )


def _run(cases: list[CaseResult], variants: list[str] | None = None) -> RunResults:
    if variants is None:
        all_v: set[str] = set()
        for c in cases:
            all_v.update(c.predictions.keys())
        variants = sorted(all_v)
    return RunResults(cases=cases, variants=variants)


def _find_check(report: AuditReport, name: str) -> MechanismCheck | None:
    for c in report.mechanism_checks:
        if c.name == name:
            return c
    return None


# ── Scenario: D8 inactive ────────────────────────────────────────────


class TestD8Inactive:
    """D8 repair_count=0, all sufficiency=strong, D8 accuracy = D accuracy.
    Must report repair INACTIVE, not PASS."""

    @pytest.fixture()
    def report(self) -> AuditReport:
        cases = [
            _case("T0", "E", {"A": "C", "C": "H", "D": "D", "D8": "D", "E": "E"},
                   sufficiency_quality="strong"),
            _case("T1", "E", {"A": "A", "C": "E", "D": "E", "D8": "E", "E": "E"},
                   sufficiency_quality="strong"),
            _case("T2", "C", {"A": "I", "C": "E", "D": "E", "D8": "E", "E": "C"},
                   sufficiency_quality="strong"),
            _case("T3", "I", {"A": "I", "C": "I", "D": "C", "D8": "C", "E": "I"},
                   sufficiency_quality="strong"),
        ]
        return audit_pipeline(_run(cases))

    def test_repair_verdict_is_inactive(self, report: AuditReport):
        check = _find_check(report, "Repair activity")
        assert check is not None
        assert check.verdict == Verdict.INACTIVE

    def test_overall_is_fail(self, report: AuditReport):
        assert report.overall_verdict == Verdict.FAIL

    def test_repair_not_triggered_labels(self, report: AuditReport):
        for case_id, labels in report.per_case_labels.items():
            assert CaseLabel.REPAIR_NOT_TRIGGERED in labels, (
                f"{case_id}: expected REPAIR_NOT_TRIGGERED"
            )

    def test_broken_mentions_repair(self, report: AuditReport):
        text = "\n".join(report.broken)
        assert "inactive" in text.lower() or "Repair" in text


# ── Scenario: Sufficiency overconfident ───────────────────────────────


class TestSufficiencyOverconfident:
    """All bundles marked 'strong' but accuracy is 20%.
    Must report sufficiency overconfident, not calibrated."""

    @pytest.fixture()
    def report(self) -> AuditReport:
        cases = [
            _case("T0", "E", {"A": "C", "D": "D", "E": "E"},
                   sufficiency_quality="strong"),
            _case("T1", "E", {"A": "A", "D": "E", "E": "E"},
                   sufficiency_quality="strong"),
            _case("T2", "C", {"A": "I", "D": "E", "E": "C"},
                   sufficiency_quality="strong"),
            _case("T3", "I", {"A": "I", "D": "C", "E": "I"},
                   sufficiency_quality="strong"),
            _case("T4", "J", {"A": "B", "D": "C", "E": "J"},
                   sufficiency_quality="strong"),
        ]
        return audit_pipeline(_run(cases))

    def test_sufficiency_verdict_is_fail(self, report: AuditReport):
        check = _find_check(report, "Sufficiency calibration")
        assert check is not None
        assert check.verdict == Verdict.FAIL

    def test_detail_says_overconfident(self, report: AuditReport):
        check = _find_check(report, "Sufficiency calibration")
        assert check is not None
        assert "overconfident" in check.detail.lower()

    def test_wrong_cases_labeled_overconfident(self, report: AuditReport):
        for case_id in ("T0", "T2", "T4"):
            labels = report.per_case_labels[case_id]
            assert CaseLabel.SUFFICIENCY_OVERCONFIDENT in labels, (
                f"{case_id}: expected SUFFICIENCY_OVERCONFIDENT"
            )


# ── Scenario: Context layer hurts ────────────────────────────────────


class TestContextLayerHurt:
    """D < C on accuracy. Must report context layer REGRESSION."""

    @pytest.fixture()
    def report(self) -> AuditReport:
        cases = [
            _case("T0", "A", {"A": "B", "C": "A", "D": "B", "E": "A"}),
            _case("T1", "B", {"A": "A", "C": "B", "D": "A", "E": "B"}),
            _case("T2", "C", {"A": "A", "C": "C", "D": "B", "E": "C"}),
            _case("T3", "D", {"A": "A", "C": "A", "D": "A", "E": "D"}),
        ]
        return audit_pipeline(_run(cases))

    def test_context_layer_verdict_is_regression(self, report: AuditReport):
        check = _find_check(report, "Context layer")
        assert check is not None
        assert check.verdict == Verdict.REGRESSION

    def test_per_case_context_layer_hurt_labels(self, report: AuditReport):
        for case_id in ("T0", "T1", "T2"):
            labels = report.per_case_labels[case_id]
            assert CaseLabel.CONTEXT_LAYER_HURT in labels, (
                f"{case_id}: C correct but D wrong → expected CONTEXT_LAYER_HURT"
            )

    def test_overall_is_fail(self, report: AuditReport):
        assert report.overall_verdict == Verdict.FAIL


# ── Scenario: Enrichment hurts ────────────────────────────────────────


class TestEnrichmentHurt:
    """C < A on accuracy. Must report enrichment REGRESSION."""

    @pytest.fixture()
    def report(self) -> AuditReport:
        cases = [
            _case("T0", "A", {"A": "A", "C": "B", "D": "B", "D8": "A", "E": "A"},
                   repair_triggered=True, repair_claim_count=3,
                   bundle_changed_from_d=True,
                   unresolved_pairs_before=2, unresolved_pairs_after=0,
                   sufficiency_quality="strong"),
            _case("T1", "B", {"A": "B", "C": "A", "D": "A", "D8": "B", "E": "B"},
                   repair_triggered=True, repair_claim_count=3,
                   bundle_changed_from_d=True,
                   unresolved_pairs_before=2, unresolved_pairs_after=0,
                   sufficiency_quality="strong"),
            _case("T2", "C", {"A": "C", "C": "A", "D": "A", "D8": "C", "E": "C"},
                   repair_triggered=True, repair_claim_count=3,
                   bundle_changed_from_d=True,
                   unresolved_pairs_before=2, unresolved_pairs_after=0,
                   sufficiency_quality="strong"),
            _case("T3", "D", {"A": "A", "C": "A", "D": "A", "D8": "A", "E": "D"},
                   repair_triggered=True, repair_claim_count=3,
                   bundle_changed_from_d=True,
                   unresolved_pairs_before=2, unresolved_pairs_after=0,
                   sufficiency_quality="weak"),
        ]
        return audit_pipeline(_run(cases))

    def test_enrichment_verdict_is_regression(self, report: AuditReport):
        check = _find_check(report, "Enrichment signal")
        assert check is not None
        assert check.verdict == Verdict.REGRESSION

    def test_per_case_enrichment_hurt_labels(self, report: AuditReport):
        for case_id in ("T0", "T1", "T2"):
            labels = report.per_case_labels[case_id]
            assert CaseLabel.ENRICHMENT_HURT in labels, (
                f"{case_id}: A correct but C wrong → expected ENRICHMENT_HURT"
            )

    def test_next_fix_mentions_enrichment(self, report: AuditReport):
        fix = report.next_fix.lower()
        assert "enrichment" in fix or "enrich" in fix


# ── Scenario: Reader variance ─────────────────────────────────────────


class TestReaderVariance:
    """D8 answer differs from D but bundle did NOT change.
    Must label READER_VARIANCE, not REPAIR_HELPED/HURT."""

    @pytest.fixture()
    def report(self) -> AuditReport:
        cases = [
            _case("T0", "A", {"A": "B", "C": "A", "D": "A", "D8": "C", "E": "A"},
                   repair_triggered=True, repair_claim_count=5,
                   bundle_changed_from_d=False,
                   unresolved_pairs_before=3, unresolved_pairs_after=3),
            _case("T1", "B", {"A": "A", "C": "B", "D": "B", "D8": "A", "E": "B"},
                   repair_triggered=True, repair_claim_count=5,
                   bundle_changed_from_d=False,
                   unresolved_pairs_before=2, unresolved_pairs_after=2),
        ]
        return audit_pipeline(_run(cases))

    def test_reader_variance_labeled(self, report: AuditReport):
        for case_id in ("T0", "T1"):
            labels = report.per_case_labels[case_id]
            assert CaseLabel.READER_VARIANCE in labels, (
                f"{case_id}: D8 differs from D, bundle unchanged → READER_VARIANCE"
            )


# ── Scenario: Successful repair ──────────────────────────────────────


class TestSuccessfulRepair:
    """D8 > D, repairs triggered, bundles changed, ambiguity resolved.
    Must report repair PASS."""

    @pytest.fixture()
    def report(self) -> AuditReport:
        cases = [
            _case("T0", "A", {"A": "B", "C": "B", "D": "B", "D8": "A", "E": "A"},
                   repair_triggered=True, repair_claim_count=8,
                   bundle_changed_from_d=True,
                   unresolved_pairs_before=4, unresolved_pairs_after=1,
                   sufficiency_quality="weak"),
            _case("T1", "B", {"A": "A", "C": "A", "D": "A", "D8": "B", "E": "B"},
                   repair_triggered=True, repair_claim_count=6,
                   bundle_changed_from_d=True,
                   unresolved_pairs_before=3, unresolved_pairs_after=0,
                   sufficiency_quality="weak"),
            _case("T2", "C", {"A": "A", "C": "C", "D": "C", "D8": "C", "E": "C"},
                   repair_triggered=True, repair_claim_count=4,
                   bundle_changed_from_d=True,
                   unresolved_pairs_before=2, unresolved_pairs_after=0,
                   sufficiency_quality="strong"),
        ]
        return audit_pipeline(_run(cases))

    def test_repair_activity_is_pass(self, report: AuditReport):
        check = _find_check(report, "Repair activity")
        assert check is not None
        assert check.verdict == Verdict.PASS

    def test_repair_effectiveness_is_pass(self, report: AuditReport):
        check = _find_check(report, "Repair effectiveness")
        assert check is not None
        assert check.verdict == Verdict.PASS

    def test_per_case_repair_helped(self, report: AuditReport):
        for case_id in ("T0", "T1"):
            labels = report.per_case_labels[case_id]
            assert CaseLabel.REPAIR_HELPED in labels, (
                f"{case_id}: D wrong, D8 correct → expected REPAIR_HELPED"
            )

    def test_overall_not_fail(self, report: AuditReport):
        assert report.overall_verdict != Verdict.FAIL


# ── Scenario: Repair dropped ─────────────────────────────────────────


class TestRepairDropped:
    """Repairs triggered, claims generated, but bundle NOT changed.
    Must report repair INACTIVE (repairs not included)."""

    @pytest.fixture()
    def report(self) -> AuditReport:
        cases = [
            _case("T0", "A", {"A": "B", "D": "B", "D8": "B", "E": "A"},
                   repair_triggered=True, repair_claim_count=10,
                   bundle_changed_from_d=False,
                   unresolved_pairs_before=5, unresolved_pairs_after=5),
            _case("T1", "B", {"A": "A", "D": "A", "D8": "A", "E": "B"},
                   repair_triggered=True, repair_claim_count=8,
                   bundle_changed_from_d=False,
                   unresolved_pairs_before=3, unresolved_pairs_after=3),
        ]
        return audit_pipeline(_run(cases))

    def test_repair_verdict_is_inactive(self, report: AuditReport):
        check = _find_check(report, "Repair activity")
        assert check is not None
        assert check.verdict == Verdict.INACTIVE

    def test_detail_mentions_not_included(self, report: AuditReport):
        check = _find_check(report, "Repair activity")
        assert check is not None
        assert "not included" in check.detail.lower()


# ── Scenario: Evidence gap is the bottleneck ──────────────────────────


class TestEvidenceGapBottleneck:
    """E = 100%, best non-oracle = 20%. Must flag evidence gap as FAIL."""

    @pytest.fixture()
    def report(self) -> AuditReport:
        cases = [
            _case("T0", "A", {"A": "B", "C": "B", "D": "B", "E": "A"}),
            _case("T1", "B", {"A": "A", "C": "B", "D": "A", "E": "B"}),
            _case("T2", "C", {"A": "A", "C": "A", "D": "A", "E": "C"}),
            _case("T3", "D", {"A": "A", "C": "A", "D": "A", "E": "D"}),
            _case("T4", "E", {"A": "A", "C": "A", "D": "A", "E": "E"}),
        ]
        return audit_pipeline(_run(cases))

    def test_evidence_gap_fail(self, report: AuditReport):
        check = _find_check(report, "Evidence quality gap")
        assert check is not None
        assert check.verdict == Verdict.FAIL

    def test_detail_mentions_bottleneck(self, report: AuditReport):
        check = _find_check(report, "Evidence quality gap")
        assert check is not None
        assert "bottleneck" in check.detail.lower()


# ── Scenario: Oracle broken ───────────────────────────────────────────


class TestOracleBroken:
    """E < 90%. Must report oracle reader FAIL."""

    @pytest.fixture()
    def report(self) -> AuditReport:
        cases = [
            _case("T0", "A", {"A": "B", "E": "A"}),
            _case("T1", "B", {"A": "A", "E": "B"}),
            _case("T2", "C", {"A": "A", "E": "A"}),
            _case("T3", "D", {"A": "A", "E": "A"}),
            _case("T4", "E", {"A": "A", "E": "E"}),
        ]
        return audit_pipeline(_run(cases))

    def test_oracle_verdict_is_fail(self, report: AuditReport):
        check = _find_check(report, "Oracle reader")
        assert check is not None
        assert check.verdict == Verdict.FAIL

    def test_detail_mentions_broken(self, report: AuditReport):
        check = _find_check(report, "Oracle reader")
        assert check is not None
        assert "broken" in check.detail.lower()


# ── Scenario: All mechanisms passing ──────────────────────────────────


class TestAllPassing:
    """C > A, D > C, repair active and helps, E >= 90%, gap < 20pp.
    Overall must be PASS."""

    @pytest.fixture()
    def report(self) -> AuditReport:
        # A=1/7, C=3/7, D=5/7, D8=6/7, E=7/7
        # C>A, D>C, D8>D, E>=90%, gap<20pp, sufficiency calibrated
        cases = [
            _case("T0", "A", {"A": "B", "C": "B", "D": "A", "D8": "A", "E": "A"},
                   repair_triggered=True, repair_claim_count=5,
                   bundle_changed_from_d=True,
                   unresolved_pairs_before=2, unresolved_pairs_after=0,
                   sufficiency_quality="strong"),
            _case("T1", "B", {"A": "A", "C": "B", "D": "B", "D8": "B", "E": "B"},
                   repair_triggered=True, repair_claim_count=4,
                   bundle_changed_from_d=True,
                   unresolved_pairs_before=1, unresolved_pairs_after=0,
                   sufficiency_quality="strong"),
            _case("T2", "C", {"A": "A", "C": "A", "D": "C", "D8": "C", "E": "C"},
                   repair_triggered=True, repair_claim_count=3,
                   bundle_changed_from_d=True,
                   unresolved_pairs_before=2, unresolved_pairs_after=0,
                   sufficiency_quality="strong"),
            _case("T3", "D", {"A": "A", "C": "D", "D": "D", "D8": "D", "E": "D"},
                   repair_triggered=True, repair_claim_count=6,
                   bundle_changed_from_d=True,
                   unresolved_pairs_before=3, unresolved_pairs_after=1,
                   sufficiency_quality="strong"),
            _case("T4", "E", {"A": "E", "C": "E", "D": "E", "D8": "E", "E": "E"},
                   repair_triggered=True, repair_claim_count=2,
                   bundle_changed_from_d=True,
                   unresolved_pairs_before=1, unresolved_pairs_after=0,
                   sufficiency_quality="strong"),
            _case("T5", "F", {"A": "A", "C": "A", "D": "A", "D8": "A", "E": "F"},
                   repair_triggered=True, repair_claim_count=2,
                   bundle_changed_from_d=True,
                   unresolved_pairs_before=1, unresolved_pairs_after=0,
                   sufficiency_quality="weak"),
            _case("T6", "G", {"A": "A", "C": "A", "D": "A", "D8": "G", "E": "G"},
                   repair_triggered=True, repair_claim_count=4,
                   bundle_changed_from_d=True,
                   unresolved_pairs_before=2, unresolved_pairs_after=0,
                   sufficiency_quality="strong"),
        ]
        return audit_pipeline(_run(cases))

    def test_overall_is_pass(self, report: AuditReport):
        assert report.overall_verdict == Verdict.PASS

    def test_no_broken_items(self, report: AuditReport):
        assert len(report.broken) == 0

    def test_enrichment_passes(self, report: AuditReport):
        check = _find_check(report, "Enrichment signal")
        assert check is not None
        assert check.verdict == Verdict.PASS

    def test_context_layer_passes(self, report: AuditReport):
        check = _find_check(report, "Context layer")
        assert check is not None
        assert check.verdict == Verdict.PASS


# ── Scenario: Missing or bad evidence per-case ────────────────────────


class TestMissingOrBadEvidence:
    """E correct but ALL non-oracle wrong. Must label MISSING_OR_BAD_EVIDENCE."""

    @pytest.fixture()
    def report(self) -> AuditReport:
        cases = [
            _case("T0", "A", {"A": "B", "C": "C", "D": "D", "E": "A"}),
        ]
        return audit_pipeline(_run(cases))

    def test_labeled_missing_or_bad(self, report: AuditReport):
        labels = report.per_case_labels["T0"]
        assert CaseLabel.MISSING_OR_BAD_EVIDENCE in labels


# ── Scenario: Baseline only correct ───────────────────────────────────


class TestBaselineOnlyCorrect:
    """A correct but C, D wrong. Must label BASELINE_ONLY_CORRECT."""

    @pytest.fixture()
    def report(self) -> AuditReport:
        cases = [
            _case("T0", "A", {"A": "A", "C": "B", "D": "C", "E": "A"}),
        ]
        return audit_pipeline(_run(cases))

    def test_labeled_baseline_only(self, report: AuditReport):
        labels = report.per_case_labels["T0"]
        assert CaseLabel.BASELINE_ONLY_CORRECT in labels

    def test_also_labeled_enrichment_hurt(self, report: AuditReport):
        labels = report.per_case_labels["T0"]
        assert CaseLabel.ENRICHMENT_HURT in labels


# ── Render smoke test ─────────────────────────────────────────────────


class TestRender:
    """Verify render() produces structured output with all sections."""

    def test_render_contains_all_sections(self):
        cases = [
            _case("T0", "A", {"A": "A", "C": "A", "D": "A", "E": "A"}),
        ]
        report = audit_pipeline(_run(cases))
        rendered = report.render()
        assert "MEDXPERTQA PIPELINE AUDIT" in rendered
        assert "OVERALL VERDICT" in rendered
        assert "Variant Accuracy" in rendered
        assert "Mechanism Checks" in rendered
        assert "Supported" in rendered
        assert "Not Supported" in rendered
        assert "Broken" in rendered
        assert "Per-Case Labels" in rendered
        assert "Next Fix" in rendered


# ── File loader tests ─────────────────────────────────────────────────


class TestLoadFromD8CleanResults:
    """Verify load_results_from_files can parse real d8clean results.json."""

    def test_load_latest_results(self):
        from eval.medxpertqa.audit_pipeline import load_results_from_files

        results_path = (
            "eval/reports/medxpertqa/d8clean_gemini-2_5-flash_20260424T202528Z/results.json"
        )
        try:
            results = load_results_from_files(results_json=results_path)
        except FileNotFoundError:
            pytest.skip("Results file not available")

        assert len(results.cases) == 10
        assert "A" in results.variants
        assert "E" in results.variants

        report = audit_pipeline(results)
        assert report.overall_verdict in (Verdict.FAIL, Verdict.AMBIGUOUS, Verdict.PASS)
        assert report.variant_accuracy["E"][0] == 9
