"""Diagnostic audit harness for MedXpertQA pipeline.

Consumes results from benchmark runs (run_d8_clean.py, run_enrichment_experiment.py)
and produces a brutal honest verdict about whether each architectural mechanism
is actually working.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger(__name__)

REPO = Path(__file__).resolve().parents[2]
REPORTS = REPO / "eval" / "reports" / "medxpertqa"


# ── Enums ───────────────────────────────────────────────────────────────


class Verdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    AMBIGUOUS = "AMBIGUOUS"
    INACTIVE = "INACTIVE"
    REGRESSION = "REGRESSION"


class CaseLabel(str, Enum):
    ENRICHMENT_HELPED = "ENRICHMENT_HELPED"
    ENRICHMENT_HURT = "ENRICHMENT_HURT"
    CONTEXT_LAYER_HELPED = "CONTEXT_LAYER_HELPED"
    CONTEXT_LAYER_HURT = "CONTEXT_LAYER_HURT"
    CONTEXT_LAYER_NEUTRAL = "CONTEXT_LAYER_NEUTRAL"
    REPAIR_HELPED = "REPAIR_HELPED"
    REPAIR_HURT = "REPAIR_HURT"
    REPAIR_INEFFECTIVE = "REPAIR_INEFFECTIVE"
    REPAIR_NOT_TRIGGERED = "REPAIR_NOT_TRIGGERED"
    READER_VARIANCE = "READER_VARIANCE"
    MISSING_OR_BAD_EVIDENCE = "MISSING_OR_BAD_EVIDENCE"
    SUFFICIENCY_OVERCONFIDENT = "SUFFICIENCY_OVERCONFIDENT"
    ALL_CORRECT = "ALL_CORRECT"
    BASELINE_ONLY_CORRECT = "BASELINE_ONLY_CORRECT"
    CANDIDATE_COLLAPSE = "CANDIDATE_COLLAPSE"
    REPAIR_BLOCKED_BY_ATTRIBUTOR = "REPAIR_BLOCKED_BY_ATTRIBUTOR"


# ── Data structures ─────────────────────────────────────────────────────


@dataclass
class CaseResult:
    case_id: str
    gold: str  # gold answer letter
    predictions: dict[str, str]  # variant -> predicted letter
    correct: dict[str, bool]  # variant -> correct?
    # Optional diagnostics
    repair_triggered: bool = False
    repair_claim_count: int = 0
    unresolved_pairs_before: int = 0
    unresolved_pairs_after: int = 0
    sufficiency_quality: str = ""  # "strong", "moderate", "weak", "insufficient"
    bundle_changed_from_d: bool = False  # did D8/D10 bundle differ from D?
    evidence_selected_option: str = ""  # what the evidence judge said
    echo_rate: float = 0.0
    discriminator_count: int = 0
    error: str = ""  # any error during processing


@dataclass
class RunResults:
    cases: list[CaseResult]
    variants: list[str]  # e.g. ["A", "C", "D", "D8", "E"]
    errors: int = 0


@dataclass
class MechanismCheck:
    name: str
    verdict: Verdict
    detail: str


@dataclass
class AuditReport:
    overall_verdict: Verdict
    variant_accuracy: dict[str, tuple[int, int]]  # variant -> (correct, total)
    mechanism_checks: list[MechanismCheck]
    per_case_labels: dict[str, list[CaseLabel]]  # case_id -> labels
    supported: list[str]  # human-readable lines
    not_supported: list[str]
    broken: list[str]
    next_fix: str

    def render(self) -> str:
        """Produce the console report string."""
        lines: list[str] = []
        sep = "═" * 60
        thin = "─" * 58

        lines.append("")
        lines.append(sep)
        lines.append(" MEDXPERTQA PIPELINE AUDIT")
        lines.append(sep)
        lines.append("")
        lines.append(f"OVERALL VERDICT: {self.overall_verdict.value}")

        # Variant accuracy
        lines.append("")
        lines.append(f"── Variant Accuracy {thin[18:]}")

        variant_labels: dict[str, str] = {
            "A": "baseline",
            "C": "enrichment",
            "D": "enrich+ctx",
            "D8": "repair",
            "D10": "strict repair",
            "E": "oracle",
        }
        for variant, (correct, total) in self.variant_accuracy.items():
            label = variant_labels.get(variant, variant)
            pct = correct / total * 100 if total else 0
            lines.append(f"  {variant} ({label}):{' ' * max(1, 16 - len(variant) - len(label))}"
                         f"{correct:>2}/{total:<2} ({pct:>5.1f}%)")

        # Mechanism checks
        lines.append("")
        lines.append(f"── Mechanism Checks {thin[18:]}")
        for check in self.mechanism_checks:
            tag = f"[{check.verdict.value}]"
            lines.append(f"  {tag:<12s} {check.name}: {check.detail}")

        # Supported
        lines.append("")
        lines.append(f"── Supported {thin[10:]}")
        if self.supported:
            for s in self.supported:
                lines.append(f"  • {s}")
        else:
            lines.append("  (none)")

        # Not supported
        lines.append("")
        lines.append(f"── Not Supported {thin[14:]}")
        if self.not_supported:
            for s in self.not_supported:
                lines.append(f"  • {s}")
        else:
            lines.append("  (none)")

        # Broken
        lines.append("")
        lines.append(f"── Broken {thin[7:]}")
        if self.broken:
            for s in self.broken:
                lines.append(f"  • {s}")
        else:
            lines.append("  (none)")

        # Per-case labels
        lines.append("")
        lines.append(f"── Per-Case Labels {thin[16:]}")
        for case_id in sorted(self.per_case_labels.keys()):
            labels = self.per_case_labels[case_id]
            if labels:
                label_str = ", ".join(lb.value for lb in labels)
                lines.append(f"  {case_id}: {label_str}")

        # Next fix
        lines.append("")
        lines.append(f"── Next Fix {thin[9:]}")
        lines.append(f"  {self.next_fix}")

        lines.append("")
        lines.append(sep)
        return "\n".join(lines)


# ── Per-case labeling ───────────────────────────────────────────────────


def _label_case(case: CaseResult, variants: list[str]) -> list[CaseLabel]:
    """Apply ALL matching labels to a single case."""
    labels: list[CaseLabel] = []

    a_correct = case.correct.get("A", False)
    c_correct = case.correct.get("C", False)
    d_correct = case.correct.get("D", False)
    c_pred = case.predictions.get("C", "")
    d_pred = case.predictions.get("D", "")
    e_correct = case.correct.get("E", False)

    # Find which repair variant is present (D8 or D10)
    repair_variant: str | None = None
    for rv in ("D11", "D10", "D8"):
        if rv in variants:
            repair_variant = rv
            break

    repair_correct = case.correct.get(repair_variant, False) if repair_variant else False
    repair_pred = case.predictions.get(repair_variant, "") if repair_variant else ""

    # Enrichment labels (A vs C)
    if "A" in variants and "C" in variants:
        if a_correct and not c_correct:
            labels.append(CaseLabel.ENRICHMENT_HURT)
        if c_correct and not a_correct:
            labels.append(CaseLabel.ENRICHMENT_HELPED)

    # Context layer labels (C vs D)
    if "C" in variants and "D" in variants:
        if c_correct and not d_correct:
            labels.append(CaseLabel.CONTEXT_LAYER_HURT)
        if d_correct and not c_correct:
            labels.append(CaseLabel.CONTEXT_LAYER_HELPED)
        if c_pred == d_pred:
            labels.append(CaseLabel.CONTEXT_LAYER_NEUTRAL)

    # Missing/bad evidence: E correct but all non-oracle wrong
    non_oracle_variants = [v for v in variants if v != "E"]
    all_non_oracle_wrong = all(not case.correct.get(v, False) for v in non_oracle_variants)
    if e_correct and all_non_oracle_wrong and non_oracle_variants:
        labels.append(CaseLabel.MISSING_OR_BAD_EVIDENCE)

    # Repair labels
    if repair_variant and repair_variant in variants and "D" in variants:
        if case.repair_triggered:
            if repair_pred != d_pred:
                if repair_correct and not d_correct:
                    labels.append(CaseLabel.REPAIR_HELPED)
                elif not repair_correct and d_correct:
                    labels.append(CaseLabel.REPAIR_HURT)
            if repair_pred == d_pred:
                labels.append(CaseLabel.REPAIR_INEFFECTIVE)
            # Reader variance: answer differs from D but bundle did NOT change
            if repair_pred != d_pred and not case.bundle_changed_from_d:
                labels.append(CaseLabel.READER_VARIANCE)
        else:
            labels.append(CaseLabel.REPAIR_NOT_TRIGGERED)

    # Sufficiency overconfident
    if case.sufficiency_quality == "strong" and all_non_oracle_wrong and non_oracle_variants:
        labels.append(CaseLabel.SUFFICIENCY_OVERCONFIDENT)

    # All correct
    all_correct = all(case.correct.get(v, False) for v in variants)
    if all_correct:
        labels.append(CaseLabel.ALL_CORRECT)

    # Baseline only correct
    if a_correct and all(
        not case.correct.get(v, False) for v in non_oracle_variants if v != "A"
    ) and len(non_oracle_variants) > 1:
        labels.append(CaseLabel.BASELINE_ONLY_CORRECT)

    # Candidate collapse: repair required but no unresolved pairs (leaders=[])
    if (case.repair_triggered and case.repair_claim_count == 0
            and case.unresolved_pairs_before == 0):
        labels.append(CaseLabel.CANDIDATE_COLLAPSE)

    # Repair blocked by attributor: repair_required but nothing to repair
    if (case.repair_triggered and case.repair_claim_count == 0
            and case.unresolved_pairs_before == 0
            and case.sufficiency_quality in ("insufficient", "")):
        labels.append(CaseLabel.REPAIR_BLOCKED_BY_ATTRIBUTOR)

    return labels


# ── Mechanism checks ────────────────────────────────────────────────────


def _check_enrichment(
    variant_accuracy: dict[str, tuple[int, int]],
) -> MechanismCheck:
    """Compare A vs C accuracy."""
    if "A" not in variant_accuracy or "C" not in variant_accuracy:
        return MechanismCheck(
            name="Enrichment signal",
            verdict=Verdict.INACTIVE,
            detail="Missing A or C variant data",
        )

    a_correct, a_total = variant_accuracy["A"]
    c_correct, c_total = variant_accuracy["C"]
    n = max(a_total, c_total)

    if c_correct > a_correct:
        return MechanismCheck(
            name="Enrichment signal",
            verdict=Verdict.PASS,
            detail=f"C {c_correct}/{n} > A {a_correct}/{n}",
        )
    elif c_correct == a_correct:
        return MechanismCheck(
            name="Enrichment signal",
            verdict=Verdict.AMBIGUOUS,
            detail=f"Enrichment neutral: C {c_correct}/{n} = A {a_correct}/{n}",
        )
    else:
        return MechanismCheck(
            name="Enrichment signal",
            verdict=Verdict.REGRESSION,
            detail=f"Enrichment hurts: C {c_correct}/{n} < A {a_correct}/{n}",
        )


def _check_context_layer(
    variant_accuracy: dict[str, tuple[int, int]],
) -> MechanismCheck:
    """Compare C vs D accuracy."""
    if "C" not in variant_accuracy or "D" not in variant_accuracy:
        return MechanismCheck(
            name="Context layer",
            verdict=Verdict.INACTIVE,
            detail="Missing C or D variant data",
        )

    c_correct, c_total = variant_accuracy["C"]
    d_correct, d_total = variant_accuracy["D"]
    n = max(c_total, d_total)

    if d_correct > c_correct:
        return MechanismCheck(
            name="Context layer",
            verdict=Verdict.PASS,
            detail=f"Context layer helps: D {d_correct}/{n} > C {c_correct}/{n}",
        )
    elif d_correct == c_correct:
        return MechanismCheck(
            name="Context layer",
            verdict=Verdict.AMBIGUOUS,
            detail=f"Context layer neutral: D {d_correct}/{n} = C {c_correct}/{n}",
        )
    else:
        return MechanismCheck(
            name="Context layer",
            verdict=Verdict.REGRESSION,
            detail=f"Context layer hurts: D {d_correct}/{n} < C {c_correct}/{n}",
        )


def _check_repair_activity(cases: list[CaseResult]) -> MechanismCheck:
    """Check whether repair mechanism is active."""
    triggered = [c for c in cases if c.repair_triggered]

    if not triggered:
        return MechanismCheck(
            name="Repair activity",
            verdict=Verdict.INACTIVE,
            detail=f"Repair mechanism inactive: 0 repairs triggered",
        )

    any_bundle_changed = any(c.bundle_changed_from_d for c in triggered)
    if not any_bundle_changed:
        return MechanismCheck(
            name="Repair activity",
            verdict=Verdict.INACTIVE,
            detail=f"Repairs generated but not included in bundle",
        )

    # Check if repair resolved ambiguity for ANY case
    all_unresolved = all(
        c.unresolved_pairs_after >= c.unresolved_pairs_before
        for c in triggered
        if c.unresolved_pairs_before > 0
    )
    cases_with_pairs = [c for c in triggered if c.unresolved_pairs_before > 0]
    if cases_with_pairs and all_unresolved:
        return MechanismCheck(
            name="Repair activity",
            verdict=Verdict.FAIL,
            detail=f"Repair did not resolve ambiguity in {len(cases_with_pairs)} cases",
        )

    return MechanismCheck(
        name="Repair activity",
        verdict=Verdict.PASS,
        detail=f"Repair active: {len(triggered)} triggered, "
               f"{sum(1 for c in triggered if c.bundle_changed_from_d)} bundles changed",
    )


def _check_repair_effectiveness(
    variant_accuracy: dict[str, tuple[int, int]],
    repair_active: bool,
) -> MechanismCheck | None:
    """Compare D vs D8/D10 accuracy. Only runs if repair is active."""
    if not repair_active:
        return None

    repair_variant: str | None = None
    for rv in ("D10", "D8"):
        if rv in variant_accuracy:
            repair_variant = rv
            break

    if not repair_variant or "D" not in variant_accuracy:
        return None

    d_correct, d_total = variant_accuracy["D"]
    r_correct, r_total = variant_accuracy[repair_variant]
    n = max(d_total, r_total)

    if r_correct > d_correct:
        return MechanismCheck(
            name="Repair effectiveness",
            verdict=Verdict.PASS,
            detail=f"Repair helps: {repair_variant} {r_correct}/{n} > D {d_correct}/{n}",
        )
    elif r_correct == d_correct:
        return MechanismCheck(
            name="Repair effectiveness",
            verdict=Verdict.AMBIGUOUS,
            detail=f"Repair did not change outcomes: {repair_variant} {r_correct}/{n} = D {d_correct}/{n}",
        )
    else:
        return MechanismCheck(
            name="Repair effectiveness",
            verdict=Verdict.REGRESSION,
            detail=f"Repair hurt accuracy: {repair_variant} {r_correct}/{n} < D {d_correct}/{n}",
        )


def _check_sufficiency_calibration(
    cases: list[CaseResult],
    variant_accuracy: dict[str, tuple[int, int]],
    variants: list[str],
) -> MechanismCheck:
    """Check whether sufficiency judgments are calibrated."""
    cases_with_quality = [c for c in cases if c.sufficiency_quality]
    if not cases_with_quality:
        return MechanismCheck(
            name="Sufficiency calibration",
            verdict=Verdict.INACTIVE,
            detail="No sufficiency data available",
        )

    strong_cases = [c for c in cases_with_quality if c.sufficiency_quality == "strong"]
    n_strong = len(strong_cases)
    n_total = len(cases_with_quality)

    # Best non-oracle accuracy
    non_oracle = [v for v in variants if v != "E"]
    best_acc = 0.0
    for v in non_oracle:
        if v in variant_accuracy:
            correct, total = variant_accuracy[v]
            if total > 0:
                best_acc = max(best_acc, correct / total)

    # All strong but accuracy < 70%
    if n_strong == n_total and best_acc < 0.70:
        pct = best_acc * 100
        return MechanismCheck(
            name="Sufficiency calibration",
            verdict=Verdict.FAIL,
            detail=f"Sufficiency overconfident: {n_strong}/{n_total} strong but only {pct:.0f}% accuracy",
        )

    # Check correlation between quality and correctness
    strong_correct = sum(
        1 for c in strong_cases
        if any(c.correct.get(v, False) for v in non_oracle)
    )
    weak_cases = [c for c in cases_with_quality if c.sufficiency_quality in ("weak", "insufficient")]
    weak_correct = sum(
        1 for c in weak_cases
        if any(c.correct.get(v, False) for v in non_oracle)
    )

    if strong_cases and weak_cases:
        strong_rate = strong_correct / len(strong_cases)
        weak_rate = weak_correct / len(weak_cases)
        if strong_rate > weak_rate:
            return MechanismCheck(
                name="Sufficiency calibration",
                verdict=Verdict.PASS,
                detail=f"Calibrated: strong={strong_rate:.0%} correct, weak={weak_rate:.0%} correct",
            )
        else:
            return MechanismCheck(
                name="Sufficiency calibration",
                verdict=Verdict.AMBIGUOUS,
                detail=f"Uncalibrated: strong={strong_rate:.0%} correct, weak={weak_rate:.0%} correct",
            )

    return MechanismCheck(
        name="Sufficiency calibration",
        verdict=Verdict.AMBIGUOUS,
        detail=f"Insufficient variation: {n_strong}/{n_total} strong, {len(weak_cases)}/{n_total} weak",
    )


def _check_oracle(
    variant_accuracy: dict[str, tuple[int, int]],
) -> MechanismCheck:
    """Check oracle reader capability."""
    if "E" not in variant_accuracy:
        return MechanismCheck(
            name="Oracle reader",
            verdict=Verdict.INACTIVE,
            detail="No oracle (E) variant data",
        )

    e_correct, e_total = variant_accuracy["E"]
    pct = e_correct / e_total * 100 if e_total else 0

    if pct >= 90:
        return MechanismCheck(
            name="Oracle reader",
            verdict=Verdict.PASS,
            detail=f"Reader capable with correct evidence: E {e_correct}/{e_total}",
        )
    else:
        return MechanismCheck(
            name="Oracle reader",
            verdict=Verdict.FAIL,
            detail=f"Reader or scoring broken: E {e_correct}/{e_total} ({pct:.0f}%)",
        )


def _check_evidence_gap(
    variant_accuracy: dict[str, tuple[int, int]],
    variants: list[str],
) -> MechanismCheck | None:
    """Evidence quality gap between oracle and best non-oracle."""
    if "E" not in variant_accuracy:
        return None

    e_correct, e_total = variant_accuracy["E"]
    e_acc = e_correct / e_total if e_total else 0

    non_oracle = [v for v in variants if v != "E"]
    best_acc = 0.0
    best_variant = ""
    for v in non_oracle:
        if v in variant_accuracy:
            correct, total = variant_accuracy[v]
            if total > 0:
                acc = correct / total
                if acc > best_acc:
                    best_acc = acc
                    best_variant = v

    gap_pp = (e_acc - best_acc) * 100
    if gap_pp > 50:
        return MechanismCheck(
            name="Evidence quality gap",
            verdict=Verdict.FAIL,
            detail=f"Evidence quality remains primary bottleneck: "
                   f"E {e_acc:.0%} vs best non-oracle ({best_variant}) {best_acc:.0%} "
                   f"({gap_pp:.0f}pp gap)",
        )
    elif gap_pp > 20:
        return MechanismCheck(
            name="Evidence quality gap",
            verdict=Verdict.AMBIGUOUS,
            detail=f"Significant evidence gap: "
                   f"E {e_acc:.0%} vs best ({best_variant}) {best_acc:.0%} ({gap_pp:.0f}pp)",
        )
    else:
        return MechanismCheck(
            name="Evidence quality gap",
            verdict=Verdict.PASS,
            detail=f"Evidence gap manageable: "
                   f"E {e_acc:.0%} vs best ({best_variant}) {best_acc:.0%} ({gap_pp:.0f}pp)",
        )


# ── Main audit function ────────────────────────────────────────────────


def audit_pipeline(results: RunResults) -> AuditReport:
    """Run the full audit and return a structured report."""
    variants = results.variants
    n = len(results.cases)

    # Compute variant accuracy
    variant_accuracy: dict[str, tuple[int, int]] = {}
    for v in variants:
        correct = sum(1 for c in results.cases if c.correct.get(v, False))
        variant_accuracy[v] = (correct, n)

    # Run mechanism checks
    checks: list[MechanismCheck] = []

    enrichment_check = _check_enrichment(variant_accuracy)
    checks.append(enrichment_check)

    context_check = _check_context_layer(variant_accuracy)
    checks.append(context_check)

    repair_activity_check = _check_repair_activity(results.cases)
    checks.append(repair_activity_check)

    repair_active = repair_activity_check.verdict == Verdict.PASS
    repair_eff_check = _check_repair_effectiveness(
        variant_accuracy, repair_active,
    )
    if repair_eff_check:
        checks.append(repair_eff_check)

    sufficiency_check = _check_sufficiency_calibration(
        results.cases, variant_accuracy, variants,
    )
    checks.append(sufficiency_check)

    oracle_check = _check_oracle(variant_accuracy)
    checks.append(oracle_check)

    evidence_gap_check = _check_evidence_gap(variant_accuracy, variants)
    if evidence_gap_check:
        checks.append(evidence_gap_check)

    # Per-case labels
    per_case_labels: dict[str, list[CaseLabel]] = {}
    for case in results.cases:
        per_case_labels[case.case_id] = _label_case(case, variants)

    # Classify findings
    supported: list[str] = []
    not_supported: list[str] = []
    broken: list[str] = []

    for check in checks:
        if check.verdict == Verdict.PASS:
            supported.append(f"{check.name}: {check.detail}")
        elif check.verdict == Verdict.AMBIGUOUS:
            not_supported.append(f"{check.name}: {check.detail}")
        elif check.verdict in (Verdict.FAIL, Verdict.REGRESSION, Verdict.INACTIVE):
            broken.append(f"{check.name}: {check.detail}")

    # Overall verdict
    verdicts = [c.verdict for c in checks]
    if any(v in (Verdict.FAIL, Verdict.REGRESSION) for v in verdicts):
        overall = Verdict.FAIL
    elif any(v == Verdict.INACTIVE for v in verdicts):
        overall = Verdict.FAIL
    elif all(v == Verdict.PASS for v in verdicts):
        overall = Verdict.PASS
    else:
        overall = Verdict.AMBIGUOUS

    # Next fix recommendation
    next_fix = _recommend_next_fix(checks)

    return AuditReport(
        overall_verdict=overall,
        variant_accuracy=variant_accuracy,
        mechanism_checks=checks,
        per_case_labels=per_case_labels,
        supported=supported,
        not_supported=not_supported,
        broken=broken,
        next_fix=next_fix,
    )


def _recommend_next_fix(
    checks: list[MechanismCheck],
) -> str:
    """Pick the single most actionable next fix based on worst finding."""
    check_map = {c.name: c for c in checks}

    # Priority order: most fundamental problems first
    repair_activity = check_map.get("Repair activity")
    if repair_activity and repair_activity.verdict == Verdict.INACTIVE:
        return "Tighten sufficiency criteria so repair actually triggers."

    sufficiency = check_map.get("Sufficiency calibration")
    if sufficiency and sufficiency.verdict == Verdict.FAIL:
        return "Sufficiency judge marks everything strong -- needs stricter criteria."

    context = check_map.get("Context layer")
    if context and context.verdict == Verdict.REGRESSION:
        return "Context layer adding noise -- investigate claim structuring."

    enrichment = check_map.get("Enrichment signal")
    if enrichment and enrichment.verdict == Verdict.REGRESSION:
        return "Enrichment hurting -- check for option leakage or misleading claims."

    evidence_gap = check_map.get("Evidence quality gap")
    if evidence_gap and evidence_gap.verdict == Verdict.FAIL:
        return "Evidence quality is the bottleneck -- improve enricher or add retrieval."

    oracle = check_map.get("Oracle reader")
    if oracle and oracle.verdict == Verdict.FAIL:
        return "Reader cannot solve even with oracle evidence -- check prompting."

    repair_eff = check_map.get("Repair effectiveness")
    if repair_eff and repair_eff.verdict == Verdict.REGRESSION:
        return "Repair hurting accuracy -- repair claims may be misleading."

    # No critical issues
    if all(c.verdict == Verdict.PASS for c in checks):
        return "All mechanisms passing. Increase sample size to confirm."

    return "Review AMBIGUOUS mechanisms and increase sample size."


# ── File loading ────────────────────────────────────────────────────────


def load_results_from_files(
    results_json: str | None = None,
    diagnostics_jsonl: str | None = None,
) -> RunResults:
    """Load RunResults from the output files of run_d8_clean.py or run_enrichment_experiment.py.

    Supports two formats:
    1. results.json from run_d8_clean.py: {variant: {case_id: {majority, correct, gold, ...}}}
    2. variant_*.jsonl from run_enrichment_experiment.py: one JSONL per variant

    If results_json is provided, loads that file directly.
    If diagnostics_jsonl is provided, loads all variant JSONL files from the same directory.
    If neither, searches for the latest results dir.
    """
    if results_json:
        rpath = Path(results_json)
        if not rpath.exists():
            raise FileNotFoundError(f"Results file not found: {rpath}")
        return _load_d8clean_results(rpath)

    if diagnostics_jsonl:
        dpath = Path(diagnostics_jsonl)
        results_dir = dpath.parent
    else:
        # Find latest results directory
        results_dir = _find_latest_results_dir()
        if not results_dir:
            raise FileNotFoundError(
                f"No results found in {REPORTS}. Run an experiment first."
            )

    # Check for results.json (d8clean format)
    rjson = results_dir / "results.json"
    if rjson.exists():
        return _load_d8clean_results(rjson)

    # Fall back to variant JSONL files (enrichment experiment format)
    return _load_enrichment_results(results_dir)


def _load_d8clean_results(rpath: Path) -> RunResults:
    """Load from run_d8_clean.py results.json format.

    Structure: {variant: {case_id: {answers, majority, correct, gold, variance, ...}}}
    Also loads cache metadata (d8_meta, d10_meta, sufficiency) if available.
    """
    data = json.loads(rpath.read_text(encoding="utf-8"))
    cache_dir = rpath.parent / "cache"

    variants_found = list(data.keys())
    cases_map: dict[str, CaseResult] = {}
    errors = 0

    # Collect all case IDs across all variants
    all_case_ids: set[str] = set()
    for variant_data in data.values():
        all_case_ids.update(variant_data.keys())

    for case_id in sorted(all_case_ids):
        predictions: dict[str, str] = {}
        correct: dict[str, bool] = {}
        gold = ""

        for vid in variants_found:
            vdata = data.get(vid, {}).get(case_id)
            if not vdata:
                continue
            predictions[vid] = vdata.get("majority", "")
            correct[vid] = vdata.get("correct", False)
            if not gold:
                gold = vdata.get("gold", "")
            if vdata.get("error"):
                errors += 1

        # Load cache metadata if available
        repair_triggered = False
        repair_claim_count = 0
        unresolved_before = 0
        unresolved_after = 0
        sufficiency_quality = ""
        bundle_changed = False
        evidence_selected_option = ""

        if cache_dir.exists():
            # D8 meta
            d8_meta_path = cache_dir / f"{case_id}__d8_meta.json"
            if d8_meta_path.exists():
                d8_meta = json.loads(d8_meta_path.read_text(encoding="utf-8"))
                rc = d8_meta.get("repair_claims", 0)
                if rc > 0:
                    repair_triggered = True
                    repair_claim_count = rc
                sufficiency_quality = d8_meta.get("sufficiency", "")

            # D10 meta
            d10_meta_path = cache_dir / f"{case_id}__d10_meta.json"
            if d10_meta_path.exists():
                d10_meta = json.loads(d10_meta_path.read_text(encoding="utf-8"))
                rc10 = d10_meta.get("repair_claims", 0)
                if rc10 > 0:
                    repair_triggered = True
                    repair_claim_count = max(repair_claim_count, rc10)
                evidence_selected_option = d10_meta.get("evidence_selected_option", "") or ""
                # Unresolved pairs
                unresolved = d10_meta.get("unresolved_pairs", [])
                unresolved_before = len(unresolved)
                if not d10_meta.get("strictly_sufficient", True):
                    unresolved_after = len(unresolved)
                # Override sufficiency from judgment quality if available
                jq = d10_meta.get("judgment_quality", "")
                if jq:
                    sufficiency_quality = jq

            # Sufficiency from cache
            suff_path = cache_dir / f"{case_id}__sufficiency.json"
            if suff_path.exists() and not sufficiency_quality:
                suff_data = json.loads(suff_path.read_text(encoding="utf-8"))
                sufficiency_quality = suff_data.get("quality", "")

            # Check if D8/D10 bundle differs from D
            d_pred = predictions.get("D", "")
            for rv in ("D8", "D10"):
                rv_pred = predictions.get(rv, "")
                if rv_pred and rv_pred != d_pred:
                    bundle_changed = True

        cases_map[case_id] = CaseResult(
            case_id=case_id,
            gold=gold,
            predictions=predictions,
            correct=correct,
            repair_triggered=repair_triggered,
            repair_claim_count=repair_claim_count,
            unresolved_pairs_before=unresolved_before,
            unresolved_pairs_after=unresolved_after,
            sufficiency_quality=sufficiency_quality,
            bundle_changed_from_d=bundle_changed,
            evidence_selected_option=evidence_selected_option,
        )

    return RunResults(
        cases=list(cases_map.values()),
        variants=variants_found,
        errors=errors,
    )


def _load_enrichment_results(results_dir: Path) -> RunResults:
    """Load from run_enrichment_experiment.py variant JSONL format.

    Each variant_X.jsonl has lines like:
    {"case_id": "Text-0", "predicted": "I", "gold": "E", "correct": false, "variant": "A", ...}
    """
    variant_files = sorted(results_dir.glob("variant_*.jsonl"))
    if not variant_files:
        raise FileNotFoundError(f"No variant_*.jsonl files in {results_dir}")

    cases_map: dict[str, CaseResult] = {}
    variants_found: list[str] = []
    errors = 0

    # Also check for diagnostic files for extra metadata
    diag_data: dict[str, dict[str, Any]] = {}  # variant -> case_id -> row
    for dpath in results_dir.glob("diagnostic_*.jsonl"):
        vid = dpath.stem.replace("diagnostic_", "")
        diag_data[vid] = {}
        for line in dpath.read_text(encoding="utf-8").strip().split("\n"):
            if not line.strip():
                continue
            row = json.loads(line)
            diag_data[vid][row.get("case_id", "")] = row

    for vfile in variant_files:
        vid = vfile.stem.replace("variant_", "")
        variants_found.append(vid)

        for line in vfile.read_text(encoding="utf-8").strip().split("\n"):
            if not line.strip():
                continue
            row = json.loads(line)
            case_id = row["case_id"]
            predicted = row.get("predicted", "")
            gold = row.get("gold", "")
            is_correct = row.get("correct", False)

            if case_id not in cases_map:
                cases_map[case_id] = CaseResult(
                    case_id=case_id,
                    gold=gold,
                    predictions={},
                    correct={},
                )

            cases_map[case_id].predictions[vid] = predicted
            cases_map[case_id].correct[vid] = is_correct
            if row.get("error"):
                errors += 1

            # Pull diagnostic metadata for D8/D10
            diag_row = diag_data.get(vid, {}).get(case_id, {})
            if vid in ("D8", "D10"):
                rc = row.get("repair_claims", 0)
                if rc > 0:
                    cases_map[case_id].repair_triggered = True
                    cases_map[case_id].repair_claim_count = rc
                suff = row.get("sufficiency", "")
                if suff:
                    cases_map[case_id].sufficiency_quality = suff
                unresolved = row.get("unresolved_pairs", [])
                if unresolved:
                    cases_map[case_id].unresolved_pairs_before = len(unresolved)

            # Echo rate and discriminator count from diagnostics
            if diag_row:
                cases_map[case_id].echo_rate = max(
                    cases_map[case_id].echo_rate,
                    diag_row.get("echo_rate", 0.0),
                )
                cases_map[case_id].discriminator_count = max(
                    cases_map[case_id].discriminator_count,
                    diag_row.get("direct_discriminator_count", 0),
                )

    # Determine bundle_changed_from_d
    for case in cases_map.values():
        d_pred = case.predictions.get("D", "")
        for rv in ("D8", "D10"):
            if rv in case.predictions and case.predictions[rv] != d_pred:
                case.bundle_changed_from_d = True

    return RunResults(
        cases=list(cases_map.values()),
        variants=variants_found,
        errors=errors,
    )


def _find_latest_results_dir() -> Path | None:
    """Find the most recent results directory under REPORTS."""
    if not REPORTS.exists():
        return None

    # Prefer d8clean runs (have richer metadata), then enrichment runs
    candidates: list[Path] = []
    for d in sorted(REPORTS.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        if (d / "results.json").exists():
            candidates.append(d)
        elif list(d.glob("variant_*.jsonl")):
            candidates.append(d)

    return candidates[0] if candidates else None


# ── CLI entry point ─────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Audit MedXpertQA pipeline results for mechanism effectiveness",
    )
    parser.add_argument(
        "--results", type=str, default=None,
        help="Path to results.json (d8clean format)",
    )
    parser.add_argument(
        "--diagnostics", type=str, default=None,
        help="Path to any diagnostic_*.jsonl file (enrichment format); "
             "loads all variant files from the same directory",
    )
    parser.add_argument(
        "--dir", type=str, default=None,
        help="Path to results directory (auto-detects format)",
    )
    args = parser.parse_args(argv)

    try:
        if args.dir:
            d = Path(args.dir)
            rjson = d / "results.json"
            if rjson.exists():
                results = load_results_from_files(results_json=str(rjson))
            else:
                results = load_results_from_files(diagnostics_jsonl=str(next(d.glob("variant_*.jsonl"))))
        elif args.results:
            results = load_results_from_files(results_json=args.results)
        elif args.diagnostics:
            results = load_results_from_files(diagnostics_jsonl=args.diagnostics)
        else:
            results = load_results_from_files()
    except FileNotFoundError as e:
        log.error("load_failed", error=str(e))
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    report = audit_pipeline(results)
    rendered = report.render()
    # Handle Windows console encoding by replacing box-drawing chars if needed
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    print(rendered)
    return 0


if __name__ == "__main__":
    sys.exit(main())
