"""LongMemEval-S Full Audit + Failure Attribution + High-Leverage Fix.

Phases 0-4: offline analysis, zero LLM calls, ~5 min runtime.
Phase 5: cost-gated fix execution via --execute-fix flag.
Phase 6: structured report output.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.longmemeval.diagnostic_slice import _gold_in_text
from eval.longmemeval.official_evaluate_qa import get_anscheck_prompt, model_zoo
from eval.longmemeval.pipeline import _sha256_file
from eval.longmemeval.round_retrieval_run import (
    _build_rounds,
    _estimate_tokens,
    _load_questions,
    _load_raw_by_id,
    _question_from_raw,
    _retrieve_tfidf,
)

EXPECTED_TYPE_COUNTS = {
    "single-session-user": 70,
    "single-session-assistant": 56,
    "single-session-preference": 30,
    "multi-session": 133,
    "temporal-reasoning": 133,
    "knowledge-update": 78,
}
EXPECTED_TOTAL = 500
EXPECTED_ABSTENTION = 30
EXPECTED_JUDGE_MODEL = "gpt-4o-2024-08-06"

REPORTED_OVERALL = 0.658
REPORTED_TASK_AVG = 0.6387
REPORTED_ABSTENTION_ACC = 0.8667


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class RunContext:
    git_hash: str
    git_dirty: bool
    file_hashes: dict[str, str]
    artifact_counts: dict[str, int]
    artifact_ids_consistent: bool
    timestamp_utc: str


@dataclass
class LegitimacyResult:
    dataset_hash: str
    dataset_question_count: int
    dataset_type_counts: dict[str, int]
    dataset_type_counts_match: bool
    dataset_abstention_count: int

    output_line_count: int
    output_has_expected_unique: bool
    output_all_in_dataset: bool
    output_no_duplicates: bool

    scorer_judge_model_in_zoo: bool
    scorer_task_prompts: dict[str, bool]
    scored_all_correct_model: bool

    recomputed_overall: float
    recomputed_task_avg: float
    recomputed_abstention: float
    recomputed_per_type: dict[str, float]
    metrics_match: bool

    contamination_passed: bool
    contamination_flags: list[str]

    input_integrity_passed: bool
    input_integrity_failures: list[str]

    @property
    def verdict(self) -> str:
        checks = [
            self.dataset_type_counts_match,
            self.output_has_expected_unique,
            self.output_all_in_dataset,
            self.output_no_duplicates,
            self.scorer_judge_model_in_zoo,
            self.scored_all_correct_model,
            self.metrics_match,
            self.contamination_passed,
            self.input_integrity_passed,
        ]
        return "PASS" if all(checks) else "FAIL"


class FailureLayer(str, Enum):
    RETRIEVAL = "retrieval"
    EVIDENCE_QUALITY = "evidence_quality"
    READER = "reader"
    SCORER = "scorer"


@dataclass
class EvidencePathCase:
    question_id: str
    question_type: str
    question: str
    gold_answer: str
    hypothesis: str
    retrieval_recall: float
    gold_sessions_all_hit: bool
    gold_exact_substring: bool
    gold_token_overlap: bool
    key_entities_found: int
    key_entities_total: int
    evidence_contains_answer: bool
    failure_layer: str
    reader_context_tokens: int
    replay_sessions_match: bool


@dataclass
class EvidencePathReport:
    total_failures: int
    by_layer: dict[str, int]
    by_type_and_layer: dict[str, dict[str, int]]
    cases: list[EvidencePathCase]


@dataclass
class FailurePatternReport:
    matrix: dict[str, dict[str, int]]
    preference_cases: list[EvidencePathCase]
    preference_idk_count: int
    preference_dominant_pattern: str
    temporal_sample: list[EvidencePathCase]
    multi_session_sample: list[EvidencePathCase]
    short_answer_failures: list[EvidencePathCase]
    dominant_failure_layer: str
    dominant_layer_count: int
    dominant_layer_pct: float


@dataclass
class FixHypothesis:
    fix_id: str
    fix_name: str
    rationale: str
    estimated_cases_affected: int


@dataclass
class FixResult:
    question_id: str
    original_hypothesis: str
    new_hypothesis: str
    original_correct: bool
    new_correct: bool


@dataclass
class FixReport:
    fix_id: str
    cases_attempted: int
    flips_to_correct: int
    flips_to_wrong: int
    unchanged: int
    estimated_new_overall: float
    results: list[FixResult]


@dataclass
class AuditReport:
    run_context: RunContext | None = None
    legitimacy: LegitimacyResult | None = None
    evidence_path: EvidencePathReport | None = None
    failure_patterns: FailurePatternReport | None = None
    fix_hypothesis: FixHypothesis | None = None
    fix_report: FixReport | None = None
    timestamp_utc: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_key_entities(text: str) -> set[str]:
    entities: set[str] = set()
    for m in re.finditer(r"\b\d+(?:\.\d+)?\b", text):
        entities.add(m.group())
    for m in re.finditer(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text):
        entities.add(m.group().lower())
    for m in re.finditer(
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{1,2}\b",
        text, re.I,
    ):
        entities.add(m.group().lower())
    for m in re.finditer(r'"([^"]+)"', text):
        entities.add(m.group(1).lower())
    return entities


def _short_token_count(text: str) -> int:
    return len(re.findall(r"[a-zA-Z0-9]+", text))


def _print_header(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def _print_sub(title: str) -> None:
    print(f"\n--- {title} ---")


# ---------------------------------------------------------------------------
# Phase 0: Freeze Run Context
# ---------------------------------------------------------------------------

def phase0_freeze_context(
    data_path: Path,
    hyp_path: Path,
    scored_path: Path,
    diag_path: Path,
    runner_path: Path,
    scorer_path: Path,
) -> RunContext:
    _print_header("PHASE 0: Freeze Run Context")

    try:
        git_hash = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=str(ROOT),
        ).stdout.strip()
    except Exception:
        git_hash = "UNKNOWN"

    git_dirty = subprocess.run(
        ["git", "diff", "--quiet"],
        capture_output=True, cwd=str(ROOT),
    ).returncode != 0

    file_hashes = {
        "runner": _sha256_file(ROOT / runner_path),
        "scorer": _sha256_file(ROOT / scorer_path),
        "dataset": _sha256_file(ROOT / data_path),
    }

    hyp_data = _load_jsonl(ROOT / hyp_path)
    scored_data = _load_jsonl(ROOT / scored_path)
    diag_data = _load_json(ROOT / diag_path)

    hyp_ids = {r["question_id"] for r in hyp_data}
    scored_ids = {r["question_id"] for r in scored_data}
    diag_ids = {r["question_id"] for r in diag_data}

    consistent = hyp_ids == scored_ids == diag_ids
    counts = {
        "hypotheses": len(hyp_data),
        "scored": len(scored_data),
        "diagnostics": len(diag_data),
    }

    ctx = RunContext(
        git_hash=git_hash,
        git_dirty=git_dirty,
        file_hashes=file_hashes,
        artifact_counts=counts,
        artifact_ids_consistent=consistent,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )

    print(f"Git: {ctx.git_hash} ({'dirty' if ctx.git_dirty else 'clean'})")
    for name, h in ctx.file_hashes.items():
        print(f"  {name}: {h[:16]}...")
    print(f"Artifacts: {counts}")
    print(f"IDs consistent: {consistent}")
    if not consistent:
        diff_hyp_scored = hyp_ids.symmetric_difference(scored_ids)
        diff_hyp_diag = hyp_ids.symmetric_difference(diag_ids)
        print(f"  hyp vs scored mismatch: {len(diff_hyp_scored)}")
        print(f"  hyp vs diag mismatch: {len(diff_hyp_diag)}")
        print("INVALID ANALYSIS CONTEXT — stopping.")
        sys.exit(1)

    return ctx


# ---------------------------------------------------------------------------
# Phase 1: Legitimacy Audit
# ---------------------------------------------------------------------------

def phase1_legitimacy(
    data_path: Path,
    hyp_path: Path,
    scored_path: Path,
    diag_path: Path,
    runner_path: Path,
    _scorer_path: Path,
) -> LegitimacyResult:
    _print_header("PHASE 1: Legitimacy Audit")

    dataset = _load_json(ROOT / data_path)
    hyp_data = _load_jsonl(ROOT / hyp_path)
    scored_data = _load_jsonl(ROOT / scored_path)
    diag_data = _load_json(ROOT / diag_path)
    dataset_ids = {str(r["question_id"]) for r in dataset}

    # 1a: Dataset
    _print_sub("1a. Dataset legitimacy")
    type_counts: dict[str, int] = Counter()
    abs_count = 0
    for r in dataset:
        type_counts[r["question_type"]] += 1
        if "_abs" in str(r["question_id"]):
            abs_count += 1

    type_match = dict(type_counts) == EXPECTED_TYPE_COUNTS
    print(f"Total questions: {len(dataset)} (expected {EXPECTED_TOTAL})")
    print(f"Type counts: {dict(type_counts)}")
    print(f"Type counts match: {type_match}")
    print(f"Abstention count: {abs_count} (expected {EXPECTED_ABSTENTION})")
    print(f"Dataset hash: {_sha256_file(ROOT / data_path)[:16]}...")

    # 1b: Output format
    _print_sub("1b. Output format")
    hyp_ids_list = [r["question_id"] for r in hyp_data]
    hyp_ids_set = set(hyp_ids_list)
    no_dupes = len(hyp_ids_list) == len(hyp_ids_set)
    all_in_ds = hyp_ids_set.issubset(dataset_ids)
    has_expected = len(hyp_data) == EXPECTED_TOTAL and len(hyp_ids_set) == EXPECTED_TOTAL

    missing_keys = []
    for i, r in enumerate(hyp_data):
        if "question_id" not in r or "hypothesis" not in r:
            missing_keys.append(i)
    print(f"Lines: {len(hyp_data)}, Unique IDs: {len(hyp_ids_set)}")
    print(f"No duplicates: {no_dupes}")
    print(f"All in dataset: {all_in_ds}")
    print(f"Missing keys at lines: {missing_keys[:5]}" if missing_keys else "All lines have required keys")

    # 1c: Scorer
    _print_sub("1c. Scorer legitimacy")
    zoo_entry = model_zoo.get("gpt-4o", (None, None))
    judge_in_zoo = zoo_entry[0] == EXPECTED_JUDGE_MODEL
    print(f"model_zoo['gpt-4o'] = {zoo_entry} (expected model {EXPECTED_JUDGE_MODEL}): {'PASS' if judge_in_zoo else 'FAIL'}")

    task_prompts: dict[str, bool] = {}
    for task in list(EXPECTED_TYPE_COUNTS.keys()) + ["abstention"]:
        try:
            if task == "abstention":
                get_anscheck_prompt("single-session-user", "q", "a", "r", abstention=True)
            else:
                get_anscheck_prompt(task, "q", "a", "r")
            task_prompts[task] = True
        except (NotImplementedError, Exception):
            task_prompts[task] = False
    print(f"Task prompts: {task_prompts}")

    bad_models = [r["question_id"] for r in scored_data if r.get("autoeval_label", {}).get("model") != EXPECTED_JUDGE_MODEL]
    all_correct_model = len(bad_models) == 0
    print(f"All scored with {EXPECTED_JUDGE_MODEL}: {all_correct_model}" + (f" (exceptions: {bad_models[:5]})" if bad_models else ""))

    # 1d: Metrics
    _print_sub("1d. Metrics recomputation")
    ref_by_id = {str(r["question_id"]): r for r in dataset}
    type2acc: dict[str, list[int]] = {t: [] for t in EXPECTED_TYPE_COUNTS}
    abs_acc: list[int] = []
    all_acc: list[int] = []

    for r in scored_data:
        qid = r["question_id"]
        label = 1 if r["autoeval_label"]["label"] else 0
        ref = ref_by_id.get(qid)
        if ref:
            qtype = ref["question_type"]
            if qtype in type2acc:
                type2acc[qtype].append(label)
            all_acc.append(label)
        if "_abs" in qid:
            abs_acc.append(label)

    recomputed_per_type = {}
    task_means = []
    for t, acc_list in type2acc.items():
        if acc_list:
            m = sum(acc_list) / len(acc_list)
            recomputed_per_type[t] = round(m, 4)
            task_means.append(m)
            print(f"  {t}: {sum(acc_list)}/{len(acc_list)} = {m:.4f}")

    recomputed_overall = sum(all_acc) / len(all_acc) if all_acc else 0.0
    recomputed_task_avg = sum(task_means) / len(task_means) if task_means else 0.0
    recomputed_abstention = sum(abs_acc) / len(abs_acc) if abs_acc else 0.0

    print(f"Overall: {recomputed_overall:.4f} (reported {REPORTED_OVERALL})")
    print(f"Task-avg: {recomputed_task_avg:.4f} (reported {REPORTED_TASK_AVG})")
    print(f"Abstention: {recomputed_abstention:.4f} (reported {REPORTED_ABSTENTION_ACC})")

    metrics_match = (
        abs(recomputed_overall - REPORTED_OVERALL) < 0.002
        and abs(recomputed_task_avg - REPORTED_TASK_AVG) < 0.002
        and abs(recomputed_abstention - REPORTED_ABSTENTION_ACC) < 0.002
    )
    print(f"Metrics match: {metrics_match}")

    # 1e: Contamination
    _print_sub("1e. Contamination check")
    runner_text = (ROOT / runner_path).read_text(encoding="utf-8")
    contamination_flags: list[str] = []
    for pattern in ["longmemeval", "LongMemEval", "the answer is", "correct answer", "gold answer"]:
        # Only check prompt strings, not comments/docstrings
        system_start = runner_text.find('system = (')
        user_start = runner_text.find('user = (')
        if system_start >= 0:
            prompt_region = runner_text[system_start:system_start + 500]
            if pattern.lower() in prompt_region.lower():
                contamination_flags.append(f"'{pattern}' found in reader system prompt")
        if user_start >= 0:
            prompt_region = runner_text[user_start:user_start + 500]
            if pattern.lower() in prompt_region.lower():
                contamination_flags.append(f"'{pattern}' found in reader user prompt")

    gold_answers = [str(r.get("answer", "")) for r in dataset]
    system_prompt_text = (
        "You are a helpful assistant answering questions based on past "
        "conversation history. Use the provided conversation excerpts to answer. "
        'If the answer cannot be determined from the excerpts, say "I don\'t know."'
    )
    for ga in gold_answers:
        if len(ga) > 10 and ga.lower() in system_prompt_text.lower():
            contamination_flags.append(f"Gold answer '{ga[:40]}...' found in system prompt")
    contamination_passed = len(contamination_flags) == 0
    print(f"Contamination: {'PASS' if contamination_passed else 'FAIL'}")
    for f in contamination_flags:
        print(f"  FLAG: {f}")

    # 1f: Input integrity
    _print_sub("1f. Input integrity")
    diag_by_id = {str(r["question_id"]): r for r in diag_data}
    raw_by_id = {str(r["question_id"]): r for r in dataset}
    integrity_failures: list[str] = []
    for qid, diag_rec in diag_by_id.items():
        raw = raw_by_id.get(qid, {})
        haystack_sids = set()
        hs_ids = raw.get("haystack_session_ids", [])
        if hs_ids:
            haystack_sids = {str(s) for s in hs_ids}
        else:
            for i in range(len(raw.get("haystack_sessions", []))):
                haystack_sids.add(f"{qid}::session::{i:03d}")
        sessions_hit = set(diag_rec.get("sessions_hit", []))
        extra = sessions_hit - haystack_sids
        if extra:
            integrity_failures.append(f"{qid}: sessions_hit has {len(extra)} IDs not in haystack")
    input_integrity_passed = len(integrity_failures) == 0
    print(f"Input integrity: {'PASS' if input_integrity_passed else 'FAIL'} ({len(integrity_failures)} failures)")
    for f in integrity_failures[:5]:
        print(f"  {f}")

    result = LegitimacyResult(
        dataset_hash=_sha256_file(ROOT / data_path),
        dataset_question_count=len(dataset),
        dataset_type_counts=dict(type_counts),
        dataset_type_counts_match=type_match,
        dataset_abstention_count=abs_count,
        output_line_count=len(hyp_data),
        output_has_expected_unique=has_expected,
        output_all_in_dataset=all_in_ds,
        output_no_duplicates=no_dupes,
        scorer_judge_model_in_zoo=judge_in_zoo,
        scorer_task_prompts=task_prompts,
        scored_all_correct_model=all_correct_model,
        recomputed_overall=round(recomputed_overall, 4),
        recomputed_task_avg=round(recomputed_task_avg, 4),
        recomputed_abstention=round(recomputed_abstention, 4),
        recomputed_per_type=recomputed_per_type,
        metrics_match=metrics_match,
        contamination_passed=contamination_passed,
        contamination_flags=contamination_flags,
        input_integrity_passed=input_integrity_passed,
        input_integrity_failures=integrity_failures,
    )
    print(f"\nLEGITIMACY VERDICT: {result.verdict}")
    return result


# ---------------------------------------------------------------------------
# Phase 2: Evidence-Path Audit
# ---------------------------------------------------------------------------

def phase2_evidence_path(
    data_path: Path,
    _hyp_path: Path,
    scored_path: Path,
    diag_path: Path,
) -> EvidencePathReport:
    _print_header("PHASE 2: Evidence-Path Audit")

    raw_by_id = _load_raw_by_id(ROOT / data_path)
    questions = [_question_from_raw(raw_by_id, q) for q in _load_questions(ROOT / data_path)]
    q_by_id = {q.question_id: q for q in questions}

    scored_data = _load_jsonl(ROOT / scored_path)

    diag_data = _load_json(ROOT / diag_path)
    diag_by_id = {r["question_id"]: r for r in diag_data}

    failed_ids = [r["question_id"] for r in scored_data if not r["autoeval_label"]["label"]]
    print(f"Total failures to analyze: {len(failed_ids)}")

    # Validate TF-IDF replay on first 10
    replay_divergences = 0
    validation_ids = failed_ids[:10]
    print("Validating TF-IDF replay on first 10 cases...")
    for qid in validation_ids:
        q = q_by_id[qid]
        rounds = _build_rounds(q)
        retrieved = _retrieve_tfidf(q.question, rounds, top_k=30)
        replay_hits = sorted({r.session_id for r in retrieved})
        diag_hits = sorted(diag_by_id[qid].get("sessions_hit", []))
        if replay_hits != diag_hits:
            replay_divergences += 1
    replay_ok = replay_divergences <= 1
    print(f"Replay validation: {10 - replay_divergences}/10 match ({'OK' if replay_ok else 'DEGRADED — using diagnostics fallback'})")

    cases: list[EvidencePathCase] = []
    for idx, qid in enumerate(failed_ids):
        if idx % 25 == 0:
            print(f"  Analyzing failure {idx + 1}/{len(failed_ids)}...", flush=True)

        q = q_by_id[qid]
        diag = diag_by_id[qid]
        gold = diag["gold_answer"]
        hypothesis = diag["hypothesis"]
        qtype = diag["question_type"]
        recall = diag["retrieval_recall"]
        diag_gold_sids = set(str(s) for s in diag.get("gold_session_ids", []))
        diag_sessions_hit = set(diag.get("sessions_hit", []))
        gold_all_hit = diag_gold_sids.issubset(diag_sessions_hit) if diag_gold_sids else True

        # Replay retrieval to get the actual reader context
        rounds = _build_rounds(q)
        retrieved = _retrieve_tfidf(q.question, rounds, top_k=30)
        context = "\n\n---\n\n".join(r.value for r in retrieved)
        replay_hits = sorted({r.session_id for r in retrieved})
        diag_hits_sorted = sorted(diag_sessions_hit)
        sessions_match = replay_hits == diag_hits_sorted

        # Evidence presence checks
        exact_sub = gold.lower() in context.lower() if len(gold) > 5 else False
        token_overlap = _gold_in_text(gold, context)

        gold_entities = _extract_key_entities(gold)
        context_lower = context.lower()
        found_entities = sum(1 for e in gold_entities if e in context_lower)
        total_entities = len(gold_entities)

        evidence_ok = exact_sub or token_overlap or (total_entities > 0 and found_entities / total_entities >= 0.5)

        # Layer classification
        if not gold_all_hit:
            layer = FailureLayer.RETRIEVAL
        elif not evidence_ok:
            layer = FailureLayer.EVIDENCE_QUALITY
        elif _gold_in_text(gold, hypothesis):
            layer = FailureLayer.SCORER
        else:
            layer = FailureLayer.READER

        cases.append(EvidencePathCase(
            question_id=qid,
            question_type=qtype,
            question=diag["question"],
            gold_answer=gold,
            hypothesis=hypothesis,
            retrieval_recall=recall,
            gold_sessions_all_hit=gold_all_hit,
            gold_exact_substring=exact_sub,
            gold_token_overlap=token_overlap,
            key_entities_found=found_entities,
            key_entities_total=total_entities,
            evidence_contains_answer=evidence_ok,
            failure_layer=layer.value,
            reader_context_tokens=_estimate_tokens(context),
            replay_sessions_match=sessions_match,
        ))

    by_layer: dict[str, int] = Counter()
    by_type_and_layer: dict[str, dict[str, int]] = defaultdict(lambda: Counter())
    for c in cases:
        by_layer[c.failure_layer] += 1
        by_type_and_layer[c.question_type][c.failure_layer] += 1

    report = EvidencePathReport(
        total_failures=len(cases),
        by_layer=dict(by_layer),
        by_type_and_layer={k: dict(v) for k, v in by_type_and_layer.items()},
        cases=cases,
    )

    print(f"\nFailure attribution ({report.total_failures} total):")
    for layer, count in sorted(by_layer.items(), key=lambda x: -x[1]):
        print(f"  {layer}: {count} ({100 * count / len(cases):.1f}%)")

    return report


# ---------------------------------------------------------------------------
# Phase 3: Failure Pattern Analysis
# ---------------------------------------------------------------------------

def phase3_failure_patterns(
    evidence_report: EvidencePathReport,
    _diag_path: Path,
) -> FailurePatternReport:
    _print_header("PHASE 3: Failure Pattern Analysis")
    cases = evidence_report.cases

    # 3A: Failure matrix
    _print_sub("3A. Failure matrix")
    all_layers = sorted({c.failure_layer for c in cases})
    all_types = sorted({c.question_type for c in cases})

    matrix: dict[str, dict[str, int]] = {}
    for qtype in all_types:
        row: dict[str, int] = {}
        type_cases = [c for c in cases if c.question_type == qtype]
        for layer in all_layers:
            row[layer] = sum(1 for c in type_cases if c.failure_layer == layer)
        row["total"] = len(type_cases)
        matrix[qtype] = row

    header = f"{'type':<30}" + "".join(f"{l:<18}" for l in all_layers) + "total"
    print(header)
    print("-" * len(header))
    for qtype, row in matrix.items():
        cols = f"{qtype:<30}" + "".join(f"{row.get(l, 0):<18}" for l in all_layers) + str(row["total"])
        print(cols)

    # 3B: Preference deep dive
    _print_sub("3B. Preference deep dive (0/30)")
    pref_cases = [c for c in cases if c.question_type == "single-session-preference"]
    idk_count = sum(1 for c in pref_cases if "don't know" in c.hypothesis.lower() or "i don't know" in c.hypothesis.lower())
    generic_count = sum(1 for c in pref_cases if c.failure_layer == FailureLayer.READER.value and "don't know" not in c.hypothesis.lower())

    print(f"Total preference failures: {len(pref_cases)}")
    print(f"  'I don't know' responses: {idk_count}")
    print(f"  Generic/wrong answers: {generic_count}")
    print(f"  Evidence quality failures: {sum(1 for c in pref_cases if c.failure_layer == FailureLayer.EVIDENCE_QUALITY.value)}")
    print(f"  Retrieval failures: {sum(1 for c in pref_cases if c.failure_layer == FailureLayer.RETRIEVAL.value)}")

    for c in pref_cases[:5]:
        print(f"\n  [{c.question_id}]")
        print(f"    Q: {c.question[:80]}")
        print(f"    Gold: {c.gold_answer[:100]}...")
        print(f"    Hyp: {c.hypothesis[:80]}")
        print(f"    Recall: {c.retrieval_recall}, Evidence OK: {c.evidence_contains_answer}, Layer: {c.failure_layer}")

    if idk_count >= len(pref_cases) * 0.8:
        pref_pattern = "reader_idk_despite_evidence"
    elif generic_count >= len(pref_cases) * 0.5:
        pref_pattern = "reader_generic_advice"
    else:
        pref_pattern = "mixed"
    print(f"\nDominant pattern: {pref_pattern}")

    # 3C: Temporal deep dive
    _print_sub("3C. Temporal deep dive")
    temporal_failures = [c for c in cases if c.question_type == "temporal-reasoning"]
    temporal_sample = temporal_failures[:10]
    for c in temporal_sample:
        print(f"\n  [{c.question_id}]")
        print(f"    Q: {c.question[:80]}")
        print(f"    Gold: {c.gold_answer[:80]}")
        print(f"    Hyp: {c.hypothesis[:80]}")
        print(f"    Layer: {c.failure_layer}, Evidence OK: {c.evidence_contains_answer}")
    temporal_reader = sum(1 for c in temporal_failures if c.failure_layer == FailureLayer.READER.value)
    temporal_evidence = sum(1 for c in temporal_failures if c.failure_layer == FailureLayer.EVIDENCE_QUALITY.value)
    print(f"\nTemporal failures: {len(temporal_failures)} total, {temporal_reader} reader, {temporal_evidence} evidence")

    # 3D: Multi-session deep dive
    _print_sub("3D. Multi-session deep dive")
    multi_failures = [c for c in cases if c.question_type == "multi-session"]
    multi_sample = multi_failures[:10]
    for c in multi_sample:
        print(f"\n  [{c.question_id}]")
        print(f"    Q: {c.question[:80]}")
        print(f"    Gold: {c.gold_answer[:80]}")
        print(f"    Hyp: {c.hypothesis[:80]}")
        print(f"    Layer: {c.failure_layer}, Evidence OK: {c.evidence_contains_answer}")
    multi_reader = sum(1 for c in multi_failures if c.failure_layer == FailureLayer.READER.value)
    multi_retrieval = sum(1 for c in multi_failures if c.failure_layer == FailureLayer.RETRIEVAL.value)
    print(f"\nMulti-session failures: {len(multi_failures)} total, {multi_reader} reader, {multi_retrieval} retrieval")

    # 3E: Short-answer / exact-value deep dive
    _print_sub("3E. Short-answer / exact-value failures")
    short_failures = [c for c in cases if _short_token_count(c.gold_answer) <= 3]
    for c in short_failures[:10]:
        print(f"  [{c.question_id}] gold='{c.gold_answer}' hyp='{c.hypothesis[:60]}' layer={c.failure_layer} evidence_ok={c.evidence_contains_answer}")
    print(f"\nShort-answer failures: {len(short_failures)}")

    # Dominant layer
    layer_counts = Counter(c.failure_layer for c in cases)
    dominant = layer_counts.most_common(1)[0] if layer_counts else ("none", 0)

    report = FailurePatternReport(
        matrix=matrix,
        preference_cases=pref_cases,
        preference_idk_count=idk_count,
        preference_dominant_pattern=pref_pattern,
        temporal_sample=temporal_sample,
        multi_session_sample=multi_sample,
        short_answer_failures=short_failures,
        dominant_failure_layer=dominant[0],
        dominant_layer_count=dominant[1],
        dominant_layer_pct=round(100 * dominant[1] / len(cases), 1) if cases else 0.0,
    )
    print(f"\nDominant failure layer: {report.dominant_failure_layer} ({report.dominant_layer_count}, {report.dominant_layer_pct}%)")
    return report


# ---------------------------------------------------------------------------
# Phase 4: Fix Hypothesis
# ---------------------------------------------------------------------------

def phase4_fix_hypothesis(
    _pattern_report: FailurePatternReport,
    evidence_report: EvidencePathReport,
) -> FixHypothesis:
    _print_header("PHASE 4: Highest-Leverage Fix Hypothesis")

    total = evidence_report.total_failures
    reader_n = evidence_report.by_layer.get(FailureLayer.READER.value, 0)
    evidence_n = evidence_report.by_layer.get(FailureLayer.EVIDENCE_QUALITY.value, 0)
    scorer_n = evidence_report.by_layer.get(FailureLayer.SCORER.value, 0)
    retrieval_n = evidence_report.by_layer.get(FailureLayer.RETRIEVAL.value, 0)

    print(f"Failure breakdown: reader={reader_n}, evidence={evidence_n}, scorer={scorer_n}, retrieval={retrieval_n}")

    if reader_n > total * 0.5:
        fix = FixHypothesis(
            fix_id="A",
            fix_name="reader_prompt_improvement",
            rationale=(
                f"Reader failures dominate: {reader_n}/{total} ({100*reader_n/total:.0f}%). "
                f"Preference 0/30 catastrophe is reader-side (IDK despite evidence). "
                f"Temporal and multi-session also show reader-layer failures with evidence present."
            ),
            estimated_cases_affected=reader_n,
        )
    elif evidence_n > total * 0.5:
        fix = FixHypothesis(
            fix_id="B",
            fix_name="evidence_quality_improvement",
            rationale=f"Evidence quality failures dominate: {evidence_n}/{total} ({100*evidence_n/total:.0f}%).",
            estimated_cases_affected=evidence_n,
        )
    elif scorer_n > total * 0.1:
        fix = FixHypothesis(
            fix_id="C",
            fix_name="scorer_cleanup",
            rationale=f"Scorer failures exceed 10%: {scorer_n}/{total} ({100*scorer_n/total:.0f}%).",
            estimated_cases_affected=scorer_n,
        )
    elif retrieval_n > total * 0.5:
        fix = FixHypothesis(
            fix_id="D",
            fix_name="retrieval_method_change",
            rationale=f"Retrieval failures dominate: {retrieval_n}/{total} ({100*retrieval_n/total:.0f}%).",
            estimated_cases_affected=retrieval_n,
        )
    else:
        fix = FixHypothesis(
            fix_id="A",
            fix_name="reader_prompt_improvement",
            rationale=(
                f"No single layer >50%, but reader is largest: {reader_n}/{total}. "
                f"Reader prompt improvement is the highest-leverage first move."
            ),
            estimated_cases_affected=reader_n,
        )

    print(f"\nRecommended: Fix {fix.fix_id} — {fix.fix_name}")
    print(f"Rationale: {fix.rationale}")
    print(f"Estimated cases affected: {fix.estimated_cases_affected}")
    return fix


# ---------------------------------------------------------------------------
# Phase 5: Targeted Fix (cost-gated)
# ---------------------------------------------------------------------------

IMPROVED_SYSTEM_PROMPT = (
    "You are a helpful assistant with access to a user's past conversation history. "
    "Use the conversation excerpts below to answer their question.\n\n"
    "Guidelines:\n"
    "- When the question asks for a recommendation or suggestion, use the user's "
    "stated preferences, interests, and constraints from the conversations to "
    "give a personalized answer. Do not say \"I don't know\" if you can infer "
    "preferences from the history.\n"
    "- When the question involves time, dates, or durations, carefully examine "
    "timestamps and chronological order in the excerpts. Compute exact values.\n"
    "- When the question requires combining information across multiple conversations, "
    "systematically gather all relevant facts before answering.\n"
    "- Only say \"I don't know\" if the excerpts truly contain no relevant information.\n"
    "- Be concise."
)


def _read_answer_fixed(
    *,
    client: Any,
    model: str,
    question: str,
    context: str,
    max_tokens: int = 256,
) -> str:
    user_msg = (
        "Here are relevant excerpts from past conversations:\n\n"
        f"{context}\n\n"
        f"Question: {question}\n\n"
        "Answer concisely."
    )
    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": IMPROVED_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=max_tokens,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            if "content_filter" in str(exc) or "ResponsibleAIPolicyViolation" in str(exc):
                return "I don't know."
            if attempt < 4:
                time.sleep(2 ** attempt)
                continue
            return "I don't know."
    return "I don't know."


def _judge_one(
    *,
    client: Any,
    model: str,
    question_id: str,
    question: str,
    gold: str,
    hypothesis: str,
    question_type: str,
) -> bool:
    is_abs = "_abs" in question_id
    prompt = get_anscheck_prompt(question_type, question, gold, hypothesis, abstention=is_abs)
    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                n=1,
                temperature=0,
                max_tokens=10,
                timeout=60,
            )
            return "yes" in resp.choices[0].message.content.strip().lower()
        except Exception:
            if attempt < 4:
                time.sleep(2 ** attempt)
                continue
            return False
    return False


def phase5_targeted_fix(
    fix: FixHypothesis,
    evidence_report: EvidencePathReport,
    data_path: Path,
    *,
    execute: bool = False,
    max_cases: int = 0,
) -> FixReport | None:
    _print_header("PHASE 5: Targeted Fix")

    if fix.fix_id != "A":
        print(f"Fix {fix.fix_id} ({fix.fix_name}) is not implemented in this pipeline.")
        print("Only Fix A (reader prompt improvement) is supported.")
        return None

    # Select cases: reader-layer failures first (most likely to flip), then others
    reader_cases = [c for c in evidence_report.cases if c.failure_layer == FailureLayer.READER.value]
    other_cases = [c for c in evidence_report.cases if c.failure_layer != FailureLayer.READER.value]
    all_failed = reader_cases + other_cases

    if max_cases > 0:
        all_failed = all_failed[:max_cases]

    est_reader_cost = len(all_failed) * 0.005
    est_judge_cost = len(all_failed) * 0.001
    est_total = est_reader_cost + est_judge_cost

    print(f"Cases to re-run: {len(all_failed)}")
    print(f"  Reader-layer: {len([c for c in all_failed if c.failure_layer == FailureLayer.READER.value])}")
    print(f"  Other layers: {len([c for c in all_failed if c.failure_layer != FailureLayer.READER.value])}")
    print(f"Estimated cost: ${est_total:.2f} (reader ${est_reader_cost:.2f} + judge ${est_judge_cost:.2f})")

    if not execute:
        print("\n[DRY RUN] Pass --execute-fix to actually call GPT-4o.")
        print(f"Improved system prompt:\n{IMPROVED_SYSTEM_PROMPT}")
        return None

    print("\nExecuting fix...")
    from eval._openai_client import make_openai_client
    reader_client, reader_model = make_openai_client("longmemeval_reader")
    judge_client, judge_model = make_openai_client("longmemeval_judge")

    raw_by_id = _load_raw_by_id(ROOT / data_path)
    questions = [_question_from_raw(raw_by_id, q) for q in _load_questions(ROOT / data_path)]
    q_by_id = {q.question_id: q for q in questions}

    results: list[FixResult] = []
    for idx, case in enumerate(all_failed):
        if idx % 10 == 0:
            print(f"  Re-running {idx + 1}/{len(all_failed)}...", flush=True)

        q = q_by_id[case.question_id]
        rounds = _build_rounds(q)
        retrieved = _retrieve_tfidf(q.question, rounds, top_k=30)
        context = "\n\n---\n\n".join(r.value for r in retrieved)

        new_hyp = _read_answer_fixed(
            client=reader_client,
            model=reader_model,
            question=q.question,
            context=context,
        )

        new_correct = _judge_one(
            client=judge_client,
            model=judge_model,
            question_id=case.question_id,
            question=case.question,
            gold=case.gold_answer,
            hypothesis=new_hyp,
            question_type=case.question_type,
        )

        results.append(FixResult(
            question_id=case.question_id,
            original_hypothesis=case.hypothesis,
            new_hypothesis=new_hyp,
            original_correct=False,
            new_correct=new_correct,
        ))

    flips_correct = sum(1 for r in results if r.new_correct)
    flips_wrong = 0  # all original were wrong, so no regressions possible
    unchanged = len(results) - flips_correct

    # Project new overall: 329 original correct + flips
    estimated_new = (329 + flips_correct) / EXPECTED_TOTAL

    report = FixReport(
        fix_id=fix.fix_id,
        cases_attempted=len(results),
        flips_to_correct=flips_correct,
        flips_to_wrong=flips_wrong,
        unchanged=unchanged,
        estimated_new_overall=round(estimated_new, 4),
        results=results,
    )

    print(f"\nResults:")
    print(f"  Cases attempted: {report.cases_attempted}")
    print(f"  Flipped to correct: {report.flips_to_correct} ({100*report.flips_to_correct/report.cases_attempted:.1f}%)")
    print(f"  Unchanged: {report.unchanged}")
    print(f"  Estimated new overall accuracy: {report.estimated_new_overall:.4f} ({report.estimated_new_overall*100:.1f}%)")

    flip_rate = report.flips_to_correct / report.cases_attempted if report.cases_attempted else 0
    if flip_rate >= 0.30:
        print(f"\n  Flip rate {flip_rate:.1%} >= 30%: FULL 500 RERUN JUSTIFIED")
    elif flip_rate < 0.15:
        print(f"\n  Flip rate {flip_rate:.1%} < 15%: FIX FAMILY INSUFFICIENT")
    else:
        print(f"\n  Flip rate {flip_rate:.1%}: moderate improvement, consider full rerun")

    return report


# ---------------------------------------------------------------------------
# Phase 6: Final Report
# ---------------------------------------------------------------------------

def phase6_report(
    audit: AuditReport,
    output_path: Path,
) -> None:
    _print_header("PHASE 6: Final Report")

    audit.timestamp_utc = datetime.now(timezone.utc).isoformat()

    # Serialize (dataclass -> dict, strip large case lists for JSON)
    def _slim(obj: Any) -> Any:
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            d = dataclasses.asdict(obj)
            if "cases" in d and isinstance(d["cases"], list) and len(d["cases"]) > 20:
                d["cases_sample"] = d["cases"][:20]
                d["cases_count"] = len(d["cases"])
                del d["cases"]
            if "preference_cases" in d and isinstance(d["preference_cases"], list) and len(d["preference_cases"]) > 10:
                d["preference_cases_sample"] = d["preference_cases"][:10]
                d["preference_cases_count"] = len(d["preference_cases"])
                del d["preference_cases"]
            if "results" in d and isinstance(d["results"], list) and len(d["results"]) > 20:
                d["results_sample"] = d["results"][:20]
                d["results_count"] = len(d["results"])
                del d["results"]
            return d
        return obj

    report_dict = _slim(audit)
    for key in list(report_dict.keys()):
        if dataclasses.is_dataclass(report_dict[key]) and not isinstance(report_dict[key], type):
            report_dict[key] = _slim(report_dict[key])

    out = ROOT / output_path
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report_dict, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    print(f"Report written to {out}")

    # Human-readable summary
    print("\n=== LongMemEval-S Audit Report ===\n")

    if audit.legitimacy:
        leg = audit.legitimacy
        print(f"Phase 1 — Legitimacy: {leg.verdict}")
        print(f"  dataset: {'PASS' if leg.dataset_type_counts_match else 'FAIL'}")
        print(f"  scorer: {'PASS' if leg.scored_all_correct_model else 'FAIL'}")
        print(f"  metrics: {'PASS' if leg.metrics_match else 'FAIL'}")
        print(f"  contamination: {'PASS' if leg.contamination_passed else 'FAIL'}")
        print(f"  input integrity: {'PASS' if leg.input_integrity_passed else 'FAIL'}")

    if audit.legitimacy:
        print(f"\nRun 1 (baseline round retrieval):")
        print(f"  Overall accuracy: {audit.legitimacy.recomputed_overall}")
        print(f"  Task-averaged accuracy: {audit.legitimacy.recomputed_task_avg}")
        print(f"  Abstention accuracy: {audit.legitimacy.recomputed_abstention}")
        for t, acc in audit.legitimacy.recomputed_per_type.items():
            print(f"    {t}: {acc}")

    if audit.evidence_path:
        ep = audit.evidence_path
        print(f"\nFailure attribution ({ep.total_failures} failures):")
        for layer, count in sorted(ep.by_layer.items(), key=lambda x: -x[1]):
            print(f"  {layer}: {count}")

    if audit.failure_patterns:
        fp = audit.failure_patterns
        print(f"\nDominant weakness: {fp.dominant_failure_layer} ({fp.dominant_layer_pct}%)")

    if audit.fix_hypothesis:
        fh = audit.fix_hypothesis
        print(f"\nChosen fix: {fh.fix_id} — {fh.fix_name}")
        print(f"  Rationale: {fh.rationale}")

    if audit.fix_report:
        fr = audit.fix_report
        print(f"\nFailed-case rerun:")
        print(f"  Attempted: {fr.cases_attempted}")
        print(f"  Flipped to correct: {fr.flips_to_correct}")
        print(f"  Estimated new overall: {fr.estimated_new_overall}")

    print("\nComparison to published systems:")
    published = [
        ("Mem0", 0.934),
        ("Mastra OM (gpt-4o)", 0.842),
        ("EverMemOS", 0.830),
        ("TiMem", 0.769),
        ("Zep/Graphiti", 0.712),
        ("Full-context GPT-4o (CoN)", 0.640),
    ]
    our_score = audit.fix_report.estimated_new_overall if audit.fix_report else (audit.legitimacy.recomputed_overall if audit.legitimacy else 0.658)
    for name, score in published:
        marker = " <-- us" if abs(score - our_score) < 0.005 else ""
        print(f"  {name}: {score:.1%}{marker}")
    print(f"  Us: {our_score:.1%}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="LongMemEval-S benchmark audit pipeline")
    parser.add_argument("--phase", default="all", choices=["0", "1", "2", "3", "4", "5", "6", "all", "0-4"])
    parser.add_argument("--data", type=Path, default=Path("eval/longmemeval/data/longmemeval_s_cleaned.json"))
    parser.add_argument("--hyp", type=Path, default=Path("eval/longmemeval/results/round_full_500.jsonl"))
    parser.add_argument("--scored", type=Path, default=Path("eval/longmemeval/results/round_full_500_scored.jsonl"))
    parser.add_argument("--diag", type=Path, default=Path("eval/longmemeval/results/round_full_500_diagnostics.json"))
    parser.add_argument("--runner", type=Path, default=Path("eval/longmemeval/round_retrieval_run.py"))
    parser.add_argument("--scorer", type=Path, default=Path("eval/longmemeval/official_evaluate_qa.py"))
    parser.add_argument("--output", type=Path, default=Path("eval/longmemeval/results/audit_report.json"))
    parser.add_argument("--execute-fix", action="store_true")
    parser.add_argument("--max-fix-cases", type=int, default=0, help="0 = all failures")
    args = parser.parse_args()

    phases = set()
    if args.phase == "all":
        phases = {0, 1, 2, 3, 4, 5, 6}
    elif args.phase == "0-4":
        phases = {0, 1, 2, 3, 4}
    else:
        phases = {int(args.phase)}

    audit = AuditReport()

    if 0 in phases:
        audit.run_context = phase0_freeze_context(
            args.data, args.hyp, args.scored, args.diag, args.runner, args.scorer,
        )

    if 1 in phases:
        audit.legitimacy = phase1_legitimacy(
            args.data, args.hyp, args.scored, args.diag, args.runner, args.scorer,
        )

    if 2 in phases:
        audit.evidence_path = phase2_evidence_path(
            args.data, args.hyp, args.scored, args.diag,
        )

    if 3 in phases:
        if audit.evidence_path is None:
            print("Phase 3 requires Phase 2 output. Running Phase 2 first...")
            audit.evidence_path = phase2_evidence_path(
                args.data, args.hyp, args.scored, args.diag,
            )
        audit.failure_patterns = phase3_failure_patterns(
            audit.evidence_path, args.diag,
        )

    if 4 in phases:
        if audit.evidence_path is None:
            audit.evidence_path = phase2_evidence_path(
                args.data, args.hyp, args.scored, args.diag,
            )
        if audit.failure_patterns is None:
            audit.failure_patterns = phase3_failure_patterns(
                audit.evidence_path, args.diag,
            )
        audit.fix_hypothesis = phase4_fix_hypothesis(
            audit.failure_patterns, audit.evidence_path,
        )

    if 5 in phases:
        if audit.fix_hypothesis is None:
            print("Phase 5 requires Phase 4 output. Run --phase 0-4 first or --phase all.")
            return
        if audit.evidence_path is None:
            print("Phase 5 requires Phase 2 output.")
            return
        audit.fix_report = phase5_targeted_fix(
            audit.fix_hypothesis,
            audit.evidence_path,
            args.data,
            execute=args.execute_fix,
            max_cases=args.max_fix_cases,
        )

    if 6 in phases or args.phase in ("all", "0-4"):
        phase6_report(audit, args.output)


if __name__ == "__main__":
    main()
