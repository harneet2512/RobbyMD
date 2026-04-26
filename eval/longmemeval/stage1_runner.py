"""Stage 1 runner: low-risk, non-architectural improvements to LongMemEval-S.

Changes applied:
  1A. Scorer cleanup — strict short-answer matching
  1B. Chronological round ordering
  1C. JSON + Chain-of-Note input formatting
  1D. Improved general reader prompt

Does NOT touch: retrieval architecture, claim extraction, substrate pipeline,
evidence verifier, token budget, supersession, question-type routing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import threading
import time
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval._openai_client import make_openai_client
from eval.longmemeval.official_evaluate_qa import get_anscheck_prompt
from eval.longmemeval.pipeline import _sha256_file
from eval.longmemeval.round_retrieval_run import (
    RoundRecord,
    _build_rounds,
    _estimate_tokens,
    _load_questions,
    _load_raw_by_id,
    _question_from_raw,
    _retrieve_tfidf,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_DATA = Path("eval/longmemeval/data/longmemeval_s_cleaned.json")
BASELINE_HYP = Path("eval/longmemeval/results/round_full_500.jsonl")
BASELINE_SCORED = Path("eval/longmemeval/results/round_full_500_scored.jsonl")
BASELINE_DIAG = Path("eval/longmemeval/results/round_full_500_diagnostics.json")
STAGE1_OUTPUT_DIR = Path("eval/longmemeval/results/stage1")

TOP_K = 30

# ---------------------------------------------------------------------------
# 1A. Scorer cleanup — strict short-answer matching
# ---------------------------------------------------------------------------

def _normalize_short(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = re.sub(r"(?<=\d)[,_](?=\d)", "", normalized)
    chars: list[str] = []
    for ch in normalized:
        if unicodedata.category(ch) == "Sc":
            continue
        chars.append(ch if ch.isalnum() else " ")
    return re.sub(r"\s+", " ", "".join(chars)).strip()


def _short_token_count(normalized: str) -> int:
    return len(re.findall(r"[a-z0-9]+", normalized))


def _boundary_match(gold_norm: str, text_norm: str) -> bool:
    pattern = rf"(?<![a-z0-9]){re.escape(gold_norm)}(?![a-z0-9])"
    return re.search(pattern, text_norm) is not None


def strict_short_answer_check(gold: str, hypothesis: str) -> bool | None:
    """For short gold answers (<=3 tokens), apply strict matching.

    Returns True if exact/boundary match found.
    Returns False if hypothesis contains a conflicting numeric value.
    Returns None to defer to the GPT-4o judge for ambiguous cases.
    """
    gold_norm = _normalize_short(gold)
    if _short_token_count(gold_norm) > 3:
        return None
    if not gold_norm:
        return None
    hyp_norm = _normalize_short(hypothesis)
    if not hyp_norm:
        return False
    if gold_norm == hyp_norm or _boundary_match(gold_norm, hyp_norm):
        return True
    gold_numbers = set(re.findall(r"\d+", gold_norm))
    if gold_numbers:
        hyp_numbers = set(re.findall(r"\d+", hyp_norm))
        if hyp_numbers and not gold_numbers.intersection(hyp_numbers):
            return False
    return None


def test_scorer_cleanup() -> None:
    """Verify scorer cleanup behaves correctly on known cases."""
    cases = [
        ("$400,000", "$350,000", False),      # different dollar amount
        ("8 days", "3 days", False),           # different day count
        ("2", "1", False),                     # different digit
        ("Chicago", "the suburbs", None),      # non-numeric, no match -> defer to judge
        ("3", "You acquired two plants: a peace lily and a succulent.", None),  # no digit in hyp -> defer
        ("5", "Based on the excerpts, six babies were born", None),            # no digit in hyp -> defer
        ("Target", "You redeemed a $5 coupon on coffee creamer last Sunday from your email inbox.", None),  # no boundary match -> defer
        ("4", "Based on the excerpts, you bought and assembled an IKEA bookshelf", None),  # no "4" in hyp -> defer
        ("2", "You went to two doctor's appointments", None),  # word-number, no digit -> defer
        ("Spotify", "You use Spotify for music.", True),       # boundary match
        ("45 minutes each way", "Your daily commute is 45 minutes each way.", None),  # >3 tokens -> skip
        ("3", "Based on the excerpts, 5 items were found", False),  # different number present
        ("2", "You have 2 appointments scheduled", True),      # exact boundary match
    ]
    passed = 0
    for gold, hyp, expected in cases:
        result = strict_short_answer_check(gold, hyp)
        ok = result == expected
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] gold='{gold}' hyp='{hyp[:50]}' expected={expected} got={result}")
        if ok:
            passed += 1
    print(f"  Scorer cleanup tests: {passed}/{len(cases)}")


# ---------------------------------------------------------------------------
# 1B. Chronological round ordering
# ---------------------------------------------------------------------------

def _sort_chronological(rounds: list[RoundRecord]) -> list[RoundRecord]:
    """Sort retrieved rounds by timestamp (chronological), then session order."""
    return sorted(rounds, key=lambda r: (r.timestamp, r.session_order, r.round_index))


# ---------------------------------------------------------------------------
# 1C. JSON + Chain-of-Note input formatting
# ---------------------------------------------------------------------------

def _format_rounds_json(rounds: list[RoundRecord]) -> list[dict[str, str]]:
    result = []
    for r in rounds:
        parts = r.key.split("\n", 1)
        speaker = "unknown"
        content = r.key
        if parts[0].startswith("user:") or parts[0].startswith("assistant:"):
            speaker = parts[0].split(":")[0]
            content = r.key
        result.append({
            "timestamp": r.timestamp,
            "session_id": r.session_id,
            "round": r.round_index,
            "speaker": speaker,
            "content": content,
        })
    return result


NOTE_EXTRACTION_SYSTEM = (
    "You extract structured reading notes from conversation excerpts to help "
    "answer a question. Extract ONLY facts directly relevant to the question. "
    "Be brief and factual. Output a JSON array of strings."
)

NOTE_EXTRACTION_USER = (
    "Question: {question}\n\n"
    "Conversation excerpts (JSON):\n{json_rounds}\n\n"
    "Extract reading notes under these categories (skip empty categories):\n"
    "1. Preferences/constraints: any stated user preferences, interests, likes, dislikes\n"
    "2. Temporal facts: dates, durations, sequences, before/after relationships, timestamps\n"
    "3. Conflicting/updated values: facts that contradict or update earlier facts\n"
    "4. Cross-conversation facts: facts spanning multiple conversations needing synthesis\n\n"
    "Return ONLY a JSON array of note strings. Example: [\"User prefers Sony cameras\", \"Trip was in March 2024\"]"
)


def _extract_reading_notes(
    *,
    client: Any,
    model: str,
    question: str,
    json_rounds: list[dict[str, str]],
) -> list[str]:
    rounds_text = json.dumps(json_rounds, indent=1, ensure_ascii=False)
    # Truncate if too long for the note extractor
    if len(rounds_text) > 60000:
        rounds_text = rounds_text[:60000] + "\n... (truncated)"

    user_msg = NOTE_EXTRACTION_USER.format(question=question, json_rounds=rounds_text)

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": NOTE_EXTRACTION_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=512,
                response_format={"type": "json_object"},
            )
            raw = (resp.choices[0].message.content or "").strip()
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(n) for n in parsed]
            if isinstance(parsed, dict):
                notes = []
                for v in parsed.values():
                    if isinstance(v, list):
                        notes.extend(str(x) for x in v)
                    elif isinstance(v, str) and v.strip():
                        notes.append(v)
                return notes
            return []
        except Exception:
            if attempt < 2:
                time.sleep(2 ** attempt)
                continue
            return []
    return []


def _build_reader_input(
    question: str,
    json_rounds: list[dict[str, str]],
    notes: list[str],
) -> str:
    reader_input = {
        "question": question,
        "retrieved_rounds": json_rounds,
        "reader_notes": notes,
    }
    return json.dumps(reader_input, indent=1, ensure_ascii=False)


# ---------------------------------------------------------------------------
# 1D. Improved general reader prompt
# ---------------------------------------------------------------------------

READER_SYSTEM_PROMPT = (
    "You are answering a question about a user based on their past conversation history. "
    "You receive structured evidence (retrieved conversation excerpts) and reading notes.\n\n"
    "Guidelines:\n"
    "- Answer from the provided evidence only.\n"
    "- If the question asks about preferences or recommendations, return the user's "
    "stated preferences and constraints from evidence rather than giving generic advice.\n"
    "- If the question asks about order, time, latest, earliest, or duration, "
    "explicitly use timestamps and chronology from the evidence.\n"
    "- If the question requires synthesis across multiple excerpts, systematically "
    "consider all provided excerpts before answering.\n"
    "- If multiple candidate values appear for the same fact, compare them and select "
    "the most supported or most recent one.\n"
    "- Only say \"I don't know\" if the evidence truly contains no relevant information.\n"
    "- Be concise."
)


def _read_answer_stage1(
    *,
    client: Any,
    model: str,
    reader_input_json: str,
    max_tokens: int = 256,
) -> str:
    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": READER_SYSTEM_PROMPT},
                    {"role": "user", "content": reader_input_json},
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


# ---------------------------------------------------------------------------
# Judge (reuse official logic)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Evidence-path analysis helpers
# ---------------------------------------------------------------------------

def _gold_in_context(gold: str, context: str) -> bool:
    """Check if gold answer content is present in the reader context."""
    if not gold or not context:
        return False
    if gold.lower().strip() in context.lower():
        return True
    # Token overlap for longer answers
    gl = gold.lower().strip()
    cl = context.lower()
    _stops = {"a", "an", "the", "and", "or", "of", "in", "to", "for", "is", "was",
              "it", "that", "this", "with", "on", "at", "by", "from", "as", "be",
              "would", "not", "they", "their", "user", "prefer", "may", "also",
              "suggestions", "responses", "those", "related", "can", "have"}
    gold_tokens = set(re.findall(r"[a-z0-9]+", gl)) - _stops
    ctx_tokens = set(re.findall(r"[a-z0-9]+", cl)) - _stops
    if not gold_tokens:
        return False
    return len(gold_tokens & ctx_tokens) / len(gold_tokens) >= 0.45


def _classify_failure_layer(
    gold: str,
    hypothesis: str,
    retrieval_recall: float,
    evidence_ok: bool,
) -> str:
    if retrieval_recall < 1.0:
        # Check if gold sessions were missed
        return "retrieval"
    if not evidence_ok:
        return "evidence_quality"
    # Check if reader got it right but scorer failed
    strict = strict_short_answer_check(gold, hypothesis)
    if strict is True:
        return "scorer"
    gold_norm = _normalize_short(gold)
    if _short_token_count(gold_norm) > 3:
        if gold.lower().strip() in hypothesis.lower():
            return "scorer"
    return "reader"


# ---------------------------------------------------------------------------
# Freeze context
# ---------------------------------------------------------------------------

def freeze_context(data_path: Path, runner_path: Path) -> dict[str, Any]:
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

    reader_prompt_hash = hashlib.sha256(READER_SYSTEM_PROMPT.encode()).hexdigest()[:16]
    note_prompt_hash = hashlib.sha256(NOTE_EXTRACTION_SYSTEM.encode()).hexdigest()[:16]

    ctx = {
        "git_hash": git_hash,
        "git_dirty": git_dirty,
        "reader_prompt_hash": reader_prompt_hash,
        "note_prompt_hash": note_prompt_hash,
        "scorer_hash": _sha256_file(ROOT / "eval/longmemeval/official_evaluate_qa.py")[:16],
        "dataset_hash": _sha256_file(ROOT / data_path)[:16],
        "stage1_runner_hash": _sha256_file(ROOT / runner_path)[:16],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    return ctx


# ---------------------------------------------------------------------------
# Main stage 1 runner
# ---------------------------------------------------------------------------

@dataclass
class Stage1CaseResult:
    question_id: str
    question_type: str
    gold: str
    old_hypothesis: str
    new_hypothesis: str
    old_correct: bool
    new_correct: bool
    new_correct_after_scorer_cleanup: bool
    retrieval_recall: float
    reader_input_tokens: int
    evidence_contains_gold: bool
    failure_layer: str
    notes_count: int
    reader_model: str


def run_stage1(
    *,
    data_path: Path,
    scored_path: Path,
    diag_path: Path,
    question_ids: list[str] | None = None,
    reader_purpose: str = "longmemeval_reader",
    note_purpose: str = "longmemeval_reader",
    output_dir: Path = STAGE1_OUTPUT_DIR,
    dry_run: bool = False,
    workers: int = 1,
) -> list[Stage1CaseResult]:
    """Run Stage 1 on specified question IDs (or all failed cases)."""

    raw_by_id = _load_raw_by_id(ROOT / data_path)
    questions = [_question_from_raw(raw_by_id, q) for q in _load_questions(ROOT / data_path)]
    q_by_id = {q.question_id: q for q in questions}

    scored = [json.loads(line) for line in (ROOT / scored_path).read_text("utf-8").splitlines() if line.strip()]

    diag_data = json.loads((ROOT / diag_path).read_text("utf-8"))
    diag_by_id = {r["question_id"]: r for r in diag_data}

    if question_ids is None:
        question_ids = [r["question_id"] for r in scored if not r["autoeval_label"]["label"]]

    # Cost estimate
    note_cost = len(question_ids) * 0.003  # ~12K input tokens at $0.15/M + 512 output
    reader_cost = len(question_ids) * 0.005  # ~16K input tokens at $0.15/M + 256 output
    judge_cost = len(question_ids) * 0.001
    total_cost = note_cost + reader_cost + judge_cost

    print(f"Stage 1 run: {len(question_ids)} cases")
    print(f"Estimated cost: ${total_cost:.2f} (notes ${note_cost:.2f} + reader ${reader_cost:.2f} + judge ${judge_cost:.2f})")

    if dry_run:
        print("[DRY RUN] Exiting before API calls.")
        return []

    # Initialize clients
    note_client, note_model = make_openai_client(note_purpose)  # type: ignore[arg-type]
    reader_client, reader_model = make_openai_client(reader_purpose)  # type: ignore[arg-type]
    judge_client, judge_model = make_openai_client("longmemeval_judge")

    print(f"Note model: {note_model}")
    print(f"Reader model: {reader_model}")
    print(f"Judge model: {judge_model}")

    out_dir = ROOT / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    hyp_path = out_dir / "stage1_hypotheses.jsonl"
    detail_path = out_dir / "stage1_details.json"

    results: list[Stage1CaseResult] = []
    details: list[dict[str, Any]] = []

    # Resume support
    completed_ids: set[str] = set()
    if hyp_path.exists():
        for line in hyp_path.read_text("utf-8").splitlines():
            if line.strip():
                completed_ids.add(json.loads(line)["question_id"])
    if detail_path.exists():
        try:
            details = json.loads(detail_path.read_text("utf-8"))
        except json.JSONDecodeError:
            details = []

    remaining = [qid for qid in question_ids if qid not in completed_ids]
    if completed_ids:
        print(f"Resuming: {len(completed_ids)} done, {len(remaining)} remaining")
    print(f"Workers: {workers}")

    def _process_one(qid: str) -> tuple[Stage1CaseResult, dict[str, Any], str, list[str]]:
        t0 = time.monotonic()
        q = q_by_id[qid]
        diag = diag_by_id[qid]
        old_hyp = diag["hypothesis"]
        gold = diag["gold_answer"]
        qtype = diag["question_type"]
        recall = diag["retrieval_recall"]

        rounds = _build_rounds(q)
        retrieved = _retrieve_tfidf(q.question, rounds, top_k=TOP_K)
        retrieved = _sort_chronological(retrieved)
        json_rounds = _format_rounds_json(retrieved)

        notes = _extract_reading_notes(
            client=note_client, model=note_model,
            question=q.question, json_rounds=json_rounds,
        )

        reader_input = _build_reader_input(q.question, json_rounds, notes)
        input_tokens = _estimate_tokens(reader_input)

        new_hyp = _read_answer_stage1(
            client=reader_client, model=reader_model,
            reader_input_json=reader_input,
        )

        new_correct = _judge_one(
            client=judge_client, model=judge_model,
            question_id=qid, question=q.question,
            gold=gold, hypothesis=new_hyp, question_type=qtype,
        )

        strict_check = strict_short_answer_check(gold, new_hyp)
        new_correct_cleaned = strict_check if strict_check is not None else new_correct

        context_text = "\n".join(r["content"] for r in json_rounds)
        evidence_ok = _gold_in_context(gold, context_text)
        layer = _classify_failure_layer(gold, new_hyp, recall, evidence_ok)

        elapsed = time.monotonic() - t0
        result = Stage1CaseResult(
            question_id=qid, question_type=qtype, gold=gold,
            old_hypothesis=old_hyp, new_hypothesis=new_hyp,
            old_correct=False, new_correct=new_correct,
            new_correct_after_scorer_cleanup=new_correct_cleaned,
            retrieval_recall=recall, reader_input_tokens=input_tokens,
            evidence_contains_gold=evidence_ok, failure_layer=layer,
            notes_count=len(notes), reader_model=reader_model,
        )
        detail = {
            **{k: getattr(result, k) for k in result.__dataclass_fields__},
            "notes": notes, "elapsed_s": round(elapsed, 2),
        }
        hyp_line = json.dumps({"question_id": qid, "hypothesis": new_hyp}, ensure_ascii=False) + "\n"
        return result, detail, hyp_line, [qid, qtype, str(new_correct_cleaned), f"{elapsed:.1f}s"]

    write_lock = threading.Lock()
    done_count = [len(completed_ids)]

    with hyp_path.open("a" if completed_ids else "w", encoding="utf-8") as hyp_f:
        if workers <= 1:
            for qid in remaining:
                result, detail, hyp_line, info = _process_one(qid)
                done_count[0] += 1
                status = "PASS" if result.new_correct_after_scorer_cleanup else "FAIL"
                print(f"  [{done_count[0]}/{len(question_ids)}] {info[0]} ({info[1]}) -> {status} ({info[3]})")
                hyp_f.write(hyp_line)
                hyp_f.flush()
                results.append(result)
                details.append(detail)
                detail_path.write_text(json.dumps(details, indent=2, ensure_ascii=False), encoding="utf-8")
        else:
            futures = {}
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for qid in remaining:
                    futures[pool.submit(_process_one, qid)] = qid
                for fut in as_completed(futures):
                    try:
                        result, detail, hyp_line, info = fut.result()
                    except Exception as exc:
                        qid = futures[fut]
                        print(f"  ERROR {qid}: {exc}")
                        continue
                    with write_lock:
                        done_count[0] += 1
                        status = "PASS" if result.new_correct_after_scorer_cleanup else "FAIL"
                        print(f"  [{done_count[0]}/{len(question_ids)}] {info[0]} ({info[1]}) -> {status} ({info[3]})")
                        hyp_f.write(hyp_line)
                        hyp_f.flush()
                        results.append(result)
                        details.append(detail)
                        detail_path.write_text(json.dumps(details, indent=2, ensure_ascii=False), encoding="utf-8")

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report_stage1(
    results: list[Stage1CaseResult],
    context: dict[str, Any],
) -> None:
    if not results:
        print("No results to report.")
        return

    total = len(results)
    flips = sum(1 for r in results if r.new_correct_after_scorer_cleanup and not r.old_correct)
    regressions = sum(1 for r in results if not r.new_correct_after_scorer_cleanup and r.old_correct)
    still_fail = total - flips - regressions

    print(f"\n{'=' * 60}")
    print("  STAGE 1 — Failed-Case Rerun Results")
    print(f"{'=' * 60}")
    print(f"\nRun context:")
    for k, v in context.items():
        print(f"  {k}: {v}")

    print(f"\nTotal cases rerun: {total}")
    print(f"Flips (fail -> pass): {flips} ({100*flips/total:.1f}%)")
    print(f"Regressions (pass -> fail): {regressions}")
    print(f"Still failing: {still_fail}")
    print(f"Estimated new overall: {(329 + flips) / 500:.4f} ({(329 + flips)}/500)")

    # Delta by question type
    print(f"\nDelta by question type:")
    type_flips: dict[str, list[int]] = defaultdict(list)
    for r in results:
        type_flips[r.question_type].append(1 if r.new_correct_after_scorer_cleanup else 0)
    for qtype in sorted(type_flips):
        f = sum(type_flips[qtype])
        t = len(type_flips[qtype])
        print(f"  {qtype}: {f}/{t} flipped ({100*f/t:.0f}%)")

    # Delta by failure layer
    print(f"\nDelta by failure layer (still-failing cases):")
    layer_counts: dict[str, int] = Counter()
    for r in results:
        if not r.new_correct_after_scorer_cleanup:
            layer_counts[r.failure_layer] += 1
    for layer, count in sorted(layer_counts.items(), key=lambda x: -x[1]):
        print(f"  {layer}: {count}")

    # Scorer cleanup impact
    judge_only = sum(1 for r in results if r.new_correct)
    cleaned = sum(1 for r in results if r.new_correct_after_scorer_cleanup)
    print(f"\nScorer cleanup impact: judge={judge_only} correct, after cleanup={cleaned} correct (delta={cleaned-judge_only})")

    # Decision rule
    flip_rate = flips / total if total else 0
    if flip_rate >= 0.30:
        print(f"\n>>> Flip rate {flip_rate:.1%} >= 30%: FULL 500 RERUN JUSTIFIED <<<")
    elif flip_rate < 0.15:
        print(f"\n>>> Flip rate {flip_rate:.1%} < 15%: STAGE 1 INSUFFICIENT <<<")
    else:
        print(f"\n>>> Flip rate {flip_rate:.1%}: moderate improvement, consider full rerun <<<")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1 LongMemEval improvements")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--scored", type=Path, default=BASELINE_SCORED)
    parser.add_argument("--diag", type=Path, default=BASELINE_DIAG)
    parser.add_argument("--output-dir", type=Path, default=STAGE1_OUTPUT_DIR)
    parser.add_argument("--reader", default="longmemeval_reader",
                        choices=["longmemeval_reader", "reader_gpt41"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--test-scorer", action="store_true")
    parser.add_argument("--question-ids", nargs="*", default=None)
    parser.add_argument("--all-500", action="store_true",
                        help="Run all 500 questions, not just failed cases")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers for API calls (default 1)")
    args = parser.parse_args()

    if args.test_scorer:
        print("Running scorer cleanup tests...")
        test_scorer_cleanup()
        return

    # Phase 0: Freeze
    print("=" * 60)
    print("  PHASE 0: Freeze Context")
    print("=" * 60)
    ctx = freeze_context(args.data, Path("eval/longmemeval/stage1_runner.py"))
    ctx["reader_purpose"] = args.reader
    for k, v in ctx.items():
        print(f"  {k}: {v}")

    # Determine question IDs
    qids = args.question_ids
    if args.all_500:
        ref = json.loads((ROOT / args.data).read_text("utf-8"))
        qids = [str(r["question_id"]) for r in ref]
        print(f"\nRunning all {len(qids)} questions")
    elif qids is None:
        print("\nRunning failed cases only")

    results = run_stage1(
        data_path=args.data,
        scored_path=args.scored,
        diag_path=args.diag,
        question_ids=qids,
        reader_purpose=args.reader,
        note_purpose=args.reader,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        workers=args.workers,
    )

    report_stage1(results, ctx)


if __name__ == "__main__":
    main()
