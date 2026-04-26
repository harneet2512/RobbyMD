"""Stage 2: substrate pipeline baseline on a 10-task stratified slice.

Compares round-retrieval (Stage 1) vs substrate claim-extraction pipeline
on the same 10 cases to estimate the value of the bigger architectural jump.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval._openai_client import make_openai_client
from eval.longmemeval.pipeline import run_substrate_case
from eval.longmemeval.round_retrieval_run import (
    _build_rounds,
    _estimate_tokens,
    _load_questions,
    _load_raw_by_id,
    _question_from_raw,
    _retrieve_tfidf,
)
from eval.longmemeval.stage1_runner import (  # type: ignore[import-not-found]
    _build_reader_input,
    _extract_reading_notes,
    _format_rounds_json,
    _judge_one,
    _read_answer_stage1,
    _sort_chronological,
)

DEFAULT_DATA = Path("eval/longmemeval/data/longmemeval_s_cleaned.json")
BASELINE_SCORED = Path("eval/longmemeval/results/round_full_500_scored.jsonl")
BASELINE_DIAG = Path("eval/longmemeval/results/round_full_500_diagnostics.json")
STAGE2_OUTPUT_DIR = Path("eval/longmemeval/results/stage2")


# ---------------------------------------------------------------------------
# 2A. Build 10-task stratified slice
# ---------------------------------------------------------------------------

def select_slice(
    _data_path: Path,
    scored_path: Path,
    diag_path: Path,
) -> list[str]:
    """Select 10 cases balanced toward weak categories.

    Target: 2 preference, 2 multi-session, 2 temporal, 2 knowledge-update,
    2 difficult exact-value/synthesis from baseline failures.
    """
    scored = [json.loads(line) for line in (ROOT / scored_path).read_text("utf-8").splitlines() if line.strip()]
    failed_ids = {r["question_id"] for r in scored if not r["autoeval_label"]["label"]}

    diag_data = json.loads((ROOT / diag_path).read_text("utf-8"))
    diag_by_id = {r["question_id"]: r for r in diag_data}

    by_type: dict[str, list[str]] = {}
    for r in diag_data:
        qtype = r["question_type"]
        by_type.setdefault(qtype, [])
        if r["question_id"] in failed_ids:
            by_type[qtype].append(r["question_id"])

    selected: list[str] = []

    # 2 preference (all failed, pick first 2)
    pref = by_type.get("single-session-preference", [])
    selected.extend(pref[:2])

    # 2 multi-session (pick first 2 failed)
    multi = by_type.get("multi-session", [])
    selected.extend(multi[:2])

    # 2 temporal (pick first 2 failed)
    temporal = by_type.get("temporal-reasoning", [])
    selected.extend(temporal[:2])

    # 2 knowledge-update (pick first 2 failed)
    ku = by_type.get("knowledge-update", [])
    selected.extend(ku[:2])

    # 2 difficult exact-value cases: short gold answers that failed
    short_failures = []
    for qid in failed_ids:
        diag = diag_by_id.get(qid, {})
        gold = diag.get("gold_answer", "")
        if len(gold.split()) <= 3 and qid not in selected:
            short_failures.append(qid)
    selected.extend(short_failures[:2])

    return selected[:10]


# ---------------------------------------------------------------------------
# 2B. Run substrate pipeline on the slice
# ---------------------------------------------------------------------------

@dataclass
class SliceCaseResult:
    question_id: str
    question_type: str
    gold: str
    # Round-retrieval results (from Stage 1 or baseline)
    round_hypothesis: str
    round_correct: bool
    round_tokens: int
    # Substrate results
    substrate_hypothesis: str
    substrate_correct: bool
    substrate_tokens: int
    substrate_direct_count: int
    substrate_supporting_count: int
    substrate_conflict_count: int
    substrate_sufficiency: str
    substrate_bundle_text_preview: str


def run_substrate_slice(
    *,
    question_ids: list[str],
    data_path: Path,
    diag_path: Path,
    reader_purpose: str = "longmemeval_reader",
    output_dir: Path = STAGE2_OUTPUT_DIR,
    dry_run: bool = False,
) -> list[SliceCaseResult]:
    """Run both round-retrieval (Stage 1 format) and substrate on the same slice."""

    raw_by_id = _load_raw_by_id(ROOT / data_path)
    questions = [_question_from_raw(raw_by_id, q) for q in _load_questions(ROOT / data_path)]
    q_by_id = {q.question_id: q for q in questions}

    diag_data = json.loads((ROOT / diag_path).read_text("utf-8"))
    diag_by_id = {r["question_id"]: r for r in diag_data}

    # Cost estimate: 10 cases × (claim extraction + reader + judge) × 2 paths
    # Substrate: claim extraction ~$0.04/case, reader ~$0.005, judge ~$0.001
    # Round-retrieval with CoN: note extraction ~$0.003, reader ~$0.005, judge ~$0.001
    total_est = len(question_ids) * (0.04 + 0.005 + 0.001 + 0.003 + 0.005 + 0.001)
    print(f"Stage 2 slice: {len(question_ids)} cases")
    print(f"Estimated cost: ${total_est:.2f}")

    if dry_run:
        print("[DRY RUN] Exiting.")
        return []

    reader_client, reader_model = make_openai_client(reader_purpose)  # type: ignore[arg-type]
    judge_client, judge_model = make_openai_client("longmemeval_judge")

    print(f"Reader model: {reader_model}")
    print(f"Judge model: {judge_model}")

    out_dir = ROOT / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[SliceCaseResult] = []

    for idx, qid in enumerate(question_ids, 1):
        q = q_by_id[qid]
        diag = diag_by_id[qid]
        gold = diag["gold_answer"]
        qtype = diag["question_type"]

        print(f"\n[{idx}/{len(question_ids)}] {qid} ({qtype})")

        # ── PATH A: Round-retrieval with Stage 1 improvements ──
        print(f"  Round-retrieval...", end="", flush=True)
        t0 = time.monotonic()

        rounds = _build_rounds(q)
        retrieved = _retrieve_tfidf(q.question, rounds, top_k=30)
        retrieved = _sort_chronological(retrieved)
        json_rounds = _format_rounds_json(retrieved)

        notes = _extract_reading_notes(
            client=reader_client,
            model=reader_model,
            question=q.question,
            json_rounds=json_rounds,
        )
        reader_input = _build_reader_input(q.question, json_rounds, notes)
        round_hyp = _read_answer_stage1(
            client=reader_client,
            model=reader_model,
            reader_input_json=reader_input,
        )
        round_correct = _judge_one(
            client=judge_client,
            model=judge_model,
            question_id=qid,
            question=q.question,
            gold=gold,
            hypothesis=round_hyp,
            question_type=qtype,
        )
        round_tokens = _estimate_tokens(reader_input)
        print(f" {'PASS' if round_correct else 'FAIL'} ({time.monotonic()-t0:.0f}s)")

        # ── PATH B: Substrate pipeline ──
        print(f"  Substrate...", end="", flush=True)
        t0 = time.monotonic()

        def make_reader(client: Any, model: str):
            def reader_fn(system: str, prompt: str) -> str:
                for attempt in range(3):
                    try:
                        resp = client.chat.completions.create(
                            model=model,
                            messages=[
                                {"role": "system", "content": system},
                                {"role": "user", "content": prompt},
                            ],
                            temperature=0.0,
                            max_tokens=256,
                        )
                        return (resp.choices[0].message.content or "").strip()
                    except Exception:
                        if attempt < 2:
                            time.sleep(2 ** attempt)
                            continue
                        return "I don't know."
                return "I don't know."
            return reader_fn

        try:
            answer, trace = run_substrate_case(
                q,
                reader_fn=make_reader(reader_client, reader_model),
            )
        except Exception as exc:
            print(f" ERROR: {exc}")
            answer = f"[ERROR] {exc}"
            trace = None

        if trace:
            substrate_hyp = answer
            substrate_correct = _judge_one(
                client=judge_client,
                model=judge_model,
                question_id=qid,
                question=q.question,
                gold=gold,
                hypothesis=substrate_hyp,
                question_type=qtype,
            )
            substrate_tokens = trace.bundle_tokens
            direct = trace.verified_direct
            supporting = trace.verified_supporting
            conflict = trace.verified_conflict
            sufficiency = trace.sufficiency
            bundle_preview = (trace.final_bundle_text or "")[:200]
        else:
            substrate_hyp = answer
            substrate_correct = False
            substrate_tokens = 0
            direct = 0
            supporting = 0
            conflict = 0
            sufficiency = "error"
            bundle_preview = ""

        print(f" {'PASS' if substrate_correct else 'FAIL'} ({time.monotonic()-t0:.0f}s)")
        print(f"    Round: {round_hyp[:80]}")
        print(f"    Substrate: {substrate_hyp[:80]}")
        print(f"    Gold: {gold[:80]}")
        print(f"    DIRECT={direct} SUPPORTING={supporting} CONFLICT={conflict} sufficiency={sufficiency}")

        results.append(SliceCaseResult(
            question_id=qid,
            question_type=qtype,
            gold=gold,
            round_hypothesis=round_hyp,
            round_correct=round_correct,
            round_tokens=round_tokens,
            substrate_hypothesis=substrate_hyp,
            substrate_correct=substrate_correct,
            substrate_tokens=substrate_tokens,
            substrate_direct_count=direct,
            substrate_supporting_count=supporting,
            substrate_conflict_count=conflict,
            substrate_sufficiency=sufficiency,
            substrate_bundle_text_preview=bundle_preview,
        ))

    # Save results
    detail_path = out_dir / "stage2_slice_results.json"
    import dataclasses
    detail_path.write_text(
        json.dumps([dataclasses.asdict(r) for r in results], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nSaved to {detail_path}")

    return results


# ---------------------------------------------------------------------------
# 2C. Comparison report
# ---------------------------------------------------------------------------

def report_comparison(results: list[SliceCaseResult]) -> None:
    if not results:
        return

    print(f"\n{'=' * 60}")
    print("  STAGE 2 — Round-Retrieval vs Substrate (10-task slice)")
    print(f"{'=' * 60}")

    round_correct = sum(1 for r in results if r.round_correct)
    substrate_correct = sum(1 for r in results if r.substrate_correct)
    both_correct = sum(1 for r in results if r.round_correct and r.substrate_correct)
    round_only = sum(1 for r in results if r.round_correct and not r.substrate_correct)
    substrate_only = sum(1 for r in results if not r.round_correct and r.substrate_correct)
    neither = sum(1 for r in results if not r.round_correct and not r.substrate_correct)

    print(f"\nOverall: Round={round_correct}/{len(results)}, Substrate={substrate_correct}/{len(results)}")
    print(f"  Both correct:    {both_correct}")
    print(f"  Round only:      {round_only}")
    print(f"  Substrate only:  {substrate_only}")
    print(f"  Neither:         {neither}")

    print(f"\nPer-case comparison:")
    for r in results:
        rc = "PASS" if r.round_correct else "FAIL"
        sc = "PASS" if r.substrate_correct else "FAIL"
        winner = "TIE" if r.round_correct == r.substrate_correct else ("ROUND" if r.round_correct else "SUBSTRATE")
        print(f"  {r.question_id} ({r.question_type}): round={rc} substrate={sc} -> {winner}")
        print(f"    D={r.substrate_direct_count} S={r.substrate_supporting_count} C={r.substrate_conflict_count}")

    # Token comparison
    avg_round = sum(r.round_tokens for r in results) / len(results)
    avg_substrate = sum(r.substrate_tokens for r in results) / len(results)
    print(f"\nToken comparison: round avg={avg_round:.0f}, substrate avg={avg_substrate:.0f}")

    # By question type
    print(f"\nBy question type:")
    types = sorted(set(r.question_type for r in results))
    for qt in types:
        qt_cases = [r for r in results if r.question_type == qt]
        r_ok = sum(1 for r in qt_cases if r.round_correct)
        s_ok = sum(1 for r in qt_cases if r.substrate_correct)
        print(f"  {qt}: round={r_ok}/{len(qt_cases)} substrate={s_ok}/{len(qt_cases)}")

    # Decision
    if substrate_correct > round_correct + 1:
        print(f"\n>>> Substrate wins clearly ({substrate_correct} vs {round_correct}): recommend broader rollout <<<")
    elif round_correct > substrate_correct + 1:
        print(f"\n>>> Round-retrieval wins ({round_correct} vs {substrate_correct}): keep Stage 1 path <<<")
    else:
        print(f"\n>>> Results are close ({round_correct} vs {substrate_correct}): more data needed <<<")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2 substrate slice comparison")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--scored", type=Path, default=BASELINE_SCORED)
    parser.add_argument("--diag", type=Path, default=BASELINE_DIAG)
    parser.add_argument("--output-dir", type=Path, default=STAGE2_OUTPUT_DIR)
    parser.add_argument("--reader", default="longmemeval_reader")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--question-ids", nargs="*", default=None)
    args = parser.parse_args()

    # Select slice
    if args.question_ids:
        question_ids = args.question_ids
    else:
        question_ids = select_slice(args.data, args.scored, args.diag)

    print(f"Selected slice ({len(question_ids)} cases):")
    diag_data = json.loads((ROOT / args.diag).read_text("utf-8"))
    diag_by_id = {r["question_id"]: r for r in diag_data}
    for qid in question_ids:
        diag = diag_by_id.get(qid, {})
        print(f"  {qid} ({diag.get('question_type', '?')}): {diag.get('question', '?')[:60]}")

    results = run_substrate_slice(
        question_ids=question_ids,
        data_path=args.data,
        diag_path=args.diag,
        reader_purpose=args.reader,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
    )

    report_comparison(results)


if __name__ == "__main__":
    main()
