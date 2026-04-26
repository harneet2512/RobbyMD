"""Protocol-compliant LongMemEval-S raw-round retrieval runner.

This runner intentionally bypasses the substrate claim-extraction pipeline.
It follows the LongMemEval paper's retrieval recommendation more closely:
retrieve raw conversation rounds and pass raw text to the reader.

Output is official LongMemEval JSONL:
  {"question_id": "...", "hypothesis": "..."}
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval._openai_client import make_openai_client
from eval.longmemeval.adapter import LongMemEvalQuestion, iter_questions


DEFAULT_DATA = Path("eval/longmemeval/data/longmemeval_s_cleaned.json")
DEFAULT_OUTPUT_DIR = Path("eval/longmemeval/results")
QUESTION_TYPE_ORDER = (
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "multi-session",
    "temporal-reasoning",
    "knowledge-update",
    "abstention",
)
EMBEDDING_MODEL_NAME = "sklearn-tfidf-word-1-2"


@dataclass(frozen=True)
class RoundRecord:
    key: str
    value: str
    session_id: str
    round_index: int
    session_order: int
    timestamp: str


def _stratum(q: LongMemEvalQuestion) -> str:
    return "abstention" if "_abs" in q.question_id else q.question_type


def _load_questions(path: Path) -> list[LongMemEvalQuestion]:
    questions = list(iter_questions(path))
    if not questions:
        raise RuntimeError(f"No LongMemEval questions loaded from {path}")
    return questions


def _stratified_sample(
    questions: list[LongMemEvalQuestion], *, n: int, seed: int
) -> list[LongMemEvalQuestion]:
    if n <= 0:
        raise ValueError("--n must be positive")
    if n > len(questions):
        raise ValueError(f"--n {n} exceeds dataset size {len(questions)}")

    rng = random.Random(seed)
    by_type: dict[str, list[LongMemEvalQuestion]] = defaultdict(list)
    for q in questions:
        by_type[_stratum(q)].append(q)
    for items in by_type.values():
        rng.shuffle(items)

    labels = [label for label in QUESTION_TYPE_ORDER if by_type.get(label)]
    if n < len(labels):
        labels = labels[:n]

    target_counts = {label: 1 for label in labels}
    remaining = n - len(labels)
    total_available = sum(len(by_type[label]) for label in labels)
    fractional: list[tuple[float, str]] = []
    for label in labels:
        ideal = remaining * (len(by_type[label]) / total_available) if total_available else 0
        whole = int(ideal)
        target_counts[label] += whole
        fractional.append((ideal - whole, label))
    left = n - sum(target_counts.values())
    for _, label in sorted(fractional, reverse=True):
        if left <= 0:
            break
        target_counts[label] += 1
        left -= 1

    selected: list[LongMemEvalQuestion] = []
    for label in labels:
        selected.extend(by_type[label][: target_counts[label]])
    rng.shuffle(selected)
    return selected


def _session_id_for(q: LongMemEvalQuestion, idx: int) -> str:
    if q.haystack_session_ids and idx < len(q.haystack_session_ids):
        return str(q.haystack_session_ids[idx])
    return f"{q.question_id}::session::{idx:03d}"


def _session_timestamp_for(q: LongMemEvalQuestion, idx: int) -> str:
    if q.haystack_dates and idx < len(q.haystack_dates):
        return str(q.haystack_dates[idx])
    return f"session_order_{idx:04d}"


def _format_message(msg: dict[str, str]) -> str:
    role = str(msg.get("role", "unknown")).strip() or "unknown"
    content = str(msg.get("content", "")).strip()
    return f"{role}: {content}"


def _rounds_for_session(
    q: LongMemEvalQuestion, session_idx: int
) -> list[RoundRecord]:
    session = q.haystack_sessions[session_idx]
    session_id = _session_id_for(q, session_idx)
    timestamp = _session_timestamp_for(q, session_idx)
    rounds: list[RoundRecord] = []
    i = 0
    round_index = 0
    while i < len(session):
        msg = session[i]
        role = str(msg.get("role", "")).lower()
        parts = [_format_message(msg)]
        if role == "user" and i + 1 < len(session):
            next_role = str(session[i + 1].get("role", "")).lower()
            if next_role == "assistant":
                parts.append(_format_message(session[i + 1]))
                i += 1
        text = "\n".join(p for p in parts if p.strip())
        if text.strip():
            value = (
                f"[timestamp={timestamp}] [session_id={session_id}] "
                f"[round={round_index}]\n{text}"
            )
            rounds.append(
                RoundRecord(
                    key=text,
                    value=value,
                    session_id=session_id,
                    round_index=round_index,
                    session_order=session_idx,
                    timestamp=timestamp,
                )
            )
            round_index += 1
        i += 1
    return rounds


def _build_rounds(q: LongMemEvalQuestion) -> list[RoundRecord]:
    rounds: list[RoundRecord] = []
    for sidx in range(len(q.haystack_sessions)):
        rounds.extend(_rounds_for_session(q, sidx))
    return rounds


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _retrieve_tfidf(
    question: str, rounds: list[RoundRecord], *, top_k: int
) -> list[RoundRecord]:
    if not rounds:
        return []
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        stop_words="english",
        max_features=50000,
    )
    matrix = vectorizer.fit_transform([r.key for r in rounds])
    query_vec = vectorizer.transform([question])
    scores = cosine_similarity(query_vec, matrix).ravel()
    ranked = sorted(
        range(len(rounds)),
        key=lambda idx: (-float(scores[idx]), rounds[idx].session_order, rounds[idx].round_index),
    )
    selected = [rounds[idx] for idx in ranked[: max(1, top_k)]]
    return sorted(selected, key=lambda r: (r.session_order, r.round_index))


def _make_reader(reader: str) -> tuple[Any, str]:
    if reader == "gpt-4o-mini":
        purpose = "llm_medcon_gpt4omini"
    elif reader == "gpt-4o":
        purpose = "longmemeval_reader"
    else:
        raise ValueError(f"Unsupported --reader {reader!r}")
    return make_openai_client(purpose)  # type: ignore[arg-type]


def _read_answer(
    *,
    client: Any,
    model: str,
    question: str,
    context: str,
    max_tokens: int,
) -> str:
    system = (
        "You are a helpful assistant answering questions based on past "
        "conversation history. Use the provided conversation excerpts to answer. "
        "If the answer cannot be determined from the excerpts, say \"I don't know.\""
    )
    user = (
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
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
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


def _retrieval_recall(q: LongMemEvalQuestion, sessions_hit: set[str]) -> tuple[list[str], float]:
    gold_ids = [str(x) for x in getattr(q, "answer_session_ids", []) or []]
    if not gold_ids:
        return gold_ids, 1.0
    hit = sum(1 for sid in gold_ids if sid in sessions_hit)
    return gold_ids, hit / len(gold_ids)


def _question_from_raw(raw_by_id: dict[str, dict[str, Any]], q: LongMemEvalQuestion) -> LongMemEvalQuestion:
    # iter_questions keeps only core fields; answer_session_ids is needed for diagnostics.
    raw = raw_by_id.get(q.question_id, {})
    object.__setattr__(q, "answer_session_ids", raw.get("answer_session_ids", []))
    return q


def _load_raw_by_id(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(obj["question_id"]): obj for obj in data}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LongMemEval-S raw-round retrieval predictions")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--reader", choices=("gpt-4o-mini", "gpt-4o"), default="gpt-4o-mini")
    parser.add_argument("--max-answer-tokens", type=int, default=256)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--diagnostics", type=Path, default=None)
    parser.add_argument("--all-questions", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    if os.environ.get("ACTIVE_PACK") != "personal_assistant":
        raise RuntimeError("Set ACTIVE_PACK=personal_assistant before running LongMemEval.")

    raw_by_id = _load_raw_by_id(args.data)
    questions = [_question_from_raw(raw_by_id, q) for q in _load_questions(args.data)]
    selected = questions if args.all_questions or args.n >= len(questions) else _stratified_sample(questions, n=args.n, seed=args.seed)

    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = args.output or DEFAULT_OUTPUT_DIR / f"round_smoke_{len(selected)}.jsonl"
    diagnostics = args.diagnostics or output.with_name(output.stem + "_diagnostics.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    diagnostics.parent.mkdir(parents=True, exist_ok=True)

    completed_ids: set[str] = set()
    diag_records: list[dict[str, Any]] = []
    if not args.no_resume and output.exists():
        for line in output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                completed_ids.add(str(json.loads(line)["question_id"]))
    if not args.no_resume and diagnostics.exists():
        try:
            diag_records = json.loads(diagnostics.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            diag_records = []

    client, reader_model = _make_reader(args.reader)
    remaining = [q for q in selected if q.question_id not in completed_ids]
    mode = "a" if completed_ids and not args.no_resume else "w"

    print("LongMemEval raw-round retrieval run")
    print(f"data={args.data}")
    print(f"n={len(selected)} seed={args.seed} strata={dict(Counter(_stratum(q) for q in selected))}")
    print(f"top_k={args.top_k} embedding_model={EMBEDDING_MODEL_NAME}")
    print(f"reader={args.reader} resolved_reader_model={reader_model}")
    print(f"output={output}")
    print(f"diagnostics={diagnostics}")
    if completed_ids:
        print(f"resuming: {len(completed_ids)} completed, {len(remaining)} remaining")

    with output.open(mode, encoding="utf-8") as out_f:
        for idx, q in enumerate(remaining, start=len(completed_ids) + 1):
            t0 = time.monotonic()
            print(f"[{idx}/{len(selected)}] {q.question_id} {q.question_type}", flush=True)
            rounds = _build_rounds(q)
            retrieved = _retrieve_tfidf(q.question, rounds, top_k=args.top_k)
            context = "\n\n---\n\n".join(r.value for r in retrieved)
            sessions_hit = {r.session_id for r in retrieved}
            gold_session_ids, recall = _retrieval_recall(q, sessions_hit)
            hypothesis = _read_answer(
                client=client,
                model=reader_model,
                question=q.question,
                context=context,
                max_tokens=args.max_answer_tokens,
            )
            elapsed = time.monotonic() - t0
            out_f.write(json.dumps({"question_id": q.question_id, "hypothesis": hypothesis}, ensure_ascii=False) + "\n")
            out_f.flush()
            diag_records.append(
                {
                    "question_id": q.question_id,
                    "question_type": q.question_type,
                    "is_abstention": "_abs" in q.question_id,
                    "question": q.question,
                    "gold_answer": q.answer,
                    "hypothesis": hypothesis,
                    "rounds_indexed": len(rounds),
                    "rounds_retrieved": len(retrieved),
                    "retrieved_tokens": _estimate_tokens(context),
                    "sessions_hit": sorted(sessions_hit),
                    "gold_session_ids": gold_session_ids,
                    "retrieval_recall": recall,
                    "reader_model": reader_model,
                    "embedding_model": EMBEDDING_MODEL_NAME,
                    "top_k": args.top_k,
                    "time_seconds": round(elapsed, 3),
                }
            )
            diagnostics.write_text(json.dumps(diag_records, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {output}")
    print(f"Wrote {diagnostics}")


if __name__ == "__main__":
    main()
