"""Protocol-compliant LongMemEval-S runner.

Input: official longmemeval_s_cleaned.json
Output: official JSONL with {"question_id": "...", "hypothesis": "..."}
Scoring: run eval/longmemeval/official_evaluate_qa.py separately.

Usage:
  python eval/longmemeval/official_run.py --n 20 --seed 42
  python eval/longmemeval/official_run.py --n 500
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval._openai_client import make_openai_client
from eval.longmemeval.adapter import LongMemEvalQuestion, iter_questions, session_to_turns
from eval.longmemeval.context import IngestionStats, make_retrying_longmemeval_extractor
from eval.longmemeval.diagnostic_slice import _gold_in_text
from eval.longmemeval.pipeline import CaseTrace, ReaderFn, run_substrate_case
from eval.longmemeval.question_router import RetrievalStrategy
from src.substrate.claims import list_active_claims
from src.substrate.on_new_turn import ExtractedClaim, on_new_turn
from src.substrate.schema import Speaker, open_database


DEFAULT_DATA = Path("eval/longmemeval/data/longmemeval_s_cleaned.json")
DEFAULT_OUTPUT_DIR = Path("eval/longmemeval/results")
DEFAULT_CACHE_DIR = Path("eval/longmemeval/cache")
QUESTION_TYPE_ORDER = (
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "multi-session",
    "temporal-reasoning",
    "knowledge-update",
    "abstention",
)


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

    selected: list[LongMemEvalQuestion] = []
    target_counts = {label: 0 for label in labels}

    # Guarantee coverage first, then fill proportionally by available count.
    for label in labels:
        target_counts[label] = 1
    remaining = n - sum(target_counts.values())
    total_available = sum(len(by_type[label]) for label in labels)
    fractional: list[tuple[float, str]] = []
    for label in labels:
        ideal_extra = remaining * (len(by_type[label]) / total_available) if total_available else 0
        whole = int(ideal_extra)
        target_counts[label] += whole
        fractional.append((ideal_extra - whole, label))
    left = n - sum(target_counts.values())
    for _, label in sorted(fractional, reverse=True):
        if left <= 0:
            break
        target_counts[label] += 1
        left -= 1

    for label in labels:
        selected.extend(by_type[label][: target_counts[label]])
    rng.shuffle(selected)
    return selected


def _make_reader(reader: str) -> tuple[ReaderFn, str]:
    if reader == "gpt-4o-mini":
        purpose = "llm_medcon_gpt4omini"
    elif reader == "gpt-4o":
        purpose = "longmemeval_reader"
    else:
        raise ValueError(f"Unsupported --reader {reader!r}; use gpt-4o-mini or gpt-4o")

    client, model = make_openai_client(purpose)  # type: ignore[arg-type]

    def _reader(system: str, user: str) -> str:
        for attempt in range(5):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=0.0,
                    max_tokens=512,
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as exc:
                if "429" in str(exc) and attempt < 4:
                    time.sleep(2 ** attempt)
                    continue
                raise

    return _reader, str(model)


def _embedding_client(name: str) -> Any:
    if name == "default":
        return None
    if name == "mock":
        from tests.e2e.conftest import MockEmbeddingClient

        return MockEmbeddingClient()
    raise ValueError(f"Unsupported --embedding {name!r}; use default or mock")


def _safe_session_cache_path(cache_dir: Path, session_id: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", session_id)[:96].strip("._") or "session"
    digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:12]
    return cache_dir / f"session_claims_{safe}_{digest}.json"


def _claim_to_cache_dict(claim: Any) -> dict[str, Any]:
    return {
        "subject": claim.subject,
        "predicate": claim.predicate,
        "value": claim.value,
        "confidence": claim.confidence,
        "value_normalised": claim.value_normalised,
        "char_start": claim.char_start,
        "char_end": claim.char_end,
    }


def _claim_from_cache_dict(obj: dict[str, Any]) -> ExtractedClaim:
    return ExtractedClaim(
        subject=str(obj.get("subject", "user")),
        predicate=str(obj.get("predicate", "user_fact")),
        value=str(obj.get("value", "")),
        confidence=float(obj.get("confidence", 0.8)),
        value_normalised=obj.get("value_normalised"),
        char_start=obj.get("char_start"),
        char_end=obj.get("char_end"),
    )


def _session_index(questions: list[LongMemEvalQuestion]) -> dict[str, list[dict[str, str]]]:
    sessions: dict[str, list[dict[str, str]]] = {}
    for q in questions:
        ids = q.haystack_session_ids or [f"{q.question_id}::session::{i:03d}" for i in range(len(q.haystack_sessions))]
        for sid, transcript in zip(ids, q.haystack_sessions, strict=False):
            sessions.setdefault(str(sid), transcript)
    return sessions


def _build_one_session_cache(
    session_id: str,
    transcript: list[dict[str, str]],
    *,
    cache_dir: Path,
) -> dict[str, Any]:
    path = _safe_session_cache_path(cache_dir, session_id)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            path.unlink(missing_ok=True)

    conn = open_database(":memory:")
    q = LongMemEvalQuestion(
        question_id=session_id,
        question="",
        answer="",
        question_type="single-session-user",
        haystack_sessions=[transcript],
        haystack_session_ids=[session_id],
    )

    def _active_claims_summary() -> str:
        active = list_active_claims(conn, session_id)
        if not active:
            return "(no active claims yet)"
        lines = [f"{c.claim_id}: {c.subject}.{c.predicate} = {c.value!r} (confidence {c.confidence})" for c in active[-5:]]
        summary = "; ".join(lines)
        if len(active) > 5:
            summary += f" ... ({len(active) - 5} more)"
        return summary

    extractor = make_retrying_longmemeval_extractor(active_claims_fn=_active_claims_summary)
    turns: list[dict[str, Any]] = []
    claims_written = 0
    for turn in session_to_turns(q, 0):
        result = on_new_turn(
            conn,
            session_id=session_id,
            speaker=Speaker(turn.speaker),
            text=turn.text,
            extractor=extractor,
            turn_id=turn.turn_id,
        )
        created = [_claim_to_cache_dict(c) for c in result.created_claims]
        claims_written += len(created)
        turns.append(
            {
                "turn_id": turn.turn_id,
                "speaker": str(turn.speaker),
                "text": turn.text,
                "claims": created,
            }
        )
    conn.close()

    payload = {
        "session_id": session_id,
        "turn_count": len(turns),
        "claims_written": claims_written,
        "turns": turns,
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    return payload


def _ensure_session_cache(
    questions_for_cache: list[LongMemEvalQuestion],
    *,
    cache_dir: Path,
    workers: int = 1,
) -> dict[str, int]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    sessions = _session_index(questions_for_cache)
    done = 0
    missing = [
        (sid, transcript)
        for sid, transcript in sessions.items()
        if not _safe_session_cache_path(cache_dir, sid).exists()
    ]
    print(
        f"session cache: {len(sessions)} unique sessions, "
        f"{len(sessions) - len(missing)} cached, {len(missing)} missing"
    )
    if workers <= 1:
        for idx, (sid, transcript) in enumerate(missing, start=1):
            print(f"[cache {idx}/{len(missing)}] {sid}", flush=True)
            _build_one_session_cache(sid, transcript, cache_dir=cache_dir)
            done += 1
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_build_one_session_cache, sid, transcript, cache_dir=cache_dir): sid
                for sid, transcript in missing
            }
            for idx, fut in enumerate(as_completed(futures), start=1):
                sid = futures[fut]
                try:
                    fut.result()
                except Exception as exc:
                    print(f"[cache {idx}/{len(missing)}] FAILED {sid}: {exc!r}", flush=True)
                    raise
                done += 1
                print(f"[cache {idx}/{len(missing)}] done {sid}", flush=True)
    return {"unique_sessions": len(sessions), "built": done, "already_cached": len(sessions) - len(missing)}


def _load_cached_turn_claims(cache_dir: Path, session_id: str) -> dict[str, list[ExtractedClaim]]:
    path = _safe_session_cache_path(cache_dir, session_id)
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, list[ExtractedClaim]] = {}
    for turn in payload.get("turns", []):
        out[str(turn["turn_id"])] = [_claim_from_cache_dict(c) for c in turn.get("claims", [])]
    return out


def _make_cached_ingester(cache_dir: Path) -> Any:
    def ingest_from_cache(q: LongMemEvalQuestion, extractor: Any | None = None) -> tuple[Any, IngestionStats]:
        conn = open_database(":memory:")
        claims_written = 0
        admitted_turns = 0
        empty_extractions = 0
        supersessions_fired = 0

        for sidx in range(len(q.haystack_sessions)):
            sid = (q.haystack_session_ids or [f"{q.question_id}::session::{i:03d}" for i in range(len(q.haystack_sessions))])[sidx]
            cached = _load_cached_turn_claims(cache_dir, str(sid))
            for turn in session_to_turns(q, sidx):
                def cached_extractor(t: Any, *, _cached=cached) -> list[ExtractedClaim]:
                    return list(_cached.get(str(getattr(t, "turn_id", "")), []))

                result = on_new_turn(
                    conn,
                    session_id=q.question_id,
                    speaker=Speaker(turn.speaker),
                    text=turn.text,
                    extractor=cached_extractor,
                    turn_id=turn.turn_id,
                )
                if result.admitted:
                    admitted_turns += 1
                    claims_written += len(result.created_claims)
                    supersessions_fired += len(result.supersession_edges)
                    if not result.created_claims:
                        empty_extractions += 1

        active = list_active_claims(conn, q.question_id)
        from src.substrate.event_frames import assemble_event_frames

        event_frames = assemble_event_frames(conn, q.question_id)
        return conn, IngestionStats(
            claims_written_count=claims_written,
            supersessions_fired_count=supersessions_fired,
            projection_nonempty=admitted_turns > 0,
            active_pack=os.environ.get("ACTIVE_PACK", ""),
            active_claim_count=len(active),
            admitted_turn_count=admitted_turns,
            empty_extraction_turn_count=empty_extractions,
            event_frames_assembled=len(event_frames),
        )

    return ingest_from_cache


def _patch_pipeline_ingestion_to_cache(cache_dir: Path) -> None:
    import eval.longmemeval.pipeline as pipeline

    pipeline.ingest_longmemeval_case = _make_cached_ingester(cache_dir)


def _patch_retrieval_top_k(top_k: int) -> None:
    """Override final/candidate retrieval depths for this process only."""
    import eval.longmemeval.question_router as router
    import eval.longmemeval.pipeline as pipeline

    original = router.classify_question

    def classify_with_top_k(question: str, question_type: str = "") -> RetrievalStrategy:
        strategy = original(question, question_type)
        return replace(strategy, top_k_candidates=max(top_k, strategy.top_k_candidates), top_k_final=top_k)

    router.classify_question = classify_with_top_k
    pipeline.classify_question = classify_with_top_k


def _diagnose_failure(trace: CaseTrace, hypothesis: str, gold: str) -> str:
    if _gold_in_text(gold, hypothesis):
        return "none"
    if not any(_gold_in_text(gold, v) or _gold_in_text(v, gold) for v in trace.extracted_claim_values_by_id.values()):
        return "extractor"
    if not any(_gold_in_text(gold, v) or _gold_in_text(v, gold) for v in trace.retrieved_claim_values_by_id.values()):
        return "retrieval"
    if not any(_gold_in_text(gold, v) or _gold_in_text(v, gold) for v in trace.final_claim_values):
        return "retrieval"
    return "reader"


def _diag_record(
    q: LongMemEvalQuestion,
    trace: CaseTrace,
    hypothesis: str,
    *,
    retrieval_top_k: int,
    reader_model: str,
    extractor_model: str,
    elapsed: float,
) -> dict[str, Any]:
    return {
        "question_id": q.question_id,
        "question_type": q.question_type,
        "is_abstention": "_abs" in q.question_id,
        "question": q.question,
        "gold_answer": q.answer,
        "hypothesis": hypothesis,
        "claims_extracted": trace.claims_written,
        "claims_retrieved": trace.retrieved_candidates,
        "bundle_tokens": trace.bundle_tokens,
        "retrieval_top_k": retrieval_top_k,
        "reader_model": reader_model,
        "extractor_model": extractor_model,
        "time_seconds": round(elapsed, 3),
        "failure_layer": _diagnose_failure(trace, hypothesis, q.answer),
        "notes": (
            "Current substrate path retrieves extracted claims and event tuples only; "
            "raw LongMemEval rounds are not preserved as reader values in this runner."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run protocol-compliant LongMemEval-S predictions")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reader", default="gpt-4o-mini", choices=("gpt-4o-mini", "gpt-4o"))
    parser.add_argument("--retrieval-top-k", type=int, default=50)
    parser.add_argument("--embedding", default="mock", choices=("mock", "default"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--diagnostics", type=Path, default=None)
    parser.add_argument("--all-questions", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--cache-scope", choices=("all", "selected"), default="all")
    parser.add_argument("--cache-workers", type=int, default=8)
    parser.add_argument("--no-session-cache", action="store_true")
    args = parser.parse_args()

    if os.environ.get("ACTIVE_PACK") != "personal_assistant":
        raise RuntimeError("Set ACTIVE_PACK=personal_assistant before running LongMemEval.")

    questions = _load_questions(args.data)
    selected = questions if args.all_questions or args.n >= len(questions) else _stratified_sample(questions, n=args.n, seed=args.seed)
    full_session_count = len(_session_index(questions))
    selected_session_count = len(_session_index(selected))

    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    output = args.output or DEFAULT_OUTPUT_DIR / f"smoke_{len(selected)}_{stamp}.jsonl"
    diagnostics = args.diagnostics or output.with_name(output.stem + "_diagnostics.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    diagnostics.parent.mkdir(parents=True, exist_ok=True)

    _patch_retrieval_top_k(args.retrieval_top_k)
    cache_stats: dict[str, int] | None = None
    if not args.no_session_cache:
        cache_questions = questions if args.cache_scope == "all" else selected
        cache_stats = _ensure_session_cache(
            cache_questions,
            cache_dir=args.cache_dir,
            workers=max(1, args.cache_workers),
        )
        _patch_pipeline_ingestion_to_cache(args.cache_dir)
    reader_fn, reader_model = _make_reader(args.reader)
    embedding_client = _embedding_client(args.embedding)
    extractor_model = os.environ.get("AZURE_OPENAI_GPT4OMINI_DEPLOYMENT") or "gpt-4o-mini"

    print("LongMemEval official prediction run")
    print(f"data={args.data}")
    print(f"n={len(selected)} seed={args.seed} strata={dict(Counter(_stratum(q) for q in selected))}")
    print(f"dataset_unique_sessions={full_session_count} selected_unique_sessions={selected_session_count}")
    print(f"reader={args.reader} resolved_reader_model={reader_model}")
    print(f"retrieval_top_k={args.retrieval_top_k} embedding={args.embedding}")
    print(f"session_cache={'disabled' if args.no_session_cache else args.cache_scope} cache_workers={args.cache_workers} cache_stats={cache_stats}")
    print(f"output={output}")
    print(f"diagnostics={diagnostics}")

    completed_ids: set[str] = set()
    diag_records: list[dict[str, Any]] = []
    if not args.no_resume and output.exists():
        for line in output.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                completed_ids.add(str(json.loads(line)["question_id"]))
            except (KeyError, json.JSONDecodeError):
                continue
    if not args.no_resume and diagnostics.exists():
        try:
            diag_records = json.loads(diagnostics.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            diag_records = []

    mode = "a" if completed_ids and not args.no_resume else "w"
    remaining_selected = [q for q in selected if q.question_id not in completed_ids]
    if completed_ids:
        print(f"resuming: {len(completed_ids)} completed, {len(remaining_selected)} remaining")

    with output.open(mode, encoding="utf-8") as out_f:
        for idx, q in enumerate(remaining_selected, start=len(completed_ids) + 1):
            t0 = time.monotonic()
            print(f"[{idx}/{len(selected)}] {q.question_id} {q.question_type}", flush=True)
            hypothesis, trace = run_substrate_case(
                q,
                embedding_client=embedding_client,
                reader_fn=reader_fn,
            )
            elapsed = time.monotonic() - t0
            out_f.write(json.dumps({"question_id": q.question_id, "hypothesis": hypothesis}, ensure_ascii=False) + "\n")
            out_f.flush()
            diag_records.append(
                _diag_record(
                    q,
                    trace,
                    hypothesis,
                    retrieval_top_k=args.retrieval_top_k,
                    reader_model=reader_model,
                    extractor_model=extractor_model,
                    elapsed=elapsed,
                )
            )
            diagnostics.write_text(json.dumps(diag_records, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {output}")
    print(f"Wrote {diagnostics}")


if __name__ == "__main__":
    main()
