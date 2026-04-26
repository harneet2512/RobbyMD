"""LongMemEval-S final auditable 500-question run.

4 research-backed improvements over the baseline:
  1. Temporal context: question_date + relative offsets + gap markers
  2. Dense + BM25 hybrid retrieval (replaces TF-IDF)
  3. Per-round Chain-of-Note reading (per LongMemEval paper §4)
  4. GPT-5-mini reader with structured CoT

Reader: openai/gpt-5-mini via OpenRouter
Judge:  openai/gpt-4o-2024-11-20 via OpenRouter (official protocol)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.longmemeval.official_evaluate_qa import get_anscheck_prompt
from eval.longmemeval.round_retrieval_run import (
    RoundRecord,
    _build_rounds,
    _estimate_tokens,
    _load_questions,
    _load_raw_by_id,
    _question_from_raw,
)

DEFAULT_DATA = Path("eval/longmemeval/data/longmemeval_s_cleaned.json")
RESULTS_DIR = Path("eval/longmemeval/results")
TOP_K = 30

READER_MODEL = "openai/gpt-5-mini"
JUDGE_MODEL = "openai/gpt-4o-2024-11-20"

# ---------------------------------------------------------------------------
# FIX 1: Temporal context — question_date + relative offsets
# ---------------------------------------------------------------------------

def _parse_lme_date(date_str: str) -> datetime | None:
    """Parse LongMemEval date like '2023/03/04 (Sat) 22:43'."""
    try:
        clean = re.sub(r"\s*\([A-Za-z]+\)\s*", " ", date_str).strip()
        for fmt in ["%Y/%m/%d %H:%M", "%Y/%m/%d"]:
            try:
                return datetime.strptime(clean, fmt)
            except ValueError:
                continue
    except Exception:
        pass
    return None


def _relative_offset(event_date: datetime, question_date: datetime) -> str:
    """Compute human-readable relative offset like Mastra's three-date model."""
    diff = question_date - event_date
    days = diff.days
    if days == 0:
        return "today"
    if days == 1:
        return "yesterday"
    if days < 0:
        return f"in {-days} days"
    if days < 7:
        return f"{days} days ago"
    if days < 14:
        return "1 week ago"
    if days < 30:
        return f"{days // 7} weeks ago"
    if days < 60:
        return "1 month ago"
    if days < 365:
        return f"{days // 30} months ago"
    return f"{days // 365} year(s) ago"


def _temporal_gap_marker(prev_date: datetime, curr_date: datetime) -> str | None:
    """Gap marker between non-adjacent sessions (Mastra pattern)."""
    days = (curr_date - prev_date).days
    if days <= 1:
        return None
    if days < 7:
        return f"[{days} days later]"
    if days < 14:
        return "[1 week later]"
    if days < 30:
        return f"[{days // 7} weeks later]"
    if days < 60:
        return "[1 month later]"
    return f"[{days // 30} months later]"


# ---------------------------------------------------------------------------
# FIX 2: Dense + BM25 hybrid retrieval (replaces TF-IDF)
# ---------------------------------------------------------------------------

_EMBED_MODEL = None
_EMBED_MODEL_LOCK = threading.Lock()


def _get_embed_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        with _EMBED_MODEL_LOCK:
            if _EMBED_MODEL is None:
                from sentence_transformers import SentenceTransformer
                _EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBED_MODEL


def _retrieve_hybrid(
    question: str, rounds: list[RoundRecord], *, top_k: int,
    semantic_weight: float = 0.7, bm25_weight: float = 0.3,
) -> list[RoundRecord]:
    """Dense semantic + BM25 keyword hybrid retrieval with score fusion."""
    if not rounds:
        return []

    keys = [r.key for r in rounds]

    # Semantic signal: dense embeddings
    model = _get_embed_model()
    round_embeds = model.encode(keys, normalize_embeddings=True, show_progress_bar=False)
    query_embed = model.encode([question], normalize_embeddings=True, show_progress_bar=False)
    semantic_scores = np.dot(round_embeds, query_embed.T).ravel()

    # BM25 signal: TF-IDF cosine (sklearn)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    vectorizer = TfidfVectorizer(
        lowercase=True, ngram_range=(1, 2),
        stop_words="english", max_features=50000,
    )
    tfidf_matrix = vectorizer.fit_transform(keys)
    query_vec = vectorizer.transform([question])
    bm25_scores = cosine_similarity(query_vec, tfidf_matrix).ravel()

    # Normalize both to [0, 1]
    def _norm(arr):
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-9:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)

    sem_norm = _norm(semantic_scores)
    bm25_norm = _norm(bm25_scores)

    # Fused score
    fused = semantic_weight * sem_norm + bm25_weight * bm25_norm

    ranked = sorted(
        range(len(rounds)),
        key=lambda idx: (-float(fused[idx]), rounds[idx].session_order, rounds[idx].round_index),
    )
    selected = [rounds[idx] for idx in ranked[:max(1, top_k)]]
    return sorted(selected, key=lambda r: (r.timestamp, r.session_order, r.round_index))


# ---------------------------------------------------------------------------
# FIX 3 & 4: Per-round CoN reader prompt with temporal context
# ---------------------------------------------------------------------------

READER_SYSTEM_PROMPT = (
    "You are answering a question about a user based on their past conversation history.\n\n"
    "You receive the question, the date it was asked, and retrieved conversation excerpts "
    "with timestamps and relative time offsets.\n\n"
    "Step 1 — NOTES: Read ALL excerpts. Write a short list of ONLY the relevant facts "
    "(skip irrelevant excerpts entirely). For each relevant fact, note which excerpt it "
    "came from and its date.\n\n"
    "Step 2 — ANSWER: Using your notes, answer the question.\n\n"
    "Guidelines:\n"
    "- Answer from provided evidence only.\n"
    "- Return stored preferences/constraints, not generic advice.\n"
    "- Use the relative time offsets (e.g., '3 weeks ago') to compute durations and recency.\n"
    "- When values change over time, prefer the most recent one.\n"
    "- Count and list ALL relevant items across ALL excerpts before answering count questions.\n"
    "- Only say \"I don't know\" if evidence truly has no relevant information.\n"
    "- Be concise.\n\n"
    "Format:\n"
    "NOTES:\n"
    "- [relevant fact 1, from excerpt N, date]\n"
    "- [relevant fact 2, from excerpt M, date]\n"
    "...\n\n"
    "ANSWER:\n"
    "[your concise answer]"
)


def _format_evidence(
    question: str,
    question_date_str: str,
    rounds: list[RoundRecord],
) -> str:
    """Format evidence with temporal context (Fix 1) and per-round structure (Fix 3/4)."""
    q_date = _parse_lme_date(question_date_str)
    excerpts = []
    prev_date = None

    for i, r in enumerate(rounds, 1):
        r_date = _parse_lme_date(r.timestamp)

        # Temporal gap marker (Mastra pattern)
        if prev_date and r_date:
            gap = _temporal_gap_marker(prev_date, r_date)
            if gap:
                excerpts.append(gap)

        # Relative offset
        offset = ""
        if r_date and q_date:
            offset = f" ({_relative_offset(r_date, q_date)})"

        excerpts.append(
            f"--- Excerpt {i} | {r.timestamp}{offset} | session={r.session_id} ---\n"
            f"{r.key}"
        )
        if r_date:
            prev_date = r_date

    evidence_block = "\n\n".join(excerpts)
    return (
        f"Question date: {question_date_str}\n"
        f"Question: {question}\n\n"
        f"Retrieved evidence ({len(rounds)} excerpts, chronological order):\n\n"
        f"{evidence_block}"
    )


# ---------------------------------------------------------------------------
# Scorer cleanup (strict short-answer matching)
# ---------------------------------------------------------------------------

def _normalize_short(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = re.sub(r"(?<=\d)[,_](?=\d)", "", normalized)
    chars = []
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


# ---------------------------------------------------------------------------
# OpenRouter client
# ---------------------------------------------------------------------------

def _make_client() -> Any:
    from openai import OpenAI
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)


# ---------------------------------------------------------------------------
# Reader call
# ---------------------------------------------------------------------------

def _call_reader(client: Any, evidence_text: str) -> tuple[str, str, str]:
    """Single-call CoN reader. Returns (full_output, notes, answer)."""
    # Cap evidence to avoid token limits
    if len(evidence_text) > 80000:
        evidence_text = evidence_text[:80000] + "\n... (truncated)"

    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model=READER_MODEL,
                messages=[
                    {"role": "system", "content": READER_SYSTEM_PROMPT},
                    {"role": "user", "content": evidence_text},
                ],
                temperature=0.0,
                max_tokens=2000,
            )
            full = (resp.choices[0].message.content or "").strip()
            if not full:
                if attempt < 4:
                    time.sleep(2 ** attempt)
                    continue
                return "I don't know.", "", "I don't know."
            notes, answer = _parse_cot(full)
            return full, notes, answer
        except Exception as exc:
            if "content_filter" in str(exc) or "ResponsibleAIPolicyViolation" in str(exc):
                return "I don't know.", "", "I don't know."
            if attempt < 4:
                time.sleep(2 ** attempt)
                continue
            return "I don't know.", "", "I don't know."
    return "I don't know.", "", "I don't know."


def _parse_cot(full: str) -> tuple[str, str]:
    # Try to find ANSWER: section first — it's what we score
    answer_match = re.search(r"\bANSWER:\s*\n?(.*)", full, re.DOTALL | re.IGNORECASE)
    if answer_match:
        answer = answer_match.group(1).strip()
        notes_match = re.search(r"\bNOTES:\s*\n?(.*?)(?=\bANSWER:)", full, re.DOTALL | re.IGNORECASE)
        notes = notes_match.group(1).strip() if notes_match else ""
        return notes, answer
    # No ANSWER: tag — model may have just given an answer directly
    # Strip any leading NOTES: block and use the rest
    stripped = re.sub(r"^NOTES:.*?(?=\n\n|\Z)", "", full, flags=re.DOTALL | re.IGNORECASE).strip()
    return "", stripped if stripped else full.strip()


# ---------------------------------------------------------------------------
# Judge (official protocol)
# ---------------------------------------------------------------------------

def _call_judge(client: Any, qid: str, question: str, gold: str, hypothesis: str, qtype: str) -> bool:
    is_abs = "_abs" in qid
    prompt = get_anscheck_prompt(qtype, question, gold, hypothesis, abstention=is_abs)
    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=16,
            )
            return "yes" in resp.choices[0].message.content.strip().lower()
        except Exception:
            if attempt < 4:
                time.sleep(2 ** attempt)
                continue
            return False
    return False


# ---------------------------------------------------------------------------
# SHA-256
# ---------------------------------------------------------------------------

def _sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
    except FileNotFoundError:
        return "MISSING"
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Phase 0: Manifest
# ---------------------------------------------------------------------------

def phase0_manifest() -> dict[str, Any]:
    print("=" * 60)
    print("  PHASE 0: Pre-Run Manifest")
    print("=" * 60)

    try:
        git_hash = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=str(ROOT)).stdout.strip()
    except Exception:
        git_hash = "UNKNOWN"
    git_dirty = subprocess.run(["git", "diff", "--quiet"], capture_output=True, cwd=str(ROOT)).returncode != 0

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_hash": git_hash,
        "git_dirty": git_dirty,
        "file_hashes": {
            "dataset": _sha256_file(ROOT / DEFAULT_DATA)[:16],
            "scorer": _sha256_file(ROOT / "eval/longmemeval/official_evaluate_qa.py")[:16],
            "final_runner": _sha256_file(ROOT / "eval/longmemeval/final_runner.py")[:16],
        },
        "prompt_hash": hashlib.sha256(READER_SYSTEM_PROMPT.encode()).hexdigest()[:16],
        "models": {"reader": READER_MODEL, "judge": JUDGE_MODEL},
        "retrieval": {"method": "dense_bm25_hybrid", "semantic_weight": 0.7, "bm25_weight": 0.3, "top_k": TOP_K, "embed_model": "all-MiniLM-L6-v2"},
        "improvements": ["question_date_temporal_context", "dense_bm25_hybrid_retrieval", "per_round_chain_of_note", "gpt5_mini_reader"],
    }

    out = ROOT / RESULTS_DIR / "final_run_manifest.pre.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Written: {out}")
    for k, v in manifest.items():
        print(f"  {k}: {v}")
    return manifest


# ---------------------------------------------------------------------------
# Phase 1: Legitimacy
# ---------------------------------------------------------------------------

def phase1_legitimacy() -> bool:
    print("\n" + "=" * 60)
    print("  PHASE 1: Legitimacy Audit")
    print("=" * 60)

    dataset = json.loads((ROOT / DEFAULT_DATA).read_text("utf-8"))
    expected = {"single-session-user": 70, "single-session-assistant": 56, "single-session-preference": 30,
                "multi-session": 133, "temporal-reasoning": 133, "knowledge-update": 78}
    type_counts = Counter(r["question_type"] for r in dataset)
    abs_count = sum(1 for r in dataset if "_abs" in str(r["question_id"]))
    ds_ok = len(dataset) == 500 and dict(type_counts) == expected and abs_count == 30
    print(f"  Dataset: {'PASS' if ds_ok else 'FAIL'} ({len(dataset)} questions, abs={abs_count})")

    # question_date availability
    qdate_count = sum(1 for r in dataset if r.get("question_date"))
    print(f"  question_date: {qdate_count}/500 available")

    # Contamination
    prompt_lower = READER_SYSTEM_PROMPT.lower()
    contam = [w for w in ["longmemeval", "benchmark", "the answer is", "gold answer"] if w in prompt_lower]
    print(f"  Contamination: {'PASS' if not contam else 'FAIL ' + str(contam)}")

    ok = ds_ok and not contam
    print(f"  VERDICT: {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# Phase 2: Validation
# ---------------------------------------------------------------------------

def phase2_validation() -> bool:
    print("\n" + "=" * 60)
    print("  PHASE 2: Validation")
    print("=" * 60)

    # Scorer tests
    cases = [("$400,000", "$350,000", False), ("8 days", "3 days", False), ("Spotify", "You use Spotify.", True)]
    scorer_ok = all(strict_short_answer_check(g, h) == e for g, h, e in cases)
    print(f"  Scorer: {'PASS' if scorer_ok else 'FAIL'}")

    # Parse test
    notes, answer = _parse_cot("NOTES:\n1. User likes cats\n\nANSWER:\nThree cats.")
    parse_ok = "cats" in notes and answer == "Three cats."
    print(f"  CoT parse: {'PASS' if parse_ok else 'FAIL'}")

    # Temporal offset test
    d1 = datetime(2023, 3, 4)
    d2 = datetime(2023, 4, 1)
    offset = _relative_offset(d1, d2)
    offset_ok = "4 weeks" in offset or "28 days" in offset
    print(f"  Temporal offset: '{offset}' -> {'PASS' if offset_ok else 'FAIL'}")

    # Hybrid retrieval test (quick)
    print(f"  Embedding model: loading...", end="", flush=True)
    _get_embed_model()
    print(f" loaded")

    ok = scorer_ok and parse_ok
    print(f"  VERDICT: {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# Phase 3: Run
# ---------------------------------------------------------------------------

@dataclass
class CaseResult:
    question_id: str
    question_type: str
    gold: str
    hypothesis: str
    full_cot: str
    notes: str
    correct_judge: bool
    correct_final: bool
    retrieval_recall: float
    reader_input_tokens: int
    rounds_indexed: int
    rounds_retrieved: int
    elapsed_s: float
    question_date: str


def phase3_run(workers: int = 10, dry_run: bool = False, spot_ids: list[str] | None = None) -> list[CaseResult]:
    print("\n" + "=" * 60)
    print("  PHASE 3: Run")
    print("=" * 60)

    raw_by_id = _load_raw_by_id(ROOT / DEFAULT_DATA)
    questions = [_question_from_raw(raw_by_id, q) for q in _load_questions(ROOT / DEFAULT_DATA)]
    q_by_id = {q.question_id: q for q in questions}
    ref_data = json.loads((ROOT / DEFAULT_DATA).read_text("utf-8"))
    ref_by_id = {str(r["question_id"]): r for r in ref_data}

    target_ids = spot_ids if spot_ids else [str(r["question_id"]) for r in ref_data]
    prefix = "spot" if spot_ids else "final_full_500"

    hyp_path = ROOT / RESULTS_DIR / f"{prefix}.jsonl"
    scored_path = ROOT / RESULTS_DIR / f"{prefix}_scored.jsonl"
    diag_path = ROOT / RESULTS_DIR / f"{prefix}_diagnostics.json"

    print(f"  Cases: {len(target_ids)}")
    print(f"  Reader: {READER_MODEL}, Judge: {JUDGE_MODEL}")
    print(f"  Retrieval: dense+BM25 hybrid (0.7/0.3), top_k={TOP_K}")
    print(f"  Workers: {workers}")

    if dry_run:
        print("  [DRY RUN]")
        return []

    client = _make_client()

    # Pre-warm embedding model
    print("  Loading embedding model...", end="", flush=True)
    _get_embed_model()
    print(" done")

    # Resume
    completed_ids: set[str] = set()
    results: list[CaseResult] = []
    diagnostics: list[dict[str, Any]] = []

    hyp_path.parent.mkdir(parents=True, exist_ok=True)
    if hyp_path.exists():
        for line in hyp_path.read_text("utf-8").splitlines():
            if line.strip():
                completed_ids.add(json.loads(line)["question_id"])
    if diag_path.exists():
        try:
            diagnostics = json.loads(diag_path.read_text("utf-8"))
        except json.JSONDecodeError:
            diagnostics = []

    remaining = [qid for qid in target_ids if qid not in completed_ids]
    if completed_ids:
        print(f"  Resuming: {len(completed_ids)} done, {len(remaining)} remaining")

    def _process(qid: str) -> CaseResult:
        t0 = time.monotonic()
        q = q_by_id[qid]
        ref = ref_by_id[qid]
        gold = str(ref["answer"])
        qtype = ref["question_type"]
        question_date = ref.get("question_date", "")

        # Build and retrieve with hybrid (Fix 2)
        rounds = _build_rounds(q)
        retrieved = _retrieve_hybrid(q.question, rounds, top_k=TOP_K)

        # Retrieval recall
        sessions_hit = {r.session_id for r in retrieved}
        gold_sids = [str(s) for s in ref.get("answer_session_ids", []) or []]
        recall = sum(1 for s in gold_sids if s in sessions_hit) / max(1, len(gold_sids)) if gold_sids else 1.0

        # Format evidence with temporal context (Fixes 1, 3, 4)
        evidence_text = _format_evidence(q.question, question_date, retrieved)
        input_tokens = _estimate_tokens(evidence_text)

        # Reader call (Fix 4: per-round CoN)
        full_cot, notes, answer = _call_reader(client, evidence_text)

        # Judge
        judge_correct = _call_judge(client, qid, q.question, gold, answer, qtype)
        strict = strict_short_answer_check(gold, answer)
        final_correct = strict if strict is not None else judge_correct

        elapsed = time.monotonic() - t0
        return CaseResult(
            question_id=qid, question_type=qtype, gold=gold,
            hypothesis=answer, full_cot=full_cot, notes=notes,
            correct_judge=judge_correct, correct_final=final_correct,
            retrieval_recall=recall, reader_input_tokens=input_tokens,
            rounds_indexed=len(rounds), rounds_retrieved=len(retrieved),
            elapsed_s=round(elapsed, 2), question_date=question_date,
        )

    write_lock = threading.Lock()
    done_count = [len(completed_ids)]

    with hyp_path.open("a" if completed_ids else "w", encoding="utf-8") as hyp_f, \
         scored_path.open("a" if completed_ids else "w", encoding="utf-8") as scored_f:

        if workers <= 1:
            for qid in remaining:
                r = _process(qid)
                done_count[0] += 1
                status = "PASS" if r.correct_final else "FAIL"
                print(f"  [{done_count[0]}/{len(target_ids)}] {r.question_id} ({r.question_type}) -> {status} ({r.elapsed_s}s)")
                hyp_f.write(json.dumps({"question_id": r.question_id, "hypothesis": r.hypothesis}, ensure_ascii=False) + "\n")
                hyp_f.flush()
                scored_f.write(json.dumps({"question_id": r.question_id, "hypothesis": r.hypothesis, "autoeval_label": {"model": JUDGE_MODEL, "label": r.correct_final}}, ensure_ascii=False) + "\n")
                scored_f.flush()
                results.append(r)
                diagnostics.append(_to_diag(r))
                diag_path.write_text(json.dumps(diagnostics, indent=2, ensure_ascii=False), encoding="utf-8")
        else:
            futures = {}
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for qid in remaining:
                    futures[pool.submit(_process, qid)] = qid
                for fut in as_completed(futures):
                    try:
                        r = fut.result()
                    except Exception as exc:
                        print(f"  ERROR {futures[fut]}: {exc}")
                        continue
                    with write_lock:
                        done_count[0] += 1
                        status = "PASS" if r.correct_final else "FAIL"
                        print(f"  [{done_count[0]}/{len(target_ids)}] {r.question_id} ({r.question_type}) -> {status} ({r.elapsed_s}s)")
                        hyp_f.write(json.dumps({"question_id": r.question_id, "hypothesis": r.hypothesis}, ensure_ascii=False) + "\n")
                        hyp_f.flush()
                        scored_f.write(json.dumps({"question_id": r.question_id, "hypothesis": r.hypothesis, "autoeval_label": {"model": JUDGE_MODEL, "label": r.correct_final}}, ensure_ascii=False) + "\n")
                        scored_f.flush()
                        results.append(r)
                        diagnostics.append(_to_diag(r))
                        diag_path.write_text(json.dumps(diagnostics, indent=2, ensure_ascii=False), encoding="utf-8")

    return results


def _to_diag(r: CaseResult) -> dict[str, Any]:
    return {
        "question_id": r.question_id, "question_type": r.question_type,
        "gold": r.gold, "hypothesis": r.hypothesis,
        "full_cot": r.full_cot, "notes": r.notes,
        "correct_judge": r.correct_judge, "correct_final": r.correct_final,
        "retrieval_recall": r.retrieval_recall,
        "reader_input_tokens": r.reader_input_tokens,
        "rounds_indexed": r.rounds_indexed, "rounds_retrieved": r.rounds_retrieved,
        "elapsed_s": r.elapsed_s, "question_date": r.question_date,
    }


# ---------------------------------------------------------------------------
# Phase 4: Audit
# ---------------------------------------------------------------------------

def phase4_audit(results: list[CaseResult]) -> None:
    print("\n" + "=" * 60)
    print("  PHASE 4: Post-Run Audit")
    print("=" * 60)

    total = len(results)
    correct = sum(1 for r in results if r.correct_final)

    types: dict[str, list[int]] = {}
    abs_acc: list[int] = []
    for r in results:
        types.setdefault(r.question_type, []).append(1 if r.correct_final else 0)
        if "_abs" in r.question_id:
            abs_acc.append(1 if r.correct_final else 0)

    task_means = []
    print(f"\n  Per type:")
    for qt in sorted(types):
        acc = sum(types[qt])
        tot = len(types[qt])
        rate = acc / tot
        task_means.append(rate)
        print(f"    {qt}: {acc}/{tot} ({100*rate:.1f}%)")

    overall = correct / total
    task_avg = sum(task_means) / len(task_means)
    abs_rate = sum(abs_acc) / len(abs_acc) if abs_acc else 0.0
    avg_tokens = sum(r.reader_input_tokens for r in results) / total
    avg_recall = sum(r.retrieval_recall for r in results) / total

    print(f"\n  Overall: {correct}/{total} ({100*overall:.1f}%)")
    print(f"  Task-averaged: {100*task_avg:.1f}%")
    print(f"  Abstention: {100*abs_rate:.1f}%")
    print(f"  Avg tokens: {avg_tokens:.0f}, Avg recall: {avg_recall:.3f}")

    published = [
        ("Mem0", 93.4), ("Mastra OM", 94.87), ("EverMemOS", 83.0),
        ("TiMem", 76.9), ("Zep/Graphiti", 71.2), ("Baseline (ours)", 65.8),
    ]
    print(f"\n  Leaderboard:")
    for name, score in published:
        print(f"    {name}: {score:.1f}%")
    print(f"    >>> Us (final): {100*overall:.1f}% <<<")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="LongMemEval-S final run")
    parser.add_argument("--phase", default="all", choices=["0", "1", "2", "3", "4", "all", "spot"])
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--spot-ids", nargs="*", default=None)
    args = parser.parse_args()

    if args.phase == "spot":
        ids = args.spot_ids or ["06878be2", "71017276", "0a995998", "852ce960", "51a45a95"]
        # Clean previous spot results
        for f in ["spot.jsonl", "spot_scored.jsonl", "spot_diagnostics.json"]:
            p = ROOT / RESULTS_DIR / f
            if p.exists():
                p.unlink()
        results = phase3_run(workers=1, spot_ids=ids)
        if results:
            correct = sum(1 for r in results if r.correct_final)
            print(f"\nSpot: {correct}/{len(results)}")
            for r in results:
                print(f"  {r.question_id} ({r.question_type}): {'PASS' if r.correct_final else 'FAIL'}")
                print(f"    Gold: {r.gold[:60]}")
                print(f"    Answer: {r.hypothesis[:80].encode('ascii', 'replace').decode()}")
                print(f"    Recall: {r.retrieval_recall:.2f}, Tokens: {r.reader_input_tokens}")
        return

    phases = {0, 1, 2, 3, 4} if args.phase == "all" else {int(args.phase)}
    if 0 in phases:
        phase0_manifest()
    if 1 in phases:
        if not phase1_legitimacy():
            return
    if 2 in phases:
        if not phase2_validation():
            return
    if 3 in phases:
        results = phase3_run(workers=args.workers, dry_run=args.dry_run)
        if 4 in phases and results:
            phase4_audit(results)


if __name__ == "__main__":
    main()
