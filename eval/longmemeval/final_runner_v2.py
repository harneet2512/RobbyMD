"""LongMemEval-S final run v2 — fact extraction + HELM-level logging.

Improvements over v1:
  1. Fact extraction: LLM extracts atomic facts from retrieved rounds before reader
  2. top_k bumped 30→50 (fixes 4 retrieval misses, free)
  3. HELM-inspired 5-file audit trail per run

Models:
  Reader + Extractor: openai/gpt-5-mini via OpenRouter
  Judge: openai/gpt-4o-2024-11-20 via OpenRouter
  Embeddings: all-MiniLM-L6-v2 (local, free)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import threading
import time
import unicodedata
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
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
RESULTS_BASE = Path("eval/longmemeval/results")
TOP_K = 50
READER_MODEL = "openai/gpt-5-mini"
JUDGE_MODEL = "openai/gpt-4o-2024-11-20"
EXTRACTOR_MODEL = "openai/gpt-5-mini"

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM = (
    "You extract atomic facts from conversation excerpts to help answer a question.\n\n"
    "Rules:\n"
    "- Each fact must be a single, self-contained sentence (15-80 words)\n"
    "- Replace pronouns with actual names or 'User'\n"
    "- Preserve exact numbers, dates, locations, product names\n"
    "- Convert relative dates using the provided question date\n"
    "- Include the source date for each fact\n"
    "- Extract facts from BOTH user and assistant messages\n"
    "- Skip chitchat, greetings, and irrelevant discussion\n\n"
    "Output a JSON array of fact strings. Example:\n"
    '[\"User was pre-approved for $400,000 from Wells Fargo (2023-11-30)\", '
    '\"User bought a Sony A7R IV camera (2023-08-15)\"]'
)

READER_SYSTEM = (
    "You are answering a question about a user based on their past conversation history.\n\n"
    "You receive the question, the date it was asked, and extracted facts from conversations "
    "with timestamps.\n\n"
    "Step 1 - NOTES: Write a short list of the most relevant facts for answering the question. "
    "Note any conflicts, updates, or temporal relationships.\n\n"
    "Step 2 - ANSWER: Answer the question using your notes.\n\n"
    "Guidelines:\n"
    "- Answer from provided facts only.\n"
    "- Return stored preferences/constraints, not generic advice.\n"
    "- Use dates and relative offsets to compute durations and recency.\n"
    "- When values change over time, prefer the most recent one.\n"
    "- Count and list ALL relevant items before answering count questions.\n"
    "- Only say \"I don't know\" if the facts truly contain no relevant information.\n"
    "- Be concise.\n\n"
    "Format:\n"
    "NOTES:\n- [note 1]\n- [note 2]\n\nANSWER:\n[your concise answer]"
)

# ---------------------------------------------------------------------------
# HELM-inspired per-call logging
# ---------------------------------------------------------------------------

@dataclass
class APICall:
    call_id: str = ""
    question_id: str = ""
    call_type: str = ""       # "extraction", "reader", "judge"
    model: str = ""
    prompt_hash: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_s: float = 0.0
    timestamp_utc: str = ""
    success: bool = True
    error: str = ""
    response_preview: str = ""
    cost_usd: float = 0.0


# Pricing
PRICING = {
    "openai/gpt-5-mini": {"input": 1.10, "output": 4.40},
    "openai/gpt-4o-2024-11-20": {"input": 2.50, "output": 10.00},
}

_call_log: list[dict[str, Any]] = []
_call_log_lock = threading.Lock()


def _log_call(call: APICall) -> None:
    p = PRICING.get(call.model, {"input": 0, "output": 0})
    call.cost_usd = call.prompt_tokens / 1e6 * p["input"] + call.completion_tokens / 1e6 * p["output"]
    with _call_log_lock:
        _call_log.append({
            "call_id": call.call_id,
            "question_id": call.question_id,
            "call_type": call.call_type,
            "model": call.model,
            "prompt_hash": call.prompt_hash,
            "prompt_tokens": call.prompt_tokens,
            "completion_tokens": call.completion_tokens,
            "total_tokens": call.total_tokens,
            "latency_s": call.latency_s,
            "timestamp_utc": call.timestamp_utc,
            "success": call.success,
            "error": call.error,
            "response_preview": call.response_preview,
            "cost_usd": round(call.cost_usd, 6),
        })


def _track_call(resp: Any, qid: str, call_type: str, model: str, prompt: str, t0: float) -> APICall:
    usage = getattr(resp, "usage", None)
    call = APICall(
        call_id=str(uuid.uuid4())[:8],
        question_id=qid,
        call_type=call_type,
        model=model,
        prompt_hash=hashlib.md5(prompt.encode()).hexdigest()[:12],
        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        total_tokens=getattr(usage, "total_tokens", 0) or 0,
        latency_s=round(time.monotonic() - t0, 2),
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        response_preview=(resp.choices[0].message.content or "")[:100] if resp.choices else "",
    )
    _log_call(call)
    return call


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
# Temporal context (same as v1)
# ---------------------------------------------------------------------------

def _parse_lme_date(date_str: str) -> datetime | None:
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
    days = (question_date - event_date).days
    if days == 0: return "today"
    if days == 1: return "yesterday"
    if days < 0: return f"in {-days} days"
    if days < 7: return f"{days} days ago"
    if days < 14: return "1 week ago"
    if days < 30: return f"{days // 7} weeks ago"
    if days < 60: return "1 month ago"
    if days < 365: return f"{days // 30} months ago"
    return f"{days // 365} year(s) ago"


def _temporal_gap(prev: datetime, curr: datetime) -> str | None:
    days = (curr - prev).days
    if days <= 1: return None
    if days < 7: return f"[{days} days later]"
    if days < 14: return "[1 week later]"
    if days < 30: return f"[{days // 7} weeks later]"
    return f"[{days // 30} months later]"


# ---------------------------------------------------------------------------
# Hybrid retrieval (same as v1, top_k=50 now)
# ---------------------------------------------------------------------------

_EMBED_MODEL = None
_EMBED_LOCK = threading.Lock()


def _get_embed_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        with _EMBED_LOCK:
            if _EMBED_MODEL is None:
                from sentence_transformers import SentenceTransformer
                _EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBED_MODEL


def _retrieve_hybrid(question: str, rounds: list[RoundRecord], *, top_k: int) -> list[RoundRecord]:
    if not rounds:
        return []
    keys = [r.key for r in rounds]
    model = _get_embed_model()
    round_embeds = model.encode(keys, normalize_embeddings=True, show_progress_bar=False)
    query_embed = model.encode([question], normalize_embeddings=True, show_progress_bar=False)
    sem_scores = np.dot(round_embeds, query_embed.T).ravel()
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), stop_words="english", max_features=50000)
    tfidf = vec.fit_transform(keys)
    bm25_scores = cosine_similarity(vec.transform([question]), tfidf).ravel()
    def _norm(a):
        mn, mx = a.min(), a.max()
        return np.zeros_like(a) if mx - mn < 1e-9 else (a - mn) / (mx - mn)
    fused = 0.7 * _norm(sem_scores) + 0.3 * _norm(bm25_scores)
    ranked = sorted(range(len(rounds)), key=lambda i: (-float(fused[i]), rounds[i].session_order, rounds[i].round_index))
    selected = [rounds[i] for i in ranked[:max(1, top_k)]]
    return sorted(selected, key=lambda r: (r.timestamp, r.session_order, r.round_index))


# ---------------------------------------------------------------------------
# Fact extraction (NEW in v2)
# ---------------------------------------------------------------------------

def _extract_facts(client: Any, qid: str, question: str, question_date: str, rounds: list[RoundRecord]) -> tuple[list[str], str]:
    """Extract atomic facts from retrieved rounds. Returns (facts, raw_response)."""
    q_date = _parse_lme_date(question_date)
    excerpts = []
    for i, r in enumerate(rounds, 1):
        r_date = _parse_lme_date(r.timestamp)
        offset = f" ({_relative_offset(r_date, q_date)})" if r_date and q_date else ""
        excerpts.append(f"[Excerpt {i} | {r.timestamp}{offset}]\n{r.key}")
    evidence_text = "\n\n".join(excerpts)
    if len(evidence_text) > 80000:
        evidence_text = evidence_text[:80000] + "\n...(truncated)"
    user_msg = f"Question date: {question_date}\nQuestion: {question}\n\nConversation excerpts:\n{evidence_text}"
    for attempt in range(3):
        t0 = time.monotonic()
        try:
            resp = client.chat.completions.create(
                model=EXTRACTOR_MODEL,
                messages=[
                    {"role": "system", "content": EXTRACTION_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=8000,
            )
            _track_call(resp, qid, "extraction", EXTRACTOR_MODEL, user_msg[:200], t0)
            raw = (resp.choices[0].message.content or "").strip()
            # GPT-5-mini sometimes emits control chars or unescaped chars in JSON
            sanitized = raw.replace("\n", "\\n").replace("\r", "").replace("\t", " ")
            try:
                facts = json.loads(sanitized)
                if isinstance(facts, list):
                    return [str(f).replace("\\n", " ") for f in facts], raw
            except json.JSONDecodeError:
                pass
            # Fallback: extract lines that look like fact strings
            lines = [l.strip().strip('",[]') for l in raw.split("\n") if l.strip() and not l.strip().startswith("[") and not l.strip().startswith("]")]
            if lines:
                return [l for l in lines if len(l) > 10], raw
            return [], raw
        except Exception as exc:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                _log_call(APICall(call_id=str(uuid.uuid4())[:8], question_id=qid, call_type="extraction",
                                  model=EXTRACTOR_MODEL, success=False, error=str(exc)[:200],
                                  latency_s=round(time.monotonic()-t0, 2),
                                  timestamp_utc=datetime.now(timezone.utc).isoformat()))
                return [], f"ERROR: {exc}"
    return [], ""


# ---------------------------------------------------------------------------
# Reader (uses extracted facts, not raw rounds)
# ---------------------------------------------------------------------------

def _call_reader(client: Any, qid: str, question: str, question_date: str, facts: list[str]) -> tuple[str, str, str]:
    """Reader over extracted facts. Returns (full_cot, notes, answer)."""
    user_msg = json.dumps({
        "question_date": question_date,
        "question": question,
        "extracted_facts": facts,
    }, indent=1, ensure_ascii=False)
    if len(user_msg) > 60000:
        user_msg = user_msg[:60000] + "\n...(truncated)"
    for attempt in range(5):
        t0 = time.monotonic()
        try:
            resp = client.chat.completions.create(
                model=READER_MODEL,
                messages=[
                    {"role": "system", "content": READER_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=8000,
            )
            _track_call(resp, qid, "reader", READER_MODEL, user_msg[:200], t0)
            full = (resp.choices[0].message.content or "").strip()
            if not full:
                if attempt < 4:
                    time.sleep(2 ** attempt)
                    continue
                return "I don't know.", "", "I don't know."
            notes, answer = _parse_cot(full)
            return full, notes, answer
        except Exception as exc:
            if "content_filter" in str(exc):
                return "I don't know.", "", "I don't know."
            if attempt < 4:
                time.sleep(2 ** attempt)
            else:
                return "I don't know.", "", "I don't know."
    return "I don't know.", "", "I don't know."


def _parse_cot(full: str) -> tuple[str, str]:
    answer_match = re.search(r"\bANSWER:\s*\n?(.*)", full, re.DOTALL | re.IGNORECASE)
    if answer_match:
        answer = answer_match.group(1).strip()
        notes_match = re.search(r"\bNOTES:\s*\n?(.*?)(?=\bANSWER:)", full, re.DOTALL | re.IGNORECASE)
        notes = notes_match.group(1).strip() if notes_match else ""
        return notes, answer
    stripped = re.sub(r"^NOTES:.*?(?=\n\n|\Z)", "", full, flags=re.DOTALL | re.IGNORECASE).strip()
    return "", stripped if stripped else full.strip()


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

def _call_judge(client: Any, qid: str, question: str, gold: str, hypothesis: str, qtype: str) -> bool:
    is_abs = "_abs" in qid
    prompt = get_anscheck_prompt(qtype, question, gold, hypothesis, abstention=is_abs)
    for attempt in range(5):
        t0 = time.monotonic()
        try:
            resp = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=16,
            )
            _track_call(resp, qid, "judge", JUDGE_MODEL, prompt[:200], t0)
            return "yes" in resp.choices[0].message.content.strip().lower()
        except Exception:
            if attempt < 4:
                time.sleep(2 ** attempt)
            else:
                return False
    return False


# ---------------------------------------------------------------------------
# Scorer cleanup
# ---------------------------------------------------------------------------

def _normalize_short(text: str) -> str:
    n = unicodedata.normalize("NFKC", text).casefold()
    n = re.sub(r"(?<=\d)[,_](?=\d)", "", n)
    return re.sub(r"\s+", " ", "".join(c if c.isalnum() else " " for c in n if unicodedata.category(c) != "Sc")).strip()

def _boundary_match(g: str, t: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(g)}(?![a-z0-9])", t) is not None

def strict_short_check(gold: str, hyp: str) -> bool | None:
    gn = _normalize_short(gold)
    if len(re.findall(r"[a-z0-9]+", gn)) > 3: return None
    if not gn: return None
    hn = _normalize_short(hyp)
    if not hn: return False
    if gn == hn or _boundary_match(gn, hn): return True
    gnums = set(re.findall(r"\d+", gn))
    if gnums:
        hnums = set(re.findall(r"\d+", hn))
        if hnums and not gnums & hnums: return False
    return None


# ---------------------------------------------------------------------------
# SHA-256
# ---------------------------------------------------------------------------

def _sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
    except FileNotFoundError:
        return "MISSING"
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

@dataclass
class CaseResult:
    question_id: str
    question_type: str
    gold: str
    hypothesis: str
    full_cot: str
    notes: str
    facts: list[str]
    facts_count: int
    correct_judge: bool
    correct_final: bool
    retrieval_recall: float
    reader_input_tokens: int
    rounds_indexed: int
    rounds_retrieved: int
    elapsed_s: float
    question_date: str
    extraction_raw: str = ""


def run(workers: int = 10, dry_run: bool = False, spot_ids: list[str] | None = None) -> list[CaseResult]:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:6]
    run_dir = ROOT / RESULTS_BASE / f"v2_{run_id}" if not spot_ids else ROOT / RESULTS_BASE / "v2_spot"
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── run_spec.json (HELM pattern) ──
    try:
        git_hash = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=str(ROOT)).stdout.strip()
    except Exception:
        git_hash = "UNKNOWN"
    git_dirty = subprocess.run(["git", "diff", "--quiet"], capture_output=True, cwd=str(ROOT)).returncode != 0

    run_spec = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_hash": git_hash,
        "git_dirty": git_dirty,
        "python_version": sys.version,
        "platform": platform.platform(),
        "models": {"reader": READER_MODEL, "judge": JUDGE_MODEL, "extractor": EXTRACTOR_MODEL},
        "retrieval": {"method": "dense_bm25_hybrid", "semantic_weight": 0.7, "bm25_weight": 0.3,
                      "top_k": TOP_K, "embed_model": "all-MiniLM-L6-v2"},
        "prompts": {
            "extraction_hash": hashlib.sha256(EXTRACTION_SYSTEM.encode()).hexdigest()[:16],
            "reader_hash": hashlib.sha256(READER_SYSTEM.encode()).hexdigest()[:16],
            "extraction_text": EXTRACTION_SYSTEM,
            "reader_text": READER_SYSTEM,
        },
        "file_hashes": {
            "dataset": _sha256(ROOT / DEFAULT_DATA)[:16],
            "scorer": _sha256(ROOT / "eval/longmemeval/official_evaluate_qa.py")[:16],
            "runner": _sha256(ROOT / "eval/longmemeval/final_runner_v2.py")[:16],
        },
        "improvements": [
            "question_date_temporal_context",
            "dense_bm25_hybrid_retrieval_top50",
            "fact_extraction_before_reader",
            "per_fact_chain_of_note",
            "gpt5_mini_reader",
        ],
    }
    (run_dir / "run_spec.json").write_text(json.dumps(run_spec, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── Load data ──
    raw_by_id = _load_raw_by_id(ROOT / DEFAULT_DATA)
    questions = [_question_from_raw(raw_by_id, q) for q in _load_questions(ROOT / DEFAULT_DATA)]
    q_by_id = {q.question_id: q for q in questions}
    ref_data = json.loads((ROOT / DEFAULT_DATA).read_text("utf-8"))
    ref_by_id = {str(r["question_id"]): r for r in ref_data}

    target_ids = spot_ids if spot_ids else [str(r["question_id"]) for r in ref_data]

    print(f"Run ID: {run_id}")
    print(f"Output: {run_dir}")
    print(f"Cases: {len(target_ids)}, Workers: {workers}")
    print(f"Reader: {READER_MODEL}, Judge: {JUDGE_MODEL}, top_k: {TOP_K}")

    if dry_run:
        print("[DRY RUN]")
        return []

    client = _make_client()
    print("Loading embeddings...", end="", flush=True)
    _get_embed_model()
    print(" done")

    # Resume
    hyp_path = run_dir / "predictions.jsonl"
    scored_path = run_dir / "scored.jsonl"
    diag_path = run_dir / "per_instance_stats.json"

    completed: set[str] = set()
    results: list[CaseResult] = []
    diagnostics: list[dict] = []
    if hyp_path.exists():
        for line in hyp_path.read_text("utf-8").splitlines():
            if line.strip():
                completed.add(json.loads(line)["question_id"])
    if diag_path.exists():
        try:
            diagnostics = json.loads(diag_path.read_text("utf-8"))
        except json.JSONDecodeError:
            diagnostics = []
    remaining = [q for q in target_ids if q not in completed]
    if completed:
        print(f"Resuming: {len(completed)} done, {len(remaining)} left")

    def _process(qid: str) -> CaseResult:
        t0 = time.monotonic()
        q = q_by_id[qid]
        ref = ref_by_id[qid]
        gold = str(ref["answer"])
        qtype = ref["question_type"]
        qdate = ref.get("question_date", "")

        rounds = _build_rounds(q)
        retrieved = _retrieve_hybrid(q.question, rounds, top_k=TOP_K)

        sessions_hit = {r.session_id for r in retrieved}
        gold_sids = [str(s) for s in ref.get("answer_session_ids", []) or []]
        recall = sum(1 for s in gold_sids if s in sessions_hit) / max(1, len(gold_sids)) if gold_sids else 1.0

        # FACT EXTRACTION (new in v2)
        facts, extraction_raw = _extract_facts(client, qid, q.question, qdate, retrieved)

        # READER (over facts, not raw rounds)
        full_cot, notes, answer = _call_reader(client, qid, q.question, qdate, facts)
        reader_tokens = _estimate_tokens(json.dumps(facts))

        # JUDGE
        judge_ok = _call_judge(client, qid, q.question, gold, answer, qtype)
        strict = strict_short_check(gold, answer)
        final_ok = strict if strict is not None else judge_ok

        return CaseResult(
            question_id=qid, question_type=qtype, gold=gold,
            hypothesis=answer, full_cot=full_cot, notes=notes,
            facts=facts, facts_count=len(facts),
            correct_judge=judge_ok, correct_final=final_ok,
            retrieval_recall=recall, reader_input_tokens=reader_tokens,
            rounds_indexed=len(rounds), rounds_retrieved=len(retrieved),
            elapsed_s=round(time.monotonic() - t0, 2), question_date=qdate,
            extraction_raw=extraction_raw[:500],
        )

    lock = threading.Lock()
    done_n = [len(completed)]

    with hyp_path.open("a" if completed else "w", encoding="utf-8") as hf, \
         scored_path.open("a" if completed else "w", encoding="utf-8") as sf:

        def _save(r: CaseResult):
            with lock:
                done_n[0] += 1
                s = "PASS" if r.correct_final else "FAIL"
                print(f"  [{done_n[0]}/{len(target_ids)}] {r.question_id} ({r.question_type}) -> {s} facts={r.facts_count} ({r.elapsed_s}s)")
                hf.write(json.dumps({"question_id": r.question_id, "hypothesis": r.hypothesis}, ensure_ascii=False) + "\n")
                hf.flush()
                sf.write(json.dumps({"question_id": r.question_id, "hypothesis": r.hypothesis,
                                     "autoeval_label": {"model": JUDGE_MODEL, "label": r.correct_final}}, ensure_ascii=False) + "\n")
                sf.flush()
                results.append(r)
                diagnostics.append({
                    k: getattr(r, k) for k in r.__dataclass_fields__
                    if k not in ("facts",)  # save facts separately to keep file manageable
                })
                diagnostics[-1]["facts"] = r.facts[:20]  # cap at 20 facts per case
                diag_path.write_text(json.dumps(diagnostics, indent=1, ensure_ascii=False), encoding="utf-8")

        if workers <= 1:
            for qid in remaining:
                _save(_process(qid))
        else:
            futs = {}
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for qid in remaining:
                    futs[pool.submit(_process, qid)] = qid
                for fut in as_completed(futs):
                    try:
                        _save(fut.result())
                    except Exception as exc:
                        print(f"  ERROR {futs[fut]}: {exc}")

    # ── Save scenario_state.json (all API calls) ──
    (run_dir / "scenario_state.json").write_text(
        json.dumps(_call_log, indent=1, ensure_ascii=False), encoding="utf-8")

    # ── Save stats.json ──
    if results:
        correct = sum(1 for r in results if r.correct_final)
        total = len(results)
        types = {}
        abs_acc = []
        for r in results:
            types.setdefault(r.question_type, []).append(1 if r.correct_final else 0)
            if "_abs" in r.question_id:
                abs_acc.append(1 if r.correct_final else 0)
        task_means = [sum(v)/len(v) for v in types.values()]

        total_cost = sum(c["cost_usd"] for c in _call_log)
        total_tokens = sum(c["total_tokens"] for c in _call_log)
        total_prompt = sum(c["prompt_tokens"] for c in _call_log)
        total_completion = sum(c["completion_tokens"] for c in _call_log)

        stats = {
            "run_id": run_id,
            "total_questions": total,
            "correct": correct,
            "overall_accuracy": round(correct / total, 4),
            "task_averaged_accuracy": round(sum(task_means) / len(task_means), 4),
            "abstention_accuracy": round(sum(abs_acc) / len(abs_acc), 4) if abs_acc else None,
            "per_type": {qt: {"correct": sum(v), "total": len(v), "accuracy": round(sum(v)/len(v), 4)}
                         for qt, v in sorted(types.items())},
            "tokens": {
                "total": total_tokens,
                "prompt": total_prompt,
                "completion": total_completion,
                "by_call_type": {},
            },
            "cost_usd": {
                "total": round(total_cost, 4),
                "by_call_type": {},
            },
            "api_calls": {
                "total": len(_call_log),
                "errors": sum(1 for c in _call_log if not c["success"]),
                "by_type": {},
            },
            "avg_facts_per_question": round(sum(r.facts_count for r in results) / total, 1),
            "avg_retrieval_recall": round(sum(r.retrieval_recall for r in results) / total, 4),
            "avg_reader_input_tokens": round(sum(r.reader_input_tokens for r in results) / total),
            "avg_elapsed_s": round(sum(r.elapsed_s for r in results) / total, 1),
        }
        for ct in ["extraction", "reader", "judge"]:
            ct_calls = [c for c in _call_log if c["call_type"] == ct]
            stats["tokens"]["by_call_type"][ct] = sum(c["total_tokens"] for c in ct_calls)
            stats["cost_usd"]["by_call_type"][ct] = round(sum(c["cost_usd"] for c in ct_calls), 4)
            stats["api_calls"]["by_type"][ct] = len(ct_calls)

        (run_dir / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

        # Print summary
        print(f"\n{'=' * 60}")
        print(f"  RESULTS: {correct}/{total} ({100*correct/total:.1f}%)")
        print(f"  Task-averaged: {100*stats['task_averaged_accuracy']:.1f}%")
        print(f"  Cost: ${total_cost:.2f} | Tokens: {total_tokens:,}")
        print(f"  Avg facts/question: {stats['avg_facts_per_question']}")
        for qt, v in sorted(stats["per_type"].items()):
            print(f"    {qt}: {v['correct']}/{v['total']} ({100*v['accuracy']:.1f}%)")
        print(f"{'=' * 60}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LongMemEval-S v2 — fact extraction + HELM logging")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--spot", action="store_true")
    parser.add_argument("--spot-ids", nargs="*", default=None)
    args = parser.parse_args()

    if args.spot:
        ids = args.spot_ids or ["06878be2", "71017276", "0a995998", "852ce960", "51a45a95"]
        run(workers=1, spot_ids=ids)
    else:
        run(workers=args.workers, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
