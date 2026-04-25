"""3-Stage MedXpertQA Benchmark: RAG + Qwen3-32B + Opus 4.7 Batch API.

Architecture:
  Stage 1 (RAG): Local FAISS retrieval of MedQA training passages
  Stage 2 (Qwen3-32B): Option elimination (10 -> 4 survivors)
  Stage 3 (Opus 4.7): Final answer selection via Batch API

Usage:
  # Build RAG index (one-time, ~20 min on CPU)
  python -m eval.medxpertqa.run_3stage --step rag-build

  # Run Qwen elimination (Modal A100)
  python -m eval.medxpertqa.run_3stage --step qwen-elim --limit 10

  # Submit Opus batch
  python -m eval.medxpertqa.run_3stage --step batch-submit --batch-type enhanced

  # Check batch status
  python -m eval.medxpertqa.run_3stage --step batch-check

  # Score results
  python -m eval.medxpertqa.run_3stage --step score
"""
from __future__ import annotations

import argparse
import ast
import concurrent.futures
import json
import os
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO / "eval" / "reports" / "medxpertqa" / "3stage"

LETTERS = "ABCDEFGHIJ"

_THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)


# ---------------------------------------------------------------------------
# Shared helpers (inlined to avoid import issues)
# ---------------------------------------------------------------------------

def parse_options(raw) -> dict[str, str]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return {}
    return {}


def format_options(opts: dict[str, str]) -> str:
    return "\n".join(f"{k}. {v}" for k, v in sorted(opts.items()))


def format_surviving_options(opts: dict[str, str], survivors: list[str]) -> str:
    return "\n".join(f"{k}. {opts[k]}" for k in survivors if k in opts)


def extract_answer(text: str) -> str:
    cleaned = _THINK_RE.sub("", text).strip().upper()
    patterns = [
        r"(?:FINAL\s+ANSWER)\s*(?:IS|:|=|-)*\s*\(?([A-J])\)?",
        r"(?:^|\n)\s*\*{0,2}ANSWER\*{0,2}\s*(?:IS|:|=|-)\s*\(?([A-J])\)?",
        r"(?:THE\s+ANSWER\s+IS|ANSWER\s*:|ANSWER\s+IS)\s+\(?([A-J])\)?",
        r"^\s*\(?([A-J])\)?\s*\.?\s*$",
        r"\b([A-J])\b",
    ]
    for pattern in patterns:
        matches = list(re.finditer(pattern, cleaned, re.MULTILINE))
        if matches:
            return matches[-1].group(1)
    for char in reversed(cleaned):
        if char in LETTERS:
            return char
    return ""


def parse_elimination(text: str) -> list[str]:
    cleaned = _THINK_RE.sub("", text).strip()
    fence = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", cleaned, re.DOTALL)
    source = fence.group(1) if fence else cleaned

    for candidate in [source, cleaned]:
        try:
            data = json.loads(candidate)
            if isinstance(data, dict) and "survivors" in data:
                return [s.upper() for s in data["survivors"] if s.upper() in LETTERS]
        except (json.JSONDecodeError, TypeError):
            continue

    letters = re.findall(r'"([A-J])"', cleaned)
    if letters:
        return list(dict.fromkeys(letters))[:5]

    return list(LETTERS[:4])


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

QWEN_ELIMINATION = """\
You are a board-certified physician taking a medical licensing exam.

CLINICAL CASE:
{question}

RELEVANT MEDICAL KNOWLEDGE:
{rag_context}

OPTIONS:
{options}

TASK: Eliminate exactly 4 options that are CLEARLY wrong. Keep 6 options that could be plausible.
For each eliminated option, give ONE SENTENCE explaining why.
List the 6 SURVIVING options.

Output as JSON:
{{"survivors": ["A", "C", "D", "E", "G", "H"], "eliminated": {{"B": "reason", "F": "reason", "I": "reason", "J": "reason"}}}}"""

OPUS_ENHANCED_SYSTEM = """\
You are a medical expert. An elimination model analyzed this case and narrowed the options. Consider the elimination reasoning carefully. Select the single best answer. Answer with the letter only."""

OPUS_ENHANCED_USER = """\
CLINICAL CASE:
{question}

RELEVANT MEDICAL KNOWLEDGE:
{rag_context}

ELIMINATION ANALYSIS:
{elimination_reasoning}

REMAINING OPTIONS:
{surviving_options}

Answer with the letter only."""

OPUS_BASELINE_SYSTEM = """\
You are a medical expert. Read the clinical case and select the single best answer. Answer with the letter only."""

OPUS_BASELINE_USER = """\
{question}

Options:
{options}

Answer with the letter only."""


# ---------------------------------------------------------------------------
# Qwen3-32B caller (Modal endpoint)
# ---------------------------------------------------------------------------

def call_qwen(prompt: str, endpoint_url: str, max_tokens: int = 4096) -> str:
    from openai import OpenAI
    client = OpenAI(base_url=endpoint_url + "/v1", api_key="dummy")
    resp = client.chat.completions.create(
        model="Qwen/Qwen3-32B",
        messages=[
            {"role": "system", "content": "You are a medical expert. Respond with JSON only. Do not explain your reasoning."},
            {"role": "user", "content": prompt + "\n\n/nothink"},
        ],
        max_tokens=max_tokens,
        temperature=0.7,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return resp.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Step: Build RAG index
# ---------------------------------------------------------------------------

def step_rag_build(args):
    from eval.medxpertqa.rag_index import build_index
    n = build_index()
    print(f"RAG index built with {n} passages")
    return 0


# ---------------------------------------------------------------------------
# Step: Qwen elimination (RAG + Qwen3-32B)
# ---------------------------------------------------------------------------

def step_qwen_elim(args):
    from datasets import load_dataset
    from eval.medxpertqa.rag_index import retrieve, format_rag_context, load_index

    ds = load_dataset("TsinghuaC3I/MedXpertQA", "Text", split="test")
    limit = args.limit or len(ds)

    endpoint = args.qwen_endpoint
    if not endpoint:
        print("ERROR: --qwen-endpoint required (Modal URL)", file=sys.stderr)
        return 1

    rag_available = load_index()
    if rag_available:
        print("RAG index loaded")
    else:
        print("WARNING: RAG index not found, running 2-stage (no RAG)")

    out_dir = RESULTS_DIR / "elimination"
    out_dir.mkdir(parents=True, exist_ok=True)

    results_file = out_dir / "results.jsonl"
    completed = set()
    if results_file.exists():
        with open(results_file, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                completed.add(r["index"])
        print(f"Resuming: {len(completed)} already done")

    print(f"Running Qwen3-32B elimination on {min(limit, len(ds))} cases")
    print(f"Endpoint: {endpoint}")

    gold_survived = 0
    total_done = 0

    def process_case(idx):
        if idx in completed:
            return None

        case = ds[idx]
        question = case["question"]
        opts = parse_options(case["options"])
        gold = case.get("label", "")
        opts_str = format_options(opts)

        # Stage 1: RAG
        if rag_available:
            passages = retrieve(question, top_k=3)
            rag_ctx = format_rag_context(passages)
        else:
            rag_ctx = "(No medical knowledge retrieval available)"

        # Stage 2: Qwen elimination
        prompt = QWEN_ELIMINATION.format(
            question=question,
            rag_context=rag_ctx,
            options=opts_str,
        )

        result = {
            "index": idx,
            "id": case.get("id", str(idx)),
            "gold": gold,
            "rag_passages": len(passages) if rag_available else 0,
        }

        try:
            raw = call_qwen(prompt, endpoint, max_tokens=2048)
            survivors = parse_elimination(raw)
            result["elimination_raw"] = raw
            result["survivors"] = survivors
            result["gold_survived"] = gold in survivors
        except Exception as e:
            print(f"  [{idx}] ERROR: {e}")
            result["elimination_raw"] = ""
            result["survivors"] = list(LETTERS[:4])
            result["gold_survived"] = gold in result["survivors"]
            result["error"] = str(e)

        return result

    # Sequential processing (safe for resume)
    with open(results_file, "a", encoding="utf-8") as f:
        for idx in range(min(limit, len(ds))):
            if idx in completed:
                continue

            result = process_case(idx)
            if result is None:
                continue

            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            total_done += 1

            if result.get("gold_survived"):
                gold_survived += 1

            gs = "Y" if result.get("gold_survived") else "N"
            surv = result.get("survivors", [])
            print(f"  [{idx}/{limit}] gold={result['gold']} survivors={surv} gold_survived={gs}")

            if total_done % 50 == 0:
                rate = gold_survived / total_done if total_done else 0
                print(f"\n  === Progress: {total_done} done, "
                      f"gold survival: {gold_survived}/{total_done} ({rate:.0%}) ===\n")

    total = total_done + len(completed)
    rate = gold_survived / total_done if total_done else 0
    print(f"\n{'='*60}")
    print(f"Qwen elimination complete: {total} cases")
    print(f"Gold survival rate: {gold_survived}/{total_done} ({rate:.0%})")
    print(f"Results: {results_file}")
    print(f"{'='*60}")
    return 0


# ---------------------------------------------------------------------------
# Step: Submit Opus batch
# ---------------------------------------------------------------------------

def step_batch_submit(args):
    import anthropic
    from datasets import load_dataset
    from eval.medxpertqa.rag_index import retrieve, format_rag_context, load_index

    batch_type = args.batch_type or "enhanced"

    ds = load_dataset("TsinghuaC3I/MedXpertQA", "Text", split="test")

    if batch_type == "enhanced":
        elim_file = RESULTS_DIR / "elimination" / "results.jsonl"
        if not elim_file.exists():
            print("ERROR: Run --step qwen-elim first", file=sys.stderr)
            return 1

        elim_data = {}
        with open(elim_file, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                elim_data[r["index"]] = r
        print(f"Loaded {len(elim_data)} elimination results")

        rag_available = load_index()
        if rag_available:
            print("RAG index loaded for enhanced prompts")

    limit = args.limit
    indices = sorted(elim_data.keys()) if batch_type == "enhanced" else list(range(len(ds)))
    if limit:
        indices = indices[:limit]

    requests = []
    for idx in indices:
        case = ds[idx]
        question = case["question"]
        opts = parse_options(case["options"])

        if batch_type == "enhanced":
            ed = elim_data[idx]
            survivors = ed.get("survivors", list(LETTERS[:4]))
            surv_opts = format_surviving_options(opts, survivors)
            elim_raw = ed.get("elimination_raw", "")

            if rag_available:
                passages = retrieve(question, top_k=3)
                rag_ctx = format_rag_context(passages)
            else:
                rag_ctx = "(No medical knowledge retrieval available)"

            user_msg = OPUS_ENHANCED_USER.format(
                question=question,
                rag_context=rag_ctx,
                elimination_reasoning=elim_raw,
                surviving_options=surv_opts,
            )
            system_text = OPUS_ENHANCED_SYSTEM
        else:
            user_msg = OPUS_BASELINE_USER.format(
                question=question,
                options=format_options(opts),
            )
            system_text = OPUS_BASELINE_SYSTEM

        requests.append({
            "custom_id": f"case-{idx}",
            "params": {
                "model": "claude-opus-4-7",
                "max_tokens": 256,
                "system": [
                    {
                        "type": "text",
                        "text": system_text,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                "messages": [{"role": "user", "content": user_msg}],
            },
        })

    print(f"Submitting {batch_type} batch: {len(requests)} requests")
    client = anthropic.Anthropic()
    batch = client.messages.batches.create(requests=requests)
    print(f"Batch ID: {batch.id}")
    print(f"Status: {batch.processing_status}")

    meta = {
        "batch_id": batch.id,
        "batch_type": batch_type,
        "n_requests": len(requests),
        "submitted_at": time.time(),
    }
    meta_file = RESULTS_DIR / f"batch_{batch_type}_meta.json"
    meta_file.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Meta saved: {meta_file}")
    return 0


# ---------------------------------------------------------------------------
# Step: Check batch status
# ---------------------------------------------------------------------------

def step_batch_check(args):
    import anthropic

    batch_id = args.batch_id
    if not batch_id:
        for bt in ["enhanced", "baseline"]:
            meta_file = RESULTS_DIR / f"batch_{bt}_meta.json"
            if meta_file.exists():
                with open(meta_file, encoding="utf-8") as f:
                    meta = json.load(f)
                batch_id = meta["batch_id"]
                print(f"Found {bt} batch: {batch_id}")
                break
    if not batch_id:
        print("ERROR: --batch-id required or run --step batch-submit first")
        return 1

    client = anthropic.Anthropic()
    batch = client.messages.batches.retrieve(batch_id)
    print(f"Batch: {batch.id}")
    print(f"Status: {batch.processing_status}")
    print(f"Counts: {batch.request_counts}")
    return 0


# ---------------------------------------------------------------------------
# Step: Score results
# ---------------------------------------------------------------------------

def step_score(args):
    import anthropic
    from datasets import load_dataset

    ds = load_dataset("TsinghuaC3I/MedXpertQA", "Text", split="test")

    # Build gold map from dataset
    gold_map = {}
    for idx in range(len(ds)):
        gold_map[f"case-{idx}"] = ds[idx].get("label", "")

    # Load elimination data for gold survival stats
    elim_file = RESULTS_DIR / "elimination" / "results.jsonl"
    elim_data = {}
    if elim_file.exists():
        with open(elim_file, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                elim_data[r["index"]] = r

    client = anthropic.Anthropic()

    for batch_type in ["enhanced", "baseline"]:
        meta_file = RESULTS_DIR / f"batch_{batch_type}_meta.json"
        if not meta_file.exists():
            continue

        with open(meta_file, encoding="utf-8") as f:
            meta = json.load(f)
        batch_id = meta["batch_id"]

        batch = client.messages.batches.retrieve(batch_id)
        if batch.processing_status != "ended":
            print(f"{batch_type} batch {batch_id}: {batch.processing_status} (not ready)")
            continue

        results = []
        correct = 0
        total = 0

        for result in client.messages.batches.results(batch_id):
            custom_id = result.custom_id
            gold = gold_map.get(custom_id, "")
            idx = int(custom_id.split("-")[1])

            if result.result.type == "succeeded":
                text = ""
                for block in result.result.message.content:
                    if getattr(block, "type", "") == "text":
                        text = block.text
                        break
                pred = extract_answer(text)
                is_correct = pred == gold
                if is_correct:
                    correct += 1
            else:
                pred = ""
                text = ""
                is_correct = False

            total += 1
            entry = {
                "custom_id": custom_id,
                "index": idx,
                "gold": gold,
                "predicted": pred,
                "correct": is_correct,
                "raw": text[:300],
            }

            if batch_type == "enhanced" and idx in elim_data:
                entry["gold_survived"] = elim_data[idx].get("gold_survived", None)
                entry["survivors"] = elim_data[idx].get("survivors", [])

            results.append(entry)

        out_file = RESULTS_DIR / f"scored_{batch_type}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        pct = correct / total * 100 if total else 0
        print(f"\n{'='*60}")
        print(f"RESULTS -- {batch_type} ({total} cases)")
        print(f"{'='*60}")
        print(f"  Correct: {correct}/{total} ({pct:.1f}%)")

        if batch_type == "enhanced":
            gold_survived_count = sum(
                1 for r in results if r.get("gold_survived", True)
            )
            gs_pct = gold_survived_count / total * 100 if total else 0
            print(f"  Gold survived elimination: {gold_survived_count}/{total} ({gs_pct:.1f}%)")
            ceiling = gold_survived_count / total * 100 if total else 0
            print(f"  Accuracy ceiling (if all survived gold correct): {ceiling:.1f}%")

        print(f"  Results saved: {out_file}")
        print(f"{'='*60}")

    # Comparison
    enh_file = RESULTS_DIR / "scored_enhanced.json"
    bas_file = RESULTS_DIR / "scored_baseline.json"
    if enh_file.exists() and bas_file.exists():
        with open(enh_file, encoding="utf-8") as f:
            enh = json.load(f)
        with open(bas_file, encoding="utf-8") as f:
            bas = json.load(f)

        enh_correct = sum(1 for r in enh if r["correct"])
        bas_correct = sum(1 for r in bas if r["correct"])
        enh_total = len(enh)
        bas_total = len(bas)

        print(f"\n{'='*60}")
        print("COMPARISON")
        print(f"{'='*60}")
        print(f"  Baseline (Opus alone):  {bas_correct}/{bas_total} ({bas_correct/bas_total*100:.1f}%)")
        print(f"  Enhanced (RAG+Qwen+Opus): {enh_correct}/{enh_total} ({enh_correct/enh_total*100:.1f}%)")
        print(f"  Delta: {enh_correct - bas_correct:+d}")

        # Per-case comparison
        bas_map = {r["custom_id"]: r["correct"] for r in bas}
        helped = []
        hurt = []
        for r in enh:
            bc = bas_map.get(r["custom_id"], False)
            ec = r["correct"]
            if ec and not bc:
                helped.append(r["custom_id"])
            elif bc and not ec:
                hurt.append(r["custom_id"])

        print(f"  Helped: {len(helped)} cases")
        print(f"  Hurt:   {len(hurt)} cases")
        print(f"{'='*60}")

    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="3-Stage MedXpertQA Benchmark: RAG + Qwen3-32B + Opus 4.7"
    )
    parser.add_argument("--step", required=True,
                        choices=["rag-build", "qwen-elim", "batch-submit",
                                 "batch-check", "score"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--qwen-endpoint", default=None,
                        help="Modal Qwen3-32B endpoint URL (without /v1)")
    parser.add_argument("--batch-id", default=None)
    parser.add_argument("--batch-type", default="enhanced",
                        choices=["enhanced", "baseline"])
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.step == "rag-build":
        return step_rag_build(args)
    elif args.step == "qwen-elim":
        return step_qwen_elim(args)
    elif args.step == "batch-submit":
        return step_batch_submit(args)
    elif args.step == "batch-check":
        return step_batch_check(args)
    elif args.step == "score":
        return step_score(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
