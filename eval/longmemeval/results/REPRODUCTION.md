# LongMemEval-S Final Run — Reproduction Guide

## Overview

This document contains everything needed to exactly reproduce the final LongMemEval-S benchmark run. Any engineer can follow these steps to recreate the results from scratch.

## Run Identity

- **Date**: 2026-04-26
- **Branch**: `longmemval`
- **Commit**: `1db0060565074b1696377adc1689a1f231e7d19e`
- **Runner**: `eval/longmemeval/final_runner.py`
- **Manifest**: `eval/longmemeval/results/final_run_manifest.pre.json`

## Models Used

| Role | Model | Provider | Purpose |
|------|-------|----------|---------|
| Reader | `openai/gpt-5-mini` | OpenRouter | Reads retrieved evidence, produces answer via chain-of-thought |
| Judge | `openai/gpt-4o-2024-11-20` | OpenRouter | Official LongMemEval judge (scores reader answers) |
| Embeddings | `all-MiniLM-L6-v2` | Local (sentence-transformers) | Dense retrieval signal |

## Improvements Over Baseline (65.8%)

| # | Improvement | What it does | Research basis |
|---|-------------|-------------|----------------|
| 1 | **Temporal context** | Injects `question_date` from dataset + computes relative offsets per excerpt ("3 weeks ago") + temporal gap markers between sessions | Mastra OM three-date model; LongMemEval paper time-aware query expansion |
| 2 | **Dense+BM25 hybrid retrieval** | Replaces TF-IDF with semantic embeddings (all-MiniLM-L6-v2) + BM25 keyword matching, fused 0.7/0.3 | Mem0 triple scoring; LongMemEval paper §4 (dense > sparse) |
| 3 | **Per-round Chain-of-Note** | Reader writes compact relevant-only notes from evidence before answering (skip irrelevant excerpts) | Yu et al. 2023 (arXiv 2311.09210); LongMemEval paper §4 (+10pp with CoN+JSON) |
| 4 | **GPT-5-mini reader** | Stronger reader model for long-context synthesis | Mastra OM: 84%→95% from gpt-4o→gpt-5-mini with zero architecture change |

## Dataset

- **File**: `eval/longmemeval/data/longmemeval_s_cleaned.json`
- **Source**: LongMemEval-S (Wu et al., ICLR 2025, arXiv 2410.10813)
- **Questions**: 500 total
- **Type counts**: single-session-user=70, single-session-assistant=56, single-session-preference=30, multi-session=133, temporal-reasoning=133, knowledge-update=78
- **Abstention**: 30 questions (marked with `_abs` in question_id)
- **All 500 questions have `question_date` field**

## Official Scorer

- **File**: `eval/longmemeval/official_evaluate_qa.py`
- **Logic**: Task-specific GPT-4o judge prompts (7 branches: 6 question types + abstention)
- **Additional**: Strict short-answer scorer cleanup for gold answers ≤3 tokens

## Environment Setup

```bash
# Python 3.12+
pip install openai sentence-transformers scikit-learn numpy

# API key
export OPENROUTER_API_KEY=<your-key>
export ACTIVE_PACK=personal_assistant
```

## Exact Reproduction Command

```bash
# Phase 0: Freeze manifest
python -m eval.longmemeval.final_runner --phase 0

# Phase 1: Legitimacy audit
python -m eval.longmemeval.final_runner --phase 1

# Phase 2: Validation
python -m eval.longmemeval.final_runner --phase 2

# Phase 3: Full 500-question run
python -m eval.longmemeval.final_runner --phase 3 --workers 10

# Phase 4: Post-run audit
python -m eval.longmemeval.final_runner --phase 4

# Or all at once:
python -m eval.longmemeval.final_runner --phase all --workers 10
```

## Output Artifacts

| File | Contents |
|------|----------|
| `final_full_500.jsonl` | 500 predictions: `{"question_id", "hypothesis"}` |
| `final_full_500_scored.jsonl` | 500 scored: `{"question_id", "hypothesis", "autoeval_label": {"model", "label"}}` |
| `final_full_500_diagnostics.json` | 500 case details: question, gold, hypothesis, full CoT output, notes, judge result, retrieval recall, tokens, timing, question_date |
| `final_run_manifest.pre.json` | Pre-run manifest: git hash, file hashes, prompt hash, model names, retrieval config |
| `final_full_500_manifest.json` | Post-run manifest: all metrics, per-type accuracy, retrieval stats |
| `REPRODUCTION.md` | This file |

## Verification

```bash
# Check completeness
wc -l eval/longmemeval/results/final_full_500.jsonl          # should be 500
wc -l eval/longmemeval/results/final_full_500_scored.jsonl    # should be 500

# Official metrics (independent verification)
python eval/longmemeval/official_print_metrics.py \
  eval/longmemeval/results/final_full_500_scored.jsonl \
  eval/longmemeval/data/longmemeval_s_cleaned.json
```

## Pipeline Detail

For each of the 500 questions:

1. **Load question** from `longmemeval_s_cleaned.json` (includes `question_date`, `haystack_sessions`, `haystack_dates`)
2. **Build rounds** from all haystack sessions — pair user+assistant messages into conversation rounds with session_id and timestamp metadata
3. **Hybrid retrieval** — encode all round keys with `all-MiniLM-L6-v2`, compute cosine similarity + TF-IDF BM25 scores, normalize both to [0,1], fuse with 0.7 semantic + 0.3 BM25 weights, take top 30
4. **Sort chronologically** by timestamp, then session_order, then round_index
5. **Format evidence** — each excerpt gets: timestamp, relative offset from question_date (e.g., "3 weeks ago"), session_id, temporal gap markers between non-adjacent sessions
6. **Reader call** (GPT-5-mini) — single-call chain-of-thought: model writes relevant-only notes then answers. System prompt instructs: use relative offsets for temporal reasoning, prefer most recent values for updates, count all items across all excerpts, return stored preferences not generic advice
7. **Parse answer** — extract ANSWER section from CoT output
8. **Judge call** (GPT-4o) — official `get_anscheck_prompt` with task-specific rubrics
9. **Scorer cleanup** — for short gold answers (≤3 tokens), strict normalized boundary matching overrides judge on numeric conflicts

## Reader System Prompt

```
You are answering a question about a user based on their past conversation history.

You receive the question, the date it was asked, and retrieved conversation excerpts
with timestamps and relative time offsets.

Step 1 — NOTES: Read ALL excerpts. Write a short list of ONLY the relevant facts
(skip irrelevant excerpts entirely). For each relevant fact, note which excerpt it
came from and its date.

Step 2 — ANSWER: Using your notes, answer the question.

Guidelines:
- Answer from provided evidence only.
- Return stored preferences/constraints, not generic advice.
- Use the relative time offsets (e.g., '3 weeks ago') to compute durations and recency.
- When values change over time, prefer the most recent one.
- Count and list ALL relevant items across ALL excerpts before answering count questions.
- Only say "I don't know" if evidence truly has no relevant information.
- Be concise.

Format:
NOTES:
- [relevant fact 1, from excerpt N, date]
- [relevant fact 2, from excerpt M, date]
...

ANSWER:
[your concise answer]
```

## Cost

~$12-15 for 500 questions via OpenRouter (GPT-5-mini reader + GPT-4o judge).

## Baseline Comparison

| System | Score | Reader |
|--------|-------|--------|
| Mem0 | 93.4% | gpt-5-mini |
| Mastra OM | 94.87% | gpt-5-mini |
| EverMemOS | 83.0% | — |
| TiMem | 76.9% | gpt-4o-mini |
| Zep/Graphiti | 71.2% | gpt-4o |
| Full-context GPT-4o (CoN) | 64.0% | gpt-4o |
| **Our baseline** | **65.8%** | gpt-4o |
| **Our final run** | **TBD** | gpt-5-mini |
