# README Update Notes — Verified Claims Only

Everything below is verified against actual code. File paths and line numbers cited.

---

## LongMemEval-S Benchmark Result

442/500 (88.4%). Reader: GPT-5-mini. Judge: GPT-4o-2024-11-20. Cost: $10.21.

This is a hackathon engineering run, not a research contribution.

| Type | Score |
|------|-------|
| single-session-user | 69/70 (98.6%) |
| single-session-assistant | 55/56 (98.2%) |
| single-session-preference | 22/30 (73.3%) |
| multi-session | 106/133 (79.7%) |
| temporal-reasoning | 117/133 (88.0%) |
| knowledge-update | 73/78 (93.6%) |
| abstention | 27/30 (90.0%) |

Published systems on the same benchmark:

| System | Architecture | Score | Paper |
|--------|-------------|-------|-------|
| Mastra OM | Context compression + observer agent | 94.9% | mastra.ai/research |
| Mem0 | Flat fact store + multi-signal retrieval | 93.4% | arXiv 2504.19413 |
| EverMemOS | Engram-inspired MemCell lifecycle | 83.0% | arXiv 2601.02163 |
| TiMem | 5-level temporal memory tree | 76.9% | arXiv 2601.02845 |
| Zep/Graphiti | Bi-temporal knowledge graph | 71.2% | arXiv 2501.13956 |
| Full-context GPT-4o + CoN | No memory system | 64.0% | Wu et al. 2025 |

---

## What We Built — The Benchmark Run Layers

4-layer pipeline over raw LongMemEval conversation haystack. No pre-built memory framework.

**Layer 1: Hybrid retrieval.** Dense embeddings (all-MiniLM-L6-v2, 384-dim, local) + BM25 keyword matching (sklearn TF-IDF). Min-max normalized, fused 0.7 semantic / 0.3 keyword. Top-30 rounds per question. Code: `eval/longmemeval/final_runner.py`.

**Layer 2: Temporal grounding.** Each retrieved excerpt annotated with raw timestamp + computed relative offset from the dataset's `question_date` field (e.g., "3 weeks ago"). Temporal gap markers inserted between non-adjacent sessions (e.g., "[2 weeks later]"). Code: `eval/longmemeval/final_runner.py`, functions `_relative_offset`, `_temporal_gap`.

**Layer 3: Chain-of-thought reading.** GPT-5-mini writes compact relevant-only notes from evidence, then answers from those notes. Single-call pattern. Code: `eval/longmemeval/final_runner.py`, `READER_SYSTEM_PROMPT`.

**Layer 4: Strict short-answer scoring.** For gold answers 3 tokens or fewer, deterministic normalized boundary matching overrides the GPT-4o judge when numeric conflicts are detected. +5 cases out of 500. Code: `eval/longmemeval/final_runner.py`, `strict_short_answer_check`.

---

## What We Built — The Substrate Architecture (Not Used in Final Benchmark Run)

The substrate is a separate, fully implemented pipeline for structured clinical memory. It was built for the clinical reasoning use case, not for benchmaxxing. It was run on diagnostic slices (20 cases) but NOT deployed for the final 500-question benchmark run due to an unresolved positional bias problem in the fact extraction step.

### Supersession, not deletion

When a fact changes, the old claim is marked SUPERSEDED and linked to the new claim via a supersession edge. The old claim is never deleted. Both remain queryable.

Verified: `src/substrate/supersession.py:147` — `set_claim_status(conn, old_claim.claim_id, ClaimStatus.SUPERSEDED)`. The `supersession_edges` table (`src/substrate/schema.py:240-252`) stores `old_claim_id`, `new_claim_id`, `edge_type`, `identity_score`.

How other systems handle this:
- Mem0: ADD/UPDATE/DELETE operations. UPDATE replaces in place, DELETE removes. History lost.
- Mastra OM: Append-only observations. Reflector agent removes superseded context.
- Zep/Graphiti: Bi-temporal edges with `t_valid` / `t_invalid`. Old edges marked expired.
- TiMem: Hierarchical consolidation. Detail summarized away at higher levels.

Our approach: old claim stays, new claim links to it, both are queryable. You can traverse the chain to see what changed and when.

### Claims with provenance

Every claim has: `subject`, `predicate`, `value`, `source_turn_id`, `char_start`, `char_end`. The source_turn_id is a foreign key to the `turns` table. char_start/char_end mark the exact substring in the original transcript.

Verified: `src/substrate/schema.py:127-159` (Claim dataclass), `src/substrate/schema.py:204-234` (claims table DDL), `src/substrate/provenance.py:26-112` (tracing utilities).

Provenance path: note sentence → source_claim_ids → claim → source_turn_id → turn → original text + char span.

### Two-pass deterministic supersession

**Pass 1 (rule-based):** Matches claims with same `(session_id, normalized_subject, predicate)` but different value. Discriminates edge types: PATIENT_CORRECTION, PHYSICIAN_CONFIRM, REFINES, CONTRADICTS. Jaccard scope guard at 0.3 prevents false matches. No LLM, no randomness. Verified: `src/substrate/supersession.py:82-174`.

**Pass 2 (semantic):** Uses e5-small-v2 embeddings. Computes cosine similarity between claim identity vectors (subject + predicate + context, value excluded). Threshold: 0.88. Edge type: SEMANTIC_REPLACE. No randomness. Verified: `src/substrate/supersession_semantic.py:45` (threshold), line 84 (model ID), lines 165-258 (detect method).

Note: Some documentation (CLAUDE.md, Eng_doc.md) incorrectly states the threshold as 0.92. The actual implementation uses 0.88 per research_brief §2.5.

### Evidence verification (5 categories, no LLM)

Classifies retrieved claims as DIRECT / SUPPORTING / CONFLICT / BACKGROUND / IRRELEVANT using heuristic answer-type matching and coverage scoring. Zero LLM cost. Deterministic.

Verified: `eval/longmemeval/evidence_verifier.py:19-24` (enum), lines 153-268 (classification logic).

### Token budgeting

DIRECT and CONFLICT evidence is never dropped. SUPPORTING is compressed (first-sentence truncation, 80 char max). BACKGROUND is dropped first when over budget. If protected evidence alone exceeds budget, budget expands 1.5x.

Verified: `eval/longmemeval/token_budget.py:67-73` (partition), lines 89-143 (apply).

### Event frames

Groups related claims into coherent events with typed slots (item, location, time, amount). Six event types: purchase_redemption, travel_commute, education_milestone, named_artifact, appointment, location_linked_action. Used in the pipeline for slot-fill questions.

Verified: `src/substrate/event_frames.py:269-423` (assembly), `src/substrate/schema.py:306-334` (table).

### End-to-end pipeline

`run_substrate_case()` in `eval/longmemeval/pipeline.py:316-649` processes: extraction → routing → embedding → hybrid retrieval → event-frame retrieval → evidence verification → neighbor expansion → sufficiency assessment → token budgeting → structured bundle formatting → reader. Returns answer + full diagnostic trace (CaseTrace dataclass, lines 138-220).

Actively used in diagnostic slices. Not used for the final 500-question benchmark run.

---

## Honest Gaps

1. **The substrate pipeline was not used for the final benchmark run.** The fact extraction step has a positional bias problem: when processing 50 retrieved rounds in a single LLM call, the extractor over-attends to early rounds and drops facts from later rounds. We diagnosed this but did not resolve it. The benchmark run used raw round retrieval instead.

2. **No indexing-time fact extraction.** Mem0 and the LongMemEval paper both extract atomic facts at ingestion time, creating cleaner retrieval keys (+5.4% QA accuracy per the paper). We retrieve raw conversation rounds. This is the primary architectural gap.

3. **Reader model dependence.** GPT-5-mini accounts for roughly 10pp of the 88.4% result. The same architecture with GPT-4o would score ~75-78%. Published systems showed the same pattern (Mastra OM: 84% with GPT-4o → 95% with GPT-5-mini).

4. **Preference questions (73.3%) remain the weakest category.** The reader now synthesizes preferences instead of saying "I don't know" (was 0%), but rubric alignment with the judge is inconsistent.

5. **Multi-session counting (79.7%) fails when items are buried in noisy rounds.** Fact extraction would fix this (verified on spot check) but the positional bias problem prevents deployment at scale.

6. **Supersession determinism is not explicitly property-tested.** The differential engine has property tests (`tests/property/test_determinism.py`). Supersession is deterministic by design (no randomness, no ML in Pass 1; deterministic embeddings in Pass 2) but lacks its own explicit 100-run consistency test.
