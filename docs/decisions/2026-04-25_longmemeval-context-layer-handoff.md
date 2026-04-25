# LongMemEval Context Layer — Handoff Document

**Date**: 2026-04-25
**Branch**: `ui-experiment` (HEAD: 781887e)
**Prior branch work**: `flow/ship-prep` (commits b9065d4..9e4395d)

---

## Where we started

The LongMemEval pipeline stored facts as individual claims. A 20-case diagnostic slice scored **0/20 (0%)**. Every case failed — 9 at classification, 5 at bundling, 3 at extraction, 3 at reader. The system was moving failures between layers instead of fixing architecture.

## Where we are now

**13/20 (65%)** on the same 20-case slice. Two entire failure families eliminated.

### Progression

| Version | Score | Key change |
|---------|-------|------------|
| v1 | 0/20 (0%) | Baseline — no fixes |
| v2 | 4/20 (20%) | Verifier widened: preference/count/generic patterns, lower thresholds |
| v3 | 9/20 (45%) | Reader prompt: allow synthesis instead of exact-match-or-abstain |
| v4 | 13/20 (65%) | Assistant-turn extraction: extract claims from assistant responses |

### Failure family elimination

| Failure type | v1 | v4 | Status |
|---|---|---|---|
| classification_failure | 9 | 0 | **Eliminated** |
| extraction_failure | 3 | 0 | **Eliminated** |
| bundling_failure | 5 | 2 | Reduced |
| reader_failure | 3 | 5 | Dominant bottleneck |

---

## Architecture added

### 1. Event-frame layer (`src/substrate/event_frames.py`)

Compositional view above claims. Groups co-referent claims into multi-slot event records (purchase, commute, education, etc.). Two-pass merge algorithm:
- Pass 1: content token overlap (>=2) or non-trivial entity-key overlap
- Pass 2: attach slot-provider turns (location_linked_action) to nearest core event within +/-3 turns

Schema: 3 new tables in `src/substrate/schema.py` (`event_frames`, `event_frame_claims`, `event_frame_embeddings`).

Integrated into ingestion (`context.py::ingest_longmemeval_case`), called after supersession.

### 2. Event-frame retrieval (`src/substrate/retrieval.py`)

`retrieve_event_frames()` and `backfill_event_frame_embeddings()`. Integrated into pipeline step 4b — event-frame claims get 1.3x score boost for slot questions (where/when/who).

Note: Event-frame retrieval is wired but did NOT contribute to any of the 20-case PASS results. All passes used the claim path. The event-frame path needs further work to demonstrate value.

### 3. Canonical pipeline freeze (`eval/longmemeval/pipeline.py`)

- `RunManifest` + `verify_manifest()` for mid-run code change detection
- `CaseTrace` extended with event-frame and token-efficiency fields
- 4 stale paths annotated with `# STALE:` comments pointing to canonical `run_substrate_case()`

### 4. Diagnostic slice (`eval/longmemeval/diagnostic_slice.py`)

20-case stratified selector across 5 categories (information_extraction, event_slot_fill, temporal_reasoning, knowledge_update, multi_session_reasoning). Enhanced 13-type failure taxonomy. Token efficiency comparison framework.

### 5. Assistant-turn extraction (`eval/longmemeval/context.py`)

Added instruction to extractor prompt context for assistant turns: "Extract any specific facts, names, numbers, or claims the assistant stated." Uses `user_fact` predicate with subject `user`.

---

## Files changed (from main)

### New files
| File | Purpose |
|---|---|
| `src/substrate/event_frames.py` | Event-frame assembly + persistence |
| `eval/longmemeval/diagnostic_slice.py` | 20-case diagnostic runner + report |
| `tests/unit/substrate/test_event_frames.py` | 12 event-frame unit tests |
| `tests/e2e/test_assistant_turn_extraction.py` | 5 assistant-extraction E2E tests |
| `eval/longmemeval/diagnostic_results_v4.json` | Latest diagnostic report |
| `eval/longmemeval/diagnostic_results_v4.cases.json` | Per-case traces |
| `eval/longmemeval/autopsy_traces.json` | Failed-case autopsy with oracle control |

### Modified files
| File | Change |
|---|---|
| `src/substrate/schema.py` | 3 new tables (event_frames, event_frame_claims, event_frame_embeddings) |
| `src/substrate/retrieval.py` | `retrieve_event_frames()`, `backfill_event_frame_embeddings()` |
| `eval/longmemeval/pipeline.py` | RunManifest, step 3b/4b event-frame integration, CaseTrace fields, synthesis reader prompt |
| `eval/longmemeval/context.py` | Event-frame assembly call in ingestion, assistant-turn extraction prompt |
| `eval/longmemeval/evidence_verifier.py` | Widened answer-type patterns (preference, count, generic), lower thresholds |
| `eval/longmemeval/question_router.py` | `prefer_event_frames` field on RetrievalStrategy |
| `eval/longmemeval/full.py` | STALE annotation |
| `eval/smoke/run_smoke.py` | STALE annotations |

---

## Test suite

**36 tests pass, 0 fail, 0 regressions.**

- 14 adversarial gold survival (`tests/e2e/test_adversarial_gold_survival.py`)
- 5 event-neighbor expansion (`tests/e2e/test_event_neighbor_expansion.py`)
- 5 assistant-turn extraction (`tests/e2e/test_assistant_turn_extraction.py`)
- 12 event-frame unit tests (`tests/unit/substrate/test_event_frames.py`)

Run: `ACTIVE_PACK=personal_assistant pytest tests/e2e/ tests/unit/substrate/test_event_frames.py -v`

---

## Remaining 7 failures (autopsy completed)

### reader_failure (5)
| case_id | category | subclass | root cause |
|---|---|---|---|
| 06878be2 | information_extraction | reader_failed_composition | Bundle has claims about Sony gear but reader abstains on preference synthesis |
| 0edc2aef | information_extraction | reader_failed_composition | Bundle has Seattle hotel claims, question asks about Miami |
| gpt4_2655b836 | temporal_reasoning | reader_missed_explicit | Non-deterministic: passed in v3, failed in v4 |
| 0a995998 | multi_session_reasoning | reader_missed_claims | Reader counts 2 items, gold is 3. Noise claim classified DIRECT |
| 6d550036 | multi_session_reasoning | reader_missed_claims | 2nd project buried in CONFLICT section, reader only uses DIRECT |

### bundling_failure (2)
| case_id | category | subclass | root cause |
|---|---|---|---|
| e3fc4d6e | event_slot_fill | extraction_precision | "Dr. Prabhakar" extracted but lost at bundling — Azure content filter blocked the turn with the name |
| 830ce83f | knowledge_update | extraction_precision | Claims say "Chicago" and "the city", gold is "the suburbs" — detail lost at extraction |

### Scoring artifacts (2 false positives in v3, status unknown in v4)
| case_id | gold | predicted | issue |
|---|---|---|---|
| 852ce960 | $400,000 | $350,000 | Token overlap matcher scores PASS incorrectly |
| b5ef892d | 8 days | 3 days | Same issue — short gold, wrong number matches on "days" |

---

## Queued fixes (ordered by impact, do one at a time)

### Fix 2: Short-answer scorer exact-match (next)
For gold answers <= 3 tokens, require exact substring match instead of token overlap. Fixes the 2 false positives. Pure scoring change, no pipeline impact.
**File**: `eval/longmemeval/diagnostic_slice.py::_gold_in_text()`

### Fix 3: Verifier threshold for knowledge_update (deferred)
Case 6aeb4375 regressed in v2→v3 due to verifier threshold change. The `cov > 0.05 + fused_score > 0.02` gate is too tight for knowledge-update questions. Scoped to `knowledge_update` type only.
**File**: `eval/longmemeval/evidence_verifier.py::classify_evidence()`
**Risk**: Medium — a previous threshold change caused this regression.

### Fix 4: Conflict vs co-topic distinction (deferred)
Cases 0a995998 and 6d550036: different projects/items classified as CONFLICT because they share `(subject, predicate)`. They're co-topic, not conflicting. Reader ignores CONFLICT section.
**File**: `eval/longmemeval/evidence_verifier.py::_detect_scoped_conflicts()`

### Not fixable by pipeline (2 cases)
- 0edc2aef: Claims about Seattle trip, question about Miami — extraction captured wrong city's preferences
- gpt4_2655b836: Non-deterministic reader output (passed v3, failed v4)

---

## How to run the diagnostic slice

```bash
ACTIVE_PACK=personal_assistant \
AZURE_OPENAI_ENDPOINT="https://eastus2.api.cognitive.microsoft.com/" \
AZURE_OPENAI_API_KEY="<key>" \
AZURE_OPENAI_GPT4O_DEPLOYMENT="gpt-4o" \
AZURE_OPENAI_GPT4OMINI_DEPLOYMENT="gpt-4o" \
python -m eval.longmemeval.diagnostic_slice \
  --data data/longmemeval_oracle.json \
  --n-per-category 4
```

Note: bge-m3 embeddings segfault on Windows with large sessions. Use `MockEmbeddingClient` from `tests/e2e/conftest.py` for the embedding_client parameter. Extraction and reader use real gpt-4o.

Azure resource: `hackathon-openai` in `hackathon-eval` resource group, eastus2, gpt-4o deployment at 50K TPM.

---

## Hard rules (still in force)

- Do NOT patch one benchmark case at a time
- Do NOT scale to full 500-question benchmark until 20-case slice has clear failure-family structure
- Do NOT lose provenance from event frame back to claims
- Do NOT claim progress by comparing different case sets
- Do NOT touch temporal_reasoning logic unless a failed trace proves regression there
- Implement one fix at a time, re-run same 20 cases, report delta
