# Bundle 2 — Phase 0 Orientation (2026-04-23)

Scratchpad capturing the current state of code Bundle 2 depends on. Two parallel Explore agents + direct reads this session. All line numbers frozen against the commit at session start (main @ `6a66e92`).

## Track A — LongMemEval

### Embedding call sites in `src/substrate/retrieval.py` — SYMMETRIC today

Queries and documents both go through `EmbeddingClient.embed([text])[0]` with no asymmetry prefix. bge-m3 was trained with an asymmetric `"Represent this sentence for searching relevant passages: "` query prefix; omitting it on the query side depresses cosine similarity.

- **Query-side (three sites, no prefix today):**
  - `retrieval.py:473` — `q_vec = effective.embed([question])[0]` inside `retrieve_relevant_claims()`
  - `retrieval.py:647` — `q_vec = effective.embed([query])[0]` inside `retrieve_hybrid()`
  - `retrieval.py:793` — `q_vec = effective.embed([query])[0]` inside `retrieve_event_tuples()`
- **Document-side (unchanged; no prefix, correct as-is):**
  - `retrieval.py:294` — `vec = effective.embed([text])[0]` inside `embed_and_store()`
  - `retrieval.py:186` — `effective.embed([text])[0]` inside `backfill_embeddings()` helper

Public retrieval entry points:
- `retrieve_relevant_claims(conn, *, session_id, question, branch=None, k=20, client=None) -> list[RankedClaim]` (line 410)
- `retrieve_hybrid(conn, *, session_id, query, entity_hint=None, top_k=20, valid_at_ts=None, weights=(1.0, 1.0, 1.0), embedding_client=None) -> list[tuple[Claim, float]]` (line 581)
- `retrieve_event_tuples(conn, *, session_id, query, top_k=16, valid_at_ts=None, embedding_client=None) -> list[tuple[EventTuple, float]]` (line 732)

### Live LME substrate variant — `eval/smoke/run_smoke.py:763` `_call_longmemeval_substrate_retrieval_con`

The active path post-FIX-2. Signature (lines 763-769):

```python
def _call_longmemeval_substrate_retrieval_con(
    case_payload: object,
    reader_env: dict[str, str],
    top_k: int = 20,
    *,
    extractions_path: "Path | None" = None,
) -> tuple[str, float, int, SubstrateStats, dict[str, Any]]:
```

Flow (lines 802-836):
1. `make_llm_extractor()` + streaming wrapper
2. `_ingest_with_real_extractor("longmemeval", q, extractor)` — writes haystack turns + claims
3. `EmbeddingClient()` + `backfill_embeddings(conn, q.question_id, client=embed_client)`
4. `retrieve_relevant_claims(conn, session_id=q.question_id, question=q.question, k=top_k, client=embed_client)` — 20 claims
5. `answer_with_con(q.question, ranked)` — two-call CoN → final answer

A.4 replaces step 4's k=20 + step 5's CoN with the adapter routing + single-call reader.

### CoN reader — `eval/longmemeval/reader_con.py`

Paper-faithful Chain-of-Note (Yu et al. 2023). Two LLM calls per question:

- Call 1 (NOTE_EXTRACTION_SYSTEM, lines 48-57): extract verbatim relevant facts into JSON `{"notes": [...]}`.
- Call 2 (ANSWER_SYSTEM, lines 59-64): answer using ONLY the extracted notes, or `"I don't know"`.
- **Hard IDK coercion (lines 251-252):**
  ```python
  if not notes:
      answer = "I don't know"
  ```

This is a double filter: `retrieve_relevant_claims` already filters candidates, then Call 1 re-filters, and any empty Call-1 output forces IDK regardless of retrieval quality. Single-call adapter replaces the whole pattern.

### Smoke harness — `eval/smoke/run_smoke.py`

- `BENCHMARKS = ("longmemeval", "acibench")` (line 56)
- CLI args (lines 262-314):
  - `--benchmark {longmemeval|acibench|both}` (line 264)
  - `--reader` (line 265)
  - `--variant {baseline|substrate|both}` (line 266)
  - `--n`, `--budget-usd`, `--dry-run`, `--legacy-lme-substrate`, `--hybrid / --no-hybrid`, `--seed`, `--stratified`, `--output-dir`
- Results dir (line 115-126): `eval/smoke/{UTC_TIMESTAMP}/{benchmark}/{variant}/`, files: `predictions.jsonl`, `metrics.json`, `LIMITATIONS.md`

### Extractor — `src/extraction/claim_extractor/extractor.py`

Factory at lines 48-54:

```python
def make_llm_extractor(
    *,
    purpose: str = "claim_extractor_gpt4omini",
    active_pack_id: str | None = None,
    env: Mapping[str, str] | None = None,
    temperature: float = 0.0,
) -> ExtractorFn:  # Callable[[Turn], list[ExtractedClaim]]
```

- Default model: gpt-4o-mini via `eval/_openai_client.make_openai_client(purpose)`.
- Routes to Azure (`AZURE_OPENAI_GPT4OMINI_DEPLOYMENT`) or direct OpenAI (`OPENAI_API_KEY`).
- Synchronous. 30s timeout. No retry; exceptions propagate.
- Ingests a single `Turn` — no conversational context.
- `ExtractedClaim` at `src/substrate/on_new_turn.py:45-60`: `subject, predicate, value, confidence, value_normalised?, char_start?, char_end?`.

### Vestigial / do-not-touch

- `eval/longmemeval/full.py` — STUB with assertion at line 56 (`"[SUBSTRATE STUB]" not in wrapped.raw_response`). Falls through to baseline then asserts. Never reached live because `run_smoke.py` doesn't import it.

## Track B — MedQA context layer

### Substrate write API

- `src/substrate/claims.insert_turn(conn, turn)` at lines 132-156
- `src/substrate/claims.insert_claim(conn, *, session_id, subject, predicate, value, confidence, source_turn_id, ...) -> Claim` at lines 236-344
- Validation: `validate_claim()` at lines 205-213 checks predicate against `active_pack().predicate_families`.
- Claim schema at `src/substrate/schema.py:127-160`:
  ```python
  Claim(claim_id, session_id, subject, predicate, value, value_normalised?, confidence,
        source_turn_id, status, created_ts, char_start?, char_end?, valid_from_ts?, valid_until_ts?)
  ```
- Turn schema at `src/substrate/schema.py:116-125`: `Turn(turn_id, session_id, speaker, text, ts, asr_confidence?)`.

### Substrate read API

- `src/substrate/claims.list_active_claims(conn, session_id) -> list[Claim]` at lines 369-387 — flat dump, grouping is adapter responsibility.
- `src/substrate/retrieval.retrieve_relevant_claims` — similarity-ranked top-k (for LME only; MedQA doesn't need retrieval since single-vignette session).

### Clinical predicate pack — `predicate_packs/clinical_general/predicates.json`

20 families: `onset, character, severity, location, radiation, aggravating_factor, alleviating_factor, associated_symptom, duration, medical_history, medication, allergy, family_history, social_history, risk_factor, vital_sign, lab_value, imaging_finding, physical_exam_finding, review_of_systems`.

Adapter's `format_structured_findings` groups by these families, renders human-readable section headers (SYMPTOMS, VITALS, LABS, HISTORY, etc.).

### OpenAI client — `eval/_openai_client.make_openai_client(purpose, env)`

Single routing choke point (lines 148-225). All readers/extractors go through it. Env vars:

- Azure: `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_GPT4OMINI_DEPLOYMENT`
- OpenAI: `OPENAI_API_KEY`

Existing purposes: `"claim_extractor_gpt4omini"`, `"llm_medcon_gpt4omini"`, `"longmemeval_reader"`. MedQA reuses `"claim_extractor_gpt4omini"` for extraction and adds `"medqa_reader_gpt4omini"` for reader.

### Existing adapter patterns (models to follow)

- `eval/longmemeval/adapter.py:52-82` — `session_to_turns()` maps haystack session → Turn list.
- `eval/aci_bench/adapter.py:52-74` — `encounter_to_turns()` — dialogue → substrate channels.

### MedQA presence check

- `**/*medqa*` — 0 files
- `**/*MedQA*` — 0 files
- `eval/data/` — contains `longmemeval/`, `acibench/`; **no `medqa/`**
- Clean slate.

### Test infrastructure

- `tests/conftest.py` — sets `sys.path`, mutes structlog.
- `tests/unit/substrate/conftest.py` — fixtures `conn` (in-memory SQLite + schema), `session_id`, `add_turn`.
- Patterns already used: `test_adapters.py`, `test_aci_bench_extractors.py`.

## Hardblock decisions (2026-04-23 session)

- **MedQA conflict with rules.md §10 / reasons.md:119-125 resolved in favor of proceeding** — user decision this session. Prerequisite: ADR at `docs/decisions/2026-04-23_medqa-reinstated-as-context-layer-ablation.md` supersedes the 2026-04-21 drop entry. Reframes MedQA as a context-layer ablation (not a substrate benchmark). Scope-limited: reported separately from the substrate-evaluation slide.
- **Track A.4 wires into `eval/smoke/run_smoke.py` directly** — not `eval/longmemeval/full.py` (vestigial) and not a new module. Live path, fewest moving parts.

## Hardblocks still in force

- Do NOT modify: `src/substrate/supersession.py`, `supersession_semantic.py`, `schema.py`; `src/extraction/claim_extractor/*`; `predicate_packs/*`; `src/differential/*`; `src/verifier/*`; `CLAUDE.md`, `PRD.md`, `Eng_doc.md`, `context.md`, `rules.md`.
- Do NOT run: full LME n=500, full MedQA n=1273. Smoke only this session; escalation requires explicit go-ahead.
- Do NOT use: Opus 4.7 for eval loops (extractor = gpt-4o-mini, LME reader = gpt-4o-2024-11-20, MedQA reader = gpt-4o-mini-2024-07-18).
