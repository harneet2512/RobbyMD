# GT v2 Study Notes — 2026-04-21

Distillation of the local-study pass over `D:\Groundtruth\src\groundtruth\memory\` and `D:\Groundtruth\docs\GT_V2_THEORY.md`. **No code is copied from GT v2.** These are ideas, patterns, and pitfalls to inform the fresh ~560 LOC implementation (revised down from 710 — see `docs/research_brief.md §4`).

Companion doc: `docs/research_brief.md` (frontier-lab + benchmark synthesis).

---

## 1. Concepts GT v2 has right (carry forward)

| Concept | Where in GT v2 | Why it matters |
|---|---|---|
| Claims as first-class objects with confidence + grounding_score + validity windows | `db/schema.py:60-87` | Enables verification gates, temporal slicing, projection queries |
| Supersession as explicit directed edges (not overwrites) | `db/schema.py:89-107`, `supersession/detect.py`, `confirm.py` | Preserves lineage; separates detection from confirmation; enables rollback; auditable |
| Active heads as a materialised projection | `projections/active_heads.py:40-51` | Fast state queries; idempotent rebuild after supersession |
| Admission filter as info-gain gate before expensive enrichment | `enrich/admission.py` | Prevents junk from entering the pipeline |
| Verification before supersession (NLI grounding) | `enrich/verify.py:71-125` | Stops hallucinated claims from corrupting state |
| Semantic identity via embeddings (subject+predicate+context, **not** object) | `supersession/identity.py` | Robust to surface form variance ("chest pain" vs "pain in the chest") |
| Slot-based canonical identity | `enrich/slots.py:99-159` | Prevents alias fragmentation across the same underlying claim |
| Async worker + queue-based enrichment | `enrich/worker.py`, `db/store.py:216-237` | Decouples fast ingest (<15ms) from expensive compute — we defer to synchronous but keep the mental pattern |
| Closed predicate families (no universal ontology) | `predicate_families.py` | Forces domain-specific vocabulary, which is right for clinical |
| Scope inference from root path | `integration/scope_inference.py` | Auto-manages isolation without config — for us, `session_id` does this job |

---

## 2. Per-module guidance for our fresh implementation

### 2.1 `schema.py` (~100 LOC)

**GT v2 reference:** `db/schema.py` (17 tables), `db/connection.py` (WAL mode, busy_timeout, read-pool).

**Keep:**
- WAL mode, `synchronous=NORMAL`, `busy_timeout=5000ms`, `cache_size=-20000` pragmas.
- vec0 virtual table pattern if we do semantic supersession (sqlite-vec extension, MIT).
- FTS5 with porter tokenizer for any full-text search (probably skip — we have closed predicate set).

**Drop for single-session chest-pain:**
- `entities` + `entity_aliases` (NER clustering overkill for closed predicates).
- `briefings`, `connections`, `scopes.parent_scope_id` (we have one scope = one session).
- `enrichment_jobs` queue (synchronous is fine).
- `event_embeddings` (retrieval is out of scope).

**Add (not in GT v2):**
- **`source_sentence_id` / per-character span columns** on `claims` — for the demo hero moment: click a note sentence → highlight the exact transcript span that sourced the claim that produced it.
- **`edge_type`** on `supersession_edges` with typed clinical vocabulary: `patient_correction | physician_confirm | semantic_replace | refines | contradicts | rules_out | dismissed_by_clinician`. See research_brief §2.1.

### 2.2 `claims.py` (~80 LOC, revised down from 120)

**GT v2 reference:** `enrich/classify.py` (LLM-based predicate classification, ~13k LOC), `enrich/materialize.py:28-70`.

**Simplify:** GT v2's classify.py is a general-purpose predicate-family classifier with 8 families and an LLM fallback. We have a **closed 14-predicate chest-pain vocabulary** (`onset, character, severity, location, radiation, aggravating_factor, alleviating_factor, associated_symptom, duration, medical_history, medication, family_history, social_history, risk_factor`). One structured Opus 4.7 prompt per turn, output JSON dict. ~80 LOC.

**Validation rules:** predicate ∈ closed family, `confidence ∈ [0,1]`, non-empty `source_turn_id` matching an existing turn. Invalid claims logged (not silently dropped — see pitfall §3.1).

### 2.3 `supersession.py` — Pass 1 deterministic (~70 LOC, down from 80)

**GT v2 reference:** `supersession/detect.py:32-130` (deterministic + semantic), `supersession/confirm.py`.

**Simplify:** single-session + closed predicates means structural matching is almost always sufficient. Pass 1 = same `(session_id, subject, predicate)`, different `value` → create edge.

**Type-discriminate** the edge: physician speaker + "yes that's right" → `physician_confirm`; patient corrects self → `patient_correction`; new claim is a narrower subset → `refines`.

**Guard:** never fires between claims in the same turn (additive, not replacing).

**Drop:** detect/confirm separation. Auto-confirm with threshold 0.0 (no human-review loop for hackathon).

### 2.4 `supersession_semantic.py` — Pass 2 (~70 LOC, down from 80; skippable)

**GT v2 reference:** `enrich/embed.py` (e5-small-v2 ONNX), `supersession/identity.py` (identity embedding = subject+predicate+context, **excludes value**), KNN search via vec0.

**Decision pending** (see research_brief §7-Q2): keep or drop.
- **If keep:** e5-small-v2 (MIT, 33M params, 384-dim). Threshold 0.88 (per Agent B's survey, lower than GT v2's 0.92 — better recall on clinical paraphrases). Identity embedding = `subject + predicate + surrounding context`, **never** includes `value`. Otherwise KNN matches on value strings and supersession breaks.
- **If drop:** save 70 LOC, lose the "pulmonary embolism onset" ≡ "PE onset" demo beat.

**Recommendation:** keep, but ship it last — if Day 4 squeezed, it's the cleanest cut.

### 2.5 `projections.py` (~100 LOC, down from 150)

**GT v2 reference:** `projections/active_heads.py`, `projections/briefing.py` (if exists).

**Keep:** `active_claims` materialised table, rebuilt after every supersession confirm. Query via `SELECT * FROM active_claims WHERE session_id=? AND status='active'`.

**Add per-branch projection** (not in GT v2): one materialised view per differential branch (Cardiac / Pulmonary / MSK / GI), filtered to claims whose predicate matches the branch's LR-table entries. Drives the `src/differential/` engine directly.

**Drop:** temporal routing, MMR diversity, knapsack budgeting. All retrieval.

**Target:** recompute ≤50 ms per projection on any claim-state change.

### 2.6 `admission.py` (~30 LOC, down from 60)

**GT v2 reference:** `enrich/admission.py` (noise regex + embedding novelty ≥0.15).

**Downgrade:** single-session + we control input. Drop embedding novelty. Keep ~30 LOC of noise-regex filtering only (reject: <3 content words, pure fillers "uh", "ok", "mhm", silence markers).

**Critical bug in GT v2 to avoid:** admission's novelty check (GT v2 lines 57-91) embeds the incoming event against existing events; on an empty index, novelty=1.0 (all admit), but after 100 events similar turns are rejected. If the demo resets between runs, threshold behaviour changes. Our mitigation: don't use embedding novelty at all.

### 2.7 `provenance.py` (~60 LOC)

**GT v2:** no direct analogue. Provenance is implicit via `source_turn_id` FK.

**Our addition:** forward/back-link utilities for the UI:
- `get_claims_for_turn(turn_id) -> list[claim_id]` (transcript panel click → highlight claims)
- `get_note_sentences_for_claim(claim_id) -> list[sentence_id]` (claim panel click → highlight note sentences)
- `get_turn_for_sentence(sentence_id) -> turn_id` (note panel click → highlight source turn)
- `span_for_claim(claim_id) -> (turn_id, char_start, char_end)` (for exact-text highlight in transcript)

This is the "provenance-as-hero" demo moment. Keep at full 60 LOC.

### 2.8 `event_bus.py` (~50 LOC, down from 60)

**GT v2:** no direct analogue. Communication is via DB polling + async queue.

**Our addition:** simple in-memory pub/sub for UI subscribers:
- `turn_added(turn_id)` → UI transcript refresh
- `claim_created(claim_id)` → UI claim-panel append
- `claim_superseded(old_id, new_id, edge_type)` → UI strike-through animation
- `differential_updated(ranking)` → UI tree re-render with ≤200 ms transition
- `verifier_updated(verifier_output)` → UI aux strip refresh

Python: simple dict of `{event_name: list[callback]}`. WebSocket broadcast at the UI boundary.

---

## 3. Pitfalls & anti-patterns (do not replicate)

### 3.1 Silent exception handlers

GT v2 has `except Exception: pass` in at least:
- `enrich/embed.py` (model loading)
- `supersession/identity.py:140` (find_similar_claims fallback)
- `enrich/worker.py:35, 44, 122` (verification, NER, slot computation)
- `enrich/slots.py:226, 270, 291, 304` (slot resolution tiers)

**Our rule** (already in CLAUDE.md §8 — confirm and enforce): no `except Exception: pass`. Log + propagate or log + return a typed `Error` result. Silent failures in supersession / verification corrupt state invisibly.

### 3.2 Over-abstraction: `PredicateFamilyMapper`

`predicate_families.py` is 119 LOC of deterministic + LLM-fallback predicate mapping for 8 general families. For our 14 closed clinical predicates, hard-code a flat dict. Saves 100 LOC.

### 3.3 Slot ordering fragmentation (unresolved in GT v2 theory doc)

If `subject` and `attribute` are extracted in inconsistent order, the slot string (`"subject/attribute"` vs `"attribute/subject"`) differs and claims about the same thing get different slots. `GT_V2_THEORY.md:91` flags this as open.

**Our fix:** subject always first, always lowercased + snake_case. E.g., `"patient_doe / onset"`. Normalize both components before joining.

### 3.4 Transitive supersession (undefined behaviour)

If A→B→C (A supersedes B, B supersedes C), is A the active head or C? GT v2 doesn't handle transitively. **Our rule:** supersession is 1:1. When A supersedes B, and B was already superseded by C, we short-circuit: A supersedes C directly. No chains.

### 3.5 Floating thresholds without validation

`config.py` has ~15 thresholds (0.92 identity, 0.15 novelty, 0.70/0.40 grounding, 0.90 auto-confirm, 0.85 entity-resolution, etc.). `GT_V2_THEORY.md:59-74` flags all as unvalidated.

**Our policy:** every threshold has (a) a default with a comment explaining why, (b) a log line capturing when it fires, (c) a revalidation plan for Day 3 against the DDXPlus-derived dialogue slice. See research_brief §5.

### 3.6 Confidence vs grounding_score confusion

`claim.confidence` = extraction confidence ("LLM said I'm 85% sure this is a claim").
`claim.grounding_score` = verification confidence ("NLI said I'm 70% sure the source entails this").

They're different. GT v2 sometimes conflates. **Keep them as distinct columns.**

### 3.7 Identity embedding includes object (wrong)

`claim_identity_vec` **must exclude `value`** — otherwise "onset: 3 days" and "onset: 4 days" embed differently and supersession can't match. GT v2 gets this right. Easy to break on re-implementation.

### 3.8 Alias fragmentation

"database" vs "db" vs "database engine" → three separate claim chains without entity resolution. GT v2 handles with `enrich/resolve.py` + `slots.py` normalization. For chest pain with closed predicates and a `session_id` fixed per encounter, we sidestep this: patient-id is `session_id`; subject is almost always `patient` or `chest_pain`; predicates are a closed set.

---

## 4. Threshold master table (for Day-3 clinical revalidation)

| Threshold | GT v2 value | GT v2 location | Our Day-1 value | Revalidation plan |
|---|---|---|---:|---|
| Semantic supersession cosine | 0.92 | `supersession/detect.py:37` | **0.88** | 20 chest-pain paraphrase pairs |
| Admission novelty | 0.15 | `config.py:73` | **N/A** (disabled) | — |
| NLI grounded | 0.70 | `config.py:84` | 0.70 | 20 note sentences, false-rejection rate |
| NLI ungrounded | 0.40 | `config.py:85` | 0.40 | same |
| Confidence high | 0.85 | `config.py:55` | 0.85 | extraction precision @ 0.85 |
| Confidence medium | 0.60 | `config.py:56` | 0.60 | — |
| Claim extraction min-confidence | 0.85 | `config.py:57` | **0.70** (permissive) | precision/recall trade-off |
| Supersession auto-confirm | 0.90 | `config.py:60` | **0.0** (always auto-confirm) | — |
| Slot embedding threshold | 0.90 | `config.py:76` | Skip Tier 2 | — |
| Slot ambiguous threshold | 0.80 | `config.py:77` | Skip Tier 3 LLM judge | — |
| Entity resolution similarity | 0.85 | `config.py:29` | Skip entity resolution | — |
| Embedding model | `intfloat/e5-small-v2` | `config.py:17` | **same** (MIT, 384-dim, fast) | — |
| NLI model | `cross-encoder/nli-MiniLM2-L6-H768` | `config.py:83` | **same** (measure clinical false-rejection) | 20 chest-pain sentences |
| NER model | `urchade/gliner_base` | `config.py:21` | **skip** | — |
| NER threshold | 0.4 | `config.py:22` | — | — |

---

## 5. Tests to re-create the spirit of

These are invariants our fresh unit tests should check. Not code — concepts.

### Schema & storage
- DB initialises without errors; all tables created; pragmas applied.
- Round-trip: insert claim → select → identical (no loss).
- Foreign keys enforced: deleting a turn cascades to its claims.

### Claim extraction
- "Patient has chest pain for 3 days" → claim(subject=patient, predicate=onset, value="3 days").
- Multi-claim utterance: "sharp chest pain, worse with breathing" → two claims.
- Negative finding: "no radiation to arms" → claim(predicate=radiation, value="none").
- Out-of-family predicate: logged + dropped.

### Verification (if NLI kept)
- Grounded: claim entailed by source turn → score ≥ 0.7.
- Ungrounded: claim contradicted by source → score < 0.4.
- Weak: neutral → 0.4 ≤ score < 0.7 (decide handling).

### Supersession Pass 1
- Same (session, subject, predicate), different value → edge created; old claim status='superseded'.
- Same predicate, same value → idempotent; no edge.
- Different predicate → no edge.
- Within-turn additive: two claims in same turn on same predicate → no supersession.
- Temporal order: newer claim must have later timestamp.
- Edge type assigned correctly based on speaker + context.

### Supersession Pass 2 (if kept)
- "chest pain onset" ≡ "pain in the chest onset" → same semantic identity cluster (cosine ≥ 0.88).
- "onset" ≡ "duration" → different clusters (no false supersession).

### Projections
- After supersession, `active_claims` contains only newer claim.
- Rebuild is idempotent (same input → same rows).
- Per-branch projection: only claims matching that branch's LR-table predicates appear.

### Provenance
- Every claim has a non-null `source_turn_id`.
- Every note sentence has non-empty `source_claim_ids`.
- `get_turn_for_sentence(s)` ∘ `get_sentences_for_claim(c)` ∘ `get_claims_for_turn(t)` round-trips.

### Event bus
- Subscribers called synchronously on publish.
- Publish with no subscribers is a no-op (no crash).

### End-to-end
- Scripted chest-pain case runs in <500 ms per turn from admission → projections.
- Determinism: same transcript → same active-claim state → same differential ranking, bit-identical (enforced by `tests/property/test_determinism.py`).

---

## 6. Gotchas surfaced during study (keep in mind)

1. **Identity embedding excludes value** — easy to mess up; break supersession invisibly.
2. **NLI model not fine-tuned for medical** — MiniLM on SNLI; may miss subtle clinical entailment like "chest pain at rest" vs "chest pain with exertion". Log scores; bimodal distribution suggests the model is confident; flat distribution suggests it's not.
3. **Verification threshold asymmetry** — `[0.4, 0.7)` range is "WEAK". GT v2 doesn't specify handling. Our decision: **store but exclude from active_claims projection** (claims shown in panel 2 but dimmed).
4. **Entity resolution silent-failure path** — GT v2's supersession matches on `subject_entity_id`; if resolution fails, id is NULL and supersession doesn't fire. Our mitigation: use raw normalized `subject_text` for matching, not an entity id.
5. **Briefing staleness** — GT v2 caches briefings with source_hash; stale until `rebuild_briefing()`. Our policy: rebuild projections synchronously after every turn; no caching optimisation.
6. **Admission cold-start bug** — 85% rejection rate on empty index vs eventual normal rate. Since we're disabling novelty-based admission, not an issue.
7. **Scope inference hashes root path** — clever but test-hostile. We use `session_id` directly; no hashing.
8. **Temporal windowing unused** — GT v2's `valid_from / valid_until` columns are defined but rarely used. We don't need them for single-session; drop.

---

## 7. Summary

GT v2 is the right conceptual foundation but **twice the scope we need**. The fresh rewrite should:

1. Adopt the claim / supersession-edge / projection data model verbatim (ideas, not code).
2. **Upgrade** to typed clinical supersession edges (white-space opportunity, research_brief §2.1).
3. **Skip** retrieval, NER entity resolution, admission novelty, async worker, scope inference — all README-vision material.
4. **Simplify** predicate mapping (closed 14-predicate dict, not `PredicateFamilyMapper`), supersession (structural-only + optional semantic), projections (no MMR/budget/temporal).
5. **Add** span-based provenance (not in GT v2) — the demo hero moment.
6. **Fix** every `except Exception: pass` site with logging + propagation.
7. **Validate** every threshold on 20 chest-pain synthetic dialogues in Day 3 (or earlier).

Total: ~560 LOC substrate + ~200 LOC verifier + ~70 LOC orchestrator + ~80 LOC tests = ~910 LOC for the core engine work, well under original 710-LOC budget for substrate alone and freeing headroom for the verifier and eval adapters.
