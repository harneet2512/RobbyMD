# Eng_doc.md — Engineering Specification

**Status**: Draft v2
**Related**: `context.md`, `PRD.md`, `CLAUDE.md`, `rules.md`

This document is the contract between planning and build. Every module, model, latency budget, and data structure declared here is binding. Deviations get a note in `docs/decisions/`.

`context.md` §7 (regulatory posture) and `rules.md` override engineering decisions in conflict.

---

## 1. System architecture

```
  mic (synthetic audio only)
        ↓
  [ASR + Diarisation]                Whisper large-v3 + WhisperX + custom vocab
        ↓   transcript turns (speaker, turn_id, text, ts)
  [Admission filter]                 drop clinically-irrelevant utterances
        ↓
  [Claim extractor]                  Opus 4.7, structured output, few-shot
        ↓   claim candidates
  [Supersession detector]            Pass 1: deterministic match
                                     Pass 2: e5-small semantic identity (0.92)
        ↓   claims with lifecycle edges
  [Claim store + projections]        SQLite + in-memory materialised views
        ↓   active claim state
  [Differential update engine]       deterministic LR-weighted log-likelihood
        ↓   ranked branch state
  [Counterfactual verifier]          support / refutation / next-best-question
        ↓
  [UI renderer]                      React + ReactFlow, 4 panels + aux strip
        ↓
  [Note generator]                   Opus 4.7, provenance-validated
```

Components communicate via the claim store event bus — not direct calls — so behaviour is replayable from a stored session.

---

## 2. Implementation posture

**All code in this repo is written fresh during the hackathon window.** No files are copied from any pre-existing codebase (see `context.md` §3 and §4). Architectural ideas and vocabulary come from prior research notes (`GT_V2_THEORY.md`) — those are thinking, not work.

Practically this means: scope the substrate so a fresh implementation is feasible. The minimum viable substrate for the demo thesis is ~500–1000 LOC in `src/substrate/`. That is the target. Anything beyond is README vision.

### 2.1 Minimum substrate (in demo)

| Module | Purpose | Approx LOC |
|---|---|---|
| `substrate/schema.py` + SQLite init | Core tables (see §4.1) | 100 |
| `substrate/claims.py` | Claim CRUD, active-state queries | 120 |
| `substrate/supersession.py` | Deterministic Pass 1 | 80 |
| `substrate/supersession_semantic.py` | e5-embed Pass 2 | 80 |
| `substrate/projections.py` | Per-branch materialised views over active claims | 150 |
| `substrate/admission.py` | Information-gain filter | 60 |
| `substrate/provenance.py` | Forward/back-link utilities for UI | 60 |
| `substrate/event_bus.py` | Pub/sub to UI | 60 |
| **Total** | | **~710** |

### 2.2 Deferred to README vision (not in demo)

7-signal retrieval fusion, ACT-R activation, MMR diversity, knapsack budget, NLI verification, async worker, scope inference, entity resolution with aliases, full slot canonicalisation, multi-session longitudinal memory. All real ideas; none on the demo critical path.

---

## 3. Model selection (open source only)

All model choices must satisfy the hackathon's open-source rule. OSI-approved licenses only, except for Opus 4.7 which is the hackathon's named sponsored API.

### 3.1 ASR stack

| Model | License | Role |
|---|---|---|
| **Whisper large-v3** | MIT | Primary transcription |
| **WhisperX** | BSD-4-Clause | Speaker diarisation on top of Whisper output |
| **Distil-Whisper large-v3** | MIT | Speed fallback (6× faster, ~1% WER loss on long-form) |
| Custom medical vocabulary | Apache 2.0 (ours) | Hotword bias + post-hoc correction for drug/anatomy terms |

**Rejected** (license not OSI-approved): Google MedASR, MedGemma (Gemma Terms); Deepgram Nova Medical, AssemblyAI (commercial APIs); any HAI-DEF-licensed model.

**Targets**: RTF ≤ 0.7 on 16–24 GB GPU; medical-term WER ≤ 12%; end-of-utterance → text ≤ 1.5 s.

**Demo moment**: side-by-side Whisper alone vs Whisper+custom-vocab on one utterance. Visceral, cheap, open-source defensible.

### 3.2 LLM

**Primary**: Claude Opus 4.7 via Anthropic API. Used for:
- Claim extraction (structured output, few-shot).
- Next-best-question natural-language rendering.
- SOAP note composition (structured output, post-hoc provenance-validated).

**Optional offline fallback**: **BioMistral-7B** (Apache 2.0, Mistral-based, pretrained on PubMed). Used only in the offline air-gapped profile for rehearsal if API access is disrupted. Not load-bearing.

### 3.3 Embeddings

`intfloat/e5-small-v2` (MIT, 33M params). Used for semantic supersession identity (Pass 2) and admission-filter novelty scoring.

### 3.4 Explicitly not using

- No ECG / imaging / IVD models (outside our criterion-1 envelope).
- No bio-NER (scispaCy, GLiNER-Bio) — over-scoped for chest-pain-only.
- No symptom-checker APIs (Infermedica, Isabel) — we are the reasoning substrate, not a wrapper.

---

## 4. Data model

### 4.1 Tables (SQLite WAL mode)

```sql
CREATE TABLE turns (
    turn_id         TEXT PRIMARY KEY,
    session_id      TEXT NOT NULL,
    speaker         TEXT NOT NULL CHECK (speaker IN ('patient','physician','system')),
    text            TEXT NOT NULL,
    ts              INTEGER NOT NULL,
    asr_confidence  REAL
);
CREATE INDEX idx_turns_session_ts ON turns(session_id, ts);

CREATE TABLE claims (
    claim_id         TEXT PRIMARY KEY,
    session_id       TEXT NOT NULL,
    subject          TEXT NOT NULL,
    predicate        TEXT NOT NULL,          -- from predicate_families
    value            TEXT NOT NULL,
    value_normalised TEXT,
    confidence       REAL NOT NULL,
    source_turn_id   TEXT NOT NULL REFERENCES turns(turn_id),
    status           TEXT NOT NULL CHECK (status IN ('active','superseded','confirmed','dismissed')),
    created_ts       INTEGER NOT NULL
);
CREATE INDEX idx_claims_active ON claims(session_id, subject, predicate, status);

CREATE TABLE supersession_edges (
    edge_id         TEXT PRIMARY KEY,
    old_claim_id    TEXT NOT NULL REFERENCES claims(claim_id),
    new_claim_id    TEXT NOT NULL REFERENCES claims(claim_id),
    reason          TEXT NOT NULL,            -- patient_correction | physician_confirm | semantic_replace
    identity_score  REAL,
    created_ts      INTEGER NOT NULL
);

CREATE TABLE decisions (
    decision_id           TEXT PRIMARY KEY,
    session_id            TEXT NOT NULL,
    kind                  TEXT NOT NULL,      -- confirm | dismiss | flag | tap_node | accept_question
    target_type           TEXT NOT NULL,
    target_id             TEXT NOT NULL,
    claim_state_snapshot  TEXT NOT NULL,      -- JSON
    ts                    INTEGER NOT NULL
);

CREATE TABLE note_sentences (
    sentence_id       TEXT PRIMARY KEY,
    session_id        TEXT NOT NULL,
    section           TEXT NOT NULL CHECK (section IN ('S','O','A','P')),
    ordinal           INTEGER NOT NULL,
    text              TEXT NOT NULL,
    source_claim_ids  TEXT NOT NULL            -- JSON array, must be non-empty
);
```

### 4.2 Predicate families (chest pain, closed set)

```
onset               character           severity            location
radiation           aggravating_factor  alleviating_factor  associated_symptom
duration            medical_history     medication          family_history
social_history      risk_factor
```

Any predicate outside this set is rejected by the extractor. Extending requires a PR with cited clinical rationale.

### 4.3 Projections

Four branch-projections (Cardiac / Pulmonary / MSK / GI). Each is a deterministic filter over active claims + the LR table, emitting:
- subset of active claims relevant to this branch
- per-node state ∈ {unasked, evidence-present, evidence-absent, contradicted}
- log-likelihood score

Recompute ≤50 ms per projection on any claim-state change.

### 4.4 LR table (`content/differentials/chest_pain/lr_table.json`)

```json
{
  "branch": "cardiac",
  "feature": "exertional_trigger",
  "predicate_path": "aggravating_factor=exertion",
  "lr_plus": 2.4,
  "lr_minus": 0.6,
  "source": "Gulati et al., 2021 AHA/ACC Chest Pain Guideline, §4.2",
  "source_url": "https://doi.org/10.1161/CIR.0000000000001029"
}
```

**Rule**: every LR has a peer-reviewed citation. No invented numbers. Approximations carry `"approximation": true` and a visual indicator in UI.

Sources: 2021 AHA/ACC Chest Pain Guideline (open); HEART score, TIMI, Wells, PERC primary papers. Do not paste proprietary content (UpToDate, AMBOSS).

---

## 5. Core components

### 5.1 Admission filter (`substrate/admission.py`)

Reject turn if: <3 content words; filler-only; embedding similarity to current active-claim set ≥0.95 (already known). Else admit.

Target false-rejection ≤5% on the DDXPlus-derived dialogue eval.

### 5.2 Claim extractor (`src/extraction/claim_extractor/`)

**Input**: current turn + 2 prior turns + current active claim set + predicate family.
**Output**: zero or more claim objects validated against schema.
**Prompt**: structured output, ≥5 few-shot examples covering multi-claim utterances, negative findings, patient self-correction (supersession), rare symptoms, ambiguous phrasing.
**Validation**: predicate ∈ predicate_families; confidence ∈ [0,1]; `source_turn_id` matches an existing turn. Invalid claims logged and dropped.
**Latency**: ≤700 ms per call.

### 5.3 Supersession detector (`substrate/supersession.py` + `supersession_semantic.py`)

**Pass 1 (deterministic)**: same `(subject, predicate)`, different `value` → edge with reason `patient_correction` or `physician_confirm`.

**Pass 2 (semantic)**: embed `(subject, predicate, value)` tuple, compare to active claims with cosine. Above 0.92 with same predicate family → `semantic_replace`.

**Guard**: never fires within the same turn.

### 5.4 Differential update engine (`src/differential/`)

```
for claim in active_claims:
  for row in lr_table where claim matches row.predicate_path:
    branch[row.branch].log_score += log(row.lr_plus) if present
                                  += log(row.lr_minus) if absent/negated
softmax across branches for displayed ranking.
```

Pure, sync, deterministic. Property test (§11.3) enforces.

Latency ≤50 ms full recompute.

### 5.5 Counterfactual verifier (`src/verifier/`)

See §6.

### 5.6 Note generator (`src/note/`)

Opus 4.7 emits `(section, text, [claim_id, ...])`. Post-hoc validator drops any sentence with empty or invalid `claim_ids`.

Latency ≤3 s for full-note generation (end of encounter, not live).

---

## 6. Counterfactual verifier (detail)

On every top-2 ranking update:

```python
@dataclass
class VerifierOutput:
    why_moved: list[str]                 # ≤2 bullets
    missing_or_contradicting: list[str]  # ≤2 bullets
    next_best_question: str              # ≤20 words
    next_question_rationale: str         # ≤1 line
    source_feature: str                  # which LR feature drove the question
```

Algorithm (deterministic selection, one LLM call at the end):

```
top2 = ranking[:2]
for branch in top2:
  support[branch]    = [f for f in lr_table[branch] if f.present_in(active) and f.lr_plus>1]
  refutation[branch] = [f for f in lr_table[branch] if f.not_present(active) and f.lr_plus>1.5]

discriminator = argmax_f in (refutation[top2[0]] | refutation[top2[1]])
                of |log LR_A(f) - log LR_B(f)| * current_uncertainty

question = opus_4_7(
  prompt=f"Write one clinical question ≤20 words about {discriminator.feature} "
         f"that distinguishes {top2[0].branch} from {top2[1].branch}."
)
```

Reuses the LR table. No new model stack. ~200 LOC. Latency ≤500 ms (dominated by the Opus call).

---

## 7. Frontend

**Stack**: React + TypeScript + Tailwind + shadcn/ui + ReactFlow + Zustand (state). All OSI-approved.

**Targets**: claim panel refresh ≤100 ms; tree update ≤200 ms; full-screen end-to-end ≤400 ms post-claim-ingest.

**Discipline**: read `/mnt/skills/public/frontend-design/SKILL.md` before touching UI. Trees must *breathe*, not snap — visible 200 ms transitions.

**Layout**: single screen at 1920×1080, reflow gracefully to 1440×900. No horizontal scroll. Persistent disclaimer header.

**Forbidden**: localStorage/sessionStorage (we're a live app, not a static artifact; use Zustand).

---

## 8. Orchestration

Five git worktrees, agents in parallel, human integrates at main.

```
main/             (human orchestrates)
wt-engine/        substrate/  (fresh implementation)
wt-extraction/    extraction/ (ASR + claim extractor + semantic supersession)
wt-trees/         content + differential/ + verifier/
wt-ui/            ui/
wt-eval/          eval/ (three benchmark harnesses)
```

Per-worktree scopes in `CLAUDE.md` §5.

---

## 9. Deployment

- **Hardware**: one workstation with 1× GPU (A100 40GB or 4090 24GB), 32 GB RAM.
- **Anthropic API**: Opus 4.7 access, expect ~30–60 calls per 5-min conversation.
- **Network**: online for API. Offline-rehearsal profile swaps Opus for local BioMistral-7B.
- **End-to-end latency**: ≤2.5 s utterance → all panels updated; tree update ≤200 ms is non-negotiable (the breathing moment).

---

## 10. Evaluation pipeline

Lives in `eval/`. Run with `make eval`. Outputs JSON + mini-charts for the video slide.

Every benchmark is published and peer-reviewed. No homemade metrics.

### 10.1 DDXPlus (`eval/ddxplus/`)

**Dataset**: public, https://github.com/mila-iqia/ddxplus. Subsample 500–730 cases via stratified sampling across the 49 pathologies (H-DDx 2025 methodology).

**Adapter**: converts DDXPlus's `{EVIDENCES, AGE, SEX, PATHOLOGY, DIFFERENTIAL_DIAGNOSIS}` record into a natural-dialogue turn stream our pipeline can ingest. The adapter is deterministic and documented.

**Variants**:
- `baseline.py` — Opus 4.7 prompted with the full case, asked for top-5 differential.
- `full.py` — turn stream → our substrate → differential engine → verifier → top-5 differential.

**Metrics**: Top-5 accuracy + HDF1 (ICD-10 hierarchical F1) per the H-DDx 2025 methodology. LLM judge pinned to `gpt-4o-2024-08-06` for Top-5 semantic-equivalence; HDF1 is computed deterministically via ICD-10 retrieval+rerank. Top-1 and Top-3 computed internally but not reported against H-DDx (which publishes Top-5 + HDF1 only in Table 2, covering 22 LLMs).

**Comparison**: report alongside H-DDx 2025 published numbers for selected comparator models.

### 10.2 LongMemEval-S (`eval/longmemeval/`)

**Dataset**: public, https://github.com/xiaowu0162/LongMemEval, `longmemeval_s` split.

**Adapter**: feeds session history into the substrate via our write API, then answers the benchmark's 500 questions using substrate retrieval.

**Variants**:
- `baseline.py` — Opus 4.7 full-context (entire session history in the prompt).
- `full.py` — Opus 4.7 + substrate as memory layer.

**Metric**: per-category accuracy — information extraction, multi-session reasoning, temporal reasoning, knowledge update, abstention. Use the official LongMemEval evaluator.

**Comparison**: TiMem 76.88%, EverMemOS 83.0%, Zep/Graphiti 71.2%, MemOS.

### 10.3 ACI-Bench (`eval/aci_bench/`)

**Dataset**: public, https://github.com/wyim/aci-bench. Run both `aci` (66 test encounters across test1/test2/test3) and `virtscribe` (24 test encounters). No slicing — full authors' test set.

**Adapter**: reads dialogue (gold or ASR), runs our extraction → substrate → note generator, compares to gold note.

**Variants**:
- `baseline.py` — Opus 4.7 dialogue → note directly.
- `full.py` — dialogue → substrate → provenance-validated note.

**Metrics**: ROUGE-1/2/L, BERTScore, MEDCON (clinical-concept F1). Use the MEDIQA-CHAT 2023 evaluation scripts.

**MEDCON dependency (hard blocker)**: requires a UMLS Terminology Services account (https://uts.nlm.nih.gov/uts/signup-login, 1–3 business-day NIH approval) plus local QuickUMLS install. Account must be requested before any other ACI-Bench work starts or this eval is blocked. BERTScore is computed division-based (full notes exceed embedding length) per the official ACI-Bench eval script — do not compute whole-note BERTScore.

**Comparison**: MEDIQA-CHAT 2023 leaderboard.

### 10.4 Outputs

`eval/reports/<timestamp>/`:
- `results.json` — raw numbers
- `ddxplus_chart.png`, `longmemeval_chart.png`, `aci_bench_chart.png`
- `summary.md` — one-page summary for README

---

## 11. Quality gates

### 11.1 Unit tests
Every public function in `substrate/` has at least one test. Verifier and differential engine ship with tests.

### 11.2 E2E test (`tests/e2e/test_demo_case.py`)
Plays the scripted chest-pain audio through the full pipeline, asserts expected claims, expected supersession at the expected turn, Cardiac in top-2 at end, verifier surfaces a discriminator from the expected feature family, SOAP note has full provenance.

### 11.3 Determinism property test (`tests/property/test_determinism.py`)
Differential update engine runs 100× on same active claims, asserts identical output every time. Blocks merge if it fails.

### 11.4 Privacy gate (`tests/privacy/test_no_phi.py`)
Static scan: every `eval/` file declared in `SYNTHETIC_DATA.md`; no real-looking SSNs/MRNs/DOBs/named-patient patterns anywhere in the repo.

### 11.5 License gate (`tests/licensing/test_open_source.py`)
Static scan of `pyproject.toml` / `package.json` / model download list. Every dependency has an OSI-approved license (allowlist: MIT, Apache-2.0, BSD, MPL, ISC, LGPL). Flags anything else. Opus 4.7 / Anthropic API is whitelisted as the hackathon's named sponsored tool.

---

## 12. Risks and mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| ASR latency too high for live feel | High | Whisper large-v3 primary, Distil-Whisper fallback, scripted audio playback is fine for demo video |
| Supersession doesn't fire cleanly on demo case | Demo-blocking | Scripted case, deterministic rule catches it by construction; property test protects |
| Trees don't "breathe" | Demo-blocking | Explicit 200 ms transitions; integration polish is the last workstream |
| Opus 4.7 rate limits during eval | Medium | Stagger runs; keep BioMistral warm as offline fallback |
| DDXPlus adapter produces unrealistic dialogue | Medium | Use the dialogue form already suggested in DDXPlus paper (structured turns); validate on 5 cases by eye before running at scale |
| LongMemEval-S numbers underwhelm | Medium | Report honestly; even partial wins on knowledge-update and temporal categories defend the architecture |
| ACI-Bench `aci` subset is small (40 test cases) | Low | Good — faster iteration; if we want more, add `virtscribe` |
| Closed model dependency (Gemma-licensed) sneaks in | Disqualifying | `test_open_source.py` enforces; review all new deps in CI |
| Reviewer asks about open-source rule wrt Opus 4.7 | Low | Event is *Built with Opus 4.7* — it's the sponsored tool, same as AWS/Vercel/GitHub |
| Confidential-fabrication risk (invented LR values) | High | Every LR has a source; `content/differentials/chest_pain/sources.md` is reviewed before merge |
| MEDCON / UMLS licence delayed or denied | Demo-blocking for ACI-Bench eval | Apply on Day 1; if delayed past Day 4, fall back to ROUGE + BERTScore only and document the gap honestly on the slide |

---

## 13. Open questions

1. Exact DDXPlus subset size (500 vs 730 vs full 1k+) — balance eval runtime vs published methodology.
2. ACI-Bench MEDCON implementation — use published eval scripts as-is.
3. Custom ASR vocabulary — start with a public drug/anatomy list, iterate on demo-case WER.
4. Semantic supersession threshold — 0.92 default, empirically validate on a 50-case subset of the DDXPlus-derived dialogue.
5. ReactFlow performance with 4 live trees at ~10 Hz during extraction bursts — test early in UI build.
