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

### 3.5 Model-usage policy

Opus 4.7 is a capable sponsored tool. It is not a scarce resource to ration, but it IS costly and it **muddies benchmark comparisons** when used everywhere. Reserve Opus 4.7 for calls whose output a judge sees or that visibly shape the demo. For eval loops and bulk extraction, match the published SOTA reader for apples-to-apples comparability.

**Demo-path (Opus 4.7 required)** — every call below produces output a judge sees in the demo video:

| Call site | Frequency | Rationale |
|---|---|---|
| Live claim extraction during the scripted visit | ~30–60 calls / 5-min conversation | Drives the live transcript → claim-panel fill |
| Counterfactual verifier's next-best-question phrasing | ≤5 calls / visit (top-2 ranking changes) | The aux-strip question viewers see on screen |
| SOAP note composition (demo run only) | 1 call / end-of-encounter | The provenance-validated note in the video |

Supersession Pass 2 semantic identity uses `e5-small-v2` only (no LLM — pure cosine over embeddings).

**Eval loops and infrastructure (Opus 4.7 NOT USED; primary reader Qwen2.5-14B-Instruct + secondary comparability readers match published SOTA)**:

| Call site | Model | Rationale |
|---|---|---|
| LongMemEval-S reader, primary (baseline + substrate variant) | `Qwen2.5-14B-Instruct` (Apache-2.0, self-hosted via vLLM on GCP L4 spot; fallback Together / Fireworks / DeepInfra API) | Reader-agnostic baseline not tied to any closed-model pricing; Apache-2.0; empirically establishes substrate delta at a fixed open-weight reader |
| LongMemEval-S reader, secondary (comparability) | `gpt-4o-mini` | Published Mem0 / Mastra OM / EverMemOS comparator axis — apples-to-apples with leaderboard |
| ACI-Bench reader, primary (baseline + substrate variant) | `Qwen2.5-14B-Instruct` | Same self-hosted stack; same reader-agnostic claim |
| ACI-Bench reader, secondary (comparability) | `gpt-4.1-mini` | 2026 cost-equivalent of [WangLab 2023 GPT-4 ICL](https://arxiv.org/abs/2305.02220) (MEDIQA-CHAT 2023 1st place) |
| SOAP note generation on the ACI-Bench eval path | **match reader model (NOT Opus 4.7)** | Apples-to-apples. Demo-path SOAP note uses Opus 4.7; eval-path SOAP note uses the active benchmark reader |
| LLM judge — LongMemEval-S accuracy | `gpt-4o-2024-08-06` (pinned) | Per LongMemEval paper mandate |
| Offline bulk reprocessing | `gpt-4.1-mini` | Non-demo path, cost-sensitive |
| Infrastructure loops (retry wrappers, validators) | N/A (deterministic code) | No LLM |

**Hosting the primary reader**: `eval/infra/deploy_qwen_gcp.sh` (primary path — GCP L4 spot + vLLM INT8 quantization) and `eval/infra/deploy_qwen_azure.sh` (documented fallback — Azure NVadsA10_v5 spot). Host machine already authenticated for `gcloud` + `az`. Rationale at `eval/infra/README.md`.

**The rule**: if a call is on the demo-video playback path and shapes what a judge sees → Opus 4.7. Otherwise → the cheapest model that preserves apples-to-apples comparison with the benchmark's published SOTA reader.

**Why the rule exists**: using Opus 4.7 as the eval reader when the published SOTA used `gpt-4o-mini` or `Qwen2.5-14B`-class readers conflates "our substrate vs their substrate" with "Opus 4.7 vs their reader" and kills the comparison. Benchmark comparability is the eval slide's entire value. Running the same substrate against two readers (primary Qwen2.5-14B for reader-agnostic claim; secondary closed-model for published-comparator alignment) makes the substrate delta attributable to the substrate, not the reader. See `reasons.md` → "Tank for the war, not the gun fight" for the full rationale.

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

### 4.2 Predicate families (pluggable domain packs)

Predicate families are declared in **domain packs** that register with the extractor at startup. Each pack defines:

- A unique `pack_id` (e.g., `clinical_general`, `personal_assistant`, `coding_agent`).
- A closed vocabulary of predicate families for that domain.
- Optional sub-slot schemas for predicates with structured values.
- Optional LR tables keyed to differential branches (clinical packs only).

The engine does not know about clinical medicine. Predicate families are data, not code. Packs register via `src/substrate/predicate_packs.py::register_pack(pack)` (module lands with the second pack); the extractor loads the active pack's families at admission time. Claims whose predicate is outside the active pack's closed vocabulary are rejected.

**Seeded packs this build**: exactly one — `clinical_general` (covers any chief complaint). Additional packs (personal_assistant for LongMemEval-S, coding_agent, legal, etc.) register via the same API with zero engine changes.

**`clinical_general` predicate set** — closed vocabulary covering chest pain, abdominal pain, dyspnoea, headache, fatigue, and any other chief complaint:

```
onset                  character                severity
location               radiation                aggravating_factor
alleviating_factor     associated_symptom       duration
medical_history        medication               allergy
family_history         social_history           risk_factor
vital_sign             lab_value                imaging_finding
physical_exam_finding  review_of_systems
```

**Structured-value sub-slots** (applicable predicates carry these; the extractor emits partial schemas when information is incomplete):

| Predicate | Sub-slots |
|---|---|
| `medication` | `name, dose, route, frequency, indication, start_date` |
| `vital_sign` | `kind ∈ {BP, HR, RR, SpO2, Temp}, value, unit, measured_at` |
| `lab_value` | `name, value, unit, reference_range, specimen_type, collected_at` |
| `imaging_finding` | `modality ∈ {X-ray, CT, MRI, US}, body_part, finding_text, reported_at` |
| `physical_exam_finding` | `body_part, finding, elicitation_method` |
| `allergy` | `agent, reaction, severity, verified_by` |

**Pack-registration schema** (sketch; full module lands with the second pack):

```python
@dataclass
class PredicatePack:
    pack_id: str                                   # e.g. "clinical_general"
    predicate_families: frozenset[str]             # closed vocabulary
    sub_slots: dict[str, frozenset[str]]           # predicate → sub-slot names
    lr_table_path: Path | None = None              # clinical packs only
    description: str = ""

def register_pack(pack: PredicatePack) -> None: ...
def active_pack() -> PredicatePack: ...
```

**Future packs** (schema-ready, not seeded this build): `personal_assistant` (predicate families like `fact_about_user, preference, scheduled_event, knowledge_update` for LongMemEval-S substrate variant), `coding_agent`, `legal`. Adding a pack requires no engine changes — just the pack declaration + its LR table (if clinical).

Any predicate outside the active pack's closed set is rejected by the extractor.

### 4.3 Projections

Four branch-projections (Cardiac / Pulmonary / MSK / GI). Each is a deterministic filter over active claims + the LR table, emitting:
- subset of active claims relevant to this branch
- per-node state ∈ {unasked, evidence-present, evidence-absent, contradicted}
- log-likelihood score

Recompute ≤50 ms per projection on any claim-state change.

### 4.4 LR table (`predicate_packs/clinical_general/differentials/chest_pain/lr_table.json`)

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

Target false-rejection ≤5% on the LongMemEval-S / ACI-Bench dialogue evals.

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

### 10.1 LongMemEval-S (`eval/longmemeval/`)

**Dataset**: public, https://github.com/xiaowu0162/LongMemEval, `longmemeval_s` split.

**Adapter**: feeds session history into the substrate via our write API, then answers the benchmark's 500 questions using substrate retrieval.

**Variants**:
- `baseline.py` — Opus 4.7 full-context (entire session history in the prompt).
- `full.py` — Opus 4.7 + substrate as memory layer.

**Metric**: per-category accuracy — information extraction, multi-session reasoning, temporal reasoning, knowledge update, abstention. Use the official LongMemEval evaluator.

**Comparison**: TiMem 76.88%, EverMemOS 83.0%, Zep/Graphiti 71.2%, MemOS.

### 10.2 ACI-Bench (`eval/aci_bench/`)

**Dataset**: public, https://github.com/wyim/aci-bench. Run both `aci` (66 test encounters across test1/test2/test3) and `virtscribe` (24 test encounters). No slicing — full authors' test set.

**Adapter**: reads dialogue (gold or ASR), runs our extraction → substrate → note generator, compares to gold note.

**Variants**:
- `baseline.py` — Opus 4.7 dialogue → note directly.
- `full.py` — dialogue → substrate → provenance-validated note.

**Metrics**: ROUGE-1/2/L, BERTScore, MEDCON (clinical-concept F1). Use the MEDIQA-CHAT 2023 evaluation scripts.

**MEDCON dependency (hard blocker)**: requires a UMLS Terminology Services account (https://uts.nlm.nih.gov/uts/signup-login, 1–3 business-day NIH approval) plus local QuickUMLS install. Account must be requested before any other ACI-Bench work starts or this eval is blocked. BERTScore is computed division-based (full notes exceed embedding length) per the official ACI-Bench eval script — do not compute whole-note BERTScore.

**Comparison**: MEDIQA-CHAT 2023 leaderboard.

### 10.3 Outputs

`eval/reports/<timestamp>/`:
- `results.json` — raw numbers
- `longmemeval_chart.png`, `aci_bench_chart.png`
- `summary.md` — one-page summary for README

(DDXPlus and MedQA were dropped 2026-04-21 — see `reasons.md` → DDXPlus and MedQA entries. Only two benchmarks remain, both scoped to the substrate's load-bearing contributions.)

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
| LongMemEval-S baseline-vs-substrate delta is noisy on small N | Medium | Smoke-first discipline — eval/smoke/run_smoke.py runs a deterministic first-10-case sanity pass before committing to full 500; published baselines pinned in eval/smoke/reference_baselines.json for ±20pp sanity check |
| LongMemEval-S numbers underwhelm | Medium | Report honestly; even partial wins on knowledge-update and temporal categories defend the architecture |
| ACI-Bench `aci` subset is small (40 test cases) | Low | Good — faster iteration; if we want more, add `virtscribe` |
| Closed model dependency (Gemma-licensed) sneaks in | Disqualifying | `test_open_source.py` enforces; review all new deps in CI |
| Reviewer asks about open-source rule wrt Opus 4.7 | Low | Event is *Built with Opus 4.7* — it's the sponsored tool, same as AWS/Vercel/GitHub |
| Confidential-fabrication risk (invented LR values) | High | Every LR has a source; `predicate_packs/clinical_general/differentials/chest_pain/sources.md` is reviewed before merge |
| MEDCON / UMLS licence delayed or denied | Demo-blocking for ACI-Bench eval | Apply on Day 1; if delayed past Day 4, fall back to ROUGE + BERTScore only and document the gap honestly on the slide |

---

## 13. Open questions

1. First smoke-run verdict on the harness once user signs off on real execution (LongMemEval-S + ACI-Bench × baseline + substrate × 10 cases each, Qwen2.5-14B reader, $50 budget cap per `eval/smoke/`).
2. ACI-Bench MEDCON implementation — use published eval scripts as-is.
3. Custom ASR vocabulary — start with a public drug/anatomy list, iterate on demo-case WER.
4. Semantic supersession threshold — 0.92 default, empirically validate on a 50-case subset of the LongMemEval-S / ACI-Bench dialogue.
5. ReactFlow performance with 4 live trees at ~10 Hz during extraction bursts — test early in UI build.
