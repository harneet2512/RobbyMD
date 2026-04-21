# CLAUDE.md — Agent Operating Instructions

Read by every Claude Code agent in this repo. Operating contract for automated work.

If anything here conflicts with `rules.md`, `context.md` §7, or `PRD.md` §10, those documents win. This is workflow; those are invariants.

---

## 1. What this repo is

A hackathon build for **Built with Opus 4.7: a Claude Code Hackathon** (Cerebral Valley × Anthropic). Live clinical reasoning substrate: conversation → claims → differentials → note, all with provenance.

Read before any non-trivial work:
1. `context.md` — hackathon framing, rules compliance, regulatory posture.
2. `PRD.md` — product requirements, UI, evals, DoD.
3. `Eng_doc.md` — engineering spec, models, data model, latency budgets, eval harness.
4. `rules.md` — non-negotiables.

---

## 2. Hard constraints (always)

Copies of the load-bearing invariants from `rules.md`:

1. **Every line of code in this repo is written during the hackathon window.** No copying from any pre-existing codebase. Prior thinking (ideas, vocabulary, `GT_V2_THEORY.md` notes) is allowed; code is not.
2. **All models and dependencies must be OSI-approved open source.** Allowlist: MIT, Apache-2.0, BSD, MPL, ISC, LGPL. Not allowed: Gemma Terms, HAI-DEF, any commercial API except Opus 4.7 (the hackathon's named sponsored tool). Enforced by `tests/licensing/test_open_source.py`.
3. **No real patient data.** All clinical data is synthetic or from public benchmarks (DDXPlus / LongMemEval / ACI-Bench). Every dataset declared in `SYNTHETIC_DATA.md`.
4. **Every claim and every note sentence has provenance** back to a conversation turn. Emitting a sentence without a provenance chain is a hard failure.
5. **The physician makes every clinical decision.** No auto-diagnose, auto-treat, auto-action. "Next best question" is a suggestion with rationale, never a directive.
6. **Deterministic differential update engine.** Same inputs → identical outputs, always. Property test enforces.
7. **Disclaimer everywhere**: app header, README, demo video bumper + outro, written summary first three sentences.
8. **No ECG, no imaging, no IVD signals, no continuous-physiologic-stream processing.** Keeps us classified as Non-Device CDS under FDA 2026 guidance.
9. **No patient-facing UI.** HCP-facing only.
10. **Only published benchmarks for evals.** DDXPlus, LongMemEval-S, ACI-Bench. No homemade "substrate ablation" charts.

Any task appearing to conflict with the above → stop and surface the conflict.

---

## 3. Project layout

```
/
├── context.md                  hackathon framing + compliance
├── PRD.md                      product requirements
├── Eng_doc.md                  engineering spec
├── CLAUDE.md                   this file
├── rules.md                    non-negotiables
├── SYNTHETIC_DATA.md           manifest — every dataset declared
├── README.md                   public-facing (vision beyond demo)
│
├── src/
│   ├── substrate/              core: claims, supersession, projections (fresh)
│   ├── extraction/
│   │   ├── asr/                Whisper large-v3 + WhisperX + custom vocab
│   │   └── claim_extractor/    Opus 4.7 prompts + validation
│   ├── differential/           trees + LR-weighted update engine
│   ├── verifier/               counterfactual + next-best-question
│   ├── note/                   SOAP gen, provenance-validated
│   └── api/                    local server for UI
│
├── ui/                         React + TS + Tailwind + shadcn + ReactFlow + Zustand
│
├── content/
│   └── differentials/chest_pain/
│       ├── branches.json
│       ├── lr_table.json
│       └── sources.md
│
├── eval/
│   ├── ddxplus/                DDXPlus adapter + runner
│   ├── longmemeval/            LongMemEval-S adapter + runner
│   ├── aci_bench/              ACI-Bench adapter + runner (MEDIQA-CHAT metrics)
│   └── reports/
│
├── tests/
│   ├── unit/
│   ├── e2e/test_demo_case.py
│   ├── property/test_determinism.py
│   ├── privacy/test_no_phi.py
│   └── licensing/test_open_source.py
│
├── scripts/
│   ├── setup.sh
│   ├── run_demo.sh
│   └── record_eval.sh
│
└── docs/
    ├── decisions/              ADRs for any deviation
    └── GT_V2_THEORY.md         prior research note (ideas only; no code reused)
```

---

## 4. Orchestration — git worktrees

Five worktrees, one agent each, human integrates at main.

```
main/            human orchestrates, does integrations
wt-engine/       feature/substrate        src/substrate/
wt-extraction/   feature/extraction       src/extraction/
wt-trees/        feature/differential     content/ + src/differential/ + src/verifier/
wt-ui/           feature/ui               ui/
wt-eval/         feature/eval             eval/
```

Setup (from repo root):

```bash
git worktree add ../wt-engine      -b feature/substrate
git worktree add ../wt-extraction  -b feature/extraction
git worktree add ../wt-trees       -b feature/differential
git worktree add ../wt-ui          -b feature/ui
git worktree add ../wt-eval        -b feature/eval
```

Per-worktree agent rules:
- Work only inside your directory tree (see §5).
- Never edit `CLAUDE.md`, `PRD.md`, `Eng_doc.md`, `context.md`, `rules.md`. If you think one is wrong, open `docs/decisions/<date>_<topic>.md` proposing the change; the human decides.
- Never edit other worktrees' files.
- Commit to feature branch. Never push to `main`.
- Before each new slice: `git pull --rebase origin main` to pick up integrated interface changes.

---

## 5. Per-worktree scopes

### 5.1 wt-engine — Substrate core

**Owns**: `src/substrate/`.

**Scope**: fresh implementation of the minimum substrate defined in `Eng_doc.md` §2.1. Claims table, deterministic supersession (Pass 1), projections, admission filter, provenance utilities, event bus. ~710 LOC target.

**Reads first**: `Eng_doc.md` §2, §4, §5.1, §5.3 (Pass 1), §5.6 deps.

**Does not**: implement 7-signal retrieval, ACT-R, MMR, NLI, async workers, scope inference. Those are README vision.

### 5.2 wt-extraction — ASR + claim extractor + semantic supersession

**Owns**: `src/extraction/asr/`, `src/extraction/claim_extractor/`, `src/substrate/supersession_semantic.py`.

**Scope**: 
- Stand up Whisper large-v3 + WhisperX on one synthetic audio clip. Compare to Distil-Whisper for fallback. Report latency + WER on medical terms in `docs/asr_benchmark.md`.
- Claim-extractor prompt for Opus 4.7 with ≥5 few-shot examples covering: multi-claim utterance, negative finding, supersession (patient correction), rare symptom, ambiguous phrasing.
- e5-small-v2 embedding + 0.92 semantic identity threshold for supersession Pass 2.

**Reads first**: `Eng_doc.md` §3.1, §5.2, §5.3 Pass 2.

**Does not**: invent new predicate families. The closed set is in `Eng_doc.md` §4.2.

### 5.3 wt-trees — Clinical content + differential engine + verifier

**Owns**: `content/differentials/chest_pain/`, `src/differential/`, `src/verifier/`.

**Scope**:
- 4 branches (Cardiac, Pulmonary, MSK, GI) as tree structures in `branches.json`.
- LR table ≥15 features per branch, every one with a peer-reviewed citation, in `lr_table.json`. Sources in `sources.md`. Primary references: 2021 AHA/ACC Chest Pain Guideline (open), HEART/TIMI/Wells/PERC primary papers. No proprietary content (UpToDate, AMBOSS).
- Deterministic LR-weighted differential update engine.
- Counterfactual verifier (`Eng_doc.md` §6).

**Reads first**: `Eng_doc.md` §4.4, §5.4, §5.5, §6. `PRD.md` §6.5.

**Hard rule**: no invented numbers. Every LR has a citation. Approximations flagged.

### 5.4 wt-ui — Frontend

**Owns**: `ui/`.

**Scope**:
- Read `/mnt/skills/public/frontend-design/SKILL.md` before writing any UI — required.
- Scaffold React + TS + Tailwind + shadcn + ReactFlow + Zustand.
- Layout shell: disclaimer header, 4 panels, aux strip.
- One panel end-to-end first — **transcript panel**, because it validates ASR → substrate → UI.

**Reads first**: `PRD.md` §6. `Eng_doc.md` §7.

**Does not**: use localStorage / sessionStorage. Use Zustand.

### 5.5 wt-eval — Benchmark harnesses

**Owns**: `eval/`.

**Scope**: three published-benchmark harnesses. Each has an adapter that ingests the benchmark's format and emits turns into the substrate's write API; each has a variant-baseline + variant-full; each reports metrics per the benchmark's own evaluator.

- `eval/ddxplus/` — DDXPlus (NeurIPS 2022). Top-5 accuracy + HDF1 (ICD-10 hierarchical F1) per H-DDx 2025 methodology on the 730-case stratified subset. See `Eng_doc.md` §10.1.
- `eval/longmemeval/` — LongMemEval-S (ICLR 2025). Per-category accuracy using the official evaluator. Run all 500 questions — no slicing. See `Eng_doc.md` §10.2.
- `eval/aci_bench/` — ACI-Bench (Nature Scientific Data 2023). MEDIQA-CHAT metrics (ROUGE-1/2/L, BERTScore, MEDCON). Full `aci` + `virtscribe` test splits — no slicing. **Blocker**: MEDCON needs UMLS licence + QuickUMLS (https://uts.nlm.nih.gov/uts/signup-login, 1–3 business-day NIH approval). See `Eng_doc.md` §10.3.

**Reads first**: `Eng_doc.md` §10. `PRD.md` §8.

**Hard rule**: no homemade metrics. Report only metrics defined by the benchmark authors. No custom "substrate ablation" bar charts.

---

## 6. Planning-first workflow

For any task >50 LOC or touching more than one file:

1. **Plan out loud** first. List files touched, public interfaces changed, the test that will prove it works.
2. **Re-read** the relevant section of `Eng_doc.md` / `PRD.md`. Don't work from memory.
3. **Check interface conflicts** with other worktrees. Pull latest main first.
4. **Write the test first** for user-visible behaviour.
5. **Run the full test suite** before proposing a commit.

Short tasks (typos, one-file fixes) skip planning.

---

## 7. Commands

```bash
# Setup
./scripts/setup.sh

# Run app locally
./scripts/run_demo.sh --case chest_pain_01
./scripts/run_demo.sh --live                      # dev mic mode

# Tests
pytest tests/unit/
pytest tests/e2e/
pytest tests/property/
pytest tests/privacy/
pytest tests/licensing/
pytest                                            # all

# Evals
make eval                                         # all three benchmarks
make eval ddxplus
make eval longmemeval
make eval aci_bench
```

---

## 8. Style

- **Python**: ruff + pyright strict. Type public signatures.
- **TypeScript**: strict mode. `any` needs a reason.
- **Comments**: why, not what. Reference PRD/Eng_doc section when implementing a spec'd behaviour.
- **Commits**: `<area>: <imperative summary>`. `area ∈ {engine, extraction, trees, verifier, ui, eval, docs, infra}`.
- **No `except Exception: pass`.** Handle or propagate.
- **Logging over print** — `structlog` JSON with `session_id` and `claim_id` on every line.

---

## 9. Anthropic API / Opus 4.7

- Model string: `claude-opus-4-7`. Verify via `product-self-knowledge` skill before coding.
- Env var: `ANTHROPIC_API_KEY`. Never commit keys.
- Cache few-shot examples locally; don't re-read from disk per call.
- 429s → exponential back-off; don't blow up the UI.
- For evals: stagger concurrency; keep BioMistral-7B warm for offline fallback.

---

## 10. Skills to read before matching work

- **Any UI / React** → `/mnt/skills/public/frontend-design/SKILL.md` (required)
- **.docx** → `/mnt/skills/public/docx/SKILL.md`
- **.pptx** → `/mnt/skills/public/pptx/SKILL.md`
- **.xlsx** → `/mnt/skills/public/xlsx/SKILL.md`
- **.pdf** → `/mnt/skills/public/pdf/SKILL.md`
- **Anthropic API / Claude Code specifics** → `/mnt/skills/public/product-self-knowledge/SKILL.md`
- **Reading uploaded binary** → `/mnt/skills/public/file-reading/SKILL.md` or `pdf-reading/SKILL.md`

Hard rule. Reading takes seconds and keeps output aligned to environment.

---

## 11. When uncertain

1. Read the relevant section of `context.md` / `PRD.md` / `Eng_doc.md`.
2. If unclear, plan conservatively, note deviation in `docs/decisions/<date>_<topic>.md`, proceed.
3. Never fabricate a clinical number, citation, or behaviour.

Confident fabrication is the biggest failure mode. "I don't know, here's the best approximation and why" always beats invented detail.

---

## 12. Demo-blocking checklist (every merge to main)

- [ ] `pytest tests/property/test_determinism.py` passes
- [ ] `pytest tests/privacy/test_no_phi.py` passes
- [ ] `pytest tests/licensing/test_open_source.py` passes
- [ ] `./scripts/run_demo.sh --case chest_pain_01` runs without crash
- [ ] Disclaimer visible: UI header, README, demo bumper
- [ ] New datasets declared in `SYNTHETIC_DATA.md`
- [ ] Any new LR has a citation in `content/differentials/chest_pain/sources.md`

---

## 13. Scope discipline

- If you're about to add a second model stack, stop.
- If you're about to add a second UI screen, stop.
- If you're about to process data other than synthetic text and audio, stop.
- If you're about to invent a benchmark or metric, stop — use DDXPlus / LongMemEval / ACI-Bench.
- If you've been stuck >2 hours on one problem, stop and escalate or cut.

Ship the smallest thing that demonstrates the thesis. Everything else is README material.
