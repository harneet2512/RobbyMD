# Research Brief — 2026-04-21

Synthesis of four parallel research passes: GT v2 local study, LLM memory architecture frontier, clinical reasoning / verifier prior art, and benchmark prior art (DDXPlus, LongMemEval-S, ACI-Bench). Inputs to the fresh ~710 LOC substrate build.

Companion doc: `docs/gt_v2_study_notes.md` (per-module GT v2 learnings).

---

## 1. Factual corrections to existing planning docs

These are wrong in current `context.md` / `PRD.md` / `Eng_doc.md` and should be patched before anything else. I am **not** editing the notes files (per CLAUDE.md §4); surfacing for you to confirm.

| # | Where | Current | Correct |
|---|---|---|---|
| 1 | `context.md` §6 and `Eng_doc.md` §10.2 | "LongMemEval-S … EMNLP 2024" | **ICLR 2025** ([paper](https://arxiv.org/abs/2410.10813)). The arXiv preprint is Oct 2024; the venue is ICLR 2025. |
| 2 | `PRD.md` §8.1 + `Eng_doc.md` §10.1 (DDXPlus metrics) | "Top-1 / Top-3 / Top-5 pathology accuracy" | **H-DDx reports Top-5 accuracy + HDF1 only** — no Top-1/3. If we want prior-art overlay (Claude-Sonnet-4, GPT-4o, etc.), we must align to Top-5 + HDF1 or explicitly note "no comparator" for Top-1/3. Recommendation: switch to Top-5 + HDF1 ([H-DDx paper](https://arxiv.org/abs/2510.03700)). |
| 3 | `Eng_doc.md` §10.3 + `CLAUDE.md` §5.5 | "ACI-Bench MEDCON F1" as if it's just a number we compute | **MEDCON requires a UMLS license + QuickUMLS**. UMLS account approval is 1–3 business days. **Start the request today** or the eval is blocked on Day 5. [UMLS signup](https://uts.nlm.nih.gov/uts/signup-login). |

---

## 2. Strategic updates to the build

### 2.1 Upgrade supersession to **typed clinical edges** (strong defensible angle)

**Finding:** Across the 12 memory systems surveyed (Zep/Graphiti, Supermemory, TiMem, EverMemOS, MemOS, mem0, A-MEM, Letta, LangMem, Cognee, Mastra OM, Anthropic memory tool), only **Supermemory** has explicit typed supersession edges (`updates` / `extends` / `derives`). Zep has untyped bitemporal invalidation; every other system either overwrites, appends, or lets the LLM self-manage.

**Nobody exposes a clinical-semantic edge vocabulary.** That's white space we can defensibly own — and it maps directly to FDA Non-Device CDS Criterion 4 ("clinician can independently review the basis") and to differential-diagnosis evidence theory.

**Proposed schema change** (from `Eng_doc.md §4.1`):

```sql
supersession_edges.reason TEXT NOT NULL CHECK (reason IN (
    'patient_correction',   -- Pass 1 deterministic
    'physician_confirm',    -- Pass 1 deterministic
    'semantic_replace',     -- Pass 2 semantic
    'refines',              -- NEW: new claim narrows an older, broader one
    'contradicts',          -- NEW: physician or later patient statement refutes
    'rules_out',            -- NEW: tied to a differential branch becoming dead
    'dismissed_by_clinician' -- NEW: physician tap to dismiss
))
```

The four new edge kinds cost us maybe 20 LOC of discrimination logic and give the verifier + UI much richer semantics (and a better demo-video beat: "watch the *type* of supersession change, not just the value").

### 2.2 Two direct prior-art papers to cite for the verifier

Agent C found we have **direct 2025-2026 ancestors** for the counterfactual verifier. These should be cited in the written summary and repo README:

1. **[Counterfactual Multi-Agent Reasoning for Clinical Diagnosis](https://arxiv.org/abs/2603.27820)** (arXiv 2603.27820, 2026). Defines the **Counterfactual Probability Gap (CPG)** — exactly measures how much a diagnosis probability shifts when a clinical finding is removed or altered. Uses CPG to pick discriminative features across 3 benchmarks × 7 LLMs, with largest gains on ambiguous cases. **Our verifier is a lightweight deterministic instantiation of CPG on an LR table.**
2. **[MedEinst](https://arxiv.org/abs/2601.06636)** (arXiv 2601.06636, Jan 2026). Introduces **Bias Trap Rate** — pairs of cases where discriminative evidence is flipped; measures if models notice. Their **ECR-Agent** uses Dynamic Causal Inference. We can also add a small "trap" subset to our DDXPlus run as a secondary eval.

**Framing upgrade for the written summary:** *"A deterministic, single-model, ~200 LOC instantiation of the counterfactual-probability-gap principle ([CF Multi-Agent Dx 2026](https://arxiv.org/abs/2603.27820)) over an AHA/ACC-cited LR table — grounded, auditable, no second model stack."*

### 2.3 Top 3 papers for depth credibility in submission

1. **[AMIE (Nature 2025)](https://www.nature.com/articles/s41586-025-08866-7)** — the headline diagnostic-dialogue system. Our deliberate contrast: explicit Bayesian LR substrate + determinism vs. AMIE's end-to-end learned uncertainty reduction.
2. **[H-DDx (arXiv 2510.03700)](https://arxiv.org/abs/2510.03700)** — defines the modern DDXPlus evaluation protocol we're using (22-LLM comparator, ICD-10 hierarchical F1, 730-case stratified sample).
3. **[Counterfactual Multi-Agent Reasoning for Clinical Diagnosis (arXiv 2603.27820)](https://arxiv.org/abs/2603.27820)** or **[MedEinst (arXiv 2601.06636)](https://arxiv.org/abs/2601.06636)** — the direct ancestor of our counterfactual verifier.

Honourable mentions: [MEDDxAgent (ACL 2025)](https://aclanthology.org/2025.acl-long.677/) as the closest architectural peer (multi-agent; we're single-model); [MedR-Bench (Nat Comms 2025)](https://www.nature.com/articles/s41467-025-64769-1) for the "exam recommendation is the weak link" motivation that info-gain exploits; [HealthBench (OpenAI)](https://openai.com/index/healthbench/) for rubric-style evaluation positioning.

### 2.4 LR table can start from AAFP 2013 (already open, already structured)

**Finding:** No turnkey machine-readable LR JSON exists, but the **[AAFP 2013 "Outpatient Diagnosis of Acute Chest Pain"](https://www.aafp.org/pubs/afp/issues/2013/0201/p177.html)** is the single richest open-access LR table. Agent C extracted ~17 rows spanning all four branches (AMI, chest-wall / costochondritis, GERD, panic, pneumonia, heart failure, PE, aortic dissection). Supplement with HEART (LR+≈13 at ≥4) and TIMI (LR+≈6.8 at ≥3) from [Liu JECCM 2021](https://jeccm.amegroups.org/article/view/4088/html) for completeness.

**Implication:** `content/differentials/chest_pain/lr_table.json` can be seeded from AAFP 2013 on Day 1, cutting hand-curation time dramatically. Every row cites `aafp.org/pubs/afp/issues/2013/0201/p177.html` + row reference.

### 2.5 LongMemEval-S: target the hard categories explicitly

**Finding:** Full-context GPT-4o on `longmemeval_s` scores **45.1% temporal reasoning** and **44.3% multi-session** (per Zep's [independent eval](https://blog.getzep.com/state-of-the-art-agent-memory/)). These are the categories where long-context drops 30–50 points and where structured-memory systems (Zep +17.3pp on TR, +13.6pp on multi-session) win biggest.

**Implication:** Our LongMemEval-S slide should report the **per-category** numbers, with focus on TR + multi-session. Overall improvement is a softer story; per-category win on TR+multi-session is a direct architectural defence of the lifecycle-tracking thesis.

### 2.6 Eval-pipeline blockers to unblock today

1. **UMLS / QuickUMLS license** (MEDCON) — [signup here](https://uts.nlm.nih.gov/uts/signup-login), 1–3 business days.
2. **LongMemEval was re-cleaned Sept 2025** — pin to a specific commit SHA before running; pre-2025-09 results are not comparable. Use [`xiaowu0162/LongMemEval` HEAD @ 2026-04-21](https://github.com/xiaowu0162/LongMemEval) and log the SHA.
3. **Zep's self-reported 71.2 vs independent 63.8** on LongMemEval-S — known reproducibility wart. Cite the independent number when overlaying comparators.
4. **LLM judge pinning** — both DDXPlus (H-DDx) and LongMemEval use `gpt-4o` as judge. Pin to `gpt-4o-2024-08-06` and log. DDXPlus H-DDx doesn't pin by default; we must.

---

## 3. SOTA comparator numbers to sit next to ours on the demo slide

### 3.1 DDXPlus (730-case H-DDx stratified subset, Top-5 + HDF1)

From [H-DDx Table 2](https://arxiv.org/html/2510.03700v1) — 22 LLMs already reported; we sit next to them:

| Model | Top-5 | HDF1 | Category |
|---|---:|---:|---|
| Claude-Sonnet-4 | **0.839** | **0.367** | proprietary |
| Gemini-2.5-Flash | 0.832 | 0.348 | proprietary |
| Claude-Sonnet-3.7 | 0.836 | 0.338 | proprietary |
| GPT-4o | 0.804 | 0.350 | proprietary |
| GPT-5 | 0.783 | 0.345 | proprietary |
| GPT-4.1 | 0.801 | 0.339 | proprietary |
| Qwen3-235B-A22B | 0.777 | 0.322 | open |
| MedGemma-27B | 0.765 | 0.331 | medical FT |
| MediPhi | 0.666 | **0.353** | medical FT |

**No published Opus 4.7 number on H-DDx.** We'd be the first to report.

### 3.2 LongMemEval-S (`longmemeval_s`, GPT-4o judge)

| System | Overall | Driver | Source |
|---|---:|---|---|
| GPT-4o full-context baseline | 60.2–71.2 | GPT-4o | [Zep blog](https://blog.getzep.com/state-of-the-art-agent-memory/) |
| Zep / Graphiti | 63.8 (indep) / 71.2 (self) | GPT-4o | [arXiv 2501.13956](https://arxiv.org/abs/2501.13956) |
| TiMem | 76.88 | GPT-4o-mini | [arXiv 2601.02845](https://arxiv.org/abs/2601.02845) |
| EverMemOS | 83.0 | — | [arXiv 2601.02163](https://arxiv.org/abs/2601.02163) |
| MemOS | +40% over baseline (per-cat not public) | — | [arXiv 2507.03724](https://arxiv.org/abs/2507.03724) |
| Mastra OM | 94.87 | GPT-5-mini | [Mastra research](https://mastra.ai/research/observational-memory) |
| Supermemory "agent swarm" | ~99 | — | [Supermemory](https://blog.supermemory.ai/we-broke-the-frontier-in-agent-memory-introducing-99-sota-memory-system/) |
| MemPalace | 96.6 | — | [mempalace.tech](https://www.mempalace.tech/benchmarks) |

**Reality check:** if we run Opus 4.7 + our substrate honestly, we will not beat Mastra OM / Supermemory overall. But our **per-category TR + multi-session** numbers — if they beat baseline Opus 4.7 full-context by +15pp on those two hard categories — that's the architectural defence we need. We are not claiming SOTA overall. We are claiming "the structured-memory thesis fixes the known failure mode, measured honestly on one published benchmark."

### 3.3 ACI-Bench (aci subset, MEDIQA-Chat 2023 Task B)

Best published comparator on the `aci`-family test split is still the [WangLab GPT-4 ICL](https://arxiv.org/abs/2305.02220) from MEDIQA-Chat 2023 — **1st place**. No 2024–2026 LLM has published ACI-Bench numbers (checked AMIE, Med42-v2, MedGemma, Opus, GPT-5, Gemini 2.5). Lowest-risk comparator: GPT-4 zero-shot baseline from the original paper:

| Metric | BART + FT-SAMSum (Division) | GPT-4 zero-shot |
|---|---:|---:|
| ROUGE-1 | **53.46** | 51.76 |
| ROUGE-2 | **25.08** | 22.58 |
| ROUGE-L | **48.62** | 45.97 |
| MEDCON | — | **57.78** |

Source: [PMC10482860](https://pmc.ncbi.nlm.nih.gov/articles/PMC10482860/).

---

## 4. Architecture adjustments to the ~710 LOC substrate

Carried from GT v2 study (companion doc); summarised here as deltas from current `Eng_doc.md §2.1` table:

| Module | Current LOC | Revised | Rationale |
|---|---:|---:|---|
| `schema.py` | 100 | 100 | Add `spans` table or JSON col for per-character provenance (utterance_id, char_start, char_end) — enables sentence-level highlight in the demo video. Add `edge_type` to `supersession_edges`. |
| `claims.py` | 120 | 80 | Closed predicate family + structured-output Opus 4.7 prompt means we skip GT v2's `classify.py` complexity (13k LOC in GT v2; we need ~80). |
| `supersession.py` (Pass 1) | 80 | 70 | Deterministic match + typed edge discrimination. |
| `supersession_semantic.py` (Pass 2) | 80 | 70 *(or 0)* | **Decision needed.** Agent A recommends skipping entirely for single-session chest-pain scope (structural match is sufficient). Agent B recommends e5-small-v2 @ 0.88 as a candidate gate with LLM arbitrator. My recommendation: **keep but shrink to ~70 LOC** — it's the one clean demo beat that shows "PE onset" and "pulmonary embolism onset" unifying. Skippable if Day-4 time pressure bites. |
| `projections.py` | 150 | 100 | No temporal routing, no diversity, no budgeting. Just per-branch active-claims views + briefing. |
| `admission.py` | 60 | 30 | **Downgrade.** Single-session, controlled input → skip embedding-novelty. Keep ~30 LOC of noise-regex filtering only. GT v2's 0.15 novelty threshold is unvalidated and rejects 85% on an empty index (cold-start bug). |
| `provenance.py` | 60 | 60 | Unchanged. Critical for the demo moment — keep full size. |
| `event_bus.py` | 60 | 50 | Pub/sub for UI; simple in-memory implementation. GT v2 doesn't have this. |
| **Verifier (`src/verifier/`)** | — (not in §2.1) | **200** | Not part of substrate budget, per `Eng_doc.md §2.1`, but worth re-stating: top-2 CPG-style discriminator over the LR table + one Opus 4.7 call to render the next-best question. |
| **Total substrate** | ~710 | **~560** | Savings freed to go into verifier polish, LR-table curation, and eval-pipeline adapters. |

Integration entry point: one `on_new_turn(turn_text, session_id, turn_id)` orchestrator (~70 LOC) chains extract → verify → supersede → rebuild projections → publish events. Synchronous is fine for hackathon; the pattern is future-async.

---

## 5. Updated thresholds — clinical revalidation plan

From GT v2 study (17 thresholds in total), the ones that actually matter for our pipeline:

| Threshold | GT v2 default | Our Day-1 value | Revalidate on |
|---|---:|---:|---|
| Semantic supersession cosine | 0.92 | **0.88** (per Agent B's survey of 2025 systems) | 20 chest-pain pairs with paraphrases |
| Admission novelty | 0.15 | **N/A** (disabling) | — |
| NLI grounded | 0.70 | 0.70 | 20 note sentences, measure false-rejection rate |
| NLI ungrounded | 0.40 | 0.40 | same as above |
| Supersession auto-confirm | 0.90 | **0.0** (auto-confirm all — no human-review loop for hackathon) | — |
| Claim extraction confidence | 0.85 | **0.70** (more permissive — let verifier + projections weed out low-grounded claims) | extraction precision/recall on 20 synthetic dialogues |
| Differential-alive posterior | n/a | 5% | visual comfort on demo case |
| Delta to fire verifier-strip update | n/a | 10pp on top-2 | demo-case feel |

---

## 6. Known novelty claim and honest disclosure

**Claim:** "A single-model, deterministic, ~710 LOC clinical-reasoning substrate with typed supersession edges and per-sentence provenance — and a ~200 LOC counterfactual verifier / info-gain next-best-question module on top — evaluated against three published benchmarks (DDXPlus via H-DDx Top-5+HDF1, LongMemEval-S per-category, ACI-Bench MEDIQA-Chat 2023)."

**Disclosure of prior art** (for honesty + depth-of-knowledge signal):
- Counterfactual verifier: direct descendant of [CF Multi-Agent Dx (2603.27820)](https://arxiv.org/abs/2603.27820) and [MedEinst ECR-Agent (2601.06636)](https://arxiv.org/abs/2601.06636). Our contribution: deterministic + single-model + ~200 LOC, not multi-agent / RL.
- Bayesian info-gain question selection: canonical prior art back to QMR-DT, DXplain, ILIAD (1980s–90s). Modern restatement: [Bayesian LLM-enhanced history-taking (ScienceDirect 2025)](https://www.sciencedirect.com/science/article/pii/S2666521225000869).
- Claim-based memory with lifecycle: Zep/Graphiti, Supermemory, LangMem all do versions. Our contribution: clinical-typed edges + per-sentence provenance in single-session scope.
- Deterministic LR-weighted DDx projection: pre-LLM prior art (QMR-DT, HEART, Wells). Our contribution: wired as a pure function off a structured claim state, not as an LLM black-box.

This disclosure is a feature for judging (depth axis, 20%), not a bug.

---

## 7. Open questions for you

1. **LongMemEval-S scope.** Agent B's note: LongMemEval-S is explicitly multi-session. Our demo is single-session. Running the substrate on it is defensible (the substrate primitives still apply), but it's a stretch. Option A: run all 500 questions; Option B: run the TR + multi-session slice only (~180 questions, smaller eval-compute budget, directly targets the win story). **Recommendation: B.**
2. **Semantic supersession Pass 2**: keep (70 LOC) or drop? **Recommendation: keep** — it's one demo beat.
3. **MEDCON / UMLS licence** — do you want me to draft the request form or will you sign up today? ([signup](https://uts.nlm.nih.gov/uts/signup-login))
4. **Factual corrections** to `PRD.md` / `context.md` / `Eng_doc.md` (§1 table above) — do you want me to prepare a patch file for each, or will you edit in-place? Per CLAUDE.md §4 I cannot touch those files directly.
5. **Typed supersession edges (§2.1)** — upgrade the schema now, or leave for post-demo?

---

## Sources (selected; full citations inline)

- [AMIE — Nature 2025](https://www.nature.com/articles/s41586-025-08866-7)
- [H-DDx — arXiv 2510.03700](https://arxiv.org/abs/2510.03700)
- [Counterfactual Multi-Agent Dx — arXiv 2603.27820](https://arxiv.org/abs/2603.27820)
- [MedEinst — arXiv 2601.06636](https://arxiv.org/abs/2601.06636)
- [MEDDxAgent — ACL 2025](https://aclanthology.org/2025.acl-long.677/)
- [MedR-Bench — Nat Comms 2025](https://www.nature.com/articles/s41467-025-64769-1)
- [LongMemEval — ICLR 2025](https://arxiv.org/abs/2410.10813)
- [Zep / Graphiti — arXiv 2501.13956](https://arxiv.org/abs/2501.13956)
- [EverMemOS — arXiv 2601.02163](https://arxiv.org/abs/2601.02163)
- [TiMem — arXiv 2601.02845](https://arxiv.org/abs/2601.02845)
- [MemOS — arXiv 2507.03724](https://arxiv.org/abs/2507.03724)
- [Supermemory research](https://supermemory.ai/research/)
- [Mastra Observational Memory](https://mastra.ai/research/observational-memory)
- [DDXPlus — NeurIPS 2022](https://arxiv.org/abs/2205.09148)
- [ACI-Bench — Nature Sci Data 2023](https://www.nature.com/articles/s41597-023-02487-3)
- [WangLab MEDIQA-Chat 2023](https://arxiv.org/abs/2305.02220)
- [AAFP 2013 Chest Pain Dx](https://www.aafp.org/pubs/afp/issues/2013/0201/p177.html)
- [FDA 2026 CDS Final Guidance](https://www.fda.gov/media/191560/download)
