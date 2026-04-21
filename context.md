# context.md

## 1. What this project is

A hackathon submission for **Built with Opus 4.7: a Claude Code Hackathon** by Cerebral Valley × Anthropic. Not a clinical product. Not a medical device. Not intended for use in patient care.

Working name: **Groundtruth Clinical Substrate** (internal). Final demo name TBD.

The product is a live clinical reasoning **substrate** that sits between a doctor and a patient during a consultation. It listens to the conversation, extracts structured clinical claims in real time, tracks how those claims evolve (including patient self-corrections), runs parallel differential-diagnosis hypothesis trees, surfaces supporting/contradicting evidence and the next-best-question for the top two hypotheses, and generates a SOAP note with full per-sentence provenance back to the source conversation turn.

One-line thesis: **in messy workflows the final output is only one layer; underneath it there must be a structured context layer that tracks facts, their sources, and their lifecycle — supersession, confirmation, contradiction — deterministically.**

## 2. Hackathon facts (authoritative)

- **Submission deadline**: Sunday, April 26, 8:00 PM EST, via the Cerebral Valley platform.
- **Deliverables**: 3-minute demo video (YouTube/Loom/similar) + public GitHub repo + 100–200 word written summary.
- **Team size**: up to 2. This is a solo build.
- **First-round judging**: April 26–27, asynchronous.
- **Finals**: April 28, 12:00 PM EST, top-6 live judging, top-3 announced at closing ceremony.
- **Judging weights** (verbatim from event page):
  - **Impact — 30%**
  - **Demo — 25%**
  - **Opus 4.7 Use — 25%**
  - **Depth & Execution — 20%**
- **Main prizes**: 1st $50k, 2nd $30k, 3rd $10k in Claude API credits.
- **Special prizes targeted**:
  - **Best use of Claude Managed Agents** ($5k) — our multi-agent build orchestration (git worktrees, sub-agents) and internal agent wiring.
  - **Most Creative Opus 4.7 Exploration** ($5k) — parallel claim extraction + supersession + counterfactual verifier is a creative composition of the model.

## 3. Hackathon rules we must satisfy (verbatim)

1. **Open Source**: *"Everything shown in the demo must be fully open source. This includes every component — backend, frontend, models, and any other parts of the project — published under an approved open source license."*
2. **New Work Only**: *"All projects must be started from scratch during the hackathon with no previous work."*
3. **Banned**: projects that violate legal/ethical/platform policies or use code/data/assets the team doesn't have rights to.

**How we comply:**

- **Every line of code in the hackathon repo is written during the hackathon window.** Prior thinking — architectural ideas, vocabulary (claims, supersession, projections, predicate families), the `GT_V2_THEORY.md` research note — is carried over as *ideas*. No files are copied from any pre-existing codebase. The substrate is **re-implemented fresh**, informed by prior thinking. If asked during judging: ideas came from earlier research notes; every line of code in this repo was written during the event window.
- **All models are OSI-approved open source**:
  - Whisper large-v3 (MIT)
  - WhisperX (BSD-4-Clause) for diarisation
  - Distil-Whisper large-v3 (MIT) for speed fallback
  - e5-small-v2 (MIT) for embeddings
  - BioMistral-7B (Apache 2.0) — optional offline fallback for clinical reasoning
  - **Not using**: MedASR or MedGemma (Gemma terms, not OSI-approved), Deepgram or any commercial ASR API.
- **Claude Opus 4.7 is the hackathon's named sponsored API.** The event is literally *Built with Opus 4.7*. Every team uses it. If a judge raises the open-source rule about Opus 4.7, the answer is: it's the event's named tool, not a third-party proprietary component we're shipping. Same category as using GitHub or Vercel.
- **No data without rights**: all clinical data is synthetic or from peer-reviewed public benchmark datasets (DDXPlus, LongMemEval, ACI-Bench — all with redistribution licenses).

## 4. Prior thinking we are carrying (ideas, not code)

Architectural thesis and vocabulary come from a pre-existing research note, `GT_V2_THEORY.md` (~7KB, architectural hypotheses only, no implementation). Concepts carried:

- Claims as first-class objects with lifecycle and provenance.
- Supersession as an explicit edge, not an overwrite.
- Deterministic projection layer over active claims.
- Information-gain admission filter.
- Semantic identity via embeddings for supersession matching.
- Predicate families as a closed, domain-specific vocabulary.

These are *ideas*. The code that implements them in this repo is written fresh during the hackathon.

## 5. Scope of the demo build

- **One chief complaint**: chest pain.
- **Four parallel differential branches**: Cardiac, Pulmonary, Musculoskeletal, GI. Branch design cross-referenced against DDXPlus's 49-pathology schema so evals line up.
- **One scripted standardised patient case** with one clean supersession moment.
- **One screen**, four panels (transcript, claim state, differential trees, SOAP note) + auxiliary strip ("Why this changed / Next best question").
- **Three published-benchmark eval runs** — see §6.

Beyond-demo vision (multi-complaint, multi-specialty, EHR/FHIR, multi-session longitudinal memory, full substrate including 7-signal retrieval and NLI verification) lives in `README.md`, not the build.

## 6. Evals — published benchmarks only

Homemade metrics are vanity numbers. Published benchmarks carry prior-art comparisons and judges recognise them. We run **three** benchmarks, each targeting a specific layer of our architecture.

| Benchmark | Source | Layer tested | Primary metric |
|---|---|---|---|
| **DDXPlus** | Fansi Tchango et al., NeurIPS 2022. 1.3M synthetic patients, 49 pathologies, differential labels. Widely used for 2025–26 LLM diagnostic evaluation (H-DDx benchmarks 22 LLMs on this). | Differential reasoning engine + verifier | Top-5 pathology accuracy + HDF1 (ICD-10 hierarchical F1) on the 730-case stratified test subset per the H-DDx 2025 methodology (matches H-DDx Table 2's 22-LLM comparator). Top-1/Top-3 computed internally only. |
| **LongMemEval-S** | Wu et al., ICLR 2025. 500 questions across 5 memory abilities. Published baselines: TiMem 76.88%, EverMemOS 83.0%, Zep/Graphiti 71.2%, MemOS, RMM. | Memory substrate (claim lifecycle + supersession + projection) | Per-category accuracy, especially knowledge-update, temporal reasoning, multi-session reasoning — the categories where long-context LLMs drop 30–60%. |
| **ACI-Bench** | Yim et al., Nature Scientific Data 2023. MEDIQA-CHAT / MEDIQA-SUM 2023 shared-task dataset. 207 doctor-patient dialogues + gold notes. | End-to-end conversation → clinical note | Standard MEDIQA-CHAT metrics: ROUGE-1/2/L, BERTScore, MEDCON (clinical-concept F1). |

**No homemade "substrate ablation" metrics. No custom bar charts of our own invention.** Every number on the demo slide traces to a published benchmark with prior-art comparison available.

Full eval spec in `Eng_doc.md` §10.

## 7. Regulatory posture

### 7.1 FDA: Non-Device Clinical Decision Support

Positioned as **Non-Device CDS** under Section 520(o)(1)(E) of the FD&C Act, per the FDA's **Clinical Decision Support Software: Final Guidance** (January 6, 2026 / re-issued January 29, 2026). All four criteria must hold:

1. **No image/IVD/signal-acquisition-system analysis.** Text derived from speech only. No ECG, imaging, continuous physiologic streams.
2. **Displays medical information** — symptoms, history, findings stated in conversation.
3. **Supports but does not direct** HCP judgement. Trees show alive/dead hypotheses; "next best question" is informational.
4. **HCP can independently review the basis** — provenance on every claim, citation on every LR, traceability on every note sentence.

**Red lines** (violating any turns the project into a regulated device):
- No time-critical / emergency framing.
- No singular diagnostic output to be acted on without review.
- No ECG / imaging / waveform / continuous-sensor processing.
- No patient-facing mode.

### 7.2 HIPAA: zero PHI

- Synthetic or public benchmark data only. DDXPlus = synthetic, ACI-Bench = research-cleared role-play, LongMemEval = synthetic chat.
- No real patient data, de-identified or otherwise.
- No third-party clinical APIs implying BAA requirements.
- `SYNTHETIC_DATA.md` declares provenance of every dataset. A static-analysis test (`tests/privacy/test_no_phi.py`) scans for real-looking identifiers.

### 7.3 Disclaimer (verbatim, appears in UI header, README, demo video bumper + outro, submission summary first three sentences)

> **Research prototype. Not a medical device.** This software is a research demonstration. It does not diagnose, treat, or recommend treatment, and is not intended for use in patient care. All patients, conversations, and data shown are synthetic or from published research benchmarks. Clinical decisions are made by the physician. This system supports the physician's reasoning by tracking claims and differential hypotheses in real time; it does not direct clinical judgement.

### 7.4 State AI-in-healthcare laws

HCP-in-the-loop posture is consistent with Colorado SB24-205, Texas HB 1709 (TRAIGA), and California SB 1120. No feature makes a coverage/diagnostic/treatment decision without HCP review.

## 8. The architectural bet: counterfactual verifier + next-best-question

On top of the deterministic differential engine, a verifier module computes, from the same claim state: (a) supporting evidence present, (b) discriminative evidence missing or contradicted, (c) the single next question that maximally reduces uncertainty between the top two hypotheses.

The difference between "showing what's alive" and "showing why it's alive, why something else is dying, and what to ask next." Implemented as ~200 lines on top of the LR table — no second model stack, no additional latency-critical path. Details in `Eng_doc.md` §6.

## 9. Scope for demo vs README vision

| Piece | Demo | README vision |
|---|---|---|
| Chest pain, 4 branches | ✅ | ✅ + respiratory distress, abdominal pain, altered mental status |
| Live claim extraction + supersession | ✅ | ✅ |
| Deterministic projection + LR-based tree update | ✅ | ✅ + Bayesian-network refinement |
| Counterfactual verifier + next-best-question strip | ✅ | ✅ + full hypothesis-driven question pipeline |
| SOAP note with per-sentence provenance | ✅ | ✅ |
| DDXPlus + LongMemEval-S + ACI-Bench runs | ✅ | ✅ + MedQA, DDXPlus-Full, full ACI-Bench test splits |
| Full substrate (7-signal retrieval, ACT-R, MMR, NLI, async worker, scope inference) | ❌ | ✅ |
| Multi-complaint, multi-specialty | ❌ | ✅ |
| EHR / FHIR integration | ❌ | ✅ |
| Multi-session longitudinal patient memory | ❌ | ✅ |

## 10. Build ordering (dependency-driven, not calendar-driven)

Work is ordered by dependency, not by calendar day. Pace is determined by effort and execution, not by the clock. Parallelism across worktrees where dependencies permit.

```
Foundation (no deps) — start in parallel:
  F1. Repo scaffold + disclaimer + SYNTHETIC_DATA.md + privacy test
  F2. Clinical content: 4 branches for chest pain + LR table with citations
  F3. ASR pipeline running end-to-end on one synthetic audio clip
  F4. Substrate core: claims table, deterministic supersession, projection layer

Builds on foundation:
  E1. Claim extraction  (deps: F3, F4)
  E2. Differential update engine  (deps: F2, F4)
  E3. Semantic supersession pass  (deps: F4)

Builds on engine:
  V1. Counterfactual verifier + next-best-question  (deps: E2)
  N1. Note generator with provenance  (deps: F4)
  U1. UI — 4 panels + aux strip  (deps: E1, E2, V1, N1)

Integration:
  I1. Scripted chest-pain case runs clean end-to-end
  I2. Determinism property test passes
  I3. End-to-end test passes

Evals (parallel, can start once E2 and N1 exist):
  B1. DDXPlus harness + run
  B2. LongMemEval-S harness + run
  B3. ACI-Bench harness + run

Submission pre-reqs (all must be green):
  S1. Compliance checklist (rules.md §9) green
  S2. Repo builds clean in <15 min
  S3. Demo video final cut (≤3 min)
  S4. Written summary (100–200 words)
  S5. GitHub repo public
  S6. Submission platform entry
```

Submit when S1–S6 are green. Not on a fixed day.

## 11. Non-goals

- Fine-tuning any model.
- Real-time voice synthesis or agent speaking back.
- EHR / FHIR integration.
- Multi-patient, multi-session, longitudinal.
- Any feature requiring FDA device oversight.
- Any feature added after the demo video's first cut (bug fixes only).

## 12. If in doubt

- **Pick smaller scope.** Shipped beats ambitious.
- **If it touches regulation, default to "we don't do that."**
- **If it cannot be traced to a source, cut it.**
- **If you're about to add a second model stack, stop.**
- **If an eval number cannot be compared to a published baseline, don't report it.**
