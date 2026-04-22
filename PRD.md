# PRD.md — Product Requirements Document

**Status**: Draft v2
**Scope**: Hackathon demo build
**Related**: `context.md` (hackathon framing + compliance), `Eng_doc.md` (engineering), `CLAUDE.md` (agent ops), `rules.md` (non-negotiables)

Before anything in this document takes effect, `context.md` §7 (regulatory posture) and `rules.md` must be satisfied. Product decisions in conflict are invalid.

---

## 1. Product Definition

A **clinical reasoning interface** that converts a doctor-patient conversation into structured clinical state and displays evolving differential hypotheses in real time to the physician.

Three outputs, each with complete provenance:

1. **Structured clinical claims** — discrete, lifecycle-tracked facts from natural conversation.
2. **Ranked differential hypotheses** — parallel diagnostic trees updated deterministically from claim state.
3. **A SOAP note** — every sentence traceable to claims and conversation turns.

The system **does not diagnose or recommend treatment.** The physician makes every clinical decision. The system supports the physician's reasoning so nothing is lost, forgotten, or hallucinated. This is the posture that keeps us inside the FDA Non-Device CDS envelope. Non-negotiable. See `context.md` §7.1.

---

## 2. Objective

During a patient interview the physician simultaneously: tracks evolving patient statements, catches contradictions, maintains multiple hypotheses, and documents the encounter. Two failures dominate: information is lost between conversation and chart; contradictions in the patient's story are missed because they cross attention boundaries. This product externalises the physician's reasoning as structured state so the physician can stay present in the room and the documentation falls out the back end.

---

## 3. Scope (demo)

| Dimension | Demo scope |
|---|---|
| Chief complaint | Chest pain (seeded + rehearsed); any other complaint also loadable via `clinical_general` pack per `Eng_doc.md §4.2` |
| Differential branches | Cardiac, Pulmonary, Musculoskeletal, GI |
| Conversations | One scripted standardised patient case with one clean supersession moment |
| Users | Physician only. No patient-facing mode |
| Deployment | Local, single workstation. Anthropic API for Opus 4.7 |
| Data | 100% synthetic or from public benchmarks (LongMemEval-S / ACI-Bench) |
| Sessions | Single-session. No cross-visit longitudinal memory |
| Language | English |

Explicitly out of scope: multi-complaint, multi-specialty, multi-session memory, EHR/FHIR, patient-facing views, voice synthesis, imaging/ECG/IVD processing.

---

## 4. User and environment

**User**: physician in a primary-care-style single-visit encounter. Not ED. Not time-critical use — time-critical framing is a higher-risk automation-bias signal under the 2026 FDA guidance and we deliberately avoid it.

**Environment**: quiet exam room, single microphone, one patient, one physician.

**Hardware**: single workstation with GPU (see `Eng_doc.md` §9).

---

## 5. User workflow

1. Physician starts a new session. One button; no patient identifiers collected.
2. Patient describes symptoms; physician asks questions naturally.
3. System transcribes speech with speaker separation.
4. Claims extracted as they arrive.
5. Claim lifecycle engine tracks active vs superseded; supersession edges created on corrections or physician confirmations.
6. Parallel differential trees re-rank deterministically after each claim. Alive branches highlighted; dead branches dimmed but never removed.
7. Counterfactual verifier shows supporting/contradicting evidence for the top two hypotheses and surfaces one next-best question.
8. Physician may tap any claim, branch, or suggested question to confirm / dismiss / flag. Every tap recorded with claim-state snapshot.
9. End of encounter: SOAP note rendered. Every sentence clickable → source claim + source turn.

---

## 6. Interface

Single screen, four regions + one auxiliary strip. Designed for one physician at one workstation.

### 6.1 Panel 1 — Transcript (left, ~25% width)

Live, scrolling, diarised.

**Fields per turn**: `speaker`, `timestamp`, `utterance_text`, `turn_id`.
**Interaction**: click a turn → highlight every downstream claim and note sentence.
**Latency**: ≤1.5 s end-of-utterance → displayed text.

### 6.2 Panel 2 — Claim state (centre-top, ~40% × 40%)

**Fields**: `claim_id`, `subject`, `predicate`, `value`, `status` ∈ {active, superseded, confirmed, dismissed}, `source_turn_id`, `confidence`, `timestamp`, `superseded_by`, `supersedes`.

**Rendering**: active black; confirmed check; superseded struck-through grey with arrow to the claim that replaced it. Hover → predicate family + source turn.

### 6.3 Panel 3 — Parallel differential trees (centre-bottom, ~40% × 50%)

Four trees side-by-side via ReactFlow.

**Per-node fields**: `feature`, `likelihood_ratio`, `source_citation`, `state` ∈ {unasked, evidence-present, evidence-absent, contradicted}.

**Per-tree score**: log-likelihood sum, softmax-ranked across branches. Re-ranks on every claim change with visible animation (~200 ms).

**Alive/dead threshold**: tree dims below configurable posterior (default 5%). Never removed — physician must always see what was ruled out.

### 6.4 Panel 4 — SOAP note (right, ~35%)

Live draft.

**Rule**: every sentence has non-empty `source_claim_ids`. Sentences without valid provenance are rejected by the post-hoc validator and never shown.

**Interaction**: click a sentence → highlight source claims + source turns.

### 6.5 Auxiliary strip — "Why this changed / Next best question"

Three fields, updated when differential ranking shifts by more than a configurable delta (default 10 pp on any top-two hypothesis):

- **Why this moved**: ≤2 bullets. E.g. *"Pulmonary ↑: pleuritic pain now present. Cardiac ↓: pain worse with breathing contradicts classic exertional pattern."*
- **Missing or contradicting evidence**: ≤2 bullets.
- **Next best question**: one question ≤20 words + a one-line rationale. E.g. *"Any recent long travel or swelling in the legs? — separates PE from MSK at the top of the differential."*

### 6.6 Persistent disclaimer

Always visible in header: *"Research prototype — not a medical device. Synthetic data. The physician makes every clinical decision."*

---

## 7. Capabilities

### C1 — Medical speech recognition

Transcribes clinical English with medical terminology accuracy, speaker diarisation, low-enough latency for a live visit.

**Stack** (all OSI-approved): Whisper large-v3 (MIT) + WhisperX (BSD) for diarisation. Distil-Whisper large-v3 (MIT) for speed fallback.

**Medical-term accuracy strategy**: custom vocabulary/hotword list derived from public drug and anatomy sources, injected at decoding time. Our code, Apache-2.0.

**Demo moment**: side-by-side generic Whisper vs Whisper + custom vocab on one utterance (e.g. "meta pearl roll" vs "metoprolol"). Two seconds, visceral.

**Targets**: medical-term WER ≤ 12%; RTF ≤ 0.7 on a 16–24 GB GPU; end-of-utterance → text ≤ 1.5 s.

### C2 — Claim extraction

**Engine**: Opus 4.7 (sponsored API) with structured-output prompting and few-shot clinical examples.
**Input**: current turn + 2 prior turns + current active claim set + closed predicate family.
**Output**: zero or more claims, validated against the schema, each with `source_turn_id` and `confidence`.
**Target**: evaluated indirectly via LongMemEval-S (`personal_assistant` pack) and ACI-Bench (`clinical_general` pack) end-to-end (see §8).

### C3 — Claim lifecycle management

States: **active**, **superseded**, **confirmed**, **dismissed**.

**Supersession rule (deterministic Pass 1)**: same `(subject, predicate)`, different `value` → supersession edge, reason `patient_correction` or `physician_confirm`.

**Supersession rule (semantic Pass 2)**: if Pass 1 misses due to phrasing, embed the `(subject, predicate, value)` tuple and match against active claims. Above 0.92 cosine identity with the same predicate family → supersession, reason `semantic_replace`. Threshold calibrated empirically.

**Guard**: supersession never fires between claims from the same turn (they're additive).

### C4 — Differential update (deterministic)

Each active claim updates per-branch log-likelihood via a published LR table (HEART, TIMI, Wells, PERC, 2021 AHA/ACC Chest Pain Guideline). Softmax across branches for ranking. Same claim set → bit-identical ranking. Enforced by a property test.

### C5 — Counterfactual verifier + next-best-question (hero feature)

For the top two branches at every update: enumerate supporting evidence present, enumerate discriminative evidence missing or contradicted, select the discriminator feature that maximises expected information gain between the top two, render as one natural-language question via Opus 4.7.

Details in `Eng_doc.md` §6. No additional model stack.

### C6 — Note generation with provenance

Opus 4.7 composes SOAP sentences as tuples `(section, text, [claim_id, ...])`. Post-hoc validator rejects any sentence without a non-empty, valid `claim_ids` chain. Evaluated against ACI-Bench (see §8).

### C7 — Decision recording

Every physician tap (confirm / dismiss / flag / tap-node / accept-question) is recorded with the full active-claim-state snapshot at the moment of the tap.

---

## 8. Evaluation — published benchmarks only

**Two** benchmarks (dropped DDXPlus + MedQA 2026-04-21 — see `reasons.md`). Each targets a specific substrate surface. No homemade "substrate ablation" bar chart. Every number on the demo slide traces to a published benchmark with prior-art comparison available. Reader + judge pins in `eval/README.md`; model-usage policy in `Eng_doc.md §3.5`.

### 8.1 LongMemEval-S (memory substrate)

**Source**: Wu et al., *LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory*, ICLR 2025.

**Our setup**: LongMemEval-S (the short variant, ~115K tokens context depth). Substrate ingests the session history as conversational turns under the `personal_assistant` pack (seeded `predicate_packs/personal_assistant/`). System answers the 500 questions using substrate-backed retrieval instead of long context.

**Variants**:
- Baseline: reader direct with full conversation history in context (no substrate).
- Our system: reader + substrate providing retrieved active claims.

**Readers**: primary `Qwen2.5-14B-Instruct` (self-hosted); secondary `gpt-4o-mini` (matches Mem0 / Mastra OM / EverMemOS leaderboard). Judge pinned `gpt-4o-2024-08-06`.

**Metrics**: per-category accuracy — information extraction, multi-session reasoning, temporal reasoning, knowledge update, abstention.

**Published prior art**: TiMem 76.88%, EverMemOS 83.0%, Zep/Graphiti 71.2%, Mastra OM 94.87%. We report alongside.

**Why this matters**: LongMemEval-S is the benchmark where long-context LLMs drop 30–60% on the hard categories (knowledge-update, temporal reasoning, multi-session reasoning) — the exact failure mode our substrate claims to fix.

### 8.2 ACI-Bench (conversation → clinical note)

**Source**: Yim et al., *ACI-BENCH: a Novel Ambient Clinical Intelligence Dataset for Benchmarking Automatic Visit Note Generation*, Nature Scientific Data 2023. MEDIQA-CHAT 2023 / MEDIQA-SUM 2023 shared-task dataset.

**Our setup**: full ACI-Bench test splits — `aci` (66 test encounters across test1/test2/test3) and `virtscribe` (24 test encounters). Pipeline loads `clinical_general` pack; ingests the provided dialogue (gold and ASR versions); produces a SOAP note; compared to the gold note. No slicing.

**Variants**:
- Baseline: reader direct, dialogue → note, no substrate.
- Our system: dialogue → claims → note via substrate.

**Readers**: primary `Qwen2.5-14B-Instruct`; secondary `gpt-4.1-mini` (modern GPT-4-class; WangLab 2023 GPT-4 ICL comparator). No LLM judge (ROUGE/BERTScore/BLEURT deterministic; MEDCON via 3-tier fallback — `docs/decisions/2026-04-21_medcon-tiered-fallback.md`).

**Metrics**: standard MEDIQA-CHAT — ROUGE-1 / ROUGE-2 / ROUGE-L, BERTScore, MEDCON (clinical-concept F1).

**Published prior art**: MEDIQA-CHAT 2023 leaderboard (WangLab GPT-4 ICL first place) + subsequent papers.

### 8.3 Smoke-first discipline

Before any full run, `eval/smoke/run_smoke.py` runs a deterministic first-10-case sanity pass with a hard `--budget-usd` cap. Verdict ✅ PASS / ⚠ ANOMALY / ❌ FAIL vs `eval/smoke/reference_baselines.json`. Smoke harness built this session; real run held until operator sign-off.

### 8.4 Eval results slide (end of demo video)

One clean slide, two mini-charts:
- LongMemEval-S: our per-category accuracy vs published substrates (TiMem, EverMemOS, Mastra OM).
- ACI-Bench: our MEDIQA-CHAT scores (ROUGE + BERTScore + MEDCON) vs WangLab 2023 GPT-4 ICL + published 2023 leaderboard.

Ten seconds on screen. Numbers, not bars for bars' sake.

---

## 9. Demo video (3 minutes)

- **00:00–00:15** — Disclaimer card, then a 5-second framing line ("What if we could see the doctor's reasoning as it happens?"). House MD reference optional, subtle.
- **00:15–00:30** — Generic Whisper vs Whisper+vocab on one clinical sentence. Visceral ASR moment.
- **00:30–01:30** — Standardised chest pain case running end-to-end. Claims populate. Trees breathe. Patient corrects onset. Supersession fires visibly. Trees recalculate. Verifier strip updates. Physician taps a suggested next question.
- **01:30–02:10** — SOAP note renders. Click a sentence → source claim highlights in Panel 2, source turn highlights in Panel 1. The provenance-is-the-hero moment.
- **02:10–02:40** — Architecture diagram: conversation → claims → supersession → projections → differential → verifier → note. 30 s.
- **02:40–02:55** — Published-benchmark results slide.
- **02:55–03:00** — Disclaimer card, closing tagline.

---

## 10. Definition of Done

Submittable when:

1. Scripted chest-pain case runs end-to-end on one workstation, producing all four panels + auxiliary strip.
2. Claims populate with provenance to source turns.
3. Supersession fires visibly in the demo case and updates claims, trees, and auxiliary strip.
4. Differential trees re-rank within 200 ms of a claim-state change.
5. Verifier strip produces a non-trivial next-best question at the correct moment.
6. SOAP note generates with every sentence traceable to source claims.
7. LongMemEval-S and ACI-Bench runs completed with numbers captured (DDXPlus + MedQA dropped 2026-04-21).
8. 3-minute demo video recorded and edited.
9. 100–200 word written summary drafted with mandatory framing from `context.md` §7.3.
10. All disclaimers in code, UI, README, video. `rules.md` §9 compliance checklist green.
11. All data declared in `SYNTHETIC_DATA.md`.
12. Repo builds from clean clone with `./scripts/setup.sh && ./scripts/run_demo.sh --case chest_pain_01` in under 15 min.

Any gap in 1–12 → priority is to close the gap, not add features.

---

## 11. Open questions (resolve during build)

1. First smoke-run verdict on `eval/smoke/run_smoke.py` (LongMemEval-S + ACI-Bench × baseline + substrate × 10 cases, Qwen2.5-14B reader, $50 budget cap) — gate before any full run.
2. LongMemEval-S adapter strategy: feed sessions via substrate write API under `personal_assistant` pack, evaluate via the official LongMemEval evaluator.
3. Custom vocabulary source for ASR — start with a public drug/anatomy list; expand only if WER on the demo case misses.
4. Semantic supersession threshold — default 0.92, tune empirically on LongMemEval-S + ACI-Bench dialogue.
5. wt-ui dispatch progress — transcript panel end-to-end on `feature/ui` scaffold (`b5529f7`); next steps: claim-state panel, differential trees, SOAP panel per `CLAUDE.md §5.4`.
