# rules.md — Non-Negotiable Rules

Invariants. Not suggestions. Cannot be violated by any contributor, any agent, or any demo compromise.

If a rule here conflicts with `PRD.md`, `Eng_doc.md`, or `CLAUDE.md`, this file wins.

If following a rule is about to break the demo, the answer is not to break the rule — it is to cut the feature, adjust scope, and re-record.

---

## §1. Hackathon compliance

### 1.1 Fresh code only
Every line of code in this repository is written during the hackathon window. No files are copied from any pre-existing codebase. Prior thinking (ideas, vocabulary, research notes such as `GT_V2_THEORY.md`) is carried over as ideas only; code is not. If asked during judging, the honest answer is: ideas came from earlier research notes; every line of code in this repo was written during the event window.

### 1.2 Open source only

Every component shipped in the demo is under an appropriate open license for its type:

- **Code** (backend, frontend, build tooling, inference pipelines): OSI-approved license — MIT, BSD, Apache-2.0, MPL, ISC, LGPL, or equivalent.
- **Model weights and datasets**: OSI-approved license **OR** open-data license permitting commercial use with attribution — CC-BY-4.0, CC-BY-SA-4.0, CDLA-Permissive-2.0, ODbL, or equivalent. Attribution is recorded in `MODEL_ATTRIBUTIONS.md` (for model weights) and `SYNTHETIC_DATA.md` (for datasets).
- **The sponsored Claude API** is permitted as the platform tool of the event (pending Discord confirmation; see `docs/decisions/licensing_clarifications.md`).

**Excluded regardless of category**: non-commercial licenses (CC-BY-NC, CC-BY-NC-SA), no-derivatives licenses (CC-BY-ND, CC-BY-NC-ND), licenses with custom use restrictions (Gemma Terms of Use, HAI-DEF, Llama 2/3 Community Licenses), and any commercial-only API for components other than the sponsored Claude API.

This split follows the Linux Foundation's OpenMDW framework (July 2025) and the broader ML-licensing consensus that model weights are data, not code. See `reasons.md` entry "Strict OSI-only licensing for model weights — rejected for industry-standard reading (2026-04-21)" for the full rationale and citations.

**Enforcement**:
- `tests/licensing/test_open_source.py` — scans `pyproject.toml` and `package.json` for code-dependency license compliance (OSI allowlist).
- `tests/licensing/test_model_attributions.py` — scans `src/` for model-load call sites (`huggingface_hub.snapshot_download`, `AutoModel.from_pretrained`, `whisperx.load_model`, `faster_whisper.WhisperModel`, `pyannote.audio.Pipeline.from_pretrained`, `SentenceTransformer`) and verifies every referenced model identifier appears in `MODEL_ATTRIBUTIONS.md` with its license and attribution line.

### 1.3 Team size, deliverables, deadline
Team size ≤2 (solo). Deliverables: 3-minute demo video + public GitHub repo + 100–200 word written summary. Deadline: **Sunday, April 26, 8:00 PM EST**.

### 1.4 Written summary opening
The 100–200 word written summary opens with the non-diagnostic framing. The phrase *"supports the physician's reasoning — the physician makes every clinical decision"* appears within the first three sentences.

---

## §2. Data and privacy (HIPAA + beyond)

### 2.1 Zero PHI
No real patient audio, transcripts, EHR records, clinical notes, or de-identified real data. No "for a quick test" exceptions.

### 2.2 All data synthetic or from public benchmarks, and declared
Every dataset, audio file, transcript, vignette, and labelled case is declared in `SYNTHETIC_DATA.md` with its provenance.

Allowed sources:
- Our own synthetic generation (with declared generation prompt).
- Actor-recorded role-play against a scripted case (with scripts checked in and attestation that no PHI appears).
- Published benchmark datasets with redistribution license: **LongMemEval-S** (MIT, synthetic), **ACI-Bench** (CC BY 4.0, research-cleared role-play). DDXPlus + MedQA dropped 2026-04-21 — see `reasons.md`.

### 2.3 No third-party clinical APIs implying BAA
No Infermedica, no Isabel, no Nuance, no clinical-workflow API that returns patient-identifiable fields or implies a Business Associate Agreement. Anthropic API (Opus 4.7) is covered by Anthropic's standard terms for hackathon use.

### 2.4 PHI sentinel must stay green
`tests/privacy/test_no_phi.py` scans for real-looking identifiers (SSN patterns, MRN-shaped IDs, real-looking DOBs, common-name matches). If it flags anything, the offending file is removed before the next commit. No exceptions.

### 2.5 Demo video
Nothing shown in the video could be mistaken for real patient data. Names are obviously fictional. Dates of birth are obviously synthetic. The disclaimer is visible throughout.

---

## §3. Regulatory (FDA Non-Device CDS)

### 3.1 Four criteria are load-bearing
The product must satisfy all four Non-Device CDS criteria of FD&C §520(o)(1)(E) under the 2026 FDA CDS Final Guidance. See `context.md` §7.1. Any feature that breaks any criterion does not ship.

### 3.2 No image / IVD / signal-stream processing
No ECG, echocardiogram, imaging (CT, MRI, X-ray), waveform data, continuous pulse-ox, telemetry, or IVD signal. Inputs are text from speech. A feature requiring any of the above does not ship.

### 3.3 Physician never bypassed
No auto-diagnose, auto-treat, auto-prescribe, auto-triage. Every clinical action is the physician's. "Next best question" is informational with a visible rationale; not a directive.

### 3.4 Every recommendation reviewable
Every branch state, every LR weight applied, every citation, every generated sentence, every suggested question exposes its basis. Black-box outputs do not ship.

### 3.5 No time-critical / emergency framing
Demo and all copy frame the product as support during a clinical encounter, not triage or emergency decision-making. The 2026 FDA guidance flags time-critical framing as higher automation-bias risk.

### 3.6 No patient-facing mode
User is always an HCP.

### 3.7 Disclaimer is a product surface, not decoration
Visible in: app header, README intro, demo video opening card, demo video closing card, written submission summary (first three sentences), GitHub repo description. Not abbreviated, not hidden, not collapsed.

Verbatim:
> **Research prototype. Not a medical device.** This software is a research demonstration. It does not diagnose, treat, or recommend treatment, and is not intended for use in patient care. All patients, conversations, and data shown are synthetic or from published research benchmarks. Clinical decisions are made by the physician. This system supports the physician's reasoning by tracking claims and differential hypotheses in real time; it does not direct clinical judgement.

### 3.8 No diagnostic accuracy claims
Eval numbers are measurements of system behaviour on a specific published benchmark with documented limitations. We do not claim "diagnoses chest pain with X% accuracy." We report measured performance on LongMemEval-S / ACI-Bench with named comparators and methodology.

### 3.9 State AI-in-healthcare laws
HCP-in-the-loop posture consistent with Colorado SB24-205, Texas HB 1709 (TRAIGA), California SB 1120. No feature takes a coverage/diagnostic/treatment action without HCP review.

---

## §4. Provenance

### 4.1 Every claim has a source turn
A claim with an empty or invalid `source_turn_id` is rejected at write time.

### 4.2 Every generated note sentence has source claims
A SOAP-note sentence without non-empty `source_claim_ids` is rejected by the post-hoc validator and not displayed.

### 4.3 Every supersession has a reason
`reason ∈ {patient_correction, physician_confirm, semantic_replace}`. No nulls.

### 4.4 Every LR in the table has a citation
Every row in `predicate_packs/clinical_general/differentials/chest_pain/lr_table.json` has a `source` field pointing to a peer-reviewed or guideline citation. Approximations carry `"approximation": true`. Invented numbers are not allowed.

### 4.5 UI provenance is clickable
Transcript, claim, and note panels are bi-directionally linked. Click any note sentence → highlight source claim + source turn.

---

## §5. Determinism

### 5.1 Differential update engine is pure
Given same active claim set + same LR table → bit-identical output. No temperature, no seed, no ML re-ranking on this path. Enforced by `tests/property/test_determinism.py`.

### 5.2 Supersession Pass 1 is pure
Deterministic matcher. No LLM. No randomness.

### 5.3 LLM calls are scoped
Opus 4.7 is used only where rule-based would be worse: claim extraction, supersession semantic Pass 2 (embedding only, deterministic threshold), next-best-question phrasing, SOAP note composition. Everywhere else is deterministic code.

### 5.4 Demo runs are reproducible
`./scripts/run_demo.sh --case chest_pain_01 --seed 42` produces identical traces across runs given fixed ASR input.

---

## §6. Evals — published benchmarks only

### 6.1 No homemade metrics
No custom "substrate ablation" charts. No invented bar graphs. Every number on the demo results slide traces to a published benchmark defined by external authors.

### 6.2 The two benchmarks (revised 2026-04-21)
- **LongMemEval-S** (ICLR 2025) — memory substrate. Per-category accuracy via official evaluator. Loads `personal_assistant` pack.
- **ACI-Bench** (Nature Scientific Data 2023) — conversation → note. MEDIQA-CHAT metrics (ROUGE, BERTScore, MEDCON via 3-tier fallback). Loads `clinical_general` pack.
- **Dropped**: DDXPlus (substrate-benchmark misalignment) and MedQA (tests reader medical knowledge, not substrate) — see `reasons.md` entries.

### 6.3 Methodology honesty
Eval reports include: dataset subset used, sample size, evaluator (LLM judge vs automatic), comparator baselines, and a `LIMITATIONS.md`. No cherry-picking. If a variant loses a metric, report it.

### 6.4 No unverified comparator claims
We do not claim "outperforms Epic / Abridge / Nuance / Med-PaLM-2 / MedGemma / Ambience" without actually running a matched comparison. Report only against baselines we actually ran or against published numbers from the benchmark's own leaderboard.

---

## §7. Content and citation

### 7.1 No proprietary clinical content verbatim
Do not paste content from UpToDate, AMBOSS, DynaMed, ClinicalKey, or any paywalled source. Reference with citation + paraphrase allowed.

### 7.2 Public clinical sources for LRs, branches, rules
Primary references: 2021 AHA/ACC Chest Pain Guideline (open), HEART score publications, TIMI risk score, Wells criteria, PERC rule, published LR studies.

### 7.3 Benchmark data respect
LongMemEval-S and ACI-Bench are used per their published licenses (MIT and CC BY 4.0 respectively). Citations in `eval/README.md` and each benchmark's per-directory README.

### 7.4 No attribution hallucination
If a number, study, or guideline cannot be confirmed against a real source, it is not in the repo.

---

## §8. Scope discipline

### 8.1 Demo scope in `PRD.md` §3 is a ceiling
One chief complaint, four branches, one scripted case, one screen, three published-benchmark evals. Do not expand mid-build.

### 8.2 Vision features live in README
Multi-complaint, multi-specialty, multi-session, EHR/FHIR, full substrate (retrieval fusion, ACT-R, MMR, NLI, async worker, scope inference) — README only.

### 8.3 No new features after first video cut
Once the demo video first cut exists, no new features land. Bug fixes only.

### 8.4 If stuck >2 hours, escalate or cut
Grinding is a waste of effort. Cut the dependent feature or surface the blocker.

---

## §9. Safety and tone

### 9.1 Do not sensationalise clinical error
No dramatising patient harm for pitch. Framing is "help the physician stay present and track reasoning," not "AI catches what doctors miss."

### 9.2 No mock-ups implying real deployment
No hospital logos, no fake EHR chrome, no screenshots that could be mistaken for a live clinical system.

### 9.3 No deceptive benchmark claims
Numbers reported with methodology and size visible. A number without its denominator and caveat is not reported.

---

## §10. Review and audit

### 10.1 Compliance checklist green at submission
Run through §11 top to bottom before submitting. The checklist result goes in the submission summary's internal review notes.

### 10.2 Deviations get a decision record
Edge-case interpretations go in `docs/decisions/<date>_<rule>.md` with rationale, human approval, and mitigation.

### 10.3 Rule updates require a PR
This file is versioned. Edits via PR titled `rules: <change>` with rationale and human sign-off. Agents propose; humans approve.

---

## §11. Compliance checklist (run before submission)

- [ ] All code in repo written during the hackathon window (no pre-existing files copied in)
- [ ] `tests/licensing/test_open_source.py` passes (no closed-source dependencies)
- [ ] `tests/privacy/test_no_phi.py` passes
- [ ] `tests/property/test_determinism.py` passes
- [ ] `tests/e2e/test_demo_case.py` passes
- [ ] `SYNTHETIC_DATA.md` declares every dataset
- [ ] Disclaimer in: app header, README, video opening, video closing, written summary first three sentences
- [ ] Every LR in `predicate_packs/clinical_general/differentials/chest_pain/lr_table.json` has a `source`
- [ ] `predicate_packs/clinical_general/differentials/chest_pain/sources.md` lists every citation with URL/DOI
- [ ] Demo video processes no ECG/imaging/IVD data
- [ ] Demo video does not frame the product as time-critical / emergency
- [ ] Demo video shows no patient-facing view
- [ ] No proprietary clinical content (UpToDate / AMBOSS / etc.) in the repo
- [ ] No real patient names, DOBs, MRNs, SSNs, addresses anywhere in code, data, or video
- [ ] Eval slide reports only LongMemEval-S / ACI-Bench numbers with methodology
- [ ] No homemade / invented metrics on the eval slide
- [ ] Written submission summary opens with non-diagnostic framing
- [ ] README intro includes the full disclaimer
- [ ] FDA Non-Device CDS four-criterion posture stated in `context.md` §7 and unchanged
- [ ] Public GitHub repo exists and is readable
- [ ] Submission platform entry complete with video link, repo link, summary

Only when every box is checked does the project get submitted.
