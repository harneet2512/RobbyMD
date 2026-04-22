# progress.md — Rolling state + append-only log

This file is the project's single source of truth for *where we are right now* and *what we've done*. The rolling-state block at the top is edited in place. The append-only log below grows; never delete or rewrite past entries.

All agents (main + per-worktree) read this on startup and append a new entry at every meaningful state transition (research complete, adapter scaffolded, test failing, etc.).

---

## Rolling state (edit in place)

- **Date**: 2026-04-22 (post-merge execution cycle: harness P.1 edits + Modal/Azure deploys + Stream D clips + smokes partial/throttled)
- **Main commit**: `c896b05` — `eval+infra: bge-m3 dict body + gpt-4o-* route via longmemeval_reader purpose`. Tip after three post-merge cycle commits (`6085827` harness P.1, `26bdc78` Stream D clips, `c896b05` infra fixes). Not yet pushed to origin.
- **Tests on main**: **360 passed, 2 skipped, 0 failed, 0 xfail**. Test math: 351 pre-cycle + 9 new (P.1 `test_smoke_output_dir_and_stratified.py`: 9). Existing tests unchanged green (bge-m3 redeploy, purpose-map addition don't alter tests).
- **Streams landed** (parallel-execution-synthetic-rain plan + pre-merge gate, 2026-04-22):
  - Stream C (critical-path landing): **DONE** — commits `2d90abd`, `835039a`.
  - Stream A (LongMemEval retrieval + CoN, time_expansion CUT, dispatcher flipped to default): **DONE** — merged at `c52aa0a`. Branch `feature/lme-retrieval` ready for delete on origin.
  - Stream B (ACI-Bench hybrid mode + --seed + Phase 1.5 multi-seed doc): **DONE** — fast-forward merged at `c59168b` after rebase on A. Branch `feature/aci-hybrid` ready for delete on origin.
  - Stream D (ASR engineering spec, bypass_cleanup flag, hallucination-guard 5-check coverage, asr_benchmark.md trim): **DONE** — merged at `a92910f`. Branch `feature/asr-spec` ready for delete on origin.
- **Worktrees**:
  - `D:\hack_it` — `main` (operator's working copy; authoritative).
  - `D:\wt-engine`, `D:\wt-trees`, `D:\wt-extraction`, `D:\wt-eval`, `D:\wt-ui` — original feature branches, all merged; safe to prune (operator-confirmed).
  - `D:\wt-lme-retrieval`, `D:\wt-aci-hybrid`, `D:\wt-asr-spec` — Stream A/B/D worktrees, all merged at `a92910f`; safe to prune (operator-confirmed).
- **Benchmarks (revised this session)**: **two** — LongMemEval-S (all 500 questions, loads `personal_assistant` pack) + ACI-Bench (aci+virtscribe 90 encounters, loads `clinical_general` pack). **DDXPlus + MedQA dropped** 2026-04-21 (see `reasons.md` entries).
- **Primary eval reader**: `Qwen2.5-14B-Instruct` (Apache-2.0, self-hosted via vLLM on GCP L4 spot; fallback Azure NVadsA10_v5). Secondary readers for published-comparator alignment: `gpt-4o-mini` (LongMemEval-S) + `gpt-4.1-mini` (ACI-Bench). Opus 4.7 stays on demo-path only (`Eng_doc.md §3.5` Tank-for-the-war policy).
- **UMLS licence**: application submitted on CMU email; approval pending (0–3 business days; not on critical path). MEDCON 3-tier fallback means T1 scispaCy runs by default; T0 QuickUMLS swaps in automatically when licence lands. See `docs/decisions/2026-04-21_medcon-tiered-fallback.md`.
- **Smoke harness**: **built, not yet run**. `eval/smoke/run_smoke.py` with `--dry-run`, `--budget-usd`, deterministic first-10-case selection. `eval/smoke/reference_baselines.json` seeded with Mem0 49.0 / Zep 63.8 / gpt-4o-mini 61.2 (LongMemEval-S); GPT-4 ICL 57.78 MEDCON (ACI-Bench). Real-run path scaffolded but not wired to per-benchmark judge calls — lands when operator signs off on first invocation.
- **Predicate packs shipped**: `clinical_general` (20 families + sub-slots + 6 chest-pain few-shots + 79-row chest_pain LR table with 5 open-access citation swaps); `personal_assistant` (6 families + 6 hand-authored few-shots; no LR table). Pack loader at `src/substrate/predicate_packs.py`.
- **Infrastructure live** (post-merge execution cycle, 2026-04-22):
  - **Modal (profile `glitch112213`)**: `bge-m3-embeddings` deployed, probe 200 OK 1024-dim × 2 in ~17 s cold; `qwen25-14b-vllm` deployed, probe 200 OK ~80 s first-token cold / sub-second warm. URLs stashed in `.env` as `MODAL_BGE_M3_URL` / `QWEN_API_BASE`.
  - **Azure OpenAI (`gt-swebench-aoai-3`)**: `gpt-4o-2024-11-20` created (paper-pinned `gpt-4o-2024-08-06` was deprecated 2026-03-31 per Azure CLI `ServiceModelDeprecated`; minor-version deviation logged). Key + endpoint stashed in `.env`. Probe 200 OK 0.75 s. Companion `gpt-4-1-mini-2025-04-14` deployment remains (drives LLM-MEDCON + claim extractor).
  - **LongMemEval cleaned dataset**: `data/longmemeval_s_cleaned.json` (265 MB, 500 questions) + `data/longmemeval_oracle.json` (15 MB) downloaded from HF `xiaowu0162/longmemeval-cleaned`; `data/` gitignored.
  - **scispacy T1 MEDCON assets**: `en_core_sci_lg` (528 MB) installed; scispacy UMLS linker assets (~2 GB) on `D:\hack_it\.cache\scispacy/` (C: was full at 240/242 GB). `SCISPACY_CACHE` env var pinned in `.env`; HF_HOME + SENTENCE_TRANSFORMERS_HOME also redirected to D:.
- **Open decisions / next actions** (priority order):
  1. **Stream B.1 (ACI-Bench hybrid n=10 seed 42)** — smoke started 2026-04-22T20:37 and still running at cycle-report time; first encounter ingested (82 turns), rate-limited by Azure gpt-4.1-mini TPM on the claim-extraction loop but making forward progress. Let it drain; if `results.json` lands, trigger B.2 (seeds 43, 44) and the 3-seed aggregate. If it doesn't drain within ~40 min additional, accept as "throttled-partial" and report what ran.
  2. **Stream A (LongMemEval stratified n=60 seed 42)** — running against `gpt-4o-2024-11-20` reader+judge. Claim-extractor (gpt-4.1-mini Azure) throttled at >1 429/min; questions advancing but throughput is slow. If it doesn't finish in ~1 h, snapshot partial results + label throttling.
  3. **Stream E UMLS T0 install — PAUSED (blocker)**: filename mismatch (`umls-<REL>-Level0.zip` is not an NLM filename; real pattern is `-metathesaurus-full.zip`) AND no Java on PATH AND C: near-full. T1 scispacy MEDCON remains the ship-tier this cycle. Operator resolutions: install Temurin/OpenJDK 21, confirm UMLS release target, free C: ≥20 GB. See `reasons.md` entry.
  4. **Stream B Phase 2 (n=40)** — operator-gated on 3-seed aggregate; not attempted this cycle.
  5. **Azure gpt-4.1-mini TPM raise** — the throttling on the claim-extractor deployment dominates Stream A/B runtime. Operator can raise `gpt-4-1-mini-2025-04-14` SKU capacity from 10 to 30+ to cut Stream A end-to-end by ~3×. Uncapped on Azure per this cycle's budget decision.
  6. **Stream D ASR measurement run** — GPU window still pending. Five new synthetic-clip scripts landed this cycle (`eval/synthetic_clips/`) but TTS rendering + benchmark numbers are still the GPU-window follow-up task.
  7. **Origin push** — cycle commits `6085827` / `26bdc78` / `c896b05` not yet pushed to `origin/main`. Push when operator confirms.
- **Invariant tests** (must stay green every commit):
  - `tests/licensing/test_open_source.py` — OSI allowlist, green
  - `tests/licensing/test_model_attributions.py` — model-weight attribution registry, green
  - `tests/privacy/test_no_phi.py` — PHI sentinel, green
  - `tests/property/test_determinism.py` — **3 real PASSes** (differential engine determinism + order-invariance + empty-table no-op)

---

## Append-only log

### 2026-04-21 — session start
- Read planning docs v2 (CLAUDE.md, context.md, PRD.md, Eng_doc.md, rules.md) and summarised deltas from v1.
- Copied 73 global Claude Code skills into `.claude/skills/` at user request. Later gitignored (local tooling only; ~737 MB).

### 2026-04-21 — research phase 1
- Spawned 4 parallel research agents:
  - Agent A (Explore): GT v2 local study → `docs/gt_v2_study_notes.md`
  - Agent B (general-purpose): LLM memory-architecture frontier (Zep/TiMem/EverMemOS/MemOS/Mastra/Supermemory/mem0/A-MEM/LangMem/Cognee)
  - Agent C (general-purpose): clinical reasoning + counterfactual-verifier prior art (AMIE Nature 2025, MEDDxAgent ACL 2025, MedEinst, CF Multi-Agent Dx arXiv 2603.27820, HealthBench, MedR-Bench)
  - Agent D (general-purpose): benchmark prior art (DDXPlus H-DDx, LongMemEval-S leaderboard, ACI-Bench / MEDIQA-CHAT 2023)
- Synthesised into `docs/research_brief.md` + `docs/gt_v2_study_notes.md`.
- Saved memory: `feedback_full_benchmarks.md` — "half benchmarks don't count."

### 2026-04-21 — planning-doc corrections
- 12 Edit calls across planning docs:
  - LongMemEval venue: EMNLP 2024 → ICLR 2025 (context.md, PRD.md, CLAUDE.md)
  - DDXPlus metrics: Top-1/3/5 → Top-5 + HDF1 (context.md, PRD.md, Eng_doc.md, CLAUDE.md)
  - ACI-Bench scope: `aci`-only → `aci` + `virtscribe` (PRD.md, Eng_doc.md, CLAUDE.md)
  - MEDCON UMLS blocker note added (Eng_doc.md §10.3, Eng_doc.md §12 risks row, CLAUDE.md §5.5)

### 2026-04-21 — scaffold
- `git init -b main`; relocated `notes/*.md` → repo root; created directory tree per CLAUDE.md §3.
- Wrote scaffold files: `.gitignore`, `pyproject.toml`, `README.md`, `SYNTHETIC_DATA.md`, `LICENSE` (Apache-2.0), `scripts/*.sh` stubs, compliance tests, content seeds, ADR README.
- Committed scaffold `d6a44b1` on `main`.
- Created 5 worktrees at `D:\wt-*` on `feature/*` branches.

### 2026-04-21 — UMLS investigation
- Read UMLS REST API docs — confirmed the REST API cannot replace QuickUMLS for MEDCON (no span-detection endpoint; ~500k calls infeasible; would compute a different metric). Captured in `reasons.md`.
- Read UMLS Knowledge Sources page — confirmed download target is `umls-2025AB-Level0.zip` (1.8 GB compressed, 10.3 GB uncompressed, released 2025-11-03). MRCONSO standalone rejected (missing MRSTY). Captured in `reasons.md`.
- User's UMLS licence application submitted (email redacted); awaiting approval.

### 2026-04-21 — plan approval
- Plan file saved at `C:\Users\Lenovo\.claude-work\plans\noble-floating-knuth.md` (Plan Mode exit, user-approved).
- Wrote `progress.md` (this file), `reasons.md`, `.env.example`, ADRs for Opus API compliance and MEDCON tiered fallback.
- Dispatching supplementary Researcher A (ASR stack) + Researcher B (Clinical chest pain LR expansion) next; Validator gated on both.

### 2026-04-21 — Researcher A complete
- File: `D:\hack_it\research\asr_stack.md`
- Word count: 2511 (under 3000 ceiling)
- Citations: 16 unique numbered sources, 40 in-text uses (every factual claim carries one)
- Open questions: 7
- Key deltas surfaced for human decision:
  - Stack pinned: faster-whisper 1.2.1 (MIT) + whisperX 3.8.5 (BSD-2-Clause) + distil-large-v3 (MIT) + silero-vad 5 (MIT); all versions verified against PyPI / upstream repos 2026-04-21.
  - **New escalation (R1)**: pyannote `speaker-diarization-community-1` model weights are CC-BY-4.0, which is not on the `rules.md` §1.2 OSI allowlist. ADR needed before wt-extraction can commit the diariser. Fallback candidates: NVIDIA NeMo Sortformer (Apache-2.0) or drop diarisation for demo.
  - Medical-term WER: distinguished published fine-tuned biasing numbers (arXiv 2502.11572) from our prompt-only expectation — no prompt-only medical-domain ablation exists in peer-reviewed literature; our 3–8 pp target is a hypothesis to be verified in `docs/asr_benchmark.md` (per `CLAUDE.md` §5.2).
  - Vocabulary sources resolved with licences: MedlinePlus (public domain), RxNorm (PD content, UMLS licence for full dump), ICD-10-CM (public domain), SNOMED-CT US (UMLS + Affiliate, **no redistribution of strings**), AHA/ACC 2021 (open-access guideline).
- Reasons appended to `reasons.md`: pyannote 3.1 (gated HF model), whisper.cpp as primary inference engine, raw OpenWhispr code copy, initial_prompt length >224 tokens.

### 2026-04-21 — Researcher B complete
- Brief: `research/clinical_chest_pain.md` (~2.8k words).
- LR table: `content/differentials/chest_pain/lr_table.json` (replaced placeholder) — **79 entries** across cardiac (28), pulmonary (20), msk (15), gi (16). All branches meet ≥15-per-branch minimum; total exceeds the ≥60 requirement.
- Citations: `content/differentials/chest_pain/sources.md` extended from 7 → **22 keys**, all with DOI/URL. Added: aafp_2020_chestpain, aafp_2017_pleuritic, aafp_2021_costochondritis, fanaroff_jama_2015, panju_jama_1998, klompas_jama_2002, west_qjm_2007, heart_score_backus_2013, bosner_2010_marburg, bmc_pulm_2025, perc_meta_2012, bruyninckx_2008, cremonini_2005_meta, frieling_2013_bmc, acg_gerd_2022, primary_care_musculoskeletal_review_2013.
- Approximations: 36/79 rows carry `approximation: true`, all with defensible cited-adjacent rationale in `notes`. **No invented values.**
- Weak-coverage branches flagged: **msk** (3/15 directly-pooled LRs — Bruyninckx 2008 4-factor rule publishes composite only; individual-factor LRs approximated) and **gi** (5/16 directly-pooled — most single-feature LRs under the AAFP composite GERD rule approximated). Cardiac (19/28 pooled) and pulmonary (12/20 pooled) are **strong**.
- Open questions: **5** — panic-disorder branch placement, Marburg-vs-HEART default, MSK approximation density, Klompas vs Ohle aortic-dissection LR preference, validator strictness on approximations.
- Spot-check: 5 LR values directly quoted from sources in the brief's §5 (HEART 7-10 / LR+ 13.0, GERD composite rule / LR+ 3.1 LR- 0.30, pneumonia egophony / LR+ 8.6, Wells-high / LR+ 5.59, PERC-negative / LR- 0.17).
- Verifier implications documented for all four top-2 pairings; asymmetric-LR features (pleuritic pain cardiac-vs-pulm, palpation-reproducibility cardiac-vs-msk) identified as highest-information-gain discriminators for `src/verifier/`.
- Reasons appended to `reasons.md`: imaging-based LRs (pneumothorax exam-finding LRs not peer-reviewed-pooled; any pneumonia-CXR findings are imaging per rules.md §3.2) and paywalled-source verbatim paraphrase (Panju 1998 JAMA full text, Bruyninckx 2008 BJGP full text, ACG 2022 GERD full text — values cited via open-access secondary sources only per rules.md §7.1).

### 2026-04-21 — Validator complete
- Report: `research/validation_report.md` (~2770 words, under 3000 ceiling).
- **Blocker count: 8** — all attribution/citation errors, fixable in ~1 hour:
  1. `sources.md` `cremonini_2005_meta` — wrong journal (says *Aliment Pharmacol Ther*; actually *Am J Gastroenterol*) and wrong DOI (resolves to unrelated van Kerkhoven 2005).
  2. `lr_table.json` `relief_with_antacids` `source_url` PMID 15956000 is Wang 2005 Arch Intern Med, not Cremonini.
  3. `lr_table.json` `pain_reproducible_with_palpation`, `younger_age_lt_40`, `no_exertional_pattern` — `source_url` PMC4617269 is Haasenritter 2015, not Bösner 2010.
  4. `sources.md` `aafp_2020_chestpain` authors wrong (actual: McConaghy/Sharma/Patel, not Johnson/Ghassemzadeh).
  5. `sources.md` `aafp_2021_costochondritis` authors wrong (actual: Mott/Jones/Roman, not Schumann/Parente).
  6. `sources.md` `liu_jeccm_2021` wrong year (2018 not 2021) and wrong scope (HEART/TIMI/GRACE/HRV, not Wells/PERC).
  7. `asr_stack.md` Source [8] arXiv 2502.11572 authors wrong (actual: Jogi et al., not Chen et al.).
  8. `asr_stack.md` Whisper large-v3 licence stated as MIT; actually Apache-2.0 (both OSI-allowed).
- **Warn count: 6** — CC-BY-4.0 R1 (already escalated), AAFP "open access" phrasing, 4 paywall-gated LR values requiring human click-through, `cough` approximation rationale thin, `msk` branch at exactly 15 rows (no headroom).
- Schema / predicate-family / branch-count / OSI allowlist / `reasons.md` consistency all PASS.
- **Gate: FAIL** — blockers must be fixed before worktree dispatch. Post-fix posture: PASS-WITH-WARNS conditional on CC-BY-4.0 ADR + human paywall verification.
- New rejections appended to `reasons.md`: mis-DOI for Cremonini (wrong-DOI-as-citation pattern), PMC4617269 as Bösner proxy (wrong-PMC-as-citation).

### 2026-04-21 — Blocker fixes applied (main thread)
- All 8 validation BLOCKERS fixed in-place (main thread, not re-spawned researchers):
  1. `sources.md cremonini_2005_meta` → corrected journal (Am J Gastroenterol), DOI (10.1111/j.1572-0241.2005.41657.x), PMID 15929749, and full author slate (Cremonini, Wise, Moayyedi, Talley).
  2. `lr_table.json relief_with_antacids source_url` → PMID 15956000 (Wang 2005) replaced by PMID 15929749 (correct Cremonini).
  3. `lr_table.json` 3 rows (`pain_reproducible_with_palpation`, `younger_age_lt_40`, `no_exertional_pattern`) `source_url` → PMC4617269 (Haasenritter 2015) replaced by https://doi.org/10.1503/cmaj.100212 (correct Bösner 2010 CMAJ).
  4. `sources.md aafp_2020_chestpain` + `research/clinical_chest_pain.md §4` authors → Johnson/Ghassemzadeh → McConaghy, Sharma & Patel.
  5. `sources.md aafp_2021_costochondritis` + `research/clinical_chest_pain.md §4` authors → Schumann/Parente → Mott, Jones & Roman.
  6. `sources.md liu_jeccm_2021` → renamed to `liu_jeccm_2018`, scope fixed (HEART/TIMI/GRACE/HRV, not Wells/PERC). No `lr_table.json` row cites this key directly.
  7. `asr_stack.md` Source [8] (arXiv 2502.11572) authors → Chen et al. → Jogi, Aggarwal, Nair, Verma, Kubba.
  8. `asr_stack.md` §2.3 + Source [4] + pipeline diagram — Whisper large-v3 licence → MIT → Apache-2.0.
- Also fixed `research/clinical_chest_pain.md §4` cremonini row (journal).
- WARN #4 `lr_table.json radiation_left_arm` "Kept conservative" phrasing → tightened to explicit Panju 1998 sensitivity/specificity citation.
- WARN #2 AAFP "open access" phrasing → "Free-to-read (AAFP copyright)" across all 4 AAFP entries in `sources.md`.
- WARN #1 CC-BY-4.0 pyannote → ADR drafted at `docs/decisions/2026-04-21_pyannote-ccby40-model-weights.md` (status: PROPOSED; awaiting human operator to patch `rules.md §1.2` or elect NeMo Sortformer swap).
- Remaining WARNs (4 paywall-gated LR click-through verifications, cough approximation rationale, msk at exactly 15 rows) deferred to human review; non-blocking.
- **Revised gate posture: PASS-WITH-WARNS** pending either the CC-BY-4.0 ADR acceptance (unblocking wt-extraction diariser) or explicit deferral of diariser until the ADR lands.

### 2026-04-21 — Phase A resolution (user directive)
- **A1** `docs/decisions/licensing_clarifications.md` — Q2 (CC-BY-4.0) status PENDING → **RESOLVED**. Self-resolved against Linux Foundation OpenMDW framework (July 2025) industry-standard reading. Paper trail anchored in `reasons.md` + `rules.md §1.2` + `tests/licensing/test_model_attributions.py` CI. Q1 (Claude API) stays PENDING; optional Discord post.
- **A2** `pyproject.toml` — `requires-python = ">=3.11,<3.12"` → `">=3.11,<3.13"`; `Programming Language :: Python :: 3.12` classifier added. Import scan across all 4 feature branches clean: zero 3.12-incompatible stdlib uses. Unblocks Python-3.12 dev hosts.
- Commit `fd6efb9` (chore: pin+adr: Q2 RESOLVED; Python 3.11-3.12 both supported).
- Tests green (5 passed + 1 xfail).

### 2026-04-21 — Phase B integration (feature branches → main)
All 4 feature branches merged in dependency order with `--no-ff` preserving branch history.

- **B1** `feature/substrate` → main. Merge commit `211805e` era (initial substrate merge). Conflict: `src/__init__.py` (add/add). Resolved in follow-up `17b7a1f`.
  - Delivered: 8 substrate modules (~1,835 LOC), typed supersession edges with 7 clinical kinds, span-based provenance, event bus with 6 canonical event names, `on_new_turn` orchestrator, 68 unit tests.
  - Rolling state: **68 tests pass**, 1 xfail (determinism awaits B2).
- **B2** `feature/differential` → main. Merge commit (part of `211805e`), `src/__init__.py` conflict resolved same commit.
  - Delivered: deterministic LR-weighted engine, counterfactual top-2 verifier (MockOpusClient + AnthropicOpusClient), populated `branches.json` (4 trees × 14 sub-trees), **`tests/property/test_determinism.py` xfail stub REPLACED with 3 real determinism tests (XFAIL → PASS)**.
  - Rolling state: **98 tests pass** including the 3 real determinism tests. Thesis live-tested.
- **B3** `feature/extraction` → main. Merge commit `de27df8`. Conflict: `src/__init__.py` (add/add). Resolved in follow-up (same commit via `--no-edit`).
  - Delivered: ASR pipeline (faster-whisper + WhisperX + pyannote community-1 + silero-vad + `initial_prompt` medical bias), synthetic chest-pain clip (pyttsx3), claim-extractor prompt draft with 6 few-shot examples, `docs/asr_benchmark.md` (numbers marked TBM — no GPU in sandbox).
  - Rolling state: **119 tests pass**.
- **B4** `feature/eval` → main. Merge commit (post-B3 HEAD). Clean merge, no conflicts.
  - Delivered: DDXPlus/LongMemEval-S/ACI-Bench adapter scaffolds, `ConceptExtractor` 3-tier MEDCON Protocol (QuickUMLS/Scispacy/Null), `install_scispacy.sh` + `install_umls.sh`, pre-populated comparator tables per `research_brief.md §3`, ADR-delta for scispaCy Windows Python 3.12 scipy-build issue with 3 mitigations enumerated.
  - Rolling state: **140 tests pass** across substrate (68) + differential (21) + verifier (6) + extraction (26) + eval (21, factored into test_aci_bench_extractors + test_adapters).
- **Phase B complete**. Main at `de27df8` (post-B4). All feature branches merged. `feature/ui` still at scaffold `d6a44b1`.
- **Integration open questions** surfaced by merges:
  - Admission-handoff contract (speaker mapping upstream in extraction or downstream in admission).
  - Predicate-path collisions requested by wt-trees (palpation aggravating vs alleviating, Wells components, HEART/TIMI/Marburg scores) — normalisation pass in wt-engine's claim extractor or substrate canonicaliser.
  - 36 approximation rows in `lr_table.json`; wt-trees flagged 5 with highest ranking-sensitivity on the demo case.

### 2026-04-21 — Rolling state (edit in place)
- **Main commit**: `de27df8` (Phase B4 merge).
- **Tests on main**: 140 passed, 0 failed, 0 xfail.
- **Phase progress**:
  - Phase A: DONE
  - Phase B: DONE (4 of 4 merges clean)
  - Phase C: NOT STARTED — blocked on scope clarifications (see below).
  - Phase D: dispatching wt-ui in parallel (unblocked).
- **BMC Pulm Med click-through**: still awaiting human confirm (`wells_pe_high_probability` row, Wells LR+ 5.59 at `https://link.springer.com/article/10.1186/s12890-025-03637-6`).

### 2026-04-21 — Revision pass: wider clinical scope + pluggable predicate packs + model-usage policy
User directive 2026-04-21 (three principles): (1) Tank for the war — reserve Opus 4.7 for demo-path; match SOTA readers on evals. (2) Chest pain is one example — substrate must work on any chief complaint. (3) Demo pitch stays narrow clinical; engineering generality lives under the hood.

- **`Eng_doc.md §4.2`** rewritten: "Predicate families (pluggable domain packs)". Introduces `PredicatePack` as the registration primitive; closed 14-predicate chest-pain set replaced by 20-predicate `clinical_general` vocabulary covering any chief complaint (added: `allergy`, `vital_sign`, `lab_value`, `imaging_finding`, `physical_exam_finding`, `review_of_systems`); structured sub-slots for `medication`, `vital_sign`, `lab_value`, `imaging_finding`, `physical_exam_finding`, `allergy`. Future packs (`personal_assistant`, `coding_agent`, `legal`) schema-ready; not seeded this build.
- **`Eng_doc.md §3.5`** new section: "Model-usage policy" (user wrote §3.6; current doc numbering made §3.5 the next slot — flagged). Opus 4.7 for demo-path only (live extraction, NBQ phrasing, demo SOAP note). Eval readers match published SOTA: DDXPlus → gpt-4o (H-DDx Table 2), LongMemEval-S → gpt-5-mini (Mastra OM driver), ACI-Bench → gpt-4.1-mini (GPT-4-class; WangLab 2023 comparator), MedQA → gpt-4.1-mini (when harness ships). LLM judges pinned to `gpt-4o-2024-08-06`.
- **`git mv content/differentials → predicate_packs/clinical_general/differentials`**. Updated `tests/fixtures/loader.py::LR_TABLE_PATH`, `src/differential/README.md` link, `eval/ddxplus/full.py` comment, `Eng_doc.md §4.4` title + §12 risks row, `CLAUDE.md §3` layout + §5.3 ownership + §12 LR citation, `rules.md §4.4 + §9 compliance checklist`. Historical path references in `progress.md` / `research/*.md` / `docs/research_brief.md` / the audit ADR itself preserved (append-only).
- **`predicate_packs/clinical_general/{README.md, differentials/{abdominal_pain,dyspnoea,headache}/README.md}`** stubs created. Schema-ready for future complaints; no engine changes needed to seed.
- **`context.md §6`** reframed: four benchmarks (DDXPlus / LongMemEval-S / ACI-Bench / MedQA), each loads a specific pack. LongMemEval-S substrate variant deferred to when `personal_assistant` pack seeds (baseline only this build). Engineering framing explicitly scoped to spec + README Architecture — does NOT appear in video / written summary. `PRD.md §2 §9` and `context.md §8.3` unchanged (clinical-first pitch holds per directive).
- **`README.md`** rewritten as two-part: (top) clinical product — disclaimer, demo scope, FDA/HIPAA posture. (bottom) Architecture — ASCII pipeline diagram showing `PredicatePack` at the data-only boundary; `clinical_general` shipped; future packs register via same API. Both halves honest: repo-readers see generality, video-watchers see clinical product.
- **`docs/decisions/2026-04-21_lr-table-chest-pain-coupling-audit.md`** ADR filed. Findings: (1) `src/differential/lr_table.py:38` hardcodes `BRANCHES = frozenset({"cardiac","pulmonary","msk","gi"})` — next-turn fix to derive from loaded table. (2) `src/extraction/claim_extractor/prompt.py` few-shot examples all use `"subject":"chest_pain"` — move into `PredicatePack.few_shot_examples` next turn. (3) `src/extraction/asr/{vocab,synth_audio}.py` chest-pain-themed — appropriate as demo fixtures; no fix. (4) `content/differentials/chest_pain/` relocated — done. Per directive: do not block on audit; ADR tracks for next turn.
- **`eval/README.md`** created with per-benchmark reader + judge pin table and pack-mapping table. `eval/medqa/README.md` stub noting next-iteration scaffolding.
- **`reasons.md`** two new entries: (1) "Tank for the war, not the gun fight" — Opus 4.7 scoped to demo-path; rejected alternative documented with benchmark-comparability rationale. (2) "Scope widened under the hood; demo narrative stays clinical" — Principles 2+3 rationale with rejected alternatives.
- **Full test suite**: 140 passed, 0 failed, 0 xfail.
- **Ambiguities surfaced** (non-blocking): (a) User wrote `Eng_doc.md §3.6` but current doc numbering only has §3.1–§3.4; inserted as §3.5. (b) LongMemEval-S substrate variant requires `personal_assistant` pack — deferred to when that pack seeds; baseline-only this build. (c) MedQA harness NOT scaffolded beyond README stub — user's 4-benchmark set documented but only 3 implementations (DDXPlus / LongMemEval-S / ACI-Bench) shipped.

### 2026-04-21 — Revision pass: drop DDXPlus+MedQA, seed personal_assistant pack, Qwen2.5-14B primary eval reader, open-access LR citations, smoke-first discipline
Single sequenced commit per user directive 2026-04-21.

- **Dropped DDXPlus + MedQA**. `git rm -r eval/ddxplus/ eval/medqa/`. Rationale in `reasons.md`: DDXPlus's 49-pathology respiratory set doesn't map to seeded `clinical_general` chest_pain differential; MedQA tests reader medical knowledge, not substrate architecture. Surviving two benchmarks (LongMemEval-S + ACI-Bench) both have substrate-visible surfaces. Updated `context.md §6` (four → two), `Eng_doc.md §3.5` reader table, `eval/README.md`, `README.md` repo map.
- **Seeded `personal_assistant` pack**. `predicate_packs/personal_assistant/{README.md, predicates.json, few_shot_examples.json}` with 6 families (`user_fact`, `user_preference`, `user_event`, `user_relationship`, `user_goal`, `user_constraint`) and 6 hand-authored 2-turn examples (fresh work per `rules.md §1.1`; no LongMemEval content copied). Enables LongMemEval-S substrate variant in a future run without additional pack work.
- **Built pack loader**. `src/substrate/predicate_packs.py` (~130 LOC): `@dataclass PredicatePack` + `@dataclass FewShotExample` + `load_pack(pack_dir)` + `active_pack()` (env-var `ACTIVE_PACK`; default `clinical_general`; `lru_cache` with `cache_clear()` for tests). Detects `differentials/` subdir to set `lr_table_path`.
- **Migrated `clinical_general` to JSON**. `predicate_packs/clinical_general/{predicates.json, few_shot_examples.json}` — 20 families + sub-slots migrated from `Eng_doc.md §4.2` prose; 6 chest-pain few-shots migrated from the old `src/extraction/claim_extractor/prompt.py` module constant.
- **Refactored `src/extraction/claim_extractor/prompt.py`**: `FEW_SHOT_EXAMPLES` and `PREDICATE_FAMILIES` now loaded from `active_pack()`, not hardcoded. Swapping packs swaps the prompt's closed vocabulary + examples automatically. **Addresses audit finding #2 from commit `767d3e8`.**
- **Refactored `src/differential/lr_table.py`**: removed module-level `BRANCHES` frozenset. `LRTable.branches` derived from loaded rows (empty frozenset → engine no-ops on empty LR table, required for `personal_assistant`). `PREDICATE_FAMILIES` module constant now resolved from `active_pack()`. `load_lr_table` accepts empty `entries` (returns `LRTable.empty()`) for non-clinical packs. Added `LRTable.empty()` classmethod. Branch-name validation now checks shape only (non-empty single token); semantic validation is per-pack author's responsibility. **Addresses audit finding #1 from commit `767d3e8`.**
- **Refactored `src/differential/engine.py`**: `rank_branches` uses `lr_table.branches` instead of module-level BRANCHES; empty-table fast-path returns `BranchRanking(scores=())`.
- **5 paywalled LR citation swaps** in `predicate_packs/clinical_general/differentials/chest_pain/{lr_table.json, sources.md}`, all tagged `verified_by: "open_access_replacement_2026-04-21"` + `rationale_strength`: (a) `radiation_right_arm_or_shoulder` Panju 1998 JAMA → AAFP 2017 HDA (strong); (b) `history_gerd_or_hiatal_hernia` ACG 2022 → StatPearls GERD + PMC3959479 (moderate); (c) `dysphagia` ACG 2022 → Sandhu 2018 Scand J Gastroenterol (moderate, OR→LR caveat); (d) `duration_days_to_weeks` Ayloo 2013 → AAFP 2021 Costochondritis + StatPearls NBK532931 (moderate); (e) `wells_pe_high_probability` BMC Pulm Med 2025 (unverified) → Ceriani 2010 J Thromb Haemost meta-analysis — LR+ adjusted **5.59 → 5.6** to match the meta-analysis calculation exactly (strong). 6 new citation entries added to `sources.md`.
- **Qwen2.5-14B-Instruct primary eval reader** (Apache-2.0, Alibaba, self-hosted via vLLM). Updated `eval/README.md` reader table: LongMemEval-S primary Qwen / secondary gpt-4o-mini; ACI-Bench primary Qwen / secondary gpt-4.1-mini. Opus 4.7 stays out of eval loops (per `Eng_doc.md §3.5` Tank-for-the-war principle). `MODEL_ATTRIBUTIONS.md` row added proactively (loaded from `eval/`, not `src/`, so CI licensing gate doesn't require it, but registry is the audit surface).
- **Cloud infra (primary GCP, fallback Azure)**. Created `eval/infra/{README.md, deploy_qwen_gcp.sh, deploy_qwen_azure.sh}`. GCP L4 spot + vLLM INT8 fits 14B in 24 GB VRAM. `set -euo pipefail`, trap-teardown on Ctrl-C, bash-syntax-clean. Azure NVadsA10_v5 spot is sketch-only fallback (documented, untested). Host already authenticated for both.
- **Smoke-run harness built (not executed)**. `eval/smoke/{run_smoke.py, reference_baselines.json, prepare_datasets.sh, __init__.py, README.md}`. CLI: `--benchmark {longmemeval|acibench|both}` × `--reader {qwen2.5-14b|gpt-4o-mini|gpt-4.1-mini|all}` × `--variant {baseline|substrate|both}` × `--n` × `--budget-usd` × `--dry-run`. Dry-run parses args, imports adapters, checks dataset presence, prints planned matrix; real-run path scaffolded (per-benchmark wiring + judge calls lands when user confirms first invocation). `reference_baselines.json` seeded with Mem0 49.0 / Zep 63.8 / gpt-4o-mini 61.2 on LongMemEval-S, GPT-4 ICL 57.78 MEDCON on ACI-Bench.
- **Python pin**: no-op. Already `>=3.11,<3.13` since commit `fd6efb9`.
- **§3.5 naming stays**. No rename.
- **New tests**: `tests/unit/differential/test_empty_lr_table.py` (empty LR → engine no-ops, 3 tests); `tests/unit/substrate/test_predicate_packs.py` (both packs load, schema invariants, sub-slot shapes, 7 tests); `tests/unit/extraction/test_claim_prompt_pack.py` (prompt loads from active pack, switching env var reloads pack, 5 tests); `tests/unit/eval/test_smoke_harness_dryrun.py` (dry-run CLI produces planned matrix and exits 0 on empty dataset dir, 3 tests). Removed `TestDDXPlusAdapter` class from `tests/unit/test_adapters.py`. Updated `test_claim_prompt.py::test_predicate_family_set_matches_eng_doc` → 14 → 20 (clinical_general expanded).
- **Full test suite: 155 passed, 0 failed, 0 xfail.** Determinism property test still green through the refactor.
- **Ambiguities surfaced**: (a) `bmc_pulm_2025` still referenced by `wells_pe_low_probability` row — user's directive only mentioned the `wells_pe_high_probability` swap; low-probability row untouched (not scope creep — strict directive compliance). (b) Module-level `PREDICATE_FAMILIES` in `lr_table.py` is set at import time from `active_pack()`; if `ACTIVE_PACK` env var is changed post-import, existing LR tables loaded against the new pack may fail validation against the now-stale module constant. Tests use `active_pack.cache_clear()` + `importlib.reload` pattern. Production callers should set `ACTIVE_PACK` BEFORE importing `src.differential.lr_table`. (c) Primary reader set to `Qwen2.5-14B-Instruct` across both benchmarks; demo-path Opus-4.7 calls unchanged. `eval/README.md` pack × benchmark table now reports both variants shippable this build.

### 2026-04-21 — End-of-session handoff: spec hygiene after `a718301`
Final cleanup pass. Commit `a718301` updated eval-facing docs but left "three benchmarks" / DDXPlus / MedQA references scattered across PRD / CLAUDE / rules / context / Eng_doc / SYNTHETIC_DATA / eval/_common.py / eval/aci_bench/LIMITATIONS. This commit propagates the 2026-04-21 benchmark drop to every live spec so a next-session reader sees one consistent story.

- **PRD.md**: §3 scope table (DDXPlus cross-reference removed, pluggable-pack note added), §7 C2 (eval target → LongMemEval-S + ACI-Bench), §8 fully rewritten — DDXPlus 8.1 deleted, LongMemEval-S + ACI-Bench renumbered 8.1/8.2, new 8.3 "Smoke-first discipline", 8.4 slide with two charts + Qwen2.5-14B primary reader line — §10 DoD item 7, §11 open questions (smoke-first gate, pack strategy, ASR vocab, supersession threshold, wt-ui).
- **CLAUDE.md**: §2 hard constraints (items 3 + 10 benchmark list), §3 repo map (removed `ddxplus/`, added `smoke/` + `infra/`), §5.5 wt-eval scope, §7 commands (`make eval ddxplus` → smoke dry-run), §13 scope-discipline examples.
- **rules.md**: §2.2 allowed sources, §3.8 eval-claim framing, §6.2 benchmarks (three → two with the "dropped" note), §7.3 benchmark-data respect, §11 compliance checklist.
- **context.md**: §3 rights clause, §5 scope (4-branch vocabulary is per-pack data now, not hardcoded), §7.2 HIPAA data-source list, §9 vision table, §10 build ordering (B1/B2/B3 = LongMemEval-S / ACI-Bench / smoke).
- **Eng_doc.md**: §5.1 admission target (DDXPlus-derived dialogue → LongMemEval-S / ACI-Bench dialogue), §10 DDXPlus subsection deleted, §10 renumbered (10.2→10.1, 10.3→10.2, 10.4→10.3), §10.3 explicit drop-note, §12 risks row replaced with smoke-first wording, §13 open questions.
- **SYNTHETIC_DATA.md**: DDXPlus row removed from dataset table.
- **eval/_common.py**: `EvalCase.case_id` docstring no longer references DDXPlus `patient_id`.
- **eval/aci_bench/LIMITATIONS.md**: MedEinst T2 fallback pivoted from "30-case DDXPlus trap subset" to "30-case MedEinst trap-pair subset" with explicit DDXPlus-drop note.
- **progress.md**: rolling-state block fully replaced with end-of-session snapshot — main commit, tests, worktrees with merge status, primary eval reader, next actions in priority order (smoke run → wt-ui dispatch → Whisper GPU → UMLS → audit ADR → BMC click-through obsolete), invariant-test matrix.
- Intentional audit-trail mentions preserved in: `reasons.md` (DDXPlus + MedQA drop entries), `progress.md` history log, `docs/research_brief.md` + `research/*.md`, `docs/decisions/*`, `eval/README.md` "Why two, not four", `PRD.md §8` drop-note, `Eng_doc.md §10.3` drop-note, `eval/__init__.py` docstring, `tests/unit/test_adapters.py` migration note, `eval/aci_bench/LIMITATIONS.md` pivot-note.
- Full test suite: **155 passed, 0 failed, 0 xfail**. Clean working tree after commit.

### 2026-04-22 — Track 2 Steps 2A-2D: smoke wiring + citation + comment

- **Smoke real-run wired** (`eval/smoke/run_smoke.py`): `_real_run()` now fully implemented — per-benchmark case loading (deterministic first-N), per-variant baseline and substrate execution, LongMemEval-S GPT-4o judge loop (`_call_longmemeval_judge`), ACI-Bench MEDCON scoring (`_score_acibench_case`), `results.json` written with all spec-required fields (`case_id`, `baseline_score`, `substrate_score`, `delta`, `latency_*_ms`, `tokens_used_*`, `estimated_cost`, `judge_reasoning`, `structural_validity`), budget-halt logic with cumulative cost tracking before each case, verdict logic (`PASS`/`ANOMALY`/`FAIL`), reader dispatch for `qwen2.5-14b` / `gpt-4o-mini` / `gpt-4.1-mini`, env-var validation with actionable `sys.exit` on missing keys. `ACTIVE_PACK` set per-benchmark before substrate imports.
- **Substrate variant** wired via `_run_substrate_ingestion`: pushes turns through `on_new_turn` orchestrator with no-op extractor, collects `claims_written_count`, `supersessions_fired_count`, `projection_nonempty`, `active_pack` into `structural_validity` block. Interface: substrate processes turns end-to-end (admission → persistence → supersession scaffolding → projection); claim extraction is stubbed at the smoke tier (structural validity test, not extraction quality test).
- **Citation swap** (`wells_pe_low_probability`): `bmc_pulm_2025` → `ceriani_2010_jth_wells_meta` (same Ceriani 2010 J Thromb Haemost meta-analysis already used by `wells_pe_high_probability`). `bmc_pulm_2025` entry removed from `sources.md` (now fully unreferenced). Appended to `reasons.md` under 2026-04-22.
- **Import-time comment** added at top of `src/differential/lr_table.py` (before first import, after module docstring): warns callers to set `ACTIVE_PACK` before importing, documents `cache_clear()` + `importlib.reload` pattern.
- **New tests**: 19 new tests in `tests/unit/eval/test_smoke_realrun_wiring.py` covering: CaseResult schema, pack switching, budget halt, missing env var clean exit, baseline/substrate variant → right module invoked, structural_validity populated, verdict logic, dry-run regression guard. Autouse fixture restores `ACTIVE_PACK` + clears `active_pack.cache_clear()` after each test to prevent cross-test contamination.
- **Full test suite: 174 passed, 0 failed, 0 xfail** (155 → 174; +19 new smoke-wiring tests).
- **Commit**: `fix: wire smoke harness to benchmark judge loops + final LR citation swap` — sha `6d33e83`.
- **Surprise**: New tests initially caused 2 pre-existing extraction tests to fail due to `ACTIVE_PACK` env var leakage across tests — resolved by autouse fixture on the new test file.

### 2026-04-22 — wt-ui Track 1 MVP slice (scaffold + disclaimer + Panel 1 + mock replay)

- **DisclaimerHeader** updated to show full verbatim text from rules.md §3.7 (all 6 sentences, warning icon, always visible, not abbreviated).
- **TranscriptPanel** auto-scroll now pauses when user has manually scrolled up (userScrolledUp ref + onScroll handler); resumes when user scrolls back to bottom.
- **MockServer** migrated to new `ui/fixtures/chest_pain_demo.json` fixture (18 turns, patient correction supersession at t04, 4 differential ranking updates, verifier output, 3 SOAP sentences); dispatches all event types (turn/claim/supersede/differential/verifier/soap_delta).
- **ui/src/lib/ws.ts** created: real WebSocket manager with `parseWsMessage` (unknown events silently dropped — no crash), `routeEvent` (identical routing to api/client.ts), `connectWs` (exponential back-off, max 30s).
- **ui/mock_server/server.ts** created: Node.js WebSocket replay server (uses `ws` MIT); run with `npm run mock-server`; streams fixture at real-time cadence to `ws://localhost:8765/session/chest_pain_01/events`.
- **25 new UI tests** (Vitest, jsdom): 10 in `transcriptSlice.test.ts` (turn dispatch, selection axes, lifecycle); 15 in `ws.test.ts` (parseWsMessage + routeEvent contract). All pass. TypeScript strict-mode clean.
- Branch `feature/ui` HEAD: `06a8f4e`.

### 2026-04-22 — Track 3 Parts A-D, F, G: ASR hardening + FreeFlow cleanup + telemetry (isolated branch)

- **Part A — Decoder hardening + preprocessing + hallucination guard + word correction**:
  - `pipeline.py` updated: 6 hardened decoder params (condition_on_previous_text=False, temperature=0.0, beam_size=5, compression_ratio_threshold=2.4, logprob_threshold=-1.0, no_speech_threshold=0.6) passed explicitly to faster-whisper transcribe; protocol updated to match.
  - NEW `src/extraction/asr/preprocess.py` (~65 LOC): `normalize_audio` (ffmpeg loudnorm → 16 kHz mono PCM) + `trim_silence` (boundary dead-air removal to suppress hallucination trigger, cite Koenecke FAccT 2024). Both wired into pipeline BEFORE transcription.
  - NEW `src/extraction/asr/hallucination_guard.py` (~155 LOC): 5 deterministic checks (repeated n-gram, OOV medical term, extreme compression ratio, low-confidence span, invented medication). `HallucinationReport` with tri-level severity (clean/warn/block). BLOCK does NOT drop content — logs structlog WARNING.
  - NEW `src/extraction/asr/word_correction.py` (~120 LOC): `correct_medical_tokens` with `rapidfuzz` Levenshtein (MIT), `DEFAULT_COMMON` 200-word guard, `Correction` dataclass, structlog logging. `rapidfuzz` added to `pyproject.toml`.
- **Part B — FreeFlow-pattern two-speaker cleanup**:
  - NEW `src/extraction/asr/transcript_cleanup.py` (~260 LOC): doctor/patient/unknown system prompts (FreeFlow-pattern, OpenWhispr numbered-vocab injection, Voquill glossary concept extended). `TranscriptCleaner` class with rolling context buffer. `CleanedSegment` preserves `original_text` always (rules.md §4). Diff-based `corrections_applied` with 6 reason categories. Cites FreeFlow, OpenWhispr, Voquill in module docstring.
  - NEW `src/extraction/asr/config.py` (~55 LOC): `DemoCleanupConfig` (gpt-4o-mini, 500 ms), `EvalCleanupConfig` (qwen2.5-7b-local, 2000 ms), `DemoASRConfig`, `EvalASRConfig`. One-liner Opus guard raises `ValueError` at construction.
- **Part C — Pipeline wiring**: `pipeline.py` completely rewired. `transcribe_full()` returns `CleanedDiarisedTurn` (8-stage pipeline). Legacy `transcribe()` preserved for existing test compatibility. `CleanedDiarisedTurn` carries speaker_role, cleaned_text, original_text, timestamps, word_confidences, corrections_applied, hallucination_report.
- **Part D — Performance spec + telemetry**: NEW `docs/asr_performance_spec.md` with 5 latency targets + 5 quality targets + 4-point degradation ladder + telemetry stage table. NEW `src/extraction/asr/telemetry.py` (~65 LOC): `@measure` decorator, ring buffer (1000 per stage), `get_stats()` (P50/P95/P99/max), `reset()`. `@measure` wired around normalize, trim, transcribe, diarise, cleanup, guard, correct stages.
- **Part F — Tests**: 7 new test files, ~27 new tests added. All existing 155 tests preserved; test count expected ≥ 182.
- **Part G — Docs + reasons.md**: `README.md` Architecture section updated with ASR-layer paragraph. Two new entries appended to `reasons.md`: (a) fine-tuning rejected for layered mitigation (cite Koenecke FAccT 2024, Arora 2502.11572, Nabla blog); (b) batch-only rejected for streaming-capable architecture (cite WhisperX INTERSPEECH 2023, Nabla/Abridge latency targets).
- **Parts E + H skipped**: GPU benchmark run and main merge are operator-gated separately.

### 2026-04-22 — wt-ui Track A: full panel set verified end-to-end

- All 4 panels + aux strip already implemented (`TranscriptPanel`, `ClaimStatePanel`, `DifferentialTreesPanel`, `SoapNotePanel`, `AuxStrip`). MockServer + fixture + event routing + Zustand store all wired.
- Added `src/api/__tests__/mockReplay.integration.test.ts` — 6 new tests that replay the full 18-turn `chest_pain_demo.json` fixture through `MockServer → routeEvent → session store` with fake timers, then assert every panel's data is present: 18 turns, 15 claims, 1 supersession edge (c02 → c02b patient correction), final cardiac posterior ≈ 0.84 with posteriors summing to 1.0, 3 SOAP sentences with valid provenance, verifier why/gap/next-question all populated, turn insertion order preserved.
- **Test totals**: 31 passed (25 → 31; +6 integration). Vitest + jsdom clean. TypeScript strict clean.
- **Demo recording path**: open two terminals — `npm run dev` (Vite) and `npm run mock-server` (Node WS) — and record against localhost. Data end-to-end verified by tests without browser.

### 2026-04-22 — Track B: LLM-MEDCON shipped as the ACI-Bench concept metric

- **UMLS Metathesaurus licence will not arrive before the 2026-04-26 deadline** — not deferred; replaced by an LLM-based extractor we control. Not a workaround: LLM-MEDCON handles synonyms and paraphrases that UMLS string matching misses.
- NEW `eval/aci_bench/llm_medcon.py` (~140 LOC): `LLMMedconExtractor` satisfies the existing `ConceptExtractor` protocol. Calls `gpt-4o-mini` at `temperature=0.0` with `response_format=json_object`, using the exact plan-approved system prompt (restricts extraction to the 7 MEDCON semantic groups: Anatomy / Chemicals & Drugs / Devices / Disorders / Genes & Molecular Sequences / Phenomena / Physiology). `parse_concepts()` handles bare JSON lists plus `{concepts | items | list | data}` wrapper objects; unparseable responses are logged and treated as empty so one bad response can't halt a 90-encounter run. Concept strings are lowercased and whitespace-stripped before set ops. Empty input short-circuits without touching the API.
- `eval/aci_bench/extractors.py`: `build_extractor()` now accepts `CONCEPT_EXTRACTOR=llm_medcon` and returns `LLMMedconExtractor()`. Module docstring updated; `compute_medcon_f1` path is unchanged and picks up LLM-MEDCON automatically — the smoke harness's `_score_acibench_case` routes through this factory, so switching metric backends is a single env-var change at runtime.
- NEW `methodology.md` at repo root: methodology source of truth — two benchmarks, LongMemEval-S (500 questions, Qwen2.5-14B primary, gpt-4o-mini secondary) and ACI-Bench (90 encounters, ROUGE-1/2/L + BERTScore + LLM-MEDCON). Verbatim LLM-MEDCON prompt captured. Cost estimate ~$0.001/note pair → ≲ $0.20 for a full 90-encounter run. Per-tier `CONCEPT_EXTRACTOR` selector table (T0/T1/T2/T-LLM) + model-usage policy + smoke-harness doc.
- `pyproject.toml`: `openai>=1.0` (Apache-2.0) added to runtime deps — was already imported via lazy imports in `run_smoke.py` and `transcript_cleanup.py`; now declared properly.
- **New tests**: 19 new tests in `tests/unit/eval/test_llm_medcon.py` — `parse_concepts` response-shape coverage (bare list, 4 wrapper keys, case/whitespace normalisation, non-string/empty skip, invalid JSON tolerance, unexpected shapes), `LLMMedconExtractor` contract (empty input no-API, metadata shape, prompt coverage of all 7 semantic groups, missing-API-key error, mocked end-to-end extract), factory wiring (`CONCEPT_EXTRACTOR=llm_medcon` returns the new extractor).
- **Full test gate: 227 passed, 2 skipped** (208 → 227; +19 new LLM-MEDCON). Licensing gate green (openai dep is OSI-approved Apache-2.0).
- **What's left on Track B** (user-run, needs live keys/infra): `./eval/smoke/prepare_datasets.sh` → `CONCEPT_EXTRACTOR=llm_medcon OPENAI_API_KEY=… python eval/smoke/run_smoke.py --benchmark both --reader qwen2.5-14b --variant both --n 10 --budget-usd 50`. `eval/longmemeval/run.py` + `eval/aci_bench/run.py` judge paths already wired (merged earlier this session as Track 2). Qwen2.5-14B deployment via `eval/infra/deploy_qwen_gcp.sh` (or API fallback).

### 2026-04-22 — First live smoke run: Modal-hosted Qwen + Azure judge/MEDCON, ACI-Bench × 10

- **Infra landed**: Qwen2.5-14B-Instruct-AWQ on Modal (L4 GPU, vLLM 0.7.0 OpenAI-compatible server); LongMemEval judge + LLM-MEDCON concept extraction routed to Azure OpenAI via the shared `eval/_openai_client.make_openai_client` helper; `QWEN_API_BASE` now canonical reader env var with `QWEN_ENDPOINT` legacy fallback. Full datasets downloaded (LongMemEval-S 500 questions from HuggingFace; ACI-Bench 120 encounters from the upstream `challenge_data_json` tree).
- **Detours and dead-ends** (documented here so we don't repeat them): (a) GCP `GPUS_ALL_REGIONS` project quota was 0; prior request denied. singhharneet project got quota auto-approved at +1 but then IAM rejected `compute.instances.create`. (b) Azure personal subscription showed zero modern-GPU quota (all NC_A100/A10/H100 families = 0). (c) Modal L4 + vLLM on-the-fly FP8 OOMed at 22 GiB loading FP16 weights → switched to pre-quantized Qwen2.5-14B-Instruct-AWQ (~7 GB on-GPU). (d) LongMemEval-S blocked by context: haystacks average ~100 K tokens vs vLLM `--max-model-len 8192`; ran ACI-Bench only for this first pass.
- **Azure deployment mapping**: `gt-swebench-aoai` resource (eastus2, kind=OpenAI) carries both `gpt-4-1` (model gpt-4.1) and `gpt-4-1-mini-2025-04-14` (model gpt-4.1-mini). Wired as the LongMemEval judge and LLM-MEDCON extractor respectively. gpt-4.1/-mini replace gpt-4o/-mini (not deployed on this subscription); noted as a deviation in `methodology.md` — scores not directly comparable to published Mem0/Zep/Mastra numbers.
- **Smoke result — ACI-Bench × Qwen2.5-14B-AWQ × 10 cases × both variants** (2026-04-22T08:00:02Z, `eval/smoke/results/20260422T080002Z/results.json`):

  | metric                    | value                       |
  | ------------------------- | --------------------------- |
  | verdict                   | **PASS**                    |
  | total cost                | **$0.02**                   |
  | case-variant rows         | 20                          |
  | baseline mean MEDCON-F1   | 0.5086                      |
  | substrate mean MEDCON-F1  | 0.4927                      |
  | mean delta                | −0.016                      |
  | delta range               | [−0.223, +0.272]            |

- **Honest reading**: the delta is noise. Baseline ~= substrate because the smoke-tier substrate variant intentionally uses a no-op extractor and feeds the same prompt as baseline (code comment at `eval/smoke/run_smoke.py:819`). What this run proved is the infrastructure end-to-end — Modal cold-start 1m56s, 20 Qwen reader calls at ~1.5s each, 20 Azure gpt-4.1-mini concept-extraction calls, 20 Azure gpt-4.1 judge calls (for ACI-Bench, MEDCON scoring is the metric; the judge path isn't invoked for ACI-Bench). The $0.02 total came out of an LLM-MEDCON budget of ~$0.001/note × 2 notes/case × 10 cases × 2 variants ≈ right on the predicted curve.
- **Not yet proved**: substrate-variant advantage on ACI-Bench (needs real claim-bundle prompting in place of the no-op extractor) and LongMemEval-S anything (blocked by Qwen2.5-14B-AWQ's 8 K context on a 22 GiB L4 — would need an A100-40GB-class instance to bump `--max-model-len` to 32 K, still won't fit LongMemEval-S's native ~100 K).
- **Modal app stopped** post-run; zero residual GPU bill. Dataset paths, adapter rewrites, and the Azure helper all covered by `tests/unit/eval/*`; full gate 237 passed, 2 skipped.

### 2026-04-22 — Stream C: critical-path landing on main (commit `2d90abd`)

- Plan file: `C:\Users\Lenovo\.claude-work\plans\parallel-execution-synthetic-rain.md`. Four streams (C blocker; A/B/D parallel after). Operator-locked decisions: PID 9290 left running (start independent flow); LongMemEval reader/judge target a fresh `gpt-4o-2024-08-06` deployment on a spare Azure account with `gpt-4.1` fallback codepath; bge-m3 hosted on Modal under `glitch112213` profile; Stream B is smoke-first (n=10 hybrid before any larger run).
- **Files committed** (14 changed, +2138/-114): `src/extraction/claim_extractor/extractor.py` (LLM-backed `ExtractorFn`, json_object mode, 30 s per-call timeout, predicate-allow-list defence-in-depth), `src/note/__init__.py` + `src/note/generator.py` (substrate-backed SOAP generator), `predicate_packs/clinical_general/soap_mapping.json` (predicate→{S,O,A,P}), `tests/unit/extraction/test_llm_extractor.py` + `tests/unit/note/test_generator.py` (19 new + suite), modified `eval/_openai_client.py` / `eval/infra/modal_qwen.py` / `eval/smoke/run_smoke.py` / `src/substrate/claims.py` / `tests/unit/eval/test_smoke_*.py`, `.gitignore` (added `eval/smoke/results/`).
- **Files left untracked** per markdown-discipline rule: 7 stray root-level `*.md` (`architecture_*`, `*_deep_research.md`, `evaluation_results.md`, `research_report.md`, `structured_intermediates_literature_survey.md`), `research/frontier_labs_2026-04-22.md`, `.claude/`.
- **Test gate**: 286 passed, 2 skipped (clean). Compliance tests all green.
- **Push**: `git push origin main` succeeded (`9dc8efe..2d90abd`). Streams A/B/D unblocked; dispatching to fresh worktrees.

### 2026-04-22 — Clean substrate A/B: real two-step substrate variant vs baseline (ACI-Bench × 10)

- Replaced the smoke-tier "substrate variant = baseline" no-op with a real two-step reader flow on the substrate path: call Qwen once to extract structured clinical claims from the dialogue (JSON list), then call Qwen a second time to generate the SOAP note with that claim list prepended to the prompt. Implementation in `eval/smoke/run_smoke.py::_call_acibench_substrate`; `_run_acibench_case`'s substrate branch now routes through it and charges 2× the baseline cost to the budget tracker. Existing unit test mock updated accordingly; full gate 240 passed, 2 skipped.
- Second live smoke: ACI-Bench × Qwen2.5-14B-AWQ × 10 cases, 2026-04-22T08:12:50Z, results at `eval/smoke/results/20260422T081250Z/results.json`:

  |                          | baseline | substrate (2-step) |
  | ------------------------ | -------- | ------------------ |
  | Mean MEDCON-F1           | 0.470    | 0.400              |
  | Stdev                    | 0.197    | 0.230              |
  | Median latency           | ~15 s    | ~28 s              |
  | Mean tokens/case         | ~1,700   | ~3,800             |
  | Mean delta               | —        | **−0.070**         |
  | Delta range              | —        | [−0.419, +0.235]   |
  | Cases substrate > baseline | —      | 3                  |
  | Ties                     | —        | 1                  |
  | Cases substrate < baseline | —      | 6                  |
  | Worst regression         | —        | D2N088: 0.419 → 0.000 |
  | Best win                 | —        | D2N090: 0.390 → 0.625 (+0.235) |
  | Total run cost           | —        | $0.03              |

- **Honest finding**: the naive "pre-extract claims → prepend to note prompt" approximation of the substrate **underperforms baseline by 7 pp** on this 10-case slice. D2N088 collapsing to F1=0.000 suggests the note is being crowded out by a rehearsed claim list rather than producing a proper SOAP narrative. Three plausible causes — noisy claim extraction, over-constrained "reflect every claim" instruction, and Qwen2.5-14B-AWQ at 8K context not being strong enough to integrate structured pre-content without distraction. Worth probing further next session: drop the "reflect every claim" instruction, use a recall-pruned claim list, or move the claim list from prompt-prepend to a separate tool-call round.
- **What the smoke does prove**: infra clean end-to-end, the substrate vs baseline comparison is now real (was artificially identical before), cost per 10-case pair is $0.03 including two extra Qwen calls per substrate case, and the scoring pipeline discriminates (stdev 0.23, range 0.4+ pp).
- Modal app stopped post-run. Total additional spend this sub-session: ~$0.35 (Modal L4 active ~25 min × $0.80/hr) + $0.01 Azure.

### 2026-04-22 — Stream B: ACI-Bench hybrid substrate variant (worktree `D:\wt-aci-hybrid`, `feature/aci-hybrid`)

- **Branch**: `feature/aci-hybrid` off `main@835039a`. Head: `59c7af8` (pending operator merge). No push performed.
- **What landed**:
  - Replaced `_call_acibench_substrate` with a single-call hybrid implementation. Prompt shape: `SECTION 1 — RAW TRANSCRIPT` / `SECTION 2 — STRUCTURED CLAIM SCAFFOLD` (grouped by SOAP section using `predicate_packs/clinical_general/soap_mapping.json`; supersession chains rendered inline per claim) / `SECTION 3 — CONFLICT-RESOLUTION RULE` (baked in verbatim; see `HYBRID_CONFLICT_RULE` constant) / `SECTION 4 — TASK`.
  - `--hybrid / --no-hybrid` CLI flag (default True) and `SmokeConfig.hybrid` field. `--no-hybrid` raises `NotImplementedError` rather than reviving the regressed 2-step path.
  - Three new module-level helpers: `_build_hybrid_prompt`, `_build_claim_scaffold`, `_build_supersession_chains`.
  - +19 unit tests: `tests/unit/eval/test_acibench_hybrid.py` (15 tests — prompt structure, empty-substrate placeholder, supersession-chain rendering, single-call invariant, edit-distance enrichment), `tests/unit/eval/test_acibench_dead_weight_dormancy.py` (4 tests — hallucination guard / differential / verifier / lr_table not invoked). Updated `test_smoke_realrun_wiring.py` mocks for the new `hybrid` kwarg.
- **Test gate**: 305 passed, 2 skipped (from baseline 286 → 305, +19 new). Clean.
- **Dead-weight verified by code inspection AND regression test**: hallucination guard (`src/extraction/asr/*`), differential engine (`src/differential/engine.py`, `lr_table.py`), verifier (`src/verifier/verifier.py`) are not touched on the ACI-Bench hybrid path. The smoke harness's `_call_acibench_substrate` only imports `eval.aci_bench.{adapter,baseline}`, `src.extraction.claim_extractor.extractor`, and substrate primitives; no transitive import chain reaches differential/verifier. Test asserts per-call dormancy.
- **Reasons.md updated**: four new 2026-04-22 entries — Stream B parity-not-dominance posture; conflict rule baked into prompt; single-call over 2-step; ±0.03 gate threshold.
- **Smoke command for operator** (gated on live Modal + Azure endpoints; not run by this agent):

  ```bash
  AZURE_OPENAI_ENDPOINT=<operator-fills> AZURE_OPENAI_API_KEY=<operator-fills> \
  AZURE_OPENAI_GPT4OMINI_DEPLOYMENT=gpt-4-1-mini-2025-04-14 \
  AZURE_OPENAI_GPT4O_DEPLOYMENT=gpt-4-1 \
  ACTIVE_PACK=clinical_general \
  CONCEPT_EXTRACTOR=llm_medcon \
  QWEN_API_BASE=<modal-url-after-deploy> \
  python eval/smoke/run_smoke.py --benchmark acibench --reader qwen2.5-14b \
      --variant both --n 10 --hybrid --budget-usd 5
  ```

- **Phase 1 / Phase 2 gate (pinned)**: Phase 1 is n=10 hybrid against the same 10 cases as `20260422T081250Z`. Decision rule: **at parity (mean delta within ±0.03 of baseline) or above → Phase 2 (n=40 stratified, operator-confirmed, $1 budget cap)**; **below parity → stop, investigate worst-regression case, log diagnostic in reasons.md, no Phase 2**. Half the prior regression magnitude (−0.070) is the discrimination threshold.
- **Hands-off reminder**: PID 9290 (live `gpt-4.1-mini` × n=10 smoke on `D:\hack_it`, started 13:37) untouched; this work is in a separate worktree and has not written to `eval/smoke/results/` or touched the `main` checkout.

### 2026-04-22 — Pre-merge gate: four fixes landed on feature branches

Four corrective commits before any of A/B/D merge to main, per the pre-merge gate prompt.

- **FIX 1** (`feature/lme-retrieval` commit `ce32aa2`): cut `eval/longmemeval/time_expansion.py` + `tests/unit/eval/test_time_expansion.py` + `DateRange` from `src/substrate/retrieval.py` + `time_window` param from `retrieve_relevant_claims` + `TestTimeWindowFilter` from `tests/unit/substrate/test_retrieval.py` + the time-expansion call site in `_call_longmemeval_substrate_retrieval_con`. Audit traced the filter to `c.created_ts` which is `now_ns()` at substrate ingestion (claims.py:246), not the original session timestamp — every `DateRange` anchored "~7 days ago" silently excluded every claim. Re-introduce after `valid_from_ts` schema change (q7).
- **FIX 2** (`feature/lme-retrieval` commit `1555cf8`): added `--legacy-lme-substrate` flag; `_run_longmemeval_case` now defaults the substrate variant to `_call_longmemeval_substrate_retrieval_con` (bge-m3 + retrieval + CoN) instead of the legacy `_call_longmemeval_substrate` (E5 + bundle-then-reader). Dry-run prints which path will be used. Updated wiring test + added a second test confirming the legacy flag still routes to the old path.
- **FIX 3** (`feature/aci-hybrid` commit `8c8081d`): added `--seed <int>` flag (default 42); plumbed through `SmokeConfig.seed` → `reader_env["seed"]` → `_call_qwen` and `_call_openai` (both pass `seed` to `chat.completions.create`; gpt-4.1 + Qwen-vLLM both honour it). Results dir name now `<UTC-timestamp>_seed<N>`; sidecar `config.json` written before any case runs. 7 new tests in `tests/unit/eval/test_smoke_seed_flag.py`. Phase 1.5 multi-seed gate documented in `reasons.md` — re-run winning arm 3× with seeds {42, 43, 44} before any Phase 2 escalation; collapse noise contribution from ~1σ to ~0.6σ; gate's false-positive rate at the ±0.03 threshold drops from ~16% to ~4%.
- **FIX 4** (no commit on feature branches; integration verification only): three-way merge on a throwaway `pre-merge-integration-test` branch surfaced two conflict regions (run_smoke.py SmokeConfig + argparse + instantiation; reasons.md). Resolved by rebasing `feature/aci-hybrid` onto `feature/lme-retrieval` so STEP 5 ff-merges. Pyright on the integration tree returned 0 errors in the four cross-worktree-noise categories (`reportMissingImports`, `reportUndefinedVariable`, `reportCallIssue` "No parameter", `reportAttributeAccessIssue` "Cannot access attribute") that flagged during the stream work — the noise resolved exactly as predicted. Pre-existing 359 strict-mode warnings on untyped third-party libs (pytest.approx, openai, structlog) are not in the gate's categories.

Each FIX 1/2 commit also got a per-decision `reasons.md` entry on `feature/lme-retrieval` (commit `c52aa0a`) explaining why the alternative (Finding B for FIX 1, additive-only for FIX 2) was rejected.

### 2026-04-22 — STEP 5: feature branches merged to main

Three sequential merges, test gate green between each. Each push gated on `pytest -q` clean.

- **Stream A** merged into main at `c52aa0a` (`docs: log pre-merge gate FIX 1 (time_expansion cut) + FIX 2 (dispatcher flip)`). Test gate: 286 → **315 passed, 2 skipped**. Pushed to `origin/main` (`835039a..c52aa0a`).
- **Stream B** rebased onto A, then ff-merged into main at `c59168b` (`smoke: add --seed flag and document Phase 1.5 multi-seed gate`). Test gate: 315 → **341 passed, 2 skipped**. Pushed to `origin/main` (`c52aa0a..c59168b`).
- **Stream D** merged into main at `a92910f` (`Merge branch 'feature/asr-spec'`). Test gate: 341 → **351 passed, 2 skipped**. Pushed to `origin/main` (`c59168b..a92910f`).

Final main commit: `a92910f`. Final test gate: **351 passed, 2 skipped, 0 failed, 0 xfail**. Open asks for operator: gpt-4o Azure deploy, Modal bge-m3 deploy + Qwen reader, Stream A smoke run, Stream B Phase 1 + Phase 1.5 smoke runs, ASR measurement GPU window + extra synthetic clips, demo video narration confirmation, origin feature-branch deletes, worktree pruning.

### 2026-04-22T20:13:07Z — Stream E UMLS T0 install
- PREFLIGHT: UMLS_API_KEY present (length=36), D: free=262 GB (>=30 GB ok), Java=NOT INSTALLED (blocker for Step 2 MetamorphoSys) — RESULT: attempting Step 1 (download) anyway; Step 2 will hard-block if Java not resolved
- STEP 1: begin authenticated UMLS 2025AB Level0 download via TGT flow — RESULT: in progress
- STEP 1 DETAIL: TGT + ST flow works (201 Created, ST obtained). Cookie jar + redirect-follow required (MOD_AUTH_CAS cookie set on first hop, consumed on second). Verified against 2023AB full-metathesaurus URL which returned HTTP 200 Content-Length=4135708549 (4.1 GB).
- STEP 1 BLOCKER A: URL `umls-2025AB-Level0.zip` returns HTTP 404. Release 2025AB is not yet published on NLM (today is 2026-04-22; UMLS naming cadence is AA=May, AB=November — 2025AB would have been Nov 2025 but is not in the catalog). Also tested 2025AA, 2024AB, 2024AA, all with `-Level0.zip` suffix: all 404. NLM does not package a file literally named "Level0" — confirmed by directory probing. The real filenames are `umls-<REL>-metathesaurus-full.zip` (verified 200 for 2024AA and 2023AB) and `umls-<REL>-mrconso.zip`. "Level 0" refers to the source-license tier, not a filename; the full zip contains all levels and MetamorphoSys filters at extract time.
- STEP 1 BLOCKER B (compound): `java` is not installed on PATH (`java: command not found`). MetamorphoSys (Step 2) is a Java GUI/CLI tool — without a JRE/JDK installed, Step 2 cannot run even if Step 1 is salvaged.
- DECISION: Pausing. Downloading ~4 GB of a different release (2024AA) that cannot then be processed by MetamorphoSys (no Java) is wasted bandwidth and also a silent deviation from the task spec (2025AB). Per failure discipline + MEMORY.md eval-legitimacy rule, flagging both issues rather than silently substituting.
- RESOLUTIONS NEEDED FROM MAIN THREAD:
  (1) Confirm desired release given 2025AB unavailable — accept 2024AA as substitute (label numbers `MEDCON-T0 @ UMLS-2024AA`), or defer to T1 scispaCy for now.
  (2) Install a JDK (e.g., `winget install Microsoft.OpenJDK.21` or Temurin 21) so MetamorphoSys can run headless. Without this, T0 path is dead on this Windows box.
- STATE: T0 NOT installed. Blocker-at-Step-1/2 (release-naming + no-Java). No files written under `.cache/umls/` beyond the empty target directory. `.env` not modified. No git changes.

### 2026-04-22T20:13:07Z onwards — Post-merge execution cycle

Cycle invoked via `/search-first` with "Post-merge execution cycle — parallel smokes, P0 confirmation, Stream D kickoff". Three compute streams (A/B/D) + Stream E UMLS install launched in parallel from `main=58b7db2`. Detailed notes:

**Commits landed** (not yet pushed):
- `6085827` eval: P.1 harness — `--output-dir` + `--stratified` + `gpt-4o-2024-11-20` in READERS + `data/longmemeval_s_cleaned.json` resolver. Adds 9 unit tests (`tests/unit/eval/test_smoke_output_dir_and_stratified.py`). 2 mocks in `test_smoke_realrun_wiring.py` updated to accept `**kwargs`; combination-count assertion in `test_smoke_harness_dryrun.py` bumped 16 → 20.
- `26bdc78` extraction: Stream D — 5 new chief-complaint dialogue scripts under `eval/synthetic_clips/` (abdominal pain, dyspnea, headache, fatigue+weight loss, dizziness/syncope), text-only, each 150–250 words with biasing-vocab block, supersession moments, and expected claim-extraction targets. `docs/asr_engineering_spec.md` §8.2 updated to strike the "3–5 additional clips" open ask.
- `c896b05` eval+infra: bge-m3 dict-body fix (FastAPI+pydantic v2 ForwardRef issue resolved by using `req: dict = Body(...)` instead of an inner BaseModel) + `_purpose_map` routing for `gpt-4o-2024-11-20` / `gpt-4o-2024-08-06` through `longmemeval_reader` purpose (Azure routing, not OpenAI direct).

**Infrastructure deployed this cycle**:
- Modal: `bge-m3-embeddings` (L4 GPU, 1024-dim embeddings) + `qwen25-14b-vllm` (L4 GPU, vLLM OpenAI-compatible on :8000). Both live under profile `glitch112213`.
- Azure: `gpt-4o-2024-11-20` (successor to paper-pinned deprecated `2024-08-06`) on `gt-swebench-aoai-3`. Reader+judge for LongMemEval this cycle route via `AZURE_OPENAI_GPT4O_LME_DEPLOYMENT`.
- scispacy T1 MEDCON: `en_core_sci_lg` (528 MB wheel) installed via D:-drive TMPDIR workaround (C: was full); UMLS linker assets (~2 GB) on `D:\hack_it\.cache\scispacy/` via `SCISPACY_CACHE` env var.

**Blockers encountered and resolved this cycle**:
- **Modal bge-m3 HTTP 422 on /embed**: FastAPI+pydantic v2 ForwardRef bug. Dropped inner BaseModel for `dict` body + explicit `Body(...)`. Redeployed. 200 OK.
- **Stream A 401 "no API key"**: `_call_openai` `_purpose_map` missing `gpt-4o-*` routes. Added to map under `longmemeval_reader` purpose. Fixed.
- **Stream B.1 `OSError: [Errno 28] No space left`**: C: drive full (240/242 GB). scispacy linker cache needed ~2 GB. Redirected via `SCISPACY_CACHE=D:/hack_it/.cache/scispacy` + HF_HOME + SENTENCE_TRANSFORMERS_HOME. Initial cache copy had a truncated jsonl (`umls_2022_ab_cat0129.jsonl`, char 486) because the previous C:-partial-download got copied forward; nuked and let scispacy re-download fresh to D:.
- **Modal Windows encoding HTTP 422-adjacent**: first `modal deploy` hit charmap encode error on stdout progress bars; set `PYTHONIOENCODING=utf-8` + `PYTHONUTF8=1` for subsequent Modal CLI calls.

**Blockers unresolved (deferred follow-up)**:
- **Stream E UMLS T0**: compound blocker — (a) assumed filename `umls-2025AB-Level0.zip` is wrong (real pattern `umls-<REL>-metathesaurus-full.zip`; 2025AB release also not yet in NLM catalog), (b) no Java on PATH for MetamorphoSys, (c) C: disk full. T1 scispacy MEDCON remains the ship-tier. Follow-up cycle needs Java install + UMLS release-target confirmation + C: cleanup.
- **Azure gpt-4.1-mini TPM throttling**: `claim_extractor_gpt4omini` purpose routed to `gpt-4-1-mini-2025-04-14` SKU capacity=10. Stream A (60 questions × ~50 sessions × ~5 turns each) and Stream B (10 encounters × ~30 turns each) both hammer this endpoint; 429s are persistent (every ~1 min in logs). Both streams retry and make forward progress; throughput is limited to Azure TPM. Operator can raise capacity to 30+ to ~3× throughput. Azure spend is uncapped per this cycle's budget decision.

**Smoke runs launched**:
- Stream B.1 ACI-Bench hybrid n=10 seed 42 → `eval/acibench/results/20260423_postmerge_hybrid_phase1_<UTC>_seed42/`
- Stream A LongMemEval stratified n=60 seed 42 → `eval/longmemeval/results/20260423_postmerge_lme_stratified_<UTC>_seed42/`

At cycle-report time both were still running with Azure throttle back-pressure; see rolling-state for the next-actions on their partial/final results.

**Reasons.md entries appended** (4): (1) `gpt-4o-2024-08-06` rejected in favour of `gpt-4o-2024-11-20` as reader/judge (Azure deprecation); (2) P0 (pre-hybrid) ACI-Bench reproduction skipped (architecturally removed from main; `NotImplementedError` at `run_smoke.py:1255`); (3) Stream E UMLS T0 install paused at Step 1/2 with compound blocker; (4) bge-m3 Modal deploy FastAPI body-parsing bug (Pydantic ForwardRef) patched to dict body + `Body(...)`.

**Memory entries added**: `reference_umls_api_key.md` (points at `D:\hack_it\.env`; raw UMLS API key never echoed).

**Modal spend this cycle**: <$0.25 (bge-m3 + Qwen cold-starts + probes; Stream B.1 ingestion-phase tokens low). Azure spend: several thousand gpt-4.1-mini extraction calls plus a handful of gpt-4o-2024-11-20 LongMemEval reader calls — rough estimate $2–$5, well within uncapped-Azure scope.
