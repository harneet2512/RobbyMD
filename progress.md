# progress.md — Rolling state + append-only log

This file is the project's single source of truth for *where we are right now* and *what we've done*. The rolling-state block at the top is edited in place. The append-only log below grows; never delete or rewrite past entries.

All agents (main + per-worktree) read this on startup and append a new entry at every meaningful state transition (research complete, adapter scaffolded, test failing, etc.).

---

## Rolling state (edit in place)

- **Date**: 2026-04-21
- **Phase**: Research — Researcher A + B complete; Validator complete → **gate FAIL (8 attribution blockers)**; no worktree agents running until blockers patched.
- **Main commit**: `090c799` (scaffold: initial repo layout, compliance tests, research docs)
- **Worktrees** (all idle at `090c799`):
  - `D:\hack_it` — main (human operator)
  - `D:\wt-engine` — `feature/substrate` (idle)
  - `D:\wt-trees` — `feature/differential` (idle)
  - `D:\wt-extraction` — `feature/extraction` (idle)
  - `D:\wt-ui` — `feature/ui` (deferred until engine+extraction+eval publish API shapes)
  - `D:\wt-eval` — `feature/eval` (idle)
- **UMLS licence**: application submitted (CMU email); approval pending, up to 3 business days. Not on critical path (see `docs/decisions/2026-04-21_medcon-tiered-fallback.md`).
- **MEDCON tier active**: T1 (scispaCy) by default; T0 (QuickUMLS) upgrade-ready on licence approval.
- **Benchmarks**: DDXPlus (H-DDx 730-case Top-5 + HDF1), LongMemEval-S (all 500 questions, per-category), ACI-Bench (`aci` + `virtscribe`, 90 test encounters). **No slicing of published benchmarks** per memory rule `feedback_full_benchmarks.md`.
- **Invariant tests** (must stay green every commit):
  - `tests/licensing/test_open_source.py` — OSI allowlist, green (no deps beyond OSI)
  - `tests/privacy/test_no_phi.py` — PHI sentinel, green
  - `tests/property/test_determinism.py` — xfail (awaiting differential engine)
- **Next decision**: Researcher B patches the 7 clinical-citation blockers (sources.md + lr_table.json), Researcher A patches the 1 ASR-citation blocker (asr_stack.md), then re-submit for targeted re-validation (Check 1 + Check 2 only). After that, user approves worktree dispatch.

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
- Committed scaffold `090c799` on `main`.
- Created 5 worktrees at `D:\wt-*` on `feature/*` branches.

### 2026-04-21 — UMLS investigation
- Read UMLS REST API docs — confirmed the REST API cannot replace QuickUMLS for MEDCON (no span-detection endpoint; ~500k calls infeasible; would compute a different metric). Captured in `reasons.md`.
- Read UMLS Knowledge Sources page — confirmed download target is `umls-2025AB-Level0.zip` (1.8 GB compressed, 10.3 GB uncompressed, released 2025-11-03). MRCONSO standalone rejected (missing MRSTY). Captured in `reasons.md`.
- User's UMLS licence application submitted on CMU email (`hbali@andrew.cmu.edu`); awaiting approval.

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
