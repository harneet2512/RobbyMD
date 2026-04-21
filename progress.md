# progress.md — Rolling state + append-only log

This file is the project's single source of truth for *where we are right now* and *what we've done*. The rolling-state block at the top is edited in place. The append-only log below grows; never delete or rewrite past entries.

All agents (main + per-worktree) read this on startup and append a new entry at every meaningful state transition (research complete, adapter scaffolded, test failing, etc.).

---

## Rolling state (edit in place)

- **Date**: 2026-04-21
- **Phase**: Research — supplementary Researcher A (ASR stack) + B (Clinical chest pain LR expansion) pending dispatch; Validator gated on both; no worktree agents running.
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
- **Next decision**: user reviews `research/validation_report.md` after researchers + validator complete, then approves worktree dispatch.

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
