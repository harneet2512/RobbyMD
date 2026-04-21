# Groundtruth Clinical Substrate

> **Research prototype. Not a medical device.** This software is a research demonstration. It does not diagnose, treat, or recommend treatment, and is not intended for use in patient care. All patients, conversations, and data shown are synthetic or from published research benchmarks. Clinical decisions are made by the physician. This system supports the physician's reasoning by tracking claims and differential hypotheses in real time; it does not direct clinical judgement.

Built for the **Built with Opus 4.7 Hackathon** (Cerebral Valley × Anthropic, April 2026).

---

## What this is

A live clinical reasoning substrate that sits between a doctor and a patient during a consultation. It extracts structured claims from the conversation, tracks how those claims evolve (including patient self-corrections), runs parallel differential-diagnosis hypothesis trees deterministically, surfaces supporting/contradicting evidence and the single next-best question for the top two hypotheses, and generates a SOAP note with full per-sentence provenance back to the source conversation turn.

**Thesis**: *in messy workflows the final output is only one layer; underneath it there must be a structured context layer that tracks facts, their sources, and their lifecycle — supersession, confirmation, contradiction — deterministically.*

The demo domain is clinical because the stakes are visceral and the thesis maps naturally to clinical reasoning. The substrate itself is domain-general.

## Regulatory posture

Positioned as a **Non-Device Clinical Decision Support (CDS)** software function under Section 520(o)(1)(E) of the Federal Food, Drug, and Cosmetic Act, per the FDA's *Clinical Decision Support Software: Final Guidance* (January 2026). Rationale and the four CDS criteria: see `context.md §7`.

**HIPAA**: zero PHI. All data is synthetic or from published research benchmarks (DDXPlus, LongMemEval-S, ACI-Bench). See `SYNTHETIC_DATA.md`.

## Repository map

See `CLAUDE.md §3` for the canonical directory layout.

```
context.md              hackathon framing + compliance
PRD.md                  product requirements
Eng_doc.md              engineering spec
CLAUDE.md               agent operating instructions
rules.md                non-negotiables
SYNTHETIC_DATA.md       dataset manifest

src/substrate/          claim store, supersession, projections
src/extraction/         ASR + claim extractor
src/differential/       parallel hypothesis trees + LR-weighted update
src/verifier/           counterfactual verifier + next-best-question
src/note/               SOAP generator with provenance
src/api/                local server for UI

ui/                     React + Tailwind + shadcn + ReactFlow + Zustand

content/differentials/chest_pain/   branches, LR table, citations

eval/ddxplus/           H-DDx 730-case stratified subset
eval/longmemeval/       500 questions, all categories
eval/aci_bench/         aci + virtscribe subsets

tests/{unit,e2e,property,privacy,licensing}/
docs/{decisions,research_brief.md,gt_v2_study_notes.md}
```

## Quickstart

```bash
./scripts/setup.sh                                       # dev deps + dataset pulls
./scripts/run_demo.sh --case chest_pain_01 --seed 42     # scripted chest-pain case
pytest                                                   # all tests
make eval                                                # three published benchmarks
```

## Licensing

This repository is licensed under the **Apache License 2.0** (see `LICENSE`). All dependencies are OSI-approved open source (MIT / Apache-2.0 / BSD / MPL / ISC / LGPL); enforced by `tests/licensing/test_open_source.py`. Claude Opus 4.7 via the Anthropic API is the hackathon's named sponsored tool.

## Scope

Demo scope, deliberately narrow: one chief complaint (chest pain), four branches (Cardiac / Pulmonary / MSK / GI), one scripted standardised patient case, single session, HCP-facing only. Vision features (multi-complaint, multi-specialty, multi-session, EHR/FHIR, full 7-signal retrieval, NLI verifier) live in this README's vision section, not in the build. See `PRD.md §3`.

## Disclaimer

See the block at the top of this README — that text appears verbatim in the app header, the demo video opening and closing cards, and the written submission summary. Required by `rules.md §3.7`.
