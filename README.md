# Groundtruth Clinical Substrate

> **Research prototype. Not a medical device.** This software is a research demonstration. It does not diagnose, treat, or recommend treatment, and is not intended for use in patient care. All patients, conversations, and data shown are synthetic or from published research benchmarks. Clinical decisions are made by the physician. This system supports the physician's reasoning by tracking claims and differential hypotheses in real time; it does not direct clinical judgement.

Built for the **Built with Opus 4.7 Hackathon** (Cerebral Valley × Anthropic, April 2026).

---

## The product — clinical reasoning for the exam room

A live clinical reasoning substrate that sits between a doctor and a patient during a consultation. It extracts structured claims from the conversation, tracks how those claims evolve (including patient self-corrections), runs parallel differential-diagnosis hypothesis trees deterministically, surfaces supporting/contradicting evidence and the single next-best question for the top two hypotheses, and generates a SOAP note with full per-sentence provenance back to the source conversation turn.

**Thesis**: *in messy workflows the final output is only one layer; underneath it there must be a structured context layer that tracks facts, their sources, and their lifecycle — supersession, confirmation, contradiction — deterministically.*

**Demo scope**: one chief complaint (chest pain), four differential branches (Cardiac / Pulmonary / MSK / GI), one scripted standardised patient case, single session, HCP-facing only. See `PRD.md §3`.

**Regulatory posture**: Non-Device Clinical Decision Support under Section 520(o)(1)(E) of the FD&C Act, per FDA's January 2026 CDS Final Guidance. HIPAA: zero PHI; data is synthetic or from published research benchmarks. See `context.md §7`.

**Demo video** (link when cut): *pending 2026-04-26*.

---

## Architecture — the substrate is domain-agnostic

The clinical product is one instance of a general claim-substrate. The engine (claim store, supersession graph, projection layer, provenance chain, event bus) does not know about medicine. All medical-specific vocabulary lives in a **pluggable predicate pack** (`predicate_packs/clinical_general/`):

```
conversation ─► ASR ─► claim extractor ─► ┌──────────────────────────┐
                                          │ PredicatePack (active)   │
                                          │  • predicate families    │
                                          │  • sub-slot schemas      │
                                          │  • LR table (clinical)   │
                                          └────────────┬─────────────┘
                                                       │
 ┌─ claim store ─► supersession graph ─► projection layer ─► differential engine ─► verifier ─► SOAP note ─┐
 │   (SQLite)      (typed edges)         (materialised views) (deterministic LR)   (CPG)       (provenance) │
 └───────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

**One pack shipped this build**: `clinical_general` (covers chest pain and any other chief complaint — abdominal pain, dyspnoea, headache, fatigue, etc.; schema accepts additional complaint branches with no engine changes). Future packs (`personal_assistant` for LongMemEval-S, `coding_agent`, `legal`, …) register via the same API; the engine treats them identically. See `Eng_doc.md §4.2`.

The generality is by construction. Any repo-reader can see it from the file tree (`predicate_packs/clinical_general/differentials/{chest_pain, abdominal_pain, dyspnoea, headache}/`) and from the fact that no engine module imports anything clinical. The demo video stays single-story clinical; the architecture speaks for itself to anyone opening the code.

---

## Repository map

```
context.md              hackathon framing + compliance
PRD.md                  product requirements
Eng_doc.md              engineering spec
CLAUDE.md               agent operating instructions
rules.md                non-negotiables
SYNTHETIC_DATA.md       dataset manifest
MODEL_ATTRIBUTIONS.md   model-weight licence + attribution registry

src/substrate/          claim store, typed supersession edges, projections, event bus
src/extraction/         ASR + claim extractor
src/differential/       parallel hypothesis trees + LR-weighted deterministic update
src/verifier/           counterfactual verifier + next-best-question
src/note/               SOAP generator with provenance
src/api/                local server for UI

ui/                     React + TS + Tailwind + shadcn + ReactFlow + Zustand

predicate_packs/
  clinical_general/     the one pack shipped this build
    differentials/
      chest_pain/       seeded + rehearsed (79 LR rows, 22 citations)
      abdominal_pain/   stub — schema-ready
      dyspnoea/         stub
      headache/         stub

eval/
  README.md             per-benchmark reader + judge table
  longmemeval/          500 questions, all categories
  aci_bench/            aci + virtscribe subsets
  smoke/                smoke-run harness (built, not yet run)
  infra/                deploy_qwen_gcp.sh + deploy_qwen_azure.sh
```

See `CLAUDE.md §3` for the canonical layout.

## Quickstart

```bash
./scripts/setup.sh                                       # dev deps + dataset pulls
./scripts/run_demo.sh --case chest_pain_01 --seed 42     # scripted chest-pain case
pytest                                                   # all tests
make eval                                                # published benchmarks
```

## Licensing

Apache 2.0 (see `LICENSE`). All **code** dependencies are OSI-approved (MIT / Apache-2.0 / BSD / MPL / ISC / LGPL). **Model weights** may additionally use open-data licences (CC-BY-4.0, CC-BY-SA-4.0, CDLA-Permissive-2.0, ODbL) per the Linux Foundation OpenMDW framework reading — see `rules.md §1.2` and `MODEL_ATTRIBUTIONS.md`. Attribution is load-bearing and enforced by `tests/licensing/test_model_attributions.py`. Claude Opus 4.7 via the Anthropic API is the hackathon's named sponsored tool.

## Disclaimer

The disclaimer block at the top of this README appears verbatim in the app header, the demo video opening and closing cards, and the written submission summary. Required by `rules.md §3.7`.
