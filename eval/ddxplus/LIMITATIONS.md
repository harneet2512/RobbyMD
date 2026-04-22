# DDXPlus harness — known limitations

Per `rules.md §6.3` (methodology honesty) and `§9.3` (no deceptive benchmark
claims). Every caveat below is reported on the demo slide alongside the
numbers.

## 1. Adapter fidelity

DDXPlus's `EVIDENCES` list is a set of structured-finding IDs; real clinical
dialogue has narrative prose, hedge language, tangents, and supersession. Our
adapter produces deterministic physician-prompt / patient-answer pairs keyed
by `release_evidences.json`. This is **less** noisy than real dialogue and
**less** expressive than the natural-language chief complaint a doctor
actually receives.

Mitigation: documented here; the same adapter runs both `baseline` and `full`
variants, so any adapter-induced bias affects both symmetrically (the
differential delta is still meaningful even if absolute numbers are shifted).

## 2. Judge pinning

Top-5 semantic equivalence is scored by an LLM judge. We pin to
`gpt-4o-2024-08-06` via `DDXPLUS_JUDGE_MODEL`. H-DDx 2025 used `gpt-4o` without
pinning a specific snapshot; our numbers are reproducible but may not
bit-for-bit reproduce H-DDx's Table 2 if OpenAI rotates the unpinned alias.

## 3. HDF1 scope

HDF1 = ICD-10 hierarchical F1 per H-DDx 2025 §3.4. We implement the
deterministic retrieval+rerank of pathology → ICD-10 chain and compute F1
against DDXPlus's gold. The ICD-10 ontology file is pinned via the
`icd10_tree.json` artefact checked in under `eval/ddxplus/data/` (see
`fetch.py`).

## 4. Subset size

We run the **730-case stratified subset** per H-DDx 2025 methodology. This
subset is H-DDx's own "full canonical comparator set" — not a slice we
invented. See `memory/feedback_full_benchmarks.md` and
`docs/research_brief.md §3.1`.

Running smaller slices (e.g. `--limit 50`) is allowed for dev smoke only and
must **never** be reported as a comparator number on the demo slide.

## 5. Comparator-set subset

`README.md`'s comparator table subsets H-DDx Table 2's 22 LLMs to 9 models
(proprietary + open + medical-FT). This is a presentation choice, not a
methodology choice. The full Table 2 is in the H-DDx paper.

## 6. No published Opus 4.7 number

H-DDx 2025 does not include Opus 4.7. Our baseline row would be the first
published number. This is noted as informational, not as a superiority claim.
