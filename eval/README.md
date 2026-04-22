# eval — Published-benchmark harnesses

Three harnesses scaffolded + one stub. Per `rules.md §6` — no homemade metrics, no slicing of canonical splits, no unverified comparator claims.

Per `Eng_doc.md §3.5` (model-usage policy): **Opus 4.7 is NOT used as the eval reader** for any benchmark. Eval readers match each benchmark's published SOTA reader so numbers are apples-to-apples. Using Opus 4.7 as the eval reader when the published SOTA used `gpt-4.1-mini` or `gpt-5-mini` conflates "our substrate vs their substrate" with "Opus 4.7 vs their reader" — and kills the comparison.

## Per-benchmark reader + judge table

| Benchmark | Reader (baseline + substrate variant) | LLM judge | SOTA precedent |
|---|---|---|---|
| **DDXPlus** | `gpt-4o` | `gpt-4o-2024-08-06` (pinned) | [H-DDx 2025 Table 2 (arXiv 2510.03700)](https://arxiv.org/abs/2510.03700) — 22 LLM comparators; GPT-4o sits mid-table (0.804 Top-5 / 0.350 HDF1) as a neutral comparator. HDF1 scoring is deterministic via ICD-10 retrieval+rerank; Top-5 semantic-equivalence uses the pinned GPT-4o judge. |
| **LongMemEval-S** | `gpt-5-mini` | `gpt-4o-2024-08-06` (pinned by LongMemEval paper) | [Mastra Observational Memory — 94.87%](https://mastra.ai/research/observational-memory) uses `gpt-5-mini` as the driver. Matching this makes our substrate-variant number directly comparable to Mastra OM (our per-category claim is on TR / multi-session / KU — the hard categories). |
| **ACI-Bench** | `gpt-4.1-mini` | None (ROUGE-1/2/L, BERTScore, BLEURT are deterministic; MEDCON computed via 3-tier concept extractor — `docs/decisions/2026-04-21_medcon-tiered-fallback.md`) | [WangLab 2023 GPT-4 ICL](https://arxiv.org/abs/2305.02220) — MEDIQA-CHAT 2023 1st place. `gpt-4.1-mini` is the modern cost-sensitive GPT-4-class equivalent; budget-permitting upgrade to full GPT-4 for a direct WangLab comparison. |
| **MedQA** (stub) | `gpt-4.1-mini` | None (multi-choice accuracy is deterministic) | MedQA canonical benchmark (Jin et al. 2021). Modern comparators include GPT-4 (~86.7%), Med-PaLM-2 (~86.5%), frontier models. Reader pinned in `eval/medqa/README.md` when the harness ships. |

## Adapter convention

Each per-benchmark directory follows this layout:

```
eval/<benchmark>/
  README.md       pinned SHA + license + methodology + comparator table
  fetch.py        downloads dataset (to eval/<benchmark>/data/, gitignored)
  adapter.py      native format → substrate turn stream
  baseline.py     reader direct-prompted, no substrate
  full.py         reader + substrate retrieval
  run.py          top-level runner; writes eval/reports/<ts>/<benchmark>/
  LIMITATIONS.md  methodology caveats per rules.md §6.3
```

## Pack × benchmark mapping

Per `context.md §6`: each benchmark loads a specific `PredicatePack` — the substrate's domain-agnosticism is tested here under the hood. Packs shipped this build: `clinical_general` (only). Consequence for the substrate variant:

| Benchmark | Pack | Substrate variant shippable this build? |
|---|---|---|
| DDXPlus | `clinical_general` + respiratory-pathologies extension | Yes |
| LongMemEval-S | `personal_assistant` (future pack) | **Baseline only** this build — substrate variant deferred until `personal_assistant` seeds |
| ACI-Bench | `clinical_general` | Yes |
| MedQA | `clinical_general` (MCQ mode) | When harness ships (stub currently) |

## Rule

If a call on the eval path needs an LLM → use the reader above for that benchmark. If it needs a judge → the pinned judge. **Do not substitute Opus 4.7 for cost.** See `reasons.md` → "Tank for the war, not the gun fight" + "Scope widened under the hood; demo narrative stays clinical" for the full rationale.

## MEDCON 3-tier fallback (ACI-Bench specifically)

See `docs/decisions/2026-04-21_medcon-tiered-fallback.md`. T1 (scispaCy) is the default; T0 (QuickUMLS) activates if UMLS licence lands; T2 (ROUGE+BERTScore+BLEURT+MedEinst Bias Trap Rate) is the hard fallback.
