# eval — Published-benchmark harnesses

**Two** harnesses scaffolded: LongMemEval-S and ACI-Bench. Per `rules.md §6` — no homemade metrics, no slicing of canonical splits, no unverified comparator claims.

Per `Eng_doc.md §3.5` (model-usage policy): **Opus 4.7 is NOT used as the eval reader** for any benchmark. Primary reader is `Qwen2.5-14B-Instruct` (Apache-2.0, self-hosted via vLLM on GCP L4 spot). Secondary comparability readers match each benchmark's published SOTA so numbers are apples-to-apples with the leaderboard.

**Why two, not four**: DDXPlus + MedQA were dropped 2026-04-21 (see `reasons.md` → DDXPlus and MedQA entries). DDXPlus's 49-pathology respiratory set didn't map to our seeded `clinical_general` chest-pain differential without a respiratory-extension pack; MedQA tests reader knowledge, not substrate architecture. LongMemEval-S + ACI-Bench directly exercise the substrate's load-bearing contributions (memory lifecycle + note-gen pipeline).

## Per-benchmark reader + judge table

| Benchmark | Primary reader (baseline + substrate variant) | Secondary reader (comparability) | LLM judge | SOTA precedent |
|---|---|---|---|---|
| **LongMemEval-S** | `Qwen2.5-14B-Instruct` (GCP L4 self-hosted) | `gpt-4o-mini` | `gpt-4o-2024-08-06` (pinned by LongMemEval paper) | Primary: reader-agnostic open-weight baseline at fixed Qwen scale. Secondary: matches [Mem0](https://mem0.ai/research) / [Mastra Observational Memory 94.87%](https://mastra.ai/research/observational-memory) / [EverMemOS](https://arxiv.org/abs/2601.02163) reporting axis. |
| **ACI-Bench** | `Qwen2.5-14B-Instruct` (same self-hosted stack) | `gpt-4.1-mini` | None (ROUGE-1/2/L, BERTScore, BLEURT are deterministic; MEDCON via 3-tier concept extractor — `docs/decisions/2026-04-21_medcon-tiered-fallback.md`) | Primary: same reader-agnostic story. Secondary: [WangLab 2023 GPT-4 ICL](https://arxiv.org/abs/2305.02220), MEDIQA-CHAT 2023 1st place; `gpt-4.1-mini` is the modern cost-sensitive GPT-4-class equivalent. |

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

Per `context.md §6`: each benchmark loads a specific `PredicatePack`. Both packs shipped this build: `clinical_general` + `personal_assistant`.

| Benchmark | Pack | Substrate variant shippable this build? |
|---|---|---|
| LongMemEval-S | `personal_assistant` (seeded this commit — `predicate_packs/personal_assistant/`) | Yes |
| ACI-Bench | `clinical_general` | Yes |

## Smoke-first discipline

Before any full run, use `eval/smoke/run_smoke.py` with `--n 10` on the first-10-by-sorted-case-id selection:

```bash
eval/smoke/prepare_datasets.sh                                    # one-shot dataset fetch
python eval/smoke/run_smoke.py --dry-run --benchmark both \
       --reader qwen2.5-14b --variant both --n 10                 # dry-run sanity
python eval/smoke/run_smoke.py --benchmark both \
       --reader qwen2.5-14b --variant both --n 10 --budget-usd 50 # real smoke, budget-capped
```

Harness enforces a hard `--budget-usd` cap with early halt + `BUDGET HALT` summary. Outputs `eval/smoke/<benchmark>/<timestamp>/{results.json, methodology.md}`. Verdict: ✅ PASS / ⚠ ANOMALY / ❌ FAIL per `eval/smoke/reference_baselines.json`.

## Hosting the primary reader

See `eval/infra/README.md` for cloud-choice rationale. Primary: `eval/infra/deploy_qwen_gcp.sh` (GCP L4 spot + vLLM INT8 — Qwen2.5-14B fits in 24 GB VRAM at INT8). Fallback: `eval/infra/deploy_qwen_azure.sh` (Azure NVadsA10_v5 spot; sketch-only, untested). Host already authenticated for both (`gcloud` + `az`).

## Rule

If a call on the eval path needs an LLM → use the reader above for that benchmark. If it needs a judge → the pinned judge. **Do not substitute Opus 4.7 for cost or capability.** See `reasons.md` → "Tank for the war, not the gun fight" + "Scope widened under the hood; demo narrative stays clinical" for the full rationale.

## MEDCON 3-tier fallback (ACI-Bench specifically)

See `docs/decisions/2026-04-21_medcon-tiered-fallback.md`. T1 (scispaCy) is the default; T0 (QuickUMLS) activates if UMLS licence lands; T2 (ROUGE+BERTScore+BLEURT+MedEinst Bias Trap Rate) is the hard fallback.
