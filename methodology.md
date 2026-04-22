# methodology.md — evaluation methodology

Source of truth for how benchmark scores reported in this repo are actually computed. Update whenever a metric is added, dropped, or swapped. Pairs with `progress.md` (state) and `reasons.md` (decisions).

**Hackathon disclaimer**: Research prototype for the "Built with Opus 4.7" hackathon. Not a medical device. No real patient data.

---

## Benchmarks

Two published benchmarks, both scored on the full published test set (no slicing, per `memory/feedback_full_benchmarks.md`):

| Benchmark       | Test set size                  | Loads pack             | Primary reader            |
| --------------- | ------------------------------ | ---------------------- | ------------------------- |
| LongMemEval-S   | 500 questions (ICLR 2025)      | `personal_assistant`   | Qwen2.5-14B-Instruct      |
| ACI-Bench       | 90 encounters (aci + virtscribe test splits) | `clinical_general` | Qwen2.5-14B-Instruct |

Secondary readers for published-comparator alignment: `gpt-4o-mini` (LongMemEval-S), `gpt-4.1-mini` (ACI-Bench). Opus 4.7 does not run in eval loops (`Eng_doc.md §3.5` model-usage policy — tank for the demo, not the gun fight).

---

## LongMemEval-S

- **Metric**: per-category accuracy as defined by the LongMemEval-S authors (Wu et al., ICLR 2025). Judged by `gpt-4o-2024-08-06` using the official judge prompts shipped with the benchmark's `evaluation/` directory.
- **Variants reported**: `baseline` (reader-only, no substrate retrieval) vs `substrate` (reader routed through the claim / supersession / projection substrate, `personal_assistant` pack loaded).
- **Reference baselines (for the anomaly gate in the smoke harness)**: Mem0 49.0, Zep 63.8, gpt-4o-mini 61.2. Source: `eval/smoke/reference_baselines.json` (built from LongMemEval-S leaderboard + Mem0/Zep papers).

---

## ACI-Bench

- **Primary metrics**: ROUGE-1 / ROUGE-2 / ROUGE-L, BERTScore (MEDIQA-CHAT 2023 configuration), plus **LLM-MEDCON** (see next section).
- **Variants reported**: `baseline` (predict_note → off-the-shelf reader on raw encounter) vs `full` (substrate-augmented SOAP generator via `eval/aci_bench/full.py`).
- **Reference baseline (smoke anomaly gate)**: WangLab 2023 MEDIQA-CHAT Task B — 57.78 MEDCON. Used for the ±20 pp sanity check only; LLM-MEDCON numbers are not directly comparable to the 57.78 figure (different extraction backend), so this check is informational.

### LLM-MEDCON (shipped variant of the MEDCON metric)

LLM-MEDCON uses gpt-4o-mini for medical concept extraction. Set-based precision / recall / F1 over extracted concepts. Restricted to the same UMLS semantic groups as MEDCON (Anatomy, Chemicals & Drugs, Devices, Disorders, Genes & Molecular Sequences, Phenomena and Physiology). Handles synonyms and paraphrases that string matching misses.

**Implementation**: `eval/aci_bench/llm_medcon.py` — `LLMMedconExtractor` satisfies the `ConceptExtractor` protocol in `eval/aci_bench/extractors.py`, so it plugs into the existing `_score_medcon` / `compute_medcon_f1` path via `CONCEPT_EXTRACTOR=llm_medcon`.

**Prompt** (verbatim, single system message sent for each note):

> Extract all medical concepts from the following clinical note. Return ONLY a JSON list of normalized concept strings. Include: diagnoses, symptoms, medications, procedures, anatomical locations, lab values, vital signs. Restrict to: Anatomy, Chemicals & Drugs, Devices, Disorders, Genes & Molecular Sequences, Phenomena and Physiology.

Extractor calls `gpt-4o-mini` at temperature 0.0 with `response_format={"type": "json_object"}`. Output is parsed as either a bare list or a wrapped object (`concepts` / `items` / `list` / `data` key); strings are lowercased and whitespace-stripped before set operations. Unparseable responses are logged and treated as empty — one bad response cannot halt a 90-encounter run.

**Cost**: ~$0.001 per note pair at gpt-4o-mini pricing (~500 input tokens + ~100 output tokens per call, two calls per encounter). Full 90-encounter ACI-Bench run ≲ $0.20 end-to-end.

### MEDCON extractor tier selector

The `CONCEPT_EXTRACTOR` env var picks one of four backends (`eval/aci_bench/extractors.py :: build_extractor`):

| Tier          | `CONCEPT_EXTRACTOR=` | Status in this build                                    |
| ------------- | -------------------- | ------------------------------------------------------- |
| T0 QuickUMLS  | `quickumls`          | Needs UMLS licence + `scripts/install_umls.sh`.         |
| T1 scispaCy   | `scispacy` (default) | Needs scispaCy install (`en_core_sci_lg` + linker).     |
| T2 Null       | `null`               | Hard fallback — MEDCON column omitted on the slide.     |
| T-LLM         | `llm_medcon`         | **Shipped variant.** gpt-4o-mini. No UMLS licence needed. |

---

## Smoke harness

Deterministic first-10-case sanity pass used to validate infrastructure + catch wiring regressions before a full-benchmark run. See `eval/smoke/README.md` and `eval/smoke/run_smoke.py`.

- Run: `python eval/smoke/run_smoke.py --benchmark both --reader qwen2.5-14b --variant both --n 10 --budget-usd 50`
- Verdict per case: `PASS` / `ANOMALY` / `FAIL`. `ANOMALY` fires when a score sits outside ±20 pp of the reference baseline for that benchmark/reader (see `eval/smoke/reference_baselines.json`).
- ACI-Bench cases in the smoke harness score against whichever `CONCEPT_EXTRACTOR` is active; for the leaderboard-bound run, set `CONCEPT_EXTRACTOR=llm_medcon` and `OPENAI_API_KEY`.

---

## Model-usage policy

Per `Eng_doc.md §3.5`:

- **Demo path** (live extraction, next-best-question phrasing, demo SOAP note): Opus 4.7 (Anthropic API, sponsored-tool exempt under `rules.md §1.2`).
- **Eval loops**: Qwen2.5-14B primary + gpt-4o-mini / gpt-4.1-mini secondaries for published-comparator alignment. Opus 4.7 is **not** called in any eval loop.
- **LLM judges**: pinned to `gpt-4o-2024-08-06`.
- **LLM-MEDCON extractor**: gpt-4o-mini (temperature 0.0, JSON-object response format).

---

## What's intentionally out of scope

- Homemade metrics. Every score reported above is defined by the benchmark authors (or, for MEDCON, adapted via LLM extraction using the same UMLS semantic groups — see LLM-MEDCON section).
- Proprietary benchmark content (UpToDate, AMBOSS, AHA/ACC paywalled guidelines). LR-table citations are open-access; see `predicate_packs/clinical_general/differentials/chest_pain/sources.md`.
- Any metric computed over real patient data. All clinical benchmarks here use synthetic or published-benchmark data declared in `SYNTHETIC_DATA.md`.
