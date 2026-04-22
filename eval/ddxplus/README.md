# DDXPlus benchmark harness

Differential-diagnosis evaluation on the H-DDx 2025 stratified subset of
DDXPlus (Fansi Tchango et al., NeurIPS 2022).

## Dataset

| Field | Value |
|---|---|
| Upstream repo | https://github.com/mila-iqia/ddxplus |
| Upstream paper | Fansi Tchango et al., *DDXPlus: A New Dataset For Automatic Medical Diagnosis*, NeurIPS 2022 ([arXiv 2205.09148](https://arxiv.org/abs/2205.09148)) |
| License | CC-BY 4.0 — redistribution + commercial use with attribution |
| Commit SHA pin | **TODO** — set on first `fetch.py` run; cached to `eval/ddxplus/data/.commit_sha` |
| Subset | H-DDx 2025 stratified 730-case test subset ([arXiv 2510.03700](https://arxiv.org/abs/2510.03700), §3.2) |
| Pathologies | 49 (full DDXPlus pathology universe, stratified) |
| Slicing note | H-DDx's 730-case stratified subset **is** the canonical methodology we compare against. This is NOT further slicing per `feedback_full_benchmarks.md`. |

## Methodology

Per H-DDx 2025 (arXiv 2510.03700, Table 2 publishes Top-5 + HDF1 for 22 LLMs).

- **Top-5 accuracy**: LLM judge for semantic-equivalence matching between predicted
  pathologies and DDXPlus's `DIFFERENTIAL_DIAGNOSIS` field. Judge pinned to
  `gpt-4o-2024-08-06` (env: `DDXPLUS_JUDGE_MODEL`) — reproducibility over freshness.
- **HDF1** (ICD-10 hierarchical F1): deterministic — ICD-10 retrieval+rerank of
  predicted pathologies, compared to gold ICD-10 chains. No LLM on this metric.
- **Top-1 / Top-3** are computed internally (for our own tracking) but not reported
  against H-DDx comparators, because H-DDx publishes Top-5 + HDF1 only.

See `LIMITATIONS.md` for caveats (adapter fidelity, judge pinning, subset size).

## Adapter

`adapter.py` converts DDXPlus's record format

```json
{
  "AGE": 42, "SEX": "F",
  "PATHOLOGY": "...",
  "EVIDENCES": ["E_123_@_V_1", "E_456", ...],
  "DIFFERENTIAL_DIAGNOSIS": [["Pathology A", 0.55], ...]
}
```

into a natural-dialogue turn stream (patient + physician utterances) that
the substrate ingests via the write API. Conversion is deterministic —
same record in, same turns out.

## Variants

- `baseline.py` — Opus 4.7 directly prompted with the full case text.
- `full.py` — substrate + differential engine + verifier; matches Eng_doc.md §10.1.
  Currently has `TODO(wt-engine)` marker pending the write API publish.

## Run

```bash
# From repo root, after ./scripts/setup.sh has populated eval/ddxplus/data/
python -m eval.ddxplus.run --variant baseline   # Opus direct baseline
python -m eval.ddxplus.run --variant full       # substrate + engine
python -m eval.ddxplus.run --dry-run            # 5-case smoke, no API calls
```

Reports to `eval/reports/<timestamp>/ddxplus/`.

## Comparator table (H-DDx 2025 Table 2)

From [H-DDx Table 2](https://arxiv.org/html/2510.03700v1) — 730-case stratified subset,
Top-5 accuracy + HDF1. Pre-populated here from `docs/research_brief.md §3.1` so this
README reads standalone for judging. **We compare against these numbers; we do NOT
redistribute H-DDx's Table 2 beyond the selected comparators below.**

| Model | Top-5 | HDF1 | Category |
|---|---:|---:|---|
| Claude-Sonnet-4 | **0.839** | **0.367** | proprietary |
| Claude-Sonnet-3.7 | 0.836 | 0.338 | proprietary |
| Gemini-2.5-Flash | 0.832 | 0.348 | proprietary |
| GPT-4o | 0.804 | 0.350 | proprietary |
| GPT-4.1 | 0.801 | 0.339 | proprietary |
| GPT-5 | 0.783 | 0.345 | proprietary |
| Qwen3-235B-A22B | 0.777 | 0.322 | open |
| MedGemma-27B | 0.765 | 0.331 | medical FT |
| MediPhi | 0.666 | **0.353** | medical FT |
| **Opus 4.7 (us)** | TBD | TBD | proprietary (baseline) |
| **Opus 4.7 + substrate (us)** | TBD | TBD | ours |

**No published Opus 4.7 number on H-DDx.** Our baseline row would be the first.

## Citation (required per rules.md §7.3)

- Fansi Tchango et al., *DDXPlus: A New Dataset For Automatic Medical Diagnosis*, NeurIPS 2022.
- H-DDx 2025 methodology: [arXiv 2510.03700](https://arxiv.org/abs/2510.03700).
