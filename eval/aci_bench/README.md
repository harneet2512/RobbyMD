# ACI-Bench benchmark harness

Conversation → clinical note evaluation (Yim et al., Nature Sci Data 2023;
MEDIQA-CHAT 2023 shared-task dataset).

## Dataset

| Field | Value |
|---|---|
| Upstream repo | https://github.com/wyim/aci-bench |
| Upstream paper | Yim et al., *ACI-BENCH: a Novel Ambient Clinical Intelligence Dataset for Benchmarking Automatic Visit Note Generation*, Nature Sci Data 2023 ([link](https://www.nature.com/articles/s41597-023-02487-3)) |
| License | CC-BY 4.0 — redistribution + commercial use with attribution |
| Commit SHA pin | **TODO** — set on first `fetch.py` run; cached to `eval/aci_bench/data/.commit_sha` |
| Splits used | `aci` test (66 encounters across test1/test2/test3) + `virtscribe` test (24 encounters) = **90 total** |
| Slicing note | We run the **full** 90-encounter test set per `memory/feedback_full_benchmarks.md`. Smaller slices are DEV ONLY and not reported. |

## Methodology

Per MEDIQA-CHAT 2023 evaluation protocol:

- **ROUGE-1 / ROUGE-2 / ROUGE-L** — standard automatic summarisation metrics.
- **BERTScore** — Division-based per official ACI-Bench eval script (full notes
  exceed BERT's embedding length; whole-note BERTScore is invalid; see
  `Eng_doc.md §10.3`).
- **MEDCON** — clinical-concept F1 restricted to 7 UMLS semantic groups. See
  "MEDCON tier active" below.
- **BLEURT** — supplementary rater; used in T2 fallback.

The MEDIQA-CHAT evaluation scripts are imported at run time from the upstream
repo (`evaluation/*.py`); we do NOT fork or reimplement them
(`rules.md §6.1`).

## MEDCON tier active

MEDCON requires a UMLS Terminology Services licence. To decouple eval quality
from UMLS approval timing, we implement MEDCON behind a `ConceptExtractor`
Protocol with three swappable backends. See
`docs/decisions/2026-04-21_medcon-tiered-fallback.md` for the decision
record. **Active backend is logged at run time; `LIMITATIONS.md` stamps the
active tier at the top.**

| Tier | Backend | Slide label | UMLS licence |
|---|---|---|---|
| **T0** | `QuickUMLSExtractor` over UMLS 2025AB Level 0 Subset | `MEDCON (official, QuickUMLS)` — directly comparable to WangLab 2023 | yes |
| **T1 (default)** | `ScispacyExtractor` — `en_core_sci_lg` + bundled UMLS linker | `MEDCON-approx (scispaCy UMLS linker)` | no |
| **T2 (hard fallback)** | `NullExtractor` — no concept extraction | MEDCON column omitted; supplemented with section-level ROUGE + Bias Trap Rate | no |

Selection is via `CONCEPT_EXTRACTOR` env var (default `scispacy`).

### Report-Both rule

If both T0 and T1 complete within the deadline, the slide shows:

> MEDCON (official): X · MEDCON-approx (scispaCy): Y · Δ = |X-Y|/X

Validated approximation beats either tier alone.

## Adapter

`adapter.py` reads ACI-Bench's dialogue JSON (gold + ASR variants), runs our
extraction → substrate → note generator, and compares the generated note to
the gold note.

## Variants

- `baseline.py` — Opus 4.7 given the dialogue, directly generates a SOAP note.
- `full.py` — dialogue → substrate → provenance-validated note generator.
  Currently has `TODO(wt-engine)` marker pending wt-engine + wt-trees.

## Run

```bash
# Day 1: T1 default (no UMLS needed)
export CONCEPT_EXTRACTOR=scispacy
python -m eval.aci_bench.run --variant baseline

# Day 3+ (UMLS licence approved):
./scripts/install_umls.sh /path/to/2025AB/META
export CONCEPT_EXTRACTOR=quickumls
export QUICKUMLS_PATH=...
python -m eval.aci_bench.run --variant full

# Worst case:
export CONCEPT_EXTRACTOR=null     # T2 fallback
```

Reports to `eval/reports/<timestamp>/aci_bench/`.

## Comparator table (MEDIQA-CHAT 2023 Task B)

From `docs/research_brief.md §3.3` — `aci` test split, Task B (note generation).
Best published baseline: WangLab GPT-4 ICL (1st place MEDIQA-CHAT 2023).
No 2024–2026 LLM has published ACI-Bench numbers (AMIE, Med42-v2, MedGemma,
Opus, GPT-5, Gemini 2.5 all absent — verified in research_brief).

| Metric | BART + FT-SAMSum (Division) | GPT-4 zero-shot | WangLab GPT-4 ICL | **Opus 4.7 (us, baseline)** | **Opus 4.7 + substrate (us)** |
|---|---:|---:|---:|---:|---:|
| ROUGE-1 | **53.46** | 51.76 | (higher, see paper) | TBD | TBD |
| ROUGE-2 | **25.08** | 22.58 | (higher, see paper) | TBD | TBD |
| ROUGE-L | **48.62** | 45.97 | (higher, see paper) | TBD | TBD |
| MEDCON | — | **57.78** | (higher, see paper) | TBD (tier stamped) | TBD (tier stamped) |

Source: [PMC10482860](https://pmc.ncbi.nlm.nih.gov/articles/PMC10482860/);
WangLab MEDIQA-CHAT 2023: [arXiv 2305.02220](https://arxiv.org/abs/2305.02220).

## Smoke test (Day 1 gate)

`eval/aci_bench/smoke_test.py` loads the active extractor and extracts CUIs
from a one-sentence ACI-Bench reference note. Must extract **≥10 CUIs** on
T1 to lock T1 as default; otherwise fall to T2 tomorrow.

```bash
python eval/aci_bench/smoke_test.py
```

## Citation (required per rules.md §7.3)

- Yim et al., *ACI-BENCH*, Nature Sci Data 2023.
- MEDIQA-CHAT 2023 evaluation: [ACL Anthology overview](https://aclanthology.org/volumes/2023.clinicalnlp-1/).
- WangLab GPT-4 ICL (1st place MEDIQA-CHAT 2023): [arXiv 2305.02220](https://arxiv.org/abs/2305.02220).
