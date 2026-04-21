# 2026-04-21 — MEDCON 3-tier fallback plan (UMLS-blocker mitigation)

**Status**: accepted
**Driver**: hack_it operator
**Affected**: `eval/aci_bench/**`, `Eng_doc.md §10.3`, `Eng_doc.md §12` risks, `rules.md §6.3` / §9.3

## Context

ACI-Bench's **MEDCON** metric (clinical-concept F1 over UMLS concepts restricted to 7 semantic groups) requires QuickUMLS + the UMLS Metathesaurus. UMLS licence approval takes **0–3 business days** from the National Library of Medicine; the hackathon deadline is **2026-04-26 20:00 EST**. Waiting on licence approval risks consuming the entire build window. User (2026-04-21) explicitly directed that eval quality must not be compromised by UMLS uncertainty.

## Decision

Implement MEDCON behind a **`ConceptExtractor` Protocol** with three concrete backends. Active backend selected by env var `CONCEPT_EXTRACTOR`. All eval code stays identical; swap is mechanical. Methodology disclosed at run time in `eval/aci_bench/LIMITATIONS.md`.

### Tier table

| Tier | Backend | Reported label on slide | UMLS licence needed |
|---|---|---|---|
| **T0** | `QuickUMLSExtractor` over UMLS **2025AB Level 0 Subset** (pinned) | `MEDCON (official, QuickUMLS)` — directly comparable to WangLab 2023 MEDIQA-CHAT Task B (57.78) | yes |
| **T1 (default)** | `ScispacyExtractor` — `en_core_sci_lg` + bundled UMLS linker (subset, AllenAI distribution) | `MEDCON-approx (scispaCy UMLS linker)` with explicit disclosure in `eval/aci_bench/LIMITATIONS.md` | no |
| **T2 (hard fallback)** | `NullExtractor` — no concept extraction | MEDCON column omitted; `eval/aci_bench/` reports ROUGE-1/2/L + BERTScore + BLEURT only, supplemented by **section-level ROUGE** (per SOAP section) + **MedEinst Bias Trap Rate** on a 30-case DDXPlus trap subset as clinical-rigor proxies | no |

### Report-Both rule

If both T0 and T1 complete within the deadline (e.g., licence approved by Day 3 and QuickUMLS index built), run both on the *same* 90-encounter ACI-Bench test set. Slide shows:

> MEDCON (official): *X* · MEDCON-approx (scispaCy): *Y* · Δ = |*X*−*Y*|/*X*

This converts approximation uncertainty into a **validated delta** — data evidence that the scispaCy approximation is trustworthy. Strictly more informative than either tier alone.

### UMLS release pinning

Pinned to **UMLS 2025AB** (released 2025-11-03) — specifically `umls-2025AB-Level0.zip`, 1.8 GB compressed / 10.3 GB uncompressed. If NLM publishes 2026AA before our eval, **do NOT auto-upgrade** — reproducibility matters (rules.md §6.3).

### Decision gates on the calendar

| When | Check | Action on yes | Action on no |
|---|---|---|---|
| **Day 1** (2026-04-21) | `./scripts/install_scispacy.sh` completes; smoke test extracts ≥10 CUIs from one ACI-Bench reference note | T1 locked as default; proceed | Fall to T2 tomorrow; reasons.md entry |
| **Day 2–3** | UMLS approval email received | Download `umls-2025AB-Level0.zip`, run `./scripts/install_umls.sh`, set `QUICKUMLS_PATH`. T0 activates on next run | Stay on T1 |
| **Day 4 morning** | T0 or T1 producing stable numbers on 10-note pilot | Proceed to full 90-encounter eval | Swap to T2; lock slide as "MEDCON omitted" with mitigation |
| **Day 4 evening** | Demo slide finalised | Ship what works | No "one more thing" — ship what works |

### What the `ConceptExtractor` interface looks like (spec; implementation in wt-eval)

```python
class ConceptExtractor(Protocol):
    name: str                               # "quickumls" | "scispacy" | "null"
    label: str                              # goes on slide + LIMITATIONS.md header
    semantic_groups: frozenset[str]         # the 7 MEDCON groups; Null returns empty
    def extract(self, text: str) -> set[str]: ...   # returns normalised concept IDs (CUIs)
```

One call per note; F1 computed over intersection/union of two extractions. Same adapter code across all three tiers.

## Alternatives considered

- **T0 only, block on licence**: risks missing deadline. Rejected; see `reasons.md` entry of 2026-04-21.
- **UMLS REST API in place of QuickUMLS**: no span detection; ~500k calls per eval run; unspecified rate limits; would produce a different metric that violates `rules.md §9.3` if labeled as MEDCON. Rejected; see `reasons.md`.
- **MRCONSO.RRF standalone** (no MRSTY) to minimise download: MRSTY is required for semantic-group filtering. Rejected; see `reasons.md`.
- **scispaCy alone, no T0 upgrade path**: foregoes Report-Both upside if licence arrives. Rejected.
- **Drop MEDCON entirely (skip straight to T2)**: loses direct comparator to WangLab 2023 (57.78) even when licence may well approve in time. Rejected as primary, kept as hard fallback.

## Consequences

- **What gets built (in `wt-eval`)**:
  - `eval/aci_bench/extractors.py` — `ConceptExtractor` Protocol + 3 implementations.
  - `eval/aci_bench/adapter.py` — reads `CONCEPT_EXTRACTOR` env var, picks backend, records `backend.name` in eval output JSON.
  - `eval/aci_bench/LIMITATIONS.md` — pre-written with 3 sections, one per tier; the eval run stamps the active tier at the top.
  - `scripts/install_scispacy.sh` — idempotent scispaCy + model install.
  - `scripts/install_umls.sh` — wrapper around `python -m quickumls.install`; takes path to extracted UMLS and builds index.
  - `.env.example` documents `CONCEPT_EXTRACTOR` + `QUICKUMLS_PATH`.

- **Eval quality invariants (preserved regardless of UMLS status)**:
  - DDXPlus (Top-5 + HDF1, 730-case H-DDx subset): 100% unaffected.
  - LongMemEval-S (all 500 questions, per-category): 100% unaffected.
  - ACI-Bench ROUGE-1/2/L + BERTScore + BLEURT: 100% unaffected.
  - ACI-Bench MEDCON: tier-dependent; worst case (T2) swaps in section-level ROUGE + Bias Trap Rate.

- **UMLS REST API still has a role** (independent of MEDCON): substrate-side live concept lookup (e.g., CUI hover in UI) using `UMLS_API_KEY`. Not on the eval critical path.

## References

- `docs/plans/foundation.md` / Part A — eval-quality protection
- `docs/research_brief.md §2.5, §3.3` — LongMemEval hard-category framing + ACI-Bench baselines
- `Eng_doc.md §10.3` — ACI-Bench spec with MEDCON blocker note
- `Eng_doc.md §12` — risks table including MEDCON/UMLS row
- `rules.md §6.1` (no homemade metrics) / §6.3 (methodology honesty) / §9.3 (no deceptive benchmark claims)
- [ACI-Bench GitHub](https://github.com/wyim/aci-bench)
- [scispaCy paper (Neumann et al., 2019)](https://aclanthology.org/W19-5034/)
- [UMLS Knowledge Sources — 2025AB release](https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html)
- [MedEinst Bias Trap Rate (arXiv 2601.06636)](https://arxiv.org/abs/2601.06636)
- [WangLab MEDIQA-CHAT 2023 GPT-4 ICL (arXiv 2305.02220)](https://arxiv.org/abs/2305.02220)
