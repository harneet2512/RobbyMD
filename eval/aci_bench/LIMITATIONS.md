# ACI-Bench harness — known limitations

Per `rules.md §6.3` (methodology honesty) and `§9.3` (no deceptive benchmark
claims). MEDCON specifics per
`docs/decisions/2026-04-21_medcon-tiered-fallback.md`.

**The active tier for a given run is stamped at the top of this file by
`run.py` at eval time** — the sections below describe all three tiers so the
template reads standalone before any run.

---

## T0 — QuickUMLS (official MEDCON)

**Backend**: `QuickUMLSExtractor` over UMLS 2025AB Level 0 Subset (pinned).
**Slide label**: `MEDCON (official, QuickUMLS)` — directly comparable to
WangLab 2023 MEDIQA-CHAT Task B (57.78).

### Caveats when active

1. **UMLS licence required** — end-user must hold a Metathesaurus licence (1–3
   business-day NIH approval). Licence details logged in
   `SYNTHETIC_DATA.md`-adjacent records.
2. **2025AB release pinning** — if NLM publishes 2026AA before our eval, we do
   NOT auto-upgrade; reproducibility over freshness.
3. **Threshold**: QuickUMLS similarity threshold 0.8 (default). Moving it
   requires an ADR + re-run of comparator numbers.
4. **Semantic-group filtering**: restricted to the 7 MEDCON groups
   (Anatomy, Chemicals & Drugs, Devices, Disorders, Living Beings, Phenomena,
   Procedures). Concepts outside these groups are intentionally dropped.
5. **Directly comparable to WangLab 2023** — we cite their 57.78 MEDCON on
   `aci` as comparator. Our 90-encounter run includes `virtscribe`, so the
   comparator number is on the `aci` subset within our run only.

---

## T1 — scispaCy (MEDCON-approx, default)

**Backend**: `ScispacyExtractor` — `en_core_sci_lg` + bundled UMLS linker.
**Slide label**: `MEDCON-approx (scispaCy UMLS linker)`.

### Caveats when active

1. **Approximation, not official MEDCON**. The scispaCy UMLS linker uses an
   AllenAI-distributed subset of UMLS and threshold-gated linking (default
   0.75). Numbers are NOT bit-for-bit comparable to WangLab 2023 (QuickUMLS).
2. **Report label is load-bearing**: slide must show `MEDCON-approx` — never
   `MEDCON` — per `rules.md §9.3` (no deceptive benchmark claims).
3. **Linker threshold**: 0.75 default. Shifting it without documenting
   shifts the distribution of matched CUIs; we do not tune this threshold
   on the test set.
4. **Report-Both upside**: if T0 also completes (licence arrives mid-week),
   `run.py` can run both against the same 90 encounters and report
   `Δ = |T0 - T1| / T0` — a **validated delta** that establishes
   trustworthiness of the approximation. Strictly more informative than
   either tier alone. Per ADR.
5. **scispaCy UMLS KB attribution**: the AllenAI UMLS KB bundled with
   `en_core_sci_lg` is NOT listed in `MODEL_ATTRIBUTIONS.md` because the
   licensing test (`tests/licensing/test_model_attributions.py`) scans `src/`
   only and scispaCy is loaded exclusively from `eval/`. If the substrate
   ever loads scispaCy from `src/`, the attribution row must be added before
   that commit merges.

---

## T2 — NullExtractor (hard fallback, MEDCON omitted)

**Backend**: `NullExtractor` — returns empty set.
**Slide label**: `MEDCON omitted — ROUGE + BERTScore + supplementary metrics only`.

### Caveats when active

1. **MEDCON column omitted** from the slide. No MEDCON number is reported —
   NOT "MEDCON ≈ 0" or any placeholder. Absence of the metric is reported as
   absence.
2. **Supplementary clinical-rigor proxies** kick in:
   - **Section-level ROUGE** — per-SOAP-section (S/O/A/P) ROUGE to catch
     content loss in a specific section even if whole-note ROUGE is stable.
   - **MedEinst Bias Trap Rate** on a 30-case MedEinst trap-pair subset
     ([arXiv 2601.06636](https://arxiv.org/abs/2601.06636)) — measures
     discriminative-evidence sensitivity. Not a MEDCON replacement; a
     different angle on clinical correctness. (Originally planned on a
     DDXPlus trap subset; DDXPlus dropped 2026-04-21, so trap cases come
     directly from MedEinst's published pair set.)
3. **Honest framing**: T2 is the "UMLS didn't land AND scispaCy install
   failed" path. The demo slide documents that MEDCON was omitted; it does
   not pretend we have a MEDCON number.

---

## Cross-tier caveats (apply regardless of tier)

### A. Full 90-encounter test set only

Per `memory/feedback_full_benchmarks.md`: we run aci test1/test2/test3
(66 encounters) + virtscribe test (24 encounters) = 90 total. Any run with
`--limit` < 90 is dev smoke; `metrics.json.is_slice == true`; those numbers
are NOT shown on the demo slide.

### B. BERTScore is division-based

ACI-Bench notes exceed BERT's 512-token context. Per the official ACI-Bench
eval script, BERTScore is computed on note divisions (SOAP sections), not
whole notes. Whole-note BERTScore is invalid (Eng_doc.md §10.3). We use the
official evaluator, which handles this correctly.

### C. Judge-free MEDCON

MEDCON is set-intersection F1 over extracted CUIs — no LLM judge. Numbers are
deterministic given the same active tier + same predictions.

### D. ASR vs gold dialogue

ACI-Bench ships both gold and ASR dialogue variants. We run on **gold**
dialogue by default — isolates the note-generation failure mode from ASR
noise. Running on ASR is possible via a CLI flag (future work).

### E. Provenance in full variant

The `full` variant's provenance-validation step rejects any generated
sentence without `source_claim_ids`. This is demonstrably stricter than the
baseline Opus-direct path; expect slightly SHORTER notes and slightly LOWER
ROUGE-recall if provenance is enforced. This is the correct-by-construction
tradeoff (rules.md §4.2), not a bug.
