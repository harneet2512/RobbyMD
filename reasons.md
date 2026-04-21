# reasons.md — Rejected options with citations

Every decision *not* taken, with its reason and a source. Append-only; new rejections at the bottom under a date header. Agents append here whenever they reject an approach.

---

## 2026-04-21

### LongMemEval venue: "EMNLP 2024"
- **Rejected**: describing LongMemEval as EMNLP 2024 in planning docs.
- **Reason**: the arXiv preprint (Oct 2024) was accepted to **ICLR 2025**, not EMNLP. Rules.md §7.4 (no attribution hallucination) requires actual publication venue.
- **Citation**: [arXiv 2410.10813](https://arxiv.org/abs/2410.10813); [OpenReview ICLR 2025](https://openreview.net/forum?id=pZiyCaVuti).

### DDXPlus metrics: Top-1 / Top-3 / Top-5 as primary comparators
- **Rejected**: reporting Top-1 / 3 / 5 pathology accuracy as the DDXPlus primary metrics on the demo slide.
- **Reason**: H-DDx 2025 Table 2 — the 22-LLM comparator — reports only **Top-5 + HDF1**. Slide with empty Top-1/3 columns vs. populated Top-5 is misleading. Aligned to Top-5 + HDF1. Top-1/3 still computed internally for our own tracking.
- **Citation**: [H-DDx, arXiv 2510.03700](https://arxiv.org/abs/2510.03700), Table 2.

### ASR: Google MedASR
- **Rejected**: using Google MedASR as primary ASR.
- **Reason**: released under Gemma Terms of Use — not an OSI-approved licence. Rules.md §1.2 enforced by `tests/licensing/test_open_source.py`.
- **Citation**: [Gemma Terms of Use](https://ai.google.dev/gemma/terms).

### Local clinical LLM: MedGemma 1.5
- **Rejected**: using MedGemma 1.5 as offline fallback or claim extractor.
- **Reason**: Gemma Terms of Use; same non-OSI issue as MedASR. Replaced by BioMistral-7B (Apache-2.0) in Eng_doc.md §3.2.
- **Citation**: [MedGemma HF card](https://huggingface.co/google/medgemma-4b-it); [Gemma Terms](https://ai.google.dev/gemma/terms).

### ASR fallback: Deepgram Nova Medical
- **Rejected**: using Deepgram as emergency ASR fallback.
- **Reason**: commercial proprietary API, not OSI-approved. Rules.md §1.2.
- **Citation**: [Deepgram pricing page](https://deepgram.com/product/pricing).

### Code reuse: Groundtruth v2 memory substrate
- **Rejected**: copying any file from `D:\Groundtruth\src\groundtruth\memory\` into this repo.
- **Reason**: rules.md §1.1 — all code must be written during the hackathon window. Ideas allowed (captured in `docs/gt_v2_study_notes.md`); code copying not allowed.
- **Citation**: rules.md §1.1; Cerebral Valley hackathon rules ("All projects must be started from scratch during the hackathon with no previous work").

### Benchmark: LongMemEval-S TR + multi-session slice
- **Rejected**: running only the ~180-question TR + multi-session slice (originally recommended in `docs/research_brief.md §7-Q1` option B).
- **Reason**: user feedback memory `feedback_full_benchmarks.md`: "full 500 questions and add to memory half benchmarks don't count." A partial slice undermines prior-art comparison and looks like cherry-picking. Running the full 500.
- **Citation**: session memory 2026-04-21; rules.md §6.3.

### Benchmark: ACI-Bench `aci` subset only (virtscribe deferred)
- **Rejected**: reporting only `aci` subset for ACI-Bench with `virtscribe` as "stretch."
- **Reason**: same "no slicing" rule. Full test set = `aci` + `virtscribe` = 90 test encounters. Patched in PRD.md §8.3, Eng_doc.md §10.3, CLAUDE.md §5.5.
- **Citation**: session memory 2026-04-21; [ACI-Bench, Nature Sci Data 2023](https://www.nature.com/articles/s41597-023-02487-3).

### MEDCON: block on UMLS licence approval
- **Rejected**: waiting for UMLS licence approval (0–3 business days) before starting ACI-Bench work.
- **Reason**: hackathon deadline is 2026-04-26 20:00 EST; 3-day wait eats the build window. Replaced by 3-tier fallback (T0 QuickUMLS / T1 scispaCy / T2 ROUGE-only) per `docs/decisions/2026-04-21_medcon-tiered-fallback.md`. UMLS is now a hot-swap upgrade, never a blocker.
- **Citation**: internal decision; ADR link above.

### MEDCON replacement: UMLS REST API
- **Rejected**: using the UMLS REST API (https://uts-ws.nlm.nih.gov) in place of QuickUMLS for clinical-concept F1.
- **Reason**: (1) no span-detection endpoint — only resolves pre-extracted terms; (2) ~500k REST calls required per eval run (90 notes × 2 [gen + ref] × ~500 candidate spans × applicable splits); (3) unspecified rate limits; (4) the resulting metric would be "MEDCON-approx-via-REST", not MEDCON — reporting it as MEDCON violates rules.md §9.3 (no deceptive benchmark claims).
- **Citation**: [UMLS REST API docs](https://documentation.uts.nlm.nih.gov/rest/home.html); session analysis 2026-04-21.

### UMLS download: MRCONSO.RRF standalone
- **Rejected**: downloading only `MRCONSO.RRF` (472 MB) to minimise install time.
- **Reason**: MRCONSO carries names + codes but **not MRSTY** (semantic types). MEDCON's semantic-group filtering (Anatomy, Chemicals & Drugs, Device, Disorders, Genes & Molecular Sequences, Phenomena, Physiology) requires MRSTY. MRCONSO alone is insufficient. Smallest valid target = Level 0 Subset (1.8 GB compressed / 10.3 GB uncompressed).
- **Citation**: [UMLS Knowledge Sources page](https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html); ACI-Bench official evaluation script in `github.com/wyim/aci-bench/evaluation/`.
