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

### Diariser: pyannote `speaker-diarization-3.1`
- **Rejected**: using the `pyannote/speaker-diarization-3.1` Hugging Face model as the WhisperX diariser backend.
- **Reason**: the 3.1 checkpoint is gated behind pyannote's HF user-terms acceptance flow and is not distributed under an OSI-compatible licence. WhisperX 3.8.5 now defaults to `pyannote/speaker-diarization-community-1` (CC-BY-4.0), which is the correct target — though CC-BY-4.0 itself requires an ADR to be added to the `rules.md` §1.2 model-weights allowlist. See `research/asr_stack.md` §R1.
- **Citation**: [WhisperX README (v3.8.5)](https://github.com/m-bain/whisperX); [pyannote/speaker-diarization-community-1 model card](https://huggingface.co/pyannote/speaker-diarization-community-1).

### ASR inference engine: `whisper.cpp` as primary
- **Rejected**: using `whisper.cpp` (MIT) as the primary inference engine in place of `faster-whisper` / CTranslate2.
- **Reason**: `whisper.cpp` is optimised for CPU-only / Apple-Silicon laptop deployments (OpenWhispr uses it for exactly that reason [OpenWhispr README](https://github.com/openwhispr/openwhispr)). Our deployment target per `Eng_doc.md` §9 is a 16–24 GB NVIDIA GPU workstation, where CTranslate2 INT8_FLOAT16 is ~2.3× faster than openai-whisper on large-v2 GPU inference ([SYSTRAN benchmarks](https://github.com/SYSTRAN/faster-whisper)). Keeping `whisper.cpp` on the shelf as an offline-only rehearsal fallback is acceptable; it is not primary.
- **Citation**: [faster-whisper README benchmarks](https://github.com/SYSTRAN/faster-whisper); [whisper.cpp README](https://github.com/ggerganov/whisper.cpp).

### Integration: copying code from OpenWhispr
- **Rejected**: lifting audio-ring-buffer / hotkey-capture / streaming-chunker code from `openwhispr/openwhispr` (MIT).
- **Reason**: even MIT-licensed code is disallowed per `rules.md` §1.1 — all code must be written during the hackathon window. OpenWhispr is a reference for the "separate VAD process + ring buffer + ASR worker" architectural pattern only, not a code source. Pattern is generic and does not need to be copied.
- **Citation**: [`rules.md` §1.1](../rules.md); [OpenWhispr repo](https://github.com/openwhispr/openwhispr).

### ASR prompt length: `initial_prompt` >224 tokens
- **Rejected**: using `initial_prompt` strings longer than ~224 tokens with faster-whisper 1.2.1.
- **Reason**: Whisper's decoder context is 448 tokens total. `faster-whisper.transcribe` raises ValueError if `prompt_tokens + max_new_tokens > 448` ([implementation](https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/transcribe.py)). A 5 s utterance at normal speech rate generates ~60–150 tokens; we reserve half the budget (~224 tokens) for the prompt to guarantee no truncation of output. Anecdotal community evidence also suggests biasing returns diminish beyond ~200 tokens.
- **Citation**: [faster-whisper transcribe.py](https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/transcribe.py); Whisper paper, [arXiv 2212.04356](https://arxiv.org/abs/2212.04356).

### LR table: pneumothorax individual-finding LRs
- **Rejected**: including pneumothorax physical-exam findings (tachycardia, absent breath sounds, hyperresonance) as rows in the chest-pain LR table.
- **Reason**: systematic search (WebSearch 2026-04-21) returned no peer-reviewed pooled LR+ / LR- for these individual pneumothorax clinical findings. All reported sensitivities and specificities are either for imaging (ultrasound 94% sensitivity / 100% specificity — falls under `rules.md` §3.2 forbidden imaging inputs) or narrative clinical descriptions without pooled LR statistics. Inclusion would violate `rules.md` §4.4 (every LR cited) and §7.4 (no attribution hallucination).
- **Citation**: WebSearch queries 2026-04-21 against StatPearls, Medscape, and PubMed for "pneumothorax likelihood ratio"; `rules.md` §3.2, §4.4, §7.4.

### LR table: paywalled-source verbatim paraphrase
- **Rejected**: paraphrasing specific wording from Panju 1998 JAMA (AMI rational clinical examination), Bruyninckx 2008 BJGP (chest-wall rule primary), and ACG 2022 GERD guideline full-text.
- **Reason**: all three sit behind institutional paywalls. Per `rules.md` §7.1, verbatim or close-paraphrase reproduction of paywalled content is not permitted. Pooled LR values from these sources are cited via the open-access secondary summaries (AAFP 2013, AAFP 2020, fanaroff_jama_2015 which re-pools Panju 1998 values for many ACS features). Where the secondary source doesn't carry the primary value, the row is flagged `approximation: true`.
- **Citation**: `rules.md` §7.1; [AAFP 2013 Table 1](https://www.aafp.org/pubs/afp/issues/2013/0201/p177.html); [Fanaroff 2015 JAMA abstract](https://jamanetwork.com/journals/jama/article-abstract/2468896).

### LR table: Ohle 2018 aortic-dissection LR revision
- **Rejected**: replacing the Klompas 2002 JAMA pulse-deficit LR+ 5.7 with the Ohle 2018 Academic Emergency Medicine revised LR+ 2.48.
- **Reason**: Klompas 2002 is the established JAMA Rational Clinical Examination pooled estimate and the value cited by AAFP 2013 (chest/back pain + pulse differential LR+ 5.3, similar magnitude). Ohle 2018 is a more recent meta-analysis but is behind a Wiley paywall (WebFetch returned 403). Until the human reviewer can access and approve the Ohle numbers, keeping the Klompas value is the conservative rules-compliant choice. Flagged as an open question in `research/clinical_chest_pain.md` §8-Q4.
- **Citation**: [Klompas 2002 JAMA abstract](https://pubmed.ncbi.nlm.nih.gov/11980527/); [Ohle 2018 ACEM abstract](https://onlinelibrary.wiley.com/doi/10.1111/acem.13360) (paywalled).

### Citation practice: DOI-as-citation without resolver verification
- **Rejected**: including a DOI in `sources.md` without end-to-end verifying that the DOI resolver actually lands on the named paper.
- **Reason**: Validator 2026-04-21 surfaced that `sources.md` `cremonini_2005_meta` carried DOI `10.1111/j.1365-2036.2005.02435.x`, which resolves to van Kerkhoven 2005 (anxiety/depression in GI endoscopy, *Aliment Pharmacol Ther*) — an unrelated paper. Actual Cremonini 2005 is in *Am J Gastroenterol* with DOI `10.1111/j.1572-0241.2005.41657.x` (PMID 15929749). The pattern — plausible-looking DOI from the same DOI prefix family, never resolver-checked — is the exact failure mode `rules.md` §7.4 exists to prevent. Going forward, every citation added to `sources.md` must have its DOI resolved once before commit.
- **Citation**: [Cremonini 2005 actual PMID](https://pubmed.ncbi.nlm.nih.gov/15929749/); [van Kerkhoven 2005 actual paper at the cited DOI](https://onlinelibrary.wiley.com/doi/10.1111/j.1365-2036.2005.02435.x); `rules.md §7.4`; `research/validation_report.md` blocker #1.

### Citation practice: PMC-ID as proxy for derivation paper without content check
- **Rejected**: using a PMC full-text URL as the `source_url` for an LR row without confirming the PMC full text is the paper the row is citing.
- **Reason**: Validator 2026-04-21 surfaced that three `lr_table.json` rows (`pain_reproducible_with_palpation`, `younger_age_lt_40`, `no_exertional_pattern`) cite "bosner_2010_marburg" but their `source_url` points to [PMC4617269](https://pmc.ncbi.nlm.nih.gov/articles/PMC4617269/), which is Haasenritter et al. 2015 *BJGP* — an MHS-adjacent paper but not the 2010 CMAJ derivation. Either the URL must be changed to the correct Bösner 2010 CMAJ DOI (`10.1503/cmaj.100212`) or a new `haasenritter_2015_bjgp` source-key added to `sources.md`. Same failure mode as the Cremonini DOI swap: cite-looks-plausible, never WebFetched.
- **Citation**: [Haasenritter 2015 BJGP at PMC4617269](https://pmc.ncbi.nlm.nih.gov/articles/PMC4617269/); [Bösner 2010 CMAJ DOI](https://doi.org/10.1503/cmaj.100212); `rules.md §7.4`; `research/validation_report.md` blocker #3.
