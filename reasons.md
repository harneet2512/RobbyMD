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

### DDXPlus — dropped for substrate-benchmark misalignment (2026-04-21)

- **Context**: benchmark selection.
- **What it is**: 1.3M synthetic patients across 49 respiratory pathologies (URTI, bronchitis, pneumonia, TB, influenza, HIV, Chagas).
- **Why it was considered**: originally picked as a differential-diagnosis benchmark to exercise the parallel-hypothesis-trees + LR-weighted update + counterfactual verifier stack.
- **Why it lost**: the 49-pathology respiratory set doesn't map to our `clinical_general` pack's seeded chest-pain differential. Running DDXPlus would require either seeding a full respiratory-pathologies differential pack (significant clinical-content work, not in scope) or reporting a meaningless number against a non-matching LR table. Neither option is defensible as a substrate-contribution claim.
- **Citation**: Tchango et al., *DDXPlus*, NeurIPS 2022 — https://arxiv.org/abs/2205.09148.
- **Superseded by**: LongMemEval-S (directly tests memory substrate — lifecycle + supersession + projection) and ACI-Bench (directly tests extraction + SOAP-note pipeline). Both load an already-seeded pack (`personal_assistant` and `clinical_general` respectively); neither requires additional content work to run.
- **Revisit trigger**: if a respiratory-pathologies differential pack ships post-hackathon — at that point DDXPlus becomes a direct test of a second seeded pack and is worth including.

### MedQA — dropped for testing reader knowledge, not substrate architecture (2026-04-21)

- **Context**: benchmark selection.
- **What it is**: USMLE-style multiple-choice medical licensing questions; 1,273 test questions (Jin et al. 2021; HuggingFace Open Medical-LLM Leaderboard).
- **Why it was considered**: medical-reasoning axis — adds a breadth claim alongside the two substrate-focused benchmarks.
- **Why it lost**: MedQA scores track the reader model's pre-trained medical knowledge, not substrate contribution. Our substrate has no meaningful surface on a closed-form MCQ — the claim store and supersession edges don't participate. Running MedQA reports which reader we picked, not whether our architecture helps. Including it would dilute the eval slide's signal and invite the apples-to-apples kill discussed in `Eng_doc.md §3.5`.
- **Citation**: Jin et al., *MedQA*, Applied Sciences 2021; [HuggingFace Open Medical-LLM Leaderboard](https://huggingface.co/blog/leaderboard-medicalllm).
- **Superseded by**: LongMemEval-S (memory-architecture contribution is measurable) and ACI-Bench (extraction + SOAP-note-generation contribution is measurable). Both have substrate-visible surfaces.
- **Revisit trigger**: never for this hackathon. Post-hackathon, MedQA becomes relevant if we build a retrieval-augmented answering layer that actively uses stored claims to justify MCQ picks — but that's a different product, not this substrate.

### Tank for the war, not the gun fight — Opus 4.7 scoped to demo-path only (2026-04-21)

- **Context**: Principle 1 of the 2026-04-21 user directive. Opus 4.7 is a capable sponsored tool, but it IS costly and using it everywhere muddies benchmark comparisons.
- **What was considered but rejected**: using Opus 4.7 as the reader for every benchmark (DDXPlus, LongMemEval-S, ACI-Bench, MedQA). Easier to implement (one client, one code path), more impressive-sounding ("we use Opus 4.7 everywhere").
- **Why it lost**: using Opus 4.7 as the eval reader when the published SOTA used `gpt-4.1-mini` or `gpt-5-mini` **kills the apples-to-apples comparison**. A Mastra OM leaderboard row of 94.87% is driven by `gpt-5-mini`; if our substrate variant runs Opus 4.7 as reader and scores e.g. 89%, no one can tell whether we lost because our substrate underperformed or because `gpt-5-mini` beats Opus 4.7 on multi-choice QA. Benchmark comparability IS the eval slide's entire value; without it, the numbers are marketing not measurement.
- **What we did instead**: `Eng_doc.md §3.5` locks the model-usage policy. Opus 4.7 for demo-path calls only (live claim extraction, verifier's next-best-question phrasing, SOAP note composition). For eval loops: match each benchmark's published SOTA reader. For bulk extraction and infrastructure: the cheapest model that preserves the comparison. `eval/README.md` enumerates per-benchmark reader + judge models with citations to the paper/leaderboard that set the precedent. `tests/licensing/test_model_attributions.py` is the OSI gate; benchmark-reader-pin gate is enforced socially by the table in `eval/README.md`.
- **Citations**:
  - User directive 2026-04-21 (Principle 1 verbatim).
  - `Eng_doc.md §3.5` — model-usage policy table.
  - `eval/README.md` — per-benchmark reader + judge pins.
  - [H-DDx 2025 Table 2 (arXiv 2510.03700)](https://arxiv.org/abs/2510.03700) — DDXPlus reader precedent (GPT-4o).
  - [Mastra Observational Memory 94.87%](https://mastra.ai/research/observational-memory) — LongMemEval-S reader precedent (gpt-5-mini).
  - [WangLab MEDIQA-CHAT 2023 GPT-4 ICL (arXiv 2305.02220)](https://arxiv.org/abs/2305.02220) — ACI-Bench reader precedent.
- **Revisit trigger**: a benchmark publishes new SOTA where Opus 4.7 IS the published reader — then use Opus 4.7 for apples-to-apples. Or a demo-path call site shifts to eval-path (unlikely; the demo video is structurally separate from the eval slide).

### Scope widened under the hood; demo narrative stays clinical (2026-04-21)

- **Context**: Principles 2 + 3 of the 2026-04-21 user directive. (2) The substrate must work on any chief complaint — abdominal pain, dyspnoea, headache, whatever a judge throws at it. (3) The 3-minute demo video tells one story: live clinical reasoning for the exam room.
- **What was considered but rejected**: (a) hardcode chest-pain assumptions throughout the engine (faster to ship; breaks the moment a judge opens the app and says "right upper quadrant pain" — exactly the Stage 2 repo-reader failure mode we want to avoid); (b) market the substrate as a general-purpose memory layer in the demo video (dilutes the clinical pitch; a 3-minute viewer leaves with no clear product memory; judges watching the video don't remember a tool they couldn't place).
- **What we did instead**: engine is domain-agnostic by construction — predicate families are data in pluggable `PredicatePack`s (`Eng_doc.md §4.2`), LR-table schema accepts multiple complaint branches, `clinical_general` is the one pack seeded this build. The file tree (`predicate_packs/clinical_general/differentials/{chest_pain,abdominal_pain,dyspnoea,headache}/`) makes the generality visible to any repo-reading judge without a word of marketing. The demo video still opens with clinical framing (`PRD.md §2, §9`; `context.md §8.3`). Both audiences (30-sec video-watcher and repo-reading Anthropic engineer) get the right signal.
- **Citations**:
  - User directive 2026-04-21 (Principles 1 / 2 / 3).
  - `Eng_doc.md §4.2` — pluggable predicate packs (amended).
  - `context.md §9` — scope for demo vs README vision.
  - `README.md` — two-part structure (clinical product top, architecture bottom).
  - `predicate_packs/clinical_general/README.md` — shipped pack spec.
  - `docs/decisions/2026-04-21_lr-table-chest-pain-coupling-audit.md` — remaining hardcodings found; next-turn fixes.
- **Revisit trigger**: demo narrative drift (PR proposes adding "general memory layer" framing to the video script or written summary) or engine-generality drift (PR hardcodes a new chest-pain assumption in substrate or differential code paths).

### Strict OSI-only licensing for model weights — rejected for industry-standard reading (2026-04-21)

- **Context**: License policy for model weights.
- **What it is**: Require every model weight in the repo to carry an OSI-approved license (MIT / BSD / Apache / etc.), same as code.
- **Why it was considered**: Simplest reading of the hackathon's "approved open source license" rule. Originally locked into `rules.md §1.2` as defensive over-interpretation.
- **Why it lost**: OSI itself does not apply its open-source definition to model weights — weights are classified as data, not software. The Linux Foundation's OpenMDW framework (July 2025) explicitly recommends CC-BY-4.0 and CDLA-Permissive-2.0 for model weights and datasets while reserving OSI licenses for code. OSI's own website uses CC-BY-4.0 for its content. The vast majority of open-weight ML releases (pyannote, many HF speech models, biomedical models) use CC-BY-4.0 for weights. A strict OSI-only reading for weights is stricter than the hackathon rule requires and stricter than the ML-licensing community treats the distinction. Applied to our stack, it would unnecessarily block `pyannote/speaker-diarization-community-1` — a clean, well-attributed, widely-used model — forcing either a week-delaying swap to NVIDIA NeMo Sortformer or dropping diarisation entirely.
- **Citations**:
  - Linux Foundation / LF AI & Data, *Simplifying AI Model Licensing with OpenMDW* (July 2025): https://lfaidata.foundation/blog/2025/07/22/simplifying-ai-model-licensing-with-openmdw/
  - OSI FAQ on license categorisation: https://opensource.org/faq
  - OpenMDW-1.0 license: https://lfaidata.foundation/projects/openmdw/
  - Model Openness Framework (MOF): https://isitopen.ai/
- **Superseded decision**: `rules.md §1.2` now permits OSI-approved licenses for code and open-data licenses (CC-BY-4.0, CC-BY-SA-4.0, CDLA-Permissive-2.0, ODbL) for model weights and datasets. `MODEL_ATTRIBUTIONS.md` enforces attribution. `tests/licensing/test_model_attributions.py` enforces that attribution in CI.
- **Revisit trigger**: Hackathon Discord responds to the licensing question (pending, see `docs/decisions/licensing_clarifications.md` Q2) with a stricter interpretation. If so, swap diariser to NeMo Sortformer (Apache-2.0) and eat the latency re-benchmark cost.

## 2026-04-22

### LR citation swap: `wells_pe_low_probability` → ceriani_2010_jth (2026-04-22)

- **Scope extension of the 2026-04-21 open-access-replacement pass**: the `wells_pe_high_probability` row was swapped to `ceriani_2010_jth_wells_meta` on 2026-04-21 (noted as ambiguity (a) in that session's log); the `wells_pe_low_probability` row still cited `bmc_pulm_2025`. This is the final citation swap to achieve open-access uniformity.
- **Swap reason**: open-access verification uniformity; Ceriani 2010 J Thromb Haemost is the same meta-analysis source already cited for `wells_pe_high_probability`. The `bmc_pulm_2025` source key is now fully unreferenced in `lr_table.json` and has been removed from `sources.md`.
- **LR value unchanged**: LR- 0.34 retained — same meta-analytic estimate, different attribution vehicle.
- **Citation**: Ceriani E et al. *Clinical prediction rules for pulmonary embolism: a systematic review and meta-analysis.* J Thromb Haemost 8(5):957–970, 2010. https://www.jthjournal.org/article/S1538-7836(22)12404-9/pdf
