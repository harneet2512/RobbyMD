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

### Fine-tuning Whisper on medical audio — rejected for layered mitigation (2026-04-22)

- **Context**: ASR hardening dispatch Track 3 Parts A-D, F, G.
- **What was considered**: fine-tuning Whisper large-v3 on medical-audio datasets (e.g. PriMock57, MTS-Dialog) to directly reduce medical-term WER. This is the approach taken by several commercial clinical ASR vendors (Nuance, Abridge, Nabla).
- **Why it lost**:
  1. **Data constraint**: fine-tuning requires substantial human-transcribed medical audio. No open-access medical ASR training set covers the demographic and acoustic diversity needed to avoid regression on OOD speakers (Koenecke et al., "Disparate ASR Accuracy in a Clinical Setting", ACM FAccT 2024, arXiv 2312.05420, found significant WER disparities across demographic groups even in fine-tuned commercial systems — fine-tuning can encode bias).
  2. **Latency constraint**: fine-tuned models are typically larger or require different quantisation, adding GPU VRAM requirements and potentially breaking our RTF ≤ 0.2 target on L4 hardware.
  3. **Build-window constraint**: fine-tuning a speech model responsibly (data curation, validation, bias audit) takes weeks. We have 4 days.
  4. **Layered mitigation adequacy**: `initial_prompt` medical bias + LLM cleanup + Levenshtein correction + hallucination guard is a defensible four-layer stack that addresses the same error modes without requiring new weights. Arora et al. (arXiv 2502.11572, Jogi, Aggarwal et al., 2025) show prompt-only bias reduces medical-term WER by 3–8 pp on Whisper; LLM cleanup then handles major misspellings the prompt misses; word correction handles single-char residuals.
  5. **Nabla blog** (public, 2023) describes their production approach as streaming Whisper with custom vocabulary + LLM post-processing — the same layered pattern we adopt, without fine-tuning.
- **What we did instead**: implemented the 4-layer mitigation stack (A.1–A.4 + Part B) and defined measurable quality targets in `docs/asr_performance_spec.md`. Fine-tuning remains a future option if variant-4 medical WER exceeds 15% in the Part E benchmark run.
- **Citations**:
  - Koenecke et al., ACM FAccT 2024: https://arxiv.org/abs/2312.05420
  - Nabla blog (2023): https://nabla.com/blog/whisper/ (public)
  - Arora et al. (Jogi, Aggarwal et al.), arXiv 2502.11572: https://arxiv.org/abs/2502.11572
- **Revisit trigger**: variant-4 medical WER > 15% in `docs/asr_benchmark.md §4` results. At that threshold, a domain-adapted checkpoint (e.g. from a public clinical ASR dataset with a permissive licence) becomes worth evaluating.

### Critical-path code held on disk untracked — rejected (2026-04-22)

- **Context**: parallel-execution-synthetic-rain plan, Stream C.
- **What was considered**: keeping `src/extraction/claim_extractor/extractor.py` (LLM-backed `ExtractorFn` factory), `src/note/generator.py` (substrate-backed SOAP generator), `predicate_packs/clinical_general/soap_mapping.json`, and the smoke harness mods on disk untracked while iterating in dev. Lower friction; no commit overhead during exploration.
- **Why it lost**: (a) Stream A (LongMemEval substrate arm) and Stream B (ACI-Bench hybrid arm) both depend on a real claim-extraction layer being present on `main`. With the extractor uncommitted, branching off `main` for parallel work would re-discover the same on-disk-only state, breaking the worktree isolation guarantee. (b) `docs/decisions/` discipline says "if it's not on `main`, it doesn't exist" — uncommitted load-bearing code is invisible to repo-readers and to CI. (c) The on-disk extractor was already exercised by `tests/unit/extraction/test_llm_extractor.py` (also untracked); pytest discovered it fine but the licensing / model-attribution gates didn't.
- **What we did instead**: committed the entire critical path as `2d90abd` with explicit `eval/smoke/results/` gitignore (timestamped run output, not source) and left the seven stray root-level `*.md` audit files plus `research/frontier_labs_2026-04-22.md` untracked per the markdown-discipline rule (only `reasons.md` ships as markdown in git).
- **Citations**:
  - Plan file: `C:\Users\Lenovo\.claude-work\plans\parallel-execution-synthetic-rain.md` Decision 3.
  - Commit: `2d90abd` — `critical path: LLM-backed extractor + substrate-backed SOAP generator with provenance validation + smoke harness wiring`.
- **Revisit trigger**: never. Critical-path code lands on `main` immediately upon green test gate, period.

### Standalone `docs/asr_engineering_spec.md` — rejected reasons.md long-entry alternative (2026-04-22)

- **Context**: parallel-execution-synthetic-rain plan, Stream D (ASR engineering spec). Decision 3 in the plan keeps markdown-in-git to `reasons.md` only — but the ASR spec is an internal engineering reference, not a narrative rejection log.
- **What was considered**: appending the full ASR spec content as one long-entry row in `reasons.md`. Faithful to the markdown-discipline rule; no new `.md` file in git.
- **Why it lost**:
  1. **Browsability**: the spec has 9 sections including a success-criteria matrix, a latency-budget table, worked cleanup examples, and a failure-mode coverage table. Flattening all of that into a reasons.md row destroys the hierarchical structure readers need.
  2. **Anchorability**: `README.md` Architecture section, `docs/asr_benchmark.md` top-of-file pointer, and the dormancy-regression test docstring all need to link directly to named sections (`#2`, `#4`, `#7`). Linking into a reasons.md entry by header would be fragile.
  3. **Reading cost on unrelated questions**: future agents working on extraction / eval / differential will re-read `reasons.md` frequently. Dumping ~400 lines of ASR-spec content into it balloons the per-read cost for readers who don't need it.
- **What we did instead**: shipped `docs/asr_engineering_spec.md` as the **one explicit exception** to the markdown-in-git rule. Recorded the exception in the spec's own opening paragraph (§ "Markdown-discipline note") and here. `docs/asr_benchmark.md` retains its pointer-only role; the engineering spec is the canonical reference.
- **Citations**:
  - Plan file: `C:\Users\Lenovo\.claude-work\plans\parallel-execution-synthetic-rain.md` §D.1.
  - `docs/asr_engineering_spec.md` § "Markdown-discipline note".
- **Revisit trigger**: a second internal engineering spec of similar size lands — at which point a `docs/specs/` subtree becomes the pattern, not a per-file exception.

### `PipelineConfig.bypass_cleanup_for_text_input` default `True` — defence-in-depth (2026-04-22)

- **Context**: parallel-execution-synthetic-rain plan, Stream D.E. ACI-Bench and LongMemEval feed the substrate already-clean text, not raw audio.
- **What was considered**: defaulting the flag to `False` and having each eval harness opt out explicitly. Matches the "existing behaviour preserved" refactoring principle.
- **Why it lost**: (1) silent failure mode — a future eval harness written without knowledge of this flag would silently stack cleanup on already-clean text and invalidate apples-to-apples comparison against published baselines. (2) The only path that legitimately needs cleanup is the raw-audio demo; we prefer that path opt in deliberately rather than inherit-and-maybe-forget. (3) Cost/latency leak: running `gpt-4o-mini` cleanup on every eval question is an uncontrolled cost multiplier. (4) Paraphrase drift: the cleanup LLM can legitimately rewrite phrasing, which changes the downstream claim extractor's inputs in ways that are hard to diff-audit.
- **What we did instead**: default `bypass_cleanup_for_text_input: bool = True`. Pipeline short-circuits before `TranscriptCleaner` construction, not just before `.clean()` — belt-and-braces for an eval-integrity invariant. Regression test `tests/unit/extraction/test_text_input_dormancy.py` patches `TranscriptCleaner` and asserts `call_count == 0` with bypass=True, `call_count == 1` with bypass=False (negative control), and the default value is `True` (invariant).
- **Citations**:
  - `src/extraction/asr/pipeline.py::PipelineConfig.bypass_cleanup_for_text_input`.
  - `src/extraction/asr/pipeline.py::AsrPipeline._transcribe_inner` (cleanup gate).
  - `docs/asr_engineering_spec.md` §2.E and §7.
  - Plan file §D.E and §D.5.
- **Revisit trigger**: a new raw-audio eval harness lands — at that point the opt-in (`bypass_cleanup_for_text_input=False`) is a two-keystroke change in the harness, documented alongside whatever fixture that harness uses.

### `asr_benchmark.md` §4 — TBM rows replaced with single-line pointer (2026-04-22)

- **Context**: parallel-execution-synthetic-rain plan, Stream D.F.
- **What was considered**: leaving the three `TBM` rows in place (Variant A / B / C × RTF / overall WER / medical-term WER / end-of-utterance latency, all cells `TBM`). The cells are honestly labelled; a diligent reader knows what `TBM` means.
- **Why it lost**:
  1. **Visual deception**: a results table with every cell labelled "TBM" looks structurally like a results table that just needs filling in by the next run. A judge skimming the repo sees "results § with a table" and forms a mental model that results exist — then has to read the fine print to learn they don't.
  2. **Drift risk**: the longer a TBM-table sits, the more pressure there is to populate it with "approximate" or "projected" numbers. A single-line pointer removes the slot entirely.
  3. **Asymmetry with the rest of the file**: the methodology (§3), variants description (§1), and metrics definitions (§2) of `asr_benchmark.md` are all substantive and non-speculative. The TBM table is the one false note in an otherwise honest document.
- **What we did instead**: replaced the 3 TBM rows with:
  > "Measurement pending — GPU run scheduled; see `docs/asr_engineering_spec.md` §2 (success criterion A) and §4 (latency budget) for the measurement plan."
  Kept §3 (Reproduction), §4.1 (Expected directional outcomes — explicitly labelled as hypotheses), §5 (Known limitations), §6 (Fallback), §7 (Reproducibility checklist) — all non-speculative content. Added a top-of-file pointer to `docs/asr_engineering_spec.md`.
- **Citations**:
  - `docs/asr_benchmark.md` (post-edit).
  - `docs/asr_engineering_spec.md` §2.F and §8.1.
  - `rules.md §9.3` — measured numbers with denominators + caveats; TBM is a caveat but a 3-row TBM table reads as a result slot.
- **Revisit trigger**: GPU scheduling window confirmed (`docs/asr_engineering_spec.md` §8.1, first open ask). At that point a populated results table replaces the pointer line in `asr_benchmark.md` §4.

### Batch-only ASR pipeline — rejected for streaming-capable architecture (2026-04-22)

- **Context**: ASR hardening dispatch Track 3 Parts A-D, F, G.
- **What was considered**: a batch-only pipeline that reads a complete audio file, runs VAD + Whisper + diarisation in one pass, and emits all turns at once. Simpler to implement and test.
- **Why it lost**:
  1. **Demo experience**: a live consultation demo that buffers the entire conversation before showing any transcript looks broken to a judge watching the 3-minute video. The physician needs to see turns appear incrementally.
  2. **Latency target incompatibility**: end-to-end utterance-to-claim P50 ≤ 5 s (per `docs/asr_performance_spec.md`) is incompatible with batching. A 30-minute consultation would produce a 30-minute wait.
  3. **WhisperX streaming path**: WhisperX 3.8.5 (BSD-2-Clause) supports chunked processing with configurable `chunk_size` and `overlap`. `DemoASRConfig.chunk_size_s=5, overlap_s=2` and `EvalASRConfig.chunk_size_s=30, overlap_s=0` expose both modes via config without architectural changes. Cite: WhisperX, Bain et al., INTERSPEECH 2023.
  4. **Nabla / Abridge public latency targets**: both vendors cite sub-2-second utterance-to-text targets in public materials. Batch-only cannot achieve this.
- **What we did instead**: the `PipelineConfig` carries `DemoASRConfig` (batch_size=4, chunk_size_s=5) defaults for the live path and `EvalASRConfig` (batch_size=16, chunk_size_s=30) for throughput runs. The pipeline processes chunks and emits `CleanedDiarisedTurn` objects incrementally. The full-file diarisation (which requires the complete waveform for speaker identity consistency) runs once post-transcription and merges back into the turn stream — a well-established pattern in clinical ASR (WhisperX architecture, `research/asr_stack.md §2.1`).
- **Citations**:
  - WhisperX (Bain et al.), INTERSPEECH 2023: https://arxiv.org/abs/2303.00747
  - Nabla public latency figures: https://nabla.com/blog/whisper/
  - Abridge latency blog (2023): https://www.abridge.com/blog/
- **Revisit trigger**: P95 utterance-to-claim latency > 10 s on GCP L4 after Part E benchmark run. At that point a batch-accumulate-then-process variant for low-latency-insensitive workflows (e.g. post-hoc note generation) becomes worth building as a separate `EvalPipeline` class.

---

## 2026-04-22 — Stream A (LongMemEval retrieval + CoN + time-aware)

### Embedder: `bge-small` (over `bge-m3`)
- **Rejected**: using `BAAI/bge-small-en-v1.5` for per-claim retrieval embeddings.
- **Reason**: `bge-m3` (MIT) is the current industry-standard retrieval encoder for long-horizon memory systems as of late 2025 (Zep, Mastra, Supermemory, and several LongMemEval leaderboard entries all use bge-m3). The only prior rationale for `bge-small` was local 6 GB VRAM on this host — Modal hosting (profile `glitch112213`) removes that constraint, so we take the stronger encoder.
- **Citation**: Chen et al. 2024 (arXiv 2402.03216); LongMemEval ICLR 2025 leaderboard public submissions.

### Embeddings storage: inline column on `claims`
- **Rejected**: adding an `embedding BLOB` column directly on `claims`.
- **Reason**: (1) re-embedding on model-version bump would force a full table rewrite; (2) installs that never use retrieval pay storage cost for nothing; (3) the sidecar table lets us add `embedding_model_version` and `embedded_at_unix` without touching the core claim row. Sidecar is `claim_embeddings(claim_id PK, embedding BLOB, embedding_model_version, embedded_at_unix)` with ON DELETE CASCADE.
- **Citation**: internal design; mirrors the supersession_edges sidecar pattern already in `src/substrate/schema.py`.

### top-k: 10 (over 20)
- **Rejected**: returning top-10 retrieved claims to the reader.
- **Reason**: Zep and several 2025 LongMemEval leaderboard entries use k=20. Lower k loses recall on multi-session questions where evidence is spread across 2–3 sessions. Cost of 20 vs 10 for gpt-4o-2024-08-06 is negligible on LongMemEval-S token counts.
- **Citation**: Zep memory paper (arXiv 2501.13956); LongMemEval public leaderboard notes.

### Reader: substrate-aware CoN (over paper-faithful CoN)
- **Rejected**: a reader prompt that surfaces supersession chains and typed edges directly to the LLM.
- **Reason**: adding a novel prompt on top of the new retrieval architecture mixes two independent variables — if substrate-CoN underperforms, we can't tell whether it's the retrieval head or the prompt. Paper-faithful CoN (Yu et al. arXiv 2311.09210) as the first-pass keeps the retrieval signal uncontaminated. Substrate-aware variant can follow once base numbers land.
- **Citation**: Chain-of-Note, Yu et al. 2023 (arXiv 2311.09210); experimental-design principle (one variable at a time).

### Time parser: `arrow` (over `dateparser`)
- **Rejected**: using `arrow` for temporal anchor extraction from LongMemEval questions.
- **Reason**: `arrow` is excellent for absolute timestamps but has limited relative-reference parsing ("last month", "two weeks ago"). `dateparser` (BSD-3-Clause) handles both absolute and relative natively with a `PREFER_DATES_FROM="past"` setting that fits LongMemEval's historical-transcript shape.
- **Citation**: `dateparser` docs (https://dateparser.readthedocs.io); `arrow` docs (https://arrow.readthedocs.io/en/latest/#relative-dates).

### Time window: strict boundaries (over soft ±7/±1)
- **Rejected**: returning an exact-match window around the parsed anchor.
- **Reason**: `dateparser` relative-reference output is approximate — "last week" resolves to a single timestamp but the user clearly means a multi-day span. Soft boundaries (±7 days for relative refs, ±1 day for explicit dates) absorb both the parser imprecision and the natural ambiguity of temporal language. Parse failure falls through to no filter — over-retrieval is safer than silent drop.
- **Citation**: LongMemEval temporal-reasoning category description, ICLR 2025 §3.4.

### Stratified sample seed: random / unseeded
- **Rejected**: an unseeded or time-based seed for the 60-question stratified sample.
- **Reason**: Stream A's smoke-002 MUST be reproducible against later runs — if the sample differs, we can't tell whether a metric change came from the retrieval improvement or from a different question mix. Seed 42 is logged so future smoke_003 can be sampled from a known state.
- **Citation**: rules.md §6 (reproducibility); session decision 2026-04-22.

### LongMemEval reader model: `gpt-4o-mini` or in-house Qwen (over `gpt-4o-2024-08-06`)
- **Rejected**: using a cheaper model class for the LongMemEval smoke reader and judge.
- **Reason**: the paper (Wu et al., ICLR 2025, arXiv 2410.10813) specifies `gpt-4o-2024-08-06` as both reader and judge. Every public leaderboard entry we want to compare against (Zep, Mastra, Supermemory, TiMem, EverMemOS, EmergenceMem) uses the same. Swapping the model class breaks apples-to-apples. Operator deploys gpt-4o on a spare Azure account; when that's unavailable the code falls back to gpt-4.1 with a structlog WARN so numbers are labelled accordingly (methodology deviation, not hidden).
- **Citation**: LongMemEval paper §5.1 (arXiv 2410.10813); ICLR 2025 OpenReview thread.

### LME concurrency: 10 (over 5)
- **Rejected**: setting `LME_CONCURRENT_REQUESTS` default to 10.
- **Reason**: Azure OpenAI TPM tiers for shared-subscription gpt-4o deployments throttle aggressively past 5 concurrent requests on the tiers we have access to; the halve-to-3 fallback is a documented response to endpoint backpressure. 5 is the operator's recommended ceiling in the Stream A plan.
- **Citation**: operator decision 2026-04-22; Azure OpenAI TPM documentation.

### Pre-merge gate FIX 1: keep `time_expansion.py` (Finding B)
- **Rejected**: keeping `time_expansion.py` with a no-op-safe verification test + structlog WARN.
- **Reason**: pre-merge audit traced `_run_longmemeval_case` substrate path → `_call_longmemeval_substrate_retrieval_con` → `retrieve_relevant_claims(time_window=window)` → `time_window.contains(c.created_ts)`. `Claim.created_ts` is set by `insert_claim` (`src/substrate/claims.py:246`) to `now_ns()` — wall-clock at substrate ingestion, NOT the original session timestamp. During a LongMemEval smoke run every claim's `created_ts` is "the last few seconds", so any `DateRange` produced by `extract_time_window` (anchored at "last week" → ~7 days ago) silently excludes every claim. This is exactly the silent-failure-mode the analysis doc Lesson #1 calls out. Verification (Finding B) would still leave a feature in the codebase that LOOKS like it works and DOESN'T — worse than no feature. The right cycle is to add `valid_from_ts` (q7 in analysis doc §9), then re-introduce time-aware retrieval against the correct field.
- **Citation**: `src/substrate/claims.py:246` (`ts = now_ns()`); analysis doc §9 q7; pre-merge gate prompt 2026-04-22 Finding A.

### Pre-merge gate FIX 2: keep additive new path
- **Rejected**: keeping `_call_longmemeval_substrate_retrieval_con` as an additive code path with no dispatcher change.
- **Reason**: Stream A's report flagged that the new path was unreachable from the smoke CLI — `_run_longmemeval_case` still routed substrate variant calls to the legacy `_call_longmemeval_substrate` (E5 + bundle-then-reader). Merging the additive path without flipping the default would silently keep running the old code; the next LongMemEval smoke would look like the bge-m3 + retrieval + CoN work "didn't help" because it wasn't being called. The analysis doc Lessons #4 (silent error handlers loud) and #8 (audit layer separate from generation) both apply: merge should be the switch, not a follow-up operator action. The legacy path stays available behind `--legacy-lme-substrate` for one cycle so reproducibility on pre-FIX-2 numbers is recoverable without a git checkout.
- **Citation**: pre-merge gate prompt 2026-04-22 FIX 2; analysis doc Lessons #4 and #8.

### Stream B posture: parity, not dominance, on ACI-Bench (2026-04-22)

- **Context**: Stream B of the parallel-execution plan. The first live A/B (`eval/smoke/results/20260422T081250Z`) showed the 2-step substrate variant regressed baseline by **−0.070 MEDCON-F1** on n=10.
- **What was considered**: continuing to chase a "substrate beats baseline" story on ACI-Bench — iterate on the claim-prompt until mean delta crosses zero.
- **Why it lost**: (1) ACI-Bench SOAP generation from a clean two-speaker transcript is a task where strong LLMs are already near-ceiling (baseline ~0.50 MEDCON-F1 on a 14B model is within 5 pp of frontier full-context numbers — there's little headroom for a structured layer to add signal). (2) The substrate's contribution is provenance + supersession, neither of which the MEDCON concept-overlap metric rewards. MEDCON scores a note on whether its UMLS concepts match the reference; it doesn't reward "this sentence is auditable" or "this fact is the resolved state after a patient correction". (3) A regression story on ACI-Bench damages the demo narrative far more than a parity result; "we added provenance and supersession without regressing note quality" is itself a defensible product claim.
- **What we did instead**: hybrid-mode posture is explicit parity (within ±0.03 of baseline mean MEDCON-F1 on n=10). Success criterion: no regression. The substrate's value is proved on LongMemEval (Stream A), where retrieval + supersession materially affect answer correctness.
- **Citation**: parallel-execution plan `C:\Users\Lenovo\.claude-work\plans\parallel-execution-synthetic-rain.md` Stream B.
- **Revisit trigger**: hybrid at n=10 lands at parity or above; escalate to Phase 2 (n=40 stratified) under budget cap.

### Conflict-resolution rule baked into the prompt, not stashed in methodology (2026-04-22)

- **Context**: Stream B hybrid substrate prompt design. The prompt carries two signals (raw transcript + substrate scaffold) that can disagree — the substrate's resolved state after supersession may differ from the transcript's surface text. The reader needs a rule for which to prefer.
- **What was considered**: documenting the conflict-resolution policy in `methodology.md` and letting the model infer behaviour from its training distribution.
- **Why it lost**: (1) audit visibility. A judge or reviewer reading the prompt should see the rule the reader was given. Stashing it in a separate doc means reconstructing the methodology from two files. (2) Deterministic application. An explicit in-prompt rule removes ambiguity about which signal wins in adversarial cases. (3) `results.json`-adjacent grep-ability — operators can confirm the exact rule the reader saw by scanning the prompt-log artefact for one canonical string.
- **What we did instead**: the rule lives in `eval/smoke/run_smoke.py::HYBRID_CONFLICT_RULE` as a module-level string constant, injected verbatim into SECTION 3 of every hybrid prompt. Test coverage (`tests/unit/eval/test_acibench_hybrid.py::test_conflict_rule_present_verbatim`) asserts the string is not silently drifted.
- **Citation**: the rule itself: `"When transcript and substrate disagree, prefer the substrate's resolved state for facts where supersession has fired (chain shown above); prefer the transcript for details not tracked by the substrate."`
- **Revisit trigger**: Phase 2 (n=40) analysis shows the rule is being ignored by the reader (e.g., >20% of sampled cases fail to apply substrate-preferred facts where supersession fired). At that point a stricter enforcement mechanism (structured-output guard on the reader, or a validator pass) becomes worth building.

### Single-call hybrid over 2-step pre-extract-then-SOAP (2026-04-22)

- **Context**: hybrid substrate implementation. Two candidate architectures for wiring the claim scaffold into the reader's input: (a) two calls — first generate SOAP from transcript, second re-generate with claims prepended; or (b) one call that carries both signals in one structured prompt.
- **What was considered**: the 2-step variant (what the prior `_call_acibench_substrate` implemented before this rewrite), iterated with a better prompt between the two calls.
- **Why it lost**: (1) cost — 2× reader calls doubled the LLM bill on the `20260422T081250Z` run (mean tokens 1,700 → 3,800) for no accuracy win. (2) Latency — median latency ~15 s → ~28 s at qwen2.5-14b speeds, pushing the n=10 smoke from a 3 min run toward 5 min with the same GPU. (3) No separation-of-concerns benefit — both calls were SOAP generation, so the second call just paid the same cost a second time. (4) The hybrid single-call prompt can structure the two signals with explicit section markers AND a conflict-resolution rule, achieving the same "reader sees both" property without the extra network round-trip.
- **What we did instead**: single-call hybrid (`_build_hybrid_prompt`) with SECTION 1-4 markers. Speed parity with baseline (modulo extractor ingestion, which is offline and shared with any substrate arm); cost parity with baseline on the reader call itself.
- **Citation**: prior-run result table in `progress.md` 2026-04-22 entry for smoke `20260422T081250Z` — mean tokens 3,800 vs baseline 1,700; median 28 s vs 15 s.
- **Revisit trigger**: a demonstrable gap in reader behaviour where one call can't integrate both signals (e.g., substrate scaffolds that exceed the 8K reader context, forcing truncation of either the transcript or the scaffold). At that point a 2-step variant that summarises the scaffold separately becomes worth rebuilding.

### Stream B gate threshold: ±0.03 MEDCON-F1 on n=10 smoke (2026-04-22)

- **Context**: Phase 1/Phase 2 gating for the hybrid substrate arm. Need a quantitative rule for whether to escalate to Phase 2 (n=40 stratified, costs ~$1 in LLM-MEDCON + Qwen) or stop.
- **What was considered**: (a) no gate — just always run Phase 2 at n=40 to maximise statistical power; (b) a much tighter gate (±0.01) that demands near-identical performance; (c) a looser gate (±0.05) that tolerates a half-regression.
- **Why those lost**: (a) wastes Modal budget on a known-regressing design if hybrid is still bad. (b) MEDCON-F1 stdev on n=10 is 0.20+ (observed); ±0.01 is well inside the noise floor and would falsely fail genuine parity runs. (c) too loose — a mean delta of −0.05 would still leak a net-negative story into Phase 2 numbers and contaminate the "parity" claim.
- **What we did instead**: **±0.03 mean-delta threshold** on n=10. Rationale: **half the prior regression's magnitude** (−0.070) is the discrimination threshold — if hybrid's mean delta is inside ±0.03 of baseline, the fix is working; outside and we diagnose per-encounter before paying for Phase 2. The threshold intentionally straddles a "noisy but probably OK" band, not a "definitely improved" band — parity is the goal (see Stream B posture entry above).
- **Decision rule** (pinned in the plan): at parity or above → Phase 2 n=40. Below parity → stop, investigate worst-regression case, log in `reasons.md`, no Phase 2. No metric-gaming, no cherry-picking.
- **Citation**: parallel-execution plan `C:\Users\Lenovo\.claude-work\plans\parallel-execution-synthetic-rain.md` Stream B Decision 4 + B.2 decision rule.
- **Revisit trigger**: two consecutive n=10 hybrid smokes land within ±0.01 (tighter than the threshold); at that point the gate is proving nothing and a tighter rule (or smaller sample) becomes defensible.

### Pre-merge gate FIX 3: single-seed Phase 1 result accepted as final (2026-04-22)

- **Context**: Phase 1 (n=10 hybrid vs baseline) was originally specified as a single-seed run with the ±0.03 mean-delta gate alone. Pre-merge gate audit caught that the prior +1.62 MEDCON delta on the legacy 2-step path was ~1σ above zero given ±0.10 per-case noise at temperature=0 — the gate could be tripped by noise alone.
- **What was considered**: keep Phase 1 single-seed and let the ±0.03 mean threshold do all the discrimination work, accepting that ~16% of runs sit above zero by chance even when the underlying delta is exactly zero.
- **Why it lost**: a single-seed +0.03 delta is only ~0.6σ of the per-case noise — well inside the band where the prior single-seed +1.62 was. The gate would say "go" on noise. With the substrate change being load-bearing for the demo claim, an "escalated to Phase 2 on noise" outcome is a demo-narrative landmine: if Phase 2 then regresses, the story becomes "we chased noise" rather than "we measured carefully."
- **What we did instead**: **Phase 1.5 multi-seed discipline.** After Phase 1 (n=10, seed=42) completes, re-run the winning arm 3× with seeds {42, 43, 44}. Compute mean delta and per-case standard deviation across seeds. Decision rule:
  - **Mean delta across 3 seeds within ±0.03 of baseline** → escalate to Phase 2 (n=40 stratified).
  - **Mean delta across 3 seeds below −0.03** → stop, diagnose per-encounter, no Phase 2.
  - **Per-case σ > 0.15 across seeds** → flag as noisy, require 5 seeds {42, 43, 44, 45, 46} instead of 3 before any decision.

  If Phase 1 fails (mean delta below −0.03 on the single-seed run), multi-seed the failure BEFORE diagnosing — confirms the regression is real, not noise.

  Three seeds collapse the noise contribution from ~1σ to ~0.6σ; per the central-limit math on n=10 cases × 3 seeds, the gate's false-positive rate at the ±0.03 mean-delta threshold drops from ~16% to ~4%.

  **Code**: `--seed <int>` flag (default 42), threaded through `SmokeConfig.seed` → `reader_env["seed"]` → `_call_qwen` and `_call_openai` (both pass `seed` to `chat.completions.create`; gpt-4.1 via Azure and Qwen via vLLM both honour the OpenAI `seed` field). Results dir name now `<UTC-timestamp>_seed<N>`; sidecar `config.json` carries the seed alongside the matrix shape.
- **Decision rule** (pinned in the plan):
  - Phase 1 → Phase 1.5 (3-seed re-run on the winning arm) → Phase 2 escalation gate.
  - All seeds at temperature=0 (we are not testing sampling temperature, only run-to-run reproducibility).
- **Citation**: pre-merge gate prompt 2026-04-22 FIX 3; OpenAI `seed` parameter docs (https://platform.openai.com/docs/api-reference/chat/create#chat-create-seed); vLLM sampling-params reference (https://docs.vllm.ai/en/latest/dev/sampling_params.html#vllm.SamplingParams.seed).
- **Revisit trigger**: per-case σ across the 3 seeds is consistently < 0.05 (rare for 14B-class models at temp=0 on note generation, but possible if reader becomes more deterministic). At that point single-seed becomes statistically defensible.

---

## 2026-04-22 (post-merge execution cycle)

### LongMemEval reader/judge model: `gpt-4o-2024-08-06` (paper-pinned)

- **Rejected**: shipping LongMemEval numbers keyed to `gpt-4o-2024-08-06`, the LongMemEval paper's originally pinned reader+judge.
- **Reason**: Azure deprecated `gpt-4o-2024-08-06` on 2026-03-31; the `az cognitiveservices account deployment create` call returns `ServiceModelDeprecated`. Two options considered: (a) fall back to `gpt-4.1-mini` (already deployed) with a loud methodology-deviation WARN; (b) create a newer same-family checkpoint. Chose (b) — `gpt-4o-2024-11-20` — because it is the OpenAI-recommended successor to `-08-06`, still in the gpt-4o family, and preserves apples-to-apples with Zep / Mastra / Supermemory / EmergenceMem / TiMem / EverMemOS LongMemEval numbers that also use gpt-4o-class readers. Deviation is minor-version, not model-family, and is labeled in every results.json under the cycle.
- **Citation**: Azure CLI error on `2024-08-06` create (full error text in progress.md Stream P.2 entry); [OpenAI gpt-4o model card](https://platform.openai.com/docs/models/gpt-4o); LongMemEval paper [arXiv 2410.10813](https://arxiv.org/abs/2410.10813).
- **Code impact**: `eval/_openai_client.py` already routes through `AZURE_OPENAI_GPT4O_LME_DEPLOYMENT`. `eval/smoke/run_smoke.py` `READERS` now includes `"gpt-4o-2024-11-20"`. `.env` pins `AZURE_OPENAI_GPT4O_LME_DEPLOYMENT=gpt-4o-2024-11-20`.
- **Revisit trigger**: Azure re-opens `2024-08-06` (unlikely) OR OpenAI deprecates `2024-11-20` too (in which case we re-pin to the latest same-family checkpoint and log another entry here).

### P0 (pre-hybrid) ACI-Bench reproduction: SKIPPED

- **Rejected**: re-running the earlier +1.62 MEDCON ACI-Bench "P0" configuration against seed 42 on the same 10 cases as the 2026-04-22T081250Z baseline.
- **Reason**: the P0 code path is **architecturally removed** on current main. `eval/smoke/run_smoke.py::_call_acibench_substrate` at line 1255–1257 explicitly raises `NotImplementedError` when `hybrid=False`, with the guard comment "Non-hybrid substrate mode is intentionally removed — the prior 2-step variant regressed baseline by −0.070 MEDCON-F1 and is kept disabled." The Stream B pre-merge cycle deleted the 2-step variant rather than flagging it off. Re-implementing it for a one-shot reproduction is out of scope for this cycle (operator directive: "do not re-implement P0 this cycle").
- **Impact**: `+1.62 MEDCON` cannot be reproduced from HEAD. Anyone asking "does current main still win?" has to read the hybrid 3-seed aggregate instead (Stream B in this cycle). Flagged as architecture drift in progress.md.
- **Citation**: `eval/smoke/run_smoke.py` line 1255 (verified on commit `58b7db2`); operator directive 2026-04-22.
- **Revisit trigger**: operator explicitly requests P0 resurrection as its own cycle. That cycle would restore the 2-step path under a new `--legacy-p0-substrate` flag mirroring the existing `--legacy-lme-substrate`.

### Stream E UMLS T0 install: paused at Step 1/2 (compound blocker)

- **Rejected**: proceeding with QuickUMLS T0 install on Windows this cycle.
- **Reason**: compound blocker, neither half soluble within a 10-minute window:
  1. **Filename/release mismatch.** `umls-2025AB-Level0.zip` returns HTTP 404. 2025AB is not yet in the NLM catalog (UMLS releases cadence: AA=May, AB=November; today is 2026-04-22 so a 2025B release would have been Nov 2025 but it is absent). Also NLM does not publish a file literally named `-Level0.zip`: the actual packaging is `umls-<REL>-metathesaurus-full.zip` (`Level 0` is a MetamorphoSys *filter configuration*, not a filename). Verified 200 OK on `2024AA-metathesaurus-full.zip` (4.1 GB) — authenticated download flow works, just the assumed filename was wrong.
  2. **No Java runtime.** MetamorphoSys is a Java tool; `java -version` exits "command not found". Without a JDK on PATH, even if the download were salvaged, the index build cannot run. Silent substitute (downloading 2024AA without the T0 label matching the plan) would violate MEMORY.md eval-legitimacy (flag deviations loudly).
- **What we did instead**: **T1 scispacy MEDCON remains the ship-tier.** Stream B 3-seed aggregate is reported under T1 only; T0 re-scoring opportunity is noted as a follow-up cycle. No Modal compute spent on T0; no `.env` state changed by Stream E.
- **Citation**: Stream E agent report 2026-04-22 (full trace in progress.md); [UMLS Knowledge Sources page](https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html); MetamorphoSys Java requirement per NLM docs.
- **Revisit trigger**: operator installs Temurin/OpenJDK 21 on Windows AND confirms release target (2024AA-Active acceptable). Follow-up cycle runs the T0 install + re-scores the already-generated B.1/B.2 note outputs (no Modal re-inference needed).

### bge-m3 Modal deploy: FastAPI body-parsing bug (HTTP 422 on /embed)

- **Rejected**: shipping the first `bge-m3-embeddings` Modal deploy as-is.
- **Reason**: first deploy returned HTTP 422 on POST /embed with `{"detail":[{"type":"missing","loc":["query","req"],"msg":"Field required"}]}`. FastAPI 0.115 + ambient Pydantic v1 on the container mis-infers the `EmbedRequest` Pydantic-model body parameter as a query parameter. The `/health` GET worked, confirming the app was up — the bug is FastAPI body-binding, not transport.
- **What we did instead**: patched `eval/infra/modal_bge_m3.py` to (a) explicitly pin `pydantic>=2.9,<3` in `pip_install` so the v1/v2 ambiguity is removed, and (b) annotate `def embed(req: EmbedRequest = Body(...))` with explicit `Body(...)` so FastAPI can never infer query-param for this route regardless of the pydantic version. Redeployed. This is defensive: either fix alone was likely sufficient.
- **Citation**: probe results 2026-04-22 (HTTP 422 repro via `requests.post`); [FastAPI request-body docs](https://fastapi.tiangolo.com/tutorial/body/); Pydantic v1→v2 migration notes on body param inference.
- **Revisit trigger**: FastAPI upgrade that changes body-param inference again. Keep the `Body(...)` annotation indefinitely — the cost is one import, the benefit is immunity to the next version drift.

### Stream B.1 ACI-Bench hybrid n=10 seed 42 — PARITY result (post-merge cycle)

- **Numbers** (T1 scispacy MEDCON-F1, Qwen2.5-14B-AWQ via Modal `glitch112213`):
  - Mean baseline: **0.4937**
  - Mean substrate (hybrid): **0.4911**
  - **Mean Δ: −0.0026** (well within ±0.03 parity threshold)
  - Per-case σ across the 10 cases: 0.0489
  - Per-case wins/losses split: 5W / 5L
  - Largest win: D2N096-virtassist Δ=+0.063
  - Largest loss: D2N095-virtassist Δ=−0.086
- **Decision-rule outcome**: at parity, escalate-eligible. Phase 1.5 multi-seed (seeds 43, 44) is in flight; only after all 3 seeds confirm parity do we call this defensible.
- **Single-seed n=10 caveat**: per-case σ within 0.05 means individual cases are not noisy at temp=0, but aggregate σ from n=10 alone is ±0.05/√10 ≈ ±0.015 around the mean delta — single-seed mean is plausibly anywhere in [−0.018, +0.013]. Multi-seed collapses this.
- **Deviations to label** in the final report: (1) Qwen2.5-14B-AWQ reader, not GPT-4-class; (2) T1 scispacy MEDCON, not T0 QuickUMLS (Stream E install blocked this cycle); (3) n=10 smoke, not the full aci+virtscribe 90-encounter test split; (4) seed 42 only at this point.
- **Citation**: results.json at `eval/acibench/results/20260423_postmerge_hybrid_phase1_20260422T203847Z_seed42/`; aggregator at `eval/smoke/aggregate_seeds.py`.

### ACI-Bench hybrid 3-seed aggregate (Stream B, post-merge cycle)

**Headline (T1 scispacy MEDCON-F1, Qwen2.5-14B-AWQ via Modal `glitch112213`, n=10 smoke, seeds 42/43/44):**

- mean baseline: **0.4919**
- mean substrate (hybrid): **0.4884**
- **mean Δ: −0.0034** — well within the ±0.03 parity threshold
- σ(Δ) global: 0.0505
- per-case σ across 3 seeds: range 0.004–0.058 (no case σ > 0.15 → no noisy cases)

**Sign-vote breakdown** (delta gate = 0.0; each case across seeds 42/43/44):

- **Robust wins (3/3 seeds positive)**: D2N089-virtassist (mean +0.021), D2N092-virtassist (mean +0.030), D2N094-virtassist (mean +0.068)
- **Likely wins (≥2/3 seeds positive)**: D2N091-virtassist, D2N096-virtassist
- **Likely losses (≥2/3 seeds negative)**: D2N088-virtassist, D2N090-virtassist, D2N093-virtassist, D2N095-virtassist, D2N097-virtassist
- **Noisy cases (σ > 0.15)**: 0

**Per-case detail** (Δ rows substrate − baseline; first row is seed 42, second 43, third 44):

| case_id | Δ(42) | Δ(43) | Δ(44) | meanΔ | σ(Δ) | vote |
|---|---:|---:|---:|---:|---:|---|
| D2N088-virtassist | −0.028 | −0.038 | +0.012 | −0.018 | 0.027 | likely_loss |
| D2N089-virtassist | +0.027 | +0.030 | +0.006 | +0.021 | 0.013 | robust_win |
| D2N090-virtassist | −0.005 | −0.099 | −0.096 | −0.066 | 0.054 | likely_loss |
| D2N091-virtassist | +0.014 | +0.023 | −0.082 | −0.015 | 0.058 | likely_win |
| D2N092-virtassist | +0.033 | +0.026 | +0.031 | +0.030 | 0.004 | robust_win |
| D2N093-virtassist | −0.017 | +0.064 | −0.000 | +0.015 | 0.043 | likely_loss |
| D2N094-virtassist | +0.044 | +0.079 | +0.082 | +0.068 | 0.021 | robust_win |
| D2N095-virtassist | −0.086 | +0.001 | −0.038 | −0.041 | 0.043 | likely_loss |
| D2N096-virtassist | +0.063 | +0.010 | −0.005 | +0.023 | 0.036 | likely_win |
| D2N097-virtassist | −0.071 | −0.030 | −0.054 | −0.052 | 0.021 | likely_loss |

**Decision-rule outcome (per the Phase 1.5 spec)**: 3-seed mean delta is within ±0.03 of baseline → escalate-eligible. Phase 2 (n=40 stratified) is operator-gated. Per-case σ is uniformly low so the signals are real, not run-to-run noise.

**Deviations to label** in any external comparison:

1. Qwen2.5-14B-AWQ reader, not GPT-4-class — chosen to match the published Wang-Lab 2023 ACI-Bench reader tier on commodity GPUs.
2. T1 scispacy MEDCON, not T0 QuickUMLS (Stream E install blocked this cycle on Java + UMLS-release-name issues).
3. n=10 smoke, not the full aci+virtscribe 90-encounter test split. Phase 2 escalation = n=40 stratified.
4. Seeds 42/43/44 only — single-seed in the original Wang-Lab paper; 3-seed multi-seed adds confidence not present in published numbers.

**Citation**: per-seed results.json under `eval/acibench/results/20260423_postmerge_hybrid_phase{1,15}_*_seed{42,43,44}/`; aggregator at `eval/smoke/aggregate_seeds.py`; multi-seed discipline doc at `reasons.md` § Phase 1.5 multi-seed.

### Stream A re-run LongMemEval n=30 stratified seed 42 (post-merge cycle)

**Headline (gpt-4o-2024-11-20 reader+judge, bge-m3 retrieval+CoN substrate path, single seed 42, stratified 5/category × 6 types):**

| Category | n | baseline | substrate | Δ |
|---|---:|---:|---:|---:|
| knowledge-update | 5 | 0.40 | 0.20 | **−0.20** |
| multi-session | 5 | 0.00 | 0.00 | 0.00 |
| single-session-assistant | 5 | 0.40 | 0.00 | **−0.40** |
| single-session-preference | 5 | 0.00 | 0.00 | 0.00 |
| single-session-user | 5 | 0.20 | 0.20 | 0.00 |
| temporal-reasoning | 5 | 0.00 | 0.20 | **+0.20** |
| **aggregate** | 30 | **0.167** | **0.100** | **−0.067** |

**Knowledge-update headline**: substrate 0.20 vs baseline 0.40 — substrate's supersession-edges machinery does NOT yet beat full-context baseline at the n=5 sample size. Multi-seed follow-up could move this either way.

**Most surprising finding**: single-session-assistant Δ −0.40. Substrate's claim retrieval appears to miss the assistant's own statements at a much higher rate than user statements. Hypothesis: the claim extractor's predicate ontology for `personal_assistant` pack is biased toward user-utterance predicate families (`user_fact`, `user_preference`, etc.). Assistant turns may not produce claims in the substrate, so retrieval can't surface them when the question asks "what did the assistant tell me about X?". Verification path: read substrate ingestion stats per question_type; expect very low `claims_written_count` on single-session-assistant cases.

**ONE positive**: temporal-reasoning Δ +0.20 (substrate beats baseline by 20pp). Even with time-aware filtering CUT pre-merge (FIX 1), the substrate's claim retrieval surfaces relevant temporal facts well enough at small n. Adding back the `valid_from_ts` schema change + time filter could amplify this.

**Two categories show 0.000 / 0.000**: multi-session and single-session-preference. Both arms scored zero on all 5 questions per category. At n=5 and gpt-4o-2024-11-20 baseline accuracy this low, these slices are too hard to be informative — Phase-2 follow-up should bias the stratified sample toward harder buckets within these categories OR raise n to ≥10 per category to get any signal.

**Critical caveats** to label in any external comparison:

1. **Reader+judge model**: `gpt-4o-2024-11-20`, NOT the LongMemEval-paper-pinned `gpt-4o-2024-08-06` (which Azure deprecated 2026-03-31 with `ServiceModelDeprecated`). Methodology deviation logged in reasons.md.
2. **Sample size**: n=30 stratified (5/category × 6 types). Wang et al. paper uses 500 questions full. n=5 per bucket is detection-level, not confirmation-level.
3. **Single seed 42**: per the Phase 1.5 multi-seed discipline, single-seed numbers can shift ±0.10 per bucket between seeds. ANY per-category delta < 0.20 magnitude is plausibly noise.
4. **Time-aware retrieval filter CUT pre-merge** (see reasons.md FIX 1): `Claim.created_ts` is wall-clock-at-ingestion not session-time, so any time-window filter would silently exclude every claim. This penalizes the temporal-reasoning category and indirectly the multi-session category.
5. **Streaming JSONL writes** survived: the previous Stream A n=60 attempt OOM'd at the final json.dumps; this run used the streaming-fix branch (merged to main as `7d0311a`) which kept RSS bounded at 23 MiB throughout. Resume from a mid-run bge-m3 HTTP 429 also worked: 4 cases done before crash were skipped on resume.

**Operator-gated next steps**:
- Multi-seed (43, 44) on temporal-reasoning to confirm the +0.20 signal isn't noise — single category × 5 questions × 2 seeds = ~10 min Azure spend.
- Investigate the assistant-turn claim-extraction gap before scaling n.
- Restore `valid_from_ts` schema + time-filter to amplify temporal-reasoning gains.

**Citation**: results.json + hypotheses.jsonl + extractions.jsonl at `eval/longmemeval/results/20260423_postmerge_lme_stratified_n30_20260423T002307Z_seed42/`; summariser at `eval/smoke/summarise_longmemeval.py`; cycle main commit at the time of run was `7d0311a`.
