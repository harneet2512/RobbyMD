# asr_stack.md — Researcher A brief

**Scope**: ASR + diarisation stack for the live clinical dictation pipeline (mic → substrate admission API). Chest-pain demo, single-session, 16–24 GB GPU assumption. OSI-approved dependencies only.

**Constraints**: `rules.md` §1.2 (OSI-only), `Eng_doc.md` §3.1 (model selection), §5.1 (admission), §9 (deployment). Does not duplicate `docs/research_brief.md`.

**Status**: research brief, no production code.

---

## 1. Executive summary

The stack is **`faster-whisper` 1.2.1 (MIT, CTranslate2 backend) running Whisper large-v3 (MIT)** as the primary streaming transcriber, **WhisperX 3.8.5 (BSD-2-Clause)** for forced-alignment + `pyannote`-based diarisation, and **Distil-Whisper `distil-large-v3` (MIT)** as a speed fallback when GPU headroom drops or latency budget (≤1.5 s utterance→text per `Eng_doc.md` §3.1) is at risk. Medical-vocabulary steering uses Whisper's native `initial_prompt` (up to ~224 prompt tokens inside Whisper's 448-token context window) seeded from RxNorm ingredient names, a small SNOMED-CT chest-pain subset, ICD-10-CM chest-pain-relevant descriptors, and the AHA/ACC 2021 chest-pain lexicon. OpenWhispr (MIT) is used only as an architectural reference; no code is copied.

**Single most important claim**: `faster-whisper` with CTranslate2 INT8 on a 24 GB GPU runs Whisper large-v3 at ~**2.3–4× real time** vs. `openai-whisper` baseline for long-form input (SYSTRAN benchmark, large-v2, beam 5, FP16: 63 s vs. 143 s on 13 min of audio) [1] — this is what makes a ≤1.5 s end-of-utterance budget credible on a single-GPU workstation before any Distil fallback is needed.

---

## 2. Component-by-component wiring plan

### 2.1 Audio flow

```
mic (synthetic only, rules.md §2.1)
  │ 16 kHz mono, 30 ms frames
  ▼
[VAD gate]                   silero-vad 5.x (MIT) [2]
  │  speech segments (≥250 ms)
  ▼
[Ring buffer + chunker]      5 s rolling windows, 1 s overlap
  │
  ▼
[Primary ASR]                faster-whisper 1.2.1 (MIT) [3]
  │ Whisper large-v3 (MIT) [4], CTranslate2 INT8_FLOAT16
  │ initial_prompt = medical vocab bias string (§4)
  │ beam_size=5, vad_filter=False (handled upstream)
  │
  │◄── fallback switch ── if GPU util >80% or queue_depth>2
  │                       swap to distil-large-v3 (MIT) [5]
  ▼
[raw segments]  (text, t_start, t_end, avg_logprob, no_speech_prob)
  │
  ▼
[Diariser + aligner]         whisperx 3.8.5 (BSD-2-Clause) [6]
  │  wav2vec2-base forced alignment → word-level timestamps
  │  pyannote/speaker-diarization-community-1 (CC-BY-4.0) [6]
  │  LICENCE NOTE: the pyannote model card is CC-BY-4.0 — use the
  │  community-1 weights; the older 3.1 model gated behind HF user
  │  terms is not used. WhisperX itself is BSD-2-Clause [6].
  ▼
[diarised turns]  (speaker_id, text, t_start, t_end, asr_confidence)
  │
  ▼
[Admission filter]           substrate/admission.py (Eng_doc §5.1)
  │  drop <3 content words, drop filler-only
  ▼
[substrate write API]        turns table (Eng_doc §4.1)
```

### 2.2 Processes

- **One GPU process** hosts faster-whisper + Distil (both loaded, hot-swap at runtime). Memory budget at INT8_FLOAT16: large-v3 ≈ 1.5 GB, distil-large-v3 ≈ 0.75 GB, plus pyannote ≈ 0.5 GB, leaving ample headroom on a 24 GB 4090 [1,5,6].
- **One CPU process** for VAD + ring buffer + admission filter + event bus publish. VAD on CPU is cheaper than marshalling frames to GPU at 30 ms cadence [2].
- WhisperX alignment + diariser called **after** each utterance-final Whisper segment (not per-window) — this matches WhisperX's documented batched pipeline [6] and avoids pyannote thrash.

### 2.3 Version pins (OSI-approved only)

| Library | Version | License | Source |
|---|---|---|---|
| `faster-whisper` | 1.2.1 | MIT | [3] |
| `ctranslate2` | ≥4.4.0 (faster-whisper 1.2.x dep) | MIT | [3] |
| `whisperx` | 3.8.5 | BSD-2-Clause | [6] |
| `pyannote.audio` | 3.3.x (as pinned by WhisperX 3.8.5) | MIT | [7] |
| `silero-vad` | 5.x | MIT | [2] |
| `openai-whisper` large-v3 weights | — | MIT | [4] |
| `distil-whisper/distil-large-v3` weights | — | MIT | [5] |

pyannote speaker-diarization-community-1 **model weights** are CC-BY-4.0 [6]. CC-BY-4.0 is OSI-compatible for attribution-preserving redistribution; we treat it as model data (not code) and attribute in `SYNTHETIC_DATA.md` and UI credits. No gated pyannote-3.1 weights are used.

---

## 3. Per-component latency estimates (16–24 GB GPU)

All numbers are **published** unless flagged "not yet measured".

| Stage | Latency (per utterance ~5 s audio) | Source |
|---|---|---|
| silero-vad (CPU) | <10 ms per 30 ms frame | [2] silero README |
| faster-whisper large-v3 INT8_FLOAT16, beam=5 | **~0.3–0.5 s** on 24 GB GPU (RTF ≈ 0.1) | [1] SYSTRAN benchmark (large-v2 as proxy; large-v3 has same arch) |
| Same, switched to distil-large-v3 | ~6.3× faster than large-v3 → **~0.05–0.08 s** | [5] HF model card |
| WhisperX forced-alignment (wav2vec2) | Reported "70× real time" batched pipeline overall | [6] WhisperX README |
| pyannote speaker-diarization-community-1 | ~2× real time GPU inference typical for pyannote 3.x | [7] pyannote.audio docs |
| End-to-end (mic → diarised turn) | **~0.8–1.3 s** target; **not yet measured; to be confirmed in wt-extraction benchmark** per `CLAUDE.md` §5.2 | — |

`Eng_doc.md` §3.1 target is RTF ≤ 0.7 and ≤1.5 s utterance→text. Published numbers make that budget comfortable; the open question is pyannote cold-start on first diarisation call. `CLAUDE.md` §5.2 requires `docs/asr_benchmark.md` to confirm.

---

## 4. Medical-term WER with and without `initial_prompt`

### 4.1 Mechanism

Whisper decodes conditioned on a preceding text prompt (the "prompt" slot of the decoder). `faster-whisper.transcribe(..., initial_prompt=...)` passes a string of up to ~224 tokens into that slot (leaving ≥224 of the 448-token decoder context for generated tokens) [3]. The prompt biases the language-model prior toward domain vocabulary without any fine-tuning.

### 4.2 Published numbers

- **Rare-word recall (not medical-specific, but the closest published zero-shot biasing study)**: Chen et al., *"Improving Rare-Word Recognition of Whisper in Zero-Shot Settings"*, arXiv 2502.11572 (IEEE SLT 2024), report **+45.6% rare-word recall** and **+60.8% unseen-word recall** over baseline prompted-biasing — but that result is after **supervised fine-tuning** on 670 h of Common Voice, not from prompt alone [8]. Distinction matters: we are using prompt-only biasing, so we **cannot claim these numbers**.
- **Prompt-only medical-domain**: no peer-reviewed study I can confirm reports prompt-only biasing on Whisper for clinical ASR. The MultiMed medical-ASR benchmark (arXiv 2409.14074) [9] reports Whisper large-v3 baseline WER but does not isolate prompt-vs-no-prompt ablation.
- **Directional evidence**: community reproductions (SYSTRAN faster-whisper docs, OpenAI Whisper discussions) consistently show single-digit to low-double-digit percentage-point WER reductions on in-domain jargon when prompts include the rare terms verbatim [3].

### 4.3 Our expectations (clearly labelled)

- **Target** per `Eng_doc.md` §3.1: medical-term WER ≤12%.
- **Expectation (unverified)**: prompt-only biasing with a ~180-token RxNorm-drug + chest-pain-anatomy string will reduce medical-term WER by 3–8 percentage points relative to unprompted large-v3 on our scripted chest-pain demo. This is a **hypothesis**; `CLAUDE.md` §5.2 mandates verification via `docs/asr_benchmark.md` before we put a number in the demo video.
- **Distil-whisper caveat**: distil-large-v3 reports ≤1 pp long-form WER loss on general English vs large-v3 [5]; loss on medical jargon is likely **larger** because the distilled decoder has weaker long-tail language modelling. We use Distil as a latency fallback, not a first choice.

---

## 5. Medical vocabulary source list

Target prompt budget: ~180 tokens (≈ 700 characters). Single prompt string, not a runtime retrieval. Each source below is checked for OSI compatibility at **distribution time**; if we redistribute derived strings, we attribute per the source's terms.

| # | Source | URL | License / terms | ~Term count | Access | Gotcha |
|---|---|---|---|---|---|---|
| 1 | **MedlinePlus Web Service (Health Topics XML)** | https://medlineplus.gov/about/developers/webservicesdocumentation/ | Public domain US government work; no redistribution restriction beyond standard NLM disclaimer [10] | ~1,000 health-topic titles (EN) | Free XML download, no key | MedlinePlus titles are lay-language; less useful for physician-side jargon than for patient-side. |
| 2 | **RxNorm (normalised drug names)** | https://www.nlm.nih.gov/research/umls/rxnorm/index.html | **RxNorm content itself is public domain** (NLM-created normalized names/codes) [11]; **full file download requires a free UMLS licence**; REST `rxnav.nlm.nih.gov` is ungated [11] | ~100k+ strings (ingredients + brand + clinical drug); we need the ~12k ingredient subset | REST API ungated; RRF download behind UMLS licence | UMLS licence already in-flight per `progress.md`. For prompt-biasing we only need the top ~150 ingredients relevant to chest pain (antiplatelet, anticoagulant, statin, beta-blocker, PPI, NSAID, GTN). |
| 3 | **SNOMED-CT US Edition (chest-pain subset)** | https://www.nlm.nih.gov/healthit/snomedct/us_edition.html | Requires UMLS Metathesaurus Licence (free, 1–3 day approval); distributed via UTS [12] | ~350k concepts full edition; our subset ≤200 concepts under `finding-site = thorax` ∩ `chest-pain` ancestor | ZIP download from UTS after licence | **Do not redistribute SNOMED strings** in the repo. Keep the prompt-string generation a local build step; commit the generation script + concept-ID list only, not the strings. SNOMED Affiliate Licence forbids redistribution of SNOMED content to non-licensees. |
| 4 | **ICD-10-CM (chest-pain descriptors)** | https://www.cms.gov/medicare/coding-billing/icd-10-codes (files via https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Publications/ICD10CM/) | ICD-10-CM is a US public-domain work maintained by NCHS/CMS; free download, no licence [13] | ~72k codes full; chapter IX (circulatory) + R07 (chest pain, R07.x) subset ≤80 codes | Free FTP / CMS download | We want the **descriptor text** not the codes themselves; the 2025 file `icd10cm_codes_2025.txt` is the clean source [13]. |
| 5 | **AHA/ACC 2021 Chest Pain Guideline (lexicon)** | https://doi.org/10.1161/CIR.0000000000001029 | Open-access guideline published in *Circulation* (open-access policy for AHA guidelines) [14] | ~50 chest-pain descriptors (crescendo, substernal, pleuritic, radiation-to-arm/jaw, exertional, etc.) | PDF + HTML via AHA Journals | Extract terminology by hand from the recommendations tables; do not paste guideline text verbatim into the repo (fair-use quotation is fine; bulk copy is not). This is also the primary LR-table source for `wt-trees` [Eng_doc §4.4]. |

### 5.1 Prompt composition strategy (recommended)

One static 150–180-token string, roughly:

```
[comma-separated RxNorm chest-pain drug ingredients ~60 tokens]
[comma-separated AHA/ACC chest-pain descriptors ~40 tokens]
[comma-separated SNOMED chest-pain findings ~40 tokens]
[comma-separated ICD-10 R07.x descriptors ~20 tokens]
```

Exact wording to be tuned empirically (see Open Questions §7).

---

## 6. Known risks and unknowns

| # | Risk / unknown | Severity | Notes |
|---|---|---|---|
| R1 | **pyannote speaker-diarization-community-1 is CC-BY-4.0** | Medium | CC-BY-4.0 is not on `rules.md` §1.2 OSI allowlist (MIT/Apache-2.0/BSD/MPL/ISC/LGPL). It **is** an OSI-compatible free-culture licence, widely accepted for models-as-data, but the strict reading of rules.md would reject it. **Escalation needed**: decision record in `docs/decisions/` to either (a) add CC-BY-4.0 to the model-weights allowlist, or (b) swap to a different diariser (options: NVIDIA NeMo Sortformer — Apache-2.0 weights [15], or skip diarisation for the demo and rely on alternating-speaker heuristics). |
| R2 | **`initial_prompt` token-length edge cases** in faster-whisper 1.2.1 | Low | Verified: faster-whisper enforces prompt+max_new_tokens ≤ 448 and raises ValueError if exceeded [3]. We target ≤224 prompt tokens (half budget), leaving room for ~224 generated tokens per window — enough for a 5 s utterance at normal speech rate. |
| R3 | **Cold-start of pyannote** on first diarisation | Medium | Not yet measured. Possible 2–5 s stall on first call; mitigate by warming the pipeline at process start. |
| R4 | **Distil-large-v3 medical-WER regression** | Medium | Distil claims ≤1 pp WER on long-form general English [5]; no published medical-domain delta. If Distil degrades medical accuracy significantly, fallback becomes cosmetic not real. |
| R5 | **Prompt-biasing has diminishing returns >200 tokens** | Low-Medium | Anecdotal from faster-whisper community; no formal study. Our ~180-token target is inside the safe zone. |
| R6 | **WhisperX 3.8.5 released 2026-04-01** | Low | Fresh release during hackathon window. Risk of an undiscovered regression; mitigate by pinning 3.8.5 exactly and noting fallback to 3.8.4. |
| R7 | **OpenWhispr architecture note** | — | OpenWhispr uses whisper.cpp + sherpa-onnx, **not** faster-whisper per its README [16]. Their choice is optimised for CPU-only laptop deployment; ours is GPU workstation. We take zero code, and only the "separate VAD+buffer+ASR process" pattern is reused — a generic pattern. |
| R8 | **SNOMED redistribution** | High if violated | Do not commit SNOMED strings to the public repo. Build-time generation only. |
| R9 | **Synthetic-only audio** (`rules.md` §2.1) | — | No risk, by construction — the ASR stack is evaluated on scripted / TTS-generated chest-pain audio declared in `SYNTHETIC_DATA.md`. |

---

## 7. Open questions (need human decision)

1. **Prompt content scope** — should `initial_prompt` include the full AHA/ACC 2021 chest-pain glossary (~40 tokens), or only drug names (~60 tokens)? More tokens → stronger bias for covered terms, weaker general language model. Recommendation: start with drugs + 20 top AHA/ACC descriptors (~80 tokens) and expand empirically.
2. **Minimum Whisper version that reliably supports `initial_prompt` with 200+ tokens** — verified present in faster-whisper 1.2.1 [3] with explicit length check; we have not confirmed behaviour on prompts >224 tokens is bias-only (no generation leakage into output). Need a 10-utterance sanity check in wt-extraction benchmark.
3. **pyannote model licence decision (R1)** — ADR needed: allow CC-BY-4.0 for model weights, or swap diariser?
4. **Distil-Whisper fallback trigger policy** — GPU-util threshold (current proposal: >80% for >2 s) or queue-depth (>2 pending utterances)? Or always-large-v3 for the demo video and distil only in production? Recommendation: demo-video always uses large-v3; fallback is README-vision feature.
5. **SNOMED subset decision** — do we invest in the ~200-concept chest-pain subset now, or defer until after UMLS licence approval? Given the full UMLS download is already on the critical path for MEDCON (`progress.md`), incremental cost is low once the licence lands.
6. **Streaming granularity** — 5 s windows with 1 s overlap is a reasonable default, but `Eng_doc.md` §9 says ≤2.5 s utterance→all-panels-updated. If we want sub-second partial-hypothesis updates in the UI, we need partial-segment emission from faster-whisper — feasible but adds UI complexity. Recommendation: full-utterance emission only for demo, partials as README vision.
7. **Do we need the admission-filter embedding-novelty check?** `docs/research_brief.md` §4 recommends downgrading `admission.py` from 60 to 30 LOC and dropping embedding-novelty. ASR stack is unaffected either way; flagging for cross-team alignment.

---

## Sources

1. SYSTRAN faster-whisper benchmarks — https://github.com/SYSTRAN/faster-whisper (README "Benchmarks" section, v1.2.1, accessed 2026-04-21). License: MIT.
2. silero-vad — https://github.com/snakers4/silero-vad. License: MIT.
3. faster-whisper 1.2.1 — https://pypi.org/project/faster-whisper/ and https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/transcribe.py (initial_prompt implementation). License: MIT. Accessed 2026-04-21.
4. Whisper large-v3 — https://huggingface.co/openai/whisper-large-v3. License: MIT. Radford et al., "Robust Speech Recognition via Large-Scale Weak Supervision", arXiv 2212.04356.
5. distil-whisper/distil-large-v3 — https://huggingface.co/distil-whisper/distil-large-v3. License: MIT. Claim: 6.3× faster than large-v3, within 1 pp long-form WER. Gandhi et al., "Distil-Whisper: Robust Knowledge Distillation via Large-Scale Pseudo Labelling", arXiv 2311.00430.
6. WhisperX 3.8.5 — https://github.com/m-bain/whisperX. License: BSD-2-Clause. Bain et al., "WhisperX: Time-Accurate Speech Transcription of Long-Form Audio", Interspeech 2023, arXiv 2303.00747. Uses `pyannote/speaker-diarization-community-1` (CC-BY-4.0).
7. pyannote.audio — https://github.com/pyannote/pyannote-audio. Library: MIT. Bredin, "pyannote.audio 2.1 speaker diarization pipeline", Odyssey 2023.
8. Chen et al., "Improving Rare-Word Recognition of Whisper in Zero-Shot Settings" — arXiv 2502.11572, IEEE SLT 2024. https://arxiv.org/abs/2502.11572.
9. MultiMed medical ASR benchmark — arXiv 2409.14074. https://arxiv.org/abs/2409.14074.
10. MedlinePlus developer services — https://medlineplus.gov/about/developers/ and https://medlineplus.gov/about/developers/webservicesdocumentation/.
11. RxNorm — https://www.nlm.nih.gov/research/umls/rxnorm/index.html and https://www.nlm.nih.gov/research/umls/rxnorm/docs/termsofservice.html. RxNorm content is public domain; full-file download requires UMLS licence.
12. SNOMED-CT US Edition — https://www.nlm.nih.gov/healthit/snomedct/us_edition.html. Distributed via UTS under UMLS Metathesaurus Licence / SNOMED Affiliate Licence.
13. ICD-10-CM (NCHS/CMS, public domain) — https://www.cms.gov/medicare/coding-billing/icd-10-codes and https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Publications/ICD10CM/.
14. Gulati et al., "2021 AHA/ACC/ASE/CHEST/SAEM/SCCT/SCMR Guideline for the Evaluation and Diagnosis of Chest Pain" — *Circulation*, 2021. DOI: 10.1161/CIR.0000000000001029. Open access.
15. NVIDIA NeMo Sortformer diarisation (alternative to pyannote) — https://github.com/NVIDIA/NeMo, Apache-2.0. Cited as fallback option only.
16. OpenWhispr — https://github.com/openwhispr/openwhispr. License: MIT. Architecture reference only; zero code reused per `rules.md` §1.1.
