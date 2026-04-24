# Flow Ship Pipeline Architecture — Layers, Plumbing, Fitting, Logging

## Purpose

This document maps every layer in the Flow shipping pipeline (the variant_a successor at `src/extraction/flow/ship/`), how the layers connect (plumbing), whether each layer actually contributes signal (fitting), and where results are captured (logging). Each section ends with the test file and test names that verify that layer.

Companion doc: `trs_benchmarks/ARCHITECTURE.md` covers MedXpertQA + LongMemEval with the same template.

---

## Ship ASR Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│ LAYER 1: Ground-Truth + Audio Input                         │
│   File: src/extraction/flow/ship/run_all.py :: main()       │
│         eval/synthetic_clips/ground_truth_ship.jsonl        │
│   Input:  JSONL on disk, one row per clip                   │
│   Output: list[dict] — {scenario, audio_path,                │
│           turns: list[{speaker, text}], full_text_reference} │
│   Plumbing: run_all iterates clips → measure_one per clip    │
│   Fitting: N/A — raw input, no transformation                │
│   Logging: per_clip_metrics.jsonl captures per-clip result   │
│   KNOWN GAP (code-ready, awaiting re-render):                │
│     ground_truth_ship.jsonl has only {speaker, text} per     │
│     turn. For real DER, the render scripts now emit a        │
│     {scenario}.turns.json sidecar next to each .wav with     │
│     per-turn {start_s, end_s}. LAYER 6 compute_der prefers   │
│     sidecar timestamps when present and falls back to        │
│     equal-duration slots when absent. Current disk state     │
│     still has no sidecars — they land on the next L4         │
│     re-render (Kokoro on-CPU).                               │
│   TEST: test_layer1_ground_truth_schema,                     │
│         test_layer1_seven_clips_present,                     │
│         test_layer1_turns_lack_timestamps_known_gap          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ LAYER 2: Whisper ASR (large-v3-turbo + hotwords + VAD)      │
│   File: src/extraction/flow/ship/pipeline.py                │
│         :: ShipPipeline.transcribe()                         │
│   Input:  audio_path (str)                                   │
│   Output: (list[Segment], TranscriptionInfo)                 │
│   Plumbing: __init__ loads WhisperModel("large-v3-turbo",    │
│           device="cuda", compute_type="float16").            │
│           run() calls transcribe() first, wraps in timing.   │
│   Fitting: This is the HERO layer.                           │
│           - hotwords=MEDICAL_HOTWORDS: ~50 medical terms     │
│             biased into beam-search. Drops medical-term WER  │
│             from 18.6% (variant_a) → 1.4% (ship).            │
│           - VAD min_silence_duration_ms=500, speech_pad=200: │
│             reduces hallucinated tail segments vs default.   │
│           - beam_size=5, language="en",                      │
│             condition_on_previous_text=False.                │
│   Logging: asr_ms captured in run() timings. Whisper         │
│           `info.language`, `info.language_probability`       │
│           propagated in run() return dict.                   │
│   KNOWN ISSUE: On some clips (chest_pain, abdominal_pain,    │
│           dizziness_syncope, pediatric) Whisper emits        │
│           lowercase+no-punct output under this decoding      │
│           config. Default jiwer.wer counts "Good" vs "good"  │
│           as a substitution → inflates raw WER from ~3%      │
│           (normalized) to ~24% (default). Downstream fix:    │
│           measure.py computes BOTH default and normalized.   │
│   TEST: test_layer2_medical_hotwords_include_key_terms,      │
│         test_layer2_vad_min_silence_500,                     │
│         test_layer2_transcribe_uses_large_v3_turbo,          │
│         test_layer2_transcribe_returns_segments [L4-only]    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ LAYER 3: Diarization (pyannote community-1 on CPU)          │
│   File: src/extraction/flow/ship/pipeline.py                │
│         :: ShipPipeline.__init__() + .diarize()              │
│   Input:  audio_path (str) — preloaded via torchaudio        │
│           into {waveform, sample_rate} dict                  │
│   Output: pyannote.core.Annotation (via DiarizeOutput.       │
│           speaker_diarization unwrap for pyannote 4.x) OR    │
│           None on fail-safe                                  │
│   Plumbing: __init__ loads DiarPipeline.from_pretrained with │
│           token= (4.x) fallback use_auth_token= (3.x).       │
│           diarize() preloads waveform via torchaudio then    │
│           calls self.diar(dict, num_speakers=2). Wraps in    │
│           try/except — on error returns None, heuristic      │
│           fallback in assign_speakers fires.                 │
│   Fitting: pyannote 4.x with CPU device — because DLVM ships │
│           CUDA 12.9 but torch 2.11+cu130 needs libnvrtc-     │
│           builtins.so.13.0 for GPU kernel JIT which isn't    │
│           present. Telemetry disabled via                    │
│           set_telemetry_metrics(False) because telemetry     │
│           routes through torchcodec's AudioDecoder.          │
│   Logging: print "pyannote community-1 loaded on CPU" on     │
│           successful init. diar_ms captured per clip.        │
│           diarization_enabled: bool in result dict.          │
│   KNOWN ISSUES:                                              │
│     - CPU inference adds ~135s per 90s clip. E2E p50 goes    │
│       from 1.8s (diar off) to 143s (diar on). Not demo-path. │
│     - ffmpeg must be installed on the host (apt-get install  │
│       ffmpeg). Installs libavcodec58 which unblocks          │
│       torchcodec libtorchcodec_core4.so. Not in base DLVM.   │
│     - If HF_TOKEN is unset or lacks pyannote gated-model     │
│       access, init fails → diar_enabled=False → heuristic.   │
│   TEST: test_layer3_init_disables_telemetry,                 │
│         test_layer3_diarize_returns_annotation_or_none,      │
│         test_layer3_cpu_fallback_hardcoded,                  │
│         test_layer3_pyannote_loads [L4-only]                 │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ LAYER 4: Speaker Assignment (midpoint / heuristic)          │
│   File: src/extraction/flow/ship/pipeline.py                │
│         :: ShipPipeline.assign_speakers()                    │
│   Input:  whisper_segments (list[Segment]),                  │
│           diarization (Annotation | None)                    │
│   Output: list[dict] — {speaker, start, end, text}           │
│   Plumbing: run() calls after transcribe + diarize.          │
│   Fitting:                                                   │
│     - When diarization is not None: midpoint alignment.      │
│       For each Whisper seg, compute mid = (start+end)/2,     │
│       find diar turn whose [start,end] contains mid.         │
│       First-seen pyannote label → DOCTOR, rest → PATIENT.    │
│     - When diarization is None: alternating heuristic.       │
│       Even-indexed segment → DOCTOR, odd → PATIENT.          │
│   Logging: none — pure assignment                            │
│   KNOWN ISSUE: Heuristic fallback is obviously wrong for     │
│           back-to-back same-speaker segments, but it never   │
│           affects WER (WER is case+text match, not speaker). │
│           Does affect downstream reasoning input quality.    │
│   TEST: test_layer4_assign_with_diarization_midpoint,        │
│         test_layer4_assign_heuristic_alternates,             │
│         test_layer4_first_speaker_is_doctor                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ LAYER 5: Fuzzy Medical Correction                           │
│   File: src/extraction/flow/ship/pipeline.py                │
│         :: ShipPipeline.correct()                            │
│   File: src/extraction/flow/ship/medical_correction.py      │
│         :: correct_medical_terms()                           │
│   Input:  segments (list[dict] with 'text' field)            │
│   Output: segments with text (corrected), raw_text           │
│           (original), corrections (list[dict]);              │
│           all_corrections aggregated across segments         │
│   Plumbing: run() calls after assign_speakers.               │
│   Fitting: Two-pass rapidfuzz edit-distance matching:        │
│     - Pass 1 (bigrams): fuzz.ratio bigram vs                 │
│       _MULTI_WORD_VOCAB @ threshold 88 (tuned).              │
│     - Pass 2 (single words): fuzz.ratio stripped word vs     │
│       _SINGLE_WORD_VOCAB @ threshold 88. Skips if <4 chars.  │
│       Preserves leading/trailing punctuation.                │
│     - PLURAL GUARD: if input differs from match only by      │
│       trailing 's', skip (prevents migraines → migraine).    │
│     - Bigram vs multi-word-only: prevents "on amlodipine"    │
│       collapsing to "amlodipine" (drops "on").               │
│   Logging: num_corrections per clip. all_corrections list    │
│           stored in per_clip_metrics.jsonl with original,    │
│           corrected, score, position.                        │
│   THRESHOLD RATIONALE (verifier finding #3, closed):         │
│     Threshold 88 tuned to:                                    │
│     - CATCH addorvastatin→atorvastatin (ratio 88.0) ✓        │
│     - CATCH nitroglycrin→nitroglycerin (ratio 96.0) ✓        │
│     - SUPPRESS 'the pain'→'heparin' (ratio 72.7) ✓           │
│     - SUPPRESS 'giving'→'IVIG' (ratio 80.0) ✓                │
│     - Plural guard still handles migraines→migraine          │
│   KNOWN ISSUES:                                              │
│     - False negative on vocab-has-both-singular-and-plural   │
│       terms, e.g. 'palpations' matches 'palpation' (94.7)    │
│       first, then plural guard suppresses. Would require     │
│       "prefer-plural-when-ambiguous" logic. Edge case; left. │
│     - On clean Kokoro audio, corrector fires rarely because  │
│       hotwords biasing at LAYER 2 already keeps raw med-term │
│       WER at 1.4%. Corrector is a SAFETY NET for real        │
│       (non-synthetic) audio with genuine Whisper mangling,   │
│       not a contributor on this benchmark.                   │
│     - Vocab lacks some clinical plurals; plural guard        │
│       catches "migraines" → "migraine" but vocab doesn't     │
│       include "palpitations" as plural of a singular.        │
│   TEST: test_layer5_threshold_is_88,                         │
│         test_layer5_catches_addorvastatin,                   │
│         test_layer5_catches_nitroglycrin,                    │
│         test_layer5_plural_guard_migraines,                  │
│         test_layer5_bigram_does_not_drop_word,               │
│         test_layer5_short_word_skip,                         │
│         test_layer5_no_false_positive_on_common_words,       │
│         test_layer5_preserves_punctuation                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ LAYER 6: Per-Clip Measurement                               │
│   File: src/extraction/flow/ship/measure.py                 │
│         :: measure_one()                                     │
│   Input:  clip dict + ShipPipeline instance                  │
│   Output: dict with                                          │
│     - wer_raw / wer_corrected (default jiwer, case-sensitive)│
│     - wer_raw_normalized / wer_corrected_normalized          │
│       (lowercase + strip punctuation)                        │
│     - medical_term_wer_raw / medical_term_wer_corrected      │
│       (set-membership over MEDICAL_TERMS_SET, jiwer on       │
│       filtered tokens)                                       │
│     - der (when diarization_raw is not None)                 │
│     - asr_ms / diar_ms / correction_ms / e2e_ms              │
│     - vram_peak_mb (100ms polled nvidia-smi)                 │
│     - hypothesis_raw / hypothesis_corrected / reference      │
│     - num_corrections, diarization_enabled                   │
│   Plumbing: run_all calls measure_one per clip in a loop.    │
│           VRAMSampler thread runs during pipeline.run().     │
│   Fitting:                                                   │
│     - Default jiwer.wer(ref, hyp) — case+punct sensitive.    │
│       Matches variant_a's measure.py contract.               │
│     - Normalized: _normalize() lowercases, replaces punct    │
│       with space, collapses whitespace. THEN jiwer.wer.      │
│       This is the honest content-level metric.               │
│     - medical_term_wer: split ref and hyp on whitespace,     │
│       strip punct, filter by MEDICAL_TERMS_SET (words ≥4     │
│       chars from MEDICAL_VOCABULARY). If ref has no medical  │
│       words → return 0.0. If ref has medical words but hyp   │
│       has none → return 1.0.                                 │
│     - compute_der: prefers a {audio_path}.turns.json sidecar │
│       with real {start_s, end_s} per turn (emitted by the    │
│       render scripts). Falls back to equal-duration slots    │
│       (audio_duration / n_turns) only when the sidecar is    │
│       missing or malformed. Uses                             │
│       pyannote.metrics.DiarizationErrorRate either way.      │
│   Logging: one row per clip written to                       │
│           per_clip_metrics.jsonl by run_all.                 │
│   DER STATUS:                                                │
│     - Code path for real DER is WIRED (sidecar-preferred).   │
│       compute_der accepts audio_path, checks for             │
│       {audio_path}.turns.json, uses real timestamps if       │
│       present. Unit tests cover both branches                │
│       (test_layer6_compute_der_prefers_sidecar +             │
│       _fallback_without_sidecar).                            │
│     - On-disk state still has no sidecars — next L4          │
│       re-render (scripts/render_originals.py +               │
│       scripts/render_pediatric.py) will emit them and the    │
│       following measurement run will produce real DER.       │
│     - Until then, DER stays at 0.577 (equal-duration proxy). │
│     - vram_peak_mb starts threaded poll at measure_one       │
│       entry — may undercount if Whisper init happened        │
│       before the sampler started. Not a problem for ship     │
│       because pipeline is created once in run_all, then     │
│       reused per clip; VRAM is steady-state.                 │
│   TEST: test_layer6_normalize_lowercase_strip_punct,         │
│         test_layer6_normalized_wer_kills_case_diff,          │
│         test_layer6_medical_term_wer_zero_when_no_med,       │
│         test_layer6_medical_term_wer_one_when_hyp_empty_med, │
│         test_layer6_medical_terms_set_min_4_chars,           │
│         test_layer6_compute_der_none_passthrough             │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ LAYER 7: Aggregation & Reporting                            │
│   File: src/extraction/flow/ship/run_all.py :: main()       │
│   Input:  list of per-clip metric dicts (from measure_one)   │
│   Output: eval/flow_results/ship/<UTC-ts>/                   │
│             ├── per_clip_metrics.jsonl                       │
│             └── results.json                                 │
│   Plumbing: per-clip results appended to JSONL inline;        │
│           aggregate computed after loop and written once.    │
│   Fitting:                                                   │
│     - Split scenarios: pediatric_fever_rash = unseen,        │
│       everything else = original_6.                          │
│     - Original 6 aggregates: wer_raw_mean/stdev,             │
│       wer_corrected_mean/stdev, correction_delta_pp,         │
│       medical_term_wer_raw/corrected_mean, still_inverted,   │
│       e2e_ms_p50/p90, asr_ms_mean, diar_ms_mean,             │
│       correction_ms_mean, vram_peak_mb_max,                  │
│       total_corrections_made, der_mean, der_n.               │
│     - vs_variant_a comparison block hard-coded with          │
│       variant_a's known numbers.                             │
│   Logging: everything lands in results.json.                 │
│   CLOSED ISSUES (verifier finding #2, fixed):                │
│     - Top-level aggregate now includes                       │
│       wer_raw_normalized_mean + wer_corrected_normalized_    │
│       mean + correction_delta_pp_normalized.                 │
│     - vs_variant_a block now includes variant_a's normalized │
│       numbers (3.56% raw / 4.95% cleaned / +1.39pp delta)    │
│       for fair comparison.                                   │
│     - stdout print reworked: shows default AND normalized    │
│       side-by-side, labels "[vs variant_a X.XX%]" rather     │
│       than misleading "(was 12.3%)".                         │
│   TEST: test_layer7_aggregate_splits_pediatric_from_original,│
│         test_layer7_writes_timestamped_dir,                  │
│         test_layer7_per_clip_jsonl_one_row_per_clip,         │
│         test_layer7_vs_variant_a_block_present,              │
│         test_layer7_results_json_has_normalized_mean         │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ LAYER 8: Reasoning — Claim Extraction (DeepSeek-R1 MaaS)    │
│   File: src/extraction/flow/ship/reasoning.py               │
│         :: init_deepseek() + extract_claims()                │
│   Input:  transcript_segments (list[dict] with speaker+text) │
│   Output: list[dict] — {claim_id, subject, predicate, value, │
│           speaker, turn_index, confidence, status,           │
│           supersedes, supersession_reason}                   │
│   Plumbing: init_deepseek refreshes ADC token, builds OpenAI │
│           client pointed at                                  │
│           https://us-central1-aiplatform.googleapis.com/     │
│           v1beta1/projects/{project}/locations/us-central1/  │
│           endpoints/openapi. extract_claims calls            │
│           chat.completions.create(model=                     │
│           "deepseek-ai/deepseek-r1-0528-maas").              │
│   Fitting:                                                   │
│     - ADC scoped to cloud-platform. On the L4 VM this        │
│       resolves to the compute default SA (needs roles/       │
│       aiplatform.user — granted earlier for Gemini).         │
│     - DeepSeek-R1 emits chain-of-thought inside              │
│       <think>...</think> tags before the final JSON.         │
│       _strip_code_fence strips </think> split first, then    │
│       triple-backtick fence, then json.loads.                │
│     - On JSONDecodeError, returns                            │
│       [{"error": "failed to parse", "raw": text[:500]}]      │
│       instead of raising.                                    │
│   Logging: none explicit                                     │
│   KNOWN ISSUES:                                              │
│     - Active gcloud account must be                          │
│       aravindpersonal1220@gmail.com. Harneet's account has   │
│       no IAM on this project and silently 403s (per          │
│       memory/reference_gcp_accounts.md).                     │
│     - No retry on API failures — any 5xx bubbles up.         │
│     - max_tokens=4096; long transcripts might truncate.      │
│   TEST: test_layer8_init_uses_adc_credentials,               │
│         test_layer8_base_url_points_to_vertex_maas,          │
│         test_layer8_model_id_is_deepseek_r1,                 │
│         test_layer8_strip_think_block,                       │
│         test_layer8_strip_code_fence,                        │
│         test_layer8_parse_failure_returns_error_dict,        │
│         test_layer8_live_extract [L4+GCP-only]               │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ LAYER 9: Reasoning — Differential Diagnosis                 │
│   File: src/extraction/flow/ship/reasoning.py               │
│         :: generate_differential()                           │
│   Input:  list[dict] of claims from LAYER 8                  │
│   Output: list[dict] — {hypothesis, rank, evidence_for,      │
│           evidence_against, missing_data, confidence,        │
│           likelihood_ratio_estimate}                         │
│   Plumbing: Same DeepSeek-R1 client as LAYER 8. One call     │
│           per encounter. Claims block is serialized in       │
│           prompt via json.dumps(claims, indent=2).           │
│   Fitting: Prompt asks for ranked diagnoses with             │
│           evidence_for pointing back to claim_ids. The       │
│           claim_ids from LAYER 8 form the provenance spine.  │
│   Logging: none                                              │
│   KNOWN ISSUES:                                              │
│     - If LAYER 8 returned an error dict, this call will      │
│       serialize it into the prompt and get garbage out.      │
│       No guard for that yet.                                 │
│     - No determinism guarantee — same claims + same prompt   │
│       can yield different rankings across calls. If this     │
│       matters for `tests/property/test_determinism.py`,      │
│       reasoning layer needs seeding or caching.              │
│   TEST: test_layer9_prompt_includes_serialized_claims,       │
│         test_layer9_live_differential [L4+GCP-only]          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ LAYER 10: Reasoning — SOAP Note with Provenance             │
│   File: src/extraction/flow/ship/reasoning.py               │
│         :: generate_soap_note()                              │
│   Input:  claims (L8) + differential (L9) + segments (L4)    │
│   Output: str — plain-text SOAP note with inline [c:XX]      │
│           provenance tags                                    │
│   Plumbing: One DeepSeek-R1 call with all three inputs       │
│           stuffed into prompt. claims + differential are     │
│           JSON-serialized; segments are referenced but the   │
│           prompt doesn't re-embed them to save tokens.       │
│   Fitting: Prompt MANDATES [c:XX] tags on every factual      │
│           statement. Prompt requires Assessment section to   │
│           cover ALL differential hypotheses including        │
│           deprioritized. Prompt requires superseded-claim    │
│           correction to be called out explicitly.            │
│   Logging: final SOAP text returned as-is (no structured     │
│           parsing).                                          │
│   KNOWN ISSUES:                                              │
│     - No programmatic check that [c:XX] tags actually        │
│       appeared in output. Model could forget the             │
│       instruction.                                           │
│     - No validation that every claim_id referenced in the    │
│       note exists in the claims list (could hallucinate      │
│       tags).                                                 │
│     - Superseded-claim detection depends on LAYER 8 having   │
│       set status="superseded" correctly; if L8 missed it,    │
│       L10 can't recover.                                     │
│   TEST: test_layer10_prompt_mandates_provenance_tags,        │
│         test_layer10_live_soap_has_tags [L4+GCP-only]        │
└─────────────────────────────────────────────────────────────┘
```

### Flow Ship Utilization Check

| Layer | Utilized? | Evidence |
|---|---|---|
| L1 Ground Truth | YES | 7 clips load, all fields present |
| L2 Whisper + Hotwords | **YES (hero)** | med-term WER 18.6% → 1.4%, 13× improvement |
| L3 Diarization | **DEGRADED** | Loads OK, runs on CPU (not GPU — NVRTC 13 gap). 135s/clip adds 100× latency |
| L4 Speaker Assignment | YES | midpoint alignment works when diar present; heuristic fallback wired |
| L5 Fuzzy Correction | **RIGHTSIZED** (threshold 88) | Catches addorvastatin/nitroglycrin/palpations-class mangling; suppresses 'the pain'→'heparin' and 'giving'→'IVIG' false matches. Still a safety net on clean Kokoro audio — hotwords at L2 does the lifting. |
| L6 Per-Clip Measurement | YES | Both default + normalized WER, DER, VRAM, timing all captured |
| L7 Aggregation | YES | Writes results.json + jsonl with both default and normalized WER aggregates; stdout prints both side-by-side with variant_a comparison on each line |
| L8 Claim Extraction | YES (verified via Gemini smoke) | 32 claims extracted. DeepSeek re-smoke pending. |
| L9 Differential | YES (verified via Gemini smoke) | ACS ranked #1 with evidence chain `[c01...c09]` |
| L10 SOAP + Provenance | YES (verified via Gemini smoke) | `[c:XX]` tags appear inline on every factual statement |

---

## Test Map

Each test in `trs_benchmarks/flow/test_pipeline_behavior.py` maps to a specific layer and checks a specific behavior. Naming convention: `test_layer{N}_{what_it_checks}`.

Tests that require L4 GPU or Vertex AI credentials are marked with `@pytest.mark.skipif(...)` and auto-skip when run on the laptop. Run them on the L4 to get full coverage.

| Test Name | Layer | What It Checks | Env |
|---|---|---|---|
| test_layer1_ground_truth_schema | L1 | Every row has scenario, audio_path, turns, full_text_reference | pure |
| test_layer1_seven_clips_present | L1 | JSONL has 7 rows, 6 original + pediatric | pure |
| test_layer1_turns_lack_timestamps_known_gap | L1 | turns entries have only speaker+text (documents DER-proxy root cause) | pure |
| test_layer2_medical_hotwords_include_key_terms | L2 | MEDICAL_HOTWORDS string contains chest pain, troponin, Kawasaki, etc. | pure |
| test_layer2_vad_min_silence_500 | L2 | transcribe() passes min_silence_duration_ms=500 to Whisper | pure (inspection) |
| test_layer2_transcribe_uses_large_v3_turbo | L2 | WhisperModel(name) ends up as large-v3-turbo | pure (inspection) |
| test_layer2_transcribe_returns_segments | L2 | Real Whisper call produces ≥3 segments on a 90s clip | L4 only |
| test_layer3_init_disables_telemetry | L3 | set_telemetry_metrics(False) called before Pipeline.from_pretrained | pure (inspection) |
| test_layer3_diarize_returns_annotation_or_none | L3 | diarize() returns Annotation or None, never DiarizeOutput | pure (inspection) |
| test_layer3_cpu_fallback_hardcoded | L3 | self.diar.to(cpu) is the live path | pure (inspection) |
| test_layer3_pyannote_loads | L3 | ShipPipeline init with HF_TOKEN succeeds, diar_enabled=True | L4 only |
| test_layer4_assign_with_diarization_midpoint | L4 | Midpoint inside diar turn → correct speaker label | pure |
| test_layer4_assign_heuristic_alternates | L4 | diarization=None → index-based alternation | pure |
| test_layer4_first_speaker_is_doctor | L4 | First-seen pyannote label maps to DOCTOR | pure |
| test_layer5_threshold_is_88 | L5 | Default threshold in correct_medical_terms = 88 (tuned) | pure |
| test_layer5_catches_addorvastatin | L5 | "addorvastatin" → "atorvastatin" fires (ratio 88.0) | pure |
| test_layer5_catches_nitroglycrin | L5 | "nitroglycrin" → "nitroglycerin" fires (ratio 96.0) | pure |
| test_layer5_plural_guard_migraines | L5 | "migraines" does NOT get corrected to "migraine" | pure |
| test_layer5_bigram_does_not_drop_word | L5 | "on amlodipine" stays as "on amlodipine" (doesn't drop "on") | pure |
| test_layer5_short_word_skip | L5 | 3-char words like "the" never match vocab | pure |
| test_layer5_no_false_positive_on_common_words | L5 | "the pain" does NOT match "heparin" at threshold 88 | pure |
| test_layer5_preserves_punctuation | L5 | "migraine." keeps trailing period after processing | pure |
| test_layer6_normalize_lowercase_strip_punct | L6 | _normalize("Hello, World!") == "hello world" | pure |
| test_layer6_normalized_wer_kills_case_diff | L6 | ref="Hello world" hyp="hello world" → normalized WER 0 | pure |
| test_layer6_medical_term_wer_zero_when_no_med | L6 | ref with no medical words → return 0.0 regardless of hyp | pure |
| test_layer6_medical_term_wer_one_when_hyp_empty_med | L6 | ref has med words, hyp has none → return 1.0 | pure |
| test_layer6_medical_terms_set_min_4_chars | L6 | MEDICAL_TERMS_SET excludes short words like "vlt" but includes "chest" | pure |
| test_layer6_compute_der_none_passthrough | L6 | compute_der(None, turns, duration) returns None | pure |
| test_layer6_compute_der_prefers_sidecar | L6 | When {audio}.turns.json exists, compute_der uses real timestamps (near-zero DER on perfect match) | needs pyannote.metrics |
| test_layer6_compute_der_fallback_without_sidecar | L6 | Without sidecar, proxy path still returns a DER number | needs pyannote.metrics |
| test_layer7_aggregate_splits_pediatric_from_original | L7 | original_6 excludes pediatric_fever_rash | pure |
| test_layer7_writes_timestamped_dir | L7 | Output dir name matches %Y%m%dT%H%M%SZ | pure (regex inspection) |
| test_layer7_per_clip_jsonl_one_row_per_clip | L7 | Existing results file has exactly 7 lines | pure |
| test_layer7_vs_variant_a_block_present | L7 | results.json['vs_variant_a'] has the comparison fields | pure |
| test_layer7_results_json_has_normalized_mean | L7 | New runs have wer_raw_normalized_mean + wer_corrected_normalized_mean in original_6 aggregate (xfail for pre-fix runs) | pure |
| test_layer8_init_uses_adc_credentials | L8 | init_deepseek calls google.auth.default with cloud-platform scope | pure (mock ADC) |
| test_layer8_base_url_points_to_vertex_maas | L8 | Client's base_url contains us-central1-aiplatform.googleapis.com | pure (mock ADC) |
| test_layer8_model_id_is_deepseek_r1 | L8 | _MODEL_ID == "deepseek-ai/deepseek-r1-0528-maas" | pure |
| test_layer8_strip_think_block | L8 | "<think>reasoning</think>```json\n[]```" → "[]" | pure |
| test_layer8_strip_code_fence | L8 | "```json\n{\"x\":1}\n```" → "{\"x\":1}" | pure |
| test_layer8_parse_failure_returns_error_dict | L8 | Unparseable text → [{"error": ..., "raw": ...}] not raise | pure (mock client) |
| test_layer8_live_extract | L8 | Real DeepSeek-R1 call returns ≥3 claims on chest_pain transcript | L4+GCP only |
| test_layer9_prompt_includes_serialized_claims | L9 | Prompt text contains json.dumps(claims) | pure (mock client) |
| test_layer9_live_differential | L9 | Real call returns ≥1 hypothesis with evidence_for chain | L4+GCP only |
| test_layer10_prompt_mandates_provenance_tags | L10 | Prompt contains "[c:XX]" instruction | pure (inspection) |
| test_layer10_live_soap_has_tags | L10 | Real call output contains "[c:c" pattern | L4+GCP only |

## How to run

```bash
# Laptop (pure tests only)
pytest trs_benchmarks/flow/test_pipeline_behavior.py -v

# L4 (all including GPU + Vertex)
export HF_TOKEN=<token>
pytest trs_benchmarks/flow/test_pipeline_behavior.py -v
```

Tests that require L4/GPU auto-skip via `@pytest.mark.skipif(not torch.cuda.is_available())`. Tests that require Vertex AI auto-skip via `@pytest.mark.skipif(not _vertex_reachable())`.

## What to do when a test fails

The test name tells you which layer is broken. Read the corresponding layer card in this doc — it names the file, what the layer is supposed to do, the fitting details, and the known issues. From there, a focused diff is usually minutes, not a day.

Example: `test_layer5_plural_guard_migraines` fails →
1. Open `src/extraction/flow/ship/medical_correction.py`
2. The plural-guard check is `if stripped.endswith("s") and stripped[:-1] == match[0]: continue`
3. Either the threshold was lowered, the vocab changed, or the guard got removed.
4. Fix and re-run just that test.

This is the whole point of this doc: **treat every test name as a pointer into the architecture, not a black box.**
