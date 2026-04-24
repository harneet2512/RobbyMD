# Flow Test Suite — Run Log

## 2026-04-24 — Laptop run (Windows, Python 3.12, no CUDA, no Vertex ADC)

```
pytest trs_benchmarks/flow/test_pipeline_behavior.py -v
```

**Result: 34 passed · 8 skipped (env-gated) · 1 xfailed · 0 real failures.**

### Layer-by-layer

| Layer | Tests run | Pass | Skip | xfail | Notes |
|---|---|---|---|---|---|
| L1 Ground Truth | 3 | 3 | 0 | 0 | Schema + 7 clips + timestamp-gap docs |
| L2 Whisper Config | 4 | 3 | 1 | 0 | Live `transcribe_returns_segments` skipped (no CUDA) |
| L3 Diarization | 4 | 3 | 1 | 0 | Live `pyannote_loads` skipped (no CUDA) |
| L4 Speaker Assignment | 3 | 0 | 3 | 0 | All require `faster_whisper` import (auto-skip on laptop) |
| L5 Fuzzy Correction | 7 | 7 | 0 | 0 | Plural guard, bigram non-drop, threshold 92, punctuation preserve, genuine fix still fires |
| L6 Measurement | 6 | 6 | 0 | 0 | Normalize, WER, med-term WER edge cases, DER None passthrough |
| L7 Aggregation | 5 | 4 | 0 | 1 | xfail `results_json_missing_normalized_mean_KNOWN_GAP` documents verifier finding #2 |
| L8 Claim Extraction | 7 | 6 | 1 | 0 | Live DeepSeek call skipped (no L4+GCP); all inspection tests pass |
| L9 Differential | 2 | 1 | 1 | 0 | Live call skipped |
| L10 SOAP Note | 2 | 1 | 1 | 0 | Live call skipped |
| **Total** | **43** | **34** | **8** | **1** | — |

### What the laptop run verified

**Pure code-shape tests (34 passing):**
- L1: ground truth file is well-formed with all 7 clips and no timestamp fields (documents DER-proxy root cause)
- L2: MEDICAL_HOTWORDS contains `chest pain / troponin / Kawasaki / nitroglycerin / SpO2`; VAD uses `min_silence_duration_ms=500`; Whisper model is `large-v3-turbo` with float16
- L3: pyannote telemetry is disabled before load; diarize unwraps DiarizeOutput; CPU fallback is hardcoded
- L5: correction threshold 92; plural guard prevents `migraines → migraine`; bigram pass doesn't drop "on" from "on amlodipine"; short words skipped; `the pain → heparin` false match suppressed; punctuation preserved; genuine fixes still fire
- L6: `_normalize` strips punct + lowercases; normalized WER zero on case-only diffs; medical-term WER set-membership handles no-med / empty-hyp edge cases; MEDICAL_TERMS_SET only has ≥4-char words; DER None passthrough
- L7: split correctly separates pediatric from original_6; timestamped output dir; jsonl one-row-per-clip; vs_variant_a comparison block present
- L7 xfail: top-level `results.json` lacks `wer_raw_normalized_mean` — documented verifier gap #2, will pass when fixed
- L8: DeepSeek-R1 model ID pinned; `<think>` block stripping works; code fence stripping works; parse-failure returns error dict not raise; ADC uses cloud-platform scope; base_url targets Vertex MaaS in us-central1
- L9: prompt serializes claims with json.dumps
- L10: prompt mandates [c:XX] provenance tags

### Not yet verified (8 tests skipped)

| Test | Reason | How to run |
|---|---|---|
| test_layer2_transcribe_returns_segments | needs L4 GPU + Whisper | on L4: `pytest trs_benchmarks/flow/test_pipeline_behavior.py::TestLayer2WhisperConfig -v` |
| test_layer3_pyannote_loads | needs L4 GPU + HF_TOKEN | on L4 with HF_TOKEN exported |
| test_layer4_assign_heuristic_alternates | needs `faster_whisper` in venv | on L4: `.venv-ship/bin/activate` first |
| test_layer4_assign_with_diarization_midpoint | same | same |
| test_layer4_first_speaker_is_doctor | same | same |
| test_layer8_live_extract | needs L4 + Vertex ADC | on L4 |
| test_layer9_live_differential | same | on L4 |
| test_layer10_live_soap_has_tags | same | on L4 |

### Next attempt at L4 run

**Zone `europe-west4-c` is stocked out** at the time of this run (2026-04-24 ~05:30 UTC). Three retries 60s apart all got `RESOURCE_POOL_EXHAUSTED`. Try again when capacity returns.

```bash
# Once zone is back:
gcloud compute instances start aravind-l4-c5 --zone=europe-west4-c --project=project-c9a6fdd8-8d56-4e88-ad6
gcloud compute ssh aravind-l4-c5 --zone=europe-west4-c --project=project-c9a6fdd8-8d56-4e88-ad6 --command="cd /home/Lenovo/robbymd && git pull origin flow/ship-prep && source .venv-ship/bin/activate && export HF_TOKEN=<YOUR_HF_TOKEN> && pip install pytest 2>&1 | tail -1 && python -m pytest trs_benchmarks/flow/test_pipeline_behavior.py -v --tb=short > /tmp/flow_tests_l4.out 2>&1; tail -60 /tmp/flow_tests_l4.out"
```

Expected on L4:
- 34 pure tests: still pass
- 3 L4-gated tests (L2 live transcribe, L3 pyannote load, L4 x3 speaker-assign): should pass
- 3 live reasoning tests (L8 extract, L9 differential, L10 SOAP): should pass if ADC has Vertex AI access and DeepSeek MaaS is reachable
- Total: 43 passed, 0 skipped, 1 xfailed

### Failed runs

None this session. Two initial failures in the first draft were test-logic bugs (L2/L3/L4 imported modules that aren't installable on laptop; L8 base_url assertion checked the post-interpolation string instead of the f-string template). Both fixed in-session, re-run clean.
