# ASR Engineering Spec

**Status (2026-04-22)**: spec complete. Measurement on target hardware
pending (see §8 "Open asks"). Code path described here is shipped and
exercised by the unit-test suite under `tests/unit/extraction/`.

> **Markdown-discipline note.** CLAUDE.md / parallel-execution plan §D.1
> restricts markdown-in-git to `reasons.md`. This document is the **one
> explicit exception** — it is an internal engineering reference, browsable
> and anchorable from `README.md` Architecture. The exception is recorded
> in `reasons.md` entry 2026-04-22 "Standalone ASR spec over reasons.md
> long-entry".

**Cross-references**
- `Eng_doc.md §3.1` — ASR latency / WER targets (upstream invariants).
- `PRD.md §3` — product requirements for the live demo path.
- `MODEL_ATTRIBUTIONS.md` — license rows for every model referenced here.
- `SYNTHETIC_DATA.md` — declaration of `eval/fixtures/asr/chest_pain_demo.wav`.
- `docs/asr_benchmark.md` — methodology + reproduction for the measurement run.
- `docs/asr_performance_spec.md` — detailed per-stage latency targets.
- `research/asr_stack.md` — stack-wiring research brief and prior-art survey.
- `reasons.md` 2026-04-22 entries — per-decision audit trail.

---

## 1. Position

Commercial single-speaker dictation tools (WisprFlow, Superwhisper, MacWhisper,
OpenWhispr) solve the problem of turning *one* person's speech into clean
text. That problem is well-served. The problem this pipeline addresses is
different in kind:

> **Canonical one-liner**: "Single-person dictation tools like WisprFlow
> don't solve the two-speaker clinical problem; this does."

A clinical consultation is a **two-speaker, turn-taking, role-asymmetric**
signal. The physician and the patient have different vocabularies, different
normalisation rules, and different correction strategies (the patient's
words are privileged verbatim for provenance; the physician's words are
normalised to medical terminology). Speaker diarisation is load-bearing —
you cannot run claim extraction on a mono transcript because the claim
"takes metoprolol daily" means something different coming from the doctor
than from the patient.

**Architecture lineage**: this pipeline extends the open-source FreeFlow
pattern ([github.com/zachlatta/freeflow](https://github.com/zachlatta/freeflow),
MIT) from single-speaker dictation to two-speaker medical dialogue. FreeFlow
demonstrates the core trick — a per-speaker system-prompt + filler-removal
+ custom-dictionary cleanup — and we extend it with:

1. `pyannote/speaker-diarization-community-1` (CC-BY-4.0) for the
   two-speaker split (WhisperX's default backend in v3.8.5+).
2. **Role-aware** system prompts — doctor, patient, unknown — wired in
   `src/extraction/asr/transcript_cleanup.py::_ROLE_TO_PROMPT`.
3. A **deterministic 5-check hallucination guard**
   (`src/extraction/asr/hallucination_guard.py`) running post-cleanup /
   pre-word-correction.
4. Text-input **dormancy** (§7 below) — the cleanup stage does NOT engage
   on ACI-Bench / LongMemEval text paths, protecting apples-to-apples
   comparison with published baselines.

Neither of the two commercial medical-ASR vendors (Nabla, Abridge) publish
their architectures in detail; this pipeline follows their *publicly
described* layered-mitigation pattern (streaming Whisper + custom vocabulary
+ LLM post-processing — see [Nabla's 2023 blog](https://nabla.com/blog/whisper/))
without lifting any code.

---

## 2. Success criteria

Six success criteria, each with a measurement method and acceptance threshold.
Criteria **A–F** map 1:1 to the deliverables of the ASR-hardening dispatch.

### A. End-to-end raw audio → diarised clean medical transcript

- **Input**: a WAV file of a two-speaker clinical conversation.
- **Output**: a stream of `CleanedDiarisedTurn` objects ready for claim
  extraction (see `src/extraction/asr/pipeline.py::CleanedDiarisedTurn`).
- **Measurement**:
  - **RTF** (real-time factor) = `wall_clock_seconds / audio_seconds`;
    end-to-end (VAD → diarise → cleanup → guard → correct).
    Target ≤ **0.2** on GCP L4 24 GB spot (per `docs/asr_performance_spec.md §1`).
    Original `Eng_doc.md §3.1` target was ≤ 0.7 on 16–24 GB GPU; ≤ 0.2 is the
    tightened target after CTranslate2 INT8_FLOAT16 evidence
    ([SYSTRAN benchmarks](https://github.com/SYSTRAN/faster-whisper)).
  - **End-of-utterance → text latency**: wall clock from the last VAD
    frame marking speech-end to the final `CleanedDiarisedTurn` being
    emitted. Target ≤ **1.5 s** per `Eng_doc.md §3.1`.
  - **Medical-term WER**: WER computed on the medical-term subset of the
    reference transcript. Target ≤ **12%** per `docs/asr_performance_spec.md §2`.
- **Hardware target**: GCP L4 24 GB spot (primary) or Modal A10 (secondary).
  A single GPU is sufficient — the pipeline is single-threaded per session
  (see `telemetry.py` note on concurrency).
- **Status**: unmeasured in this environment. Fixture + scripts are ready
  (`docs/asr_benchmark.md §3`); the GPU scheduling window is the first
  open ask in §8 below.

### B. Hallucination resistance — 5-check guard

The guard (`src/extraction/asr/hallucination_guard.py`) runs post-cleanup
and pre-word-correction with five deterministic checks. Cheap, O(n) in
segment length, no ML imports.

1. **`repeated_ngram`** — flags n-gram loops (e.g. "thank you thank you
   thank you thank you"). Motivated by Koenecke et al., ACM FAccT 2024
   ([arXiv 2312.05420](https://arxiv.org/abs/2312.05420)), which documents
   Whisper's loop-pathology on silent / near-silent frames.
2. **`oov_medical_term`** — flags TitleCased or medical-suffixed tokens
   (length 5–20) absent from the active pack's `medical_vocabulary`. Targets
   confabulated anatomy / procedure names.
3. **`extreme_compression`** — flags segments whose characters-per-second
   ratio exceeds 15 chars/s (normal conversational speech: ~10–14 chars/s).
   A Whisper hallucination typically generates faster than the underlying
   speech rate.
4. **`low_confidence_span`** — flags any run of ≥ 5 consecutive words with
   avg per-word confidence < 0.30. Signals the decoder was guessing.
5. **`invented_medication`** — flags drug-suffixed tokens (-statin, -mab,
   -nib, -pril, -sartan, etc.) absent from the RxNorm vocabulary. A
   placeholder ~30-drug set is wired now; full RxNorm lands once UMLS
   licence clears. Arora et al.
   ([arXiv 2502.11572](https://arxiv.org/abs/2502.11572), Jogi, Aggarwal
   et al., 2025) document medical-term hallucination as a distinct
   pathology from general hallucination — hence a dedicated check.

Severity:
- `clean` → 0 checks flagged.
- `warn`  → 1–2 checks flagged AND no `invented_medication` flag.
- `block` → 3+ checks flagged OR `invented_medication` is True.

Each check has a **crafted-failure unit test** in
`tests/unit/extraction/test_hallucination_guard_each_check.py`
(parametrised; one test per check plus an aggregate-name invariant test).
The pre-existing `tests/unit/extraction/test_hallucination_guard.py` covers
each checker function directly; the per-check file exercises the same
checks via the aggregate `check()` entry point — the surface the pipeline
actually calls.

**Acceptance threshold (production rollout, not this spec)**: ≤ 0.5% of
segments BLOCK-flagged on clean speech (`docs/asr_performance_spec.md §2`).
Unmeasured pre-GPU-run.

### C. Medical vocabulary lift via `initial_prompt` biasing

Whisper supports prefix conditioning via its `initial_prompt` argument (up
to 224 tokens — see `reasons.md` 2026-04-21 "ASR prompt length" entry).
`src/extraction/asr/vocab.py::build_initial_prompt` assembles a medical
glossary prompt from RxNorm + AHA/ACC chest-pain guidelines + ICD-10
anchors.

- **Measurement**: medical-term WER of variant B (biased) vs variant A
  (unbiased baseline) on the synthetic chest-pain clip.
- **Target**: 3–8 pp absolute medical-term WER reduction.
- **Honest framing**: Arora et al. ([arXiv 2502.11572](https://arxiv.org/abs/2502.11572))
  report 40–60% **relative** reductions on rare medical terms using
  **contextual-biasing with explicit biasing loss** — a stronger training
  intervention than our prompt-only approach. Prompt-only is the weaker
  of the two: it modulates the decoder's prior without changing weights.
  We should therefore expect the lower end of Arora et al.'s range at
  best. The 3–8 pp target is deliberately conservative; anything above
  that on the GPU run is a bonus. Document the measured vs expected gap
  in `docs/asr_benchmark.md §4` once the run lands.
- **Cross-reference**: `research/asr_stack.md §4.3` for the directional
  rationale.

### D. Per-speaker cleanup

`src/extraction/asr/transcript_cleanup.py` extends FreeFlow's single-speaker
pattern with role-aware system prompts:

- `SYSTEM_PROMPT_DOCTOR` — filler removal, backtracking resolution, medical
  misspelling correction, numeric/unit normalisation. Example:
  "BP one-twenty over eighty" → "blood pressure 120/80 mmHg".
- `SYSTEM_PROMPT_PATIENT` — preserve the patient's **exact** complaint
  language; tag lay phrases with `[likely: <medical-term>]` rather than
  replace. Example: "feels like someone's sitting on my chest
  [likely: pressure-type chest pain]". Cross-reference the doctor's earlier
  mention of a medication where possible: "my blood pressure pill" +
  doctor previously said "amlodipine 5 mg" → link both for the claim
  extractor.
- `SYSTEM_PROMPT_UNKNOWN` — filler + punctuation only, no medical
  normalisation. Safer default than guessing role.

**Provenance invariant**: `CleanedSegment.original_text` is ALWAYS the raw
ASR output, regardless of what the cleanup LLM returns. The claim extractor
consumes `cleaned_text`; every emitted claim carries a chain back to
`original_text` per `rules.md §4`.

Cleanup model is `gpt-4o-mini` for the demo path (`DemoCleanupConfig`) and
`qwen2.5-7b-local` for eval throughput (`EvalCleanupConfig`). Opus 4.7 is
never a cleanup model (`Eng_doc.md §3.5` "Tank for the war" policy;
enforced by `config.py::_guard_no_opus`).

### E. Dormancy on text-input eval paths

Load-bearing invariant. ACI-Bench and LongMemEval feed the substrate
**already-clean text**, not raw audio. Running `TranscriptCleaner` on top
of that text has two failure modes:

1. **Paraphrase drift** — the cleanup LLM "helpfully" re-wraps clinical
   phrasing, which invalidates apples-to-apples comparison against the
   published ACI-Bench / LongMemEval baselines.
2. **Silent extra LLM call** — an uncontrolled cost and latency leak per
   eval question.

`PipelineConfig.bypass_cleanup_for_text_input: bool = True` (default!)
short-circuits the cleanup stage in
`AsrPipeline._transcribe_inner` before `TranscriptCleaner` is even
constructed. The default is deliberately **`True`** — the only path that
opts in is the raw-audio demo, which must flip the flag deliberately. Any
new eval harness inherits the dormant-cleanup guarantee for free.

**Regression guard**:
`tests/unit/extraction/test_text_input_dormancy.py` patches
`TranscriptCleaner` and asserts `call_count == 0` when bypass is True,
`call_count == 1` when bypass is False (negative control), and the default
value is `True` (defence-in-depth invariant).

See also §7 below.

### F. Honest TBM handling in `docs/asr_benchmark.md`

Before this dispatch, `docs/asr_benchmark.md §4` carried three rows with
every cell marked `TBM` (to be measured). Three TBM rows look superficially
like a results table that just needs filling in — a visually deceptive
artefact.

This dispatch replaces those three rows with a single pointer line:

> "Measurement pending — GPU run scheduled; see
> `docs/asr_engineering_spec.md` §2 (success criterion A) and §4
> (latency budget) for the measurement plan."

The methodology + reproduction sections of `docs/asr_benchmark.md` (which
are non-speculative) remain intact. Directional-outcome hypotheses (§4.1
"Expected directional outcomes") also remain — they are explicitly labelled
as hypotheses, not measurements. Recorded in `reasons.md` 2026-04-22
"TBM rows replaced with single-line pointer".

---

## 3. Failure-mode coverage

Known failure modes and their current coverage status.

| Failure mode | Current coverage | Gap / mitigation |
|---|---|---|
| Cross-talk (simultaneous speakers) | pyannote community-1 handles overlapping speech via its overlap-detection head. | Unmeasured on our fixtures. ADR-able. |
| Background hospital noise | silero-vad trims boundary silence; Whisper large-v3 is trained with noise augmentation. | Noise-robust WER not measured. Open ask: one synthetic clip with overlaid ward ambience. |
| Accented speakers | Whisper large-v3 is multilingual + accent-robust in training data. | No accented-speaker synthetic clip yet — Koenecke et al. 2024 document significant WER disparities across demographic groups; this is a known gap. |
| Rapid clinician speech (monologue) | `extreme_compression` guard caps at 15 chars/s; fast clinicians may legitimately trip this as a WARN. | Threshold re-calibratable per pack; default kept conservative. |
| Whisper loop-pathology on silence | `repeated_ngram` guard + `no_speech_threshold=0.6` + boundary silence trim in preprocess. | 3-layer defence; this is the failure mode with strongest coverage. |
| Medical confabulation | `oov_medical_term` + `invented_medication` + RxNorm post-check. | RxNorm currently ~30-drug placeholder; full UMLS-licenced dump is a hot-swap upgrade. |
| Diariser collapse to 1 speaker | Diariser returns single-speaker stream; `_guess_speaker_role` falls back to "unknown". | Degrades to single-speaker mode rather than crashing; cleanup uses the `SYSTEM_PROMPT_UNKNOWN` neutral prompt. |

---

## 4. Latency budget per stage

Per-stage instrumentation via `src/extraction/asr/telemetry.py::@measure`.
Targets below are P95 on GCP L4 24 GB unless otherwise noted, copied from
`docs/asr_performance_spec.md` for anchor-ability.

| Stage | P50 target | P95 target | Wraps |
|---|---:|---:|---|
| `normalize` | 100 ms | 300 ms | `preprocess.normalize_audio` (ffmpeg loudnorm) |
| `trim` | 50 ms | 200 ms | `preprocess.trim_silence` (ffmpeg silenceremove) |
| `transcribe` | 600 ms | 2000 ms | `_Transcriber.transcribe` (faster-whisper + CT2 INT8_FLOAT16) |
| `diarise` | 400 ms | 1500 ms | `_AlignerDiarizer.align_and_diarise` (WhisperX + pyannote) |
| `cleanup` | 200 ms | 500 ms | `TranscriptCleaner.clean` (gpt-4o-mini; DORMANT on text-input) |
| `guard` | 1 ms | 5 ms | `hallucination_guard.check` (pure Python) |
| `correct` | 10 ms | 100 ms | `word_correction.correct_medical_tokens` (rapidfuzz Levenshtein) |
| **End-to-end utterance-to-claim** | **5.0 s** | **7.0 s** | includes downstream Opus 4.7 claim extraction |

Aggregate statistics at any time: `telemetry.get_stats()` returns
`{count, p50, p95, p99, max}` per stage. No threading overhead — the pipeline
is single-threaded per session.

---

## 5. Engineering moves

### 5.1 VAD tuning

`silero-vad v5` (MIT; self-contained 2 MB ONNX model) runs ahead of
Whisper. Two reasons for an explicit VAD gate rather than relying on
Whisper's internal `vad_filter`:

1. **Silent-frame hallucination suppression**: Whisper confabulates on
   silence; trimming silent boundaries before transcription is the cheapest
   defence.
2. **Clinical audio is pause-heavy**: typical conversational speech has
   ~15% silence; a clinical exam (patient pauses, physician writes notes)
   runs 20–30%. Silero's threshold is tuned for general speech; clinical
   audio benefits from a slightly lower threshold so we don't clip
   mid-utterance pauses. We currently ship silero's defaults; the first
   round of GPU-run feedback will inform whether we override.

### 5.2 WhisperX alignment quality

WhisperX (BSD-2-Clause) uses wav2vec2 forced alignment for **word-level**
timestamps, not just segment-level. This is load-bearing for:

- Per-speaker word attribution (one WhisperX segment can straddle a
  speaker change; word-level alignment lets us split correctly).
- Provenance: every claim's `t_start` / `t_end` bound maps to a specific
  word window, not just an utterance.

The collapse step (`_collapse_turns`) merges consecutive same-speaker
WhisperX segments into one turn — so one utterance = one Turn downstream.

### 5.3 Diarisation

`pyannote/speaker-diarization-community-1` (CC-BY-4.0) is the current
baseline. Licensing evolution tracked in `reasons.md` 2026-04-21
"Strict OSI-only licensing for model weights" entry. Alternative under
watch: NVIDIA NeMo Sortformer (Apache-2.0); swap reserved for the case
where CC-BY-4.0 is ruled out by the hackathon licensing committee
(low-probability).

No newer community-1 successor has shipped at the time of writing; we'll
re-evaluate at the next release. A "Sortformer-2026" release would be an
automatic candidate if it publishes similar or better DER on clinical
multi-speaker audio under Apache-2.0.

### 5.4 Contextual biasing

`src/extraction/asr/vocab.py::build_initial_prompt` assembles the
`initial_prompt` from three sources: RxNorm top-100 medications,
AHA/ACC 2021 chest-pain guideline key terms, ICD-10 anchors. Budget: 224
tokens (Whisper's decoder budget is 448; we reserve half for the prompt
to guarantee no output truncation — see `reasons.md` 2026-04-21 "ASR
prompt length").

**Primary citation**: Arora et al., "Contextual Biasing for Clinical ASR",
[arXiv 2502.11572](https://arxiv.org/abs/2502.11572) (Jogi, Aggarwal et al.,
2025). Their approach uses *training-time* biasing loss (40–60% relative
rare-term WER reduction); ours is prompt-only (weaker; expected 3–8 pp
absolute). Honest gap documented in §2.C above.

---

## 6. Per-speaker cleanup — worked examples

From `src/extraction/asr/transcript_cleanup.py`. The following illustrate
how the role-aware prompts handle representative utterances.

### Doctor segment — numeric / unit normalisation

```
raw:     "BP one-twenty over eighty, heart rate eighty-two, O2 sat ninety-eight percent"
cleaned: "BP 120/80 mmHg, HR 82, SpO2 98%"
```

The doctor prompt is licensed to normalise numeric phrasing to clinical
shorthand; this is the kind of transform that costs nothing in meaning
and saves tokens downstream. A `CleanupCorrection` row is recorded for
each span with `reason_category = "grammar"` or `"medical_term_correction"`.

### Doctor segment — medical misspelling correction

```
raw:     "prescribed met-oh-pro-lol 50 milligrams twice daily and aspirine 81"
cleaned: "prescribed metoprolol 50 mg twice daily and aspirin 81 mg"
```

Numbered-vocabulary injection (OpenWhispr pattern) biases the cleanup
LLM toward authoritative spellings when the raw ASR produces phonetic
approximations. The raw text is preserved on `CleanedSegment.original_text`
for provenance.

### Patient segment — verbatim preservation + lay-language tag

```
raw:     "my chest felt like someone was sitting on it, right here"
cleaned: "my chest felt like someone was sitting on it [likely: pressure-type
          chest pain], right here"
```

The patient prompt is **not** licensed to replace "someone sitting on my
chest" with "pressure-type chest pain"; the `[likely: …]` tag makes the
medical interpretation available to the claim extractor while preserving
the patient's exact words. This is the specific feature that sets us
apart from single-speaker dictation: the patient's vocabulary is clinical
evidence, not noise to be cleaned.

### Cross-reference across turns

```
turn N-4  doctor:   "I'll start you on amlodipine 5 mg for the blood pressure."
turn N    patient:  "my blood pressure pill"
```

The conversation context buffer (last `max_context=5` cleaned turns)
carries the doctor's "amlodipine 5 mg" into turn N's user message, so
the patient's cleanup pass can tag: "my blood pressure pill
[likely: amlodipine 5 mg]". The tag is best-effort; it does not override
the patient's phrase. Downstream, the claim extractor sees both the
verbatim phrase and the likely medical referent.

---

## 7. Dormancy assertion

Code-level guarantee (reprised from §2.E for spec-completeness):

```python
# src/extraction/asr/pipeline.py, _transcribe_inner
if self.config.bypass_cleanup_for_text_input:
    logger.info(
        "asr_pipeline.cleanup_bypassed_for_text_input",
        reason="bypass_cleanup_for_text_input=True",
    )
elif self.config.enable_cleanup:
    try:
        cleaner = TranscriptCleaner(...)
    except Exception as exc:
        logger.warning("asr_pipeline.cleaner_init_failed", error=str(exc))
```

Three invariants this configuration enforces:

1. **Default-dormant**: `PipelineConfig.bypass_cleanup_for_text_input: bool = True`.
   Any `PipelineConfig()` constructed without keyword arguments inherits
   the dormant default.
2. **Short-circuit before construction**: the cleaner is never even
   *instantiated* when bypass is True. This matters because
   `TranscriptCleaner.__init__` does NOT make an LLM call — but belt-and-braces
   is the right posture for an eval-integrity invariant.
3. **Regression test guards the invariant**:
   `tests/unit/extraction/test_text_input_dormancy.py` patches
   `src.extraction.asr.pipeline.TranscriptCleaner`, runs a full pipeline
   pass, asserts the mock's `call_count == 0`. A negative control test
   (bypass=False) asserts `call_count == 1`. A default-value test asserts
   the flag defaults to True.

ACI-Bench / LongMemEval harnesses construct pipelines without touching this
flag, inherit the dormant default, and therefore never accidentally run
`TranscriptCleaner` on pre-cleaned text. The only place that must opt in
is the raw-audio demo path (and any future live-audio eval harness).

See `reasons.md` 2026-04-22 entry "bypass_cleanup_for_text_input default
True — defence-in-depth".

---

## 8. Open asks for the human

Three asks, prioritised.

### 8.1 GPU instance + cloud + scheduling window for the measurement run

Primary: GCP L4 24 GB spot, us-central1, ~60 minutes.
Secondary: Modal A10 via profile `glitch112213`, same ~60-minute budget.

Deliverables from the run, populated into `docs/asr_benchmark.md §4` and
cross-linked from this spec:

- RTF (P50 / P95) for all three variants (A baseline, B biased, C distil).
- Overall WER and medical-term WER for all three variants.
- End-of-utterance latency for variant B (the production path).
- Per-stage telemetry (normalize / trim / transcribe / diarise / cleanup
  (bypassed in eval) / guard / correct).
- Hardware profile: GPU model, CUDA version, driver, CTranslate2 version,
  faster-whisper version, WhisperX version.

### 8.2 Additional synthetic clips beyond the chest-pain demo

Closed 2026-04-22. Five text scripts landed under `eval/synthetic_clips/`
covering: abdominal pain, dyspnea, headache, fatigue + weight loss, and
dizziness + syncope. Each script contains a two-speaker dialogue, a
biasing-vocabulary block, an explicit supersession moment, and the expected
claim-extraction targets keyed to the `clinical_general` predicate families.
Scripts are text-only (no TTS rendered); TTS rendering + `SYNTHETIC_DATA.md`
declaration remain open for whenever the generalisation-claim measurement
run is scheduled. One "ambient hospital noise" clip (same chest-pain
transcript overlaid with public-domain ward ambience) also remains open for
the noise-robustness claim under §3.

### 8.3 Demo video script confirmation — WisprFlow one-liner framing

Decision 2 in the parallel-execution plan says the demo-video narration
includes the WisprFlow one-liner verbatim (or a close paraphrase). This
spec has no direct control over the video; confirmation from the operator
that the line is in the shooting script before the voice-over record is
the ask. Canonical phrasing:

> "Single-person dictation tools like WisprFlow don't solve the two-speaker
> clinical problem; this does."

If the one-liner is cut from the video, the canonical distinction this
spec embodies loses its on-screen anchor, and judges watching the 3-minute
video will not understand why a diarised pipeline exists. This is a
product-narrative ask, not a code ask.

---

## 9. Change log

| Date | Change | Author |
|---|---|---|
| 2026-04-22 | Initial spec landed alongside `bypass_cleanup_for_text_input` flag, per-check hallucination tests, and trimmed `asr_benchmark.md §4`. | wt-asr-spec |
