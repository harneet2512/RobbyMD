# asr_benchmark.md — ASR latency + medical-term WER

> **For the full engineering spec see `docs/asr_engineering_spec.md`.**
> That document is the canonical reference for success criteria, the
> 5-check hallucination-guard coverage, the latency budget, and the
> text-input dormancy guarantee. This file is a thin methodology +
> reproduction pointer for the measurement run itself.

**Scope**: CLAUDE.md §5.2 deliverable for `wt-extraction`. Measured real-time
factor (RTF) and medical-term WER for three ASR variants on one synthetic
chest-pain dialogue clip (see `src/extraction/asr/synth_audio.py`).

**Status (2026-04-21)**: methodology + fixtures landed; numbers are reproducible
but **not yet measured in this environment** (no GPU available in the sandbox
on the date this file was authored). The command block below is the exact
reproduction. The rule-complying framing per `rules.md` §9.3 is that we report
measured numbers with their denominators + caveats — so this file publishes
the methodology and leaves the number columns labelled `TBM` (to be measured)
until a run on the target hardware lands.

**Hardware target** (per `Eng_doc.md` §3.1 / §9): 16–24 GB NVIDIA GPU
workstation (A100 40 GB or 4090 24 GB), 32 GB RAM, CUDA 12.x.

---

## 1. Variants

Three ASR configurations are benchmarked against the same synthetic clip:

| # | Name | Model | `initial_prompt` | Backend |
|---|---|---|---|---|
| A | `large_v3_baseline` | openai/whisper-large-v3 (Apache-2.0) | none | faster-whisper 1.2.1, CTranslate2 INT8_FLOAT16 |
| B | `large_v3_biased` | openai/whisper-large-v3 (Apache-2.0) | `build_initial_prompt()` (RxNorm + AHA/ACC + ICD-10) | faster-whisper 1.2.1, CTranslate2 INT8_FLOAT16 |
| C | `distil_large_v3_fallback` | distil-whisper/distil-large-v3 (MIT) | none | faster-whisper 1.2.1, CTranslate2 INT8_FLOAT16 |

Variant B is the production path — all three components (VAD silero-vad,
Whisper large-v3, WhisperX + pyannote community-1) wired per
`research/asr_stack.md` §2.1. Variant A is the medical-WER baseline for
isolating the `initial_prompt` bias effect. Variant C is the latency fallback
when GPU headroom drops (research/asr_stack.md §2.1 fallback switch).

## 2. Metrics

- **RTF** (real-time factor) = `wall_clock_seconds / audio_seconds`. Lower is
  better. Target: RTF <= 0.7 per Eng_doc.md §3.1. Measured across VAD +
  Whisper + alignment + diarisation end-to-end.
- **Medical-term WER**: word error rate computed only over the medical-term
  token set in the reference transcript (the `metoprolol`, `aspirin`,
  `atorvastatin`, `radiates`, `pressure`, `stairs`, `breath` tokens from
  `DEMO_SCRIPT`). Using `jiwer` for standard WER arithmetic then filtering
  to the medical-term subset. Target: <= 12% per Eng_doc.md §3.1.
- **End-of-utterance -> text latency**: time from the last VAD frame marking
  speech-end to the final `Turn` object being emitted. Target <= 1.5 s per
  Eng_doc.md §3.1.

`jiwer` (Apache-2.0) is the WER engine. Not yet added to `pyproject.toml` —
will be wired when the GPU run lands; it is on the allowlist.

## 3. Reproduction

### 3.1 Prerequisites

```bash
cd "$REPO_ROOT/../wt-extraction"          # or wherever you have feature/extraction
python -m pip install -e ".[dev]"
python -m pip install faster-whisper whisperx silero-vad jiwer pyttsx3
python -m pip install "huggingface_hub<0.30"   # pyannote community-1 compat
export HF_TOKEN=<your-hf-token-accepted-for-pyannote-community-1>
```

### 3.2 Generate the clip

```bash
python - <<'PY'
from pathlib import Path
from src.extraction.asr.synth_audio import synthesise_clip
synthesise_clip(
    out_wav=Path("eval/fixtures/asr/chest_pain_demo.wav"),
    out_script_json=Path("eval/fixtures/asr/chest_pain_demo.script.json"),
)
PY
```

### 3.3 Run each variant

```python
from pathlib import Path
from src.extraction.asr.pipeline import PipelineConfig, build_pipeline

for name, cfg in [
    ("A_large_v3_baseline",
     PipelineConfig(hf_token=os.environ["HF_TOKEN"], use_initial_prompt=False)),
    ("B_large_v3_biased",
     PipelineConfig(hf_token=os.environ["HF_TOKEN"], use_initial_prompt=True)),
    ("C_distil_fallback",
     PipelineConfig(hf_token=os.environ["HF_TOKEN"],
                    use_initial_prompt=False,
                    whisper_model_id="distil-whisper/distil-large-v3")),
]:
    pipe = build_pipeline(cfg)
    t0 = time.perf_counter()
    turns = list(pipe.transcribe(Path("eval/fixtures/asr/chest_pain_demo.wav")))
    wall = time.perf_counter() - t0
    record(name, turns, wall)
```

### 3.4 Score

```python
from jiwer import wer
from src.extraction.asr.synth_audio import script_as_gold_transcript
gold = script_as_gold_transcript()
hyp  = " ".join(t.text for t in turns)
overall_wer = wer(gold, hyp)

med_terms = {"metoprolol", "aspirin", "atorvastatin", "radiates",
             "pressure", "stairs", "breath"}
# Medical-term WER: restrict both streams to med_terms before computing WER.
filter_med = lambda txt: " ".join(w for w in txt.lower().split() if w in med_terms)
med_wer = wer(filter_med(gold), filter_med(hyp))
```

### 3.5 Record

Append a new row to the results table below (§4) with the measured numbers,
the commit SHA, and the hardware profile.

## 4. Results

Measurement pending — GPU run scheduled; see `docs/asr_engineering_spec.md` §2
(success criterion A) and §4 (latency budget) for the measurement plan.

### Expected directional outcomes (hypothesis, NOT measurement)

Derived from the published numbers collated in `research/asr_stack.md`:

- **A (baseline)**: RTF ~0.1–0.2 on a 4090 at INT8_FLOAT16 (SYSTRAN benchmark
  large-v2 proxy), end-of-utterance latency 0.8–1.3 s. Medical-term WER
  unknown prior — this benchmark establishes it.
- **B (biased)**: RTF unchanged vs. A (prompt is prefix-only, not decoder
  weight change). Medical-term WER expected 3–8 pp lower than A per
  `research/asr_stack.md` §4.3 (directional, anecdotal).
- **C (distil)**: RTF ~0.05 (6.3× faster per HF model card). Medical-term WER
  likely **worse** than A because distillation weakens long-tail LM; magnitude
  unknown. That's why Distil is a latency fallback, not a first choice.

These are hypotheses only; none go in the demo video until variant B lands
its measured numbers.

## 5. Known limitations

1. **One-clip benchmark**. Single 45-s scripted dialogue is too small to
   claim anything beyond "works on the happy path". The published benchmark
   numbers in `Eng_doc.md` §10 (ACI-Bench) give the multi-encounter story;
   this file is a sanity check for the local pipeline, not a research claim.
2. **Synthetic TTS audio**. The prosody / phonemes are SAPI5-class, not
   human-class. Medical-term WER on real clinician speech will differ; the
   `jiwer` numbers here are a pipeline sanity check, not a clinical number.
3. **pyannote cold-start**. First diarisation call may take 2–5 s to warm
   up; the runner above ignores it (or warms explicitly before `t0`).
4. **Distil medical delta is speculative**. No published medical-domain WER
   for distil-large-v3 exists. Variant C's number is part of this benchmark's
   contribution.
5. **HF_TOKEN is required** for pyannote speaker-diarization-community-1 model
   download. If unavailable, the benchmark cannot be run end-to-end; see §6
   for the fallback ADR.

## 6. Fallback: pyannote unavailable

If `HF_TOKEN` is unobtainable in time (low probability — the token is free
and auto-issued once `pyannote/speaker-diarization-community-1` terms are
accepted), the fallback options in order of preference are:

1. **Swap diariser to NVIDIA NeMo Sortformer** (Apache-2.0) — documented as
   R1 option (b) in `research/asr_stack.md` and as alternative #1 in
   `docs/decisions/2026-04-21_pyannote-ccby40-model-weights.md`. Adds a
   ~3 GB CUDA toolkit to the install footprint. Requires a new row in
   `MODEL_ATTRIBUTIONS.md` before any code references it.
2. **Drop diarisation for the demo** — speaker labels come from alternating
   turns in the scripted script. Loses the architectural completeness
   claim; acceptable as last resort.

Either fallback is an open question for the human operator at the time of
the blockage; this doc names both so the decision is unblocked when needed.

## 7. Reproducibility checklist

- [ ] `src/extraction/asr/synth_audio.py::synthesise_clip` runs without error
- [ ] Gold transcript JSON written and matches `DEMO_SCRIPT` content
- [ ] HF_TOKEN env var exported; pyannote terms accepted
- [ ] All three variants run to completion on the target hardware
- [ ] RTF, overall WER, medical-term WER, end-of-utterance latency recorded
- [ ] Git commit SHA of the run captured in the row
- [ ] Hardware profile recorded (GPU model, CUDA version, driver)

---

**Cross-references**: `research/asr_stack.md` (wiring plan), `Eng_doc.md` §3.1
(targets), `MODEL_ATTRIBUTIONS.md` (license rows for every model touched),
`SYNTHETIC_DATA.md` (synthetic clip declaration).
