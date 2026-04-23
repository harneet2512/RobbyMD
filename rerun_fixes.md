# Bundle 4 (Flow Variant A) — re-run fix list

**Written**: 2026-04-23
**Branch**: `bundle4-variant-a-run` (base for the next iteration)
**Last measurement**: `eval/flow_results/variant_a/20260423T163104Z/results.json` (commit `8d3f634`)

First run shipped real measured numbers on 6 synthetic clinical dialogues, but with two defects that make the result non-demo-ready. This document lists exactly what needs to change before the next measurement run, why, and how to verify.

---

## Issue 1 — Cleanup stage is making WER worse

**Symptom (measured on last run):**
- WER raw (Whisper-only): **12.1 %**
- WER cleaned (+ BioMistral): **19.9 %**

The cleanup LLM is actively destroying transcription accuracy. This is a quality regression in the cleanup stage itself — Whisper's output is better than Whisper + BioMistral.

**Root cause (ranked):**

1. **The prompt invites paraphrasing.** Current system prompt (`src/extraction/flow/variant_a/pipeline.py::CLEANUP_SYSTEM_PROMPT`):
   > *"remove filler words and disfluencies, **normalize medical terminology to standard forms**, fix obvious transcription errors in drug names and anatomical terms, and preserve every clinically relevant claim."*
   The word "normalize" tells the model to rewrite `"I got short of breath"` → `"Patient experiencing dyspnea"`. That's a rewrite, not a cleanup. WER is token-level, so valid-but-different phrasings count as errors.

2. **BioMistral's pretraining bias.** Continued-pretrained on PubMed Central (formal medical prose). Its maximum-likelihood response to a conversational segment is to produce formal medical writing.

3. **No "copy if clean" escape hatch.** The prompt never says *"if the segment has no filler words, return it verbatim."* An instruction-tuned model assumes it must always improve the input.

4. **`max_tokens = 2 × input length`** (in `cleanup_segment`) gives the model room to expand. Even at `temperature=0` this invites adding clarifiers.

5. **Per-segment cleanup, no cross-segment context.** Short Whisper segments (2-5 words) get rewritten because the LLM has almost no context to anchor on.

**Fix — code changes to `src/extraction/flow/variant_a/pipeline.py`:**

### (a) Tighter prompt (subsequence-only)

```python
CLEANUP_SYSTEM_PROMPT = (
    "You are a transcript cleaner. Your output must be a subsequence of the "
    "input: you may only delete filler tokens (uh, um, er, ah, like, you know) "
    "and obvious duplicate words. You must NOT paraphrase, NOT reorder, NOT "
    "normalize terminology, NOT expand abbreviations, NOT add clarifications, "
    "NOT add punctuation that was absent. If the input has no filler, return "
    "it byte-for-byte unchanged. Output only the cleaned text, no preface."
)
```

### (b) Cap `max_tokens` at input length, not 2×

In `cleanup_segment`:

```python
"max_tokens": max(16, len(raw_text.split()) + 4),  # +4 for tokenizer slack
```

### (c) Fast-path skip when no filler detected

At the top of `cleanup_segment`:

```python
_FILLER_RE = re.compile(r"\b(?:uh+|um+|er+|ah+|like|you know)\b", re.I)

def cleanup_segment(self, raw_text: str) -> str:
    if not raw_text.strip():
        return ""
    if not _FILLER_RE.search(raw_text):
        return raw_text  # no filler tokens — pass through unchanged
    # ... existing LLM call ...
```

This eliminates LLM calls on clean segments entirely (likely the majority for TTS-rendered audio which has no spontaneous disfluencies), and removes the biggest source of rewriting.

**Acceptance criterion for issue 1:**
- `wer_cleaned_mean` < `wer_raw_mean` on the same 6 clips.
- Target: cleaned WER within 0.5 percentage points of raw WER (since TTS clips have no real disfluencies, the cleanup stage should be a near-no-op).
- Medical-term WER should stay ≤ 6.7 % (not regress from the baseline that raw Whisper already achieved).

---

## Issue 2 — DER = `null` (diarisation disabled)

**Symptom:** The brief's required DER metric is not produced. Speaker labels fall back to an alternating-turn heuristic in `pipeline.py::run()` (`enable_diarisation=False` by default).

**Root cause — hard dependency conflict:**
- `pyannote/speaker-diarization-community-1` declares `pyannote.audio >= 4.0.0` in its `config.yaml`.
- `pyannote.audio 4.0.x` requires `torch >= 2.8`.
- `vllm 0.6.3` pins `torch == 2.4.0` (hard pin via xformers 0.0.27).
- One Python venv cannot hold torch 2.4 and torch 2.8 simultaneously. Installing one uninstalls the other.

On the last run we chose to keep vLLM (medical LLM is load-bearing) and disabled the diariser.

**Fix — upgrade vLLM to a torch-2.8-compatible release:**

### (a) Update `scripts/_bundle4_l4_setup.sh` (or a new `_b4_upgrade.sh`)

Change the pip-install block to:

```bash
pip install --upgrade \
    "torch==2.8.*" "torchaudio==2.8.*" \
    --index-url https://download.pytorch.org/whl/cu121

pip install --upgrade \
    "faster-whisper>=1.2.1" \
    "pyannote.audio>=4.0.0" \
    "kokoro>=0.9.2" \
    "soundfile==0.12.1" \
    "pydub" \
    "librosa" \
    "jiwer==3.0.4" \
    "pyannote.metrics>=4.0.0" \
    "huggingface_hub" \
    "requests"

pip install --upgrade "vllm>=0.8.0"  # torch-2.8 compatible
```

### (b) Re-enable diarisation in `pipeline.py::VariantAPipeline.__init__`

Flip the default:

```python
def __init__(
    self,
    ...,
    enable_diarisation: bool = True,  # was False
) -> None:
```

### (c) Adjust `b4_measure.sh` vLLM flags for the new version

vLLM 0.8+ no longer needs `--enforce-eager` and supports a wider `--gpu-memory-utilization`. Use:

```bash
vllm serve BioMistral/BioMistral-7B-DARE-AWQ-QGS128-W4-GEMM \
    --served-model-name biomistral-7b-dare \
    --host 127.0.0.1 --port 8000 \
    --quantization awq --dtype float16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.55 \
    --disable-log-requests
```

### (d) Re-measure DER

Once diarisation is back, `measure.py::measure_one` already handles the DER computation when `result.get("diarisation_enabled")` is true. No code change needed there.

**Acceptance criterion for issue 2:**
- `results.json["der_mean"]` is a real number (not `null`).
- `n_ok == 6`, `n_failed == 0`.
- `results.json["stack"]` string no longer says "Diarisation deferred".

---

## Other resolved-during-last-run items to preserve

These aren't re-fixes — they're lessons the next run should inherit so it doesn't re-discover them:

1. **Use `BioMistral/BioMistral-7B-DARE-AWQ-QGS128-W4-GEMM`** (official org), not LoneStriker's mirror. LoneStriker ships a bogus `model.safetensors.index.json` pointing at non-existent shards (merge-kit artefact); vLLM's HF-API refetch path honours it even if you delete it locally.

2. **pyairports stub**: vLLM's outlines integration imports `pyairports`, PyPI's `pyairports==0.0.1` is a namespace squat with no `airports` submodule. The setup script needs to create the stub after pip install:

   ```bash
   SP="$HOME/robbymd/.venv-flow-a/lib/python3.10/site-packages"
   mkdir -p "$SP/pyairports"
   printf '' > "$SP/pyairports/__init__.py"
   printf 'AIRPORT_LIST = []\nclass Airports:\n    def __init__(self,*a,**k): pass\n    def lookup(self,*a,**k): return None\n' > "$SP/pyairports/airports.py"
   ```

   (Check if vLLM 0.8+ fixed this transitive; the stub can likely be dropped with the upgrade.)

3. **Mistral chat template rejects `system` role** — already fixed in `pipeline.py::cleanup_segment` (system prompt merged into user message). Keep as-is; don't regress.

4. **Kokoro TTS renders on CPU** — `KPipeline(lang_code="a", device="cpu")` in `render_tts.py`. The CUDA path crashed with `Cannot load symbol cublasLtCreate` against torch 2.4. Re-test on torch 2.8; if CUDA works, flip to GPU (saves a few minutes per render cycle — tiny model, not a big deal either way).

5. **Stage markers under `~/robbymd/.b4_markers/`** — the resumable setup script is already idempotent. Do NOT blow them away on restart; that's what lets us resume in 10 minutes instead of 60 after a spot preemption.

---

## Re-run procedure (executable checklist)

**Pre-conditions:**
- Free L4 GPU available (see GPU-availability section below)
- HF token accepted for `pyannote/speaker-diarization-community-1` (already done on Aravind's HF account)
- Branch `bundle4-variant-a-run` on origin has the code fixes landed locally + pushed

**Steps:**

1. **Apply code fixes locally on laptop**, commit, push to `bundle4-variant-a-run`:
   - `src/extraction/flow/variant_a/pipeline.py`: new CLEANUP_SYSTEM_PROMPT, max_tokens cap, filler fast-path, `enable_diarisation: bool = True`.
   - `scripts/_bundle4_l4_setup.sh`: pin torch 2.8, pyannote 4.0, vllm 0.8.
   - `scripts/_bundle4_l4_measure.sh`: drop `--enforce-eager` if kept; keep the rest.

2. **Decide which L4 to use** (see availability section). Preferred: whichever of `aravind-l4-gpu` / `harneet-l4-gpu` comes back from spot preemption first (persistent disk has cached models on both — faster resume).

3. **Wipe stage markers** (force full re-setup because torch/vllm/pyannote all change major versions):
   ```bash
   gcloud compute ssh <INSTANCE> --zone=<ZONE> --project=<PROJECT> \
     --command='rm -rf /home/Lenovo/robbymd/.b4_markers /home/Lenovo/robbymd/.venv-flow-a'
   ```

4. **Pull the fixed code + relaunch setup in tmux:**
   ```bash
   gcloud compute ssh <INSTANCE> --zone=<ZONE> --project=<PROJECT> --command='
     cd /home/Lenovo/robbymd && git fetch origin --quiet && \
     git reset --hard origin/bundle4-variant-a-run && \
     tmux kill-server 2>/dev/null; tmux new-session -d -s b4 && \
     tmux send-keys -t b4 "export HF_TOKEN=... && bash /home/Lenovo/b4_setup.sh" Enter'
   ```

5. **Wait for `DONE_SETUP`**, then launch `b4_measure.sh` in its own tmux session. Same poller scripts (`_poll_l4_setup.sh` → `_poll_l4_measure.sh`) apply.

6. **Expected new timing** (with vLLM 0.8 + full new pip install): setup ~60-75 min (torch+vllm+pyannote are all re-downloaded), measurement ~8-15 min (diarisation adds ~1-2 s per clip, cleanup LLM calls drop dramatically due to filler fast-path).

7. **Pull results** (`gcloud compute scp ...`), commit to `bundle4-variant-a-run` (local commit, then `git push` from laptop because L4 has no GitHub creds), append a progress.md entry dated 2026-04-24 (or whenever the re-run happens) with before/after WER + real DER.

**Acceptance criteria for the re-run to be declared "done":**
- `wer_cleaned_mean` < `wer_raw_mean` (cleanup now helping, not hurting)
- `medical_term_wer_mean` ≤ 0.07 (didn't regress from raw-Whisper baseline)
- `der_mean` is a real number (not null); reasonable target 0.10-0.18 on clean two-speaker TTS
- `n_ok == 6`, `n_failed == 0`
- `results.json["stack"]` no longer references "Diarisation deferred"
- progress.md updated with before/after comparison
- branch pushed, commit hash recorded

---

## GPU availability (as of 2026-04-23T16:40Z)

| Instance | Project | Zone | Status | Notes |
|---|---|---|---|---|
| **aravind-l4-b5** | Aravind | europe-west4-b | RUNNING | Presumably Bundle 5's — check if idle before using |
| aravind-l4-gpu | Aravind | europe-west4-a | TERMINATED | Persistent disk has full venv + models + markers from last run — fastest resume target |
| harneet-l4-gpu | Harneet | us-central1-a | TERMINATED | Fresh state, full setup cost if restarted |

Both terminated instances are currently stockout in their zones. Spot capacity returns sporadically; previous observation was europe-west4-a came back within minutes on one attempt and 15+ on others. A 90-second-interval poller is the right pattern — already scripted at `scripts/_wait_for_l4.sh`.

**Plan for actually firing the re-run:**

1. **When a free instance appears**: start `scripts/_wait_for_l4.sh` in the background; it writes `/tmp/l4_winner.env` and exits when either preempted L4 comes back.
2. **After the caveat-fix code is on origin**: wipe `.b4_markers` on the winning instance, pull, relaunch setup.
3. **While setup runs (60-75 min)**: I draft the "after" progress.md entry template so we can fill in numbers immediately on completion.

Estimated wall-clock from "L4 available" to "results committed + pushed": ~90-100 minutes.

---

## Deeper review — findings the two headline issues hid

Second-pass review through per-clip metrics, raw-vs-cleaned hypotheses, and code. Ordered by how much each matters for the next run.

### D1. The cleanup regression is concentrated on medically-dense clips

Per-clip `wer_cleaned - wer_raw`:

| Scenario | Δ | Comment |
|---|---:|---|
| chest_pain | **+0.17** | cleanup doubled WER |
| abdominal_pain | **+0.23** | cleanup more than doubled WER |
| dizziness_syncope | +0.05 | small damage |
| dyspnea | +0.02 | near-no-op |
| fatigue_weight_loss | +0.01 | near-no-op |
| headache | **−0.01** | cleanup slightly helped |

Looking at the actual outputs:

- `chest_pain` RAW (first sentence): `I've been having this pressure in my chest. It started about two hours ago.` → CLEANED: `Pressure in chest started two hours ago.` — compressed to medical-note shorthand.
- `abdominal_pain` RAW: `I've had this pain in my belly since last night. It's getting worse.` → CLEANED: `Belly pain since last night, worsening.` — same compression pattern.
- `headache` CLEANED: first-person flipped to second-person — `I've got a really bad headache` → `You have a really bad headache`.

**Insight:** BioMistral's PubMed bias kicks in hardest on medically-dense content. PubMed is full of clinical case notes like `Patient reports pressure in chest for 2h`, not patient-voice transcripts. The model is doing maximum-likelihood medical-note reformatting — and that's exactly the content-type that PubMed trained it on.

**Implication for the fix:** the subsequence-only prompt in Issue 1 is the right call, but the **filler-regex fast-path is especially important** — the chest_pain and abdominal_pain raw Whisper outputs had zero disfluencies, so the LLM should never have been called at all. The fast-path would eliminate both of the worst cases entirely.

### D2. `first_token_ms` doesn't measure streaming first-token latency

Current code in `pipeline.py::transcribe`:

```
t0 = time.perf_counter()
generator, _info = self.whisper.transcribe(audio_path, ...)
for seg in generator:
    if first_token_ms is None:
        first_token_ms = (time.perf_counter() - t0) * 1000.0
```

faster-whisper's generator yields after the first 30-second audio chunk is processed. For a 60-90 s clip with beam_size=5, the first yield is hundreds of ms to a few seconds in. **This is first-segment-after-ingest latency, not streaming first-word latency.**

Wispr Flow's published "700 ms p99" is true streaming: audio streamed in, partial hypothesis out before audio ends. Our 832 ms p50 / 1227 ms p90 is not apples-to-apples with that number.

**Fix (small code change):**

- Rename the metric to `first_segment_after_ingest_ms` everywhere in `measure.py` and `results.json`.
- For a true streaming-first-word comparison, add a second measurement path: pre-chunk the audio into 5-second overlapping windows, call `transcribe()` on each in sequence, measure time from `t0` to first non-empty word emitted. Report as `streaming_first_word_ms`.
- Update `results.json["competitive_context"]` to clarify: Wispr Flow 700 ms is streaming-first-word; our first-token value is first-segment-after-ingest.

### D3. `medical_term_wer` has an alignment correctness bug

Current code in `measure.py::medical_term_wer`:

```
ref_medical = [t for t in ref_tokens if t in MEDICAL_TERMS]
hyp_medical = [t for t in hyp_tokens if t in MEDICAL_TERMS]
return jiwer.wer(" ".join(ref_medical), " ".join(hyp_medical))
```

Subsetting tokens drops positional information. Three concrete failure modes:

1. **Reorder produces false errors.** Ref `chest pain` transcribed as `pain chest` — both correct individually, but `jiwer.wer("chest pain", "pain chest") = 1.0`.
2. **Substituted non-medical token erases a medical position.** Ref `chest pain` heard as `jest pain`. `jest` is not in MEDICAL_TERMS so it gets filtered out before alignment, changing the problem from "chest was substituted" to "chest was deleted".
3. **Insertion of a medical term in hyp.** Hyp `chest pain troponin` when ref was just `chest pain`. Hyp_medical gets a phantom `troponin` that jiwer charges as insertion — but for medical-term recall this might actually be worth counting differently.

**Fix:** use jiwer's word-level alignment on the full strings, then compute error rate restricted to positions where the reference token is medical:

```
out = jiwer.process_words(reference.lower(), hypothesis.lower())
# out.alignments maps each ref position to equal | substitute | delete | (adjacent insert)
errors = 0
total = 0
for ref_pos, ref_tok in enumerate(out.references[0]):
    if ref_tok.strip(_PUNCT) not in MEDICAL_TERMS:
        continue
    total += 1
    # walk alignments to see if this ref position was matched equal
    if not _position_matched_equal(out.alignments[0], ref_pos):
        errors += 1
return errors / max(1, total)
```

The position-mapping is a few lines of boilerplate; the key point is that alignment-based medical-term error rate is the correct definition and the current metric is biased.

### D4. n = 6 — mean metrics have huge confidence intervals

Per-clip WER with paired-test analysis:

| Metric | mean | stdev | 95 % CI (n=6, t≈2.57) |
|---|---:|---:|---|
| wer_raw | 0.121 | 0.064 | [0.054, 0.188] |
| wer_cleaned | 0.199 | 0.129 | [0.063, 0.335] |
| medical_term_wer | 0.067 | 0.097 | [−0.035, 0.168] |

Paired difference (cleaned − raw): mean 0.078, stdev 0.098, 95 % CI **[−0.025, 0.181]**. The cleanup regression is not statistically significant at n=6 — we're confident qualitatively (we can read the chest_pain / abdominal_pain outputs) but the headline numbers carry more noise than signal.

**Fix — do at least one of:**

- Report stdev and 95 % CI alongside every mean in `results.json` (small code change, high honesty payoff).
- Add 3-6 more synthetic scenarios (target n=10-12) so CIs tighten. Kokoro rendering is ~1 min/clip on CPU.
- If Bundle 5 (Variant B) ends up measured on the same 6 clips, paired comparison B-vs-A is statistically stronger than unpaired means — which is already the plan.

### D5. `word_timestamps=True` is dead cost

`transcribe()` sets `word_timestamps=True` but nothing downstream consumes word-level timestamps — `align_speakers` uses segment midpoints and the fallback is index-based. Per faster-whisper 1.2 docs, `word_timestamps=True` adds roughly 5-10 % latency because it runs a second alignment pass.

**Fix:** set `word_timestamps=False` unless real diarisation alignment needs them. When diarisation comes back in Issue 2's vLLM 0.8 upgrade, flip it back on but gate it on `enable_diarisation=True`.

### D6. `max_tokens` budget is in words, not model tokens

Current in `cleanup_segment`:

```
"max_tokens": max(64, len(raw_text.split()) * 3)
```

`len(raw_text.split())` counts whitespace-separated words. `max_tokens` takes model tokens. BioMistral's tokenizer produces roughly 1.3-1.5 tokens per English word, higher for drug names and medical Latin. Effective budget is 4-5× input, not 3×.

Combined with Issue 1's subsequence-only prompt and the proposed tight cap, this becomes non-issue. But when sizing budgets in general, use `len(tokenizer.encode(raw_text))` from the HF tokenizer, not word count.

### D7. Clean-audio calibration risk

Kokoro-82M produces studio-clean synthetic audio: no room reverb, no background noise, no speaker overlap, invariant microphone distance. Real clinical audio has all of these. Our 12.1 % raw WER is a best-case number; expect 20-30 % on real clinic recordings.

**Fix (not for the rerun, for honesty when the number gets quoted):** add a `"calibration_note"` field to `results.json` flagging that the test set is clean TTS and measured WER will rise on real audio. Keeps the downstream story truthful when a judge or reader sees the numbers.

### Order of improvements for the rerun

1. **Must-have** (already listed in Issue 1 + Issue 2 above): subsequence prompt, filler fast-path, `max_tokens` cap by tokenizer length, vLLM 0.8 + pyannote 4 upgrade.
2. **Should-have** — low-cost code changes, add to the rerun: `word_timestamps=False` (D5), rename `first_token_ms` → `first_segment_after_ingest_ms` (D2), add stdev + 95 % CI per metric in `results.json` (D4), add `"calibration_note"` (D7). Roughly 30 min of total code change; makes the JSON substantially more defensible.
3. **Nice-to-have** — may defer: fix `medical_term_wer` alignment (D3) — current metric is biased but still directionally useful; add scenarios to reach n=10-12 (D4, broader fix).
4. **Separate workstream** — not for this rerun: real streaming first-word latency measurement (D2's deeper fix).
