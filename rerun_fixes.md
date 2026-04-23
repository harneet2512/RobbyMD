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
