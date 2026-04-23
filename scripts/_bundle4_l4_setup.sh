#!/usr/bin/env bash
# Bundle 4 Variant A — resumable setup. Each phase drops a marker in
# ~/robbymd/.b4_markers/ so a spot-preempted restart picks up where it left
# off. Re-run this script after `rm -rf ~/robbymd/.b4_markers
# ~/robbymd/.venv-flow-a` to force a clean rebuild (required when the pinned
# dep matrix changes — as it did 2026-04-23 for the vLLM 0.8 / torch 2.8 /
# pyannote.audio 4.0 upgrade that re-enables diarisation).
#
# Logs: ~/robbymd/b4_setup.log   Status: ~/robbymd/b4_setup.status

set -u
cd "$HOME/robbymd"

LOG="$HOME/robbymd/b4_setup.log"
STATUS="$HOME/robbymd/b4_setup.status"
MARK="$HOME/robbymd/.b4_markers"
mkdir -p "$MARK"
exec > >(tee -a "$LOG") 2>&1

step() { echo -e "\n=== $(date -u +%H:%M:%SZ) $* ==="; }
done_marker() { touch "$MARK/$1"; }
is_done() { [[ -f "$MARK/$1" ]]; }
fail() { echo "FAIL: $*"; echo "FAIL: $*" > "$STATUS"; exit 1; }

echo "RUNNING" > "$STATUS"

# -------- Stage 0: apt deps --------
if ! is_done stage0_apt; then
  step "0. apt deps (python3.10-venv, build-essential, ffmpeg)"
  if ! python3.10 -c 'import ensurepip' 2>/dev/null; then
    sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq || fail "apt update"
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
      python3.10-venv python3.10-dev build-essential ffmpeg \
      || fail "apt install"
  fi
  done_marker stage0_apt
else
  echo "skip stage0 (already done)"
fi

# -------- Stage 1: branch + HF (always runs, cheap) --------
step "1. Branch + HF token"
git fetch origin --quiet || fail "git fetch"
git checkout bundle4-variant-a-run 2>/dev/null \
  || git checkout -b bundle4-variant-a-run origin/bundle4-variant-a-run \
  || fail "checkout"
git reset --hard origin/bundle4-variant-a-run || fail "reset"
git log --oneline -3
[[ -n "${HF_TOKEN:-}" ]] || fail "HF_TOKEN unset"

# -------- Stage 2: venv + pip install (NEW MATRIX, 2026-04-23 rerun) --------
#   torch 2.8, vllm 0.8, pyannote.audio 4.0, faster-whisper 1.2.x
#   This combination lets vLLM (torch 2.8-compatible) coexist with
#   pyannote community-1 (requires pyannote.audio 4.x, which needs torch
#   2.8) in a single venv — the blocker that forced diarisation to be
#   disabled on the first run.
if ! is_done stage2_pip; then
  step "2. venv + pip install (~15 min — major version bumps, larger downloads)"
  if [[ ! -f .venv-flow-a/bin/activate ]]; then
    rm -rf .venv-flow-a
    python3.10 -m venv .venv-flow-a || fail "venv create"
  fi
  # shellcheck disable=SC1091
  source .venv-flow-a/bin/activate
  python -m pip install --quiet --upgrade pip setuptools wheel || fail "pip upgrade"

  # torch 2.8 + matching torchaudio. The cu121/cu124 indexes only ship
  # up to torch 2.5/2.6; torch 2.8 wheels live on cu126+. L4's driver
  # (580) is CUDA 12.x-compatible and handles cu126 fine.
  python -m pip install --quiet \
      "torch==2.8.*" "torchaudio==2.8.*" \
      --index-url https://download.pytorch.org/whl/cu126 \
    || fail "torch"

  # Application stack. pyannote.audio 4.0 pulls pyannote-core 6.x which
  # requires numpy>=2.0 — do NOT pin numpy<2 anymore.
  python -m pip install --quiet \
      "faster-whisper>=1.2.1" \
      "pyannote.audio>=4.0.0" \
      "pyannote.metrics>=4.0.0" \
      "kokoro>=0.9.2" \
      "soundfile==0.12.1" \
      "pydub" \
      "librosa" \
      "jiwer==3.0.4" \
      "huggingface_hub" \
      "requests" \
    || fail "asr+tts+metrics"

  # vLLM 0.8+ for torch-2.8 compatibility
  python -m pip install --quiet "vllm>=0.8.0" || fail "vllm"

  python -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__, torch.cuda.get_device_name(0), round(torch.cuda.get_device_properties(0).total_memory/1e9, 1), 'GB')" \
    || fail "CUDA check"

  # pyairports shim: vLLM's guided-decoding path imports outlines which
  # imports pyairports.airports. PyPI's `pyairports==0.0.1` is a
  # namespace-squat stub with no `airports` submodule. Stub it here
  # unconditionally — if vLLM 0.8+ already fixed the transitive, this is a
  # no-op; if not, this prevents the 400 Internal Server Error on every
  # chat/completions request.
  step "2b. pyairports stub (belt-and-braces)"
  SP="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
  if ! python -c "from pyairports.airports import AIRPORT_LIST" 2>/dev/null; then
    mkdir -p "$SP/pyairports"
    : > "$SP/pyairports/__init__.py"
    cat > "$SP/pyairports/airports.py" << 'PYSTUB'
AIRPORT_LIST = []
class Airports:
    def __init__(self, *a, **k): pass
    def lookup(self, *a, **k): return None
PYSTUB
    python -c "from pyairports.airports import AIRPORT_LIST; print('pyairports stub active, len =', len(AIRPORT_LIST))"
  else
    echo "pyairports natively importable — stub not needed"
  fi

  done_marker stage2_pip
else
  echo "skip stage2 (already done)"
  # shellcheck disable=SC1091
  source .venv-flow-a/bin/activate
fi

# -------- Stage 3: licence-verify + model downloads --------
if ! is_done stage3_models; then
  step "3. Licence-verify + model downloads (~20 min if cache warm, ~45 min cold)"
  python scripts/download_and_verify_variant_a.py || fail "licence/download"
  done_marker stage3_models
else
  echo "skip stage3 (already done)"
fi

# -------- Stage 4: TTS render --------
if ! is_done stage4_tts; then
  step "4. Render 6 TTS clips (Kokoro on CPU)"
  python -m src.extraction.flow.variant_a.render_tts || fail "TTS render"
  ls -la eval/synthetic_clips/audio/
  done_marker stage4_tts
else
  echo "skip stage4 (already done)"
fi

# -------- Stage 5: ground truth --------
if ! is_done stage5_gt; then
  step "5. Ground-truth JSONL"
  python -m src.extraction.flow.variant_a.build_ground_truth || fail "ground truth"
  wc -l eval/synthetic_clips/ground_truth.jsonl
  done_marker stage5_gt
else
  echo "skip stage5 (already done)"
fi

# -------- Stage 6: local commit (push happens from laptop — L4 has no GH creds) --------
if ! is_done stage6_push; then
  step "6. Local commit of rendered clips + ground truth"
  git config user.email "aravindpersonal1220@gmail.com"
  git config user.name "bundle4-l4-agent"
  git add eval/synthetic_clips/ground_truth.jsonl eval/synthetic_clips/audio/ progress.md 2>/dev/null || true
  if ! git diff --cached --quiet; then
    git commit -m "render: 6 TTS clips via Kokoro-82M + ground-truth JSONL

Kokoro-82M (Apache-2.0) with am_michael (DOCTOR) + af_bella (PATIENT).
seed=42, 24 kHz mono, 300 ms inter-turn silence. Regex accepts both
DR/PT (Stream D scripts) and DOCTOR/PATIENT. Licence audit block
appended to progress.md.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>" \
      || fail "commit ms1"
    # Push is done from the laptop after SCP, not from the L4 (no GH creds).
  fi
  done_marker stage6_push
else
  echo "skip stage6 (already done)"
fi

echo "DONE_SETUP" > "$STATUS"
echo -e "\n=== $(date -u +%H:%M:%SZ) setup complete. launch vLLM + measurement next. ==="
