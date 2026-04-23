#!/usr/bin/env bash
# Bundle 4 Variant A — Phase 2: start vLLM + run measurement.
# Expected to run after _bundle4_l4_setup.sh finishes successfully.
# Logs: ~/robbymd/b4_measure.log   Status: ~/robbymd/b4_measure.status

set -u
cd "$HOME/robbymd"

LOG="$HOME/robbymd/b4_measure.log"
STATUS="$HOME/robbymd/b4_measure.status"
exec > >(tee -a "$LOG") 2>&1

step() { echo -e "\n=== $(date -u +%H:%M:%SZ) $* ==="; }
fail() { echo "FAIL: $*"; echo "FAIL: $*" > "$STATUS"; exit 1; }

echo "RUNNING" > "$STATUS"

# shellcheck disable=SC1091
source .venv-flow-a/bin/activate

BIOMISTRAL_REPO="${BIOMISTRAL_REPO:-$(grep 'BIOMISTRAL_REPO=' "$LOG" /home/Lenovo/robbymd/b4_setup.log 2>/dev/null | tail -1 | sed 's/.*BIOMISTRAL_REPO=//')}"
if [[ -z "$BIOMISTRAL_REPO" ]]; then
  # Infer from the first successful candidate in the licence audit
  BIOMISTRAL_REPO=$(python -c "
import json, sys
from pathlib import Path
p = Path('progress.md')
if not p.exists(): sys.exit()
blocks = p.read_text().split('## Licence audit — Bundle 4 Variant A download')
if len(blocks) < 2: sys.exit()
block = blocks[-1].split('\`\`\`json')[-1].split('\`\`\`')[0]
data = json.loads(block)
for e in data:
    if 'biomistral_chosen' in e:
        print(e['biomistral_chosen']); sys.exit()
")
fi
[[ -n "$BIOMISTRAL_REPO" ]] || fail "cannot determine BIOMISTRAL_REPO"
echo "BIOMISTRAL_REPO=$BIOMISTRAL_REPO"

# Override with BioMistral org's clean AWQ packaging — LoneStriker's mirror
# shipped a merge-kit artefact (bogus model.safetensors.index.json pointing
# to nonexistent shards) that vLLM 0.6.3 cannot sidestep.
BIOMISTRAL_REPO="BioMistral/BioMistral-7B-DARE-AWQ-QGS128-W4-GEMM"
echo "Pinning BIOMISTRAL_REPO=$BIOMISTRAL_REPO (BioMistral org's own AWQ; single-file weights, no stray index)"

# Pre-fetch weights (idempotent — hits cache if already downloaded)
python -c "from huggingface_hub import snapshot_download; p = snapshot_download('$BIOMISTRAL_REPO', token='$HF_TOKEN'); print('weights cached at', p)" || fail "biomistral download"

step "1. Start vLLM in tmux session vllm"
tmux kill-session -t vllm 2>/dev/null || true
tmux new-session -d -s vllm
# vLLM 0.8+ flags. Dropped --enforce-eager (vLLM 0.8 handles the AWQ
# kernel compile without it and keeps CUDA graph optimisations for faster
# first-token). gpu-memory-utilization bumped to 0.60 to reserve a bit
# more headroom for pyannote 4.x diariser (re-enabled this run).
tmux send-keys -t vllm "source $PWD/.venv-flow-a/bin/activate && export HF_TOKEN=$HF_TOKEN && vllm serve $BIOMISTRAL_REPO --served-model-name biomistral-7b-dare --host 127.0.0.1 --port 8000 --quantization awq --dtype float16 --max-model-len 8192 --gpu-memory-utilization 0.60 --disable-log-requests 2>&1 | tee /home/Lenovo/robbymd/vllm.log" Enter

step "2. Wait for vLLM health (up to 5 min)"
ok=0
for i in $(seq 1 60); do
  sleep 5
  if curl -s -m 2 http://127.0.0.1:8000/v1/models 2>/dev/null | grep -q "biomistral"; then
    ok=1; break
  fi
done
[[ "$ok" == "1" ]] || fail "vllm never became healthy — see vllm.log"
echo "vLLM ready after ~${i}*5 sec"
curl -s http://127.0.0.1:8000/v1/models | python -c "import sys,json; print(json.dumps(json.load(sys.stdin), indent=2))"

step "3. Benchmark BioMistral single-stream throughput"
python -c "
import time, requests
t = time.time()
r = requests.post('http://127.0.0.1:8000/v1/chat/completions', json={
  'model': 'biomistral-7b-dare',
  'messages': [{'role':'user','content':'Say 100 words about chest pain differential.'}],
  'max_tokens': 150, 'temperature': 0.0,
})
dt = (time.time() - t) * 1000
tok = r.json()['usage']['completion_tokens']
print(f'BioMistral single-stream: {tok} tokens in {dt:.0f}ms = {tok/(dt/1000):.1f} tok/s')
" || fail "biomistral smoke"

step "4. Run measurement harness (~30-60 min for 6 clips)"
python -m src.extraction.flow.variant_a.run_all || fail "measurement"

step "5. Commit results locally (push skipped; no GitHub creds on L4)"
git config user.email "aravindpersonal1220@gmail.com"
git config user.name "bundle4-l4-agent"
git add eval/flow_results/variant_a/ progress.md 2>/dev/null || true
if ! git diff --cached --quiet; then
  git commit -m "flow/A: measurement complete on 6 TTS clips

Whisper-large-v3-turbo + pyannote community-1 + BioMistral cleanup on
NVIDIA L4 24GB. Metrics in eval/flow_results/variant_a/<timestamp>/results.json.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
fi
echo "Results ready at: $(ls -td eval/flow_results/variant_a/*/ | head -1)"

step "6. Shutdown vLLM"
tmux kill-session -t vllm 2>/dev/null || true

echo "DONE_MEASURE" > "$STATUS"
echo -e "\n=== $(date -u +%H:%M:%SZ) measurement phase complete. ==="
