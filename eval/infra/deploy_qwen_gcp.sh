#!/usr/bin/env bash
# deploy_qwen_gcp.sh — Spin up a GCP L4 spot instance and serve Qwen2.5-14B-Instruct via vLLM.
#
# Host must already be authenticated for gcloud and have a project selected.
# On Ctrl-C (or any exit), the spot instance is torn down so we don't leave a GPU running.
# Shellcheck-clean (no execution in CI; structure only). See eval/infra/README.md for the cloud-choice rationale.

set -euo pipefail

# ----- configurable via env vars -----
: "${QWEN_GCP_PROJECT:=$(gcloud config get-value project 2>/dev/null || true)}"
: "${QWEN_GCP_REGION:=us-central1}"
: "${QWEN_GCP_ZONE:=us-central1-a}"
: "${QWEN_GCP_INSTANCE_NAME:=qwen-25-14b-eval}"
: "${QWEN_GCP_MACHINE_TYPE:=g2-standard-8}"     # L4 attached by default; 8 vCPU / 32 GB RAM
: "${QWEN_GCP_ACCELERATOR:=type=nvidia-l4,count=1}"
: "${QWEN_GCP_IMAGE_FAMILY:=common-cu124-ubuntu-2204}"
: "${QWEN_GCP_IMAGE_PROJECT:=deeplearning-platform-release}"
: "${QWEN_GCP_DISK_SIZE_GB:=200}"
: "${QWEN_MODEL_ID:=Qwen/Qwen2.5-14B-Instruct}"
: "${QWEN_VLLM_PORT:=8000}"
: "${QWEN_VLLM_QUANTIZATION:=fp8}"              # fp8 or awq; INT8 equiv so 14B fits in 24 GB VRAM

if [[ -z "${QWEN_GCP_PROJECT}" ]]; then
    echo "ERROR: no GCP project set. Run 'gcloud config set project <PROJECT_ID>' first." >&2
    exit 2
fi

echo "[qwen-gcp] Target: project=${QWEN_GCP_PROJECT} zone=${QWEN_GCP_ZONE} instance=${QWEN_GCP_INSTANCE_NAME}"
echo "[qwen-gcp] Model: ${QWEN_MODEL_ID} via vLLM (${QWEN_VLLM_QUANTIZATION}) on port ${QWEN_VLLM_PORT}"

# ----- teardown handler -----
cleanup() {
    local exit_code=$?
    echo
    echo "[qwen-gcp] Tearing down spot instance (exit=${exit_code})..."
    gcloud compute instances delete "${QWEN_GCP_INSTANCE_NAME}" \
        --project="${QWEN_GCP_PROJECT}" \
        --zone="${QWEN_GCP_ZONE}" \
        --quiet 2>/dev/null || true
    echo "[qwen-gcp] Cleanup done."
    exit "${exit_code}"
}
trap cleanup EXIT INT TERM

# ----- spin up L4 spot instance -----
echo "[qwen-gcp] Creating L4 spot instance..."
gcloud compute instances create "${QWEN_GCP_INSTANCE_NAME}" \
    --project="${QWEN_GCP_PROJECT}" \
    --zone="${QWEN_GCP_ZONE}" \
    --machine-type="${QWEN_GCP_MACHINE_TYPE}" \
    --accelerator="${QWEN_GCP_ACCELERATOR}" \
    --image-family="${QWEN_GCP_IMAGE_FAMILY}" \
    --image-project="${QWEN_GCP_IMAGE_PROJECT}" \
    --boot-disk-size="${QWEN_GCP_DISK_SIZE_GB}GB" \
    --provisioning-model=SPOT \
    --instance-termination-action=DELETE \
    --maintenance-policy=TERMINATE \
    --metadata=install-nvidia-driver=True

echo "[qwen-gcp] Waiting for SSH to come up..."
for _ in $(seq 1 30); do
    if gcloud compute ssh "${QWEN_GCP_INSTANCE_NAME}" \
        --project="${QWEN_GCP_PROJECT}" \
        --zone="${QWEN_GCP_ZONE}" \
        --command="echo ok" \
        --quiet 2>/dev/null
    then
        break
    fi
    sleep 10
done

# ----- install vLLM + serve -----
echo "[qwen-gcp] Installing vLLM + pulling model..."
gcloud compute ssh "${QWEN_GCP_INSTANCE_NAME}" \
    --project="${QWEN_GCP_PROJECT}" \
    --zone="${QWEN_GCP_ZONE}" \
    --quiet \
    --command="set -euo pipefail; \
        python3 -m pip install --user --upgrade pip; \
        python3 -m pip install --user 'vllm>=0.6.0'; \
        nohup python3 -m vllm.entrypoints.openai.api_server \
            --model '${QWEN_MODEL_ID}' \
            --quantization '${QWEN_VLLM_QUANTIZATION}' \
            --port ${QWEN_VLLM_PORT} \
            --gpu-memory-utilization 0.92 \
            > /tmp/vllm.log 2>&1 &"

# ----- local port-forward so eval/smoke/run_smoke.py can hit localhost -----
echo "[qwen-gcp] Forwarding local :${QWEN_VLLM_PORT} → instance :${QWEN_VLLM_PORT}."
echo "[qwen-gcp] Point QWEN_API_BASE=http://127.0.0.1:${QWEN_VLLM_PORT}/v1 in your eval env."
echo "[qwen-gcp] Ctrl-C teardown when done."
gcloud compute ssh "${QWEN_GCP_INSTANCE_NAME}" \
    --project="${QWEN_GCP_PROJECT}" \
    --zone="${QWEN_GCP_ZONE}" \
    --ssh-flag="-L ${QWEN_VLLM_PORT}:localhost:${QWEN_VLLM_PORT}" \
    --command="tail -f /tmp/vllm.log"
