#!/usr/bin/env bash
# deploy_qwen_azure.sh — Sketch-only Azure fallback: NVadsA10_v5 spot + vLLM INT8.
#
# NOT exercised in CI; kept shellcheck-clean as documentation of the fallback path
# if GCP L4 spot capacity is unavailable. See eval/infra/README.md for rationale.
# Host must already be authenticated (az login). Teardown on Ctrl-C.

set -euo pipefail

: "${QWEN_AZURE_SUBSCRIPTION:=$(az account show --query id -o tsv 2>/dev/null || true)}"
: "${QWEN_AZURE_LOCATION:=eastus}"
: "${QWEN_AZURE_RG:=qwen-eval-rg}"
: "${QWEN_AZURE_VM_NAME:=qwen-25-14b-eval}"
: "${QWEN_AZURE_VM_SIZE:=Standard_NV36ads_A10_v5}"   # A10 GPU, 24 GB VRAM
: "${QWEN_AZURE_IMAGE:=Canonical:0001-com-ubuntu-server-jammy:22_04-lts-gen2:latest}"
: "${QWEN_MODEL_ID:=Qwen/Qwen2.5-14B-Instruct}"
: "${QWEN_VLLM_PORT:=8000}"
: "${QWEN_VLLM_QUANTIZATION:=fp8}"

if [[ -z "${QWEN_AZURE_SUBSCRIPTION}" ]]; then
    echo "ERROR: no Azure subscription. Run 'az login' first." >&2
    exit 2
fi

echo "[qwen-azure] Sub=${QWEN_AZURE_SUBSCRIPTION} location=${QWEN_AZURE_LOCATION} vm=${QWEN_AZURE_VM_NAME}"
echo "[qwen-azure] Model ${QWEN_MODEL_ID} via vLLM (${QWEN_VLLM_QUANTIZATION}) on port ${QWEN_VLLM_PORT}"

# ----- teardown handler -----
cleanup() {
    local exit_code=$?
    echo
    echo "[qwen-azure] Tearing down resource group (exit=${exit_code})..."
    az group delete \
        --name "${QWEN_AZURE_RG}" \
        --yes --no-wait 2>/dev/null || true
    echo "[qwen-azure] Cleanup requested (async)."
    exit "${exit_code}"
}
trap cleanup EXIT INT TERM

# ----- create resource group + VM -----
echo "[qwen-azure] Creating resource group..."
az group create \
    --name "${QWEN_AZURE_RG}" \
    --location "${QWEN_AZURE_LOCATION}" \
    --subscription "${QWEN_AZURE_SUBSCRIPTION}" \
    --output none

echo "[qwen-azure] Creating A10 spot VM..."
az vm create \
    --name "${QWEN_AZURE_VM_NAME}" \
    --resource-group "${QWEN_AZURE_RG}" \
    --location "${QWEN_AZURE_LOCATION}" \
    --image "${QWEN_AZURE_IMAGE}" \
    --size "${QWEN_AZURE_VM_SIZE}" \
    --priority Spot \
    --eviction-policy Delete \
    --max-price -1 \
    --admin-username "azureuser" \
    --generate-ssh-keys \
    --output none

echo "[qwen-azure] Opening port ${QWEN_VLLM_PORT}..."
az vm open-port \
    --resource-group "${QWEN_AZURE_RG}" \
    --name "${QWEN_AZURE_VM_NAME}" \
    --port "${QWEN_VLLM_PORT}" \
    --output none

PUBLIC_IP=$(az vm show \
    --resource-group "${QWEN_AZURE_RG}" \
    --name "${QWEN_AZURE_VM_NAME}" \
    --show-details \
    --query "publicIps" \
    --output tsv)

echo "[qwen-azure] VM public IP: ${PUBLIC_IP}"
echo "[qwen-azure] Installing vLLM + pulling model (will take several minutes)..."
ssh -o StrictHostKeyChecking=no "azureuser@${PUBLIC_IP}" "set -euo pipefail; \
    sudo apt-get update && sudo apt-get install -y python3-pip nvidia-cuda-toolkit; \
    python3 -m pip install --user --upgrade pip; \
    python3 -m pip install --user 'vllm>=0.6.0'; \
    nohup python3 -m vllm.entrypoints.openai.api_server \
        --model '${QWEN_MODEL_ID}' \
        --quantization '${QWEN_VLLM_QUANTIZATION}' \
        --port ${QWEN_VLLM_PORT} \
        --gpu-memory-utilization 0.92 \
        > /tmp/vllm.log 2>&1 &"

echo "[qwen-azure] Tunnelling local :${QWEN_VLLM_PORT} → ${PUBLIC_IP}:${QWEN_VLLM_PORT}."
echo "[qwen-azure] QWEN_API_BASE=http://127.0.0.1:${QWEN_VLLM_PORT}/v1"
ssh -N -L "${QWEN_VLLM_PORT}:localhost:${QWEN_VLLM_PORT}" "azureuser@${PUBLIC_IP}"
