"""Modal deployment of Qwen2.5-14B-Instruct via vLLM, OpenAI-compatible endpoint.

Drop-in replacement for `deploy_qwen_gcp.sh` when GCP GPU quota or Azure
GPU quota is blocked (both were zero for us on 2026-04-22). Modal's
workspace comes with pre-approved GPU access; deployment is one command
and the endpoint URL is stable across cold starts.

Deploy
------
    modal deploy eval/infra/modal_qwen.py

Result: a persistent HTTPS URL exposing vLLM's OpenAI-compatible API at
`/v1/chat/completions` etc. Set `QWEN_API_BASE=<url>/v1` in the smoke
harness env.

GPU choice
----------
L4 (24 GB VRAM, sm_89): native FP8 tensor cores. Qwen2.5-14B in FP8 fits
with ~10 GB headroom for KV cache. Cheapest option that runs FP8.
Fallback: A100-40GB if L4 capacity is unavailable.

Cold start
----------
First request downloads ~28 GB of Qwen2.5-14B-Instruct weights from
HuggingFace and runs vLLM's on-the-fly FP8 quantization. Expect 8–12 min
on the very first invocation. Subsequent cold starts hit the persisted
HF cache volume and are 30–90 s.
"""
from __future__ import annotations

import modal

MINUTES = 60

MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"
# Pin the HF revision so a silent upstream change doesn't move our scores.
# Commit sha from HuggingFace's main branch at 2026-04-22.
MODEL_REVISION = "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"

app = modal.App("qwen25-14b-vllm")

# vLLM 0.7.0 is the latest Modal-image-compatible release with FP8 on Ada (L4).
# hf-transfer speeds the first-pull HF download from ~15 min to ~5 min.
vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "vllm==0.7.0",
        "huggingface-hub[hf-transfer]==0.27.1",
        "hf-transfer==0.1.9",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            # vLLM sometimes writes compiled kernels to ~/.cache/vllm.
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        }
    )
)

# Volumes persist across invocations so we pay the model download once.
hf_cache = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("vllm-cache", create_if_missing=True)


@app.function(
    image=vllm_image,
    gpu="L4",
    scaledown_window=15 * MINUTES,
    timeout=60 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/vllm": vllm_cache,
    },
)
@modal.concurrent(max_inputs=20)
@modal.web_server(port=8000, startup_timeout=20 * MINUTES)
def serve() -> None:
    """Launch vLLM's OpenAI-compatible API server on :8000."""
    import subprocess

    cmd = [
        "vllm",
        "serve",
        MODEL_NAME,
        "--revision",
        MODEL_REVISION,
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--quantization",
        "fp8",
        "--gpu-memory-utilization",
        "0.90",
        "--max-model-len",
        "8192",
        # OpenAI-compatible served-model id must match what the smoke harness sends.
        "--served-model-name",
        "Qwen/Qwen2.5-14B-Instruct",
    ]
    subprocess.Popen(cmd)
