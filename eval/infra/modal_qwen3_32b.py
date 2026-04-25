"""Modal deployment of Qwen3-32B via vLLM on A100-80GB.

Replacement for MedGemma (non-OSI) and OpenBioLLM-70B (Llama license).
Qwen3-32B is Apache-2.0 and scores competitively on medical benchmarks.

Deploy
------
    modal deploy --profile anthony2512young-blip eval/infra/modal_qwen3_32b.py

Result: persistent HTTPS URL with vLLM OpenAI-compatible API at /v1/chat/completions.
Set QWEN3_API_BASE=<url>/v1 in environment.

GPU: A100-80GB. Qwen3-32B FP16 is ~64GB; fits with ~16GB KV cache headroom.
"""
from __future__ import annotations

import modal

MINUTES = 60

MODEL_NAME = "Qwen/Qwen3-32B"

app = modal.App("qwen3-32b-vllm")

vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "vllm>=0.8.0",
        "huggingface-hub[hf-transfer]",
        "hf-transfer",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        }
    )
)

hf_cache = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("vllm-cache", create_if_missing=True)


@app.function(
    image=vllm_image,
    gpu="A100-80GB",
    scaledown_window=10 * MINUTES,
    timeout=60 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/vllm": vllm_cache,
    },
)
@modal.concurrent(max_inputs=20)
@modal.web_server(port=8000, startup_timeout=20 * MINUTES)
def serve() -> None:
    """Launch vLLM OpenAI-compatible API server for Qwen3-32B."""
    import subprocess

    cmd = [
        "vllm",
        "serve",
        MODEL_NAME,
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--gpu-memory-utilization",
        "0.90",
        "--max-model-len",
        "8192",
        "--served-model-name",
        "Qwen/Qwen3-32B",
        "--dtype",
        "bfloat16",
    ]
    subprocess.Popen(cmd)
