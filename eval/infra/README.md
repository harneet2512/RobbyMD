# eval/infra — Primary-reader hosting scripts

Primary eval reader is `Qwen2.5-14B-Instruct` (Apache-2.0) per `Eng_doc.md §3.5` and `eval/README.md`. Two deployment paths — GCP (primary) and Azure (sketch-only fallback). The dev host is already authenticated for both (`gcloud` + `az` pick up existing credentials; no auth step in the scripts).

## Primary — `deploy_qwen_gcp.sh`

GCP L4 spot instance + vLLM INT8 quantization so 14B weights fit in 24 GB VRAM. L4 spot price as of 2026-04 is roughly **$0.20–0.30/hr preemptible**.

```bash
./eval/infra/deploy_qwen_gcp.sh                          # default region us-central1, L4 spot
QWEN_GCP_REGION=us-west1 ./eval/infra/deploy_qwen_gcp.sh # override region
```

On Ctrl-C (or script exit) the instance is torn down via a `trap` handler so you don't leave a GPU running overnight. Script is `set -euo pipefail`, shellcheck-clean.

**Rationale for GCP as primary**: well-trodden 2025–26 ML-ops path for Qwen-14B-class vLLM serving, mature `gcloud` CLI, reliable L4 spot availability in `us-central1`.

## Fallback — `deploy_qwen_azure.sh`

Azure NVadsA10_v5 spot instance + same vLLM install. Sketch-only; **not exercised** — kept shellcheck-clean but not validated against a live Azure spot quota. Escape hatch if GCP L4 spot capacity is unavailable on a given day.

```bash
./eval/infra/deploy_qwen_azure.sh                          # default region eastus, A10 spot
QWEN_AZURE_LOCATION=westus2 ./eval/infra/deploy_qwen_azure.sh
```

## Last resort — managed API

If neither cloud yields spot capacity, Together, Fireworks, or DeepInfra all host `Qwen2.5-14B-Instruct` at comparable per-token prices. Set:

```bash
export QWEN_API_BASE="https://api.together.xyz/v1"
export QWEN_API_KEY="..."
```

The smoke harness (`eval/smoke/run_smoke.py`) reads these env vars when `--reader qwen2.5-14b` is passed.

## Auth assumption

The dev host authenticated for both clouds 2026-04-21. The deploy scripts do NOT call `gcloud auth login` or `az login` — they assume existing credentials are already visible to the CLI. If running on a fresh machine, authenticate first:

```bash
gcloud auth login && gcloud config set project <PROJECT_ID>
az login
```
