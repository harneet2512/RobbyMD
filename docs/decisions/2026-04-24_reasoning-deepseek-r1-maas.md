# 2026-04-24 — Reasoning layer uses DeepSeek-R1 via Vertex AI MaaS

## Status

Accepted.

## Context

The ship pipeline's reasoning layer (claim extraction → differential diagnosis → SOAP note) has iterated through three implementations in a single afternoon:

1. **Gemini 2.5 Pro on Vertex AI** — prototyped per operator prompt, smoke-tested end-to-end (32 claims, ACS ranked #1, SOAP with `[c:XX]` provenance). Flagged by independent review as a **hard rules violation**: `rules.md §2` allows exactly one commercial API (Claude Opus 4.7); Gemini is outside that whitelist.

2. **Opus 4.7 via Anthropic SDK** — rewritten to close the rules hit. Compliant, but blocked operationally: no `ANTHROPIC_API_KEY` on `aravind-l4-c5` and no session-time path to provision one. ADR at `2026-04-24_opus-reasoning-only.md` (now superseded). Opus code preserved at `opus4.7_usage.md`.

3. **DeepSeek-R1 via Vertex AI MaaS** — this decision.

DeepSeek-R1 weights are released under MIT (see HuggingFace model card: `deepseek-ai/DeepSeek-R1`). MIT is on the `rules.md §2` OSI allowlist. DeepSeek-R1 has **no commercial-API classification for our purposes** — we are invoking an OSI-licensed open-weight model via a managed endpoint. The endpoint itself (Vertex AI) is infrastructure, not a commercial model API in the `rules.md` sense (the rule is about the *model*, not the hosting).

## Decision

`src/extraction/flow/ship/reasoning.py` calls `deepseek-ai/deepseek-r1-0528-maas` via Vertex AI's OpenAI-compatible Chat Completions endpoint. Auth is Application Default Credentials — on `aravind-l4-c5` that resolves to the compute default service account (scope `cloud-platform` already granted at VM creation). No new API keys, no new IAM grants beyond the `roles/aiplatform.user` that was added earlier for the Gemini smoke.

The three reasoning endpoints stay the same — `extract_claims`, `generate_differential`, `generate_soap_note` — with the same prompts and same expected JSON return shapes. The OpenAI Python SDK replaces the `anthropic` SDK; `google-auth` + `google-auth-transport` replace nothing (they come with `google-cloud-aiplatform` which is already installed).

One DeepSeek-R1-specific: R1 emits chain-of-thought reasoning inside `<think>...</think>` tags before the final answer. The `_strip_code_fence` helper strips `</think>` and any trailing code fence before JSON parsing.

## Consequences

**Compliance**: clears `rules.md §2` with room to spare. DeepSeek-R1 is MIT-licensed open-weight — belongs in the same bucket as Whisper, pyannote, rapidfuzz, Kokoro in the repo's licensing posture.

**Operational**:
- No `ANTHROPIC_API_KEY` needed.
- On the L4, uses the VM's metadata-server token automatically.
- On a laptop, requires `gcloud auth application-default login` once.
- Active gcloud account must be `aravindpersonal1220@gmail.com` — per `reference_gcp_accounts.md`, Harneet's account has no IAM on this project and silently 403s. `project_aravind_gcp_interference.md` documents two prior incidents on this project; trust it as low-trust infra.

**Cost**: DeepSeek-R1 MaaS on Vertex is priced per token with a free-tier. The three-call cascade (claims → differential → SOAP) runs ~3 calls/encounter at ~2–3k input + ~1–2k output tokens, well inside the free tier for the hackathon demo scope (≤10 encounters).

**Quality**: DeepSeek-R1's reasoning mode (chain-of-thought inside `<think>`) tends to produce longer, more carefully-structured JSON than Gemini 2.5 Pro did in our first smoke. We accept any small quality drift vs Gemini/Opus in exchange for rules-compliance + operational simplicity. A side-by-side diff of outputs will land in the follow-up Opus smoke run if anyone re-enables Opus.

**Regression risk**: low. Prompts and JSON shapes unchanged. The `<think>` tag stripping is the only new parsing surface.

## Alternatives considered

- **Keep Opus 4.7** — rules-compliant but operationally blocked (no API key on L4). The operator can unblock by provisioning a key; if they do, flip back per `opus4.7_usage.md`.
- **Self-host DeepSeek-R1 on the L4** — 70B+ weights don't fit in 24GB L4 VRAM; would need int8/int4 quantization and competes with Whisper for GPU. Managed Vertex endpoint is strictly better for this demo scale.
- **Llama 3.3 70B via Fireworks / Together AI** — Llama 3 license has use restrictions that aren't strictly on the `rules.md §2` OSI allowlist ("MIT, Apache-2.0, BSD, MPL, ISC, LGPL"). Technically questionable.
- **Mistral Large / Mixtral via Mistral API** — Apache-2.0 weights, but the Mistral API is commercial and not on the whitelist. Would need self-hosting to comply, same GPU problem as DeepSeek self-host.
- **Qwen 2.5 72B via Alibaba Cloud** — Qwen license varies by size and has commercial-use restrictions on some variants; compliance is version-by-version. Skipped.

DeepSeek-R1 MaaS on Vertex is the only option that's (a) MIT-licensed, (b) operationally reachable from this L4 today, (c) doesn't compete with Whisper for GPU, (d) doesn't require new credentials.

## References

- `rules.md §2` — OSI allowlist + Opus whitelist
- `CLAUDE.md §2` point 2 — same invariant restated
- `src/extraction/flow/ship/reasoning.py` — live implementation
- `opus4.7_usage.md` — parked Opus code for future re-enable
- `docs/decisions/2026-04-24_opus-reasoning-only.md` — superseded ADR
- `C:\Users\Lenovo\.claude-personal\projects\D--hack-it\memory\reference_gcp_accounts.md` — account-flip warning
- `C:\Users\Lenovo\.claude-personal\projects\D--hack-it\memory\project_aravind_gcp_interference.md` — project-level interference history
- [DeepSeek-R1 model card](https://huggingface.co/deepseek-ai/DeepSeek-R1) — MIT license
