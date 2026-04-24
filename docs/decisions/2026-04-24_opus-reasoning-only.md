# 2026-04-24 — Reasoning layer must use Opus 4.7 (no Gemini, no other commercial API)

## Status

Accepted.

## Context

The ship pipeline's reasoning layer (claim extraction → differential diagnosis → SOAP note) was initially prototyped against **Gemini 2.5 Pro on Vertex AI** during the 2026-04-24 build session. The hackathon operator prompt explicitly named Gemini, so it got wired in and smoke-tested end-to-end (32 claims, ACS ranked #1, SOAP note with `[c:XX]` provenance tags — working output at `eval/flow_results/ship/20260424T025318Z/step8_gemini_smoke.txt`).

An independent code review then flagged this as a **hard rules violation**:

- `rules.md §2`: *"All models and dependencies must be OSI-approved open source. Allowlist: MIT, Apache-2.0, BSD, MPL, ISC, LGPL. Not allowed: Gemma Terms, HAI-DEF, **any commercial API except Opus 4.7** (the hackathon's named sponsored tool)."*
- `CLAUDE.md §2` point 2 restates the same invariant.
- `CLAUDE.md §9`: *"Model string: `claude-opus-4-7`."*

Gemini is a closed commercial API outside the Opus-4.7 whitelist. Keeping it would fail `tests/licensing/test_open_source.py` on intent even if not on a scanned-identifier match, and more importantly it's a straight-up constitution violation for a hackathon whose thesis rides on compliance.

## Decision

Replace the Gemini 2.5 Pro call site with **Claude Opus 4.7 via the Anthropic Python SDK**. One file, `src/extraction/flow/ship/reasoning.py`, loses its `vertexai` dependency and gains an `anthropic` dependency. Model string hard-coded to `claude-opus-4-7` per CLAUDE.md §9.

The three reasoning endpoints stay the same — `extract_claims`, `generate_differential`, `generate_soap_note` — with the same prompts and return shapes. Only the transport changes.

## Consequences

**Compliance**: clears `rules.md §2` and `CLAUDE.md §2`. No other change to the pipeline is needed — this is a one-file swap.

**Operational**: `ANTHROPIC_API_KEY` env var replaces Vertex AI ADC on the L4. No IAM grants needed. Cost is billed to whoever owns the key instead of to Aravind's GCP project.

**Caching**: long-term the 3 sequential calls per encounter would benefit from Anthropic's prompt caching (the claims block is verbatim in two prompts), but that's a follow-up — not needed to unblock compliance.

**Gemini smoke output preserved**: `eval/flow_results/ship/20260424T025318Z/step8_gemini_smoke.txt` stays in-repo as historical record of what Gemini produced. The file is explicitly labeled `gemini` in its name, so it doesn't pretend to be Opus output. When Opus is re-smoked, the new output will land alongside as `step8_opus_smoke.txt`.

**Regression risk**: zero functional risk — the prompts and expected JSON shapes are unchanged. The only thing that could diverge is response style, and since we're parsing structured JSON out of code-fenced blocks, that's absorbed.

## Alternatives considered

- **Keep Gemini and write an exemption into rules.md** — rejected. The rule exists because the hackathon thesis is "built on open source + Opus 4.7." Widening the exception for convenience would undermine that thesis and the `test_open_source.py` intent.
- **Swap to an open-source model (Llama 3.3 / Mistral / DeepSeek)** — plausible per `rules.md §2` (MIT/Apache licenses qualify). But routing reasoning through a self-hosted OSS model adds ~20GB VRAM + inference latency on the L4 and would compete with Whisper for GPU. Opus 4.7 is on the whitelist specifically because the hackathon wants the reasoning quality bar that the sponsored model provides, without the local compute cost. Open-source reasoning is a fine follow-up if the hackathon posture changes; not needed for v1.
- **Drop the reasoning layer entirely** — rejected. The ship thesis is "ASR → claims → differential → note with provenance." Removing the last three stages breaks the demo.

## References

- `rules.md §2` — OSI-only + Opus 4.7 whitelist
- `CLAUDE.md §2`, `§9` — model string and compliance reiteration
- `src/extraction/flow/ship/reasoning.py` — implementation
- `eval/flow_results/ship/20260424T025318Z/step8_gemini_smoke.txt` — pre-switch Gemini output, retained as historical artifact
- `progress.md` Bundle-follow-up entry (2026-04-24) — documents the discovery + fix
