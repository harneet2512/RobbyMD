# Opus 4.7 usage — parked

This document captures the Opus-4.7-via-Anthropic-SDK reasoning layer that was briefly wired into `src/extraction/flow/ship/reasoning.py` on 2026-04-24 and then pulled out the same day. It is retained so the work can be re-picked-up if the hackathon posture changes.

## Why Opus 4.7 was considered

`rules.md §2` whitelists **one** commercial API: Claude Opus 4.7, the hackathon's named sponsored tool. The hackathon is literally titled *"Built with Opus 4.7: a Claude Code Hackathon."* For the reasoning layer (claim extraction → differential → SOAP), Opus 4.7 is:

- Rules-compliant (explicitly allowed in `rules.md §2`)
- The strongest closed model the hackathon permits
- Low friction on the workstation (existing Claude Code auth)

## Why Opus 4.7 was pulled back out

- **No session API key on the L4.** `aravind-l4-c5` has no `ANTHROPIC_API_KEY` and no Anthropic Workbench creds. Getting one provisioned requires operator time we are choosing not to spend.
- **Prefer OSS.** `rules.md §2` reads the Opus whitelist as a concession, not a preference. The same section favors OSI-licensed weights. DeepSeek-R1 is MIT-licensed, already running as Vertex AI MaaS on Aravind's project (see `reference_gcp_accounts.md`), and satisfies the same reasoning-layer interface contract.
- **Ops continuity.** Swapping back to Opus 4.7 later is a one-file diff against the artefact below.

## The code that was committed, then replaced

The file below ran in `src/extraction/flow/ship/reasoning.py` as of commit `b6136ab`. Diff against `HEAD` to see the exact swap; the Anthropic SDK version is preserved here verbatim so a future contributor can paste-replace.

```python
"""
Claude Opus 4.7 reasoning layer via the Anthropic API.
"""
from __future__ import annotations
import json
import os
from typing import List, Optional
import anthropic

_MODEL_ID = "claude-opus-4-7"
_MAX_TOKENS = 4096


def init_claude(api_key: Optional[str] = None) -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))


def _extract_text(response: anthropic.types.Message) -> str:
    parts: List[str] = []
    for block in response.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "".join(parts).strip()


def extract_claims(client: anthropic.Anthropic, transcript_segments: list) -> list:
    transcript_text = "\n".join(
        f"{s['speaker']}: {s['text']}" for s in transcript_segments
    )
    prompt = f"""<same claim-extraction prompt as in the live reasoning.py>"""
    response = client.messages.create(
        model=_MODEL_ID,
        max_tokens=_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    text = _extract_text(response)
    # strip ```json ... ``` fence, json.loads, fallback on parse error
    ...


# generate_differential, generate_soap_note, smoke_test all follow the
# same pattern: client.messages.create(model="claude-opus-4-7", ...).
```

## How to re-enable

1. `pip install anthropic` into the ship venv.
2. Export `ANTHROPIC_API_KEY` on whatever machine runs `reasoning.py`.
3. Replace `src/extraction/flow/ship/reasoning.py` with the preserved Opus-SDK version (full source in commit `b6136ab`).
4. Remove `openai` + `google-auth` usage (they're the DeepSeek-MaaS dependencies).
5. Update `docs/decisions/2026-04-24_opus-reasoning-only.md` to `Status: Accepted` again, and mark the DeepSeek ADR as `Superseded`.
6. Re-run `src/extraction/flow/ship/reasoning.py::smoke_test` on one transcript and commit the output as `eval/flow_results/ship/<ts>/step8_opus_smoke.txt`.

## Budget note

Opus 4.7 input+output pricing at the claim-extraction + differential + SOAP cascade is ~3 calls per encounter, average ~2–3k input tokens and ~1–2k output tokens. Roughly \$0.10–0.20 per encounter at list. Prompt caching on the claims block (reused in two of the three calls) would cut that ~30%.

## Pointers

- Live reasoning layer: `src/extraction/flow/ship/reasoning.py`
- Opus-era commit: `b6136ab`
- Opus ADR (now superseded): `docs/decisions/2026-04-24_opus-reasoning-only.md`
- DeepSeek ADR: `docs/decisions/2026-04-24_reasoning-deepseek-r1-maas.md`
- Rules: `rules.md §2`, `CLAUDE.md §2` + `§9`
- Historical Gemini smoke output: `eval/flow_results/ship/20260424T025318Z/step8_gemini_smoke.txt`
