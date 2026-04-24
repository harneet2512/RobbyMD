"""
Claude Opus 4.7 reasoning layer via the Anthropic API.

Claim extraction, differential diagnosis, and SOAP note generation from the
ship-pipeline transcript segments. Uses the hackathon's named sponsored tool
(Opus 4.7) per rules.md §2 / CLAUDE.md §2. Replaces an earlier Gemini 2.5 Pro
implementation that violated the open-source/Opus-only constraint; see
docs/decisions/2026-04-24_opus-reasoning-only.md for the switch rationale.
"""
from __future__ import annotations

import json
import os
from typing import List, Optional

import anthropic

_MODEL_ID = "claude-opus-4-7"
_MAX_TOKENS = 4096


def init_claude(api_key: Optional[str] = None) -> anthropic.Anthropic:
    """Build an Anthropic client. Reads ANTHROPIC_API_KEY from env if not passed."""
    return anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))


def _extract_text(response: anthropic.types.Message) -> str:
    parts: List[str] = []
    for block in response.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "".join(parts).strip()


def _strip_code_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```json"):
        t = t[len("```json") :].strip()
    elif t.startswith("```"):
        t = t[3:].strip()
    if t.endswith("```"):
        t = t[:-3].strip()
    return t


def extract_claims(client: anthropic.Anthropic, transcript_segments: list) -> list:
    transcript_text = "\n".join(
        f"{s['speaker']}: {s['text']}" for s in transcript_segments
    )
    prompt = f"""You are a clinical claim extractor. Given this doctor-patient transcript, extract every factual clinical claim as structured data.

TRANSCRIPT:
{transcript_text}

For each claim, output JSON with:
- claim_id: sequential (c01, c02, ...)
- subject: the clinical entity (e.g., "chest pain", "fever", "SpO2")
- predicate: the attribute (e.g., "onset", "location", "value", "trigger")
- value: the stated value
- speaker: who stated it (DOCTOR or PATIENT)
- turn_index: which turn number (0-indexed)
- confidence: high/medium/low
- status: "active" (default) or "superseded" if contradicted later in the transcript
- supersedes: claim_id of the claim this supersedes (if any)
- supersession_reason: why (if any)

Output ONLY a JSON array of claim objects. No other text."""
    response = client.messages.create(
        model=_MODEL_ID,
        max_tokens=_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    text = _extract_text(response)
    try:
        return json.loads(_strip_code_fence(text))
    except json.JSONDecodeError:
        return [{"error": "failed to parse claims", "raw": text[:500]}]


def generate_differential(client: anthropic.Anthropic, claims: list) -> list:
    prompt = f"""You are a clinical reasoning engine. Given these extracted claims from a patient encounter, generate a differential diagnosis.

CLAIMS:
{json.dumps(claims, indent=2)}

For each hypothesis, output JSON with:
- hypothesis: diagnosis name
- rank: 1 = most likely
- evidence_for: list of claim_ids supporting this
- evidence_against: list of claim_ids arguing against this
- missing_data: list of tests/questions that would help confirm or rule out
- confidence: high/medium/low
- likelihood_ratio_estimate: rough LR if known

Output ONLY a JSON array of hypothesis objects, ranked by likelihood. No other text."""
    response = client.messages.create(
        model=_MODEL_ID,
        max_tokens=_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    text = _extract_text(response)
    try:
        return json.loads(_strip_code_fence(text))
    except json.JSONDecodeError:
        return [{"error": "failed to parse differential", "raw": text[:500]}]


def generate_soap_note(
    client: anthropic.Anthropic,
    claims: list,
    differential: list,
    transcript_segments: list,
) -> str:
    prompt = f"""You are a clinical documentation engine. Generate a SOAP note from this encounter data.

CLAIMS:
{json.dumps(claims, indent=2)}

DIFFERENTIAL:
{json.dumps(differential, indent=2)}

Rules:
1. Every factual statement in the note MUST include a provenance tag [c:XX] referencing the claim_id it came from.
2. The Assessment section must mention ALL hypotheses from the differential, including those that were deprioritized, with the evidence for and against each.
3. If any claim was superseded, note the correction explicitly.
4. Use standard SOAP format: Subjective, Objective, Assessment, Plan.

Output the SOAP note as plain text with [c:XX] tags inline."""
    response = client.messages.create(
        model=_MODEL_ID,
        max_tokens=_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    return _extract_text(response)


def smoke_test(transcript_segments: list, api_key: Optional[str] = None) -> dict:
    """Run all three stages on a single transcript for manual inspection."""
    client = init_claude(api_key=api_key)
    claims = extract_claims(client, transcript_segments)
    differential = generate_differential(client, claims)
    note = generate_soap_note(client, claims, differential, transcript_segments)
    return {"claims": claims, "differential": differential, "note": note}
