"""
DeepSeek-R1 reasoning layer via Vertex AI MaaS (OpenAI-compatible endpoint).

Claim extraction, differential diagnosis, and SOAP note generation from the
ship-pipeline transcript segments. DeepSeek-R1 weights are MIT-licensed
(OSI-approved), satisfying `rules.md §2`. The model runs as a managed
service on Aravind's GCP project (`project-c9a6fdd8-8d56-4e88-ad6`,
`us-central1`) per `reference_gcp_accounts.md` — no self-hosted inference,
no dedicated GPU.

Superseded implementations retained for history:
- Gemini 2.5 Pro (rules violation — any commercial API other than Opus 4.7
  is forbidden). Smoke output at
  `eval/flow_results/ship/20260424T025318Z/step8_gemini_smoke.txt`.
- Opus 4.7 via Anthropic SDK (whitelisted but no session API key available
  on the L4). Parked in `opus4.7_usage.md` for future re-enable.

Auth: Application Default Credentials (ADC). On a GCE VM with
`--scopes=cloud-platform`, the metadata server provides a token automatically.
On a workstation, run `gcloud auth application-default login` first.

ACCOUNT WARNING (`reference_gcp_accounts.md`): the active gcloud account
must be `aravindpersonal1220@gmail.com` — `singhharneet2512@gmail.com` has
no IAM on this project and silently 403s on the MaaS endpoint.
"""
from __future__ import annotations

import json
import os
from typing import Optional

import google.auth
import google.auth.transport.requests
from openai import OpenAI

_MODEL_ID = "deepseek-ai/deepseek-r1-0528-maas"
_DEFAULT_PROJECT = "project-c9a6fdd8-8d56-4e88-ad6"
_LOCATION = "us-central1"
_MAX_TOKENS = 4096


def _refresh_adc_token() -> str:
    """Pull a fresh OAuth token from Application Default Credentials."""
    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    credentials.refresh(google.auth.transport.requests.Request())
    return credentials.token


def init_deepseek(project_id: Optional[str] = None) -> OpenAI:
    """Build an OpenAI-compatible client pointed at Vertex AI MaaS."""
    project = project_id or os.environ.get("GCP_PROJECT") or _DEFAULT_PROJECT
    token = _refresh_adc_token()
    base_url = (
        f"https://{_LOCATION}-aiplatform.googleapis.com/v1beta1/projects/"
        f"{project}/locations/{_LOCATION}/endpoints/openapi"
    )
    return OpenAI(base_url=base_url, api_key=token)


def _strip_code_fence(text: str) -> str:
    t = text.strip()
    # DeepSeek-R1 emits reasoning inside <think>...</think>; strip it first.
    if "</think>" in t:
        t = t.split("</think>", 1)[1].strip()
    if t.startswith("```json"):
        t = t[len("```json") :].strip()
    elif t.startswith("```"):
        t = t[3:].strip()
    if t.endswith("```"):
        t = t[:-3].strip()
    return t


def _chat(client: OpenAI, prompt: str) -> str:
    response = client.chat.completions.create(
        model=_MODEL_ID,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=_MAX_TOKENS,
    )
    return response.choices[0].message.content or ""


def extract_claims(client: OpenAI, transcript_segments: list) -> list:
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

Output ONLY a JSON array of claim objects. No other text, no commentary."""
    text = _chat(client, prompt)
    try:
        return json.loads(_strip_code_fence(text))
    except json.JSONDecodeError:
        return [{"error": "failed to parse claims", "raw": text[:500]}]


def generate_differential(client: OpenAI, claims: list) -> list:
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
    text = _chat(client, prompt)
    try:
        return json.loads(_strip_code_fence(text))
    except json.JSONDecodeError:
        return [{"error": "failed to parse differential", "raw": text[:500]}]


def generate_soap_note(
    client: OpenAI,
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
    return _chat(client, prompt)


def smoke_test(transcript_segments: list, project_id: Optional[str] = None) -> dict:
    """Run all three stages on a single transcript for manual inspection."""
    client = init_deepseek(project_id=project_id)
    claims = extract_claims(client, transcript_segments)
    differential = generate_differential(client, claims)
    note = generate_soap_note(client, claims, differential, transcript_segments)
    return {"claims": claims, "differential": differential, "note": note}
