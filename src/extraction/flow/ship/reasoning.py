"""
Gemini 2.5 Pro reasoning layer via Vertex AI.

Claim extraction, differential diagnosis, and SOAP note generation from the
ship-pipeline transcript segments. Separate from the ASR measurement — can
fail without invalidating the WER/DER numbers.
"""
from __future__ import annotations

import json
from typing import Optional

import vertexai
from vertexai.generative_models import GenerativeModel

_MODEL_ID = "gemini-2.5-pro"


def init_gemini(project_id: str, location: str = "us-central1") -> GenerativeModel:
    vertexai.init(project=project_id, location=location)
    return GenerativeModel(_MODEL_ID)


def _strip_code_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```json"):
        t = t[len("```json") :].strip()
    elif t.startswith("```"):
        t = t[3:].strip()
    if t.endswith("```"):
        t = t[:-3].strip()
    return t


def extract_claims(model: GenerativeModel, transcript_segments: list) -> list:
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
    response = model.generate_content(prompt)
    try:
        return json.loads(_strip_code_fence(response.text))
    except json.JSONDecodeError:
        return [{"error": "failed to parse claims", "raw": response.text[:500]}]


def generate_differential(model: GenerativeModel, claims: list) -> list:
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
    response = model.generate_content(prompt)
    try:
        return json.loads(_strip_code_fence(response.text))
    except json.JSONDecodeError:
        return [{"error": "failed to parse differential", "raw": response.text[:500]}]


def generate_soap_note(
    model: GenerativeModel,
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
    response = model.generate_content(prompt)
    return response.text


def smoke_test(project_id: str, transcript_segments: list) -> dict:
    """One-shot smoke test: run all three stages on a single transcript.

    Returns {claims, differential, note} for manual inspection.
    """
    model = init_gemini(project_id)
    claims = extract_claims(model, transcript_segments)
    differential = generate_differential(model, claims)
    note = generate_soap_note(model, claims, differential, transcript_segments)
    return {"claims": claims, "differential": differential, "note": note}
