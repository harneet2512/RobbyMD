"""D11 Stage 1 -- Clinical abstraction via Qwen3-32B.

Extracts structured clinical features from a vignette without answering
the question or ranking any option.
"""
from __future__ import annotations

import os
import re
from typing import Callable

import structlog

from eval.medxpertqa.d11.types import ClinicalAbstraction
from eval.medxpertqa.retry_utils import retry_with_backoff

log = structlog.get_logger(__name__)

GROQ_MODEL = "qwen/qwen3-32b"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

ChatFn = Callable[[str, str, int], str]

_ABSTRACTION_PROMPT = """\
You are a clinical case analyst. Given a clinical vignette, extract structured features.

Rules:
- Do NOT answer any clinical question
- Do NOT rank or recommend any option
- Do NOT copy the vignette verbatim -- abstract into medical features
- Identify the core clinical problem, key findings, temporal patterns

VIGNETTE:
{vignette}

Respond in this EXACT format:
CLINICAL_PROBLEM: [one sentence describing the core clinical issue]
KEY_FINDINGS: [comma-separated list of significant clinical findings]
TEMPORAL_PATTERN: [acute|subacute|chronic|progressive|episodic|unknown]
BODY_SYSTEM: [cardiovascular|respiratory|neurological|gastrointestinal|musculoskeletal|renal|endocrine|hematological|dermatological|psychiatric|multisystem|other]
SPECIALTY_HINT: [most relevant medical specialty]
TASK_TYPE: [diagnosis|mechanism|management|anatomy|pathology|adverse_effect|next_step|other]
MISSING_CONTEXT: [comma-separated list of information that would help but is not provided, or "none"]

/no_think
"""

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

_VALID_TEMPORAL = {
    "acute", "subacute", "chronic", "progressive", "episodic", "unknown",
}
_VALID_BODY_SYSTEM = {
    "cardiovascular", "respiratory", "neurological", "gastrointestinal",
    "musculoskeletal", "renal", "endocrine", "hematological",
    "dermatological", "psychiatric", "multisystem", "other",
}
_VALID_TASK_TYPE = {
    "diagnosis", "mechanism", "management", "anatomy", "pathology",
    "adverse_effect", "next_step", "other",
}


def _groq_client():  # noqa: ANN202
    from openai import OpenAI

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set")
    return OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)


def _call_groq(prompt: str, label: str = "", max_tokens: int = 2048) -> str:
    client = _groq_client()

    def _call() -> str:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.3,
        )
        return resp.choices[0].message.content or ""

    raw = retry_with_backoff(_call, label=label)
    return _THINK_RE.sub("", raw).strip()


def _parse_line(text: str, header: str) -> str:
    """Extract value after 'HEADER: value' from text."""
    pattern = re.compile(rf"^{re.escape(header)}\s*:\s*(.+)$", re.MULTILINE | re.IGNORECASE)
    m = pattern.search(text)
    return m.group(1).strip() if m else ""


def _parse_csv(value: str) -> list[str]:
    """Split comma-separated value into stripped non-empty items."""
    if not value or value.lower() == "none":
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def abstract_clinical_case(
    vignette: str,
    options: dict[str, str] | None = None,
    case_id: str = "",
    chat_fn: ChatFn | None = None,
) -> ClinicalAbstraction:
    """Call Qwen3-32B to abstract clinical case into structured features."""
    prompt = _ABSTRACTION_PROMPT.format(vignette=vignette)
    _fn = chat_fn or (lambda p, l, m: _call_groq(p, label=l, max_tokens=m))
    raw = _fn(prompt, f"abstraction:{case_id}", 2048)

    clinical_problem = _parse_line(raw, "CLINICAL_PROBLEM") or "unknown"
    key_findings = _parse_csv(_parse_line(raw, "KEY_FINDINGS"))

    temporal_raw = _parse_line(raw, "TEMPORAL_PATTERN").lower()
    temporal_pattern = temporal_raw if temporal_raw in _VALID_TEMPORAL else "unknown"

    body_raw = _parse_line(raw, "BODY_SYSTEM").lower()
    body_system = body_raw if body_raw in _VALID_BODY_SYSTEM else "other"

    specialty_hint = _parse_line(raw, "SPECIALTY_HINT") or "general medicine"

    task_raw = _parse_line(raw, "TASK_TYPE").lower()
    task_type = task_raw if task_raw in _VALID_TASK_TYPE else "other"

    missing_context = _parse_csv(_parse_line(raw, "MISSING_CONTEXT"))

    result = ClinicalAbstraction(
        clinical_problem=clinical_problem,
        key_findings=key_findings,
        temporal_pattern=temporal_pattern,
        body_system=body_system,
        specialty_hint=specialty_hint,
        task_type=task_type,
        missing_context=missing_context,
    )
    log.info(
        "abstraction.done",
        case_id=case_id,
        n_findings=len(key_findings),
        temporal=temporal_pattern,
        body_system=body_system,
    )
    return result
