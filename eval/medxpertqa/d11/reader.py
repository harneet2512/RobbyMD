"""D11 Stage 5 -- Gemini reader with compiled evidence bundle.

Calls Gemini via Vertex AI to answer the clinical question using
the full differential-compiled evidence bundle.
"""
from __future__ import annotations

import os
import re
import subprocess

import requests
import structlog

from eval.medxpertqa.adapter import ANSWER_OPTIONS
from eval.medxpertqa.d11.types import FinalBundle

log = structlog.get_logger(__name__)

HARNEET_PROJECT = "project-26227097-98fa-4016-a54"

_THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)


# ---------------------------------------------------------------------------
# Answer extractor -- reused from baseline._extract_answer
# ---------------------------------------------------------------------------

def _extract_answer(response: str) -> str:
    """Extract a single letter (A-J) from the model response.

    Tiered pattern matching: boxed > "final answer" > "ANSWER:" >
    phrasing patterns > option/choice > last standalone letter.
    """
    closed_stripped = _THINK_RE.sub("", response)
    # Handle unclosed think blocks (truncated output)
    think_open = "<think>" in closed_stripped.lower()
    think_close = "</think>" in closed_stripped.lower()
    if think_open and not think_close:
        idx_open = closed_stripped.lower().rfind("<think>")
        after_open = closed_stripped[idx_open + len("<think>"):]
        search_target = after_open if len(after_open) > 200 else closed_stripped
    else:
        search_target = closed_stripped

    text = search_target.upper()

    tier1 = [
        r"\\BOXED\s*\{\s*([A-J])\s*\}",
        r"(?:FINAL\s+ANSWER)\s*(?:IS|:|=|-|—)?\s*\*{0,2}\(?([A-J])\)?\*{0,2}\b",
        r"(?:^|\n)\s*\*{0,2}ANSWER\*{0,2}\s*(?:IS|:|=|-|—)\s*\(?([A-J])\)?\b",
    ]
    for pattern in tier1:
        matches = list(re.finditer(pattern, text, flags=re.IGNORECASE | re.MULTILINE))
        if matches:
            return matches[-1].group(1).upper()

    tier2 = [
        r"(?:THE\s+ANSWER\s+IS|MY\s+ANSWER\s+IS|ANSWER\s*:|ANSWER\s+IS)\s+\*{0,2}\(?([A-J])\)?\*{0,2}\b",
        r"(?:I\s*(?:'LL|'D|\s+WILL|\s+WOULD)\s+(?:GO\s+WITH|PICK|CHOOSE|SELECT))\s+\(?([A-J])\)?\b",
        r"(?:I\s+(?:THINK|BELIEVE))\s+\(?([A-J])\)?\s+(?:IS|IS\s+THE|IS\s+A|IS\s+THE\s+CORRECT)",
        r"(?:GOOD\s+CANDIDATE|CORRECT\s+ANSWER|RIGHT\s+ANSWER)\s+(?:IS|:|=|WOULD\s+BE)\s+\(?([A-J])\)?\b",
    ]
    for pattern in tier2:
        matches = list(re.finditer(pattern, text, flags=re.IGNORECASE | re.MULTILINE))
        if matches:
            return matches[-1].group(1).upper()

    tier3 = [
        r"(?:OPTION|CHOICE)\s*(?:IS|:|=)?\s*\(?([A-J])\)?\b",
        r"\(([A-J])\)",
        r"\b([A-J])\b\s*[\.\!]?\s*$",
        r"^\s*([A-J])\b",
    ]
    for pattern in tier3:
        matches = list(re.finditer(pattern, text, flags=re.IGNORECASE | re.MULTILINE))
        if matches:
            return matches[-1].group(1).upper()

    for char in reversed(text):
        if char in "ABCDEFGHIJ":
            return char
    return ""


# ---------------------------------------------------------------------------
# Auth helper
# ---------------------------------------------------------------------------

def _get_token() -> str:
    """Get GCP access token for Vertex AI."""
    try:
        r = subprocess.run(
            "gcloud auth print-access-token --account=singhharneet2512@gmail.com",
            capture_output=True, text=True, timeout=10, shell=True,
        )
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_READER_PROMPT = """\
You are a clinical reasoning model. Use the evidence bundle to answer the question.

Rules:
- Prefer case-specific discriminators over generic facts
- Note trap warnings
- Acknowledge missing information
- Select one answer
- Cite evidence from the bundle

VIGNETTE:
{vignette}

OPTIONS:
{options}

EVIDENCE BUNDLE:
{bundle}

After your reasoning, state your final answer on its own line as:
ANSWER: [A-J]
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def read_with_bundle(
    vignette: str,
    options: dict[str, str],
    bundle: FinalBundle,
    case_id: str = "",
    model: str = "gemini-2.5-flash",
) -> tuple[str, str]:
    """Call Gemini reader with final bundle.

    Returns (predicted_letter, raw_response).
    """
    opts_str = "\n".join(f"{k}. {options[k]}" for k in ANSWER_OPTIONS if k in options)
    prompt = _READER_PROMPT.format(
        vignette=vignette,
        options=opts_str,
        bundle=bundle.full_text,
    )

    token = _get_token()
    if not token:
        token = os.environ.get("HARNEET_TOKEN", "")
    if not token:
        log.error("reader.no_token", case_id=case_id)
        return ("", "")

    url = (
        f"https://us-central1-aiplatform.googleapis.com/v1/projects/"
        f"{HARNEET_PROJECT}/locations/us-central1/publishers/google/"
        f"models/{model}:generateContent"
    )

    try:
        resp = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json={
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"maxOutputTokens": 4096, "temperature": 0.3},
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        log.error("reader.api_error", case_id=case_id, err=repr(e)[:200])
        return ("", "")

    answer = _extract_answer(raw_text)

    log.info("reader.done", case_id=case_id, model=model, answer=answer)
    return (answer, raw_text)
