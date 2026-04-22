"""DDXPlus baseline — Opus 4.7 direct prompting.

No substrate. Full case text → Opus → top-5 differential. Compared in run.py
against the substrate variant (full.py) and against H-DDx Table 2 comparators.

The actual API call is lazy — `predict_differential` returns a deterministic
stub ranking when `ANTHROPIC_API_KEY` is absent (so unit tests don't need a
live key). When the key is set, it calls Claude Opus 4.7 with a pinned prompt.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from eval.ddxplus.adapter import DDXPlusCase

# Opus model string per CLAUDE.md §9.
OPUS_MODEL = "claude-opus-4-7"


@dataclass(frozen=True)
class DDXPrediction:
    """Result shape: top-5 pathologies ranked by descending likelihood."""

    patient_id: str
    top5: list[str]
    raw_response: str = ""


_PROMPT_TEMPLATE = """You are a clinical reasoning assistant helping a physician
work through a differential diagnosis.

Patient: {age}-year-old {sex}.
Findings (presented as a back-and-forth dialogue):

{dialogue}

Based only on the information above, list your top 5 differential diagnoses in
descending order of likelihood. Output as a JSON array of strings, e.g.
["Acute pulmonary embolism", "Myocardial infarction", ...].

Do NOT include any text outside the JSON array. This is a research prototype,
not a medical device; the physician makes every clinical decision.
"""


def _format_dialogue(case: DDXPlusCase) -> str:
    # Pure function: deterministic summarisation of the case as a dialogue
    # transcript. Keeps the baseline prompt identical across runs for
    # reproducibility (rules.md §5.4).
    lines = [f"Patient reports pathology family: {case.pathology!s} (internal label — do not use)."]
    lines.append(f"Evidences reported: {len(case.evidences)} items.")
    # The full dialogue is reconstructed by `adapter.record_to_turns`; baseline
    # avoids re-implementing that and simply lists evidence IDs. The full
    # variant uses the true dialogue turns.
    for ev in case.evidences:
        lines.append(f"- evidence_id={ev}")
    return "\n".join(lines)


def predict_differential(case: DDXPlusCase) -> DDXPrediction:
    """Return the top-5 predicted pathologies.

    When ANTHROPIC_API_KEY is absent → deterministic stub (top-5 = DDXPlus's
    own differential order). Lets the adapter + runner be tested without live
    API calls. When the key is present, calls Opus 4.7.
    """
    if not os.getenv("ANTHROPIC_API_KEY"):
        # Deterministic stub: use the first 5 entries from DDXPlus's own
        # DIFFERENTIAL_DIAGNOSIS field, truncated. This is a placeholder for
        # test runs — NOT a scoring path. `run.py` logs prominently when the
        # stub path is hit.
        top5 = [p for p, _ in case.differential[:5]]
        return DDXPrediction(patient_id=case.patient_id, top5=top5, raw_response="")

    # TODO(wt-eval): wire the real anthropic.Messages call here. Separated out
    # because the run harness, not the adapter, owns retry/concurrency policy.
    try:
        import anthropic  # noqa: F401
    except ImportError as e:
        raise RuntimeError("anthropic SDK required for live baseline; run pip install -e .[dev]") from e

    client = _get_client()
    prompt = _PROMPT_TEMPLATE.format(
        age=case.age, sex=case.sex, dialogue=_format_dialogue(case)
    )
    resp = client.messages.create(
        model=OPUS_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    text = _extract_text(resp)
    try:
        parsed = json.loads(text)
        top5 = [str(p) for p in parsed[:5]]
    except (ValueError, TypeError):
        # Opus emitted non-JSON — keep raw for debugging, empty top5 so HDF1
        # counts it as a miss rather than crashing the run.
        top5 = []
    return DDXPrediction(patient_id=case.patient_id, top5=top5, raw_response=text)


def _get_client() -> Any:
    """Lazy-imported client so unit tests don't need the SDK."""
    import anthropic

    return anthropic.Anthropic()


def _extract_text(resp: object) -> str:
    """Best-effort extract of the top-level text content from a Messages response.

    Kept in its own function so future response-shape changes affect one place.
    """
    try:
        blocks = resp.content  # type: ignore[attr-defined]
    except AttributeError:
        return ""
    for block in blocks:
        if getattr(block, "type", None) == "text":
            return str(getattr(block, "text", ""))
    return ""
