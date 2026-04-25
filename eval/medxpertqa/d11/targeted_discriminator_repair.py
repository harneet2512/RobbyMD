"""D11 Stage 4 -- Targeted discriminator repair.

Generates more specific discriminators for pairs that remained unresolved
or were missing from the initial tournament. Only runs when repair_required
is True in the SufficiencyAudit.
"""
from __future__ import annotations

import os
import re

import structlog

from eval.medxpertqa.d11.types import (
    CandidateHypothesis,
    ClinicalAbstraction,
    RepairClaim,
)
from eval.medxpertqa.retry_utils import retry_with_backoff

log = structlog.get_logger(__name__)

GROQ_MODEL = "qwen/qwen3-32b"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

MAX_REPAIR_PAIRS = 5

_REPAIR_PROMPT = """\
You are a clinical discriminator. Given two candidate hypotheses and a clinical case, identify the key discriminating factor.

This pair was previously unresolved. Provide a MORE specific discriminator using case details.

Rules:
- Do NOT choose a final diagnosis
- Do NOT explain both candidates in full
- Identify ONE specific discriminator that separates these two candidates
- Reference a specific case clue
- Be MORE specific than a general textbook distinction

CLINICAL PROBLEM: {clinical_problem}
KEY FINDINGS: {key_findings}

CANDIDATE_A: {cand_a_id} -- {cand_a_label}
CANDIDATE_B: {cand_b_id} -- {cand_b_label}

Respond:
DISCRIMINATOR: [the key clinical feature that separates A from B]
CASE_CLUE: [the specific finding from the case relevant to this discriminator]
SUPPORTS: [candidate_id that this discriminator supports, or "neither" or "unclear"]
RULES_OUT: [candidate_id that this discriminator argues against, or "neither" or "unclear"]
CONFIDENCE: [low|medium|high]

/no_think
"""


# ---------------------------------------------------------------------------
# Groq helpers
# ---------------------------------------------------------------------------

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
    pattern = re.compile(rf"^{re.escape(header)}\s*:\s*(.+)$", re.MULTILINE | re.IGNORECASE)
    m = pattern.search(text)
    return m.group(1).strip() if m else ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def repair_discriminators(
    unresolved_pairs: list[tuple[str, str]],
    missing_discriminators: list[tuple[str, str]],
    abstraction: ClinicalAbstraction,
    candidates: list[CandidateHypothesis],
    case_id: str = "",
) -> list[RepairClaim]:
    """Generate targeted discriminators for unresolved/missing pairs.

    Combines unresolved_pairs and missing_discriminators, deduplicates,
    and runs up to MAX_REPAIR_PAIRS repair calls.
    """
    cand_map = {c.candidate_id: c for c in candidates}
    findings_str = ", ".join(abstraction.key_findings) if abstraction.key_findings else "none specified"

    # Deduplicate pairs (normalize ordering)
    seen: set[tuple[str, str]] = set()
    all_pairs: list[tuple[str, str]] = []
    for pair in unresolved_pairs + missing_discriminators:
        key = (min(pair[0], pair[1]), max(pair[0], pair[1]))
        if key not in seen:
            seen.add(key)
            all_pairs.append(key)

    all_pairs = all_pairs[:MAX_REPAIR_PAIRS]

    results: list[RepairClaim] = []

    for pair in all_pairs:
        cand_a = cand_map.get(pair[0])
        cand_b = cand_map.get(pair[1])
        if not cand_a or not cand_b:
            log.warning("repair.missing_candidate", case_id=case_id, pair=pair)
            continue

        prompt = _REPAIR_PROMPT.format(
            clinical_problem=abstraction.clinical_problem,
            key_findings=findings_str,
            cand_a_id=cand_a.candidate_id,
            cand_a_label=cand_a.candidate_label,
            cand_b_id=cand_b.candidate_id,
            cand_b_label=cand_b.candidate_label,
        )

        try:
            raw = _call_groq(prompt, label=f"repair:{case_id}:{pair[0]}v{pair[1]}")
            claim_text = _parse_line(raw, "DISCRIMINATOR") or "unresolved"
            supports = _parse_line(raw, "SUPPORTS") or "unclear"
            rules_out = _parse_line(raw, "RULES_OUT") or "unclear"
            case_clue = _parse_line(raw, "CASE_CLUE") or ""
            confidence_raw = _parse_line(raw, "CONFIDENCE").lower()
            confidence = confidence_raw if confidence_raw in ("low", "medium", "high") else "low"

            results.append(RepairClaim(
                pair=pair,
                claim=claim_text,
                supports=supports,
                rules_out=rules_out,
                case_clue=case_clue,
                confidence=confidence,
            ))
        except Exception as e:
            log.warning(
                "repair.pair_failed",
                case_id=case_id,
                pair=pair,
                err=repr(e)[:200],
            )

    log.info("repair.done", case_id=case_id, n_repairs=len(results))
    return results
