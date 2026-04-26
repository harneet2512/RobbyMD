"""D11 Stage 3 -- Pairwise discriminator tournament.

Deterministic pair selection (no model call), then one model call per pair
to generate the clinical discriminator between two candidate hypotheses.
"""
from __future__ import annotations

import os
import re
from typing import Callable

import structlog

from eval.medxpertqa.d11.types import (
    CandidateEvidence,
    CandidateHypothesis,
    ClinicalAbstraction,
    PairwiseDiscriminator,
)
from eval.medxpertqa.retry_utils import retry_with_backoff

log = structlog.get_logger(__name__)

GROQ_MODEL = "qwen/qwen3-32b"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

ChatFn = Callable[[str, str, int], str]

MAX_PAIRS = 5

_DISCRIMINATOR_PROMPT = """\
You are a clinical discriminator. Given two candidate hypotheses and a clinical case, identify the key discriminating factor.

Rules:
- Do NOT choose a final diagnosis
- Do NOT explain both candidates in full
- Identify ONE specific discriminator that separates these two candidates
- Reference a specific case clue

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
WHY_DECISIVE: [one sentence explaining why this discriminator matters]

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


# ---------------------------------------------------------------------------
# Deterministic pair selection
# ---------------------------------------------------------------------------

def select_pairs(
    evidence: list[CandidateEvidence],
    trap_candidates: list[str],
) -> list[tuple[str, str]]:
    """Deterministic pair selection -- NO model call.

    Priority:
    1. Two supported candidates without explicit discriminator
    2. Supported + trap candidate
    3. Leading + runner-up without rule-out
    4. Any unresolved pair

    Returns up to MAX_PAIRS pairs.
    """
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def _add(a: str, b: str) -> bool:
        key = (min(a, b), max(a, b))
        if key in seen or len(pairs) >= MAX_PAIRS:
            return False
        seen.add(key)
        pairs.append(key)
        return True

    supported = [e for e in evidence if e.net_status == "supported"]
    contradicted = [e for e in evidence if e.net_status == "contradicted"]
    unresolved = [e for e in evidence if e.net_status == "unresolved"]
    trap_ev = [e for e in evidence if e.candidate_id in trap_candidates]

    # Priority 1: pairs of supported candidates
    for i, a in enumerate(supported):
        for b in supported[i + 1:]:
            _add(a.candidate_id, b.candidate_id)

    # Priority 2: supported + trap
    for s in supported:
        for t in trap_ev:
            if s.candidate_id != t.candidate_id:
                _add(s.candidate_id, t.candidate_id)

    # Priority 3: supported + unresolved (leading vs runner-up)
    for s in supported:
        for u in unresolved:
            _add(s.candidate_id, u.candidate_id)

    # Priority 4: any remaining unresolved pairs
    all_ids = [e.candidate_id for e in evidence]
    for i, a in enumerate(all_ids):
        for b in all_ids[i + 1:]:
            _add(a, b)

    log.info("pairs.selected", n_pairs=len(pairs))
    return pairs[:MAX_PAIRS]


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def _parse_line(text: str, header: str) -> str:
    pattern = re.compile(rf"^{re.escape(header)}\s*:\s*(.+)$", re.MULTILINE | re.IGNORECASE)
    m = pattern.search(text)
    return m.group(1).strip() if m else ""


def _parse_discriminator(
    raw: str,
    pair: tuple[str, str],
) -> PairwiseDiscriminator:
    """Parse a single discriminator response."""
    discriminator = _parse_line(raw, "DISCRIMINATOR") or "unresolved"
    case_clue = _parse_line(raw, "CASE_CLUE") or ""
    supports = _parse_line(raw, "SUPPORTS") or "unclear"
    rules_out = _parse_line(raw, "RULES_OUT") or "unclear"
    confidence_raw = _parse_line(raw, "CONFIDENCE").lower()
    confidence = confidence_raw if confidence_raw in ("low", "medium", "high") else "low"
    why_decisive = _parse_line(raw, "WHY_DECISIVE") or ""

    return PairwiseDiscriminator(
        pair=pair,
        discriminator=discriminator,
        case_clue=case_clue,
        supports=supports,
        rules_out=rules_out,
        confidence=confidence,
        why_decisive=why_decisive,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_pairwise_tournament(
    pairs: list[tuple[str, str]],
    abstraction: ClinicalAbstraction,
    candidates: list[CandidateHypothesis],
    case_id: str = "",
    chat_fn: ChatFn | None = None,
) -> list[PairwiseDiscriminator]:
    """Generate discriminator for each pair."""
    cand_map = {c.candidate_id: c for c in candidates}
    findings_str = ", ".join(abstraction.key_findings) if abstraction.key_findings else "none specified"
    _fn = chat_fn or (lambda p, l, m: _call_groq(p, label=l, max_tokens=m))
    results: list[PairwiseDiscriminator] = []

    for pair in pairs:
        cand_a = cand_map.get(pair[0])
        cand_b = cand_map.get(pair[1])
        if not cand_a or not cand_b:
            log.warning("tournament.missing_candidate", case_id=case_id, pair=pair)
            continue

        prompt = _DISCRIMINATOR_PROMPT.format(
            clinical_problem=abstraction.clinical_problem,
            key_findings=findings_str,
            cand_a_id=cand_a.candidate_id,
            cand_a_label=cand_a.candidate_label,
            cand_b_id=cand_b.candidate_id,
            cand_b_label=cand_b.candidate_label,
        )

        try:
            raw = _fn(prompt, f"discriminator:{case_id}:{pair[0]}v{pair[1]}", 2048)
            disc = _parse_discriminator(raw, pair)
            results.append(disc)
        except Exception as e:
            log.warning(
                "tournament.pair_failed",
                case_id=case_id,
                pair=pair,
                err=repr(e)[:200],
            )
            results.append(PairwiseDiscriminator(
                pair=pair,
                discriminator="error",
                case_clue="",
                supports="unclear",
                rules_out="unclear",
                confidence="low",
                why_decisive="",
            ))

    log.info("tournament.done", case_id=case_id, n_discriminators=len(results))
    return results
