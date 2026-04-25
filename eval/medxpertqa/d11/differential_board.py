"""D11 Stage 2 -- Differential board with 3 parallel reasoning roles.

Three independent specialist roles run concurrently via ThreadPoolExecutor:
  A. Mechanism specialist -- supporting/contradicting clues per candidate
  B. Option skeptic -- missing required clues and contradictions
  C. Trap detector -- identifies tempting-but-wrong candidates
"""
from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import structlog

from eval.medxpertqa.d11.types import (
    BoardResults,
    CandidateHypothesis,
    ClinicalAbstraction,
    MechanismOutput,
    SkepticOutput,
    TrapOutput,
)
from eval.medxpertqa.retry_utils import retry_with_backoff

log = structlog.get_logger(__name__)

GROQ_MODEL = "qwen/qwen3-32b"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_MECHANISM_PROMPT = """\
You are a clinical mechanism specialist. Given a clinical case and candidate hypotheses, analyze how each candidate's mechanism fits the clinical picture.

Rules:
- Do NOT choose a correct answer
- Do NOT rank the candidates
- For each candidate, list supporting and contradicting clues from the case

CLINICAL PROBLEM: {clinical_problem}
KEY FINDINGS: {key_findings}

CANDIDATES:
{candidates_block}

For each candidate, respond:
CANDIDATE: {{candidate_id}}
SUPPORTING: [comma-separated clues from the case that support this]
CONTRADICTING: [comma-separated clues from the case that contradict this]

/no_think
"""

_SKEPTIC_PROMPT = """\
You are a clinical skeptic. For each candidate hypothesis, identify why it may be wrong or incomplete.

Rules:
- Do NOT choose a correct answer
- Focus on what is MISSING that would be required for each candidate
- Identify contradictions between the case and each candidate

CLINICAL PROBLEM: {clinical_problem}
KEY FINDINGS: {key_findings}

CANDIDATES:
{candidates_block}

For each candidate:
CANDIDATE: {{candidate_id}}
MISSING_REQUIRED: [comma-separated findings that SHOULD be present if this were correct but are NOT mentioned]
CONTRADICTIONS: [comma-separated case findings that argue AGAINST this candidate]

/no_think
"""

_TRAP_PROMPT = """\
You are a clinical trap detector. Identify candidates that are tempting but likely wrong.

Rules:
- Do NOT choose a correct answer
- Identify candidates that look superficially correct but have a specific flaw
- Explain what makes them tempting and what case clue exposes the trap

CLINICAL PROBLEM: {clinical_problem}
KEY FINDINGS: {key_findings}

CANDIDATES:
{candidates_block}

For each candidate that is a potential trap:
TRAP_CANDIDATE: {{candidate_id}}
WHY_TEMPTING: [why this looks correct at first glance]
EXPOSING_CLUE: [specific case finding that reveals this is wrong]

For candidates that are NOT traps, do not include them.

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


def _call_groq(prompt: str, label: str = "", max_tokens: int = 3072) -> str:
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
# Formatters
# ---------------------------------------------------------------------------

def _format_candidates(candidates: list[CandidateHypothesis]) -> str:
    return "\n".join(
        f"{c.candidate_id}: {c.candidate_label}" for c in candidates
    )


def _format_findings(findings: list[str]) -> str:
    return ", ".join(findings) if findings else "none specified"


def _parse_csv(value: str) -> list[str]:
    """Split comma-separated value into stripped non-empty items."""
    if not value or value.lower() in ("none", "n/a", ""):
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def _parse_mechanism(raw: str, candidates: list[CandidateHypothesis]) -> list[MechanismOutput]:
    """Parse mechanism specialist output into MechanismOutput list."""
    results: list[MechanismOutput] = []
    candidate_ids = {c.candidate_id for c in candidates}

    # Split into candidate blocks
    blocks = re.split(r"(?m)^CANDIDATE\s*:\s*", raw)
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        lines = block.split("\n")
        cid = lines[0].strip()
        if cid not in candidate_ids:
            continue

        supporting: list[str] = []
        contradicting: list[str] = []
        for line in lines[1:]:
            line = line.strip()
            sup_match = re.match(r"SUPPORTING\s*:\s*(.+)", line, re.IGNORECASE)
            con_match = re.match(r"CONTRADICTING\s*:\s*(.+)", line, re.IGNORECASE)
            if sup_match:
                supporting = _parse_csv(sup_match.group(1))
            elif con_match:
                contradicting = _parse_csv(con_match.group(1))

        results.append(MechanismOutput(
            candidate_id=cid,
            supporting_clues=supporting,
            contradicting_clues=contradicting,
        ))

    # Ensure all candidates are represented
    seen = {r.candidate_id for r in results}
    for c in candidates:
        if c.candidate_id not in seen:
            results.append(MechanismOutput(
                candidate_id=c.candidate_id,
                supporting_clues=[],
                contradicting_clues=[],
            ))

    return results


def _parse_skeptic(raw: str, candidates: list[CandidateHypothesis]) -> list[SkepticOutput]:
    """Parse option skeptic output into SkepticOutput list."""
    results: list[SkepticOutput] = []
    candidate_ids = {c.candidate_id for c in candidates}

    blocks = re.split(r"(?m)^CANDIDATE\s*:\s*", raw)
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        lines = block.split("\n")
        cid = lines[0].strip()
        if cid not in candidate_ids:
            continue

        missing: list[str] = []
        contradictions: list[str] = []
        for line in lines[1:]:
            line = line.strip()
            miss_match = re.match(r"MISSING_REQUIRED\s*:\s*(.+)", line, re.IGNORECASE)
            con_match = re.match(r"CONTRADICTIONS?\s*:\s*(.+)", line, re.IGNORECASE)
            if miss_match:
                missing = _parse_csv(miss_match.group(1))
            elif con_match:
                contradictions = _parse_csv(con_match.group(1))

        results.append(SkepticOutput(
            candidate_id=cid,
            missing_required_clues=missing,
            contradictions=contradictions,
        ))

    seen = {r.candidate_id for r in results}
    for c in candidates:
        if c.candidate_id not in seen:
            results.append(SkepticOutput(
                candidate_id=c.candidate_id,
                missing_required_clues=[],
                contradictions=[],
            ))

    return results


def _parse_traps(raw: str, candidates: list[CandidateHypothesis]) -> list[TrapOutput]:
    """Parse trap detector output into TrapOutput list."""
    results: list[TrapOutput] = []
    candidate_ids = {c.candidate_id for c in candidates}

    blocks = re.split(r"(?m)^TRAP_CANDIDATE\s*:\s*", raw)
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        lines = block.split("\n")
        cid = lines[0].strip()
        if cid not in candidate_ids:
            continue

        why_tempting = ""
        exposing_clue = ""
        for line in lines[1:]:
            line = line.strip()
            why_match = re.match(r"WHY_TEMPTING\s*:\s*(.+)", line, re.IGNORECASE)
            exp_match = re.match(r"EXPOSING_CLUE\s*:\s*(.+)", line, re.IGNORECASE)
            if why_match:
                why_tempting = why_match.group(1).strip()
            elif exp_match:
                exposing_clue = exp_match.group(1).strip()

        results.append(TrapOutput(
            candidate_id=cid,
            is_trap=True,
            why_tempting=why_tempting,
            exposing_clue=exposing_clue,
        ))

    # Non-trap candidates get is_trap=False
    trap_ids = {r.candidate_id for r in results}
    for c in candidates:
        if c.candidate_id not in trap_ids:
            results.append(TrapOutput(
                candidate_id=c.candidate_id,
                is_trap=False,
                why_tempting="",
                exposing_clue="",
            ))

    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_differential_board(
    abstraction: ClinicalAbstraction,
    candidates: list[CandidateHypothesis],
    case_id: str = "",
) -> BoardResults:
    """Run 3 independent reasoning roles in parallel via Groq."""
    candidates_block = _format_candidates(candidates)
    findings_str = _format_findings(abstraction.key_findings)

    mechanism_prompt = _MECHANISM_PROMPT.format(
        clinical_problem=abstraction.clinical_problem,
        key_findings=findings_str,
        candidates_block=candidates_block,
    )
    skeptic_prompt = _SKEPTIC_PROMPT.format(
        clinical_problem=abstraction.clinical_problem,
        key_findings=findings_str,
        candidates_block=candidates_block,
    )
    trap_prompt = _TRAP_PROMPT.format(
        clinical_problem=abstraction.clinical_problem,
        key_findings=findings_str,
        candidates_block=candidates_block,
    )

    results: dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            pool.submit(_call_groq, mechanism_prompt, f"mechanism:{case_id}"): "mechanism",
            pool.submit(_call_groq, skeptic_prompt, f"skeptic:{case_id}"): "skeptic",
            pool.submit(_call_groq, trap_prompt, f"trap:{case_id}"): "trap",
        }
        for fut in as_completed(futures):
            role = futures[fut]
            try:
                results[role] = fut.result()
            except Exception as e:
                log.warning("board.role_failed", case_id=case_id, role=role, err=repr(e)[:200])
                results[role] = ""

    mechanism_outputs = _parse_mechanism(results.get("mechanism", ""), candidates)
    skeptic_outputs = _parse_skeptic(results.get("skeptic", ""), candidates)
    trap_outputs = _parse_traps(results.get("trap", ""), candidates)

    log.info(
        "board.done",
        case_id=case_id,
        n_mechanism=len(mechanism_outputs),
        n_skeptic=len(skeptic_outputs),
        n_traps=sum(1 for t in trap_outputs if t.is_trap),
    )

    return BoardResults(
        mechanism_outputs=mechanism_outputs,
        skeptic_outputs=skeptic_outputs,
        trap_outputs=trap_outputs,
    )
