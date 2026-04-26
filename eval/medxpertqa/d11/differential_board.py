"""D11 Stage 2 -- Differential board with 3 parallel reasoning roles.

Three independent specialist roles run concurrently via ThreadPoolExecutor:
  A. Mechanism specialist -- supporting/contradicting clues per candidate
  B. Option skeptic -- missing required clues and contradictions
  C. Trap detector -- identifies tempting-but-wrong candidates

Supports a combined single-call mode (``run_combined_board``) that issues
one model call returning all three role outputs as JSON, reducing total
API calls per case from 3 to 1.
"""
from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

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

ChatFn = Callable[[str, str, int], str]

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_MECHANISM_PROMPT = """\
You are a clinical mechanism specialist. Given a clinical case and candidate hypotheses, analyze how EVERY candidate's mechanism fits the clinical picture.

CRITICAL: You MUST produce a CANDIDATE block for EVERY candidate listed below. Do NOT skip any candidate. If a candidate has no supporting case evidence, write SUPPORTING: none and explain why.

Rules:
- Do NOT choose a correct answer
- Do NOT rank the candidates
- Analyze ALL {n_candidates} candidates below
- For each candidate, list supporting and contradicting clues from the case

CLINICAL PROBLEM: {clinical_problem}
KEY FINDINGS: {key_findings}

CANDIDATES ({n_candidates} total -- you must analyze ALL {n_candidates}):
{candidates_block}

For EVERY candidate above, respond with this exact format:
CANDIDATE: {{candidate_id}}
SUPPORTING: [comma-separated clues from the case that support this, or "none"]
CONTRADICTING: [comma-separated clues from the case that contradict this, or "none"]

You MUST have exactly {n_candidates} CANDIDATE blocks.

/no_think
"""

_SKEPTIC_PROMPT = """\
You are a clinical skeptic. For EVERY candidate hypothesis below, identify why it may be wrong or incomplete.

CRITICAL: You MUST produce a CANDIDATE block for EVERY candidate listed below. Do NOT skip any candidate. If you cannot find missing clues or contradictions, write "none" explicitly.

Rules:
- Do NOT choose a correct answer
- Focus on what is MISSING that would be required for each candidate
- Identify contradictions between the case and each candidate
- Analyze ALL {n_candidates} candidates

CLINICAL PROBLEM: {clinical_problem}
KEY FINDINGS: {key_findings}

CANDIDATES ({n_candidates} total -- you must analyze ALL {n_candidates}):
{candidates_block}

For EVERY candidate above:
CANDIDATE: {{candidate_id}}
MISSING_REQUIRED: [comma-separated findings that SHOULD be present if this were correct but are NOT mentioned, or "none"]
CONTRADICTIONS: [comma-separated case findings that argue AGAINST this candidate, or "none"]

You MUST have exactly {n_candidates} CANDIDATE blocks.

/no_think
"""

_TRAP_PROMPT = """\
You are a clinical trap detector. For EVERY candidate below, assess whether it is a trap (tempting but wrong).

CRITICAL: You MUST produce a TRAP_CANDIDATE block for EVERY candidate listed below, not just the traps. For non-traps, set WHY_TEMPTING and EXPOSING_CLUE to "none".

Rules:
- Do NOT choose a correct answer
- Assess ALL {n_candidates} candidates
- Identify candidates that look superficially correct but have a specific flaw

CLINICAL PROBLEM: {clinical_problem}
KEY FINDINGS: {key_findings}

CANDIDATES ({n_candidates} total -- you must assess ALL {n_candidates}):
{candidates_block}

For EVERY candidate above:
TRAP_CANDIDATE: {{candidate_id}}
IS_TRAP: [yes | no]
WHY_TEMPTING: [why this looks correct at first glance, or "none"]
EXPOSING_CLUE: [specific case finding that reveals this is wrong, or "none"]

You MUST have exactly {n_candidates} TRAP_CANDIDATE blocks.

/no_think
"""

_SINGLE_REPAIR_PROMPT = """\
You are a clinical analyst. The previous board analysis omitted candidate "{candidate_id}" ({candidate_label}).

Analyze ONLY this one candidate. Do NOT answer the question. Do NOT choose a correct answer. Do NOT discuss other candidates.

CLINICAL PROBLEM: {clinical_problem}
KEY FINDINGS: {key_findings}

Respond with EXACTLY this JSON object (no other text):
{{
  "candidate_id": "{candidate_id}",
  "supporting_clues": ["case clue 1", "case clue 2"],
  "contradicting_clues": ["case clue that argues against"],
  "mechanism_fit": "strong | partial | weak | none",
  "missing_required_clues": ["finding expected but absent"],
  "contradictions": ["case finding arguing against"],
  "is_trap": false,
  "why_tempting": "",
  "exposing_clue": ""
}}

Rules:
- Use empty arrays [] if no clues apply, not "none"
- supporting_clues: case-specific evidence that supports this candidate
- contradicting_clues: case-specific evidence that argues against
- mechanism_fit: how well the candidate mechanism explains the clinical picture
- is_trap: true only if the candidate is tempting but specifically wrong
- Output ONLY the JSON object, nothing else

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

        is_trap = True
        why_tempting = ""
        exposing_clue = ""
        for line in lines[1:]:
            line = line.strip()
            trap_match = re.match(r"IS_TRAP\s*:\s*(.+)", line, re.IGNORECASE)
            why_match = re.match(r"WHY_TEMPTING\s*:\s*(.+)", line, re.IGNORECASE)
            exp_match = re.match(r"EXPOSING_CLUE\s*:\s*(.+)", line, re.IGNORECASE)
            if trap_match:
                val = trap_match.group(1).strip().lower()
                is_trap = val in ("yes", "true")
            elif why_match:
                why_tempting = why_match.group(1).strip()
            elif exp_match:
                exposing_clue = exp_match.group(1).strip()

        if why_tempting.lower() in ("none", "n/a", ""):
            why_tempting = ""
        if exposing_clue.lower() in ("none", "n/a", ""):
            exposing_clue = ""

        results.append(TrapOutput(
            candidate_id=cid,
            is_trap=is_trap,
            why_tempting=why_tempting,
            exposing_clue=exposing_clue,
        ))

    seen = {r.candidate_id for r in results}
    for c in candidates:
        if c.candidate_id not in seen:
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
    chat_fn: ChatFn | None = None,
) -> BoardResults:
    """Run 3 independent reasoning roles in parallel, then repair coverage."""
    candidates_block = _format_candidates(candidates)
    findings_str = _format_findings(abstraction.key_findings)
    n_cands = len(candidates)

    fmt = dict(
        clinical_problem=abstraction.clinical_problem,
        key_findings=findings_str,
        candidates_block=candidates_block,
        n_candidates=n_cands,
    )

    mechanism_prompt = _MECHANISM_PROMPT.format(**fmt)
    skeptic_prompt = _SKEPTIC_PROMPT.format(**fmt)
    trap_prompt = _TRAP_PROMPT.format(**fmt)

    _fn = chat_fn or (lambda p, l, m: _call_groq(p, label=l, max_tokens=m))
    max_tok = max(3072, n_cands * 300)

    results: dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            pool.submit(_fn, mechanism_prompt, f"mechanism:{case_id}", max_tok): "mechanism",
            pool.submit(_fn, skeptic_prompt, f"skeptic:{case_id}", max_tok): "skeptic",
            pool.submit(_fn, trap_prompt, f"trap:{case_id}", max_tok): "trap",
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

    board = BoardResults(
        mechanism_outputs=mechanism_outputs,
        skeptic_outputs=skeptic_outputs,
        trap_outputs=trap_outputs,
    )

    # Coverage repair: find candidates with empty evidence across all roles
    board = _repair_missing_candidates(
        board, abstraction, candidates, case_id, _fn,
    )

    log.info(
        "board.done",
        case_id=case_id,
        n_mechanism=len(board.mechanism_outputs),
        n_skeptic=len(board.skeptic_outputs),
        n_traps=sum(1 for t in board.trap_outputs if t.is_trap),
    )

    return board


def _repair_missing_candidates(
    board: BoardResults,
    abstraction: ClinicalAbstraction,
    candidates: list[CandidateHypothesis],
    case_id: str,
    chat_fn: ChatFn,
) -> BoardResults:
    """Repair missing candidates one at a time with JSON output."""
    from eval.medxpertqa.d11.board_coverage_validator import validate_board_coverage

    coverage = validate_board_coverage(candidates, board)
    if coverage.coverage_complete:
        return board

    missing_ids = set(coverage.empty_candidate_ids + coverage.missing_candidate_ids)
    if not missing_ids:
        return board

    cand_map = {c.candidate_id: c for c in candidates}
    mech_by_cid = {m.candidate_id: m for m in board.mechanism_outputs}
    skep_by_cid = {s.candidate_id: s for s in board.skeptic_outputs}
    trap_by_cid = {t.candidate_id: t for t in board.trap_outputs}

    repaired = 0
    fallbacks = 0

    log.info(
        "board.repair_missing",
        case_id=case_id,
        n_missing=len(missing_ids),
        missing_ids=sorted(missing_ids),
    )

    for cid in sorted(missing_ids):
        cand = cand_map.get(cid)
        if not cand:
            continue

        record = _repair_single_candidate(
            cid, cand.candidate_label, abstraction, case_id, chat_fn,
        )

        if record is not None:
            mech_by_cid[cid] = MechanismOutput(
                cid,
                _safe_list(record.get("supporting_clues")),
                _safe_list(record.get("contradicting_clues")),
            )
            skep_by_cid[cid] = SkepticOutput(
                cid,
                _safe_list(record.get("missing_required_clues")),
                _safe_list(record.get("contradictions")),
            )
            is_trap = bool(record.get("is_trap", False))
            trap_by_cid[cid] = TrapOutput(
                cid,
                is_trap,
                str(record.get("why_tempting", "")) if is_trap else "",
                str(record.get("exposing_clue", "")) if is_trap else "",
            )
            repaired += 1
        else:
            mech_by_cid[cid] = MechanismOutput(cid, [], [])
            skep_by_cid[cid] = SkepticOutput(
                cid,
                ["Board repair failed; candidate not fully evaluated."],
                [],
            )
            trap_by_cid[cid] = TrapOutput(cid, False, "", "")
            fallbacks += 1
            log.warning(
                "board.repair_fallback_used",
                case_id=case_id,
                candidate_id=cid,
            )

    log.info(
        "board.repair_done",
        case_id=case_id,
        repaired=repaired,
        fallbacks=fallbacks,
    )

    return BoardResults(
        mechanism_outputs=list(mech_by_cid.values()),
        skeptic_outputs=list(skep_by_cid.values()),
        trap_outputs=list(trap_by_cid.values()),
    )


def _safe_list(val: object) -> list[str]:
    """Coerce value to list[str], handling None and non-list types."""
    if val is None:
        return []
    if isinstance(val, list):
        return [str(x) for x in val if x]
    return []


def _extract_repair_json(
    text: str,
    candidate_id: str,
    candidate_label: str = "",
) -> dict | None:
    """Extract JSON object from repair response, with multiple fallbacks."""
    stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    for source in [stripped, text.strip()]:
        obj = _try_parse_json_object(source, candidate_id)
        if obj is not None:
            return _normalize_cid(obj, candidate_id, candidate_label)

    # Try code fence
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)```", stripped, re.DOTALL)
    if fence_match:
        obj = _try_parse_json_object(fence_match.group(1).strip(), candidate_id)
        if obj is not None:
            return _normalize_cid(obj, candidate_id, candidate_label)

    # Try largest balanced { ... } — handles truncated responses
    best = _extract_largest_balanced_json(stripped)
    if best is not None:
        return _normalize_cid(best, candidate_id, candidate_label)

    return None


def _try_parse_json_object(text: str, candidate_id: str) -> dict | None:
    """Try json.loads; handle dict and list results."""
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, list):
            for item in obj:
                if isinstance(item, dict) and item.get("candidate_id") == candidate_id:
                    return item
            if obj and isinstance(obj[0], dict):
                return obj[0]
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def _extract_largest_balanced_json(text: str) -> dict | None:
    """Find the largest balanced JSON object in text.

    Handles truncated responses by trying progressively smaller
    substrings ending at each closing brace.
    """
    idx = text.find("{")
    if idx < 0:
        return None

    candidates: list[str] = []
    depth = 0
    for i in range(idx, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                candidates.append(text[idx:i + 1])

    # Try from largest to smallest
    for candidate_str in reversed(candidates):
        try:
            obj = json.loads(candidate_str)
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, ValueError):
            continue

    return None


def _normalize_cid(
    obj: dict,
    candidate_id: str,
    candidate_label: str = "",
) -> dict:
    """Normalize candidate_id: accept label exact-match as valid."""
    returned_cid = obj.get("candidate_id", "")
    if returned_cid == candidate_id:
        return obj
    if not returned_cid and candidate_label:
        returned_label = obj.get("candidate_label", "")
        if returned_label and returned_label.strip().lower() == candidate_label.strip().lower():
            obj["candidate_id"] = candidate_id
            return obj
    if not returned_cid:
        obj["candidate_id"] = candidate_id
    return obj


def _repair_single_candidate(
    candidate_id: str,
    candidate_label: str,
    abstraction: ClinicalAbstraction,
    case_id: str,
    chat_fn: ChatFn,
    max_attempts: int = 3,
) -> dict | None:
    """Repair a single missing candidate with JSON output. Returns parsed dict or None."""
    findings_str = _format_findings(abstraction.key_findings)

    prompt = _SINGLE_REPAIR_PROMPT.format(
        candidate_id=candidate_id,
        candidate_label=candidate_label,
        clinical_problem=abstraction.clinical_problem,
        key_findings=findings_str,
    )

    for attempt in range(1, max_attempts + 1):
        try:
            raw = chat_fn(prompt, f"board_repair_1x1:{case_id}:{candidate_id}", 2048)
            record = _extract_repair_json(raw, candidate_id, candidate_label)

            if record is None:
                log.warning(
                    "board.repair_parse_failed",
                    case_id=case_id,
                    candidate_id=candidate_id,
                    attempt=attempt,
                    preview=raw[:300],
                )
                continue

            returned_cid = record.get("candidate_id", "")
            if returned_cid and returned_cid != candidate_id:
                log.warning(
                    "board.repair_wrong_cid",
                    case_id=case_id,
                    expected=candidate_id,
                    got=returned_cid,
                    attempt=attempt,
                )
                continue

            record["candidate_id"] = candidate_id
            log.info(
                "board.repair_candidate_ok",
                case_id=case_id,
                candidate_id=candidate_id,
                attempt=attempt,
                sup=len(_safe_list(record.get("supporting_clues"))),
                con=len(_safe_list(record.get("contradicting_clues"))),
            )
            return record

        except Exception as e:
            log.warning(
                "board.repair_call_failed",
                case_id=case_id,
                candidate_id=candidate_id,
                attempt=attempt,
                err=repr(e)[:200],
            )

    log.warning(
        "board.repair_exhausted",
        case_id=case_id,
        candidate_id=candidate_id,
    )
    return None


# ---------------------------------------------------------------------------
# Combined board (single call returning all 3 roles as JSON)
# ---------------------------------------------------------------------------

_COMBINED_BOARD_PROMPT = """\
You are a clinical case analysis board with three independent specialist roles.
Given a clinical case and candidate hypotheses, provide ALL THREE analyses.

CRITICAL: You MUST analyze ALL {n_candidates} candidates. Do NOT skip any.

CLINICAL PROBLEM: {clinical_problem}
KEY FINDINGS: {key_findings}

CANDIDATES ({n_candidates} total -- analyze ALL):
{candidates_block}

Respond with EXACTLY this JSON structure (no other text).
Each array MUST have exactly {n_candidates} entries, one per candidate:
{{
  "mechanism_specialist": [
    {{"candidate_id": "<id>", "supporting": ["clue1"], "contradicting": ["clue1"]}}
  ],
  "option_skeptic": [
    {{"candidate_id": "<id>", "missing_required": ["finding1"], "contradictions": ["contra1"]}}
  ],
  "trap_detector": [
    {{"candidate_id": "<id>", "is_trap": true/false, "why_tempting": "...", "exposing_clue": "..."}}
  ]
}}

Rules:
- Do NOT choose a correct answer
- Do NOT rank the candidates
- ALL {n_candidates} candidates must appear in ALL THREE sections
- If no case evidence applies, use empty arrays [] and set is_trap to false
- For non-traps, set why_tempting and exposing_clue to ""

/no_think
"""

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)```", re.DOTALL)


def _extract_combined_json(text: str) -> dict | None:
    """Extract JSON from combined board response."""
    stripped = _THINK_RE.sub("", text).strip()

    for attempt_text in [stripped, text]:
        try:
            return json.loads(attempt_text)
        except (json.JSONDecodeError, ValueError):
            pass

        m = _JSON_FENCE_RE.search(attempt_text)
        if m:
            try:
                return json.loads(m.group(1).strip())
            except (json.JSONDecodeError, ValueError):
                pass

        idx = attempt_text.find("{")
        if idx >= 0:
            try:
                return json.loads(attempt_text[idx:])
            except (json.JSONDecodeError, ValueError):
                pass

    return None


def _json_to_board(data: dict, candidates: list[CandidateHypothesis]) -> BoardResults:
    """Convert combined JSON response to BoardResults."""
    candidate_ids = {c.candidate_id for c in candidates}

    mechanism_outputs: list[MechanismOutput] = []
    for entry in data.get("mechanism_specialist", []):
        cid = entry.get("candidate_id", "")
        if cid not in candidate_ids:
            continue
        mechanism_outputs.append(MechanismOutput(
            candidate_id=cid,
            supporting_clues=entry.get("supporting", []),
            contradicting_clues=entry.get("contradicting", []),
        ))
    seen_mech = {m.candidate_id for m in mechanism_outputs}
    for c in candidates:
        if c.candidate_id not in seen_mech:
            mechanism_outputs.append(MechanismOutput(c.candidate_id, [], []))

    skeptic_outputs: list[SkepticOutput] = []
    for entry in data.get("option_skeptic", []):
        cid = entry.get("candidate_id", "")
        if cid not in candidate_ids:
            continue
        skeptic_outputs.append(SkepticOutput(
            candidate_id=cid,
            missing_required_clues=entry.get("missing_required", []),
            contradictions=entry.get("contradictions", []),
        ))
    seen_skep = {s.candidate_id for s in skeptic_outputs}
    for c in candidates:
        if c.candidate_id not in seen_skep:
            skeptic_outputs.append(SkepticOutput(c.candidate_id, [], []))

    trap_outputs: list[TrapOutput] = []
    for entry in data.get("trap_detector", []):
        cid = entry.get("candidate_id", "")
        if cid not in candidate_ids:
            continue
        trap_outputs.append(TrapOutput(
            candidate_id=cid,
            is_trap=bool(entry.get("is_trap", False)),
            why_tempting=entry.get("why_tempting", ""),
            exposing_clue=entry.get("exposing_clue", ""),
        ))
    seen_trap = {t.candidate_id for t in trap_outputs}
    for c in candidates:
        if c.candidate_id not in seen_trap:
            trap_outputs.append(TrapOutput(c.candidate_id, False, "", ""))

    return BoardResults(
        mechanism_outputs=mechanism_outputs,
        skeptic_outputs=skeptic_outputs,
        trap_outputs=trap_outputs,
    )


def run_combined_board(
    abstraction: ClinicalAbstraction,
    candidates: list[CandidateHypothesis],
    case_id: str = "",
    chat_fn: ChatFn | None = None,
    fallback_to_separate: bool = True,
) -> BoardResults:
    """Run all 3 board roles in a single model call returning JSON.

    Falls back to 3 separate calls if JSON parsing fails and
    ``fallback_to_separate`` is True.
    """
    if chat_fn is None:
        return run_differential_board(abstraction, candidates, case_id)

    candidates_block = _format_candidates(candidates)
    findings_str = _format_findings(abstraction.key_findings)

    prompt = _COMBINED_BOARD_PROMPT.format(
        clinical_problem=abstraction.clinical_problem,
        key_findings=findings_str,
        candidates_block=candidates_block,
        n_candidates=len(candidates),
    )

    try:
        max_tok = max(4096, len(candidates) * 400)
        raw = chat_fn(prompt, f"combined_board:{case_id}", max_tok)
        data = _extract_combined_json(raw)
        if data is not None and isinstance(data, dict):
            board = _json_to_board(data, candidates)
            log.info(
                "combined_board.done",
                case_id=case_id,
                n_mechanism=len(board.mechanism_outputs),
                n_skeptic=len(board.skeptic_outputs),
                n_traps=sum(1 for t in board.trap_outputs if t.is_trap),
            )
            return board
        else:
            log.warning("combined_board.json_failed", case_id=case_id, preview=raw[:200])
    except Exception as e:
        log.warning("combined_board.call_failed", case_id=case_id, err=repr(e)[:200])

    if fallback_to_separate:
        log.info("combined_board.fallback_to_separate", case_id=case_id)
        return run_differential_board(abstraction, candidates, case_id, chat_fn=chat_fn)

    return BoardResults(
        mechanism_outputs=[MechanismOutput(c.candidate_id, [], []) for c in candidates],
        skeptic_outputs=[SkepticOutput(c.candidate_id, [], []) for c in candidates],
        trap_outputs=[TrapOutput(c.candidate_id, False, "", "") for c in candidates],
    )
