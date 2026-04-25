"""Merge differential-board role outputs into per-candidate evidence.

Takes the three role outputs (mechanism specialist, option skeptic,
trap detector) and fuses them into a single :class:`CandidateEvidence`
record per candidate.  Pure Python — no API calls.
"""
from __future__ import annotations

import re

import structlog

from eval.medxpertqa.d11.types import (
    BoardResults,
    CandidateEvidence,
    CandidateHypothesis,
    Claim,
)

log = structlog.get_logger(__name__)

# --- Case-specificity heuristics ---------------------------------------------------

_CASE_SPECIFIC_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bin this case\b",
        r"\bthe patient\b",
        r"\bpresented with\b",
        r"\bhistory of\b",
        r"\bthis patient\b",
        r"\bpatient's\b",
        r"\bon examination\b",
        r"\blab shows\b",
        r"\bfindings show\b",
        r"\breported\b",
    ]
]

_NONE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"^\s*\[?\s*none\b",
        r"^\s*\[?\s*n/a\b",
        r"\bnot contradicted\b",
        r"\bno contradictions?\b",
        r"\bno missing\b",
        r"\bno specific\s+(missing|contradiction)",
        r"^\s*\[?\s*-?\s*$",
    ]
]


def _is_none_response(text: str) -> bool:
    """Return True if the text is a 'no data' response from the LLM."""
    if len(text.strip()) < 5:
        return True
    for pat in _NONE_PATTERNS:
        if pat.search(text):
            return True
    return False


_GENERIC_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bcan cause\b",
        r"\bis associated with\b",
        r"\bcommonly presents\b",
        r"\bmay lead to\b",
        r"\bis characterized by\b",
        r"\bis a common\b",
        r"\btypically\b",
        r"\bin general\b",
        r"\bis known to\b",
        r"\boften\b",
    ]
]


def _is_case_specific(text: str) -> bool:
    """Return True if the claim text references case-specific findings."""
    for pat in _CASE_SPECIFIC_PATTERNS:
        if pat.search(text):
            return True
    return False


def _is_generic(text: str) -> bool:
    """Return True if the claim text is a general medical statement."""
    for pat in _GENERIC_PATTERNS:
        if pat.search(text):
            return True
    return False


def _determine_strength(
    text: str,
    case_specific: bool,
    is_discriminator: bool,
) -> str:
    """Assign claim strength: strong > medium > weak.

    - strong: case_specific AND discriminates between candidates
    - medium: case_specific OR discriminates (but not both)
    - weak:   generic (neither)
    """
    if case_specific and is_discriminator:
        return "strong"
    if case_specific or is_discriminator:
        return "medium"
    return "weak"


def _build_claim(
    text: str,
    source_role: str,
    related_candidate_id: str,
    is_discriminator: bool = False,
) -> Claim:
    """Construct a single :class:`Claim` with heuristic fields."""
    case_specific = _is_case_specific(text)
    if not case_specific:
        case_specific = not _is_generic(text)
    strength = _determine_strength(text, case_specific, is_discriminator)
    return Claim(
        claim=text,
        case_specific=case_specific,
        source_role=source_role,
        strength=strength,
        related_candidate_id=related_candidate_id,
    )


_STRENGTH_WEIGHTS = {"strong": 3.0, "medium": 2.0, "weak": 1.0}
_MISSING_CLUE_PENALTY = 0.75


def _score_claims(claims: list[Claim]) -> float:
    return sum(_STRENGTH_WEIGHTS.get(c.strength, 0.0) for c in claims)


def _determine_net_status(
    candidate_id: str,
    supporting: list[Claim],
    contradicting: list[Claim],
    missing: list[str],
    is_trap: bool,
) -> str:
    """Compute net_status using weighted evidence balance.

    Scoring:
      strong claim: ±3, medium: ±2, weak: ±1
      generic support (case_specific=False): +0.25
      missing required clue: -0.75
      trap flag: -2.0 (the exposing_clue is already in contradictions)
      fatal contradiction (strong + case_specific): blocks "supported"
    """
    support_score = 0.0
    case_specific_support_score = 0.0
    for c in supporting:
        w = _STRENGTH_WEIGHTS.get(c.strength, 0.0)
        if c.case_specific:
            support_score += w
            case_specific_support_score += w
        else:
            support_score += 0.25

    contradiction_score = _score_claims(contradicting)
    missing_penalty = len(missing) * _MISSING_CLUE_PENALTY
    trap_penalty = 2.0 if is_trap else 0.0

    fatal_contradictions = [
        c for c in contradicting
        if c.strength == "strong" and c.case_specific
    ]

    total_negative = contradiction_score + missing_penalty + trap_penalty

    # 1. supported: meaningful case-specific support outweighs all negatives
    #    Fatal contradictions block only if they represent substantial negative weight
    fatal_blocks = (
        fatal_contradictions
        and contradiction_score >= support_score * 0.5
    )
    if (
        case_specific_support_score >= 2.0
        and support_score > total_negative
        and not fatal_blocks
    ):
        return "supported"

    # 2. trap: trap flag + weak support (trap detector's judgment holds)
    if is_trap and case_specific_support_score < 4.0:
        return "trap"

    # 3. contradicted: negatives dominate or fatal + weak support
    if contradiction_score > support_score:
        return "contradicted"
    if fatal_contradictions and case_specific_support_score < 2.0:
        return "contradicted"

    # 4. unresolved: some support but not enough to be confident
    if support_score > 0:
        return "unresolved"

    # 5. insufficient
    return "insufficient"


def _find_universal_claims(
    all_supporting: dict[str, list[str]],
    candidate_ids: list[str],
) -> set[str]:
    """Identify claim texts that appear for ALL candidates (cannot discriminate)."""
    if len(candidate_ids) <= 1:
        return set()
    per_candidate_sets: list[set[str]] = [
        {c.lower().strip() for c in all_supporting.get(cid, [])}
        for cid in candidate_ids
    ]
    if not per_candidate_sets:
        return set()
    return set.intersection(*per_candidate_sets)


def attribute_evidence(
    board: BoardResults,
    candidates: list[CandidateHypothesis],
) -> list[CandidateEvidence]:
    """Merge board outputs into per-candidate evidence.

    Walks the three role outputs (mechanism, skeptic, trap) and fuses
    them into a :class:`CandidateEvidence` per candidate.

    Args:
        board: Combined output from the three differential-board roles.
        candidates: The candidate hypotheses for this case.

    Returns:
        One :class:`CandidateEvidence` per candidate, ordered to match
        the input ``candidates`` list.
    """
    cid_set = {c.candidate_id for c in candidates}
    cid_to_label = {c.candidate_id: c.candidate_label for c in candidates}

    # Index board outputs by candidate_id for O(1) lookup.
    mech_by_cid = {m.candidate_id: m for m in board.mechanism_outputs if m.candidate_id in cid_set}
    skep_by_cid = {s.candidate_id: s for s in board.skeptic_outputs if s.candidate_id in cid_set}
    trap_by_cid = {t.candidate_id: t for t in board.trap_outputs if t.candidate_id in cid_set}

    # Collect all raw supporting clues per candidate (for universal-claim detection).
    all_raw_supporting: dict[str, list[str]] = {}
    for cid in cid_set:
        mech = mech_by_cid.get(cid)
        all_raw_supporting[cid] = list(mech.supporting_clues) if mech else []

    candidate_ids = [c.candidate_id for c in candidates]
    universal_claims = _find_universal_claims(all_raw_supporting, candidate_ids)

    results: list[CandidateEvidence] = []
    for cand in candidates:
        cid = cand.candidate_id
        mech = mech_by_cid.get(cid)
        skep = skep_by_cid.get(cid)
        trap = trap_by_cid.get(cid)

        supporting: list[Claim] = []
        contradicting: list[Claim] = []
        generic: list[Claim] = []
        missing: list[str] = []

        # --- Mechanism specialist ---
        if mech:
            for clue in mech.supporting_clues:
                if _is_none_response(clue):
                    continue
                claim = _build_claim(
                    clue,
                    source_role="mechanism_specialist",
                    related_candidate_id=cid,
                    is_discriminator=clue.lower().strip() not in universal_claims,
                )
                if clue.lower().strip() in universal_claims:
                    claim = Claim(
                        claim=claim.claim,
                        case_specific=False,
                        source_role=claim.source_role,
                        strength="weak",
                        related_candidate_id=cid,
                    )
                    generic.append(claim)
                else:
                    supporting.append(claim)

            for clue in mech.contradicting_clues:
                if _is_none_response(clue):
                    continue
                claim = _build_claim(
                    clue,
                    source_role="mechanism_specialist",
                    related_candidate_id=cid,
                    is_discriminator=True,
                )
                contradicting.append(claim)

        # --- Option skeptic ---
        if skep:
            missing.extend(c for c in skep.missing_required_clues if not _is_none_response(c))
            for contra in skep.contradictions:
                if _is_none_response(contra):
                    continue
                claim = _build_claim(
                    contra,
                    source_role="option_skeptic",
                    related_candidate_id=cid,
                    is_discriminator=True,
                )
                contradicting.append(claim)

        # --- Trap detector ---
        is_trap = False
        if trap and trap.is_trap:
            is_trap = True
            if trap.exposing_clue:
                contradicting.append(
                    _build_claim(
                        trap.exposing_clue,
                        source_role="trap_detector",
                        related_candidate_id=cid,
                        is_discriminator=True,
                    )
                )

        net_status = _determine_net_status(
            cid, supporting, contradicting, missing, is_trap,
        )

        results.append(
            CandidateEvidence(
                candidate_id=cid,
                candidate_label=cid_to_label.get(cid, ""),
                supporting_claims=supporting,
                contradicting_claims=contradicting,
                missing_required_clues=missing,
                generic_claims=generic,
                net_status=net_status,
            )
        )

    log.info(
        "evidence_attributor.done",
        n_candidates=len(results),
        statuses={r.candidate_id: r.net_status for r in results},
    )
    return results
