"""Strict sufficiency auditor — structural analysis of the evidence graph.

Determines whether the accumulated evidence is strong enough to support
a confident answer.  Pure structural checks — NO model calls.
"""
from __future__ import annotations

import structlog

from eval.medxpertqa.d11.types import (
    CandidateEvidence,
    Claim,
    PairwiseDiscriminator,
    SufficiencyAudit,
)

log = structlog.get_logger(__name__)


def _supported_candidates(evidence: list[CandidateEvidence]) -> list[CandidateEvidence]:
    """Return candidates with net_status == 'supported'."""
    return [e for e in evidence if e.net_status == "supported"]


def _rank_candidates(supported: list[CandidateEvidence]) -> tuple[list[str], list[str]]:
    """Rank supported candidates into leader(s) and runner-ups.

    Leader = most supporting claims with fewest contradictions (unique).
    Runner-ups = remaining supported candidates.
    """
    if not supported:
        return [], []

    def _score(ev: CandidateEvidence) -> tuple[int, int]:
        """(supporting_count, -contradicting_count) — higher is better."""
        return (len(ev.supporting_claims), -len(ev.contradicting_claims))

    ranked = sorted(supported, key=_score, reverse=True)

    best_score = _score(ranked[0])
    leaders: list[str] = []
    runners: list[str] = []

    for ev in ranked:
        if _score(ev) == best_score:
            leaders.append(ev.candidate_id)
        else:
            runners.append(ev.candidate_id)

    return leaders, runners


def _find_unresolved_pairs(
    leaders: list[str],
    runners: list[str],
    discriminators: list[PairwiseDiscriminator],
) -> list[tuple[str, str]]:
    """Pairs of (leader, runner-up) without a decisive discriminator."""
    # Index discriminators by pair (both orderings).
    disc_index: dict[tuple[str, str], PairwiseDiscriminator] = {}
    for d in discriminators:
        disc_index[d.pair] = d
        disc_index[(d.pair[1], d.pair[0])] = d

    unresolved: list[tuple[str, str]] = []

    # Check leader-vs-runner pairs
    for leader in leaders:
        for runner in runners:
            pair_key = (leader, runner)
            disc = disc_index.get(pair_key)
            if disc is None:
                unresolved.append(pair_key)
            elif disc.rules_out in ("unclear", "neither"):
                unresolved.append(pair_key)

    # Check leader-vs-leader pairs (co-equal candidates need discrimination too)
    for i, leader_a in enumerate(leaders):
        for leader_b in leaders[i + 1:]:
            pair_key = (leader_a, leader_b)
            disc = disc_index.get(pair_key)
            if disc is None:
                unresolved.append(pair_key)
            elif disc.rules_out in ("unclear", "neither"):
                unresolved.append(pair_key)

    return unresolved


def _find_missing_discriminators(
    leaders: list[str],
    runners: list[str],
    discriminators: list[PairwiseDiscriminator],
) -> list[tuple[str, str]]:
    """Pairs that SHOULD have a discriminator but don't (no entry at all)."""
    existing_pairs: set[frozenset[str]] = set()
    for d in discriminators:
        existing_pairs.add(frozenset(d.pair))

    missing: list[tuple[str, str]] = []
    for leader in leaders:
        for runner in runners:
            if frozenset((leader, runner)) not in existing_pairs:
                missing.append((leader, runner))

    # Leader-vs-leader pairs also need discriminators
    for i, leader_a in enumerate(leaders):
        for leader_b in leaders[i + 1:]:
            if frozenset((leader_a, leader_b)) not in existing_pairs:
                missing.append((leader_a, leader_b))

    return missing


def _collect_generic_claims(evidence: list[CandidateEvidence]) -> list[Claim]:
    """All claims where case_specific is False across candidates."""
    generic: list[Claim] = []
    for ev in evidence:
        generic.extend(ev.generic_claims)
        for claim in ev.supporting_claims:
            if not claim.case_specific:
                generic.append(claim)
    return generic


def _collect_case_specific_decisive_claims(
    evidence: list[CandidateEvidence],
) -> list[Claim]:
    """Claims that are case_specific AND distinguish this candidate from another.

    A claim distinguishes if it appears for only one candidate (it is in
    supporting_claims and case_specific=True and strength in {strong, medium}).
    """
    decisive: list[Claim] = []
    for ev in evidence:
        for claim in ev.supporting_claims:
            if claim.case_specific and claim.strength in ("strong", "medium"):
                decisive.append(claim)
    return decisive


def _find_misleading_claims(
    evidence: list[CandidateEvidence],
    leaders: list[str],
) -> list[Claim]:
    """Strong claims that contradict the leading candidate(s)."""
    misleading: list[Claim] = []
    leader_set = set(leaders)
    for ev in evidence:
        if ev.candidate_id in leader_set:
            for claim in ev.contradicting_claims:
                if claim.strength == "strong":
                    misleading.append(claim)
    return misleading


def _determine_bundle_quality(
    leaders: list[str],
    runners: list[str],
    unresolved_pairs: list[tuple[str, str]],
    misleading_claims: list[Claim],
    generic_count: int,
    decisive_count: int,
    supported: list[CandidateEvidence],
) -> tuple[str, bool]:
    """Compute bundle_quality and repair_required.

    Strong criteria (ALL must hold for 'strong'):
      1. len(leaders) == 1
      2. Every runner-up ruled out by a discriminator
      3. Every (leader, runner-up) pair has a case-specific discriminator
      4. decisive_count > generic_count
      5. len(unresolved_pairs) == 0
      6. No misleading claim contradicts leader with strength 'strong'

    Returns:
        (bundle_quality, repair_required)
    """
    if not supported:
        return "insufficient", True

    criterion_1 = len(leaders) == 1
    criterion_2 = len(unresolved_pairs) == 0  # every runner-up resolved
    # criterion_3 already folded into criterion_2 (unresolved checks pair coverage)
    criterion_4 = generic_count < decisive_count
    criterion_5 = len(unresolved_pairs) == 0
    criterion_6 = len(misleading_claims) == 0

    all_pass = all([criterion_1, criterion_2, criterion_4, criterion_5, criterion_6])

    if all_pass:
        return "strong", False

    # Determine the most descriptive failure mode.
    if generic_count > 0 and decisive_count == 0:
        return "generic", True

    if len(misleading_claims) > 0:
        return "misleading", True

    # Multiple candidates with similar support levels.
    if len(leaders) > 1:
        return "conflicting", True

    # Only 1-2 criteria failed.
    failures = sum(
        1 for c in [criterion_1, criterion_2, criterion_4, criterion_5, criterion_6] if not c
    )
    if failures <= 2:
        return "underdetermined", True

    return "insufficient", True


def audit_sufficiency(
    evidence: list[CandidateEvidence],
    discriminators: list[PairwiseDiscriminator],
    trap_candidates: list[str],
) -> SufficiencyAudit:
    """Structural analysis of the evidence graph.  NO model call.

    Args:
        evidence: Per-candidate evidence from the evidence attributor.
        discriminators: Pairwise discriminators from the board.
        trap_candidates: candidate_ids flagged as traps.

    Returns:
        A :class:`SufficiencyAudit` with quality assessment and repair guidance.
    """
    supported = _supported_candidates(evidence)
    leaders, runners = _rank_candidates(supported)

    unresolved_pairs = _find_unresolved_pairs(leaders, runners, discriminators)
    missing_discs = _find_missing_discriminators(leaders, runners, discriminators)

    generic_claims = _collect_generic_claims(evidence)
    decisive_claims = _collect_case_specific_decisive_claims(evidence)
    misleading_claims = _find_misleading_claims(evidence, leaders)

    generic_count = len(generic_claims)
    decisive_count = len(decisive_claims)

    board_missing = [e for e in evidence if e.net_status == "board_missing"]

    bundle_quality, repair_required = _determine_bundle_quality(
        leaders=leaders,
        runners=runners,
        unresolved_pairs=unresolved_pairs,
        misleading_claims=misleading_claims,
        generic_count=generic_count,
        decisive_count=decisive_count,
        supported=supported,
    )

    # Fallback candidates: board_missing or those with repair failure note
    fallback_ids = {e.candidate_id for e in evidence if e.net_status == "board_missing"}
    for e in evidence:
        for clue in e.missing_required_clues:
            if "board repair failed" in clue.lower():
                fallback_ids.add(e.candidate_id)

    # Block strong if any fallback candidate is in top reasoning
    top_ids = set(leaders + runners)
    for pair in unresolved_pairs:
        top_ids.update(pair)
    fallback_in_top = fallback_ids & top_ids

    if fallback_ids and bundle_quality == "strong":
        if fallback_in_top:
            bundle_quality = "underdetermined"
            repair_required = True
        elif board_missing:
            bundle_quality = "underdetermined"
            repair_required = True

    audit = SufficiencyAudit(
        bundle_quality=bundle_quality,
        leading_candidates=leaders,
        runner_up_candidates=runners,
        unresolved_pairs=unresolved_pairs,
        missing_discriminators=missing_discs,
        misleading_claims=misleading_claims,
        generic_claim_count=generic_count,
        case_specific_decisive_claim_count=decisive_count,
        repair_required=repair_required,
    )

    log.info(
        "sufficiency_audit.done",
        quality=bundle_quality,
        leaders=leaders,
        runners=runners,
        unresolved=len(unresolved_pairs),
        missing_disc=len(missing_discs),
        generic=generic_count,
        decisive=decisive_count,
        repair_required=repair_required,
    )
    return audit
