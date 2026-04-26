"""Deterministic LR-weighted differential update engine.

Per Eng_doc.md §5.4 and rules.md §5.1. Pure, sync, no LLM, no seed, no temperature.
Same input set → bit-identical output, always. `tests/property/test_determinism.py`
enforces this invariant.

Algorithm:
    for claim in active_claims (sorted deterministically):
        for row in lr_table matching claim.predicate_path (same-branch-level):
            if claim.polarity is present and row.lr_plus is not None:
                branch[row.branch].log_score += log(row.lr_plus)
            elif not claim.polarity and row.lr_minus is not None:
                branch[row.branch].log_score += log(row.lr_minus)
    ranking = softmax(log_scores) sorted by posterior desc, branch_id asc as tiebreak

The softmax is over a closed four-branch set (Cardiac/Pulmonary/MSK/GI — Eng_doc.md
§4.3). Prior is uniform (log 1.0 across branches) — matches PRD.md §6.3 where the
four trees start equal at conversation start. Displayed probabilities sum to 1.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from src.differential.lr_table import LRRow, LRTable
from src.differential.types import ActiveClaim, PhysicianOverride


@dataclass(frozen=True, slots=True)
class AppliedLR:
    """Audit trail for a single LR multiplication — feeds the UI's 'why moved' panel."""

    claim_id: str
    branch: str
    feature: str
    predicate_path: str
    lr_value: float
    log_lr: float
    direction: str  # "lr_plus" (feature present) | "lr_minus" (feature absent)
    approximation: bool


@dataclass(frozen=True, slots=True)
class BranchScore:
    branch: str
    log_score: float
    posterior: float  # softmax-normalised ∈ [0,1]
    applied: tuple[AppliedLR, ...]  # audit trail, deterministically ordered


@dataclass(frozen=True, slots=True)
class BranchRanking:
    """Deterministic ranked list of branches.

    Sort order: posterior descending, with branch-id ascending as the stable
    tiebreak. `.top_n(n)` returns the first n.
    """

    scores: tuple[BranchScore, ...]

    def top_n(self, n: int) -> tuple[BranchScore, ...]:
        return self.scores[:n]

    def by_branch(self, branch: str) -> BranchScore | None:
        for s in self.scores:
            if s.branch == branch:
                return s
        return None


def _applied_key(a: AppliedLR) -> tuple[str, str, str]:
    """Stable sort key for the audit trail within each branch."""
    return (a.claim_id, a.predicate_path, a.feature)


def _apply_row(claim: ActiveClaim, row: LRRow) -> AppliedLR | None:
    """Compute the single-row log-LR contribution for one claim. None if no LR applies."""
    if claim.polarity:
        if row.lr_plus is None:
            return None
        lr = row.lr_plus
        direction = "lr_plus"
    else:
        if row.lr_minus is None:
            return None
        lr = row.lr_minus
        direction = "lr_minus"

    return AppliedLR(
        claim_id=claim.claim_id,
        branch=row.branch,
        feature=row.feature,
        predicate_path=row.predicate_path,
        lr_value=lr,
        log_lr=math.log(lr),
        direction=direction,
        approximation=row.approximation,
    )


def rank_branches(
    active_claims: Iterable[ActiveClaim],
    lr_table: LRTable,
    overrides: Iterable[PhysicianOverride] = (),
) -> BranchRanking:
    """Compute LR-weighted branch ranking with optional physician steering.

    Deterministic: any iterable that yields claims in the same order as another
    yields the same result. We explicitly materialise and sort by `claim_id` to
    neutralise iteration-order noise from upstream callers — required by
    rules.md §5.1.

    Physician overrides enter the same log-odds accumulator as LR evidence.
    An upgrade adds weight to a branch; a downgrade subtracts. Both appear
    in the AppliedLR audit trail so the full reasoning chain — machine AND
    physician — is reconstructible.
    """
    branches = lr_table.branches
    if not branches:
        return BranchRanking(scores=())

    claims = sorted(active_claims, key=lambda c: (c.claim_id, c.predicate_path, c.polarity))

    log_scores: dict[str, float] = dict.fromkeys(branches, 0.0)
    applied: dict[str, list[AppliedLR]] = {b: [] for b in branches}

    for claim in claims:
        for row in lr_table.rows_for(claim.predicate_path):
            hit = _apply_row(claim, row)
            if hit is None:
                continue
            log_scores[hit.branch] += hit.log_lr
            applied[hit.branch].append(hit)

    # Physician overrides: same math, same audit trail.
    for ov in sorted(overrides, key=lambda o: (o.decision_id, o.branch)):
        if ov.branch not in log_scores:
            continue
        sign = 1.0 if ov.direction == "upgrade" else -1.0
        log_delta = sign * ov.weight
        log_scores[ov.branch] += log_delta
        applied[ov.branch].append(AppliedLR(
            claim_id=ov.decision_id,
            branch=ov.branch,
            feature=f"physician_{ov.direction}",
            predicate_path="physician_override",
            lr_value=math.exp(log_delta),
            log_lr=log_delta,
            direction=ov.direction,
            approximation=False,
        ))

    max_log = max(log_scores.values())
    unnorm = {b: math.exp(log_scores[b] - max_log) for b in log_scores}
    z = sum(unnorm.values())
    posteriors = {b: unnorm[b] / z for b in unnorm}

    scored = tuple(
        BranchScore(
            branch=b,
            log_score=log_scores[b],
            posterior=posteriors[b],
            applied=tuple(sorted(applied[b], key=_applied_key)),
        )
        for b in sorted(log_scores.keys())
    )
    ranked = tuple(sorted(scored, key=lambda s: (-s.posterior, s.branch)))
    return BranchRanking(scores=ranked)
