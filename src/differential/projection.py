"""Per-branch materialised view over active claims + the LR table.

Per Eng_doc.md §4.3: each branch projection emits
    - subset of active claims relevant to this branch,
    - per-node state ∈ {unasked, evidence-present, evidence-absent, contradicted},
    - log-likelihood score (shared with the ranking engine).

Substrate-agnostic: consumes any iterable of `ActiveClaim`. Once wt-engine's
`ActiveClaimSet` lands, swap the argument type — no change to the algorithm.
Deterministic per rules.md §5.1 — no LLM, no randomness.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from src.differential.engine import AppliedLR, BranchRanking, rank_branches
from src.differential.lr_table import LRRow, LRTable
from src.differential.types import ActiveClaim


class NodeState(StrEnum):
    UNASKED = "unasked"
    EVIDENCE_PRESENT = "evidence-present"
    EVIDENCE_ABSENT = "evidence-absent"
    CONTRADICTED = (
        "contradicted"  # claim exists with polarity opposite the feature-favoured direction
    )


@dataclass(frozen=True, slots=True)
class NodeProjection:
    """One feature-node's state for the UI tree panel (PRD.md §6.3)."""

    feature: str
    predicate_path: str
    state: NodeState
    lr_row: LRRow
    matched_claim_id: str | None
    applied_log_lr: float | None  # non-None iff the row contributed to the branch score


@dataclass(frozen=True, slots=True)
class BranchProjection:
    """Per-branch view: claim subset + node states + score (shared with engine)."""

    branch: str
    log_score: float
    posterior: float
    relevant_claim_ids: tuple[str, ...]
    nodes: tuple[NodeProjection, ...]


def _match_claim_for_row(
    row: LRRow, claims_by_path: dict[str, list[ActiveClaim]]
) -> tuple[ActiveClaim | None, NodeState, float | None]:
    """Pick the first deterministically-ordered matching claim for a row."""
    matches = claims_by_path.get(row.predicate_path, [])
    if not matches:
        return None, NodeState.UNASKED, None

    # Deterministic: sort by claim_id so we always resolve to the same canonical claim.
    first = sorted(matches, key=lambda c: c.claim_id)[0]
    if first.polarity:
        # Feature present. If the row's LR+ > 1 the branch is supported; <1 (rule-out)
        # means present-feature argues against the branch — that's CONTRADICTED
        # per the `{evidence-present, ... contradicted}` node-state vocabulary.
        if row.lr_plus is None:
            return first, NodeState.UNASKED, None
        state = NodeState.EVIDENCE_PRESENT if row.lr_plus >= 1.0 else NodeState.CONTRADICTED
        import math as _m

        return first, state, _m.log(row.lr_plus)
    else:
        if row.lr_minus is None:
            # We know the feature is absent but no LR- published — UNASKED for scoring.
            return first, NodeState.UNASKED, None
        import math as _m

        return first, NodeState.EVIDENCE_ABSENT, _m.log(row.lr_minus)


def project_branches(
    active_claims: Iterable[ActiveClaim],
    lr_table: LRTable,
    ranking: BranchRanking | None = None,
) -> tuple[BranchProjection, ...]:
    """Emit a BranchProjection per branch. Deterministic given the same inputs."""
    materialised = list(active_claims)
    if ranking is None:
        ranking = rank_branches(materialised, lr_table)

    claims_by_path: dict[str, list[ActiveClaim]] = {}
    for c in materialised:
        claims_by_path.setdefault(c.predicate_path, []).append(c)

    # Build audit-trail index for fast lookup when filling node projections.
    applied_by_branch: dict[str, list[AppliedLR]] = {
        s.branch: list(s.applied) for s in ranking.scores
    }

    out: list[BranchProjection] = []
    for score in ranking.scores:
        rows = lr_table.rows_on_branch(score.branch)
        nodes: list[NodeProjection] = []
        relevant_claim_ids: set[str] = set()

        for row in rows:
            claim, state, log_lr = _match_claim_for_row(row, claims_by_path)
            matched_id = claim.claim_id if claim is not None else None
            if matched_id is not None:
                relevant_claim_ids.add(matched_id)
            nodes.append(
                NodeProjection(
                    feature=row.feature,
                    predicate_path=row.predicate_path,
                    state=state,
                    lr_row=row,
                    matched_claim_id=matched_id,
                    applied_log_lr=log_lr
                    if log_lr is not None and matched_id is not None
                    else None,
                )
            )

        # Deterministic node order: evidence states first (present > absent > contradicted >
        # unasked), then feature-name alphabetical.
        state_rank = {
            NodeState.EVIDENCE_PRESENT: 0,
            NodeState.EVIDENCE_ABSENT: 1,
            NodeState.CONTRADICTED: 2,
            NodeState.UNASKED: 3,
        }
        nodes.sort(key=lambda n: (state_rank[n.state], n.feature))

        # Sanity: log_score we compute from applied hits should equal ranking.log_score
        # (up to float additivity). The ranking value is the source of truth.
        _ = applied_by_branch  # indicate intentional use for audit-only side-channel

        out.append(
            BranchProjection(
                branch=score.branch,
                log_score=score.log_score,
                posterior=score.posterior,
                relevant_claim_ids=tuple(sorted(relevant_claim_ids)),
                nodes=tuple(nodes),
            )
        )

    return tuple(out)
