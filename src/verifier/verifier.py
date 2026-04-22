"""Counterfactual verifier — deterministic discriminator selection + LLM phrasing.

Per Eng_doc.md §6. On every top-2 ranking update:
    1. For each of the top-2 branches, compute support (features present with LR+>1)
       and refutation (features ABSENT with LR+>1.5).
    2. Pick the single discriminator feature maximising
           |log LR_A(f) - log LR_B(f)| * uncertainty
       across the union of the two branches' refutation sets. The "uncertainty"
       multiplier keeps the verifier quiet when the top-2 gap is already large
       (the ranking is confident — no question needed).
    3. Call Opus 4.7 once to phrase that discriminator as a ≤20-word clinical
       question. The selection itself is deterministic; only the phrasing is LLM.

Offline fallback: `MockOpusClient` templates a canned question when the
`ANTHROPIC_API_KEY` env var is unset. Lets unit + property tests run without network.
"""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Protocol

from src.differential.engine import BranchRanking
from src.differential.lr_table import LRRow, LRTable
from src.differential.types import ActiveClaim
from src.verifier.prompts import SYSTEM_PROMPT, build_user_prompt

logger = logging.getLogger(__name__)

MODEL_ID = "claude-opus-4-7"  # CLAUDE.md §9; rules.md §5.3


@dataclass(frozen=True, slots=True)
class VerifierOutput:
    """Shape matches Eng_doc.md §6 — this is the payload the UI aux strip consumes."""

    why_moved: tuple[str, ...]
    missing_or_contradicting: tuple[str, ...]
    next_best_question: str
    next_question_rationale: str
    source_feature: str


@dataclass(frozen=True, slots=True)
class Discriminator:
    """Deterministically-selected feature that best distinguishes top-2 branches."""

    feature: str
    predicate_path: str
    branch_a: str  # top-1 branch
    branch_b: str  # top-2 branch
    lr_a: float | None
    lr_b: float | None
    log_gap: float  # |log LR_A - log LR_B|, with missing LR treated as 1.0 (log 0)
    direction: str  # "+" (feature presence argues for one branch) | "-" (absence argues)
    uncertainty: float  # 1 - (posterior_a - posterior_b); close-to-0 when the gap is wide
    score: float  # log_gap * uncertainty — the argmax target
    row_a: LRRow | None = field(default=None)
    row_b: LRRow | None = field(default=None)


class OpusClient(Protocol):
    """Minimal protocol the verifier calls to render the question."""

    def render_next_best_question(
        self,
        branch_a: str,
        branch_b: str,
        feature: str,
        predicate_path: str,
        lr_a: float | None,
        lr_b: float | None,
        direction: str,
    ) -> str: ...


class MockOpusClient:
    """Offline fallback. Deterministic: same discriminator → same question.

    Used when `ANTHROPIC_API_KEY` is not set so the test suite, CI, and
    demo-rehearsal offline mode stay green. Real calls go through
    `AnthropicOpusClient` below.
    """

    _TEMPLATE = (
        "Can you clarify whether {feature_phrase} is present - this would help "
        "distinguish {branch_a} from {branch_b}?"
    )

    def render_next_best_question(
        self,
        branch_a: str,
        branch_b: str,
        feature: str,
        predicate_path: str,
        lr_a: float | None,
        lr_b: float | None,
        direction: str,
    ) -> str:
        feature_phrase = feature.replace("_", " ")
        text = self._TEMPLATE.format(
            feature_phrase=feature_phrase, branch_a=branch_a, branch_b=branch_b
        )
        # Enforce the ≤20-word bound defensively.
        words = text.split()
        if len(words) > 20:
            text = " ".join(words[:19]).rstrip(",.;:-") + "?"
        return text


class AnthropicOpusClient:
    """Thin wrapper over the Anthropic Messages API. No side effects on import."""

    def __init__(self, api_key: str | None = None) -> None:
        # Imported lazily so the module can load in offline CI without the SDK.
        from anthropic import Anthropic

        self._client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    def render_next_best_question(
        self,
        branch_a: str,
        branch_b: str,
        feature: str,
        predicate_path: str,
        lr_a: float | None,
        lr_b: float | None,
        direction: str,
    ) -> str:
        user_prompt = build_user_prompt(
            branch_a=branch_a,
            branch_b=branch_b,
            feature=feature,
            predicate_path=predicate_path,
            lr_a=lr_a,
            lr_b=lr_b,
            direction=direction,
        )
        msg = self._client.messages.create(
            model=MODEL_ID,
            max_tokens=64,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        # The SDK returns a list of content blocks; the first is text.
        text_blocks = [blk.text for blk in msg.content if getattr(blk, "type", "") == "text"]
        raw = text_blocks[0].strip() if text_blocks else ""
        # Enforce ≤20-word bound server-side (rules.md §5.3 — LLM output is bounded).
        words = raw.split()
        if len(words) > 20:
            raw = " ".join(words[:19]).rstrip(",.;:-") + "?"
        return raw


def _default_client() -> OpusClient:
    """Pick the right client based on ANTHROPIC_API_KEY presence. No-op if unset."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return AnthropicOpusClient()
        except Exception as e:  # pragma: no cover — SDK import / construction failures
            logger.warning("AnthropicOpusClient unavailable (%s); falling back to mock.", e)
    return MockOpusClient()


def _present_claim_paths(claims: Iterable[ActiveClaim]) -> set[str]:
    return {c.predicate_path for c in claims if c.polarity}


def _absent_claim_paths(claims: Iterable[ActiveClaim]) -> set[str]:
    return {c.predicate_path for c in claims if not c.polarity}


def _safe_log(lr: float | None) -> float:
    return math.log(lr) if lr is not None and lr > 0 else 0.0


def _branch_rows_by_feature(lr_table: LRTable, branch: str) -> dict[str, LRRow]:
    """Last-wins on duplicate feature names per branch (there aren't any today)."""
    return {r.feature: r for r in lr_table.rows_on_branch(branch)}


def _row_for_path(lr_table: LRTable, branch: str, predicate_path: str) -> LRRow | None:
    """Return a row on `branch` whose predicate_path matches, or None if the branch
    does not score on that feature (i.e. the feature is branch-silent — neutral)."""
    for r in lr_table.rows_on_branch(branch):
        if r.predicate_path == predicate_path:
            return r
    return None


def select_discriminator(
    ranking: BranchRanking,
    lr_table: LRTable,
    active_claims: Iterable[ActiveClaim],
) -> Discriminator | None:
    """Deterministically pick the single best discriminator for the top-2 branches.

    Returns None when there is no useful discriminator (e.g. no LR mismatch; or the
    top-2 gap is already so wide the verifier would be noise).
    """
    top2 = ranking.top_n(2)
    if len(top2) < 2:
        return None
    branch_a, branch_b = top2[0].branch, top2[1].branch
    posterior_gap = top2[0].posterior - top2[1].posterior
    # Uncertainty: 1 when tied, 0 when fully confident in top-1.
    uncertainty = max(0.0, 1.0 - posterior_gap)
    if uncertainty <= 0.0:
        return None

    claims_list = list(active_claims)
    present_paths = _present_claim_paths(claims_list)
    absent_paths = _absent_claim_paths(claims_list)

    # Eng_doc.md §6 algorithm: discriminator = argmax over refutation(A) union refutation(B).
    # Refutation set = features that would argue FOR branch X but which the patient has
    # NOT yet volunteered (predicate_path not in present_paths). LR+ > 1.5 gate keeps
    # weak rows out.
    candidate_rows: dict[str, LRRow] = {}  # feature → row on one branch
    for branch in (branch_a, branch_b):
        for row in lr_table.rows_on_branch(branch):
            if (
                row.lr_plus is not None
                and row.lr_plus > 1.5
                and row.predicate_path not in present_paths
            ):
                # Don't propose a question we've already answered negatively.
                if row.predicate_path in absent_paths:
                    continue
                # Prefer rows with larger LR+ when the same feature surfaces on both branches.
                prev = candidate_rows.get(row.feature)
                if prev is None or (row.lr_plus or 0) > (prev.lr_plus or 0):
                    candidate_rows[row.feature] = row

    best: Discriminator | None = None
    for feature, row in candidate_rows.items():
        row_a = _row_for_path(lr_table, branch_a, row.predicate_path)
        row_b = _row_for_path(lr_table, branch_b, row.predicate_path)
        lr_a = row_a.lr_plus if row_a is not None else None
        lr_b = row_b.lr_plus if row_b is not None else None
        log_gap = abs(_safe_log(lr_a) - _safe_log(lr_b))
        if log_gap <= 0.0:
            continue
        score = log_gap * uncertainty
        # Direction indicator: "+" marks that the feature's presence argues FOR the
        # branch with the larger LR+. Absence-argues cases fold into log_gap naturally.
        direction = "+"
        candidate = Discriminator(
            feature=feature,
            predicate_path=row.predicate_path,
            branch_a=branch_a,
            branch_b=branch_b,
            lr_a=lr_a,
            lr_b=lr_b,
            log_gap=log_gap,
            direction=direction,
            uncertainty=uncertainty,
            score=score,
            row_a=row_a,
            row_b=row_b,
        )
        # Deterministic tiebreak: higher score first; on a tie, feature name asc.
        if (
            best is None
            or candidate.score > best.score
            or (candidate.score == best.score and candidate.feature < best.feature)
        ):
            best = candidate

    return best


def _why_moved_bullets(ranking: BranchRanking, limit: int = 2) -> tuple[str, ...]:
    """Top features that *raised* the log-score of the top-1 branch — deterministic."""
    if not ranking.scores:
        return ()
    top = ranking.scores[0]
    hits = [a for a in top.applied if a.log_lr > 0]
    hits.sort(key=lambda a: (-a.log_lr, a.feature))
    return tuple(
        f"{a.feature} ({'LR+' if a.direction == 'lr_plus' else 'LR-'} {a.lr_value:g})"
        for a in hits[:limit]
    )


def _missing_bullets(disc: Discriminator | None, limit: int = 2) -> tuple[str, ...]:
    """Features whose absence is holding the top-1 branch back (or leaving it tied)."""
    if disc is None:
        return ()
    # One bullet from the selected discriminator; a stub line for the second slot keeps
    # the UI layout stable even when only one candidate survives the gate.
    parts = [f"{disc.feature} (LR+ ~{(disc.lr_a or disc.lr_b or 1.0):g}, not yet volunteered)"]
    return tuple(parts[:limit])


def verify(
    ranking: BranchRanking,
    lr_table: LRTable,
    active_claims: Iterable[ActiveClaim],
    opus_client: OpusClient | None = None,
) -> VerifierOutput:
    """Build a `VerifierOutput` for the UI aux strip.

    Deterministic except for the Opus call that renders `next_best_question`
    (and even that is stubbed via `MockOpusClient` when offline).
    """
    claims_list = list(active_claims)
    disc = select_discriminator(ranking, lr_table, claims_list)
    client = opus_client if opus_client is not None else _default_client()

    if disc is None:
        # No useful question — verifier stays quiet. UI shows a neutral placeholder.
        return VerifierOutput(
            why_moved=_why_moved_bullets(ranking),
            missing_or_contradicting=(),
            next_best_question="",
            next_question_rationale="",
            source_feature="",
        )

    question = client.render_next_best_question(
        branch_a=disc.branch_a,
        branch_b=disc.branch_b,
        feature=disc.feature,
        predicate_path=disc.predicate_path,
        lr_a=disc.lr_a,
        lr_b=disc.lr_b,
        direction=disc.direction,
    )

    rationale = (
        f"log-LR gap {disc.log_gap:.2f} between {disc.branch_a} and {disc.branch_b} "
        f"on {disc.feature}"
    )

    return VerifierOutput(
        why_moved=_why_moved_bullets(ranking),
        missing_or_contradicting=_missing_bullets(disc),
        next_best_question=question,
        next_question_rationale=rationale,
        source_feature=disc.feature,
    )
