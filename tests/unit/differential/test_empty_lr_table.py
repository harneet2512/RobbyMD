"""Empty-LR-table regression test per 2026-04-21 refactor.

`personal_assistant` pack has no differential hypotheses; its `active_pack()`
carries no `lr_table_path`. The differential engine must no-op gracefully on
an empty `LRTable` rather than crash.

Addresses audit finding #1 from commit `767d3e8` (hardcoded `BRANCHES`
frozenset replaced by per-table derived branches).
"""

from __future__ import annotations

from src.differential.engine import rank_branches
from src.differential.lr_table import LRTable
from src.differential.types import ActiveClaim


def test_empty_lr_table_has_empty_branches() -> None:
    table = LRTable.empty(chief_complaint="none")
    assert table.branches == frozenset()
    assert table.rows == ()
    assert table.chief_complaint == "none"


def test_rank_branches_on_empty_table_returns_empty_ranking() -> None:
    """Engine no-ops on an empty LR table — required for non-clinical packs."""
    table = LRTable.empty()
    ranking = rank_branches([], table)
    assert ranking.scores == ()
    assert ranking.top_n(5) == ()


def test_rank_branches_on_empty_table_ignores_claims() -> None:
    """Even with active claims, no branches → empty ranking; no crash."""
    table = LRTable.empty()
    claims = [
        ActiveClaim(
            claim_id="c1",
            predicate_path="onset=sudden",
            polarity=True,
            confidence=0.9,
            source_turn_id="t1",
        ),
        ActiveClaim(
            claim_id="c2",
            predicate_path="character=sharp",
            polarity=True,
            confidence=0.9,
            source_turn_id="t2",
        ),
    ]
    ranking = rank_branches(claims, table)
    assert ranking.scores == ()
