"""Determinism property test per rules.md §5.1.

The differential update engine is pure: given the same active claim set and the same
LR table, the output ranking is bit-identical on every call. No temperature, no seed,
no ML re-ranking on this path.

Runs the engine 100× on the same input and asserts identical output. Blocks merge
if it ever fails.

NOTE: substrate + differential engine not yet implemented. Marked xfail until
src/differential/engine.py lands in the wt-trees worktree.
"""
from __future__ import annotations

import pytest


@pytest.mark.xfail(
    reason="Substrate + differential engine not yet implemented (wt-trees scope).",
    strict=True,
)
def test_differential_engine_is_deterministic() -> None:
    # Will be replaced once src/differential/engine.py + src/substrate/claims.py land:
    #
    # from src.differential.engine import rank_branches
    # from src.substrate.claims import ActiveClaimSet
    # from src.differential.lr_table import load_lr_table
    #
    # claims = ActiveClaimSet.from_fixture("tests/fixtures/chest_pain_mid_case.json")
    # lrs = load_lr_table("content/differentials/chest_pain/lr_table.json")
    # first = rank_branches(claims, lrs)
    # for _ in range(99):
    #     assert rank_branches(claims, lrs) == first
    raise NotImplementedError("Awaiting src/differential/engine.py")
