"""Determinism property test per rules.md §5.1.

The differential update engine is pure: given the same active claim set and the
same LR table, the output ranking is bit-identical on every call. No temperature,
no seed, no ML re-ranking on this path.

Runs the engine 100x on the canonical chest-pain mid-case fixture and asserts
identical output every time. Blocks merge if it ever fails (rules.md §5.1,
Eng_doc.md §5.4 + §11.3).
"""

from __future__ import annotations

import random

from src.differential.engine import rank_branches
from src.differential.lr_table import load_lr_table
from tests.fixtures.loader import LR_TABLE_PATH, load_mid_case_claims

_N = 100


def test_differential_engine_is_deterministic() -> None:
    """100 consecutive calls produce bit-identical BranchRanking."""
    table = load_lr_table(LR_TABLE_PATH)
    claims = load_mid_case_claims()
    first = rank_branches(claims, table)
    for _ in range(_N - 1):
        nxt = rank_branches(claims, table)
        assert nxt == first, "differential engine violated rules.md §5.1 determinism"


def test_engine_order_independent_across_permutations() -> None:
    """Determinism survives caller-side re-ordering of the active-claim iterable."""
    table = load_lr_table(LR_TABLE_PATH)
    claims = load_mid_case_claims()
    first = rank_branches(claims, table)
    rng = random.Random(20260421)
    for _ in range(_N - 1):
        shuffled = claims.copy()
        rng.shuffle(shuffled)
        nxt = rank_branches(shuffled, table)
        assert nxt == first, "differential engine output depends on iteration order"


def test_engine_deterministic_on_empty_input() -> None:
    table = load_lr_table(LR_TABLE_PATH)
    first = rank_branches([], table)
    for _ in range(_N - 1):
        assert rank_branches([], table) == first
