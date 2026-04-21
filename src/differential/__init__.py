"""Deterministic LR-weighted differential update engine.

Per Eng_doc.md §5.4 and rules.md §5.1. Pure, sync, no LLM on this path.
Same input → bit-identical output.

Public surface:
    load_lr_table(path) -> LRTable
    rank_branches(active_claims, lr_table) -> BranchRanking
    BranchProjection — per-branch materialised view

See src/differential/README.md for the algorithm and prior-art references.
"""

from __future__ import annotations

from src.differential.engine import BranchRanking, BranchScore, rank_branches
from src.differential.lr_table import LRRow, LRTable, load_lr_table
from src.differential.projection import BranchProjection, project_branches

__all__ = [
    "BranchProjection",
    "BranchRanking",
    "BranchScore",
    "LRRow",
    "LRTable",
    "load_lr_table",
    "project_branches",
    "rank_branches",
]
