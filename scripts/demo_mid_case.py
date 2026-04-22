"""Smoke-demo: mid-case chest-pain ranking + next-best-question.

Run with: python -m scripts.demo_mid_case  (from the wt-trees root)

Prints the BranchRanking produced by src/differential on the hand-crafted fixture,
the top-2 branches, and the verifier's deterministically-selected next-best
question (rendered by MockOpusClient offline).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.differential import load_lr_table, rank_branches  # noqa: E402
from src.verifier import MockOpusClient, verify  # noqa: E402
from tests.fixtures.loader import LR_TABLE_PATH, load_mid_case_claims  # noqa: E402


def main() -> None:
    table = load_lr_table(LR_TABLE_PATH)
    claims = load_mid_case_claims()
    ranking = rank_branches(claims, table)

    print("=" * 66)
    print("Active claims (synthetic chest-pain mid-case):")
    for c in claims:
        sign = "+" if c.polarity else "-"
        print(f"  [{sign}] {c.claim_id}  {c.predicate_path}")

    print("\nRanked branches (posterior desc):")
    for s in ranking.scores:
        print(
            f"  {s.branch:10s}  posterior={s.posterior:.4f}  "
            f"log_score={s.log_score:+.3f}  (|applied|={len(s.applied)})"
        )

    out = verify(ranking, table, claims, opus_client=MockOpusClient())
    top1, top2 = ranking.scores[0], ranking.scores[1]
    print(f"\nTop-2:  {top1.branch} (p={top1.posterior:.3f}) vs {top2.branch} (p={top2.posterior:.3f})")
    print(f"Why moved        : {list(out.why_moved)}")
    print(f"Missing/absent   : {list(out.missing_or_contradicting)}")
    print(f"Source feature   : {out.source_feature}")
    print(f"Rationale        : {out.next_question_rationale}")
    print(f"Next-best question:\n  \"{out.next_best_question}\"")
    print("=" * 66)


if __name__ == "__main__":
    main()
