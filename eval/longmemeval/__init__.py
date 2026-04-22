"""LongMemEval-S benchmark harness.

Upstream: https://github.com/xiaowu0162/LongMemEval
Methodology: ICLR 2025 — all 500 questions, per-category accuracy via the
official evaluator. Judge pinned to LONGMEMEVAL_JUDGE_MODEL
(default gpt-4o-2024-08-06).

See eval/longmemeval/README.md for dataset pin, license, and comparator table.
"""
