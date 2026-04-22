# LongMemEval-S harness — known limitations

Per `rules.md §6.3` and `§9.3`. Every caveat is reported alongside the numbers
on the demo slide.

## 1. Sept 2025 re-cleanup

LongMemEval authors re-cleaned the dataset on **2025-09-XX**. Pre-Sept-2025
numbers (including some of the comparator numbers in
`docs/research_brief.md §3.2`) are **not directly comparable** to our run.
The runner logs the commit SHA in every `metrics.json`; reviewers verify
alignment before overlaying.

## 2. Self-reported vs independent comparator spread

Zep reports 71.2 on `longmemeval_s`; independent re-runs report 63.8 (per
Zep's own blog). We cite the independent number when overlaying, per
`rules.md §6.4`. If a comparator only has a self-reported number, we mark it
so on the slide.

## 3. Single-session substrate running on multi-session benchmark

LongMemEval-S is multi-session by design; our demo substrate is single-session
(PRD.md §3). Running the substrate on LongMemEval-S is a **stretch test** for
memory primitives — the write API handles multi-session with no ideation, but
cross-session retrieval quality is unproven on this benchmark until the full
eval runs. Per `docs/research_brief.md §7` recommendation: we still run all
500 questions (not a slice) per the `feedback_full_benchmarks.md` rule, and
report honestly.

## 4. Judge pinning

Judge pinned to `gpt-4o-2024-08-06` (env: `LONGMEMEVAL_JUDGE_MODEL`).
LongMemEval's official evaluator accepts any OpenAI chat model; pinning
ensures reproducibility across days/weeks. Different judges (gpt-4o-mini,
gpt-5-mini in Mastra's run) produce different numbers; overlay comparators
must match judge.

## 5. Headline framing: per-category, not overall

Per `docs/research_brief.md §3.2` reality check, we do NOT claim overall SOTA
— systems like Mastra OM (94.87) and Supermemory agent swarm (~99) are ahead.
Our defensible claim is the per-category win on **temporal reasoning +
multi-session** against baseline full-context Opus. Overall is secondary.

## 6. Retrieval quality depends on wt-engine

The `full` variant's retrieval is only as good as wt-engine's
`SubstrateRetriever.top_k`. If retrieval returns irrelevant claims, the full
variant will *underperform* the baseline. This is a correctness property of
wt-engine, not a methodology problem — but it is a failure mode worth
disclosing.
