# LongMemEval-S benchmark harness

Long-term memory benchmark for chat assistants (Wu et al., ICLR 2025).

## Dataset

| Field | Value |
|---|---|
| Upstream repo | https://github.com/xiaowu0162/LongMemEval |
| Upstream paper | Wu et al., *LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory*, ICLR 2025 ([arXiv 2410.10813](https://arxiv.org/abs/2410.10813)) |
| License | MIT — redistribution + commercial use |
| Commit SHA pin | **TODO** — pinned to HEAD@2026-04-21 after `fetch.py` runs; cached to `eval/longmemeval/data/.commit_sha` |
| Split used | `longmemeval_s` — all 500 questions |
| Slicing note | Per `memory/feedback_full_benchmarks.md`: we run **all 500 questions**. Hard-category focus (temporal reasoning + multi-session) is a **reporting lens** on the full run, not a subset selection. |

## Sept 2025 cleanup caveat

The LongMemEval authors re-cleaned the dataset in **September 2025**. Numbers
from pre-Sept-2025 evaluations are **not comparable** to the current split.
Our runner logs the exact commit SHA in every report artefact; consumers
verify alignment before overlaying comparators. Zep's self-reported 71.2 vs
independent 63.8 is the known reproducibility wart; see
`docs/research_brief.md §2.6` (blocker 3).

## Methodology

Per LongMemEval ICLR 2025 §4 (official evaluator):

- **Per-category accuracy** across five categories:
  - `information_extraction`
  - `multi_session_reasoning`
  - `temporal_reasoning`
  - `knowledge_update`
  - `abstention`
- LLM judge pinned to `gpt-4o-2024-08-06` via `LONGMEMEVAL_JUDGE_MODEL`.
- Official evaluator script: `<ddxplus-repo>/src/evaluation/evaluate_qa.py`
  (imported at run time; we don't fork or reimplement the evaluator — doing so
  would violate `rules.md §6.1` "no homemade metrics").

See `LIMITATIONS.md` for caveats (memory interface, retrieval quality, judge
variance).

## Adapter

`adapter.py` converts LongMemEval's nested session-history JSON

```json
{
  "question_id": "...",
  "question": "...",
  "answer": "...",
  "question_type": "temporal_reasoning",
  "haystack_sessions": [ [ {"role": "user", "content": "..."}, ... ], ... ]
}
```

into substrate-write calls. Each session's messages become a turn stream (one
turn per message, `user` → `patient`, `assistant` → `physician` mapping, both
kept as `system` if ambiguous). The substrate is then queried via its
retrieval API to assemble the context Opus 4.7 answers the question against.

## Variants

- `baseline.py` — Opus 4.7 with the full `haystack_sessions` concatenated into
  the prompt (~115K tokens on `longmemeval_s`).
- `full.py` — Opus 4.7 with substrate-backed retrieval instead of long
  context. Currently stubbed with `TODO(wt-engine)` pending the substrate
  write + retrieval API.

## Run

```bash
python -m eval.longmemeval.run --variant baseline
python -m eval.longmemeval.run --variant full
python -m eval.longmemeval.run --dry-run      # 5-question smoke
```

Reports to `eval/reports/<timestamp>/longmemeval/`.

## Comparator table

From `docs/research_brief.md §3.2` — `longmemeval_s`, GPT-4o judge. The
spread between self-reported and independent numbers is preserved so the
demo slide can choose its comparator honestly per `rules.md §6.4`.

| System | Overall | Judge | Source |
|---|---:|---|---|
| GPT-4o full-context baseline | 60.2–71.2 | GPT-4o | [Zep blog](https://blog.getzep.com/state-of-the-art-agent-memory/) |
| Zep / Graphiti | 63.8 (indep) / 71.2 (self) | GPT-4o | [arXiv 2501.13956](https://arxiv.org/abs/2501.13956) |
| TiMem | 76.88 | GPT-4o-mini | [arXiv 2601.02845](https://arxiv.org/abs/2601.02845) |
| EverMemOS | 83.0 | — | [arXiv 2601.02163](https://arxiv.org/abs/2601.02163) |
| MemOS | +40% over baseline (per-cat not public) | — | [arXiv 2507.03724](https://arxiv.org/abs/2507.03724) |
| Mastra OM | 94.87 | GPT-5-mini | [Mastra research](https://mastra.ai/research/observational-memory) |
| Supermemory "agent swarm" | ~99 | — | [Supermemory blog](https://blog.supermemory.ai/we-broke-the-frontier-in-agent-memory-introducing-99-sota-memory-system/) |
| MemPalace | 96.6 | — | [mempalace.tech](https://www.mempalace.tech/benchmarks) |
| **Opus 4.7 full-context (us, baseline)** | TBD | `gpt-4o-2024-08-06` | this repo |
| **Opus 4.7 + substrate (us)** | TBD | `gpt-4o-2024-08-06` | this repo |

**Honest defensibility framing** (per `docs/research_brief.md §3.2` reality
check): our claim is NOT overall SOTA. Our claim is **per-category win on TR +
multi-session** — the failure modes where long-context LLMs drop 30–50 points
and where structured-memory systems (Zep +17.3pp on TR, +13.6pp on multi-session)
win biggest. Headline table is per-category; overall is secondary.

## Citation (required per rules.md §7.3)

- Wu et al., *LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory*, ICLR 2025.
