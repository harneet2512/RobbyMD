# Local LongMemEval smoke — Azure gpt-4o reader+judge

One-command smoke to verify the bge-m3 asymmetric query-prefix fix (`src/substrate/retrieval.py`, step (a) of the fix-order waterfall). Runs on Windows / macOS / Linux without a GPU. The embedding backend goes over Modal; the reader + judge go over Azure OpenAI.

## Why this exists

`src/substrate/retrieval.py` previously called bge-m3 symmetrically (no prefix on either side). bge-m3 is asymmetric: queries expect the instruction prefix `"Represent this sentence for searching relevant passages: "`, documents do not. The symptom on LongMemEval-S was cosine similarity clustering at 0.28–0.40, the CoN reader's Call-1 note-extraction returning empty, and `reader_con.py:251-252` short-circuiting to "I don't know" on most questions.

This smoke measures whether fixing the prefix alone (step (a)) is enough. If yes, step (b) MMR + top-K 40 is not needed. If no, that's the next escalation.

## Prerequisites

### Azure OpenAI env vars (required)

Routing in `eval/_openai_client.py:148-225` switches to Azure when `AZURE_OPENAI_ENDPOINT` is set. The `longmemeval_reader` + `judge_gpt4o` purposes both map to the deployment named in `AZURE_OPENAI_GPT4O_LME_DEPLOYMENT` (falls back to `AZURE_OPENAI_GPT4O_DEPLOYMENT` with a WARN log if unset).

```
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_API_KEY=<your-key>
AZURE_OPENAI_GPT4O_LME_DEPLOYMENT=<your-deployment-name-for-gpt-4o>
AZURE_OPENAI_API_VERSION=2024-10-21   # optional; this is the default
```

Put these in your shell environment or a `.env` file — do not commit them.

### bge-m3 embedding backend (required; one of)

- **Modal (recommended on laptops)**: set `MODAL_BGE_M3_URL` to the deployed Modal endpoint that hosts bge-m3. The embedding client POSTs to `{URL}/embed`.
- **Local `sentence-transformers`**: `pip install sentence-transformers`. Runs on CPU — ~30× slower than Modal and will download ~2.3 GB of model weights on first call. Not recommended for smoke iteration.

The wrapper preflights both paths and fails fast if neither is reachable.

### LongMemEval-S dataset

Already committed: `data/longmemeval_s_cleaned.json` (265 MB). The smoke auto-resolves it via `eval/smoke/run_smoke.py:2031-2043`.

## Run

```
python -m eval.smoke.run_lme_local_smoke
```

Runs `run_smoke.main` with these pinned flags:

```
--benchmark longmemeval
--reader gpt-4o-2024-11-20
--variant both                   # baseline + substrate variants
--n 10 --stratified              # stratified across 6 LME question types
--seed 42
--budget-usd 10                  # hard cap; run halts if exceeded
--output-dir eval/smoke/results/lme_local_<UTC>
```

## Expected runtime + cost

- Runtime: 5–10 min (Azure reader latency dominates; judge ~2s/call; 10 questions × 2 variants = 40 reader calls + 20 judge calls).
- Cost: ≈ $0.30 at `gpt-4o-2024-11-20` Azure pricing for n=10 both-variants. Budget cap of $10 gives 30× safety margin.

## Smoke-pass criteria (Bundle 2 §A.5 applied to step (a))

- ✅ Substrate variant produces actual answers — NOT "I don't know" on 8+/10.
- ✅ ≥ 3/10 substrate answers judged correct.
- ✅ Retrieval cosine similarity > 0.5 on ≥ 5/10 questions (up from the pre-fix 0.28–0.40 band).
- ✅ Baseline variant still works and produces plausible results.
- ⚠ ANOMALY: substrate worse than baseline on 8+/10 (possibly retrieval is now surfacing distractors → consider step (b) MMR).
- ❌ FAIL: substrate still returns "I don't know" on 8+/10 (prefix alone insufficient → escalate to step (b)).

## Decision gate

| Verdict | Action |
|---|---|
| PASS (exit 0) | Step (a) is sufficient. No need to implement step (b) MMR or (c) reranker. |
| ANOMALY (exit 1) | Inspect `hypotheses.jsonl` in the output dir. If the substrate now *finds* claims but the reader picks distractors, escalate to step (b). |
| FAIL (exit 2) | Escalate to step (b): top-K 40 + MMR diversification in `src/substrate/retrieval.py`. Do NOT skip (b) and jump to (c) reranker. |
| PREFLIGHT (exit 3) | Fix env vars and retry. |

## Output artifacts

```
eval/smoke/results/lme_local_<UTC>/
├── hypotheses.jsonl           # one row per question × variant (streamed live)
├── extractions.jsonl          # claim-extractor outputs (for replay / triage)
└── results.json               # aggregate + verdict
```

Inspect `hypotheses.jsonl` to see per-question `answer`, `judged_correct`, `retrieval_confidence` (max cosine), and the retrieved-claims preview.
