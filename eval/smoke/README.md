# eval/smoke — Smoke-run harness

Deterministic first-10-case sanity pass before any full benchmark run. Enforces:

- Hard `--budget-usd` cap (early halt with `BUDGET HALT` summary).
- Published-baseline sanity check (±20pp vs `reference_baselines.json`).
- Substrate-delta non-zero sanity (catches the "substrate did nothing" failure mode).
- `--dry-run` mode that parses args + imports adapters + reports planned matrix without any API call or cost.

## Typical flow

```bash
# One-shot dataset fetch (skips if already cloned):
./eval/smoke/prepare_datasets.sh

# Dry-run sanity (zero cost, no API calls, no GPU):
python eval/smoke/run_smoke.py --dry-run \
    --benchmark both \
    --reader qwen2.5-14b \
    --variant both \
    --n 10

# Real smoke, budget-capped, with reader hosted on GCP via eval/infra/deploy_qwen_gcp.sh
# (set QWEN_API_BASE=http://127.0.0.1:8000/v1 after starting the tunnel):
python eval/smoke/run_smoke.py \
    --benchmark both \
    --reader qwen2.5-14b \
    --variant both \
    --n 10 \
    --budget-usd 50
```

## Outputs

For each real run: `eval/smoke/<benchmark>/<timestamp>/{results.json, methodology.md}`.

- `results.json`: per-case breakdown (latency, tokens, estimated cost, raw output, judge-scored correctness, structural-validity checks) + aggregate.
- `methodology.md`: case-selection method (deterministic: first N by sorted case_id), reader version, prompt version, variant definition, budget actual vs estimate.

## Verdicts

ASCII-only markers so Windows cmd / PowerShell (cp1252 default) renders cleanly.

- `[OK] PASS`: harness completes, substrate produces valid outputs, baseline within ±20pp of `reference_baselines.json`, non-zero substrate delta, cost ≤ 1.5× estimate.
- `[WARN] ANOMALY`: harness completes but one criterion fails — details surfaced.
- `[FAIL] FAIL`: harness crashes, substrate produces empty/invalid outputs, or cost blows past hard cap.

## Status

**Built, not run.** Smoke runs happen in a separate invocation after the user reviews the harness. Per `reasons.md` → "Smoke-first benchmark discipline (2026-04-21)": never run a full benchmark before the smoke verdict is ✅ PASS or ⚠ ANOMALY with a documented override.
