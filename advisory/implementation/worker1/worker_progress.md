# Worker 1 — Benchmark Safety + Evaluation Integrity (fix/benchmark-integrity)

**Branch**: `fix/benchmark-integrity`
**Status**: complete, 380 passed / 2 skipped / 0 failed.
**Executed by**: orchestrator main thread (sub-agent stalled in plan mode).

## Files changed

| File | Change |
|------|--------|
| `eval/_openai_client.py` | `_DIRECT_DEFAULTS`: bumped `judge_gpt4o`, `longmemeval_reader`, `longmemeval_judge` from `gpt-4o-2024-08-06` (retired 2026-03-31) → `gpt-4o-2024-11-20`. Added inline citation comment. |
| `eval/aci_bench/baseline.py:47–52` | Replaced silent gold-leak fallback with `RuntimeError`. Message explains the inflation risk. |
| `eval/longmemeval/baseline.py:53–59` | Same. |
| `eval/aci_bench/full.py` | Wrapped the constructed `ACINotePrediction` in a local `wrapped` variable; added `assert "[SUBSTRATE STUB]" not in wrapped.raw_response` before return. Fires today (bypass is real); will pass silently once Worker 3 wires the real substrate. |
| `eval/longmemeval/full.py` | Same pattern. |
| `tests/unit/eval/test_openai_client.py:23` | Updated existing assertion to `gpt-4o-2024-11-20`. |
| `tests/unit/eval/test_lme_client_fallback.py:28,34,131,132` | Updated three existing assertions to `gpt-4o-2024-11-20`. |
| `tests/unit/eval/test_baseline_no_key.py` | NEW. Four tests: ACI raises RuntimeError, LME raises RuntimeError, both messages explain why. |
| `tests/unit/eval/test_judge_model_current.py` | NEW. Three tests: `judge_gpt4o`, `longmemeval_reader`, `longmemeval_judge` all resolve to `gpt-4o-2024-11-20`, never the deprecated snapshot. |
| `tests/unit/eval/test_full_variant_bypass_detection.py` | NEW. Two tests: ACI and LME `--variant full` both raise `AssertionError` with `"Bypass detected"` today. Will need to be inverted when Worker 3 wires the real substrate. |

## Resolution of Worker 1 sub-agent's three flagged questions

1. **Commit artifacts with code?** — Yes. Single commit on `fix/benchmark-integrity` carries both code changes and the four `advisory/implementation/worker1/` artifacts.
2. **Existing test asserts deprecated model?** — Updated. Two test files (`test_openai_client.py`, `test_lme_client_fallback.py`) had four assertions to the old model id; all four updated.
3. **Bypass-assertion placement?** — Sub-agent's interpretation was correct. Built the `wrapped` prediction object first so the `[SUBSTRATE STUB]` sentinel is present in `wrapped.raw_response`, then asserted, then returned. Today the assertion fires (bypass is real); after Worker 3 the wrapper no longer carries the sentinel and the assertion passes silently.

## Override of sub-agent's conservatism on `longmemeval_*` purposes

Sub-agent flagged that `longmemeval_reader` and `longmemeval_judge` were pinned to `gpt-4o-2024-08-06` for paper-faithfulness with LongMemEval ICLR 2025 (Wu et al.). Override applied: bump those too. Rationale documented in `implementation_notes.md` (the model is *retired* — calls return 404 — so paper-faithfulness with the original snapshot is no longer mechanically achievable).

## Test results

```
tests/unit/eval/                                            131 passed
tests/unit/                                                 ~250 passed
tests/property/                                             9 passed (3 determinism + 6 pre-existing property)
tests/unit/substrate/                                       82 passed (no temporal-validity columns on this branch)
                                                            ─────────────
                                                            380 passed, 2 skipped, 0 failed in 2.60s
```

The 2 skipped are pre-existing skips not caused by Worker 1.

## Benchmark-integrity report (`benchmark_integrity_report.md`)

See sibling file. Headline: **today both ACI-Bench `--variant full` and LongMemEval `--variant full` are baseline-with-stub-label**. The new bypass-detection assertion makes that fact loud. Worker 3 (`feature/lme-temporal`) and the shared substrate backend factor will close the bypass.

## Commit

To follow this artifact write — see `git log -1 fix/benchmark-integrity`.
