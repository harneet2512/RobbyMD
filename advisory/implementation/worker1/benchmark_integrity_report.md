# Benchmark Integrity Report (Worker 1)

Per the orchestrator's CRITICAL CONSTRAINT:

> **Do not allow benchmark runs where full benchmark path == baseline pipeline.**
> **If detected, stop execution and report.**

## Headline: bypass IS detected, today, in both benchmarks

As of `fix/benchmark-integrity` commit, both `--variant full` paths still
fall back to baseline. The bypass is now **loud, not silent**: any attempt to
run `python -m eval.aci_bench --variant full` or
`python -m eval.longmemeval --variant full` will raise `AssertionError` with
the message:

```
Bypass detected: --variant full fell back to baseline.
Wire the real substrate (Worker 3 + shared backend) before running ...
```

This is the safety guard. **Worker 2's substrate-core branch must land first**,
followed by **Worker 3's substrate-backend factor + LME temporal projection**,
before `--variant full` can complete a benchmark run on either side.

## Bypass mechanics

`eval/aci_bench/full.py:36–47` (and the parallel `eval/longmemeval/full.py:35–48`):

1. Creates a `SubstrateStub(session_id=...)` — in-memory no-op.
2. Calls `stub.write_turns(...)` — writes to no-op.
3. Calls `baseline_predict(enc)` — invokes the real baseline.
4. Wraps the baseline's `raw_response` with prefix `"[SUBSTRATE STUB] ..."`.
5. **NEW**: asserts `"[SUBSTRATE STUB]" not in wrapped.raw_response` — fires today.
6. Returns the wrapped prediction (unreachable today due to step 5).

The substrate is **never queried**. The note generator, extractor,
supersession, predicate packs, retrieval head — none execute. Any number
reported from `--variant full` today would be a baseline number wearing a
substrate label.

## What changed in this PR

| Concern | Before | After |
|---------|--------|-------|
| `--variant full` actually runs substrate | ❌ silent bypass to baseline | ❌ bypass still present, but **now loud** |
| `--variant baseline` without API key | ❌ silent gold-leak (returns gold note as prediction) | ✅ `RuntimeError` |
| Judge model | ❌ `gpt-4o-2024-08-06` (retired 2026-03-31, returns 404) | ✅ `gpt-4o-2024-11-20` |
| LongMemEval reader / judge | ❌ same retired snapshot | ✅ same current snapshot |
| Tests defending the above | 0 | 9 new + 5 updated |

## Audit log of every place that could silently degrade an eval run

Searched repo for the failure modes the orchestrator's constraint targets.
All locations are now guarded:

| Failure mode | Location | Status after Worker 1 |
|--------------|----------|----------------------|
| Baseline returns gold without API key | `eval/aci_bench/baseline.py:47` | ✅ raises `RuntimeError` |
| Baseline returns gold without API key | `eval/longmemeval/baseline.py:53` | ✅ raises `RuntimeError` |
| `--variant full` falls back to baseline | `eval/aci_bench/full.py:36–47` | ✅ raises `AssertionError` |
| `--variant full` falls back to baseline | `eval/longmemeval/full.py:35–48` | ✅ raises `AssertionError` |
| Deprecated judge model (404 in production) | `eval/_openai_client.py:_DIRECT_DEFAULTS` | ✅ bumped + locked by test |
| Smoke harness `--variant substrate` falls through | `eval/smoke/run_smoke.py` (the only path that runs real substrate today) | ✅ no degradation; this is the load-bearing path |

## Hand-off list — what Worker 3 must do to lift the bypass

In approximate dependency order:

1. Land Worker 2's `feature/substrate-core` (temporal validity columns,
   already committed; merge into main when ready).
2. Factor `eval/_substrate_backend.py` exposing
   `acibench_substrate_predict(enc) -> ACINotePrediction` and
   `longmemeval_substrate_predict(q) -> LongMemEvalPrediction`. Bodies
   come from `eval/smoke/run_smoke.py::_call_acibench_substrate` and
   `_call_longmemeval_substrate_retrieval_con`.
3. Replace the `SubstrateStub` + `baseline_predict` fallback in
   `eval/aci_bench/full.py:36–47` with `acibench_substrate_predict(enc)`.
   Remove the `"[SUBSTRATE STUB] ..."` prefix from `raw_response`.
4. Same for `eval/longmemeval/full.py:35–48`.
5. Invert the two tests in
   `tests/unit/eval/test_full_variant_bypass_detection.py` — they should
   assert that the assertion **does not** fire.
6. (Optional defense in depth) Keep the bypass-detection assertion: now
   that the sentinel is gone, it passes silently. Cheap insurance.

## Provenance discipline (per cursor.md / advisory/validation/research_report.md)

The judge model bump from `gpt-4o-2024-08-06` → `gpt-4o-2024-11-20` is a
**methodology deviation** from the LongMemEval ICLR 2025 paper baseline.
Per the legitimacy rules in `advisory/validation/sota_and_patterns.md` §7
finding #2 ("vendor numbers and paper numbers are not stackable"), any
LongMemEval number we report must be labelled with the actual judge model
in use, not the paper's pinned model. The inline comment in
`_openai_client.py::_DIRECT_DEFAULTS` documents this; downstream reporting
must do the same.

## Time / scope

- ~30 LOC of production code changes
- ~110 LOC of new tests
- ~20 LOC of test assertion updates
- 380 total tests passing on this branch
- Total branch effort: complete in one main-thread cycle (sub-agent stalled
  in plan mode; orchestrator executed)
