# Worker 1 — Test Output

## Eval unit tests (focused)

```
============================= test session starts =============================
platform win32 -- Python 3.12.0, pytest-9.0.2
collected 131 items

tests\unit\eval\test_acibench_extractors.py ...........                  [  8%]
tests\unit\eval\test_acibench_hybrid.py ...............                  [ 19%]
tests\unit\eval\test_acibench_judge_call.py ...                          [ 22%]
tests\unit\eval\test_baseline_no_key.py ....                             [ 25%]   <-- NEW
tests\unit\eval\test_full_variant_bypass_detection.py ..                 [ 26%]   <-- NEW
tests\unit\eval\test_judge_model_current.py ...                          [ 29%]   <-- NEW
tests\unit\eval\test_lme_adapter.py ........                             [ 35%]
tests\unit\eval\test_lme_client_fallback.py .........                    [ 41%]   <-- 4 lines updated
tests\unit\eval\test_lme_substrate_runner.py .........                   [ 48%]
tests\unit\eval\test_openai_client.py ..........                         [ 56%]   <-- 1 line updated
tests\unit\eval\test_reader_con.py .......                               [ 61%]
tests\unit\eval\test_smoke_harness_dryrun.py ...                         [ 64%]
tests\unit\eval\test_smoke_output_dir_and_stratified.py .........        [ 71%]
tests\unit\eval\test_smoke_realrun_wiring.py ....................        [ 86%]
tests\unit\eval\test_smoke_seed_flag.py .......                          [ 91%]
tests\unit\eval\test_smoke_streaming.py ..........                       [100%]

============================= 131 passed in 1.25s =============================
```

## Full unit + property suites (regression check)

```
======================= 380 passed, 2 skipped in 2.60s ========================
```

## Tests added by Worker 1

- `test_baseline_no_key.py::test_aci_baseline_raises_runtime_error_without_anthropic_key`
- `test_baseline_no_key.py::test_lme_baseline_raises_runtime_error_without_anthropic_key`
- `test_baseline_no_key.py::test_aci_baseline_message_explains_why`
- `test_baseline_no_key.py::test_lme_baseline_message_explains_why`
- `test_judge_model_current.py::test_judge_gpt4o_is_not_deprecated`
- `test_judge_model_current.py::test_longmemeval_reader_is_not_deprecated`
- `test_judge_model_current.py::test_longmemeval_judge_is_not_deprecated`
- `test_full_variant_bypass_detection.py::test_aci_bench_full_variant_assertion_fires_on_bypass`
- `test_full_variant_bypass_detection.py::test_lme_full_variant_assertion_fires_on_bypass`

**9 new tests, all passing.**

## Tests updated by Worker 1

- `test_openai_client.py:23` — assertion bumped from `gpt-4o-2024-08-06` to `gpt-4o-2024-11-20`
- `test_lme_client_fallback.py:28` — same
- `test_lme_client_fallback.py:34` — same
- `test_lme_client_fallback.py:131` — same
- `test_lme_client_fallback.py:132` — same

**5 assertions updated, all passing.**

## Tests not affected by Worker 1

- `tests/property/test_determinism.py` — green (Worker 1 doesn't touch substrate)
- `tests/property/test_temporal_validity.py` — does not exist on this branch (Worker 2's branch only)
- `tests/unit/substrate/` — green (no substrate changes here)
- `tests/unit/test_aci_bench_extractors.py` — green
- `tests/unit/test_adapters.py` — green
- `tests/unit/verifier/test_verifier.py` — green

## Time

Total test run: **2.60 s** for 380 tests + 2 skipped. Fast enough to run on every save during Wave B.
