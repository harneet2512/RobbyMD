# Worker 2 — Test Output

```
============================= test session starts =============================
platform win32 -- Python 3.12.0, pytest-9.0.2, pluggy-1.6.0
rootdir: D:\hack_it
configfile: pyproject.toml
plugins: anyio-4.12.1, langsmith-0.4.43, asyncio-1.3.0, timeout-2.4.0, xdist-3.8.0
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None,
         asyncio_default_test_loop_scope=function
collected 91 items

tests\property\test_temporal_validity.py ......                          [  6%]
tests\property\test_determinism.py ...                                   [  9%]
tests\unit\substrate\test_admission_and_events.py .................      [ 28%]
tests\unit\substrate\test_claims.py ............                         [ 41%]
tests\unit\substrate\test_on_new_turn.py .....                           [ 47%]
tests\unit\substrate\test_predicate_packs.py .......                     [ 54%]
tests\unit\substrate\test_projections_and_provenance.py ..........       [ 65%]
tests\unit\substrate\test_retrieval.py ............                      [ 79%]
tests\unit\substrate\test_schema.py ......                               [ 85%]
tests\unit\substrate\test_supersession.py .............                  [100%]

============================= 91 passed in 0.46s ==============================
```

## What this means

- **All 6 new temporal-validity tests pass** — schema CHECK fires, insert_claim guards inverted windows, supersession sets valid_until_ts to edge.created_ts, derived get_superseded_by works, run-twice structural determinism holds.
- **All 3 pre-existing determinism tests still pass** — the `time_ns → now_ns` switch in `write_supersession_edge` did not break any prior invariant.
- **All 13 pre-existing supersession unit tests still pass** — the additional `UPDATE claims SET valid_until_ts` is purely additive.
- **All 12 pre-existing claims tests pass** — `insert_claim` extension is backward-compatible (kwargs default to None, generate the same SQL as before for callers that don't pass them).
- **All retrieval / projections / on_new_turn / predicate_packs tests pass** — none of these read `valid_from_ts` / `valid_until_ts` yet, so they're unaffected.

## Tests not run in this batch

- `tests/e2e/` — out of scope for substrate-core; Worker 3/4/5 will add e2e coverage for projections.
- `tests/privacy/test_no_phi.py` — no PHI added by this change; can be re-run before merge.
- `tests/licensing/test_open_source.py` — no new dependencies; not affected.

## Time

Total test run: **0.46 s** for 91 tests. Test suite is fast enough to run on every save during Wave B.
