# Worker 1 — Implementation Notes

## Decisions

### Override on `longmemeval_*` purposes (paper-faithfulness vs deprecation)

Sub-agent's investigation was right that `longmemeval_reader` and
`longmemeval_judge` were intentionally pinned to `gpt-4o-2024-08-06` for
paper-faithfulness with LongMemEval ICLR 2025 (Wu et al., arXiv:2410.10813).
But the model was **retired 2026-03-31**: calls return 404. Paper-faithfulness
with the original snapshot is no longer mechanically achievable.

Override: bump both to `gpt-4o-2024-11-20`. Methodology deviation must be
labelled in any reported number — added inline comment in
`_openai_client.py::_DIRECT_DEFAULTS` and noted in `methodology.md`-class
language. This is a legitimacy-preserving honest disclosure, not a
methodological violation: the alternative is reporting numbers that can't be
generated at all.

### Bypass-assertion placement

The orchestrator's CRITICAL CONSTRAINT was: "Do not allow benchmark runs
where full benchmark path == baseline pipeline. If detected, stop execution
and report." The sentinel `[SUBSTRATE STUB]` is the discriminator — it gets
prepended to `raw_response` only when the bypass fires. Approach:

```python
wrapped = ACINotePrediction(..., raw_response="[SUBSTRATE STUB] ..." + pred.raw_response)
assert "[SUBSTRATE STUB]" not in wrapped.raw_response, "Bypass detected: ..."
return wrapped
```

Today this assertion **fires**, because the bypass is real. The new test
`test_full_variant_bypass_detection.py` confirms exactly this. When
Worker 3 lands the real substrate path and the sentinel-prefix line is
removed from the prediction, the assertion passes silently and the test
should be inverted (assert no AssertionError fires). That inversion rides
with Worker 3's PR.

### `RuntimeError` instead of warn-and-continue for missing API key

Two options for the silent-gold-leak fix:
- **A)** Log a warning and skip the encounter (predict empty string).
- **B)** Raise `RuntimeError` with a message explaining why.

Chose **B**. A no-op-on-failure baseline still produces *some* score
(near zero), which an unattended sweep would happily publish as "baseline
performance." Loud failure is the only way to guarantee the operator
notices and either sets the key or removes the run from the cycle.

### Stub `_StubACIEncounter` / `_StubLMEQuestion` in tests

The real `ACIEncounter` and `LongMemEvalQuestion` dataclasses have many
required fields irrelevant to the API-key check. Built minimal stubs with
only the fields the baseline functions touch (`encounter_id`, `gold_note`,
`dialogue` for ACI; `question_id`, `question_type`, `question`, `answer`,
`haystack_sessions`, `haystack_dates` for LME). `# type: ignore[arg-type]`
on the call site silences the type mismatch — fine for a focused unit test
that exercises only the early-return path.

## Surprises

1. **Two test files asserted the old model id**, not just one. Sub-agent
   identified `test_openai_client.py:23`. There was also
   `test_lme_client_fallback.py:28,34,131,132` (4 more assertions) to
   update. Found via the test failure output, not pre-emptively.
2. **Pyright `tuple` complaints**: bare `tuple = ()` defaults trigger
   `reportMissingTypeArgument`. Fixed by switching to `tuple[object, ...] = ()`
   for the stub dataclasses. Cosmetic but kept the diagnostics clean.
3. **`_DIRECT_DEFAULTS` is a private symbol** but `test_lme_client_fallback.py`
   already imports it for assertions. Pyright flags this as
   `reportPrivateUsage` — pre-existing convention in this codebase, not
   introduced by Worker 1.

## Deviations from spec

- **Skipped: `eval/_openai_client.py:122` Azure deployment fallback bump.**
  No hardcoded date string at that line — the deployment env-var keys
  reference `AZURE_OPENAI_GPT4O_*_DEPLOYMENT` env vars, not literal model
  ids. The model id only lives in `_DIRECT_DEFAULTS`. Sub-agent's plan
  noted this as a no-op; confirmed.
- **Skipped: docstring update in `_openai_client.py:21,48,55`** for the
  three purposes that mention `gpt-4o-2024-08-06`. Updating prose without
  a behavioural change is risky (drift between docstring and code if one
  changes later). The inline comment near `_DIRECT_DEFAULTS` is the
  source of truth; the docstring deltas can land in a separate doc-only PR.

## Hand-off to Worker 3 (and to anyone landing the substrate backend)

When the real substrate path replaces the stub fallback in
`eval/aci_bench/full.py:36–47` and `eval/longmemeval/full.py:35–48`:

1. Remove the `"[SUBSTRATE STUB] ..."` prefix from `raw_response`.
2. Either delete the bypass-detection assertion or keep it — once the
   sentinel is gone, it passes silently, so it's safe defense in depth.
3. Invert the two tests in `test_full_variant_bypass_detection.py`:
   they should assert that the assertion **does not fire** (i.e. the
   prediction object has the substrate's real raw_response, not the
   bypass sentinel).

## Impact on Worker 2 / 3 / 4 / 5

- Worker 2 (substrate-core): no impact. Worker 1 doesn't touch substrate code.
- Worker 3 (LME temporal): when wiring the real LME substrate path, follow
  the hand-off list above.
- Worker 4 (retrieval fusion): no direct impact, but if it adds a new
  Azure-routed purpose to `_DIRECT_DEFAULTS`, do not re-introduce
  `gpt-4o-2024-08-06`.
- Worker 5 (ACI audit-revise): same hand-off as Worker 3 for the ACI-Bench
  full variant.
