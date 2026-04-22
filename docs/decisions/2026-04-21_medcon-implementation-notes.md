# 2026-04-21 — MEDCON implementation notes (delta vs tiered-fallback ADR)

**Status**: accepted
**Relates to**: `docs/decisions/2026-04-21_medcon-tiered-fallback.md`
**Driver**: wt-eval agent (scaffolding pass)
**Affected**: `scripts/install_scispacy.sh`, `eval/aci_bench/smoke_test.py`, `Eng_doc.md §12` risks table

## Context

The parent ADR (2026-04-21_medcon-tiered-fallback) defines a 3-tier
`ConceptExtractor` with T1 scispaCy as the default Day-1 gate:

> Day 1 (2026-04-21) — `./scripts/install_scispacy.sh` completes; smoke test
> extracts ≥10 CUIs from one ACI-Bench reference note.

This note records the delta observed during Day-1 scaffolding in `wt-eval` —
the T1 install does **not** complete cleanly on Windows without additional
build tooling. It does not invalidate the ADR; it adds an implementation
caveat the ADR's alternatives section did not anticipate.

## What happened

`scripts/install_scispacy.sh` (scaffolded to spec: `scispacy==0.5.4` +
`en_core_sci_lg-0.5.4.tar.gz`) failed on the scaffolding machine with:

```
Building wheel for scipy (pyproject.toml) — FAILED
Unknown compiler(s): [['icl'], ['cl'], ['cc'], ['gcc'], ['clang'], ['clang-cl'], ['pgcc']]
```

Root cause: `scispacy==0.5.4` pins `scipy>=1.4,<1.10`; pip-resolves to
`scipy==1.9.3`, which Meson-builds from source on Python 3.12 because no
prebuilt Windows wheel ships for that combination. The fix path requires a
C compiler (MSVC Build Tools or mingw-w64), which is a laptop-level install,
not a `pip install` step.

Host env: Windows 11, Python 3.12.0, no MSVC. Project's `pyproject.toml`
declares `requires-python = ">=3.11,<3.12"` — Python 3.11 would likely resolve
to a scipy version with a prebuilt Windows wheel.

## Decision (operator-level mitigations)

The T1 install is still the **intended** Day-1 path. Three operator-level
unblocks, in order of preference:

1. **Use Python 3.11 (matches `pyproject.toml`).** scispaCy 0.5.4's scipy
   pin has prebuilt Windows wheels for cp311. Create a 3.11 venv, re-run
   `./scripts/install_scispacy.sh`, re-run the smoke. Most likely path.
2. **Install MSVC Build Tools 2019+** (Desktop development with C++) on
   Python 3.12 and re-run. Adds ~4 GB of tooling; overkill for a hackathon
   unless the operator already has them.
3. **Relax the scispaCy pin** to `>=0.5.5` and retry on 3.12. 0.5.5 onwards
   bumps scipy upper bound; Windows wheels exist for newer scipy on 3.12.
   Requires validating that the UMLS linker bundled with `en_core_sci_lg`
   still works; the attribution line in
   `eval/aci_bench/extractors.py:ScispacyExtractor` does not change.

None of these materially affect the ADR's tier table, extractor Protocol, or
Report-Both rule — the swap point is the environment, not the code.

## Fallback plan if T1 install cannot be unblocked by Day 2

Per parent ADR decision-gate row:

> Day 1 — smoke test extracts ≥10 CUIs. Action on no: fall to T2 tomorrow;
> reasons.md entry.

Concretely: if by end of Day 2 the operator cannot get `scispacy` importing
AND QuickUMLS is not ready, set `CONCEPT_EXTRACTOR=null`. T2 is already
scaffolded (`NullExtractor`); the slide-render path already handles the
"MEDCON omitted" case; section-level ROUGE + MedEinst Bias Trap Rate take
over as clinical-rigor proxies.

## Scripts already handle the diagnostic path

`scripts/install_scispacy.sh` runs a smoke import + model load before
exiting 0. On failure, its output points directly at the compiler / build
error, which is the correct signal for the operator to pick one of the three
mitigations above. The script is idempotent — safe to re-run after
installing MSVC or switching Python.

## Consequences

- `eval/aci_bench/smoke_test.py` runs cleanly in T2 mode (verified with
  `CONCEPT_EXTRACTOR=null`). T1 mode will run once any of the three
  mitigations lands.
- No change to `extractors.py` Protocol, factory, or F1 maths.
- No change to `LIMITATIONS.md` tier sections — they already cover T2 as
  the documented hard-fallback state.
- `Eng_doc.md §12` risks table already calls out MEDCON / UMLS; this note
  adds a co-dependency: scispaCy on Windows + Python 3.12 requires a C
  compiler. Tracked here; not worth promoting into `Eng_doc.md` unless the
  install continues to fail across mitigation attempts.

## References

- `docs/decisions/2026-04-21_medcon-tiered-fallback.md` — parent ADR
- `scripts/install_scispacy.sh` — Day-1 install path
- `eval/aci_bench/smoke_test.py` — Day-1 gate
- [scispaCy 0.5.4 release](https://github.com/allenai/scispacy/releases/tag/v0.5.4)
- [SciPy Windows wheels availability table](https://pypi.org/project/scipy/#files)
