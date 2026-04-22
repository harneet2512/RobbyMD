# 2026-04-21 — Audit: chest-pain coupling in shipped code (next-turn fixes)

**Status**: audit complete; no immediate code changes (per user directive 2026-04-21: *"flag findings in an ADR for next-turn fix; do not block on the audit"*).
**Driver**: hack_it operator
**Affected**: `src/differential/lr_table.py`, `src/extraction/claim_extractor/prompt.py`, descriptive coupling in `src/extraction/asr/`.

## Context

Principle 2 of the 2026-04-21 directive: *the substrate must work on any chief complaint — abdominal pain, dyspnoea, headache, whatever a judge throws at it*. During the `predicate_packs/` refactor I audited shipped code for chest-pain assumptions that would break if a different complaint loaded. Findings below.

## Findings

### 1. `src/differential/lr_table.py:38` — hardcoded `BRANCHES` frozenset

```python
BRANCHES: frozenset[str] = frozenset({"cardiac", "pulmonary", "msk", "gi"})
```

This is chest-pain-specific. Abdominal pain would have `{hepatobiliary, gi, genitourinary, vascular, msk}`, headache `{primary, secondary_vascular, meningeal, raised_icp}`, etc. The LR-table validator (`test_unknown_branch_rejected`) uses this frozenset to reject rows with unknown branch names on load.

**Recommended fix (next turn)**: derive `BRANCHES` from the active LR table at load time — e.g. `branches = frozenset(row["branch"] for row in table["entries"])` — and validate each row's branch appears in the same set. Per-pack validation; the engine stays generic.

**Why not fixed in this commit**: risk-scoping. `BRANCHES` is covered by `test_unknown_branch_rejected` in the green 140-test suite; re-engineering it requires new tests to preserve the existing rejection semantics. Safer to fix when the second complaint lands (first real exercise of the generality).

### 2. `src/extraction/claim_extractor/prompt.py` — chest-pain few-shot examples

All 6 few-shot examples use `"subject":"chest_pain"` with chest-pain-specific predicate values. The prompt's closed predicate list (drawn from the active pack) is domain-agnostic, but the few-shots bias Opus 4.7 toward chest-pain-shaped outputs.

**Recommended fix (next turn)**: move few-shot examples into the `PredicatePack` itself — `PredicatePack.few_shot_examples: list[ExtractionExample]`. Loading a different pack swaps in its own examples. Alternatively, keep one domain-neutral example set + allow packs to append domain-specific shots.

**Why not fixed in this commit**: the Phase-1 demo is chest-pain; the existing hardcoded few-shots deliver the demo. Generalising without a second pack to test against risks extraction regression without upside. Fix when the second pack (or second complaint) seeds.

### 3. `src/extraction/asr/vocab.py` + `synth_audio.py` — chest-pain demo fixtures

Both modules are explicitly chest-pain-themed: `DEMO_SCRIPT` is the scripted 10-utterance chest-pain dialogue; `initial_prompt` biasing is a chest-pain-flavoured RxNorm/ICD-10 set.

**This is appropriate — no fix needed.** These are demo-case fixtures, not engine code. They are clearly labelled (module docstrings explicitly say "chest-pain dialogue") and do not affect substrate generality. A second complaint would add its own `synth_audio.py`-style demo script + its own `vocab.py`-style `initial_prompt` bias — sibling fixtures, not replacements.

### 4. `content/differentials/chest_pain/` relocated

This commit moved the directory to `predicate_packs/clinical_general/differentials/chest_pain/` via `git mv`. Sibling stubs exist at `abdominal_pain/`, `dyspnoea/`, `headache/` with READMEs noting future scope. No engine changes required to add new complaint branches — just a new directory + `branches.json` + `lr_table.json` + `sources.md`.

## Decision

**Audit complete — next-turn fixes tracked here.** Findings #1 and #2 are on the engine-generality critical path; fix when `personal_assistant` (for LongMemEval-S substrate variant) or a second clinical complaint seeds (whichever comes first). Findings #3 and #4 require no action.

## Revisit trigger

- When the second `PredicatePack` seeds (`personal_assistant` for LongMemEval-S substrate variant, or a second clinical complaint like `abdominal_pain`), findings #1 and #2 become blockers.
- If a Stage 2 judge opens the repo, loads a non-chest-pain complaint, and the `BRANCHES` frozenset silently rejects the new complaint's branch names instead of reporting "no LR table for this complaint" — that's the failure mode Finding #1 would expose. Fix before that happens.

## References

- User directive 2026-04-21 Principle 2 (*"chest pain is one example, not the product"*)
- `Eng_doc.md §4.2` — pluggable predicate packs (amended 2026-04-21)
- `rules.md §1.1` — fresh code only (no pre-existing GT v2 reuse)
- `predicate_packs/clinical_general/README.md` — shipped pack spec
- `reasons.md` — entry *"Scope widened under the hood; demo narrative stays clinical (2026-04-21)"*
