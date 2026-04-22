# src/differential — Deterministic LR-weighted differential engine

Per [`Eng_doc.md §5.4`](../../Eng_doc.md) and [`rules.md §5.1`](../../rules.md).

The engine consumes an iterable of `ActiveClaim` objects (structurally compatible
with whatever wt-engine's substrate emits) and the validated LR table in
[`predicate_packs/clinical_general/differentials/chest_pain/lr_table.json`](../../predicate_packs/clinical_general/differentials/chest_pain/lr_table.json).
It emits a `BranchRanking`: four branches (Cardiac, Pulmonary, MSK, GI) each
carrying a log-score, a softmax posterior, and a deterministic audit trail of
every LR multiplication applied.

## Invariants

1. **Pure and deterministic** — same inputs produce bit-identical outputs. No
   temperature, no seed, no stochastic re-ranking (`rules.md §5.1`). Enforced by
   [`tests/property/test_determinism.py`](../../tests/property/test_determinism.py) which
   runs the engine 100× on the canonical mid-case fixture.
2. **Closed predicate family** — `PREDICATE_FAMILIES` in `lr_table.py` mirrors
   the 14-entry closed set in `Eng_doc.md §4.2`. Extending requires a PR with
   cited clinical rationale (`rules.md §7.2`).
3. **Every LR has a citation** — loader rejects rows missing `source` or
   `source_url` (`rules.md §4.4`). Approximations are permitted when flagged.
4. **Iteration-order independence** — the engine sorts claims internally so
   upstream ordering cannot influence scoring.

## Algorithm

```
for claim in sorted(active_claims):
    for row in lr_table.rows_for(claim.predicate_path):
        if claim.polarity and row.lr_plus is not None:
            log_scores[row.branch] += log(row.lr_plus)
        elif not claim.polarity and row.lr_minus is not None:
            log_scores[row.branch] += log(row.lr_minus)
posteriors = softmax(log_scores)          # uniform prior over 4 branches
ranking = sorted by posterior desc, branch_id asc (stable)
```

Latency: four-branch full recompute in well under the 50 ms budget from
`Eng_doc.md §4.3`. Runs sync on the UI critical path.

## Prior art

This is a lightweight **deterministic instantiation of the Counterfactual
Probability Gap** principle on a fixed LR table, rather than an end-to-end
learned system. Two direct ancestors:

- **Counterfactual Multi-Agent Reasoning for Clinical Diagnosis** (arXiv
  [2603.27820](https://arxiv.org/abs/2603.27820), 2026). Defines the CPG metric
  as the shift in diagnosis probability when a clinical finding is removed or
  altered. Uses CPG to pick discriminative features across three benchmarks and
  seven LLMs. Our verifier re-uses that metric against a pooled LR table instead
  of against ensemble probabilities.
- **MedEinst** (arXiv [2601.06636](https://arxiv.org/abs/2601.06636), 2026).
  Introduces the **Bias Trap Rate** — pairs of cases where discriminative
  evidence is flipped — and the ECR-Agent's dynamic causal-inference loop.
  Motivates our verifier's quiet-when-confident gate: if the top-2 gap is wide
  the discriminator score is zero and the UI stays silent.

See [`docs/research_brief.md §2.2`](../../docs/research_brief.md) for the full
depth-signal argument.

## Counterfactual verifier

The verifier (in `src/verifier/`) is where CPG-style selection actually fires.
On every top-2 ranking change:

```
top2      = ranking[:2]
refute[X] = {rows r in branch X where lr_plus>1.5 and r.path not already asked}
disc      = argmax_r in (refute[A] ∪ refute[B]) of
              |log LR_A(r) - log LR_B(r)| * uncertainty
question  = opus_4_7(… describe disc in ≤20 words …)
```

`uncertainty = 1 - (posterior_a - posterior_b)` keeps the verifier silent when
the top-1 hypothesis is already dominant. Only the question phrasing goes
through Opus; discriminator selection itself is pure code so the demo is
replayable from a stored session.

Offline: `MockOpusClient` templates a canned question when
`ANTHROPIC_API_KEY` is unset, so CI and property tests run without network.

## File layout

| File | LOC (approx) | Purpose |
|---|---:|---|
| `engine.py` | 140 | Pure scoring; `rank_branches(...)` |
| `lr_table.py` | 150 | Loader + validator + indices |
| `projection.py` | 110 | Per-branch materialised view (UI-facing) |
| `types.py` | 25 | `ActiveClaim` shape, shared with wt-engine |
| `__init__.py` | 25 | Public surface |
