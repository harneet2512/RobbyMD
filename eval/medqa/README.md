# eval/medqa — MedQA context-layer ablation

**Not a substrate benchmark.** See `docs/decisions/2026-04-23_medqa-reinstated-as-context-layer-ablation.md` for the scope-limited framing that supersedes the 2026-04-21 drop (`reasons.md:119-125`).

This directory runs USMLE-style multiple-choice questions through two conditions:

- **Baseline** — raw vignette + 4 options → reader → letter answer.
- **Context-layer** — vignette → claim extractor (`clinical_general` predicate pack) → substrate (in-memory, one vignette per session) → structured findings grouped by predicate family + vignette + options → reader → letter answer.

Same N questions, paired conditions, McNemar 2×2. We are measuring whether a structured-extraction layer helps a *weak reader* (gpt-4o-mini) organize clinical reasoning on pre-structured prose. Results are reported separately from LongMemEval / ACI-Bench; never aggregated into a substrate claim.

## Dataset

- Source: [`GBaker/MedQA-USMLE-4-options`](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options) on HuggingFace.
- Paper: *What Disease Does This Patient Have? A Large-Scale Open Domain Question Answering Dataset from Medical Exams* — Jin et al., arXiv:2009.13081 (EMNLP 2021).
- Test split: 1273 questions, 4 options each. Downloaded to `eval/data/medqa/test.jsonl` (not committed — reconstructable via `scripts/download_medqa.py` or the snippet in `SYNTHETIC_DATA.md`).

### Schema

Each line is one JSON object:

```
{
  "question":          "…clinical vignette prose…",
  "options":           {"A": "…", "B": "…", "C": "…", "D": "…"},
  "answer":            "…full text of the correct option…",
  "answer_idx":        "A" | "B" | "C" | "D",
  "meta_info":         "step1" | "step2" | "step3",
  "metamap_phrases":   [...]   // unused by us
}
```

## Files

- `adapter.py` — extractor → substrate → structured findings → prompt formatting. Single-vignette flow. No retrieval (each vignette is its own session); claims are dumped flat via `list_active_claims`.
- `run_medqa_smoke.py` — standalone n=50 smoke harness. Paired baseline + context-layer, deterministic first-N selection, McNemar contingency, markdown report. Stays out of `eval/smoke/run_smoke.py`'s `BENCHMARKS` tuple until the smoke signal is observed.
- `results/` (gitignored) — timestamped run artifacts (`smoke_results.json`, `smoke_report.md`).

## How to download

```bash
pip install datasets --break-system-packages   # if not already
python - <<'PY'
from datasets import load_dataset
ds = load_dataset('GBaker/MedQA-USMLE-4-options', split='test')
ds.to_json('D:/hack_it/eval/data/medqa/test.jsonl')
print(f'{len(ds)} questions saved')
PY
```

## How to run smoke (n=50)

```bash
python eval/medqa/run_medqa_smoke.py \
    --n 50 \
    --reader gpt-4o-mini-2024-07-18 \
    --seed 42 \
    --budget-usd 20 \
    --dry-run
```

Drop `--dry-run` after the cost estimate clears $10. The run produces `eval/medqa/results/<UTC-timestamp>/smoke_results.json` and `…/smoke_report.md`.

## Pass / anomaly / fail criteria (from `docs/decisions/2026-04-23_medqa-reinstated-as-context-layer-ablation.md`)

| Outcome at n=50 | Action |
|---|---|
| delta ≥ 2pp, wins ≥ regressions, parse ≥ 95%, ≥ 3 claims/vignette | Escalate to full n=1273 (separate approval cycle). |
| 0 < delta < 2pp OR wins ≈ regressions | ANOMALY — do not escalate. |
| delta ≤ 0 OR extraction fails > 10% OR parse < 95% | FAIL — archive unused, update ADR to `abandoned`. |

## What this directory is NOT

- Not a substrate benchmark. Supersession, temporal edges, retrieval, multi-session — none of these fire on a single-vignette MCQ.
- Not a claim about overall clinical reasoning ability. Only about the organizational effect of a structured-extraction layer.
- Not part of the main demo path. Read-only in the demo; no UI surface.
