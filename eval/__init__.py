"""Benchmark harnesses for the hack_it clinical-reasoning substrate.

Two published benchmarks (no homemade metrics, no slicing — rules.md §6):
- longmemeval: LongMemEval-S (ICLR 2025) — all 500 questions, per-category;
  runs with `personal_assistant` pack for substrate variant.
- aci_bench: ACI-Bench (Nature Sci Data 2023) — aci + virtscribe test splits,
  90 encounters; runs with `clinical_general` pack.

DDXPlus + MedQA were dropped 2026-04-21 — see `reasons.md` and
`eval/README.md`. Each surviving subpackage provides fetch.py, adapter.py,
baseline.py, full.py, run.py, README.md, LIMITATIONS.md per CLAUDE.md §5.5.
"""
