"""Benchmark harnesses for the hack_it clinical-reasoning substrate.

Three published benchmarks (no homemade metrics, no slicing — rules.md §6):
- ddxplus: DDXPlus (NeurIPS 2022) via H-DDx 2025 (Top-5 + HDF1, 730-case stratified).
- longmemeval: LongMemEval-S (ICLR 2025) — all 500 questions, per-category.
- aci_bench: ACI-Bench (Nature Sci Data 2023) — aci + virtscribe test splits, 90 encounters.

Each subpackage scaffolds fetch.py, adapter.py, baseline.py, full.py, run.py,
README.md, and LIMITATIONS.md per CLAUDE.md §5.5 and Eng_doc.md §10.
"""
