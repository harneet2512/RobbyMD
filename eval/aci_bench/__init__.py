"""ACI-Bench benchmark harness.

Upstream: https://github.com/wyim/aci-bench
Methodology: MEDIQA-CHAT 2023 — full `aci` + `virtscribe` test splits
(90 encounters), ROUGE-1/2/L, BERTScore, MEDCON.

MEDCON is implemented via a 3-tier `ConceptExtractor` (see extractors.py) per
`docs/decisions/2026-04-21_medcon-tiered-fallback.md`.

See eval/aci_bench/README.md for dataset pin, license, and comparator table.
"""
