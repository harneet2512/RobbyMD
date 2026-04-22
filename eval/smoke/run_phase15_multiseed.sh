#!/usr/bin/env bash
# Stream B.2 Phase 1.5 multi-seed re-runs (seeds 43 + 44).
#
# Launches both seeds in sequence against the same 10 ACI-Bench cases as
# the seed-42 Phase 1 run. Sequential to avoid doubling gpt-4.1-mini TPM
# pressure on the claim extractor. If you've raised the Azure SKU capacity,
# flip the two `python ...` calls into background with trailing ` &`.
#
# Requires .env populated with Modal + Azure creds; mimics the seed-42
# invocation exactly except for --seed and --output-dir.

set -eu
cd "$(dirname "$0")/../.."

# shellcheck disable=SC1091
set -a && source .env && set +a
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

for SEED in 43 44; do
    TS=$(date -u +%Y%m%dT%H%M%SZ)
    OUTDIR="eval/acibench/results/20260423_postmerge_hybrid_phase15_${TS}_seed${SEED}"
    echo "=== Stream B.2 seed=${SEED} -> ${OUTDIR} ==="
    python eval/smoke/run_smoke.py \
        --benchmark acibench \
        --reader qwen2.5-14b \
        --variant both \
        --n 10 \
        --seed "${SEED}" \
        --output-dir "${OUTDIR}"
    echo "=== Seed ${SEED} done ==="
done

echo "=== B.3 aggregate ==="
python eval/smoke/aggregate_seeds.py \
    --benchmark acibench \
    --pattern "eval/acibench/results/20260423_postmerge_hybrid_phase1*_seed*/results.json"
