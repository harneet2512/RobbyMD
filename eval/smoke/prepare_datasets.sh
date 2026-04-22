#!/usr/bin/env bash
# prepare_datasets.sh — One-shot dataset fetch for smoke runs.
#
# Clones the two canonical benchmark repos into eval/data/ (gitignored) at
# pinned commit SHAs per eval/<benchmark>/README.md. Deterministic first-10-case
# selection is done by run_smoke.py at runtime (sorted case_id).
#
# Re-runnable: skips clone if the dir already exists. shellcheck-clean.

set -euo pipefail

# Resolve repo root (script is at eval/smoke/prepare_datasets.sh).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_DIR="${REPO_ROOT}/eval/data"

# Pinned commit SHAs — update these (and eval/<b>/README.md) together when we re-pin.
LONGMEMEVAL_REPO="https://github.com/xiaowu0162/LongMemEval"
LONGMEMEVAL_SHA="${LONGMEMEVAL_SHA:-HEAD}"     # TODO(operator): pin SHA before first real run
ACIBENCH_REPO="https://github.com/wyim/aci-bench"
ACIBENCH_SHA="${ACIBENCH_SHA:-HEAD}"           # TODO(operator): pin SHA before first real run

mkdir -p "${DATA_DIR}"

clone_or_skip() {
    local name="$1" repo="$2" sha="$3"
    local target="${DATA_DIR}/${name}"
    if [[ -d "${target}/.git" ]]; then
        echo "[datasets] ${name}: already cloned at ${target} — skipping."
        return 0
    fi
    echo "[datasets] ${name}: cloning ${repo} into ${target}..."
    git clone --quiet "${repo}" "${target}"
    if [[ "${sha}" != "HEAD" ]]; then
        git -C "${target}" checkout --quiet "${sha}"
    fi
    local head
    head="$(git -C "${target}" rev-parse HEAD)"
    echo "[datasets] ${name}: pinned at ${head}"
}

clone_or_skip "longmemeval" "${LONGMEMEVAL_REPO}" "${LONGMEMEVAL_SHA}"
clone_or_skip "acibench" "${ACIBENCH_REPO}" "${ACIBENCH_SHA}"

# ----- LongMemEval-S data file (not in the repo; lives on HuggingFace) -----
# The LongMemEval repo README directs users to `wget` the cleaned JSON from
# HuggingFace at runtime (see repo README §Setup › Data). We fetch the 500-
# question cleaned split and place it at the path `adapter.py` expects.
LM_DATA_DIR="${DATA_DIR}/longmemeval/data"
LM_JSON_PATH="${LM_DATA_DIR}/longmemeval_s.json"
mkdir -p "${LM_DATA_DIR}"
if [[ ! -f "${LM_JSON_PATH}" ]]; then
    echo "[datasets] LongMemEval-S: fetching longmemeval_s_cleaned.json from HuggingFace..."
    curl -L --fail --show-error \
        "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json" \
        -o "${LM_JSON_PATH}"
fi

# ----- summary -----
echo
echo "[datasets] === Summary ==="
if [[ -f "${LM_JSON_PATH}" ]]; then
    lm_count=$(python3 -c "import json; print(len(json.load(open('${LM_JSON_PATH}'))))")
    echo "[datasets] LongMemEval-S: ${lm_count} cases at ${LM_JSON_PATH}"
else
    echo "[datasets] LongMemEval-S: data file not found at ${LM_JSON_PATH}"
fi
# ACI-Bench ships three JSON files under challenge_data_json/ — one per test
# split (test1 = clinicalnlp_taskB, test2 = clinicalnlp_taskC, test3 = clef_taskC).
# The adapter (eval/aci_bench/adapter.py) now reads this flat JSON layout.
AB_JSON_DIR="${DATA_DIR}/acibench/data/challenge_data_json"
if [[ -d "${AB_JSON_DIR}" ]]; then
    ab_files=(
        "clinicalnlp_taskB_test1.json"
        "clinicalnlp_taskC_test2.json"
        "clef_taskC_test3.json"
    )
    total=0
    for f in "${ab_files[@]}"; do
        path="${AB_JSON_DIR}/${f}"
        if [[ -f "${path}" ]]; then
            n=$(python3 -c "import json; print(len(json.load(open('${path}'))['data']))")
            echo "[datasets] ACI-Bench ${f}: ${n} rows"
            total=$((total + n))
        else
            echo "[datasets] ACI-Bench: expected file missing at ${path}"
        fi
    done
    echo "[datasets] ACI-Bench total rows: ${total}"
else
    echo "[datasets] ACI-Bench: JSON dir not found at ${AB_JSON_DIR}"
fi

echo
echo "[datasets] Next: python eval/smoke/run_smoke.py --dry-run --benchmark both --reader qwen2.5-14b --variant both --n 10"
