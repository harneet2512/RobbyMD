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

# ----- summary -----
echo
echo "[datasets] === Summary ==="
if [[ -d "${DATA_DIR}/longmemeval" ]]; then
    # LongMemEval-S JSON lives at data/longmemeval_s.json; count case entries.
    lm_file="${DATA_DIR}/longmemeval/data/longmemeval_s.json"
    if [[ -f "${lm_file}" ]]; then
        lm_count=$(python3 -c "import json; print(len(json.load(open('${lm_file}'))))")
        echo "[datasets] LongMemEval-S: ${lm_count} cases at ${lm_file}"
    else
        echo "[datasets] LongMemEval-S: data file not found at expected path ${lm_file}"
    fi
fi
if [[ -d "${DATA_DIR}/acibench" ]]; then
    ab_test1="${DATA_DIR}/acibench/data/challenge_data/test1"
    if [[ -d "${ab_test1}" ]]; then
        ab_count=$(find "${ab_test1}" -maxdepth 1 -type d | wc -l)
        echo "[datasets] ACI-Bench test1: ~${ab_count} encounter dirs at ${ab_test1}"
    else
        echo "[datasets] ACI-Bench: test1 dir not found at expected path ${ab_test1}"
    fi
fi

echo
echo "[datasets] Next: python eval/smoke/run_smoke.py --dry-run --benchmark both --reader qwen2.5-14b --variant both --n 10"
