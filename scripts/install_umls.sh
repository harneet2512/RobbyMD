#!/usr/bin/env bash
# install_umls.sh — build a QuickUMLS index from an extracted UMLS Metathesaurus.
#
# Context: docs/decisions/2026-04-21_medcon-tiered-fallback.md
# Tier 0 (official MEDCON): QuickUMLS over UMLS 2025AB Level 0 Subset.
#
# Usage:
#   ./scripts/install_umls.sh <path to extracted UMLS META directory>
#
# Expected META layout (after unzipping umls-2025AB-Level0.zip and running
# MetamorphoSys):
#   <META>/MRCONSO.RRF
#   <META>/MRSTY.RRF
#   <META>/MRREL.RRF  (optional)
#
# Pinned version: UMLS 2025AB (released 2025-11-03). Do NOT auto-upgrade per
# ADR: reproducibility over freshness. If NLM publishes 2026AA, require an ADR
# before switching.
#
# Outputs the index to $QUICKUMLS_PATH (or ./.cache/quickumls_2025AB/ by
# default). Writes an env hint to stdout that the caller should source/export.
#
# Idempotent: if the output directory already contains a built index, skip
# rebuild unless FORCE=1.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <path to extracted UMLS META directory>" >&2
    echo "       (e.g. /path/to/2025AB/META or /mnt/umls/2025AB-Level0/META)" >&2
    exit 2
fi

UMLS_META="$1"
if [[ ! -d "${UMLS_META}" ]]; then
    echo "[install_umls] FAIL: ${UMLS_META} does not exist or is not a directory." >&2
    exit 1
fi
if [[ ! -f "${UMLS_META}/MRCONSO.RRF" ]] || [[ ! -f "${UMLS_META}/MRSTY.RRF" ]]; then
    echo "[install_umls] FAIL: MRCONSO.RRF and MRSTY.RRF must both exist in ${UMLS_META}." >&2
    echo "              Did MetamorphoSys finish? Both files are load-bearing for MEDCON." >&2
    exit 1
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

QUICKUMLS_PATH="${QUICKUMLS_PATH:-${repo_root}/.cache/quickumls_2025AB}"
FORCE="${FORCE:-0}"

echo "[install_umls] UMLS META dir: ${UMLS_META}"
echo "[install_umls] QuickUMLS output: ${QUICKUMLS_PATH}"

if [[ -d "${QUICKUMLS_PATH}" ]] && [[ -f "${QUICKUMLS_PATH}/database_backend.flag" ]] && [[ "${FORCE}" != "1" ]]; then
    echo "[install_umls] Index already present at ${QUICKUMLS_PATH}. Set FORCE=1 to rebuild."
    echo
    echo "export QUICKUMLS_PATH=\"${QUICKUMLS_PATH}\""
    echo "export CONCEPT_EXTRACTOR=quickumls"
    exit 0
fi

echo "[install_umls] pip install quickumls (best-effort; must be on OSI allowlist — MIT)"
python -m pip install "quickumls>=1.4.2"

mkdir -p "${QUICKUMLS_PATH}"

echo "[install_umls] Building index — this can take 20–60 min on first run."
python -m quickumls.install "${UMLS_META}" "${QUICKUMLS_PATH}"

echo "[install_umls] Smoke: load index + extract on one short sentence."
python - <<PYEOF
import sys

try:
    from quickumls import QuickUMLS
except ImportError as e:
    print(f"[install_umls] FAIL import quickumls: {e}", file=sys.stderr)
    sys.exit(1)

matcher = QuickUMLS("${QUICKUMLS_PATH}")
matches = matcher.match("The patient has chest pain and dyspnea.", best_match=True, ignore_syntax=False)
print(f"[install_umls] OK — extracted {len(matches)} match group(s) on smoke sentence")
PYEOF

echo "[install_umls] Done."
echo
echo "Export these in your shell (or .env) to activate T0 MEDCON:"
echo "  export QUICKUMLS_PATH=\"${QUICKUMLS_PATH}\""
echo "  export CONCEPT_EXTRACTOR=quickumls"
