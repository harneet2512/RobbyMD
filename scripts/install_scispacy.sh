#!/usr/bin/env bash
# install_scispacy.sh — idempotent scispaCy T1 install for MEDCON-approx.
#
# Context: docs/decisions/2026-04-21_medcon-tiered-fallback.md
# Tier 1 (default): en_core_sci_lg + bundled UMLS linker.
# Runs on Day 1 — no UMLS licence required. Upgrade to T0 via install_umls.sh
# once the UMLS Metathesaurus licence lands.
#
# Exit codes:
#   0 — scispaCy + model installed, smoke import passes
#   1 — install failed (see stderr)
#
# Idempotent: re-running with pip already satisfied is a no-op.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

SCISPACY_VERSION="${SCISPACY_VERSION:-0.5.4}"
SCISPACY_MODEL="${SCISPACY_MODEL:-en_core_sci_lg}"
MODEL_URL="https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v${SCISPACY_VERSION}/${SCISPACY_MODEL}-${SCISPACY_VERSION}.tar.gz"

echo "[install_scispacy] scispaCy ${SCISPACY_VERSION}, model ${SCISPACY_MODEL}"
echo "[install_scispacy] pip install scispacy==${SCISPACY_VERSION}"
python -m pip install "scispacy==${SCISPACY_VERSION}"

echo "[install_scispacy] pip install ${MODEL_URL}"
python -m pip install "${MODEL_URL}"

echo "[install_scispacy] smoke: import scispacy + load ${SCISPACY_MODEL}"
python - <<PYEOF
import importlib
import sys

try:
    import scispacy  # noqa: F401
    import spacy
except ImportError as e:
    print(f"[install_scispacy] FAIL import: {e}", file=sys.stderr)
    sys.exit(1)

try:
    nlp = spacy.load("${SCISPACY_MODEL}")
except OSError as e:
    print(f"[install_scispacy] FAIL load model: {e}", file=sys.stderr)
    sys.exit(1)

doc = nlp("The patient reports crushing substernal chest pain radiating to the left arm.")
print(f"[install_scispacy] OK — smoke doc has {len(doc.ents)} entities")
PYEOF

echo "[install_scispacy] Done."
echo
echo "Next: set CONCEPT_EXTRACTOR=scispacy (default in .env.example) and run:"
echo "  python eval/aci_bench/smoke_test.py"
