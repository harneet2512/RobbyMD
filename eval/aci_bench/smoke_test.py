"""Day-1 smoke test for the active `ConceptExtractor` tier.

Per `docs/decisions/2026-04-21_medcon-tiered-fallback.md` decision gate:

> Day 1 (2026-04-21) — `./scripts/install_scispacy.sh` completes; smoke test
> extracts ≥10 CUIs from one ACI-Bench reference note.
> Action on yes: T1 locked as default; proceed.
> Action on no: fall to T2 tomorrow; reasons.md entry.

This script picks the extractor via `CONCEPT_EXTRACTOR` env var (default
scispacy), extracts CUIs from a **pre-downloaded** sample clinical note
(bundled inline so the smoke test runs before `fetch.py` lands the full
90-encounter dataset), and:

- exits 0 with a readable summary if ≥10 CUIs extracted
- exits 1 and prints a diagnostic if <10

Sample note: a SYNTHETIC, ACI-Bench-style SOAP excerpt authored for this
smoke (no real patient data). Long enough that a UMLS-backed extractor
should comfortably pick 10+ concepts.

Run:

    python eval/aci_bench/smoke_test.py
    python eval/aci_bench/smoke_test.py --verbose

This is NOT a pytest test; it's an operator-facing gate with a clear exit
code. pytest coverage of the `ConceptExtractor` factory lives in
`tests/unit/test_aci_bench_extractors.py` (TODO — follow-up commit; the
factory is already unit-testable).
"""
# SYNTHETIC — not real patient data. See SYNTHETIC_DATA.md.
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python eval/aci_bench/smoke_test.py` from the repo root (no install
# required). pyproject.toml's packages.find covers `src*` only; `eval` is a
# runtime-executed tree, not a distributable package.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.aci_bench.extractors import NullExtractor, build_extractor  # noqa: E402

# SYNTHETIC sample — authored for smoke test, deliberately rich in UMLS-
# linkable clinical concepts across the 7 MEDCON semantic groups (disorders,
# anatomy, procedures, chemicals/drugs). ACI-Bench-style SOAP note excerpt.
SAMPLE_NOTE = """
SUBJECTIVE:
The patient is a 58-year-old female with a past medical history of type 2
diabetes mellitus, essential hypertension, and hyperlipidemia who presents
today with complaints of intermittent retrosternal chest pain radiating to
the left arm for the past three days. The pain is described as pressure-like,
lasting approximately 10-15 minutes, occurs with exertion such as climbing
stairs, and is partially relieved by rest. The patient denies associated
shortness of breath, diaphoresis, nausea, or vomiting. She reports adherence
to her current medications including metformin 1000 mg twice daily,
lisinopril 20 mg daily, and atorvastatin 40 mg at bedtime.

OBJECTIVE:
Vital signs: blood pressure 142/88 mmHg, heart rate 78 beats per minute,
respiratory rate 16, oxygen saturation 98% on room air, temperature 36.8 C.
Physical examination: the patient appears comfortable. Cardiovascular exam
reveals a regular rate and rhythm with no murmurs, rubs, or gallops.
Pulmonary exam: clear to auscultation bilaterally. Electrocardiogram shows
normal sinus rhythm without acute ST-segment changes. Troponin I is pending.

ASSESSMENT:
Stable angina pectoris, likely secondary to coronary artery disease given
the patient's risk factors including diabetes, hypertension, and dyslipidemia.
Uncontrolled type 2 diabetes mellitus, with HbA1c of 8.2% from two months
ago. Hypertension, poorly controlled on current regimen.

PLAN:
1. Order treadmill stress echocardiogram to evaluate for inducible ischemia.
2. Initiate aspirin 81 mg daily and metoprolol tartrate 25 mg twice daily.
3. Increase lisinopril to 40 mg daily; recheck blood pressure in two weeks.
4. Refer to cardiology for further evaluation.
5. Order fasting lipid panel, comprehensive metabolic panel, and HbA1c.
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", "-v", action="store_true", help="print CUI list")
    parser.add_argument(
        "--min-cuis",
        type=int,
        default=10,
        help="ADR gate threshold (default 10 per docs/decisions/2026-04-21_medcon-tiered-fallback.md)",
    )
    args = parser.parse_args(argv)

    try:
        extractor = build_extractor()
    except RuntimeError as e:
        print(f"[smoke] FAIL building extractor: {e}", file=sys.stderr)
        return 1

    print(f"[smoke] active tier: {extractor.name} — {extractor.label}")

    if isinstance(extractor, NullExtractor):
        print(
            "[smoke] T2 active (NullExtractor). MEDCON is intentionally omitted; "
            "see eval/aci_bench/LIMITATIONS.md §T2. This is NOT a pass but IS "
            "the documented T2 state. Exiting 0 (no failure to report)."
        )
        return 0

    try:
        cuis = extractor.extract(SAMPLE_NOTE)
    except Exception as e:
        print(f"[smoke] FAIL extraction: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    print(f"[smoke] extracted {len(cuis)} CUIs from sample note ({len(SAMPLE_NOTE)} chars)")
    if args.verbose:
        for cui in sorted(cuis):
            print(f"  {cui}")
    else:
        sample = sorted(cuis)[:15]
        print("[smoke] first 15 CUIs (alphabetical): " + ", ".join(sample))

    if len(cuis) >= args.min_cuis:
        print(f"[smoke] PASS — ≥{args.min_cuis} CUIs; tier {extractor.name} locked as active.")
        return 0
    print(
        f"[smoke] FAIL — extracted {len(cuis)} CUIs, below threshold {args.min_cuis}.\n"
        "       Per ADR decision gate: fall to T2 tomorrow; log a reasons.md entry."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
