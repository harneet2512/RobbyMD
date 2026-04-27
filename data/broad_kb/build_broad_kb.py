"""
Build the Broad Knowledge Base for RobbyMD Tier 2 differential.

Combines:
  1. GetTheDiagnosis.org (297 diagnoses, 1,238 entries with sens/spec → compute LRs)
  2. HPO phenotype annotations (282K disease-symptom rows, 12K+ diseases)

Outputs:
  - broad_kb.json: unified knowledge base with LR data where available
  - stats.json: coverage statistics

Usage:
    python data/broad_kb/build_broad_kb.py
"""

import json
import csv
import re
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data" / "broad_kb"
EVAL_DIR = ROOT / "eval"


def parse_hpo_ontology(obo_path: Path) -> dict[str, str]:
    """Parse hp.obo to get HPO ID → human-readable name mapping."""
    terms: dict[str, str] = {}
    current_id = None
    with open(obo_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "[Term]":
                current_id = None
            elif line.startswith("id: HP:"):
                current_id = line.split("id: ")[1]
            elif line.startswith("name: ") and current_id:
                terms[current_id] = line.split("name: ")[1]
    return terms


def parse_hpo_annotations(tab_path: Path, hpo_names: dict[str, str]) -> dict:
    """Parse phenotype.hpoa into disease → symptom mappings with frequency."""
    diseases: dict[str, dict] = defaultdict(lambda: {
        "name": "",
        "source": "HPO",
        "findings": [],
    })

    with open(tab_path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 8:
                continue

            db_id = parts[0]
            disease_name = parts[1]
            qualifier = parts[2]
            hpo_id = parts[3]
            reference = parts[4]
            frequency_raw = parts[7] if len(parts) > 7 else ""

            if qualifier == "NOT":
                continue

            symptom_name = hpo_names.get(hpo_id, hpo_id)

            freq = parse_frequency(frequency_raw)

            key = f"{db_id}"
            diseases[key]["name"] = disease_name
            diseases[key]["findings"].append({
                "finding": symptom_name,
                "hpo_id": hpo_id,
                "frequency": freq,
                "reference": reference,
            })

    return dict(diseases)


def parse_frequency(raw: str) -> float | None:
    """Convert HPO frequency annotations to a float 0-1."""
    if not raw:
        return None

    # Fraction format: "3/10"
    m = re.match(r"(\d+)/(\d+)", raw)
    if m:
        num, den = int(m.group(1)), int(m.group(2))
        return round(num / den, 4) if den > 0 else None

    # Percentage format: "45%"
    m = re.match(r"([\d.]+)%", raw)
    if m:
        return round(float(m.group(1)) / 100, 4)

    # HPO frequency terms
    freq_map = {
        "HP:0040280": 1.0,      # Obligate (100%)
        "HP:0040281": 0.90,     # Very frequent (80-99%)
        "HP:0040282": 0.55,     # Frequent (30-79%)
        "HP:0040283": 0.12,     # Occasional (5-29%)
        "HP:0040284": 0.025,    # Very rare (1-4%)
        "HP:0040285": 0.005,    # Excluded (0%)
    }
    return freq_map.get(raw)


def parse_getthediagnosis(json_path: Path) -> dict:
    """Parse scraped GetTheDiagnosis data, compute LRs from sens/spec."""
    diagnoses: dict[str, dict] = defaultdict(lambda: {
        "name": "",
        "source": "GetTheDiagnosis.org",
        "findings": [],
    })

    with open(json_path, encoding="utf-8") as f:
        entries = json.load(f)

    for entry in entries:
        dx = entry["diagnosis"]
        sens = entry.get("sensitivity")
        spec = entry.get("specificity")

        finding_data: dict = {
            "finding": entry["finding"],
            "section": entry.get("section", ""),
            "pmid": entry.get("pmid"),
        }

        if sens is not None and spec is not None and sens > 0 and spec < 100:
            sens_frac = sens / 100.0
            spec_frac = spec / 100.0
            lr_pos = round(sens_frac / (1 - spec_frac), 2) if spec_frac < 1 else 999.0
            lr_neg = round((1 - sens_frac) / spec_frac, 4) if spec_frac > 0 else 0.0
            finding_data["sensitivity"] = sens
            finding_data["specificity"] = spec
            finding_data["lr_positive"] = lr_pos
            finding_data["lr_negative"] = lr_neg

        diagnoses[dx]["name"] = dx
        diagnoses[dx]["findings"].append(finding_data)

    return dict(diagnoses)


def build_unified_kb():
    """Merge HPO and GetTheDiagnosis into a unified broad KB."""

    print("Parsing HPO ontology...")
    hpo_names = parse_hpo_ontology(DATA_DIR / "hp.obo")
    print(f"  {len(hpo_names)} HPO terms loaded")

    print("Parsing HPO annotations...")
    hpo_diseases = parse_hpo_annotations(
        DATA_DIR / "phenotype_annotations.tab", hpo_names
    )
    print(f"  {len(hpo_diseases)} diseases from HPO")

    gtd_path = EVAL_DIR / "getthediagnosis_data.json"
    gtd_diseases: dict = {}
    if gtd_path.exists():
        print("Parsing GetTheDiagnosis.org data...")
        gtd_diseases = parse_getthediagnosis(gtd_path)
        lr_count = sum(
            1
            for d in gtd_diseases.values()
            for f in d["findings"]
            if "lr_positive" in f
        )
        print(f"  {len(gtd_diseases)} diagnoses, {lr_count} findings with computed LRs")
    else:
        print("  GetTheDiagnosis data not found, skipping")

    # Build stats
    total_hpo_findings = sum(len(d["findings"]) for d in hpo_diseases.values())
    total_gtd_findings = sum(len(d["findings"]) for d in gtd_diseases.values())
    gtd_with_lr = sum(
        1
        for d in gtd_diseases.values()
        for f in d["findings"]
        if "lr_positive" in f
    )

    stats = {
        "hpo": {
            "diseases": len(hpo_diseases),
            "total_findings": total_hpo_findings,
            "source": "Human Phenotype Ontology (hpo.jax.org)",
            "license": "HPO License (open for research + clinical use)",
            "version": "2026-02-16",
        },
        "getthediagnosis": {
            "diseases": len(gtd_diseases),
            "total_findings": total_gtd_findings,
            "findings_with_lr": gtd_with_lr,
            "source": "GetTheDiagnosis.org (Kohlberg & Hammer, 2016)",
            "license": "Open access / Creative Commons",
        },
        "combined": {
            "total_diseases": len(hpo_diseases) + len(gtd_diseases),
            "total_findings": total_hpo_findings + total_gtd_findings,
            "findings_with_computed_lr": gtd_with_lr,
        },
    }

    # Save GetTheDiagnosis KB (the one with LRs — most useful for Tier 2)
    gtd_out = DATA_DIR / "getthediagnosis_kb.json"
    with open(gtd_out, "w", encoding="utf-8") as f:
        json.dump(gtd_diseases, f, indent=2, ensure_ascii=False)
    print(f"\nGetTheDiagnosis KB saved: {gtd_out}")

    # Save HPO KB (broader but no LRs — useful for symptom matching)
    hpo_out = DATA_DIR / "hpo_kb.json"
    with open(hpo_out, "w", encoding="utf-8") as f:
        json.dump(hpo_diseases, f, indent=2, ensure_ascii=False)
    print(f"HPO KB saved: {hpo_out}")

    # Save stats
    stats_out = DATA_DIR / "stats.json"
    with open(stats_out, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats saved: {stats_out}")

    print(f"\n{'='*60}")
    print(f"BROAD KNOWLEDGE BASE BUILT")
    print(f"{'='*60}")
    print(f"  GetTheDiagnosis: {len(gtd_diseases)} diagnoses, {gtd_with_lr} findings with LRs")
    print(f"  HPO:             {len(hpo_diseases)} diseases, {total_hpo_findings} symptom links")
    print(f"  TOTAL:           {len(hpo_diseases) + len(gtd_diseases)} conditions covered")
    print(f"{'='*60}")


if __name__ == "__main__":
    build_unified_kb()
