"""
Tier 2 Broad Differential Engine for RobbyMD.

When no curated predicate pack exists, this engine provides a differential
from two open-access knowledge bases:
  - GetTheDiagnosis.org: 297 diagnoses with computed LRs from published sens/spec
  - HPO: 12,997 diseases with symptom-frequency associations

Architecture:
  Tier 1: Deep predicate pack (e.g. chest_pain/) — deterministic Bayesian, cited LRs
  Tier 2: This module — broad differential from open-access data
  Tier 3: Claim extraction + provenance only (no differential)
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

KB_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "broad_kb"


@dataclass
class RankedDiagnosis:
    name: str
    posterior: float
    log_odds: float
    matching_findings: list[dict] = field(default_factory=list)
    tier: str = "broad"
    source: str = ""


@dataclass
class BroadDifferential:
    query_findings: list[str]
    ranked: list[RankedDiagnosis]
    tier: str = "2"
    coverage: str = ""
    total_diseases_searched: int = 0


class BroadDifferentialEngine:
    """Tier 2 engine using GetTheDiagnosis.org LR data."""

    def __init__(self):
        self._gtd_kb: dict | None = None
        self._hpo_kb: dict | None = None

    def _load_gtd(self) -> dict:
        if self._gtd_kb is None:
            path = KB_DIR / "getthediagnosis_kb.json"
            if not path.exists():
                raise FileNotFoundError(
                    f"Broad KB not built. Run: python data/broad_kb/build_broad_kb.py"
                )
            with open(path, encoding="utf-8") as f:
                self._gtd_kb = json.load(f)
        return self._gtd_kb

    def _load_hpo(self) -> dict:
        if self._hpo_kb is None:
            path = KB_DIR / "hpo_kb.json"
            if not path.exists():
                return {}
            with open(path, encoding="utf-8") as f:
                self._hpo_kb = json.load(f)
        return self._hpo_kb

    def compute_differential(
        self,
        findings: list[str],
        top_n: int = 10,
        prior: float = 0.01,
    ) -> BroadDifferential:
        """
        Given a list of clinical findings (free text), compute a ranked
        differential using LR-based Bayesian updating from GetTheDiagnosis data.

        Args:
            findings: list of clinical finding strings (e.g. ["chest pain", "dyspnea"])
            top_n: number of top diagnoses to return
            prior: uniform prior probability for each disease
        """
        kb = self._load_gtd()
        findings_lower = [f.lower().strip() for f in findings]

        results: list[RankedDiagnosis] = []

        for dx_key, dx_data in kb.items():
            dx_findings = dx_data.get("findings", [])
            if not dx_findings:
                continue

            log_odds = math.log(prior / (1 - prior))
            matched: list[dict] = []

            for kb_finding in dx_findings:
                lr_pos = kb_finding.get("lr_positive")
                if lr_pos is None:
                    continue

                finding_name = kb_finding["finding"].lower().strip()

                for query_f in findings_lower:
                    if self._fuzzy_match(query_f, finding_name):
                        log_odds += math.log(max(lr_pos, 0.01))
                        matched.append({
                            "finding": kb_finding["finding"],
                            "lr_positive": lr_pos,
                            "lr_negative": kb_finding.get("lr_negative"),
                            "sensitivity": kb_finding.get("sensitivity"),
                            "specificity": kb_finding.get("specificity"),
                            "pmid": kb_finding.get("pmid"),
                        })
                        break

            if not matched:
                continue

            posterior = 1.0 / (1.0 + math.exp(-log_odds))

            results.append(RankedDiagnosis(
                name=dx_data["name"],
                posterior=round(posterior, 4),
                log_odds=round(log_odds, 4),
                matching_findings=matched,
                tier="broad-lr",
                source="GetTheDiagnosis.org",
            ))

        results.sort(key=lambda r: r.posterior, reverse=True)

        return BroadDifferential(
            query_findings=findings,
            ranked=results[:top_n],
            tier="2",
            coverage=f"{len(kb)} diagnoses with LR data from GetTheDiagnosis.org",
            total_diseases_searched=len(kb),
        )

    def _fuzzy_match(self, query: str, target: str) -> bool:
        """Simple substring matching for finding names."""
        if query in target or target in query:
            return True
        query_words = set(query.split())
        target_words = set(target.split())
        if len(query_words & target_words) >= max(1, len(query_words) - 1):
            return True
        return False


def resolve_tier(chief_complaint: str, pack_dir: Path | None = None) -> str:
    """Determine which tier to use for a given chief complaint."""
    if pack_dir and (pack_dir / "branches.json").exists():
        return "1"

    kb_path = KB_DIR / "getthediagnosis_kb.json"
    if kb_path.exists():
        with open(kb_path, encoding="utf-8") as f:
            kb = json.load(f)
        complaint_lower = chief_complaint.lower()
        for dx_data in kb.values():
            if complaint_lower in dx_data["name"].lower():
                has_lr = any("lr_positive" in f for f in dx_data.get("findings", []))
                if has_lr:
                    return "2"

    hpo_path = KB_DIR / "hpo_kb.json"
    if hpo_path.exists():
        return "2-hpo"

    return "3"


if __name__ == "__main__":
    engine = BroadDifferentialEngine()

    print("=" * 60)
    print("TIER 2 BROAD DIFFERENTIAL — Demo")
    print("=" * 60)

    test_cases = [
        {
            "name": "Chest Pain Presentation",
            "findings": ["chest pain", "dyspnea", "troponin", "ST depression"],
        },
        {
            "name": "Abdominal Pain Presentation",
            "findings": ["abdominal pain", "nausea", "fever", "guarding"],
        },
        {
            "name": "Headache Presentation",
            "findings": ["headache", "neck stiffness", "photophobia", "fever"],
        },
        {
            "name": "Joint Pain Presentation",
            "findings": ["joint pain", "swelling", "fever", "erythema"],
        },
    ]

    for case in test_cases:
        print(f"\n--- {case['name']} ---")
        print(f"Findings: {case['findings']}")
        result = engine.compute_differential(case["findings"], top_n=5)
        print(f"Searched: {result.total_diseases_searched} diagnoses")

        if not result.ranked:
            print("  No matching diagnoses found in broad KB")
            continue

        for i, dx in enumerate(result.ranked, 1):
            lr_info = ", ".join(
                f"{f['finding']} (LR+={f['lr_positive']})"
                for f in dx.matching_findings
            )
            print(f"  {i}. {dx.name} — posterior: {dx.posterior:.1%}")
            print(f"     Evidence: {lr_info}")
            print(f"     Source: {dx.source}")
