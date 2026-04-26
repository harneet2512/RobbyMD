"""Aftercare package — the physician-approved patient-facing artifact.

Generated at encounter close from:
- Active claims projection (what we know)
- SOAP note (the clinical summary)
- Verifier final state (red-flag source)
- Decision log (what the doctor decided)

The package is LOCKED after physician approval. The Patient Agent serves
exactly this content — nothing more.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any

import structlog

from src.substrate.claims import list_active_claims
from src.substrate.schema import Claim, ClaimStatus

log = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class MedicationInstruction:
    medication: str
    instructions: str
    source_claim_id: str


@dataclass(frozen=True, slots=True)
class FollowUpItem:
    action: str
    timeframe: str | None
    source_claim_id: str | None


@dataclass(frozen=True, slots=True)
class AftercarePackage:
    """The locked patient-facing artifact. Immutable after physician approval."""

    encounter_id: str
    summary: str
    follow_up_plan: tuple[FollowUpItem, ...]
    medication_instructions: tuple[MedicationInstruction, ...]
    red_flag_symptoms: tuple[str, ...]
    medical_terms: tuple[str, ...]
    approved: bool
    approval_note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "encounter_id": self.encounter_id,
            "summary": self.summary,
            "follow_up_plan": [
                {"action": f.action, "timeframe": f.timeframe}
                for f in self.follow_up_plan
            ],
            "medication_instructions": [
                {"medication": m.medication, "instructions": m.instructions}
                for m in self.medication_instructions
            ],
            "red_flag_symptoms": list(self.red_flag_symptoms),
            "medical_terms": list(self.medical_terms),
            "approved": self.approved,
            "approval_note": self.approval_note,
        }


def generate_aftercare_package(
    conn: sqlite3.Connection,
    session_id: str,
    soap_note: str | None = None,
    red_flags: tuple[str, ...] = (),
) -> AftercarePackage:
    """Generate an aftercare package from the current encounter state."""
    claims = list_active_claims(conn, session_id)

    medications = _extract_medications(claims)
    follow_ups = _extract_follow_ups(claims)
    summary = _build_summary(claims, soap_note)
    terms = _extract_medical_terms(claims)

    package = AftercarePackage(
        encounter_id=session_id,
        summary=summary,
        follow_up_plan=tuple(follow_ups),
        medication_instructions=tuple(medications),
        red_flag_symptoms=red_flags,
        medical_terms=tuple(terms),
        approved=False,
    )
    log.info(
        "aftercare.package_generated",
        encounter_id=session_id,
        n_red_flags=len(red_flags),
    )
    return package


def _extract_medications(claims: list[Claim]) -> list[MedicationInstruction]:
    meds: list[MedicationInstruction] = []
    for c in claims:
        if c.predicate == "medication" and c.status in (
            ClaimStatus.ACTIVE,
            ClaimStatus.CONFIRMED,
        ):
            meds.append(
                MedicationInstruction(
                    medication=c.value,
                    instructions=f"Take {c.value} as prescribed by your doctor.",
                    source_claim_id=c.claim_id,
                )
            )
    return meds


def _extract_follow_ups(claims: list[Claim]) -> list[FollowUpItem]:
    follow_ups: list[FollowUpItem] = []
    for c in claims:
        if c.predicate == "risk_factor" and c.status in (
            ClaimStatus.ACTIVE,
            ClaimStatus.CONFIRMED,
        ):
            follow_ups.append(
                FollowUpItem(
                    action=f"Follow up regarding {c.value}",
                    timeframe=None,
                    source_claim_id=c.claim_id,
                )
            )
    if not follow_ups:
        follow_ups.append(
            FollowUpItem(
                action="Follow up with your doctor as directed",
                timeframe="within 1 week",
                source_claim_id=None,
            )
        )
    return follow_ups


def _build_summary(claims: list[Claim], soap_note: str | None) -> str:
    if soap_note:
        lines: list[str] = []
        in_plan = False
        for line in soap_note.split("\n"):
            lower = line.strip().lower()
            if lower.startswith("plan") or lower.startswith("p:"):
                in_plan = True
            if in_plan and line.strip():
                lines.append(line.strip())
        if lines:
            return "\n".join(lines)

    parts: list[str] = []
    for c in claims:
        if c.status in (ClaimStatus.ACTIVE, ClaimStatus.CONFIRMED):
            parts.append(f"- {c.predicate}: {c.value}")
    if parts:
        return "Based on your visit, the following was discussed:\n" + "\n".join(
            parts
        )
    return "Your visit summary is being prepared."


_MEDICAL_VOCAB = frozenset({
    "troponin",
    "ecg",
    "ekg",
    "electrocardiogram",
    "angiogram",
    "ct scan",
    "mri",
    "x-ray",
    "ultrasound",
    "stress test",
    "cardiac catheterization",
    "omeprazole",
    "aspirin",
    "nitroglycerin",
    "heparin",
    "d-dimer",
    "bnp",
})


def _extract_medical_terms(claims: list[Claim]) -> list[str]:
    terms: set[str] = set()
    for c in claims:
        val_lower = c.value.lower()
        for term in _MEDICAL_VOCAB:
            if term in val_lower:
                terms.add(term)
    return sorted(terms)


_packages: dict[str, AftercarePackage] = {}


def cache_package(package: AftercarePackage) -> None:
    _packages[package.encounter_id] = package


def get_cached_package(encounter_id: str) -> AftercarePackage | None:
    return _packages.get(encounter_id)


def approve_package(
    encounter_id: str, approval_note: str | None = None
) -> AftercarePackage | None:
    pkg = _packages.get(encounter_id)
    if pkg is None:
        return None
    approved = AftercarePackage(
        encounter_id=pkg.encounter_id,
        summary=pkg.summary,
        follow_up_plan=pkg.follow_up_plan,
        medication_instructions=pkg.medication_instructions,
        red_flag_symptoms=pkg.red_flag_symptoms,
        medical_terms=pkg.medical_terms,
        approved=True,
        approval_note=approval_note,
    )
    _packages[encounter_id] = approved
    log.info("aftercare.package_approved", encounter_id=encounter_id)
    return approved
