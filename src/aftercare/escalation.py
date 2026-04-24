"""Escalation store — in-memory for hackathon, persistent for production.

The Patient Agent writes escalations when it detects red-flag matches or
care-boundary drift. The Doctor Agent reads them for review.
"""
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

import structlog

log = structlog.get_logger(__name__)


class TriggerType(StrEnum):
    RED_FLAG_MATCH = "red_flag_match"
    CARE_BOUNDARY_DRIFT = "care_boundary_drift"
    PATIENT_REQUESTED = "patient_requested"


class Urgency(StrEnum):
    ROUTINE = "routine"
    URGENT = "urgent"
    EMERGENCY = "emergency"


@dataclass(frozen=True, slots=True)
class Escalation:
    escalation_id: str
    encounter_id: str
    timestamp: str  # ISO 8601
    trigger_type: TriggerType
    patient_message: str
    matched_red_flag: str | None
    drift_type: str | None
    patient_safe_response: str
    doctor_reviewed: bool = False
    doctor_response: str | None = None
    urgency: Urgency | None = None

    def to_dict(self) -> dict:
        return {
            "escalation_id": self.escalation_id,
            "encounter_id": self.encounter_id,
            "timestamp": self.timestamp,
            "trigger_type": self.trigger_type.value,
            "patient_message": self.patient_message,
            "matched_red_flag": self.matched_red_flag,
            "drift_type": self.drift_type,
            "patient_safe_response": self.patient_safe_response,
            "doctor_reviewed": self.doctor_reviewed,
            "doctor_response": self.doctor_response,
            "urgency": self.urgency.value if self.urgency else None,
        }


class EscalationStore:
    """Thread-safe in-memory escalation store."""

    def __init__(self) -> None:
        self._escalations: list[Escalation] = []
        self._lock = threading.Lock()

    def create(
        self,
        encounter_id: str,
        trigger_type: TriggerType,
        patient_message: str,
        patient_safe_response: str,
        matched_red_flag: str | None = None,
        drift_type: str | None = None,
    ) -> Escalation:
        esc = Escalation(
            escalation_id=f"esc_{uuid.uuid4().hex[:12]}",
            encounter_id=encounter_id,
            timestamp=datetime.now(UTC).isoformat(),
            trigger_type=trigger_type,
            patient_message=patient_message,
            matched_red_flag=matched_red_flag,
            drift_type=drift_type,
            patient_safe_response=patient_safe_response,
        )
        with self._lock:
            self._escalations.append(esc)
        log.info(
            "escalation.created",
            escalation_id=esc.escalation_id,
            trigger_type=trigger_type.value,
            encounter_id=encounter_id,
        )
        return esc

    def get(self, escalation_id: str) -> Escalation | None:
        with self._lock:
            for e in self._escalations:
                if e.escalation_id == escalation_id:
                    return e
        return None

    def list_for_encounter(
        self, encounter_id: str, unreviewed_only: bool = False
    ) -> list[Escalation]:
        with self._lock:
            results = [
                e for e in self._escalations if e.encounter_id == encounter_id
            ]
            if unreviewed_only:
                results = [e for e in results if not e.doctor_reviewed]
            return results

    def mark_reviewed(
        self, escalation_id: str, doctor_response: str, urgency: Urgency
    ) -> Escalation | None:
        with self._lock:
            for i, e in enumerate(self._escalations):
                if e.escalation_id == escalation_id:
                    updated = Escalation(
                        escalation_id=e.escalation_id,
                        encounter_id=e.encounter_id,
                        timestamp=e.timestamp,
                        trigger_type=e.trigger_type,
                        patient_message=e.patient_message,
                        matched_red_flag=e.matched_red_flag,
                        drift_type=e.drift_type,
                        patient_safe_response=e.patient_safe_response,
                        doctor_reviewed=True,
                        doctor_response=doctor_response,
                        urgency=urgency,
                    )
                    self._escalations[i] = updated
                    log.info(
                        "escalation.reviewed",
                        escalation_id=escalation_id,
                        urgency=urgency.value,
                    )
                    return updated
        return None


_store = EscalationStore()


def get_escalation_store() -> EscalationStore:
    return _store
