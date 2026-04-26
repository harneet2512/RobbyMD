"""Patient Agent tools — approved-view access only.

Seven tools giving the Patient Agent access to ONLY the physician-approved
aftercare information. No reasoning graph, no hypothesis probabilities,
no differential ranking.
"""
from __future__ import annotations

import json
import os
import sqlite3
from typing import Any

import structlog

from src.aftercare.escalation import TriggerType, get_escalation_store
from src.aftercare.package import get_cached_package
from src.aftercare.red_flags import check_symptoms_against_flags

log = structlog.get_logger(__name__)


def patient_tool_manifest() -> list[dict[str, Any]]:
    """Tool definitions for the Patient managed agent."""
    return [
        {
            "type": "custom",
            "name": "get_approved_summary",
            "description": "Returns the doctor-approved patient-facing visit summary.",
            "input_schema": {
                "type": "object",
                "properties": {"encounter_id": {"type": "string"}},
                "required": ["encounter_id"],
            },
        },
        {
            "type": "custom",
            "name": "get_follow_up_plan",
            "description": "Returns next steps, pending tests, and appointment guidance.",
            "input_schema": {
                "type": "object",
                "properties": {"encounter_id": {"type": "string"}},
                "required": ["encounter_id"],
            },
        },
        {
            "type": "custom",
            "name": "explain_approved_term",
            "description": "Explains a medical term using only the visit context.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "encounter_id": {"type": "string"},
                    "term": {"type": "string", "description": "Medical term to explain"},
                },
                "required": ["encounter_id", "term"],
            },
        },
        {
            "type": "custom",
            "name": "check_red_flag_symptoms",
            "description": (
                "Checks patient-reported symptoms against the approved red-flag "
                "list. If matched, you MUST call escalate_to_doctor."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "encounter_id": {"type": "string"},
                    "symptoms": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Symptoms the patient reports",
                    },
                },
                "required": ["encounter_id", "symptoms"],
            },
        },
        {
            "type": "custom",
            "name": "get_medication_instructions",
            "description": "Returns plain-language medication instructions from the visit.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "encounter_id": {"type": "string"},
                    "medication": {"type": "string"},
                },
                "required": ["encounter_id"],
            },
        },
        {
            "type": "custom",
            "name": "detect_care_boundary_drift",
            "description": (
                "Analyzes a patient message to determine if it moves outside "
                "the approved aftercare boundary."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "encounter_id": {"type": "string"},
                    "patient_message": {"type": "string"},
                },
                "required": ["encounter_id", "patient_message"],
            },
        },
        {
            "type": "custom",
            "name": "escalate_to_doctor",
            "description": (
                "Sends a structured escalation to the Doctor Agent. MANDATORY "
                "when red flags match or care-boundary drift is detected."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "encounter_id": {"type": "string"},
                    "trigger_type": {
                        "type": "string",
                        "enum": [
                            "red_flag_match",
                            "care_boundary_drift",
                            "patient_requested",
                        ],
                    },
                    "patient_message": {"type": "string"},
                    "matched_red_flag": {"type": "string"},
                    "patient_safe_response": {"type": "string"},
                },
                "required": ["encounter_id", "trigger_type", "patient_message"],
            },
        },
    ]


class PatientToolDispatcher:
    """Executes patient tools against ONLY the approved aftercare view."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def dispatch(self, tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
        handler = getattr(self, f"_handle_{tool_name}", None)
        if handler is None:
            return {"error": f"Unknown tool: {tool_name}"}
        try:
            return handler(tool_input)
        except Exception as exc:
            log.error("patient_tool.error", tool=tool_name, error=str(exc))
            return {"error": str(exc)}

    def _handle_get_approved_summary(self, inp: dict[str, Any]) -> dict[str, Any]:
        pkg = get_cached_package(inp["encounter_id"])
        if not pkg:
            return {
                "summary": (
                    "Your visit summary is being prepared. "
                    "Please check back shortly."
                )
            }

        result: dict[str, Any] = {"summary": pkg.summary}
        if not pkg.approved:
            result["note"] = (
                "Your doctor has not yet reviewed this summary. "
                "It was generated from your visit automatically."
            )
        return result

    def _handle_get_follow_up_plan(self, inp: dict[str, Any]) -> dict[str, Any]:
        pkg = get_cached_package(inp["encounter_id"])
        if not pkg:
            return {"follow_up": "Follow up with your doctor as directed."}
        return {
            "follow_up_items": [
                {"action": f.action, "timeframe": f.timeframe}
                for f in pkg.follow_up_plan
            ]
        }

    def _handle_explain_approved_term(self, inp: dict[str, Any]) -> dict[str, Any]:
        term = inp["term"]
        pkg = get_cached_package(inp["encounter_id"])

        if os.environ.get("ANTHROPIC_API_KEY"):
            try:
                return _explain_with_opus(term, pkg)
            except Exception as exc:
                log.warning("opus_explain.failed", error=str(exc))

        return {
            "term": term,
            "explanation": (
                f"'{term}' was mentioned during your visit. Please ask your "
                "doctor for a detailed explanation at your next appointment."
            ),
        }

    def _handle_check_red_flag_symptoms(self, inp: dict[str, Any]) -> dict[str, Any]:
        pkg = get_cached_package(inp["encounter_id"])
        if not pkg or not pkg.red_flag_symptoms:
            return {"matches": [], "any_matched": False}

        matches = check_symptoms_against_flags(
            inp["symptoms"], pkg.red_flag_symptoms
        )
        return {
            "matches": [
                {
                    "patient_symptom": m.patient_symptom,
                    "matched_flag": m.matched_flag,
                    "is_match": m.is_match,
                }
                for m in matches
            ],
            "any_matched": any(m.is_match for m in matches),
        }

    def _handle_get_medication_instructions(
        self, inp: dict[str, Any]
    ) -> dict[str, Any]:
        pkg = get_cached_package(inp["encounter_id"])
        medication = inp.get("medication")

        if not pkg:
            return {
                "instructions": (
                    "Please follow the medication instructions "
                    "provided by your doctor."
                )
            }

        if medication:
            for med in pkg.medication_instructions:
                if medication.lower() in med.medication.lower():
                    return {
                        "medication": med.medication,
                        "instructions": med.instructions,
                    }
            return {
                "medication": medication,
                "instructions": (
                    f"No specific instructions found for {medication}. "
                    "Please contact your doctor."
                ),
            }

        return {
            "medications": [
                {"medication": m.medication, "instructions": m.instructions}
                for m in pkg.medication_instructions
            ]
        }

    def _handle_detect_care_boundary_drift(
        self, inp: dict[str, Any]
    ) -> dict[str, Any]:
        message = inp["patient_message"]

        if os.environ.get("ANTHROPIC_API_KEY"):
            try:
                return _detect_drift_with_opus(message, inp["encounter_id"])
            except Exception as exc:
                log.warning("opus_drift.failed", error=str(exc))

        return _detect_drift_keywords(message)

    def _handle_escalate_to_doctor(self, inp: dict[str, Any]) -> dict[str, Any]:
        store = get_escalation_store()
        esc = store.create(
            encounter_id=inp["encounter_id"],
            trigger_type=TriggerType(inp["trigger_type"]),
            patient_message=inp["patient_message"],
            patient_safe_response=inp.get(
                "patient_safe_response", "Your care team has been notified."
            ),
            matched_red_flag=inp.get("matched_red_flag"),
            drift_type=inp.get("drift_type"),
        )
        return {
            "escalation_id": esc.escalation_id,
            "status": "sent",
            "message": "Escalation sent to the care team.",
        }


def _explain_with_opus(
    term: str, pkg: Any | None
) -> dict[str, Any]:
    from anthropic import Anthropic

    client = Anthropic()
    context = pkg.summary if pkg else "a clinical visit"
    msg = client.messages.create(
        model="claude-opus-4-7-20250415",
        max_tokens=200,
        system=(
            "You explain medical terms to patients in plain language. "
            "Use ONLY the visit context provided. Do not add information "
            "beyond the visit. Be warm, clear, and brief (2-3 sentences)."
        ),
        messages=[
            {
                "role": "user",
                "content": (
                    f"Visit context: {context}\n\n"
                    f"Explain this term to the patient: {term}"
                ),
            }
        ],
    )
    text = (
        msg.content[0].text
        if msg.content
        else f"'{term}' is a medical term from your visit."
    )
    return {"term": term, "explanation": text}


def _detect_drift_with_opus(
    message: str, encounter_id: str
) -> dict[str, Any]:
    from anthropic import Anthropic

    client = Anthropic()
    pkg = get_cached_package(encounter_id)
    context = pkg.summary if pkg else "a clinical visit"

    msg = client.messages.create(
        model="claude-opus-4-7-20250415",
        max_tokens=100,
        system=(
            "Classify this patient message as either SAFE (within approved "
            "aftercare scope) or DRIFT (outside scope).\n"
            "Drift categories: requesting_new_diagnosis, "
            "questioning_doctor_decision, requesting_treatment_change, "
            "reporting_new_symptom, high_anxiety_escalation, "
            "requesting_pending_result_interpretation, "
            "off_topic_medical_question.\n"
            'Return JSON only: {"drift_detected": bool, '
            '"drift_type": string|null, "confidence": float}'
        ),
        messages=[
            {
                "role": "user",
                "content": (
                    f"Visit context: {context}\n\nPatient message: {message}"
                ),
            }
        ],
    )
    text = (
        msg.content[0].text
        if msg.content
        else '{"drift_detected": false, "drift_type": null, "confidence": 0.5}'
    )
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"drift_detected": False, "drift_type": None, "confidence": 0.0}


def _detect_drift_keywords(message: str) -> dict[str, Any]:
    lower = message.lower()
    drift_patterns: dict[str, list[str]] = {
        "requesting_new_diagnosis": [
            "do i have",
            "could it be",
            "what disease",
            "diagnose",
        ],
        "questioning_doctor_decision": [
            "miss",
            "missed",
            "wrong",
            "should have",
            "second opinion",
            "mistake",
        ],
        "requesting_treatment_change": [
            "stop taking",
            "change medication",
            "switch to",
            "don't want to take",
        ],
        "high_anxiety_escalation": [
            "going to die",
            "am i dying",
            "cancer",
            "worst case",
        ],
        "requesting_pending_result_interpretation": [
            "what do my results mean",
            "test results",
            "lab results",
        ],
    }
    for drift_type, keywords in drift_patterns.items():
        if any(kw in lower for kw in keywords):
            return {
                "drift_detected": True,
                "drift_type": drift_type,
                "confidence": 0.7,
            }
    return {"drift_detected": False, "drift_type": None, "confidence": 0.9}
