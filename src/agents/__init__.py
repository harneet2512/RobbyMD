"""Aftercare agents — Claude Managed Agents for doctor and patient aftercare.

Two agents, one substrate, tool-level access control.
"""
from src.agents.doctor_agent import DoctorAgent
from src.agents.patient_agent import PatientAgent

__all__ = ["DoctorAgent", "PatientAgent"]
