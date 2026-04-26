"""Claude Managed Agents — clinical reasoning agents operating against the substrate.

Five agents, one substrate, tool-level access control + always_ask permissions.
"""
from src.agents.doctor_agent import DoctorAgent
from src.agents.patient_agent import PatientAgent
from src.agents.handoff_agent import HandoffAgent
from src.agents.bias_monitor_agent import BiasMonitorAgent
from src.agents.note_coauthor_agent import NoteCoauthorAgent

__all__ = [
    "DoctorAgent",
    "PatientAgent",
    "HandoffAgent",
    "BiasMonitorAgent",
    "NoteCoauthorAgent",
]
