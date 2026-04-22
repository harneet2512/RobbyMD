"""Pure-Python unit tests for the three benchmark adapters.

Exercises the deterministic transforms (record → Turn list) without touching
the substrate, API, or external files. Keeps the wt-eval scaffold CI-green
before the full fetchers run.
"""
from __future__ import annotations

from eval.aci_bench.adapter import ACIEncounter, encounter_to_turns
from eval.ddxplus.adapter import DDXPlusCase, record_to_turns
from eval.longmemeval.adapter import LongMemEvalQuestion, session_to_turns


class TestDDXPlusAdapter:
    def test_record_to_turns_opens_with_physician(self) -> None:
        case = DDXPlusCase(
            patient_id="P1",
            age=42,
            sex="F",
            pathology="Acute pulmonary embolism",
            evidences=["E_1", "E_2_@_V_1"],
            differential=[("Acute pulmonary embolism", 0.7)],
            initial_evidence=None,
        )
        ev_dict = {
            "E_1": {"question_en": "Do you have chest pain?"},
            "E_2": {
                "question_en": "Where is the pain?",
                "value_meaning": {"1": {"en": "left side"}},
            },
        }
        turns = record_to_turns(case, ev_dict)
        assert turns[0].speaker == "physician"
        assert "42-year-old" in turns[0].text
        # Each evidence yields a physician + patient pair.
        assert any(t.speaker == "patient" and t.text == "Yes." for t in turns)
        assert any(t.speaker == "patient" and "left side" in t.text for t in turns)

    def test_record_to_turns_is_deterministic(self) -> None:
        case = DDXPlusCase(
            patient_id="P1",
            age=30,
            sex="M",
            pathology="X",
            evidences=["E_1"],
            differential=[],
            initial_evidence=None,
        )
        ev_dict = {"E_1": {"question_en": "Q?"}}
        t1 = record_to_turns(case, ev_dict)
        t2 = record_to_turns(case, ev_dict)
        assert t1 == t2

    def test_unknown_evidence_is_skipped(self) -> None:
        case = DDXPlusCase(
            patient_id="P1",
            age=30,
            sex="M",
            pathology="X",
            evidences=["E_UNKNOWN"],
            differential=[],
            initial_evidence=None,
        )
        turns = record_to_turns(case, {})
        # Only the opener physician turn — no evidence was resolved.
        assert len(turns) == 1
        assert turns[0].speaker == "physician"


class TestLongMemEvalAdapter:
    def test_session_to_turns_preserves_role_tag(self) -> None:
        q = LongMemEvalQuestion(
            question_id="Q1",
            question="...",
            answer="...",
            question_type="temporal_reasoning",
            haystack_sessions=[
                [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}],
            ],
        )
        turns = session_to_turns(q, 0)
        assert len(turns) == 2
        assert turns[0].speaker == "system"   # both roles map to system per adapter
        assert "[user]" in turns[0].text
        assert "[assistant]" in turns[1].text
        # Turn IDs are stable and include session index.
        assert turns[0].turn_id.startswith("Q1::session::000::msg::")

    def test_empty_content_skipped(self) -> None:
        q = LongMemEvalQuestion(
            question_id="Q2",
            question="...",
            answer="...",
            question_type="abstention",
            haystack_sessions=[
                [{"role": "user", "content": ""}, {"role": "user", "content": "hi"}],
            ],
        )
        turns = session_to_turns(q, 0)
        assert len(turns) == 1


class TestACIBenchAdapter:
    def test_encounter_to_turns_maps_speakers(self) -> None:
        enc = ACIEncounter(
            encounter_id="D_1",
            split="aci",
            subsplit="test1",
            dialogue=[
                {"speaker": "DOCTOR", "utterance": "Good morning."},
                {"speaker": "PATIENT", "utterance": "Hi, I have chest pain."},
                {"speaker": "UNKNOWN", "utterance": "system event"},
            ],
            gold_note="...",
        )
        turns = encounter_to_turns(enc)
        assert turns[0].speaker == "physician"
        assert turns[1].speaker == "patient"
        assert turns[2].speaker == "system"
        # Stable turn IDs.
        assert turns[0].turn_id == "D_1::turn::0000"
        assert turns[2].turn_id == "D_1::turn::0002"

    def test_empty_utterances_skipped(self) -> None:
        enc = ACIEncounter(
            encounter_id="D_2",
            split="aci",
            subsplit="test1",
            dialogue=[
                {"speaker": "DOCTOR", "utterance": ""},
                {"speaker": "PATIENT", "utterance": "Yes."},
            ],
            gold_note="...",
        )
        turns = encounter_to_turns(enc)
        assert len(turns) == 1
