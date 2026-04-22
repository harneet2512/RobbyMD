"""Pure-Python unit tests for the benchmark adapters.

Exercises the deterministic transforms (record → Turn list) without touching
the substrate, API, or external files. Keeps the wt-eval scaffold CI-green
before the full fetchers run.

DDXPlus adapter tests removed 2026-04-21 — DDXPlus dropped from benchmark set
(see `reasons.md` → "DDXPlus — dropped for substrate-benchmark misalignment").
"""
from __future__ import annotations

from eval.aci_bench.adapter import ACIEncounter, encounter_to_turns
from eval.longmemeval.adapter import LongMemEvalQuestion, session_to_turns


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
