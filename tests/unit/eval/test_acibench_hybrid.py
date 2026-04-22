"""Unit tests for the ACI-Bench hybrid substrate variant.

Stream B deliverable (2026-04-22). The hybrid variant replaces the prior
2-step `pre-extract → prepend → separate-SOAP-call` that regressed
baseline by −0.070 MEDCON-F1 on the `20260422T081250Z` n=10 smoke.
Hybrid ships a single reader call whose prompt carries:

- SECTION 1: raw transcript (baseline fidelity)
- SECTION 2: structured claim scaffold (substrate audit layer)
- SECTION 3: conflict-resolution rule (baked in for audit visibility)
- SECTION 4: task instruction

These tests assert prompt structure, edge cases (empty scaffold,
supersession-chain rendering), and that the LLM is called exactly once
(speed parity with baseline).

All LLM clients are mocked — no network, no API key required.
"""

from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.smoke import run_smoke
from eval.smoke.run_smoke import (
    HYBRID_CONFLICT_RULE,
    HYBRID_EMPTY_SCAFFOLD,
    HYBRID_SECTION_1_MARKER,
    HYBRID_SECTION_2_MARKER,
    HYBRID_SECTION_3_MARKER,
    HYBRID_SECTION_4_MARKER,
    _build_claim_scaffold,
    _build_hybrid_prompt,
    _build_supersession_chains,
)


@pytest.fixture(autouse=True)
def _restore_active_pack():
    """Save and restore ACTIVE_PACK; the hybrid helpers read it for pack resolution."""
    original = os.environ.get("ACTIVE_PACK")
    os.environ["ACTIVE_PACK"] = "clinical_general"
    # Clear the lru_cache so active_pack() re-resolves.
    try:
        from src.substrate.predicate_packs import active_pack as _ap
        _ap.cache_clear()
    except ImportError:
        pass
    yield
    if original is None:
        os.environ.pop("ACTIVE_PACK", None)
    else:
        os.environ["ACTIVE_PACK"] = original
    try:
        from src.substrate.predicate_packs import active_pack as _ap2
        _ap2.cache_clear()
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Helpers — build a small in-memory substrate with deterministic claims.
# ---------------------------------------------------------------------------


def _make_populated_substrate(
    *,
    session_id: str = "enc_test_hybrid",
    with_supersession: bool = False,
) -> sqlite3.Connection:
    """Return an in-memory connection with a couple of claims wired in.

    When `with_supersession=True`, a PATIENT_CORRECTION edge is written
    between two claims (c02 superseded by c02b) so the scaffold can show
    the chain.
    """
    from src.substrate.claims import insert_claim, insert_turn
    from src.substrate.schema import (
        ClaimStatus,
        EdgeType,
        Speaker,
        Turn,
        open_database,
    )
    from src.substrate.supersession import write_supersession_edge

    conn = open_database(":memory:")
    # Turn 1 (patient utterance).
    insert_turn(
        conn,
        Turn(
            turn_id="tu_01",
            session_id=session_id,
            speaker=Speaker.PATIENT,
            text="My chest started hurting about two hours ago.",
            ts=1,
        ),
    )
    # Turn 2 (patient self-correction).
    insert_turn(
        conn,
        Turn(
            turn_id="tu_02",
            session_id=session_id,
            speaker=Speaker.PATIENT,
            text="Actually, it started about four hours ago, not two.",
            ts=2,
        ),
    )

    c1 = insert_claim(
        conn,
        session_id=session_id,
        subject="patient",
        predicate="onset",
        value="2 hours ago",
        confidence=0.9,
        source_turn_id="tu_01",
        claim_id="c02",
    )
    c2 = insert_claim(
        conn,
        session_id=session_id,
        subject="patient",
        predicate="severity",
        value="7/10",
        confidence=0.85,
        source_turn_id="tu_01",
        claim_id="c03",
    )

    if with_supersession:
        c1b = insert_claim(
            conn,
            session_id=session_id,
            subject="patient",
            predicate="onset",
            value="4 hours ago",
            confidence=0.92,
            source_turn_id="tu_02",
            claim_id="c02b",
        )
        write_supersession_edge(
            conn,
            old_claim_id=c1.claim_id,
            new_claim_id=c1b.claim_id,
            edge_type=EdgeType.PATIENT_CORRECTION,
            identity_score=None,
        )
        # Mark the old claim superseded so list_active_claims drops it.
        from src.substrate.claims import set_claim_status
        set_claim_status(conn, c1.claim_id, ClaimStatus.SUPERSEDED)

    return conn


# ---------------------------------------------------------------------------
# _build_hybrid_prompt — structural shape + conflict rule verbatim
# ---------------------------------------------------------------------------


class TestHybridPromptStructure:
    """The hybrid prompt must carry 4 explicit sections in order."""

    def test_all_four_section_markers_present(self) -> None:
        prompt = _build_hybrid_prompt(
            transcript_text="PATIENT: I have chest pain.",
            claim_scaffold="## SUBJECTIVE\n- [c:c01] patient / severity = 7/10",
        )
        assert HYBRID_SECTION_1_MARKER in prompt
        assert HYBRID_SECTION_2_MARKER in prompt
        assert HYBRID_SECTION_3_MARKER in prompt
        assert HYBRID_SECTION_4_MARKER in prompt

    def test_section_markers_are_in_order(self) -> None:
        prompt = _build_hybrid_prompt(
            transcript_text="PATIENT: I have chest pain.",
            claim_scaffold="(scaffold)",
        )
        i1 = prompt.index(HYBRID_SECTION_1_MARKER)
        i2 = prompt.index(HYBRID_SECTION_2_MARKER)
        i3 = prompt.index(HYBRID_SECTION_3_MARKER)
        i4 = prompt.index(HYBRID_SECTION_4_MARKER)
        assert i1 < i2 < i3 < i4, f"Sections out of order: {i1}, {i2}, {i3}, {i4}"

    def test_transcript_present_verbatim(self) -> None:
        transcript = (
            "DOCTOR: Where does it hurt?\nPATIENT: Center of my chest, 7/10."
        )
        prompt = _build_hybrid_prompt(
            transcript_text=transcript, claim_scaffold="(x)"
        )
        assert transcript in prompt

    def test_scaffold_present_verbatim(self) -> None:
        scaffold = (
            "## SUBJECTIVE\n"
            "- [c:c01] patient / severity = 7/10\n"
            "- [c:c02] patient / onset = 2 hours ago"
        )
        prompt = _build_hybrid_prompt(
            transcript_text="(transcript)", claim_scaffold=scaffold
        )
        assert scaffold in prompt

    def test_conflict_rule_present_verbatim(self) -> None:
        """The conflict-resolution rule must appear character-for-character.

        Audit requirement: operators must be able to grep `results.json`-
        adjacent artefacts for this exact string and prove the rule was
        on the reader's plate. Any drift breaks downstream audits.
        """
        prompt = _build_hybrid_prompt(
            transcript_text="(t)", claim_scaffold="(s)"
        )
        assert HYBRID_CONFLICT_RULE in prompt
        # Sanity — the rule actually mentions both signals.
        assert "substrate" in HYBRID_CONFLICT_RULE.lower()
        assert "transcript" in HYBRID_CONFLICT_RULE.lower()
        assert "supersession" in HYBRID_CONFLICT_RULE.lower()

    def test_task_instruction_tells_reader_to_produce_soap(self) -> None:
        prompt = _build_hybrid_prompt(
            transcript_text="(t)", claim_scaffold="(s)"
        )
        assert "SUBJECTIVE" in prompt
        assert "OBJECTIVE" in prompt
        assert "ASSESSMENT" in prompt
        assert "PLAN" in prompt


# ---------------------------------------------------------------------------
# _build_claim_scaffold — SOAP grouping + chain rendering
# ---------------------------------------------------------------------------


class TestClaimScaffoldShape:
    """The scaffold groups claims by SOAP section using the active pack's mapping."""

    def test_empty_substrate_returns_stable_placeholder(self) -> None:
        """Zero claims → a stable string that doesn't break the reader."""
        from src.substrate.schema import open_database

        conn = open_database(":memory:")
        scaffold = _build_claim_scaffold(conn, "enc_empty", "clinical_general")
        conn.close()
        assert scaffold == HYBRID_EMPTY_SCAFFOLD

    def test_grouped_by_soap_section_using_pack_mapping(self) -> None:
        conn = _make_populated_substrate()
        scaffold = _build_claim_scaffold(conn, "enc_test_hybrid", "clinical_general")
        conn.close()
        # Both `onset` and `severity` map to Subjective per clinical_general.
        assert "## SUBJECTIVE" in scaffold
        # Only one section should be present when all claims map to S.
        assert "## OBJECTIVE" not in scaffold

    def test_scaffold_includes_claim_id_provenance(self) -> None:
        conn = _make_populated_substrate()
        scaffold = _build_claim_scaffold(conn, "enc_test_hybrid", "clinical_general")
        conn.close()
        assert "[c:c02]" in scaffold
        assert "[c:c03]" in scaffold

    def test_scaffold_shows_supersession_chain_inline(self) -> None:
        """A superseded onset → new onset must show the resolution chain."""
        conn = _make_populated_substrate(with_supersession=True)
        scaffold = _build_claim_scaffold(conn, "enc_test_hybrid", "clinical_general")
        conn.close()

        # Only the resolved claim is active.
        assert "[c:c02b]" in scaffold
        # The chain shows old → new with edge_type and old value.
        assert "c02" in scaffold  # old claim id appears in chain
        assert "c02b" in scaffold  # new claim id appears in chain
        assert "patient_correction" in scaffold
        # Old value quoted so the reader sees what it used to say.
        assert "2 hours ago" in scaffold


class TestBuildSupersessionChains:
    """The chain helper returns `new_claim_id → [edge entries]`."""

    def test_no_supersession_returns_empty_dict(self) -> None:
        conn = _make_populated_substrate(with_supersession=False)
        chains = _build_supersession_chains(conn, "enc_test_hybrid")
        conn.close()
        assert chains == {}

    def test_supersession_chain_keys_by_new_claim_id(self) -> None:
        conn = _make_populated_substrate(with_supersession=True)
        chains = _build_supersession_chains(conn, "enc_test_hybrid")
        conn.close()
        assert "c02b" in chains
        entries = chains["c02b"]
        assert len(entries) == 1
        entry = entries[0]
        assert entry["old_claim_id"] == "c02"
        assert entry["edge_type"] == "patient_correction"
        assert entry["old_value"] == "2 hours ago"


# ---------------------------------------------------------------------------
# _call_acibench_substrate — end-to-end single-call hybrid integration
# ---------------------------------------------------------------------------


class TestHybridSingleCallIntegration:
    """The hybrid substrate path must make exactly one reader call per case."""

    def _stub_encounter(self) -> Any:
        class _StubEnc:
            encounter_id = "enc_hybrid_1"
            split = "aci"
            subsplit = "test1"
            dialogue = [
                {"speaker": "DOCTOR", "utterance": "What brings you in?"},
                {"speaker": "PATIENT", "utterance": "Chest pain for 2 hours."},
                {"speaker": "PATIENT", "utterance": "Actually more like 4 hours."},
            ]
            gold_note = "SUBJECTIVE: chest pain 4h. OBJECTIVE: unremarkable."

        return _StubEnc()

    def test_single_reader_call_and_no_two_step_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Hybrid must call the reader exactly once (not twice like the 2-step)."""
        enc = self._stub_encounter()

        reader_calls: list[tuple[str, str]] = []

        def _mock_qwen(system: str, user: str, _env: dict) -> tuple[str, int]:
            reader_calls.append((system, user))
            return "SUBJECTIVE: chest pain, 4h onset. OBJECTIVE: none.", 123

        monkeypatch.setattr(run_smoke, "_call_qwen", _mock_qwen)

        # Stub extractor so we don't hit the LLM extractor either.
        def _stub_extractor_factory() -> Any:
            def _noop(_turn: Any) -> list:
                return []
            return _noop

        monkeypatch.setattr(
            "src.extraction.claim_extractor.extractor.make_llm_extractor",
            _stub_extractor_factory,
        )

        note, latency_ms, tokens, stats = run_smoke._call_acibench_substrate(
            enc,
            reader="qwen2.5-14b",
            reader_env={"endpoint": "http://mock"},
            baseline_note_for_edit_distance=None,
            hybrid=True,
        )

        # Exactly one reader call.
        assert len(reader_calls) == 1, (
            f"hybrid must make 1 reader call, got {len(reader_calls)}"
        )

        system_prompt, user_prompt = reader_calls[0]
        # All 4 section markers present in the user prompt.
        assert HYBRID_SECTION_1_MARKER in user_prompt
        assert HYBRID_SECTION_2_MARKER in user_prompt
        assert HYBRID_SECTION_3_MARKER in user_prompt
        assert HYBRID_SECTION_4_MARKER in user_prompt
        # Conflict rule verbatim.
        assert HYBRID_CONFLICT_RULE in user_prompt
        # Empty scaffold placeholder (noop extractor → zero claims).
        assert HYBRID_EMPTY_SCAFFOLD in user_prompt

        assert note.startswith("SUBJECTIVE")
        assert tokens == 123
        # Latency is wall-clock on a mocked path; >=0 is the right invariant.
        # On very fast Windows clocks this can round to exactly 0.0.
        assert latency_ms >= 0
        assert stats.active_pack == "clinical_general"

    def test_no_hybrid_raises_not_implemented(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`hybrid=False` must raise — the 2-step path is intentionally dead."""
        enc = self._stub_encounter()
        monkeypatch.setattr(run_smoke, "_call_qwen", lambda *_, **__: ("x", 1))

        with pytest.raises(NotImplementedError):
            run_smoke._call_acibench_substrate(
                enc,
                reader="qwen2.5-14b",
                reader_env={"endpoint": "http://mock"},
                hybrid=False,
            )

    def test_edit_distance_populated_when_baseline_supplied(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When baseline note is passed, stats gain substrate_vs_baseline_edit_distance."""
        enc = self._stub_encounter()

        monkeypatch.setattr(
            run_smoke, "_call_qwen", lambda *_, **__: ("different note text", 10)
        )

        def _stub_extractor_factory() -> Any:
            def _noop(_turn: Any) -> list:
                return []
            return _noop

        monkeypatch.setattr(
            "src.extraction.claim_extractor.extractor.make_llm_extractor",
            _stub_extractor_factory,
        )

        _note, _lat, _tok, stats = run_smoke._call_acibench_substrate(
            enc,
            reader="qwen2.5-14b",
            reader_env={"endpoint": "http://mock"},
            baseline_note_for_edit_distance="completely unrelated baseline note",
            hybrid=True,
        )
        # Non-None and strictly positive because the strings differ.
        assert stats.substrate_vs_baseline_edit_distance is not None
        assert stats.substrate_vs_baseline_edit_distance > 0.0
