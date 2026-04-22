"""Unit tests for `src/note/generator.py`.

Cover:
- Empty active claims → returns zero-claim result (no crash).
- Claims grouped correctly by predicate using `soap_mapping.json`.
- `[c:…]` provenance tag parsing — valid + unknown IDs + untagged sentences.
- `sentence_with_provenance_ratio` metric correctness.
- Error handling: LLM failure returns a structured empty SOAPResult.
"""
from __future__ import annotations

import sqlite3
from types import SimpleNamespace

import pytest

from src.note.generator import (
    SOAPResult,
    _parse_note_with_provenance,
    generate_soap_note,
)
from src.substrate.claims import insert_claim, insert_turn
from src.substrate.predicate_packs import active_pack
from src.substrate.schema import Speaker, Turn, open_database


@pytest.fixture
def _clinical_pack(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the clinical_general pack for each test."""
    monkeypatch.setenv("ACTIVE_PACK", "clinical_general")
    active_pack.cache_clear()
    yield
    active_pack.cache_clear()


def _seed_db(
    conn: sqlite3.Connection,
    session_id: str,
    claim_specs: list[tuple[str, str, str, float]],
) -> list[str]:
    """Insert one turn + one claim per spec. Return the claim_ids in order."""
    turn_id = "turn-1"
    insert_turn(
        conn,
        Turn(
            turn_id=turn_id,
            session_id=session_id,
            speaker=Speaker.PATIENT,
            text="I have chest pain and shortness of breath.",
            ts=1_000_000,
        ),
    )
    claim_ids: list[str] = []
    for i, (subject, predicate, value, confidence) in enumerate(claim_specs):
        claim = insert_claim(
            conn,
            session_id=session_id,
            subject=subject,
            predicate=predicate,
            value=value,
            confidence=confidence,
            source_turn_id=turn_id,
            claim_id=f"c{i}",
        )
        claim_ids.append(claim.claim_id)
    return claim_ids


def _fake_response(content: str, prompt_tokens: int = 100, completion_tokens: int = 50) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        ),
    )


def _fake_client(content: str) -> SimpleNamespace:
    """Single-response client — both draft and annotation calls get the same string.

    For tests that don't care about the annotation payload; the annotation
    parser will treat non-JSON as a failure and return empty provenance,
    which is fine for these tests.
    """
    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **_: _fake_response(content)
            )
        )
    )


def _fake_client_seq(draft_text: str, annotation_json: str) -> SimpleNamespace:
    """P0 flow: first call returns the draft, second returns annotation JSON.

    Uses a mutable counter to dispatch based on call order. When more than
    two calls happen, falls back to the annotation response.
    """
    call_count = [0]

    def _create(**_: object) -> SimpleNamespace:
        i = call_count[0]
        call_count[0] += 1
        if i == 0:
            return _fake_response(draft_text)
        return _fake_response(annotation_json)

    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=_create)
        )
    )


# ── _parse_note_with_provenance (pure function) ──────────────────────────
class TestParseNoteWithProvenance:
    def test_tags_stripped_from_output(self) -> None:
        raw = "The patient reports chest pain [c:c1]."
        known = [
            SimpleNamespace(
                claim_id="c1",
                subject="",
                predicate="",
                value="",
                confidence=0.0,
            )
        ]
        note, prov = _parse_note_with_provenance(raw, known)  # type: ignore[arg-type]
        assert "[c:c1]" not in note
        assert note == "The patient reports chest pain ."
        assert len(prov) == 1
        assert prov[0][1] == ("c1",)

    def test_unknown_claim_ids_dropped(self) -> None:
        raw = "The patient reports chest pain [c:c1][c:hallucinated_id]."
        known = [
            SimpleNamespace(
                claim_id="c1",
                subject="",
                predicate="",
                value="",
                confidence=0.0,
            )
        ]
        _note, prov = _parse_note_with_provenance(raw, known)  # type: ignore[arg-type]
        assert len(prov) == 1
        assert prov[0][1] == ("c1",)  # hallucinated ID dropped

    def test_sentence_without_tags_kept(self) -> None:
        raw = "SUBJECTIVE:\nThe patient reports chest pain [c:c1].\nAssessment is pending."
        known = [SimpleNamespace(claim_id="c1", subject="", predicate="", value="", confidence=0.0)]
        _note, prov = _parse_note_with_provenance(raw, known)  # type: ignore[arg-type]
        # Two sentences captured; only first has provenance.
        texts = [s for s, _ in prov]
        id_groups = [ids for _, ids in prov]
        assert "The patient reports chest pain ." in texts
        assert "Assessment is pending." in texts
        assert ("c1",) in id_groups
        assert () in id_groups

    def test_section_headings_preserved(self) -> None:
        raw = "SUBJECTIVE:\nPatient reports chest pain [c:c1].\nOBJECTIVE:\nVitals stable."
        known = [SimpleNamespace(claim_id="c1", subject="", predicate="", value="", confidence=0.0)]
        note, _ = _parse_note_with_provenance(raw, known)  # type: ignore[arg-type]
        assert "SUBJECTIVE:" in note
        assert "OBJECTIVE:" in note

    def test_multiple_tags_per_sentence(self) -> None:
        raw = "Chest pain radiating to left arm [c:c1][c:c2]."
        known = [
            SimpleNamespace(claim_id="c1", subject="", predicate="", value="", confidence=0.0),
            SimpleNamespace(claim_id="c2", subject="", predicate="", value="", confidence=0.0),
        ]
        _note, prov = _parse_note_with_provenance(raw, known)  # type: ignore[arg-type]
        assert len(prov) == 1
        assert set(prov[0][1]) == {"c1", "c2"}

    def test_empty_input(self) -> None:
        note, prov = _parse_note_with_provenance("", [])
        assert note == ""
        assert prov == ()


# ── generate_soap_note (integration with real DB + mocked LLM) ───────────
class TestGenerateSoapNote:
    def test_empty_claims_returns_sensible_result(self, _clinical_pack: None) -> None:
        conn = open_database(":memory:")
        client = _fake_client("SUBJECTIVE:\nNo claims available.\nOBJECTIVE:\nNo claims available.")
        result = generate_soap_note(
            conn,
            session_id="s1",
            dialogue_text="",
            reader="gpt-4.1-mini",
            reader_env={},
            client=client,
        )
        assert isinstance(result, SOAPResult)
        assert result.active_claim_count == 0
        assert "SUBJECTIVE:" in result.note_text
        assert result.sentence_with_provenance_ratio == 0.0  # no claims = no tags
        conn.close()

    def test_active_claim_count_matches_db(self, _clinical_pack: None) -> None:
        conn = open_database(":memory:")
        _seed_db(
            conn,
            "s1",
            [
                ("chest_pain", "severity", "7/10", 0.9),
                ("chest_pain", "onset", "2 hours ago", 0.9),
            ],
        )
        # P0: draft end-to-end, then annotate. Mock both calls.
        draft = (
            "SUBJECTIVE:\n"
            "Patient reports chest pain at 7/10 severity.\n"
            "Onset was 2 hours ago.\n"
        )
        annotation = (
            '{"annotations": ['
            '{"sentence_index": 0, "claim_ids": []},'
            '{"sentence_index": 1, "claim_ids": ["c0"]},'
            '{"sentence_index": 2, "claim_ids": ["c1"]}'
            ']}'
        )
        client = _fake_client_seq(draft, annotation)
        result = generate_soap_note(
            conn,
            session_id="s1",
            dialogue_text="dialogue",
            reader="gpt-4.1-mini",
            reader_env={},
            client=client,
        )
        assert result.active_claim_count == 2
        all_ids = {cid for _, ids in result.sentence_provenance for cid in ids}
        assert all_ids == {"c0", "c1"}
        assert result.sentence_with_provenance_ratio > 0
        conn.close()

    def test_provenance_ratio_reflects_untagged_sentences(
        self, _clinical_pack: None
    ) -> None:
        conn = open_database(":memory:")
        _seed_db(conn, "s1", [("chest_pain", "severity", "7/10", 0.9)])
        # P0 flow: draft has 2 sentences, annotation tags only the first.
        draft = "Patient reports chest pain. Plan is to observe."
        annotation = (
            '{"annotations": ['
            '{"sentence_index": 0, "claim_ids": ["c0"]},'
            '{"sentence_index": 1, "claim_ids": []}'
            ']}'
        )
        client = _fake_client_seq(draft, annotation)
        result = generate_soap_note(
            conn,
            session_id="s1",
            dialogue_text="dialogue",
            reader="gpt-4.1-mini",
            reader_env={},
            client=client,
        )
        assert result.active_claim_count == 1
        # 2 sentences, 1 with provenance → 0.5
        assert result.sentence_with_provenance_ratio == pytest.approx(0.5, abs=0.01)
        conn.close()

    def test_hallucinated_claim_id_dropped(self, _clinical_pack: None) -> None:
        conn = open_database(":memory:")
        _seed_db(conn, "s1", [("chest_pain", "severity", "7/10", 0.9)])
        # P0: annotation returns one real ID and one hallucinated ID.
        draft = "Patient reports chest pain."
        annotation = (
            '{"annotations": ['
            '{"sentence_index": 0, "claim_ids": ["c0", "c99_fake"]}'
            ']}'
        )
        client = _fake_client_seq(draft, annotation)
        result = generate_soap_note(
            conn,
            session_id="s1",
            dialogue_text="dialogue",
            reader="gpt-4.1-mini",
            reader_env={},
            client=client,
        )
        all_ids = {cid for _, ids in result.sentence_provenance for cid in ids}
        assert all_ids == {"c0"}  # hallucinated ID excluded
        conn.close()

    def test_api_error_returns_empty_result(self, _clinical_pack: None) -> None:
        conn = open_database(":memory:")
        _seed_db(conn, "s1", [("chest_pain", "severity", "7/10", 0.9)])

        def _raise(**_: object) -> object:
            raise RuntimeError("boom")

        bad_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=_raise))
        )
        result = generate_soap_note(
            conn,
            session_id="s1",
            dialogue_text="dialogue",
            reader="gpt-4.1-mini",
            reader_env={},
            client=bad_client,
        )
        assert result.note_text == ""
        assert result.sentence_provenance == ()
        assert result.active_claim_count == 1  # DB still queried successfully
        assert result.tokens_used == 0
        conn.close()

    def test_tokens_used_extracted(self, _clinical_pack: None) -> None:
        conn = open_database(":memory:")
        client = _fake_client("SUBJECTIVE:\nNone.")
        result = generate_soap_note(
            conn,
            session_id="s1",
            dialogue_text="",
            reader="gpt-4.1-mini",
            reader_env={},
            client=client,
        )
        assert result.tokens_used == 150  # 100 prompt + 50 completion per _fake_response
        conn.close()


# ── sentence_with_provenance_ratio property ──────────────────────────────
class TestRatioProperty:
    def test_zero_when_no_sentences(self) -> None:
        r = SOAPResult(
            note_text="",
            sentence_provenance=(),
            active_claim_count=0,
            tokens_used=0,
            latency_ms=0.0,
        )
        assert r.sentence_with_provenance_ratio == 0.0

    def test_one_when_all_tagged(self) -> None:
        r = SOAPResult(
            note_text="",
            sentence_provenance=(("a", ("c1",)), ("b", ("c2",))),
            active_claim_count=2,
            tokens_used=0,
            latency_ms=0.0,
        )
        assert r.sentence_with_provenance_ratio == 1.0

    def test_mixed(self) -> None:
        r = SOAPResult(
            note_text="",
            sentence_provenance=(("a", ("c1",)), ("b", ())),
            active_claim_count=1,
            tokens_used=0,
            latency_ms=0.0,
        )
        assert r.sentence_with_provenance_ratio == pytest.approx(0.5, abs=0.01)
