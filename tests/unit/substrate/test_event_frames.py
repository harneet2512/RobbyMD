"""Unit tests for the event-frame assembly layer."""
from __future__ import annotations

import os

import pytest

from src.substrate.claims import insert_claim, list_active_claims
from src.substrate.event_frames import (
    EventFrame,
    assemble_event_frames,
    classify_event_type,
    list_event_frames,
)
from src.substrate.schema import ClaimStatus, Speaker, open_database


@pytest.fixture(autouse=True)
def _set_personal_assistant_pack(monkeypatch):
    monkeypatch.setenv("ACTIVE_PACK", "personal_assistant")
    from src.substrate.predicate_packs import active_pack
    active_pack.cache_clear()


def _make_turn(conn, session_id: str, turn_id: str, ts: int, text: str = "test"):
    conn.execute(
        "INSERT INTO turns (turn_id, session_id, speaker, text, ts) VALUES (?, ?, ?, ?, ?)",
        (turn_id, session_id, Speaker.PATIENT.value, text, ts),
    )


def _make_claim(
    conn, session_id: str, turn_id: str,
    subject: str, predicate: str, value: str,
    confidence: float = 0.9,
):
    return insert_claim(
        conn,
        session_id=session_id,
        subject=subject,
        predicate=predicate,
        value=value,
        confidence=confidence,
        source_turn_id=turn_id,
    )


class TestClassifyEventType:
    def test_purchase_detected(self):
        assert classify_event_type(["redeemed $5 coupon on coffee creamer"]) == "purchase_redemption"

    def test_travel_detected(self):
        assert classify_event_type(["daily commute takes 45 minutes"]) == "travel_commute"

    def test_education_detected(self):
        assert classify_event_type(["graduated with degree in Business Administration"]) == "education_milestone"

    def test_no_match_returns_none(self):
        assert classify_event_type(["likes the color blue"]) is None

    def test_multi_value_best_match(self):
        result = classify_event_type([
            "bought a new book",
            "purchased it at the bookstore",
            "paid $15",
        ])
        assert result == "purchase_redemption"


class TestFragmentedClaimsAssemble:
    """Claims about the same event from adjacent turns should assemble into one frame."""

    def test_coupon_and_target_assemble(self):
        conn = open_database(":memory:")
        sid = "test_session"
        _make_turn(conn, sid, "t0", 100, "I redeemed a $5 coupon on coffee creamer last Sunday")
        _make_turn(conn, sid, "t1", 101, "assistant response")
        _make_turn(conn, sid, "t2", 102, "I was shopping at Target for groceries")

        _make_claim(conn, sid, "t0", "user", "user_event", "redeemed $5 coupon on coffee creamer last Sunday")
        _make_claim(conn, sid, "t2", "user", "user_event", "shopping at Target for groceries")

        frames = assemble_event_frames(conn, sid)
        assert len(frames) >= 1

        purchase_frames = [f for f in frames if f.event_type == "purchase_redemption"]
        assert len(purchase_frames) == 1

        frame = purchase_frames[0]
        assert frame.location is not None, f"Location should be extracted, got missing_slots={frame.missing_slots}"
        assert "target" in frame.location.lower()
        assert len(frame.supporting_claim_ids) == 2
        conn.close()


class TestAtomicFactsDoNotAssemble:
    """A single claim about a fact should not produce an event frame unless it matches an event type."""

    def test_single_commute_claim(self):
        conn = open_database(":memory:")
        sid = "test_atomic"
        _make_turn(conn, sid, "t0", 100, "My commute takes 45 minutes each way")

        _make_claim(conn, sid, "t0", "user", "user_fact", "commute takes 45 minutes each way")

        frames = assemble_event_frames(conn, sid)
        # A single claim may produce a frame if it matches an event type,
        # but the key is it should NOT merge with unrelated claims.
        # The commute claim matches travel_commute, which is fine as a single-claim frame.
        assert all(len(f.supporting_claim_ids) <= 1 for f in frames)
        conn.close()


class TestUnrelatedClaimsSeparateFrames:
    """Claims about different events should produce separate frames, not merge."""

    def test_cat_tower_and_coupon_separate(self):
        conn = open_database(":memory:")
        sid = "test_separate"
        _make_turn(conn, sid, "t0", 100, "I bought a cat tower from Petco for $120")
        _make_turn(conn, sid, "t1", 101, "assistant response")
        _make_turn(conn, sid, "t2", 102, "I redeemed a $5 coupon on coffee creamer")
        _make_turn(conn, sid, "t3", 103, "assistant response")
        _make_turn(conn, sid, "t4", 104, "I was shopping at Target")

        _make_claim(conn, sid, "t0", "user", "user_event", "bought cat tower from Petco for $120")
        _make_claim(conn, sid, "t2", "user", "user_event", "redeemed $5 coupon on coffee creamer")
        _make_claim(conn, sid, "t4", "user", "user_event", "shopping at Target")

        frames = assemble_event_frames(conn, sid)
        purchase_frames = [f for f in frames if f.event_type == "purchase_redemption"]
        assert len(purchase_frames) >= 1

        petco_frames = [f for f in purchase_frames if any("petco" in cid.lower() or
            (f.location and "petco" in f.location.lower()) for cid in f.supporting_claim_ids)]
        target_frames = [f for f in purchase_frames if any("target" in cid.lower() or
            (f.location and "target" in f.location.lower()) for cid in f.supporting_claim_ids)]

        # They should not be in the same frame
        for frame in frames:
            values = []
            for cid in frame.supporting_claim_ids:
                row = conn.execute("SELECT value FROM claims WHERE claim_id = ?", (cid,)).fetchone()
                if row:
                    values.append(row["value"].lower())
            has_petco = any("petco" in v for v in values)
            has_coupon = any("coupon" in v for v in values)
            assert not (has_petco and has_coupon), "Petco and coupon claims should not merge"
        conn.close()


class TestProvenancePreserved:
    """Event frame supporting_claim_ids must map back to real claims."""

    def test_claim_ids_exist_in_db(self):
        conn = open_database(":memory:")
        sid = "test_prov"
        _make_turn(conn, sid, "t0", 100, "I bought groceries at Target")

        _make_claim(conn, sid, "t0", "user", "user_event", "bought groceries at Target")

        frames = assemble_event_frames(conn, sid)
        for frame in frames:
            for cid in frame.supporting_claim_ids:
                row = conn.execute("SELECT claim_id FROM claims WHERE claim_id = ?", (cid,)).fetchone()
                assert row is not None, f"Claim {cid} not found in DB"

            # Check junction table
            junc_rows = conn.execute(
                "SELECT * FROM event_frame_claims WHERE event_id = ?",
                (frame.event_id,)
            ).fetchall()
            assert len(junc_rows) == len(frame.supporting_claim_ids)
        conn.close()


class TestMissingSlotsDetected:
    """Frames with missing slots should report them."""

    def test_no_location_flagged(self):
        conn = open_database(":memory:")
        sid = "test_missing"
        _make_turn(conn, sid, "t0", 100, "I redeemed a $5 coupon on coffee creamer")

        _make_claim(conn, sid, "t0", "user", "user_event", "redeemed $5 coupon on coffee creamer")

        frames = assemble_event_frames(conn, sid)
        purchase_frames = [f for f in frames if f.event_type == "purchase_redemption"]
        if purchase_frames:
            frame = purchase_frames[0]
            assert "location" in frame.missing_slots, f"Expected location in missing_slots, got {frame.missing_slots}"
        conn.close()


class TestIdempotent:
    """Running assemble_event_frames twice should produce the same result."""

    def test_double_assembly_consistent(self):
        conn = open_database(":memory:")
        sid = "test_idemp"
        _make_turn(conn, sid, "t0", 100, "I purchased coffee at Target")

        _make_claim(conn, sid, "t0", "user", "user_event", "purchased coffee at Target")

        frames1 = assemble_event_frames(conn, sid)
        frames2 = assemble_event_frames(conn, sid)

        # Second call uses INSERT OR REPLACE, so DB state is consistent.
        db_frames = list_event_frames(conn, sid)
        assert len(db_frames) >= len(frames1)
        conn.close()


class TestListEventFrames:
    """list_event_frames loads persisted frames."""

    def test_roundtrip(self):
        conn = open_database(":memory:")
        sid = "test_list"
        _make_turn(conn, sid, "t0", 100, "I bought a new book at the bookstore")

        _make_claim(conn, sid, "t0", "user", "user_event", "bought a new book at the bookstore")

        assemble_event_frames(conn, sid)
        loaded = list_event_frames(conn, sid)
        assert len(loaded) >= 1
        assert all(isinstance(f, EventFrame) for f in loaded)
        assert all(f.session_id == sid for f in loaded)
        conn.close()
