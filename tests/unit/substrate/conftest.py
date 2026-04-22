"""Shared fixtures for the substrate unit tests.

Each test gets a fresh in-memory SQLite DB so they are hermetic and can
run in any order. We also pre-seed a session id + a small helper that
inserts a turn and returns its id, saving boilerplate in the actual
tests.
"""
from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from typing import Protocol

import pytest

from src.substrate import open_database
from src.substrate.claims import insert_turn, new_turn_id, now_ns
from src.substrate.schema import Speaker, Turn


class AddTurnFn(Protocol):
    """Fixture signature: `(text, speaker=PATIENT) -> turn_id`."""

    def __call__(self, text: str, speaker: Speaker = ...) -> str: ...


@pytest.fixture()
def conn() -> Iterator[sqlite3.Connection]:
    """Fresh in-memory SQLite with the substrate schema applied."""
    c = open_database(":memory:")
    try:
        yield c
    finally:
        c.close()


@pytest.fixture()
def session_id() -> str:
    return "sess_test_01"


@pytest.fixture()
def add_turn(conn: sqlite3.Connection, session_id: str) -> AddTurnFn:
    """Factory: add a turn and return its id."""

    def _add(text: str, speaker: Speaker = Speaker.PATIENT) -> str:
        tid = new_turn_id()
        turn = Turn(
            turn_id=tid,
            session_id=session_id,
            speaker=speaker,
            text=text,
            ts=now_ns(),
        )
        insert_turn(conn, turn)
        return tid

    return _add
