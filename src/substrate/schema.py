"""SQLite schema + typed data model for the substrate.

Implements `Eng_doc.md` §4.1 tables with two upgrades from
`docs/research_brief.md` §2.1 and `docs/gt_v2_study_notes.md` §2.1:

1.  **Typed supersession edges** — `supersession_edges.edge_type` is
    `NOT NULL` with a `CHECK` constraint over the seven clinical
    edge kinds (see `EdgeType`).
2.  **Span-based provenance** — the `claims` table carries
    `(source_turn_id, char_start, char_end)` so the UI can highlight
    the exact transcript substring that sourced the claim
    (`PRD.md` §6.2 + §6.4 provenance-is-the-hero demo beat).

PRAGMAs follow GT v2 study notes §2.1: WAL mode, `synchronous=NORMAL`,
`busy_timeout=5000`, `cache_size=-20000`, foreign keys on.

Nothing in this module talks to an LLM or an embedder. Pure storage.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

import structlog

log = structlog.get_logger(__name__)


# ------------------------------------------------------------------ enums ---


class Speaker(StrEnum):
    """Who is speaking in a turn. Matches `turns.speaker` CHECK constraint."""

    PATIENT = "patient"
    PHYSICIAN = "physician"
    SYSTEM = "system"


class ClaimStatus(StrEnum):
    """Lifecycle state. Matches `claims.status` CHECK constraint."""

    ACTIVE = "active"
    SUPERSEDED = "superseded"
    CONFIRMED = "confirmed"
    DISMISSED = "dismissed"


class EdgeType(StrEnum):
    """Typed supersession edge vocabulary (research_brief §2.1).

    - `PATIENT_CORRECTION` — patient restates with a different value (Pass 1).
    - `PHYSICIAN_CONFIRM` — physician confirms a prior patient claim (Pass 1).
    - `SEMANTIC_REPLACE` — same identity under cosine ≥ 0.88 (Pass 2).
    - `REFINES` — new value is a narrower subset of the old one (Pass 1).
    - `CONTRADICTS` — later speaker refutes an earlier value (Pass 1).
    - `RULES_OUT` — tied to a differential branch becoming dead (verifier).
    - `DISMISSED_BY_CLINICIAN` — physician tap in the UI (decisions table).
    """

    PATIENT_CORRECTION = "patient_correction"
    PHYSICIAN_CONFIRM = "physician_confirm"
    SEMANTIC_REPLACE = "semantic_replace"
    REFINES = "refines"
    CONTRADICTS = "contradicts"
    RULES_OUT = "rules_out"
    DISMISSED_BY_CLINICIAN = "dismissed_by_clinician"


class NoteSection(StrEnum):
    """SOAP section for `note_sentences.section`."""

    SUBJECTIVE = "S"
    OBJECTIVE = "O"
    ASSESSMENT = "A"
    PLAN = "P"


# The closed 14-predicate chest-pain vocabulary (Eng_doc.md §4.2).
# Anything outside is rejected by claim-extraction validation.
PREDICATE_FAMILIES: Final[frozenset[str]] = frozenset({
    "onset",
    "character",
    "severity",
    "location",
    "radiation",
    "aggravating_factor",
    "alleviating_factor",
    "associated_symptom",
    "duration",
    "medical_history",
    "medication",
    "family_history",
    "social_history",
    "risk_factor",
})


# ------------------------------------------------------------ dataclasses ---


@dataclass(frozen=True, slots=True)
class Turn:
    """One conversation turn (`Eng_doc.md` §4.1 `turns`)."""

    turn_id: str
    session_id: str
    speaker: Speaker
    text: str
    ts: int
    asr_confidence: float | None = None


@dataclass(frozen=True, slots=True)
class Claim:
    """One claim (`Eng_doc.md` §4.1 `claims` + span + temporal-validity upgrades).

    `char_start` / `char_end` index into the source turn's `text` and bound
    the substring the extractor believes produced this claim. Both `None`
    means the extractor could not localise the span (still admissible;
    the UI falls back to full-turn highlight).

    `valid_from_ts` / `valid_until_ts` carry the temporal-validity window —
    aligned with Zep (arXiv:2501.13956 — valid_from / valid_until on KG edges)
    and Chronos (arXiv:2603.16862 — start_datetime / end_datetime on event
    tuples). Default behaviour: `valid_from_ts == created_ts`,
    `valid_until_ts == None` (unbounded). Pass-1 supersession sets
    `valid_until_ts` on the superseded claim to the supersession edge's
    `created_ts`. RobbyMD's contribution vs Chronos: deterministic algorithmic
    supersession (vs LLM-based revision).
    """

    claim_id: str
    session_id: str
    subject: str
    predicate: str
    value: str
    value_normalised: str | None
    confidence: float
    source_turn_id: str
    status: ClaimStatus
    created_ts: int
    char_start: int | None = None
    char_end: int | None = None
    valid_from_ts: int | None = None
    valid_until_ts: int | None = None


@dataclass(frozen=True, slots=True)
class SupersessionEdge:
    """Lifecycle edge between two claims.

    `identity_score` is the Pass-2 cosine similarity when `edge_type ==
    SEMANTIC_REPLACE`; `None` for the deterministic Pass-1 kinds.
    """

    edge_id: str
    old_claim_id: str
    new_claim_id: str
    edge_type: EdgeType
    identity_score: float | None
    created_ts: int


@dataclass(frozen=True, slots=True)
class NoteSentence:
    """One rendered SOAP-note sentence (`Eng_doc.md` §4.1 `note_sentences`)."""

    sentence_id: str
    session_id: str
    section: NoteSection
    ordinal: int
    text: str
    source_claim_ids: tuple[str, ...]


# ----------------------------------------------------------------- schema ---


_SCHEMA_SQL: Final[str] = """
CREATE TABLE IF NOT EXISTS turns (
    turn_id         TEXT PRIMARY KEY,
    session_id      TEXT NOT NULL,
    speaker         TEXT NOT NULL CHECK (speaker IN ('patient','physician','system')),
    text            TEXT NOT NULL,
    ts              INTEGER NOT NULL,
    asr_confidence  REAL
);
CREATE INDEX IF NOT EXISTS idx_turns_session_ts ON turns(session_id, ts);

CREATE TABLE IF NOT EXISTS claims (
    claim_id          TEXT PRIMARY KEY,
    session_id        TEXT NOT NULL,
    subject           TEXT NOT NULL,
    predicate         TEXT NOT NULL,
    value             TEXT NOT NULL,
    value_normalised  TEXT,
    confidence        REAL NOT NULL,
    source_turn_id    TEXT NOT NULL REFERENCES turns(turn_id) ON DELETE CASCADE,
    status            TEXT NOT NULL CHECK (
                          status IN ('active','superseded','confirmed','dismissed')
                      ),
    created_ts        INTEGER NOT NULL,
    char_start        INTEGER,
    char_end          INTEGER,
    -- Temporal validity windows. Aligned with Zep (arXiv:2501.13956 —
    -- valid_from / valid_until on KG edges) and Chronos (arXiv:2603.16862
    -- — start_datetime / end_datetime on event tuples). RobbyMD's contribution:
    -- deterministic Pass-1 supersession algorithmically sets valid_until_ts
    -- on the superseded claim (vs LLM-based revision in Chronos).
    valid_from_ts     INTEGER,
    valid_until_ts    INTEGER,
    CHECK (
        valid_until_ts IS NULL
        OR valid_from_ts IS NULL
        OR valid_until_ts > valid_from_ts
    )
);
CREATE INDEX IF NOT EXISTS idx_claims_active
    ON claims(session_id, subject, predicate, status);
CREATE INDEX IF NOT EXISTS idx_claims_temporal
    ON claims(valid_from_ts, valid_until_ts);

CREATE TABLE IF NOT EXISTS supersession_edges (
    edge_id         TEXT PRIMARY KEY,
    old_claim_id    TEXT NOT NULL REFERENCES claims(claim_id) ON DELETE CASCADE,
    new_claim_id    TEXT NOT NULL REFERENCES claims(claim_id) ON DELETE CASCADE,
    edge_type       TEXT NOT NULL CHECK (edge_type IN (
                        'patient_correction','physician_confirm','semantic_replace',
                        'refines','contradicts','rules_out','dismissed_by_clinician'
                    )),
    identity_score  REAL,
    created_ts      INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_supersession_old ON supersession_edges(old_claim_id);
CREATE INDEX IF NOT EXISTS idx_supersession_new ON supersession_edges(new_claim_id);

CREATE TABLE IF NOT EXISTS decisions (
    decision_id           TEXT PRIMARY KEY,
    session_id            TEXT NOT NULL,
    kind                  TEXT NOT NULL,
    target_type           TEXT NOT NULL,
    target_id             TEXT NOT NULL,
    claim_state_snapshot  TEXT NOT NULL,
    ts                    INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS note_sentences (
    sentence_id       TEXT PRIMARY KEY,
    session_id        TEXT NOT NULL,
    section           TEXT NOT NULL CHECK (section IN ('S','O','A','P')),
    ordinal           INTEGER NOT NULL,
    text              TEXT NOT NULL,
    source_claim_ids  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_note_session_section
    ON note_sentences(session_id, section, ordinal);

-- Sidecar table for per-claim retrieval embeddings (Stream A, feature/lme-retrieval).
-- Kept separate from `claims` so we can re-embed without rewriting the main table
-- and so installations that never use retrieval pay no storage cost beyond the
-- empty table. FK with ON DELETE CASCADE keeps the sidecar in step with any
-- claim-level deletions (e.g. per-case :memory: DBs are discarded wholesale).
CREATE TABLE IF NOT EXISTS claim_embeddings (
    claim_id                TEXT PRIMARY KEY REFERENCES claims(claim_id) ON DELETE CASCADE,
    embedding               BLOB NOT NULL,
    embedding_model_version TEXT NOT NULL,
    embedded_at_unix        INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_claim_embeddings_model
    ON claim_embeddings(embedding_model_version);
"""


def open_database(path: str | Path) -> sqlite3.Connection:
    """Open (or create) a SQLite DB, apply PRAGMAs, and initialise the schema.

    Idempotent — safe to call against an existing DB. `path` may be
    `":memory:"` for unit tests.
    """
    is_memory = str(path) == ":memory:"
    conn = sqlite3.connect(str(path), isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row

    # PRAGMAs per gt_v2_study_notes §2.1. WAL only makes sense on-disk.
    if not is_memory:
        conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA cache_size=-20000")
    conn.execute("PRAGMA foreign_keys=ON")

    conn.executescript(_SCHEMA_SQL)
    log.debug("substrate.db_opened", path=str(path))
    return conn
