"""Supersession Pass 1 — deterministic structural matcher.

Per `Eng_doc.md` §5.3 Pass 1 + upgrade to typed edges
(`docs/research_brief.md` §2.1):

Match: same `(session_id, normalised_subject, predicate)`, different `value`.

Discriminate edge_type from speaker + context (new claim's source turn):

- physician speaker on the *new* claim → `PHYSICIAN_CONFIRM`
- patient speaker on the *new* claim and the old claim was also patient →
  `PATIENT_CORRECTION` (self-correction)
- patient speaker on the *new* claim but old claim was physician →
  `CONTRADICTS`
- if the new value is a strict token-subset of the old value → `REFINES`

No LLM, no randomness (rules.md §5.2). Guard (`Eng_doc.md` §5.3): supersession
never fires within the same turn — claims emitted together from one utterance
are additive, not replacing.
"""
from __future__ import annotations

import sqlite3
import uuid

import structlog

from src.substrate.claims import now_ns, set_claim_status
from src.substrate.schema import Claim, ClaimStatus, EdgeType, Speaker, SupersessionEdge

log = structlog.get_logger(__name__)


def _new_edge_id() -> str:
    return f"ed_{uuid.uuid4().hex[:12]}"


def _tokens(value: str) -> set[str]:
    import re

    return {t for t in re.split(r"[\s,;/]+", value.lower().strip()) if t}


def _classify_edge(
    *,
    old_claim: Claim,
    new_claim: Claim,
    new_turn_speaker: Speaker,
    old_turn_speaker: Speaker,
) -> EdgeType:
    """Return the typed edge kind for a Pass-1 supersession.

    Order of discrimination (first match wins):
    1. REFINES — new value is a strict subset of old (narrower).
    2. PHYSICIAN_CONFIRM — physician speaks the new claim.
    3. PATIENT_CORRECTION — patient self-corrects (both speakers patient).
    4. CONTRADICTS — later speaker refutes the earlier speaker's value.
    """
    old_tokens = _tokens(old_claim.value)
    new_tokens = _tokens(new_claim.value)
    if old_tokens and new_tokens and new_tokens < old_tokens:
        return EdgeType.REFINES
    if new_turn_speaker is Speaker.PHYSICIAN:
        return EdgeType.PHYSICIAN_CONFIRM
    if (
        new_turn_speaker is Speaker.PATIENT
        and old_turn_speaker is Speaker.PATIENT
    ):
        return EdgeType.PATIENT_CORRECTION
    return EdgeType.CONTRADICTS


def _get_speaker(conn: sqlite3.Connection, turn_id: str) -> Speaker:
    row = conn.execute("SELECT speaker FROM turns WHERE turn_id = ?", (turn_id,)).fetchone()
    if row is None:
        # Impossible under normal flow: claims.insert_claim validates
        # source_turn_id exists before persisting. Loud error beats silent None.
        raise ValueError(f"turn {turn_id!r} not found when classifying supersession")
    return Speaker(row["speaker"])


def detect_pass1(
    conn: sqlite3.Connection,
    new_claim: Claim,
) -> SupersessionEdge | None:
    """Deterministic Pass-1 supersession for a freshly-inserted claim.

    Returns the created edge (and marks the old claim superseded) or `None` if
    no prior matching active/confirmed claim exists.

    Guard (Eng_doc.md §5.3): never fires if the candidate prior claim is from
    the same turn — within-turn claims are additive.
    """
    row = conn.execute(
        "SELECT * FROM claims WHERE session_id = ? AND subject = ? AND predicate = ?"
        " AND status IN ('active','confirmed') AND claim_id != ?"
        " AND source_turn_id != ?"
        " ORDER BY created_ts DESC LIMIT 1",
        (
            new_claim.session_id,
            new_claim.subject,
            new_claim.predicate,
            new_claim.claim_id,
            new_claim.source_turn_id,
        ),
    ).fetchone()
    if row is None:
        return None

    from src.substrate.claims import row_to_claim  # local import avoids circular at module load

    old_claim = row_to_claim(row)

    # Idempotent: same value means no supersession. Keep both as active.
    if old_claim.value.strip().lower() == new_claim.value.strip().lower():
        return None

    # Scope guard: only supersede if values describe the same fact.
    import re as _re
    _noise = {"the","a","an","is","was","to","for","and","or","of","in","on","at",
              "it","my","i","me","we","up","so","no","not","but","with","has","had",
              "be","do","did","will","been","just","very","really","also","about",
              "some","from","that","this","more","than","each","during"}
    old_content = set(_re.findall(r"[a-z0-9]+", old_claim.value.lower())) - _noise
    new_content = set(_re.findall(r"[a-z0-9]+", new_claim.value.lower())) - _noise
    if old_content and new_content:
        jaccard = len(old_content & new_content) / len(old_content | new_content)
        if jaccard < 0.3:
            return None

    new_speaker = _get_speaker(conn, new_claim.source_turn_id)
    old_speaker = _get_speaker(conn, old_claim.source_turn_id)
    edge_type = _classify_edge(
        old_claim=old_claim,
        new_claim=new_claim,
        new_turn_speaker=new_speaker,
        old_turn_speaker=old_speaker,
    )

    edge = write_supersession_edge(
        conn,
        old_claim_id=old_claim.claim_id,
        new_claim_id=new_claim.claim_id,
        edge_type=edge_type,
        identity_score=None,
    )
    set_claim_status(conn, old_claim.claim_id, ClaimStatus.SUPERSEDED)
    # Close the temporal validity window on the superseded claim. Reuse the
    # edge's `created_ts` rather than calling now_ns() again — keeps the
    # supersession event atomic (one timestamp, two records). Aligned with
    # Zep's KG-edge `valid_until` pattern (arXiv:2501.13956). If the old
    # claim's `valid_from_ts` happens to equal `edge.created_ts` (extreme
    # clock collision), guard the CHECK constraint by leaving valid_until
    # NULL — the supersession event itself is still recorded in the edges
    # table and `status=SUPERSEDED` is the canonical signal.
    conn.execute(
        "UPDATE claims SET valid_until_ts = ?"
        " WHERE claim_id = ?"
        " AND (valid_from_ts IS NULL OR valid_from_ts < ?)",
        (edge.created_ts, old_claim.claim_id, edge.created_ts),
    )
    # Collapse transitive chains (gt_v2_study_notes §3.4). If old_claim itself
    # superseded an older claim, we do nothing extra here: the older claim is
    # already SUPERSEDED and the chain is one-hop in the edges table. That is
    # acceptable because `list_active_claims` filters by status; chains are
    # traversed only when the UI asks for provenance.
    log.info(
        "substrate.supersession_pass1",
        session_id=new_claim.session_id,
        old_claim_id=old_claim.claim_id,
        new_claim_id=new_claim.claim_id,
        edge_type=edge_type.value,
    )
    return edge


def write_supersession_edge(
    conn: sqlite3.Connection,
    *,
    old_claim_id: str,
    new_claim_id: str,
    edge_type: EdgeType,
    identity_score: float | None,
) -> SupersessionEdge:
    """Insert a typed supersession edge (no status side-effect here).

    Callers that also need to flip the old claim's `status` to
    `SUPERSEDED` should do so explicitly via
    `claims.set_claim_status`. Kept separate so Pass 2 and the UI
    dismissal path can reuse this helper without implying a status
    change policy.
    """
    edge_id = _new_edge_id()
    # Use the substrate's monotonic-bumped clock (claims.now_ns) so edge
    # timestamps stay strictly greater than any claim's valid_from_ts —
    # otherwise the temporal-validity update in detect_pass1 silently
    # skips, breaking determinism across runs that have warmed _last_ts.
    ts = now_ns()
    conn.execute(
        "INSERT INTO supersession_edges"
        " (edge_id, old_claim_id, new_claim_id, edge_type, identity_score, created_ts)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        (edge_id, old_claim_id, new_claim_id, edge_type.value, identity_score, ts),
    )
    return SupersessionEdge(
        edge_id=edge_id,
        old_claim_id=old_claim_id,
        new_claim_id=new_claim_id,
        edge_type=edge_type,
        identity_score=identity_score,
        created_ts=ts,
    )


def record_clinician_dismissal(
    conn: sqlite3.Connection,
    *,
    dismissed_claim_id: str,
    replacement_claim_id: str | None,
) -> SupersessionEdge | None:
    """Record a physician tap dismissing a claim (UI → substrate).

    If `replacement_claim_id` is `None`, only the status is set to
    `DISMISSED` — no edge is created (no new claim to point to). Otherwise
    an edge with `edge_type=DISMISSED_BY_CLINICIAN` is written.
    """
    set_claim_status(conn, dismissed_claim_id, ClaimStatus.DISMISSED)
    if replacement_claim_id is None:
        return None
    return write_supersession_edge(
        conn,
        old_claim_id=dismissed_claim_id,
        new_claim_id=replacement_claim_id,
        edge_type=EdgeType.DISMISSED_BY_CLINICIAN,
        identity_score=None,
    )
