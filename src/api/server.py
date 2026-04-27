"""FastAPI + WebSocket server bridging the substrate to the working surface.

Serves a single WebSocket endpoint /ws/{session_id}. On connect, sends
a full snapshot. Then subscribes to the EventBus and forwards events
as JSON to all connected clients. Accepts inbound physician decisions.

Run:
    uvicorn src.api.server:app --host 0.0.0.0 --port 8420 --reload
"""
from __future__ import annotations

import asyncio
import logging
import sqlite3
import threading
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.api.ws_protocol import serialize, parse_client_message
from src.substrate.event_bus import (
    CLAIM_CREATED,
    CLAIM_SUPERSEDED,
    CLAIM_STATUS_CHANGED,
    PROJECTION_UPDATED,
    TURN_ADDED,
    NOTE_SENTENCE_ADDED,
    EventBus,
)
from src.substrate.schema import open_database

log = logging.getLogger(__name__)

_bus = EventBus()
_conn: sqlite3.Connection | None = None
_loop: asyncio.AbstractEventLoop | None = None
_clients: dict[str, list[asyncio.Queue[str]]] = {}
_demo_thread: threading.Thread | None = None
_agent_sessions: dict[str, Any] = {}


def _broadcast(session_id: str, msg: str) -> None:
    for q in _clients.get(session_id, []):
        if _loop:
            _loop.call_soon_threadsafe(q.put_nowait, msg)


def _on_turn_added(payload: dict[str, Any]) -> None:
    sid = payload.get("session_id", "demo")
    turn_id = payload.get("turn_id")
    if not turn_id or _conn is None:
        return
    row = _conn.execute(
        "SELECT turn_id, session_id, speaker, text, ts, asr_confidence "
        "FROM turns WHERE turn_id = ?",
        (turn_id,),
    ).fetchone()
    if row:
        _broadcast(sid, serialize("turn.full", {
            "turn_id": row[0],
            "session_id": row[1],
            "speaker": row[2],
            "text": row[3],
            "ts": row[4],
            "asr_confidence": row[5],
        }))


def _on_claim_created(payload: dict[str, Any]) -> None:
    sid = payload.get("session_id", "demo")
    _broadcast(sid, serialize("claim.created", payload))


def _on_claim_superseded(payload: dict[str, Any]) -> None:
    sid = payload.get("session_id", "demo")
    _broadcast(sid, serialize("claim.superseded", payload))


def _on_claim_status_changed(payload: dict[str, Any]) -> None:
    sid = payload.get("session_id", "demo")
    _broadcast(sid, serialize("claim.status_changed", payload))


def _on_projection_updated(payload: dict[str, Any]) -> None:
    sid = payload.get("session_id", "demo")
    ranking_data = _build_ranking(sid)
    _broadcast(sid, serialize("ranking.updated", ranking_data))
    _maybe_broadcast_verifier(sid, ranking_data)


def _on_note_sentence_added(payload: dict[str, Any]) -> None:
    sid = payload.get("session_id", "demo")
    _broadcast(sid, serialize("note_sentence.full", payload))


def _maybe_broadcast_verifier(session_id: str, ranking_data: dict[str, Any]) -> None:
    """Run verifier after ranking update and broadcast result."""
    try:
        from src.substrate.claims import list_active_claims
        from src.differential.engine import rank_branches
        from src.differential.lr_table import load_lr_table
        from src.differential.types import ActiveClaim
        from src.substrate.predicate_packs import active_pack
        from src.verifier.verifier import verify
        from dataclasses import asdict

        if _conn is None:
            return

        claims = list_active_claims(_conn, session_id)
        if not claims:
            return

        pack = active_pack()
        lr_table = load_lr_table(pack.lr_table_path) if pack.lr_table_path else None
        if not lr_table:
            return

        active = []
        for c in claims:
            path = f"{c.predicate}={c.value_normalised or c.value}"
            polarity = not (c.value_normalised or c.value).startswith("negated:")
            active.append(ActiveClaim(
                claim_id=c.claim_id,
                predicate_path=path,
                polarity=polarity,
            ))

        ranking = rank_branches(tuple(active), lr_table)
        verifier_output = verify(ranking, lr_table, active)

        payload = {
            "why_moved": list(verifier_output.why_moved),
            "missing_or_contradicting": list(verifier_output.missing_or_contradicting),
            "next_best_question": verifier_output.next_best_question,
            "next_question_rationale": verifier_output.next_question_rationale,
            "source_feature": verifier_output.source_feature,
        }
        _broadcast(session_id, serialize("verifier.updated", payload))

    except Exception as e:
        log.warning("verifier broadcast failed: %s", e)


def _handle_decision(session_id: str, msg: dict[str, Any]) -> None:
    """Process physician decisions from the UI (confirm/dismiss claims)."""
    msg_type = msg.get("type", "")
    claim_id = msg.get("claim_id")
    if not claim_id or _conn is None:
        return

    try:
        if msg_type == "decision.confirm":
            _conn.execute(
                "UPDATE claims SET status = 'confirmed' WHERE claim_id = ? AND session_id = ?",
                (claim_id, session_id),
            )
            _conn.commit()
            _bus.publish(CLAIM_STATUS_CHANGED, {
                "claim_id": claim_id,
                "session_id": session_id,
                "status": "confirmed",
            })
        elif msg_type == "decision.dismiss":
            _conn.execute(
                "UPDATE claims SET status = 'dismissed' WHERE claim_id = ? AND session_id = ?",
                (claim_id, session_id),
            )
            _conn.commit()
            _bus.publish(CLAIM_STATUS_CHANGED, {
                "claim_id": claim_id,
                "session_id": session_id,
                "status": "dismissed",
            })
            _bus.publish(PROJECTION_UPDATED, {
                "session_id": session_id,
                "active_count": -1,
            })
    except Exception as e:
        log.warning("decision handling failed: %s", e)


def _build_ranking(session_id: str) -> dict[str, Any]:
    """Build hypothesis ranking from current active claims."""
    try:
        from src.substrate.claims import list_active_claims
        from src.differential.engine import rank_branches
        from src.differential.lr_table import load_lr_table
        from src.differential.types import ActiveClaim
        from src.substrate.predicate_packs import active_pack

        if _conn is None:
            return {"scores": _default_scores()}

        claims = list_active_claims(_conn, session_id)
        pack = active_pack()
        lr_table = load_lr_table(pack.lr_table_path) if pack.lr_table_path else None

        if not lr_table:
            return {"scores": _default_scores()}

        active = []
        for c in claims:
            path = f"{c.predicate}={c.value_normalised or c.value}"
            polarity = not (c.value_normalised or c.value).startswith("negated:")
            active.append(ActiveClaim(
                claim_id=c.claim_id,
                predicate_path=path,
                polarity=polarity,
            ))

        ranking = rank_branches(tuple(active), lr_table)
        branch_labels = {"cardiac": "ACS", "pulmonary": "PE", "msk": "MSK", "gi": "GERD"}

        scores = []
        for bs in ranking.scores:
            applied_list = []
            for a in bs.applied:
                applied_list.append({
                    "claim_id": a.claim_id,
                    "branch": a.branch,
                    "feature": a.feature,
                    "predicate_path": a.predicate_path,
                    "lr_value": a.lr_value,
                    "log_lr": a.log_lr,
                    "direction": a.direction,
                    "approximation": a.approximation,
                })
            scores.append({
                "branch": bs.branch,
                "label": branch_labels.get(bs.branch, bs.branch.upper()),
                "posterior": bs.posterior,
                "log_score": bs.log_score,
                "claim_count": len(bs.applied),
                "applied": applied_list,
            })
        return {"scores": scores}

    except Exception as e:
        log.warning("ranking build failed: %s", e)
        return {"scores": _default_scores()}


def _default_scores() -> list[dict[str, Any]]:
    return [
        {"branch": "cardiac", "label": "ACS", "posterior": 0.25, "log_score": 0.0, "claim_count": 0, "applied": []},
        {"branch": "pulmonary", "label": "PE", "posterior": 0.25, "log_score": 0.0, "claim_count": 0, "applied": []},
        {"branch": "msk", "label": "MSK", "posterior": 0.25, "log_score": 0.0, "claim_count": 0, "applied": []},
        {"branch": "gi", "label": "GERD", "posterior": 0.25, "log_score": 0.0, "claim_count": 0, "applied": []},
    ]


_demo_started_sessions: set[str] = set()
_session_start_ns: dict[str, int] = {}


def _build_snapshot(session_id: str) -> dict[str, Any]:
    if session_id not in _session_start_ns:
        _session_start_ns[session_id] = int(time.time() * 1_000_000_000)

    turns_list: list[dict[str, Any]] = []
    claims_list: list[dict[str, Any]] = []
    if _conn is not None:
        try:
            turn_rows = _conn.execute(
                "SELECT turn_id, session_id, speaker, text, ts, asr_confidence "
                "FROM turns WHERE session_id = ? ORDER BY ts",
                (session_id,),
            ).fetchall()
            for r in turn_rows:
                turns_list.append({
                    "turn_id": r[0],
                    "session_id": r[1],
                    "speaker": r[2],
                    "text": r[3],
                    "ts": r[4],
                    "asr_confidence": r[5],
                })
        except Exception as e:
            log.warning("snapshot turn query failed: %s", e)

        try:
            rows = _conn.execute(
                "SELECT claim_id, session_id, subject, predicate, value, "
                "confidence, source_turn_id, status, created_ts "
                "FROM claims WHERE session_id = ? AND status = 'active' "
                "ORDER BY created_ts",
                (session_id,),
            ).fetchall()
            for r in rows:
                turn_row = _conn.execute(
                    "SELECT text, speaker FROM turns WHERE turn_id = ?",
                    (r[6],),
                ).fetchone()
                claims_list.append({
                    "claim_id": r[0],
                    "session_id": r[1],
                    "subject": r[2],
                    "predicate": r[3],
                    "value": r[4],
                    "confidence": r[5],
                    "source_turn_id": r[6],
                    "source_turn_text": turn_row[0] if turn_row else "",
                    "source_turn_speaker": turn_row[1] if turn_row else "system",
                    "char_start": None,
                    "char_end": None,
                    "status": r[7],
                    "created_ts_ns": r[8],
                })
        except Exception as e:
            log.warning("snapshot claim query failed: %s", e)

    return {
        "encounter": {
            "session_id": session_id,
            "patient_name": "MR TORRES",
            "patient_age": 52,
            "patient_sex": "M",
            "chief_complaint": "chest pain",
            "started_at_ns": _session_start_ns[session_id],
        },
        "turns": turns_list,
        "claims": claims_list,
        "ranking": _build_ranking(session_id),
        "verifier": None,
    }


def _maybe_start_demo(session_id: str) -> None:
    """Start the demo replay for a session, at most once."""
    global _demo_thread
    if session_id in _demo_started_sessions:
        return
    _demo_started_sessions.add(session_id)

    import os
    if os.environ.get("DEMO_MODE", "1") == "1":
        from src.api.demo_replay import start_demo
        _demo_thread = threading.Thread(
            target=start_demo,
            args=(_conn, _bus, session_id),
            daemon=True,
        )
        _demo_thread.start()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _conn, _loop

    _loop = asyncio.get_running_loop()
    _conn = open_database(":memory:")

    _bus.subscribe(TURN_ADDED, _on_turn_added)
    _bus.subscribe(CLAIM_CREATED, _on_claim_created)
    _bus.subscribe(CLAIM_SUPERSEDED, _on_claim_superseded)
    _bus.subscribe(CLAIM_STATUS_CHANGED, _on_claim_status_changed)
    _bus.subscribe(PROJECTION_UPDATED, _on_projection_updated)
    _bus.subscribe(NOTE_SENTENCE_ADDED, _on_note_sentence_added)

    from src.substrate.clinical_interpreter import subscribe_interpreter
    subscribe_interpreter(_bus, _conn)

    yield

    if _conn:
        _conn.close()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/ws/{session_id}")
async def ws_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()

    queue: asyncio.Queue[str] = asyncio.Queue()
    _clients.setdefault(session_id, []).append(queue)

    try:
        snapshot = _build_snapshot(session_id)
        await websocket.send_text(serialize("snapshot", snapshot))

        _maybe_start_demo(session_id)

        async def sender():
            while True:
                msg = await queue.get()
                await websocket.send_text(msg)

        sender_task = asyncio.create_task(sender())

        try:
            while True:
                raw = await websocket.receive_text()
                msg = parse_client_message(raw)
                log.info("client message: %s", msg)
                _handle_decision(session_id, msg)
        except WebSocketDisconnect:
            pass
        finally:
            sender_task.cancel()

    finally:
        _clients[session_id].remove(queue)
        if not _clients[session_id]:
            del _clients[session_id]


# ═══════════════════════════════════════════════════════════════════════════
# Aftercare agent REST endpoints — Claude Managed Agents
# ═══════════════════════════════════════════════════════════════════════════


class AgentMessageRequest(BaseModel):
    message: str


class AgentCreateRequest(BaseModel):
    encounter_id: str
    use_managed: bool = True


class AftercareApproveRequest(BaseModel):
    approval_note: str | None = None


@app.post("/api/agents/doctor/session")
async def create_doctor_session(req: AgentCreateRequest):
    """Create a Doctor Agent session for an encounter."""
    from src.agents.orchestrator import create_orchestrator

    if _conn is None:
        return {"error": "Database not initialized"}

    orch = create_orchestrator(_conn, use_managed=req.use_managed)
    session = orch.create_doctor_session(req.encounter_id)
    session_id = f"doctor_{req.encounter_id}"
    _agent_sessions[session_id] = session
    return {"session_id": session_id, "encounter_id": req.encounter_id, "agent": "doctor"}


@app.post("/api/agents/patient/session")
async def create_patient_session(req: AgentCreateRequest):
    """Create a Patient Agent session for an encounter."""
    from src.agents.orchestrator import create_orchestrator

    if _conn is None:
        return {"error": "Database not initialized"}

    orch = create_orchestrator(_conn, use_managed=req.use_managed)
    session = orch.create_patient_session(req.encounter_id)
    session_id = f"patient_{req.encounter_id}"
    _agent_sessions[session_id] = session
    return {"session_id": session_id, "encounter_id": req.encounter_id, "agent": "patient"}


@app.post("/api/agents/{session_id}/message")
async def send_agent_message(session_id: str, req: AgentMessageRequest):
    """Send a message to an agent session and return the response."""
    session = _agent_sessions.get(session_id)
    if session is None:
        return {"error": f"Session {session_id} not found"}

    events = []
    text_parts = []
    for event in session.send_message(req.message):
        events.append(event)
        if event["type"] == "text":
            text_parts.append(event["text"])

    return {
        "session_id": session_id,
        "response": "".join(text_parts),
        "events": events,
    }


@app.post("/api/aftercare/{encounter_id}/generate")
async def generate_aftercare(encounter_id: str):
    """Generate an aftercare package for an encounter."""
    from src.aftercare.package import generate_aftercare_package, cache_package
    from src.aftercare.red_flags import generate_red_flag_list

    if _conn is None:
        return {"error": "Database not initialized"}

    red_flags = generate_red_flag_list(complaint="chest_pain")
    package = generate_aftercare_package(
        _conn, encounter_id, red_flags=red_flags,
    )
    cache_package(package)
    return package.to_dict()


@app.post("/api/aftercare/{encounter_id}/approve")
async def approve_aftercare(encounter_id: str, req: AftercareApproveRequest):
    """Approve an aftercare package (physician action)."""
    from src.aftercare.package import approve_package

    result = approve_package(encounter_id, req.approval_note)
    if result is None:
        return {"error": "Package not found. Generate it first."}
    return result.to_dict()


@app.get("/api/aftercare/{encounter_id}")
async def get_aftercare(encounter_id: str):
    """Get the aftercare package for an encounter."""
    from src.aftercare.package import get_cached_package

    pkg = get_cached_package(encounter_id)
    if pkg is None:
        return {"error": "No aftercare package for this encounter"}
    return pkg.to_dict()


@app.get("/api/escalations/{encounter_id}")
async def list_escalations(encounter_id: str, unreviewed_only: bool = False):
    """List escalations for an encounter."""
    from src.aftercare.escalation import get_escalation_store

    store = get_escalation_store()
    escalations = store.list_for_encounter(encounter_id, unreviewed_only=unreviewed_only)
    return {"escalations": [e.to_dict() for e in escalations]}


# ═══════════════════════════════════════════════════════════════════════════
# Shift Handoff Agent endpoints
# ═══════════════════════════════════════════════════════════════════════════


class HandoffCreateRequest(BaseModel):
    encounter_id: str
    use_managed: bool = True


@app.post("/api/encounters/{encounter_id}/snapshot")
async def create_encounter_snapshot(encounter_id: str):
    """Persist the full reasoning state for handoff."""
    from src.persistence.snapshots import write_encounter_snapshot

    if _conn is None:
        return {"error": "Database not initialized"}
    path = write_encounter_snapshot(_conn, encounter_id)
    return {"encounter_id": encounter_id, "path": str(path)}


@app.post("/api/agents/handoff/session")
async def create_handoff_session(req: HandoffCreateRequest):
    """Create a Shift Handoff Agent session."""
    from src.agents.orchestrator import create_orchestrator

    if _conn is None:
        return {"error": "Database not initialized"}
    orch = create_orchestrator(_conn, use_managed=req.use_managed)
    session = orch.create_handoff_session(req.encounter_id)
    session_id = f"handoff_{req.encounter_id}"
    _agent_sessions[session_id] = session
    return {"session_id": session_id, "encounter_id": req.encounter_id, "agent": "handoff"}


# ═══════════════════════════════════════════════════════════════════════════
# Diagnostic Bias Monitor endpoints
# ═══════════════════════════════════════════════════════════════════════════


class BiasMonitorCreateRequest(BaseModel):
    use_managed: bool = True


@app.post("/api/encounters/{encounter_id}/history-entry")
async def create_history_entry(encounter_id: str):
    """Append a closed encounter to the bias monitor history."""
    from src.persistence.encounter_history import append_encounter_entry

    if _conn is None:
        return {"error": "Database not initialized"}
    entry = append_encounter_entry(_conn, encounter_id)
    return entry


@app.post("/api/agents/bias-monitor/session")
async def create_bias_monitor_session(req: BiasMonitorCreateRequest):
    """Create a Diagnostic Bias Monitor session."""
    from src.agents.orchestrator import create_orchestrator

    if _conn is None:
        return {"error": "Database not initialized"}
    orch = create_orchestrator(_conn, use_managed=req.use_managed)
    session = orch.create_bias_monitor_session()
    session_id = "bias_monitor"
    _agent_sessions[session_id] = session
    return {"session_id": session_id, "agent": "bias_monitor"}


# ═══════════════════════════════════════════════════════════════════════════
# Clinical Note Co-Author endpoints
# ═══════════════════════════════════════════════════════════════════════════


class NoteCoauthorCreateRequest(BaseModel):
    encounter_id: str
    use_managed: bool = True


class NoteEditRequest(BaseModel):
    edited_note_text: str


class ApproveSuggestionRequest(BaseModel):
    approved: bool
    action_type: str | None = None
    params: dict | None = None


@app.post("/api/agents/note-coauthor/session")
async def create_note_coauthor_session(req: NoteCoauthorCreateRequest):
    """Create a Clinical Note Co-Author session."""
    from src.agents.orchestrator import create_orchestrator

    if _conn is None:
        return {"error": "Database not initialized"}
    orch = create_orchestrator(_conn, use_managed=req.use_managed)
    session = orch.create_note_coauthor_session(req.encounter_id)
    session_id = f"note_coauthor_{req.encounter_id}"
    _agent_sessions[session_id] = session
    return {"session_id": session_id, "encounter_id": req.encounter_id, "agent": "note_coauthor"}


@app.post("/api/notes/{encounter_id}/draft")
async def generate_note_draft(encounter_id: str):
    """Generate initial SOAP note draft from substrate."""
    from src.agents.tools.note_coauthor_tools import NoteCoauthorToolDispatcher

    if _conn is None:
        return {"error": "Database not initialized"}
    dispatcher = NoteCoauthorToolDispatcher(_conn)
    return dispatcher.dispatch("generate_note_draft", {"encounter_id": encounter_id})


@app.post("/api/notes/{encounter_id}/edit")
async def submit_note_edit(encounter_id: str, req: NoteEditRequest):
    """Submit physician edit and detect claim conflicts."""
    from src.agents.tools.note_coauthor_tools import NoteCoauthorToolDispatcher

    if _conn is None:
        return {"error": "Database not initialized"}
    dispatcher = NoteCoauthorToolDispatcher(_conn)
    return dispatcher.dispatch("submit_physician_edit", {
        "encounter_id": encounter_id,
        "edited_note_text": req.edited_note_text,
    })


@app.post("/api/agents/{session_id}/approve-suggestion")
async def approve_suggestion(session_id: str, req: ApproveSuggestionRequest):
    """Approve or reject a pending always_ask suggestion."""
    if not req.approved:
        return {"approved": False, "message": "Suggestion rejected by physician."}

    session = _agent_sessions.get(session_id)
    if session is None:
        return {"error": f"Session {session_id} not found"}

    if req.action_type and req.params:
        from src.agents.tools.note_coauthor_tools import NoteCoauthorToolDispatcher
        if _conn is None:
            return {"error": "Database not initialized"}
        dispatcher = NoteCoauthorToolDispatcher(_conn)
        result = dispatcher.execute_approved_action(req.action_type, req.params)
        return {"approved": True, "result": result}

    return {"approved": True, "message": "Suggestion approved."}
