import { MockServer } from "@/mock/MockServer";
import { useSession } from "@/store/session";
import {
  CLAIM_CREATED,
  CLAIM_FULL,
  CLAIM_STATUS_CHANGED,
  CLAIM_SUPERSEDED,
  NOTE_SENTENCE_ADDED,
  NOTE_SENTENCE_FULL,
  PROJECTION_UPDATED,
  RANKING_UPDATED,
  TURN_ADDED,
  TURN_FULL,
  VERIFIER_UPDATED,
  type SubstrateEvent,
  type BranchRanking,
  type Claim,
  type EdgeType,
  type Turn,
  type NoteSentence,
  type VerifierOutput,
} from "@/types/substrate";

export interface Connection {
  sessionId: string;
  disconnect: () => void;
}

function routeEvent(e: SubstrateEvent): void {
  const s = useSession.getState();
  switch (e.event) {
    case TURN_ADDED:
      return;
    case TURN_FULL:
      s.upsertTurn(e.payload);
      return;
    case CLAIM_CREATED:
      return;
    case CLAIM_FULL:
      s.upsertClaim(e.payload);
      return;
    case CLAIM_SUPERSEDED: {
      s.appendEdge({
        edge_id: `edge_${e.payload.old_claim_id}_${e.payload.new_claim_id}`,
        old_claim_id: e.payload.old_claim_id,
        new_claim_id: e.payload.new_claim_id,
        edge_type: e.payload.edge_type,
        identity_score: e.payload.identity_score,
        created_ts: Date.now() * 1_000_000,
      });
      s.setClaimStatus(e.payload.old_claim_id, "superseded");
      return;
    }
    case CLAIM_STATUS_CHANGED:
      s.setClaimStatus(e.payload.claim_id, e.payload.status);
      return;
    case PROJECTION_UPDATED:
      return;
    case NOTE_SENTENCE_ADDED:
      return;
    case NOTE_SENTENCE_FULL:
      s.upsertSentence(e.payload);
      return;
    case RANKING_UPDATED:
      s.setRanking(e.payload);
      return;
    case VERIFIER_UPDATED:
      s.setVerifier(e.payload);
      return;
    default: {
      const _never: never = e;
      void _never;
    }
  }
}

// ── Mock connection (fixture replay, no server needed) ──

export function connectMock(): Connection {
  const server = new MockServer({ playbackMs: 650 });
  useSession.getState().startSession(server.sessionId);
  const unsub = server.subscribe(routeEvent);
  server.start();
  return {
    sessionId: server.sessionId,
    disconnect: () => {
      unsub();
      server.stop();
      useSession.getState().endSession();
    },
  };
}

// ── Live connection (WebSocket to FastAPI backend on port 8420) ──

const WS_PORT = 8420;
const RECONNECT_MS = 2000;
const MAX_RECONNECT_MS = 30_000;

function mapBackendClaim(c: Record<string, unknown>): Claim {
  return {
    claim_id: c.claim_id as string,
    session_id: c.session_id as string,
    subject: (c.subject as string) ?? "chest_pain",
    predicate: c.predicate as string,
    value: c.value as string,
    value_normalised: (c.value_normalised as string | null) ?? (c.value as string),
    confidence: c.confidence as number,
    source_turn_id: c.source_turn_id as string,
    status: (c.status as Claim["status"]) ?? "active",
    created_ts: (c.created_ts_ns as number) ?? (c.created_ts as number) ?? 0,
    char_start: (c.char_start as number | null) ?? null,
    char_end: (c.char_end as number | null) ?? null,
  };
}

function mapBackendTurn(t: Record<string, unknown>): Turn {
  return {
    turn_id: t.turn_id as string,
    session_id: t.session_id as string,
    speaker: t.speaker as Turn["speaker"],
    text: t.text as string,
    ts: t.ts as number,
    asr_confidence: (t.asr_confidence as number | null) ?? null,
  };
}

function mapBackendRanking(data: Record<string, unknown>): { scores: Array<{ branch: string; log_score: number; posterior: number; applied: unknown[] }> } {
  const hypotheses = (data.hypotheses ?? data.scores ?? []) as Array<Record<string, unknown>>;
  return {
    scores: hypotheses.map((h) => ({
      branch: h.branch as string,
      log_score: (h.log_score as number) ?? 0,
      posterior: (h.posterior as number) ?? 0,
      applied: (h.applied as unknown[]) ?? [],
    })),
  };
}

function handleBackendMessage(raw: string): void {
  let msg: { type: string; data: Record<string, unknown> };
  try {
    msg = JSON.parse(raw);
  } catch {
    return;
  }
  const { type, data } = msg;

  switch (type) {
    case "snapshot":
      handleSnapshot(data);
      break;

    case "turn.full":
      routeEvent({ event: TURN_FULL, payload: mapBackendTurn(data) });
      break;

    case "claim.created":
      routeEvent({ event: CLAIM_FULL, payload: mapBackendClaim(data) });
      break;

    case "claim.superseded":
      routeEvent({
        event: CLAIM_SUPERSEDED,
        payload: {
          old_claim_id: data.old_claim_id as string,
          new_claim_id: data.new_claim_id as string,
          edge_type: ((data.edge_type as string) ?? "patient_correction") as EdgeType,
          identity_score: (data.identity_score as number | null) ?? null,
        },
      });
      break;

    case "claim.status_changed":
      routeEvent({
        event: CLAIM_STATUS_CHANGED,
        payload: {
          claim_id: data.claim_id as string,
          status: data.status as Claim["status"],
        },
      });
      break;

    case "ranking.updated":
      routeEvent({
        event: RANKING_UPDATED,
        payload: mapBackendRanking(data) as unknown as BranchRanking,
      });
      break;

    case "verifier.updated":
      routeEvent({
        event: VERIFIER_UPDATED,
        payload: data as unknown as VerifierOutput,
      });
      break;

    case "note_sentence.full":
      routeEvent({
        event: NOTE_SENTENCE_FULL,
        payload: data as unknown as NoteSentence,
      });
      break;
  }
}

function handleSnapshot(snap: Record<string, unknown>): void {
  const turns = (snap.turns ?? []) as Array<Record<string, unknown>>;
  for (const t of turns) {
    routeEvent({ event: TURN_FULL, payload: mapBackendTurn(t) });
  }

  const claims = (snap.claims ?? []) as Array<Record<string, unknown>>;
  for (const c of claims) {
    routeEvent({ event: CLAIM_FULL, payload: mapBackendClaim(c) });
  }

  const ranking = snap.ranking as Record<string, unknown> | undefined;
  if (ranking) {
    routeEvent({
      event: RANKING_UPDATED,
      payload: mapBackendRanking(ranking) as unknown as BranchRanking,
    });
  }

  const verifier = snap.verifier as Record<string, unknown> | null;
  if (verifier) {
    routeEvent({
      event: VERIFIER_UPDATED,
      payload: verifier as unknown as VerifierOutput,
    });
  }
}

export function connectLive(sessionId: string): Connection {
  useSession.getState().startSession(sessionId);

  let ws: WebSocket | null = null;
  let stopped = false;
  let retryMs = RECONNECT_MS;
  let retryTimer: ReturnType<typeof setTimeout> | null = null;

  function open() {
    if (stopped) return;
    const url = `ws://${window.location.hostname}:${WS_PORT}/ws/${sessionId}`;
    ws = new WebSocket(url);

    ws.onopen = () => {
      retryMs = RECONNECT_MS;
      if (retryTimer) {
        clearTimeout(retryTimer);
        retryTimer = null;
      }
    };

    ws.onmessage = (ev) => {
      handleBackendMessage(ev.data as string);
    };

    ws.onclose = () => {
      if (stopped) return;
      retryTimer = setTimeout(() => {
        retryMs = Math.min(retryMs * 2, MAX_RECONNECT_MS);
        open();
      }, retryMs);
    };

    ws.onerror = () => ws?.close();
  }

  open();

  return {
    sessionId,
    disconnect: () => {
      stopped = true;
      if (retryTimer) clearTimeout(retryTimer);
      if (ws) ws.close();
      useSession.getState().endSession();
    },
  };
}
