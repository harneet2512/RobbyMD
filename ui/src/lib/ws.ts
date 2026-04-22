import { useSession } from "@/store/session";
import type { SubstrateEvent } from "@/types/substrate";
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
} from "@/types/substrate";

/**
 * ws.ts — WebSocket connection manager (real transport layer).
 *
 * Phase 1 uses MockServer in-process. This module wires the future real
 * WebSocket (ws://localhost:<port>/session/:id/events) once src/api/ ships.
 * The routeEvent function is identical to api/client.ts — the only difference
 * is transport.
 *
 * Usage:
 *   const conn = connectWs("ws://localhost:8080/session/sess_abc123/events");
 *   // ... later:
 *   conn.disconnect();
 *
 * Unknown event kinds (future events not yet handled) are silently dropped
 * per contract — unknown events must not crash the UI.
 *
 * Reconnect: exponential back-off (1 s → 2 s → 4 s → … max 30 s).
 */

export type ConnectionStatus = "connecting" | "connected" | "reconnecting" | "disconnected";

export interface WsConnection {
  /** Current WebSocket URL being used. */
  url: string;
  /** Permanently close the connection (no reconnect). */
  disconnect: () => void;
}

/**
 * Parse an incoming WebSocket message body into a SubstrateEvent or null.
 * Returns null for unknown event types so unknown events don't crash.
 */
export function parseWsMessage(data: string): SubstrateEvent | null {
  let parsed: unknown;
  try {
    parsed = JSON.parse(data);
  } catch {
    return null;
  }
  if (
    typeof parsed !== "object" ||
    parsed === null ||
    !("event" in parsed) ||
    typeof (parsed as Record<string, unknown>).event !== "string"
  ) {
    return null;
  }
  const eventName = (parsed as Record<string, unknown>).event as string;
  const KNOWN_EVENTS: Set<string> = new Set([
    TURN_ADDED,
    TURN_FULL,
    CLAIM_CREATED,
    CLAIM_FULL,
    CLAIM_SUPERSEDED,
    CLAIM_STATUS_CHANGED,
    PROJECTION_UPDATED,
    NOTE_SENTENCE_ADDED,
    NOTE_SENTENCE_FULL,
    RANKING_UPDATED,
    VERIFIER_UPDATED,
  ]);
  if (!KNOWN_EVENTS.has(eventName)) {
    // Unknown event type — silently drop; do not crash. See contract above.
    return null;
  }
  // Trust the shape — schema validated server-side.
  return parsed as SubstrateEvent;
}

/** Route a parsed SubstrateEvent into the Zustand store. */
export function routeEvent(e: SubstrateEvent): void {
  const s = useSession.getState();
  switch (e.event) {
    case TURN_ADDED:
      return; // Notification-only — wait for TURN_FULL.
    case TURN_FULL:
      s.upsertTurn(e.payload);
      return;
    case CLAIM_CREATED:
      return; // Notification-only — wait for CLAIM_FULL.
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
      return; // Server triggers ranking + verifier re-emit; no local change.
    case NOTE_SENTENCE_ADDED:
      return; // Notification-only — wait for NOTE_SENTENCE_FULL.
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
      // TypeScript exhaustiveness guard — will error if a new event is added
      // without a handler.
      const _never: never = e;
      void _never;
    }
  }
}

const MAX_BACKOFF_MS = 30_000;

/**
 * Open a WebSocket connection to the real substrate API server.
 * Reconnects with exponential back-off on close/error.
 *
 * @param url  Full WebSocket URL, e.g. ws://localhost:8080/session/abc/events
 * @param onStatus  Optional callback for status changes.
 */
export function connectWs(
  url: string,
  onStatus?: (status: ConnectionStatus) => void,
): WsConnection {
  let ws: WebSocket | null = null;
  let stopped = false;
  let retryMs = 1_000;
  let retryTimer: ReturnType<typeof setTimeout> | null = null;

  function connect() {
    if (stopped) return;
    onStatus?.("connecting");
    ws = new WebSocket(url);

    ws.onopen = () => {
      retryMs = 1_000; // Reset backoff on successful connect.
      onStatus?.("connected");
    };

    ws.onmessage = (evt) => {
      const e = parseWsMessage(evt.data as string);
      if (e) routeEvent(e);
    };

    ws.onclose = () => {
      if (stopped) {
        onStatus?.("disconnected");
        return;
      }
      onStatus?.("reconnecting");
      retryTimer = setTimeout(() => {
        retryMs = Math.min(retryMs * 2, MAX_BACKOFF_MS);
        connect();
      }, retryMs);
    };

    ws.onerror = () => {
      // onclose fires immediately after onerror — handled there.
    };
  }

  connect();

  return {
    url,
    disconnect() {
      stopped = true;
      if (retryTimer != null) clearTimeout(retryTimer);
      if (ws) ws.close();
      onStatus?.("disconnected");
    },
  };
}
