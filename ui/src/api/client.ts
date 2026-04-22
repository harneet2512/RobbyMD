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
} from "@/types/substrate";

/**
 * API client — the transport shim between the substrate (future
 * `src/api/` WebSocket) and the Zustand store.
 *
 * Phase 1: only the MockServer is wired. The real client will replace
 * `MockServer` with a WebSocket to `ws://localhost:PORT/session/:id/events`
 * once `src/api/` lands. The rest of the pipeline (routeEvent below) does
 * not change — it consumes `SubstrateEvent` regardless of transport.
 *
 * Event-bus contract handshake (for when `src/api/` gets built):
 *   Produced by substrate (src/substrate/event_bus.py):
 *     - turn.added            -> store.upsertTurn requires companion turn.full
 *     - claim.created         -> store.upsertClaim requires companion claim.full
 *     - claim.superseded      -> store.appendEdge + setClaimStatus(old, "superseded")
 *     - claim.status_changed  -> store.setClaimStatus
 *     - projection.updated    -> (no store change; triggers verifier re-emit server-side)
 *     - note_sentence.added   -> store.upsertSentence requires companion note_sentence.full
 *   UI expects the server to additionally emit full-entity events
 *   (*.full) alongside the notification-only events. This is *not* in
 *   event_bus.py today — it's the contract ask for src/api/.
 */

export interface Connection {
  sessionId: string;
  disconnect: () => void;
}

function routeEvent(e: SubstrateEvent): void {
  const s = useSession.getState();
  switch (e.event) {
    case TURN_ADDED:
      // notification-only; waiting for TURN_FULL to land the entity.
      return;
    case TURN_FULL:
      s.upsertTurn(e.payload);
      return;
    case CLAIM_CREATED:
      // notification-only; waiting for CLAIM_FULL.
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
      // Triggers ranking + verifier re-emit on the server; no local change.
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
      // Exhaustiveness — TypeScript will flag an unhandled variant.
      const _never: never = e;
      void _never;
    }
  }
}

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
