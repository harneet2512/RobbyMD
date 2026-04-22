import demoFixture from "../../fixtures/chest_pain_demo.json";
import {
  CLAIM_CREATED,
  CLAIM_FULL,
  CLAIM_SUPERSEDED,
  NOTE_SENTENCE_ADDED,
  NOTE_SENTENCE_FULL,
  PROJECTION_UPDATED,
  RANKING_UPDATED,
  TURN_ADDED,
  TURN_FULL,
  VERIFIER_UPDATED,
  type BranchRanking,
  type Claim,
  type NoteSentence,
  type SubstrateEvent,
  type SupersessionEdge,
  type Turn,
  type VerifierOutput,
} from "@/types/substrate";

/**
 * MockServer — stands in for the future `src/api/` WebSocket until
 * wt-engine + wt-extraction are wired into an HTTP server. Emits events
 * in the exact shape documented in `src/substrate/event_bus.py`, plus
 * the `*.full` extension events that carry the entire entity so the UI
 * does not have to GET after every notification.
 *
 * Now driven from `ui/fixtures/chest_pain_demo.json` — a 18-turn scripted
 * case that includes turns, claims, a supersession edge (patient correction),
 * differential ranking updates, verifier output, and SOAP note deltas.
 *
 * Replay uses the fixture's `t_ms` timestamps scaled by `speedMultiplier`.
 * Default speed = 1.0 (real-time). For tests, pass a high speedMultiplier
 * (e.g. 100) so all events fire quickly.
 *
 * The server does *not* mutate the Zustand store directly — it publishes
 * to its own subscriber list. `api/client.ts` routes events into the
 * store. This keeps the transport boundary visible and swappable.
 */

type Listener = (e: SubstrateEvent) => void;

export interface MockServerOptions {
  /** t_ms scale factor; higher = faster replay. Default 1.0. */
  speedMultiplier?: number;
  /** Deprecated alias for backwards compat — treated as 1000/playbackMs speed. */
  playbackMs?: number;
  autoStart?: boolean;
}

// ---- Fixture shapes ----

interface FixtureTurnEvent {
  t_ms: number;
  type: "turn";
  turn: Turn;
}

interface FixtureClaimEvent {
  t_ms: number;
  type: "claim";
  claim: Claim;
}

interface FixtureSupersede {
  t_ms: number;
  type: "supersede";
  edge: SupersessionEdge;
}

interface FixtureDifferential {
  t_ms: number;
  type: "differential";
  ranking: BranchRanking;
}

interface FixtureVerifier {
  t_ms: number;
  type: "verifier";
  verifier: VerifierOutput;
}

interface FixtureSoapDelta {
  t_ms: number;
  type: "soap_delta";
  delta: NoteSentence;
}

type FixtureEvent =
  | FixtureTurnEvent
  | FixtureClaimEvent
  | FixtureSupersede
  | FixtureDifferential
  | FixtureVerifier
  | FixtureSoapDelta;

interface Fixture {
  case_id: string;
  events: FixtureEvent[];
}

// ---- MockServer ----

export class MockServer {
  private listeners = new Set<Listener>();
  private timers: number[] = [];
  private started = false;
  private readonly speedMultiplier: number;
  private readonly fixture: Fixture;

  constructor(opts: MockServerOptions = {}) {
    // Backwards compat: playbackMs previously set inter-turn interval; map to speed.
    if (opts.playbackMs != null) {
      // 650 ms/turn was the old default → treat as speedMultiplier = 1.
      this.speedMultiplier = 650 / opts.playbackMs;
    } else {
      this.speedMultiplier = opts.speedMultiplier ?? 1.0;
    }
    this.fixture = demoFixture as unknown as Fixture;
    if (opts.autoStart) this.start();
  }

  /** session_id derived from case_id so it matches the fixture's payload. */
  get sessionId(): string {
    return `sess_${this.fixture.case_id}`;
  }

  subscribe(fn: Listener): () => void {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  private emit(event: SubstrateEvent): void {
    for (const l of this.listeners) l(event);
  }

  /** Replay all fixture events at scaled cadence. Idempotent. */
  start(): void {
    if (this.started) return;
    this.started = true;

    for (const ev of this.fixture.events) {
      const delay = Math.max(0, Math.round(ev.t_ms / this.speedMultiplier));
      const t = window.setTimeout(() => this.dispatchFixtureEvent(ev), delay);
      this.timers.push(t);
    }
  }

  private dispatchFixtureEvent(ev: FixtureEvent): void {
    switch (ev.type) {
      case "turn": {
        const { turn } = ev;
        this.emit({ event: TURN_ADDED, payload: { turn_id: turn.turn_id, session_id: turn.session_id } });
        this.emit({ event: TURN_FULL, payload: turn });
        return;
      }
      case "claim": {
        const { claim } = ev;
        this.emit({
          event: CLAIM_CREATED,
          payload: { claim_id: claim.claim_id, session_id: claim.session_id, predicate: claim.predicate, status: claim.status },
        });
        this.emit({ event: CLAIM_FULL, payload: claim });
        this.emit({ event: PROJECTION_UPDATED, payload: { session_id: claim.session_id, active_count: -1 } });
        return;
      }
      case "supersede": {
        const { edge } = ev;
        this.emit({
          event: CLAIM_SUPERSEDED,
          payload: { old_claim_id: edge.old_claim_id, new_claim_id: edge.new_claim_id, edge_type: edge.edge_type, identity_score: edge.identity_score },
        });
        return;
      }
      case "differential": {
        this.emit({ event: RANKING_UPDATED, payload: ev.ranking });
        return;
      }
      case "verifier": {
        this.emit({ event: VERIFIER_UPDATED, payload: ev.verifier });
        return;
      }
      case "soap_delta": {
        const { delta } = ev;
        this.emit({
          event: NOTE_SENTENCE_ADDED,
          payload: { sentence_id: delta.sentence_id, session_id: delta.session_id, section: delta.section, source_claim_ids: delta.source_claim_ids },
        });
        this.emit({ event: NOTE_SENTENCE_FULL, payload: delta });
        return;
      }
      default: {
        // Exhaustiveness — unknown fixture event types are silently dropped.
        const _never: never = ev;
        void _never;
      }
    }
  }

  stop(): void {
    for (const t of this.timers) window.clearTimeout(t);
    this.timers = [];
    this.started = false;
  }
}
