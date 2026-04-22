import fixture from "@/mock/chest_pain_mid_case.transcript.json";
import {
  CLAIM_CREATED,
  CLAIM_FULL,
  PROJECTION_UPDATED,
  TURN_ADDED,
  TURN_FULL,
  type Claim,
  type SubstrateEvent,
  type Turn,
} from "@/types/substrate";

/**
 * MockServer — stands in for the future `src/api/` WebSocket until
 * wt-engine + wt-extraction are wired into an HTTP server. Emits events
 * in the exact shape documented in `src/substrate/event_bus.py`, plus
 * the `*.full` extension events that carry the entire entity so the UI
 * does not have to GET after every notification.
 *
 * Replay is cadence-controlled so the demo can breathe. Default ~600ms
 * between turns; override via `playbackMs` for tests.
 *
 * The server does *not* mutate the Zustand store directly — it publishes
 * to its own subscriber list. `api/client.ts` routes events into the
 * store. This keeps the transport boundary visible and swappable.
 */
type Listener = (e: SubstrateEvent) => void;

interface MockServerOptions {
  playbackMs?: number;
  autoStart?: boolean;
}

function nowNs(): number {
  return Math.floor(performance.now() * 1_000_000);
}

interface FixtureTurn {
  turn_id: string;
  speaker: "patient" | "physician" | "system";
  text: string;
  ts_offset_ms: number;
}

interface FixtureClaim {
  claim_id: string;
  subject: string;
  predicate: string;
  value: string;
  polarity: boolean;
  source_turn_id: string;
  confidence: number;
  char_start: number | null;
  char_end: number | null;
}

interface Fixture {
  session_id: string;
  turns: FixtureTurn[];
  claims: FixtureClaim[];
}

export class MockServer {
  private listeners = new Set<Listener>();
  private timers: number[] = [];
  private started = false;
  private readonly playbackMs: number;
  private readonly fixture: Fixture;

  constructor(opts: MockServerOptions = {}) {
    this.playbackMs = opts.playbackMs ?? 650;
    // Single cast — fixture JSON is validated by shape at compile-time by
    // matching the interface. The StrEnum values line up.
    this.fixture = fixture as unknown as Fixture;
    if (opts.autoStart) this.start();
  }

  get sessionId(): string {
    return this.fixture.session_id;
  }

  subscribe(fn: Listener): () => void {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  private emit(event: SubstrateEvent): void {
    for (const l of this.listeners) l(event);
  }

  /** Replay the fixture in order. Idempotent. */
  start(): void {
    if (this.started) return;
    this.started = true;
    const session_id = this.fixture.session_id;

    this.fixture.turns.forEach((ft, idx) => {
      const fireAt = idx * this.playbackMs;
      const t = window.setTimeout(() => {
        const turn: Turn = {
          turn_id: ft.turn_id,
          session_id,
          speaker: ft.speaker,
          text: ft.text,
          ts: nowNs(),
          asr_confidence: 0.95,
        };
        // Matches src/substrate/event_bus.py TURN_ADDED payload shape.
        this.emit({
          event: TURN_ADDED,
          payload: { turn_id: ft.turn_id, session_id },
        });
        // Full-entity extension — see SubstrateEvent in types/substrate.ts.
        this.emit({ event: TURN_FULL, payload: turn });

        // Emit claims derived from this turn. In the real pipeline, the
        // claim-extractor lands these after extraction — here they ride
        // alongside the turn for demo simplicity.
        const claims = this.fixture.claims.filter(
          (c) => c.source_turn_id === ft.turn_id,
        );
        claims.forEach((fc, cIdx) => {
          const tc = window.setTimeout(() => {
            const claim: Claim = {
              claim_id: fc.claim_id,
              session_id,
              subject: fc.subject,
              predicate: fc.predicate,
              value: fc.value,
              value_normalised: fc.value,
              confidence: fc.confidence,
              source_turn_id: fc.source_turn_id,
              status: "active",
              created_ts: nowNs(),
              char_start: fc.char_start,
              char_end: fc.char_end,
            };
            this.emit({
              event: CLAIM_CREATED,
              payload: {
                claim_id: claim.claim_id,
                session_id,
                predicate: claim.predicate,
                status: claim.status,
              },
            });
            this.emit({ event: CLAIM_FULL, payload: claim });
            this.emit({
              event: PROJECTION_UPDATED,
              payload: { session_id, active_count: -1 },
            });
          }, 120 * (cIdx + 1));
          this.timers.push(tc);
        });
      }, fireAt);
      this.timers.push(t);
    });
  }

  stop(): void {
    for (const t of this.timers) window.clearTimeout(t);
    this.timers = [];
    this.started = false;
  }
}
