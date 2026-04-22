/**
 * mock_server/server.ts — Node.js WebSocket replay server.
 *
 * Reads ui/fixtures/chest_pain_demo.json and broadcasts events to all
 * connected clients at realistic cadence (default: fixture t_ms timestamps;
 * speed multiplier via --speed flag).
 *
 * This server stands in for the real src/api/ WebSocket until the Python
 * substrate is wired to a server process. The browser client connects to
 * ws://localhost:<port>/session/<case_id>/events and receives SubstrateEvent
 * JSON frames in the same shape as src/substrate/event_bus.py.
 *
 * Start: npm run mock-server
 * Options:
 *   --port <number>   WebSocket port (default: 8765)
 *   --speed <number>  Time multiplier (default: 1.0 = real-time)
 *   --case <string>   Fixture case_id (default: chest_pain_01)
 *   --loop            Repeat fixture on completion
 *
 * MIT-licensed dependency: ws ^8.x.
 *
 * CLAUDE.md §5.4 — wt-ui scope: ui/.
 */

import { createServer } from "node:http";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { WebSocketServer, type WebSocket } from "ws";

// ---- Fixture types ----

interface TurnPayload {
  turn_id: string;
  session_id: string;
  speaker: "physician" | "patient" | "system";
  text: string;
  ts: number;
  asr_confidence: number;
}

interface ClaimPayload {
  claim_id: string;
  session_id: string;
  subject: string;
  predicate: string;
  value: string;
  value_normalised: string;
  confidence: number;
  source_turn_id: string;
  status: string;
  created_ts: number;
  char_start: number | null;
  char_end: number | null;
}

interface SupersedeEdge {
  edge_id: string;
  old_claim_id: string;
  new_claim_id: string;
  edge_type: string;
  identity_score: number | null;
  created_ts: number;
}

interface BranchScore {
  branch: string;
  log_score: number;
  posterior: number;
  applied: unknown[];
}

interface VerifierPayload {
  why_moved: string[];
  missing_or_contradicting: string[];
  next_best_question: string;
  next_question_rationale: string;
  source_feature: string;
}

interface SoapDelta {
  sentence_id: string;
  session_id: string;
  section: string;
  ordinal: number;
  text: string;
  source_claim_ids: string[];
}

type FixtureEvent =
  | { t_ms: number; type: "turn"; turn: TurnPayload }
  | { t_ms: number; type: "claim"; claim: ClaimPayload }
  | { t_ms: number; type: "supersede"; edge: SupersedeEdge }
  | { t_ms: number; type: "differential"; ranking: { scores: BranchScore[] } }
  | { t_ms: number; type: "verifier"; verifier: VerifierPayload }
  | { t_ms: number; type: "soap_delta"; delta: SoapDelta };

interface Fixture {
  case_id: string;
  events: FixtureEvent[];
}

// ---- CLI args ----

function parseArgs(): { port: number; speed: number; caseId: string; loop: boolean } {
  const argv = process.argv.slice(2);
  let port = 8765;
  let speed = 1.0;
  let caseId = "chest_pain_01";
  let loop = false;
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === "--port" && argv[i + 1]) port = parseInt(argv[++i], 10);
    else if (argv[i] === "--speed" && argv[i + 1]) speed = parseFloat(argv[++i]);
    else if (argv[i] === "--case" && argv[i + 1]) caseId = argv[++i];
    else if (argv[i] === "--loop") loop = true;
  }
  return { port, speed, caseId, loop };
}

// ---- Translate fixture events → SubstrateEvent frames ----

/**
 * Convert a fixture event into one or more SubstrateEvent JSON strings.
 * Matches the wire format expected by ui/src/lib/ws.ts routeEvent().
 */
function toWireFrames(ev: FixtureEvent): string[] {
  switch (ev.type) {
    case "turn":
      return [
        JSON.stringify({ event: "turn.added", payload: { turn_id: ev.turn.turn_id, session_id: ev.turn.session_id } }),
        JSON.stringify({ event: "turn.full", payload: ev.turn }),
      ];
    case "claim":
      return [
        JSON.stringify({
          event: "claim.created",
          payload: { claim_id: ev.claim.claim_id, session_id: ev.claim.session_id, predicate: ev.claim.predicate, status: ev.claim.status },
        }),
        JSON.stringify({ event: "claim.full", payload: ev.claim }),
        JSON.stringify({ event: "projection.updated", payload: { session_id: ev.claim.session_id, active_count: -1 } }),
      ];
    case "supersede":
      return [
        JSON.stringify({
          event: "claim.superseded",
          payload: {
            old_claim_id: ev.edge.old_claim_id,
            new_claim_id: ev.edge.new_claim_id,
            edge_type: ev.edge.edge_type,
            identity_score: ev.edge.identity_score,
          },
        }),
      ];
    case "differential":
      return [JSON.stringify({ event: "ranking.updated", payload: ev.ranking })];
    case "verifier":
      return [JSON.stringify({ event: "verifier.updated", payload: ev.verifier })];
    case "soap_delta":
      return [
        JSON.stringify({
          event: "note_sentence.added",
          payload: { sentence_id: ev.delta.sentence_id, session_id: ev.delta.session_id, section: ev.delta.section, source_claim_ids: ev.delta.source_claim_ids },
        }),
        JSON.stringify({ event: "note_sentence.full", payload: ev.delta }),
      ];
    default:
      return [];
  }
}

// ---- Replay logic ----

function broadcast(clients: Set<WebSocket>, frames: string[]): void {
  for (const client of clients) {
    if (client.readyState === 1 /* OPEN */) {
      for (const frame of frames) {
        client.send(frame);
      }
    }
  }
}

function scheduleReplay(
  fixture: Fixture,
  clients: Set<WebSocket>,
  speed: number,
  loop: boolean,
): void {
  const events = fixture.events;
  const timers: ReturnType<typeof setTimeout>[] = [];

  function schedule(events: FixtureEvent[]): void {
    for (const ev of events) {
      const delay = Math.round(ev.t_ms / speed);
      const t = setTimeout(() => {
        const frames = toWireFrames(ev);
        broadcast(clients, frames);
      }, delay);
      timers.push(t);
    }

    if (loop && events.length > 0) {
      const lastMs = events[events.length - 1].t_ms;
      const loopDelay = Math.round((lastMs + 3000) / speed);
      const loopTimer = setTimeout(() => {
        console.log("[mock-server] Looping fixture...");
        schedule(events);
      }, loopDelay);
      timers.push(loopTimer);
    }
  }

  schedule(events);

  // Return cleanup (not exposed here — process will exit on Ctrl-C)
}

// ---- Main ----

const { port, speed, caseId, loop } = parseArgs();

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const fixturePath = join(__dirname, "..", "fixtures", `${caseId.replace(/[^a-z0-9_-]/gi, "")}_demo.json`);

let fixture: Fixture;
try {
  fixture = JSON.parse(readFileSync(fixturePath, "utf-8")) as Fixture;
  console.log(`[mock-server] Loaded fixture: ${fixturePath} (${fixture.events.length} events)`);
} catch (err) {
  console.error(`[mock-server] Failed to load fixture at ${fixturePath}:`, err);
  process.exit(1);
}

const httpServer = createServer((req, res) => {
  res.writeHead(200, { "Content-Type": "text/plain" });
  res.end(`hack_it mock WebSocket server — connect to ws://localhost:${port}/session/${fixture.case_id}/events\n`);
});

const wss = new WebSocketServer({ server: httpServer, path: `/session/${fixture.case_id}/events` });

const clients = new Set<WebSocket>();

wss.on("connection", (ws, req) => {
  clients.add(ws);
  console.log(`[mock-server] Client connected (${clients.size} total) — ${req.socket.remoteAddress ?? "unknown"}`);

  ws.on("close", () => {
    clients.delete(ws);
    console.log(`[mock-server] Client disconnected (${clients.size} remaining)`);
  });

  ws.on("error", (err) => {
    console.error("[mock-server] WebSocket error:", err.message);
  });
});

httpServer.listen(port, () => {
  console.log(`[mock-server] Listening on ws://localhost:${port}/session/${fixture.case_id}/events`);
  console.log(`[mock-server] Speed: ${speed}x | Loop: ${loop}`);
  scheduleReplay(fixture, clients, speed, loop);
});

process.on("SIGINT", () => {
  console.log("\n[mock-server] Shutting down.");
  wss.close();
  httpServer.close();
  process.exit(0);
});
