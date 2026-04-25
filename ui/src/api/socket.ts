import type { ServerMessage } from "./types";
import { useEncounterStore } from "@/store/encounter";
import { useTraceStore } from "@/store/trace";
import { claimToProse, groupClaimsByTurn } from "@/lib/claim-prose";
import { DECISION_GLYPHS } from "@/lib/glyphs";

let ws: WebSocket | null = null;
let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

function handleMessage(msg: ServerMessage) {
  const encounter = useEncounterStore.getState();
  const trace = useTraceStore.getState();

  switch (msg.type) {
    case "snapshot": {
      encounter.applySnapshot(msg.data);
      const grouped = groupClaimsByTurn(msg.data.claims);
      for (const entry of grouped) {
        trace.append(entry);
      }
      break;
    }

    case "claim.created": {
      const d = msg.data;
      encounter.addClaim(d);

      const prose = claimToProse(d.predicate, d.value);
      const turnKey = `turn-${d.source_turn_id}`;

      trace.appendOrMerge({
        id: turnKey,
        ts_ns: d.created_ts_ns,
        kind: "claims",
        fragments: [prose],
        claimIds: [d.claim_id],
        sourceTurnText: d.source_turn_text,
        sourceTurnSpeaker: d.source_turn_speaker,
        confidence: d.confidence,
      });
      break;
    }

    case "claim.superseded":
      trace.markSuperseded(msg.data.old_claim_id);
      break;

    case "ranking.updated":
      encounter.updateRanking(msg.data);
      break;

    case "verifier.updated":
      encounter.updateVerifier(msg.data);
      if (msg.data.next_best_question) {
        trace.append({
          id: `vq-${Date.now()}`,
          ts_ns: Date.now() * 1_000_000,
          kind: "verifier",
          fragments: [msg.data.next_best_question],
          claimIds: [],
          confidence: 1,
        });
      }
      break;

    case "decision.recorded": {
      const d = msg.data;
      const glyph = DECISION_GLYPHS[d.action] ?? d.action;
      trace.append({
        id: `dec-${d.claim_id}-${d.ts_ns}`,
        ts_ns: d.ts_ns,
        kind: "decision",
        fragments: [d.hypothesis],
        claimIds: [d.claim_id],
        decisionGlyph: glyph,
        decisionInitials: d.initials,
        confidence: 1,
      });
      break;
    }

    case "asr.gap":
      trace.append({
        id: `gap-${msg.data.ts_ns}`,
        ts_ns: msg.data.ts_ns,
        kind: "gap",
        fragments: [`audio unclear — ${msg.data.duration_s}s not captured`],
        claimIds: [],
        confidence: 1,
      });
      break;
  }
}

export function connect(sessionId: string) {
  if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return;

  const url = `ws://${window.location.hostname}:8420/ws/${sessionId}`;
  ws = new WebSocket(url);

  ws.onopen = () => {
    useEncounterStore.getState().setConnected(true);
    if (reconnectTimer) {
      clearTimeout(reconnectTimer);
      reconnectTimer = null;
    }
  };

  ws.onmessage = (ev) => {
    try {
      const msg: ServerMessage = JSON.parse(ev.data);
      handleMessage(msg);
    } catch {
      /* malformed message */
    }
  };

  ws.onclose = () => {
    useEncounterStore.getState().setConnected(false);
    reconnectTimer = setTimeout(() => connect(sessionId), 2000);
  };

  ws.onerror = () => ws?.close();
}

export function sendDecision(action: string, claimId: string) {
  if (ws?.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: `decision.${action}`, claim_id: claimId }));
  }
}

export function sendClose() {
  if (ws?.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "encounter.close" }));
  }
}
