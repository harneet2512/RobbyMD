/**
 * ws.test.ts
 *
 * Contract tests for the WebSocket utility module (ui/src/lib/ws.ts).
 * Tests focus on the pure functions — parseWsMessage and routeEvent — since
 * connectWs requires a real WebSocket which is not available in jsdom.
 *
 * Per CLAUDE.md §5.4: Vitest only, no Playwright / visual tests.
 * Per dispatch brief §8: unknown event kinds must not crash.
 */

import { beforeEach, describe, expect, it, vi } from "vitest";
import { parseWsMessage, routeEvent } from "@/lib/ws";
import { useSession } from "@/store/session";

// Reset store state before each test.
beforeEach(() => {
  useSession.setState({
    sessionId: null,
    turns: {},
    turnOrder: [],
    claims: {},
    claimOrder: [],
    edges: [],
    noteSentences: {},
    noteOrder: [],
    ranking: null,
    verifier: null,
    selectedAxis: null,
    selectedTurnId: null,
    selectedClaimId: null,
    selectedSentenceId: null,
  });
});

// ---- parseWsMessage ----

describe("parseWsMessage", () => {
  it("returns null for invalid JSON", () => {
    expect(parseWsMessage("not-json")).toBeNull();
  });

  it("returns null for JSON without event field", () => {
    expect(parseWsMessage(JSON.stringify({ foo: "bar" }))).toBeNull();
  });

  it("returns null for unknown event type", () => {
    const msg = JSON.stringify({ event: "unknown.future.event", payload: {} });
    expect(parseWsMessage(msg)).toBeNull();
  });

  it("returns null for empty string", () => {
    expect(parseWsMessage("")).toBeNull();
  });

  it("parses turn.full event correctly", () => {
    const payload = {
      turn_id: "t01",
      session_id: "sess_test",
      speaker: "physician",
      text: "Hello",
      ts: 1000000000,
      asr_confidence: 0.95,
    };
    const msg = JSON.stringify({ event: "turn.full", payload });
    const result = parseWsMessage(msg);
    expect(result).not.toBeNull();
    expect(result?.event).toBe("turn.full");
  });

  it("parses claim.full event correctly", () => {
    const payload = {
      claim_id: "c01",
      session_id: "sess_test",
      subject: "patient",
      predicate: "onset",
      value: "yesterday",
      value_normalised: "yesterday",
      confidence: 0.9,
      source_turn_id: "t01",
      status: "active",
      created_ts: 1000000000,
      char_start: 0,
      char_end: 9,
    };
    const msg = JSON.stringify({ event: "claim.full", payload });
    const result = parseWsMessage(msg);
    expect(result?.event).toBe("claim.full");
  });

  it("parses ranking.updated event", () => {
    const payload = { scores: [{ branch: "cardiac", log_score: 3.1, posterior: 0.65, applied: [] }] };
    const msg = JSON.stringify({ event: "ranking.updated", payload });
    const result = parseWsMessage(msg);
    expect(result?.event).toBe("ranking.updated");
  });

  it("parses verifier.updated event", () => {
    const payload = {
      why_moved: ["Cardiac up: exertional pain."],
      missing_or_contradicting: ["Wells criteria not elicited."],
      next_best_question: "Any leg swelling or recent travel?",
      next_question_rationale: "Separates PE from cardiac.",
      source_feature: "wells_pe_immobilisation",
    };
    const msg = JSON.stringify({ event: "verifier.updated", payload });
    const result = parseWsMessage(msg);
    expect(result?.event).toBe("verifier.updated");
  });
});

// ---- routeEvent ----

describe("routeEvent", () => {
  it("turn.full event calls upsertTurn and adds to store", () => {
    const turn = {
      turn_id: "t01",
      session_id: "sess_test",
      speaker: "physician" as const,
      text: "Good afternoon.",
      ts: 1000000000,
      asr_confidence: 0.97,
    };
    routeEvent({ event: "turn.full", payload: turn });

    const s = useSession.getState();
    expect(s.turns["t01"]).toEqual(turn);
    expect(s.turnOrder).toContain("t01");
  });

  it("claim.full event calls upsertClaim and adds to store", () => {
    const claim = {
      claim_id: "c01",
      session_id: "sess_test",
      subject: "patient",
      predicate: "onset",
      value: "yesterday",
      value_normalised: "yesterday",
      confidence: 0.9,
      source_turn_id: "t01",
      status: "active" as const,
      created_ts: 1000000000,
      char_start: 0,
      char_end: 9,
    };
    routeEvent({ event: "claim.full", payload: claim });

    const s = useSession.getState();
    expect(s.claims["c01"]).toEqual(claim);
    expect(s.claimOrder).toContain("c01");
  });

  it("claim.superseded marks old claim as superseded and appends edge", () => {
    // Set up an existing claim to be superseded.
    useSession.getState().upsertClaim({
      claim_id: "c_old",
      session_id: "sess_test",
      subject: "patient",
      predicate: "onset",
      value: "three_days_ago",
      value_normalised: "three_days_ago",
      confidence: 0.8,
      source_turn_id: "t02",
      status: "active",
      created_ts: 1000000000,
      char_start: 0,
      char_end: 10,
    });

    routeEvent({
      event: "claim.superseded",
      payload: {
        old_claim_id: "c_old",
        new_claim_id: "c_new",
        edge_type: "patient_correction",
        identity_score: 0.97,
      },
    });

    const s = useSession.getState();
    expect(s.claims["c_old"]?.status).toBe("superseded");
    expect(s.edges).toHaveLength(1);
    expect(s.edges[0]?.edge_type).toBe("patient_correction");
  });

  it("ranking.updated event updates ranking in store", () => {
    const ranking = {
      scores: [
        { branch: "cardiac", log_score: 5.2, posterior: 0.67, applied: [] },
        { branch: "pulmonary", log_score: 0.3, posterior: 0.15, applied: [] },
        { branch: "msk", log_score: -1.1, posterior: 0.11, applied: [] },
        { branch: "gi", log_score: -1.5, posterior: 0.07, applied: [] },
      ],
    };
    routeEvent({ event: "ranking.updated", payload: ranking });

    const s = useSession.getState();
    expect(s.ranking).not.toBeNull();
    expect(s.ranking?.scores).toHaveLength(4);
    expect(s.ranking?.scores[0]?.branch).toBe("cardiac");
  });

  it("verifier.updated event updates verifier in store", () => {
    const verifier = {
      why_moved: ["Cardiac up."],
      missing_or_contradicting: ["Wells not elicited."],
      next_best_question: "Any leg swelling?",
      next_question_rationale: "Separates PE from cardiac.",
      source_feature: "wells_pe_immobilisation",
    };
    routeEvent({ event: "verifier.updated", payload: verifier });

    const s = useSession.getState();
    expect(s.verifier?.next_best_question).toBe("Any leg swelling?");
    expect(s.verifier?.why_moved).toHaveLength(1);
  });

  it("turn.added event (notification-only) does not modify turns", () => {
    const spy = vi.spyOn(useSession.getState(), "upsertTurn");
    routeEvent({ event: "turn.added", payload: { turn_id: "t01", session_id: "sess_test" } });
    // turn.added is notification-only; the store should not be mutated until turn.full arrives.
    expect(spy).not.toHaveBeenCalled();
    expect(useSession.getState().turnOrder).toHaveLength(0);
  });

  it("note_sentence.full event adds sentence to store", () => {
    const sentence = {
      sentence_id: "s01",
      session_id: "sess_test",
      section: "S" as const,
      ordinal: 1,
      text: "Patient reports chest pain.",
      source_claim_ids: ["c01", "c02"],
    };
    routeEvent({ event: "note_sentence.full", payload: sentence });

    const s = useSession.getState();
    expect(s.noteSentences["s01"]).toEqual(sentence);
    expect(s.noteOrder).toContain("s01");
  });
});
