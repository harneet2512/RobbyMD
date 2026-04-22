/**
 * transcriptSlice.test.ts
 *
 * Contract-level tests for the Zustand transcript slice (turn dispatch,
 * selection axis, and selector shape). Per CLAUDE.md §5.4 — Vitest only,
 * no Playwright / visual tests.
 *
 * These tests exercise the store in isolation — no React, no DOM rendering.
 * Zustand stores are plain JS objects and can be called directly.
 */

import { beforeEach, describe, expect, it } from "vitest";
import { useSession } from "@/store/session";
import type { Turn } from "@/types/substrate";

// Reset store between tests so each test starts from a known state.
beforeEach(() => {
  useSession.getState().endSession();
  // Manually clear all collections so tests are fully isolated.
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

const MOCK_TURN: Turn = {
  turn_id: "t01",
  session_id: "sess_test_01",
  speaker: "physician",
  text: "Good afternoon. What brings you in today?",
  ts: 1000000000,
  asr_confidence: 0.97,
};

const MOCK_TURN_2: Turn = {
  turn_id: "t02",
  session_id: "sess_test_01",
  speaker: "patient",
  text: "I have chest pain.",
  ts: 4000000000,
  asr_confidence: 0.93,
};

describe("turn dispatch → state", () => {
  it("upsertTurn adds a new turn to turns map and turnOrder", () => {
    const { upsertTurn } = useSession.getState();
    upsertTurn(MOCK_TURN);

    const s = useSession.getState();
    expect(s.turns["t01"]).toEqual(MOCK_TURN);
    expect(s.turnOrder).toContain("t01");
    expect(s.turnOrder).toHaveLength(1);
  });

  it("upsertTurn updates existing turn without duplicating turnOrder entry", () => {
    const { upsertTurn } = useSession.getState();
    upsertTurn(MOCK_TURN);
    // Update the same turn_id with different text.
    upsertTurn({ ...MOCK_TURN, text: "updated text" });

    const s = useSession.getState();
    expect(s.turns["t01"]?.text).toBe("updated text");
    // turnOrder should still have only one entry for t01.
    expect(s.turnOrder.filter((id) => id === "t01")).toHaveLength(1);
  });

  it("multiple turns append to turnOrder in insertion order", () => {
    const { upsertTurn } = useSession.getState();
    upsertTurn(MOCK_TURN);
    upsertTurn(MOCK_TURN_2);

    const s = useSession.getState();
    expect(s.turnOrder).toEqual(["t01", "t02"]);
    expect(Object.keys(s.turns)).toHaveLength(2);
  });
});

describe("selection axis", () => {
  it("selectTurn sets selectedAxis=turn and clears other axes", () => {
    const { upsertTurn, selectTurn } = useSession.getState();
    upsertTurn(MOCK_TURN);
    selectTurn("t01");

    const s = useSession.getState();
    expect(s.selectedAxis).toBe("turn");
    expect(s.selectedTurnId).toBe("t01");
    expect(s.selectedClaimId).toBeNull();
    expect(s.selectedSentenceId).toBeNull();
  });

  it("selectTurn(null) clears selection axis to null", () => {
    const { selectTurn } = useSession.getState();
    selectTurn("t01");
    selectTurn(null);

    const s = useSession.getState();
    expect(s.selectedAxis).toBeNull();
    expect(s.selectedTurnId).toBeNull();
  });

  it("clearSelection resets all axes", () => {
    const { selectTurn, clearSelection } = useSession.getState();
    selectTurn("t01");
    clearSelection();

    const s = useSession.getState();
    expect(s.selectedAxis).toBeNull();
    expect(s.selectedTurnId).toBeNull();
  });

  it("switching from turn to claim axis clears selectedTurnId", () => {
    const { selectTurn, selectClaim } = useSession.getState();
    selectTurn("t01");
    selectClaim("c01");

    const s = useSession.getState();
    expect(s.selectedAxis).toBe("claim");
    expect(s.selectedClaimId).toBe("c01");
    expect(s.selectedTurnId).toBeNull();
  });
});

describe("startSession / endSession lifecycle", () => {
  it("startSession resets all collections and sets sessionId", () => {
    const { upsertTurn, startSession } = useSession.getState();
    upsertTurn(MOCK_TURN);
    startSession("sess_new");

    const s = useSession.getState();
    expect(s.sessionId).toBe("sess_new");
    expect(s.turnOrder).toHaveLength(0);
    expect(Object.keys(s.turns)).toHaveLength(0);
  });

  it("endSession clears sessionId", () => {
    const { startSession, endSession } = useSession.getState();
    startSession("sess_abc");
    endSession();

    expect(useSession.getState().sessionId).toBeNull();
  });
});

describe("selector shape contract", () => {
  it("turns record is keyed by turn_id and values match Turn interface", () => {
    useSession.getState().upsertTurn(MOCK_TURN);
    const { turns } = useSession.getState();
    const t = turns["t01"];
    expect(t).toBeDefined();
    // Check required fields present.
    expect(typeof t?.turn_id).toBe("string");
    expect(typeof t?.session_id).toBe("string");
    expect(["patient", "physician", "system"]).toContain(t?.speaker);
    expect(typeof t?.text).toBe("string");
    expect(typeof t?.ts).toBe("number");
  });
});
