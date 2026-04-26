/**
 * End-to-end integration test — fixture → MockServer → routeEvent → store.
 *
 * Exercises the full demo data path without a browser: the same pipeline
 * the demo video is recorded against. If this passes, the 4 panels will
 * render real data during the mock replay.
 *
 * Rules:
 *   - Fake timers so the fixture's t_ms schedule advances deterministically.
 *   - Reset session state before every test to avoid cross-test leakage.
 *   - Disconnect (which calls endSession) in afterEach.
 *
 * Fixture: `ui/fixtures/chest_pain_demo.json` — 30 turns, 23 claims,
 * 2 supersessions (patient correction + severity refinement), 5 differential
 * updates, 8 SOAP sentences, 5 verifier outputs. Max t_ms = 184000.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { connectMock, type Connection } from "@/api/client";
import { useSession } from "@/store/session";

const RESET = {
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
};

describe("fixture replay end-to-end", () => {
  let conn: Connection | null = null;

  beforeEach(() => {
    vi.useFakeTimers();
    useSession.setState(RESET);
  });

  afterEach(() => {
    conn?.disconnect();
    conn = null;
    vi.useRealTimers();
  });

  it("full fixture lands in store: 30 turns, 23 claims, 2 edges, 5 rankings, 8 sentences, 5 verifiers", () => {
    conn = connectMock();
    vi.advanceTimersByTime(200_000);

    const s = useSession.getState();
    expect(s.turnOrder).toHaveLength(30);
    expect(s.claimOrder).toHaveLength(23);
    expect(s.edges).toHaveLength(2);
    expect(s.noteOrder).toHaveLength(8);
    expect(s.ranking).not.toBeNull();
    expect(s.ranking?.scores).toHaveLength(4);
    expect(s.verifier).not.toBeNull();
  });

  it("supersession: c02 → c02b marks c02 superseded and leaves c02b active", () => {
    conn = connectMock();
    vi.advanceTimersByTime(25_000);

    const s = useSession.getState();
    expect(s.edges[0]?.old_claim_id).toBe("c02");
    expect(s.edges[0]?.new_claim_id).toBe("c02b");
    expect(s.edges[0]?.edge_type).toBe("patient_correction");
    expect(s.claims["c02"]?.status).toBe("superseded");
    expect(s.claims["c02b"]?.status).toBe("active");
  });

  it("final ranking: cardiac ~0.84 and posteriors sum to 1.0", () => {
    conn = connectMock();
    vi.advanceTimersByTime(200_000);

    const s = useSession.getState();
    const cardiac = s.ranking?.scores.find((x) => x.branch === "cardiac");
    expect(cardiac?.posterior).toBeCloseTo(0.84, 2);

    const sum = (s.ranking?.scores ?? []).reduce((acc, x) => acc + x.posterior, 0);
    expect(sum).toBeCloseTo(1.0, 2);
  });

  it("SOAP sentences only cite claim ids that exist in the store", () => {
    conn = connectMock();
    vi.advanceTimersByTime(200_000);

    const s = useSession.getState();
    for (const sid of s.noteOrder) {
      const sentence = s.noteSentences[sid];
      expect(sentence).toBeDefined();
      for (const cid of sentence!.source_claim_ids) {
        expect(s.claims[cid], `sentence ${sid} cites unknown claim ${cid}`).toBeDefined();
      }
    }
  });

  it("verifier emits why_moved, gap list, and next-best-question with rationale", () => {
    conn = connectMock();
    vi.advanceTimersByTime(200_000);

    const v = useSession.getState().verifier;
    expect(v?.why_moved.length).toBeGreaterThan(0);
    expect(v?.missing_or_contradicting.length).toBeGreaterThan(0);
    expect(v?.next_best_question.length).toBeGreaterThan(0);
    expect(v?.next_question_rationale.length).toBeGreaterThan(0);
  });

  it("turn order preserves fixture insertion order (t01..t30)", () => {
    conn = connectMock();
    vi.advanceTimersByTime(200_000);

    const expected = Array.from({ length: 30 }, (_, i) => `t${String(i + 1).padStart(2, "0")}`);
    expect(useSession.getState().turnOrder).toEqual(expected);
  });
});
