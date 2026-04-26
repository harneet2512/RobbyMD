import { create } from "zustand";
import type {
  BranchRanking,
  Claim,
  NoteSentence,
  SupersessionEdge,
  Turn,
  VerifierOutput,
} from "@/types/substrate";

/**
 * Session store — the single source of truth for everything the UI renders.
 *
 * Per Eng_doc.md §7 "no localStorage / sessionStorage" — the substrate is
 * the authority and the store is an in-memory projection of events from the
 * event bus. On reload we reconnect and replay.
 *
 * Selected-id fields drive the provenance-is-hero highlight (rules.md §4.5,
 * PRD.md §6.1). Only one "source" axis at a time — clicking a turn sets
 * selectedTurnId and clears the others, etc.
 */
export type SelectionAxis = "turn" | "claim" | "sentence" | null;
export type UserClaimDecision = "promoted" | "suppressed";
export type UserBranchDecision = "promoted" | "demoted";

interface SessionState {
  // identity
  sessionId: string | null;

  // entities (keyed by id for O(1) patch; arrays derived via selectors)
  turns: Record<string, Turn>;
  turnOrder: string[]; // monotonic append; turn_id in creation order
  claims: Record<string, Claim>;
  claimOrder: string[];
  edges: SupersessionEdge[];
  noteSentences: Record<string, NoteSentence>;
  noteOrder: string[];

  // derived from differential engine + verifier (Panel 3 + aux strip)
  ranking: BranchRanking | null;
  verifier: VerifierOutput | null;

  // UI selection (provenance highlight)
  selectedAxis: SelectionAxis;
  selectedTurnId: string | null;
  selectedClaimId: string | null;
  selectedSentenceId: string | null;
  userSuppressedClaimIds: Set<string>;
  claimDecisionsById: Record<string, UserClaimDecision>;
  branchDecisionsById: Record<string, UserBranchDecision>;

  // ingress — called by api/client.ts when events arrive
  upsertTurn: (turn: Turn) => void;
  upsertClaim: (claim: Claim) => void;
  appendEdge: (edge: SupersessionEdge) => void;
  setClaimStatus: (claimId: string, status: Claim["status"]) => void;
  suppressClaim: (claimId: string) => void;
  unsuppressClaim: (claimId: string) => void;
  clearUserSuppression: () => void;
  recordClaimDecision: (claimId: string, decision: UserClaimDecision | null) => void;
  recordBranchDecision: (branchId: string, decision: UserBranchDecision | null) => void;
  upsertSentence: (sentence: NoteSentence) => void;
  setRanking: (ranking: BranchRanking) => void;
  setVerifier: (verifier: VerifierOutput) => void;

  // selection actions (exactly one axis at a time)
  selectTurn: (turnId: string | null) => void;
  selectClaim: (claimId: string | null) => void;
  selectSentence: (sentenceId: string | null) => void;
  clearSelection: () => void;

  // lifecycle
  startSession: (sessionId: string) => void;
  endSession: () => void;
}

export const useSession = create<SessionState>((set) => ({
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
  userSuppressedClaimIds: new Set(),
  claimDecisionsById: {},
  branchDecisionsById: {},

  upsertTurn: (turn) =>
    set((s) => {
      if (s.turns[turn.turn_id]) {
        return { turns: { ...s.turns, [turn.turn_id]: turn } };
      }
      return {
        turns: { ...s.turns, [turn.turn_id]: turn },
        turnOrder: [...s.turnOrder, turn.turn_id],
      };
    }),

  upsertClaim: (claim) =>
    set((s) => {
      if (s.claims[claim.claim_id]) {
        return { claims: { ...s.claims, [claim.claim_id]: claim } };
      }
      return {
        claims: { ...s.claims, [claim.claim_id]: claim },
        claimOrder: [...s.claimOrder, claim.claim_id],
      };
    }),

  appendEdge: (edge) => set((s) => ({ edges: [...s.edges, edge] })),

  setClaimStatus: (claimId, status) =>
    set((s) => {
      const existing = s.claims[claimId];
      if (!existing) return {};
      return {
        claims: { ...s.claims, [claimId]: { ...existing, status } },
      };
    }),

  suppressClaim: (claimId) =>
    set((s) => {
      const next = new Set(s.userSuppressedClaimIds);
      next.add(claimId);
      return { userSuppressedClaimIds: next };
    }),

  unsuppressClaim: (claimId) =>
    set((s) => {
      const next = new Set(s.userSuppressedClaimIds);
      next.delete(claimId);
      return { userSuppressedClaimIds: next };
    }),

  clearUserSuppression: () => set({ userSuppressedClaimIds: new Set() }),

  recordClaimDecision: (claimId, decision) =>
    set((s) => {
      const next = { ...s.claimDecisionsById };
      if (decision) next[claimId] = decision;
      else delete next[claimId];
      return { claimDecisionsById: next };
    }),

  recordBranchDecision: (branchId, decision) =>
    set((s) => {
      const next = { ...s.branchDecisionsById };
      if (decision) next[branchId] = decision;
      else delete next[branchId];
      return { branchDecisionsById: next };
    }),

  upsertSentence: (sentence) =>
    set((s) => {
      if (s.noteSentences[sentence.sentence_id]) {
        return {
          noteSentences: {
            ...s.noteSentences,
            [sentence.sentence_id]: sentence,
          },
        };
      }
      return {
        noteSentences: {
          ...s.noteSentences,
          [sentence.sentence_id]: sentence,
        },
        noteOrder: [...s.noteOrder, sentence.sentence_id],
      };
    }),

  setRanking: (ranking) => set({ ranking }),
  setVerifier: (verifier) => set({ verifier }),

  selectTurn: (turnId) =>
    set({
      selectedAxis: turnId ? "turn" : null,
      selectedTurnId: turnId,
      selectedClaimId: null,
      selectedSentenceId: null,
    }),
  selectClaim: (claimId) =>
    set({
      selectedAxis: claimId ? "claim" : null,
      selectedTurnId: null,
      selectedClaimId: claimId,
      selectedSentenceId: null,
    }),
  selectSentence: (sentenceId) =>
    set({
      selectedAxis: sentenceId ? "sentence" : null,
      selectedTurnId: null,
      selectedClaimId: null,
      selectedSentenceId: sentenceId,
    }),
  clearSelection: () =>
    set({
      selectedAxis: null,
      selectedTurnId: null,
      selectedClaimId: null,
      selectedSentenceId: null,
    }),

  startSession: (sessionId) =>
    set({
      sessionId,
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
      userSuppressedClaimIds: new Set(),
      claimDecisionsById: {},
      branchDecisionsById: {},
    }),
  endSession: () => set({
    sessionId: null,
    userSuppressedClaimIds: new Set(),
    claimDecisionsById: {},
    branchDecisionsById: {},
  }),
}));
