import { create } from "zustand";
import type {
  EncounterMeta,
  HypothesisState,
  VerifierEvent,
  ClaimEvent,
  RankingEvent,
  Snapshot,
} from "@/api/types";

export interface FragmentState {
  kind: "hypothesis" | "claim";
  id: string;
  anchorX: number;
  anchorY: number;
}

interface EncounterStore {
  session: EncounterMeta | null;
  hypotheses: HypothesisState[];
  previousHypotheses: HypothesisState[];
  claims: Map<string, ClaimEvent>;
  verifier: VerifierEvent | null;
  activeFragment: FragmentState | null;
  connected: boolean;

  applySnapshot: (snap: Snapshot) => void;
  addClaim: (claim: ClaimEvent) => void;
  updateRanking: (ranking: RankingEvent) => void;
  updateVerifier: (v: VerifierEvent) => void;
  openFragment: (
    kind: "hypothesis" | "claim",
    id: string,
    x: number,
    y: number
  ) => void;
  closeFragment: () => void;
  setConnected: (c: boolean) => void;
}

export const useEncounterStore = create<EncounterStore>((set, get) => ({
  session: null,
  hypotheses: [],
  previousHypotheses: [],
  claims: new Map(),
  verifier: null,
  activeFragment: null,
  connected: false,

  applySnapshot: (snap) => {
    const claimMap = new Map<string, ClaimEvent>();
    for (const c of snap.claims) {
      claimMap.set(c.claim_id, c);
    }
    set({
      session: snap.encounter,
      hypotheses: snap.ranking.hypotheses,
      previousHypotheses: [],
      claims: claimMap,
      verifier: snap.verifier,
    });
  },

  addClaim: (claim) => {
    const next = new Map(get().claims);
    next.set(claim.claim_id, claim);
    set({ claims: next });
  },

  updateRanking: (ranking) => {
    set({
      previousHypotheses: get().hypotheses,
      hypotheses: ranking.hypotheses,
    });
  },

  updateVerifier: (v) => set({ verifier: v }),

  openFragment: (kind, id, x, y) =>
    set({ activeFragment: { kind, id, anchorX: x, anchorY: y } }),

  closeFragment: () => set({ activeFragment: null }),

  setConnected: (c) => set({ connected: c }),
}));
