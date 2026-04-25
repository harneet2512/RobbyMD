import { create } from "zustand";
import type { TraceEntry } from "@/lib/claim-prose";

interface TraceStore {
  entries: TraceEntry[];
  append: (entry: TraceEntry) => void;
  appendOrMerge: (entry: TraceEntry) => void;
  markSuperseded: (claimId: string) => void;
  clear: () => void;
}

export const useTraceStore = create<TraceStore>((set) => ({
  entries: [],

  append: (entry) =>
    set((s) => {
      if (s.entries.some((e) => e.id === entry.id)) return s;
      return { entries: [...s.entries, entry] };
    }),

  appendOrMerge: (entry) =>
    set((s) => {
      const existing = s.entries.find((e) => e.id === entry.id);
      if (existing) {
        const newFragments = entry.fragments.filter(
          (f) => !existing.fragments.includes(f)
        );
        const newClaimIds = entry.claimIds.filter(
          (id) => !existing.claimIds.includes(id)
        );
        if (newFragments.length === 0 && newClaimIds.length === 0) return s;
        const merged: TraceEntry = {
          ...existing,
          fragments: [...existing.fragments, ...newFragments],
          claimIds: [...existing.claimIds, ...newClaimIds],
          confidence: Math.min(existing.confidence, entry.confidence),
        };
        return {
          entries: s.entries.map((e) => (e.id === entry.id ? merged : e)),
        };
      }
      return { entries: [...s.entries, entry] };
    }),

  markSuperseded: (claimId) =>
    set((s) => ({
      entries: s.entries.map((e) =>
        e.claimIds.includes(claimId)
          ? { ...e, supersededClaimId: claimId }
          : e
      ),
    })),

  clear: () => set({ entries: [] }),
}));

export type { TraceEntry };
