import type { Claim, NoteSentence } from "@/types/substrate";

/**
 * Client-side mirror of `src/substrate/provenance.py` forward/back-link
 * utilities. Same semantics; operates on the Zustand-held entity maps
 * instead of SQLite rows.
 *
 * Used by the provenance-highlight interaction (rules.md §4.5,
 * PRD.md §6.1 / §6.4).
 */

export function claimIdsForTurn(
  claims: Record<string, Claim>,
  claimOrder: string[],
  turnId: string,
): string[] {
  return claimOrder.filter((id) => {
    const c = claims[id];
    return c != null && c.source_turn_id === turnId;
  });
}

export function sentenceIdsForClaim(
  sentences: Record<string, NoteSentence>,
  noteOrder: string[],
  claimId: string,
): string[] {
  return noteOrder.filter((id) => {
    const s = sentences[id];
    return s != null && s.source_claim_ids.includes(claimId);
  });
}

export function turnIdForSentence(
  claims: Record<string, Claim>,
  sentences: Record<string, NoteSentence>,
  sentenceId: string,
): string | null {
  const s = sentences[sentenceId];
  if (!s || s.source_claim_ids.length === 0) return null;
  const firstClaimId = s.source_claim_ids[0];
  if (!firstClaimId) return null;
  const c = claims[firstClaimId];
  return c ? c.source_turn_id : null;
}

export function sentenceIdsForTurn(
  claims: Record<string, Claim>,
  claimOrder: string[],
  sentences: Record<string, NoteSentence>,
  noteOrder: string[],
  turnId: string,
): string[] {
  const claimIds = claimIdsForTurn(claims, claimOrder, turnId);
  const hit = new Set<string>();
  for (const cid of claimIds) {
    for (const sid of sentenceIdsForClaim(sentences, noteOrder, cid)) {
      hit.add(sid);
    }
  }
  return [...hit];
}
