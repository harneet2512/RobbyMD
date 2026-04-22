import { useMemo } from "react";
import { cn } from "@/lib/cn";
import { Badge } from "@/components/ui/badge";
import { useSession } from "@/store/session";
import type { ClaimStatus } from "@/types/substrate";

/**
 * Panel 2 — Claim state (PRD.md §6.2).
 *
 * Shows every extracted claim with lifecycle status (active / superseded
 * / confirmed / dismissed). Superseded claims struck-through with an
 * arrow to their replacement.
 *
 * Clicking a claim becomes the "claim" selection axis (highlights source
 * turn + downstream sentences). Reciprocal with TranscriptPanel.
 */

const TONE_BY_STATUS: Record<ClaimStatus, "active" | "superseded" | "confirmed" | "dismissed"> = {
  active: "active",
  superseded: "superseded",
  confirmed: "confirmed",
  dismissed: "dismissed",
};

export function ClaimStatePanel() {
  const claims = useSession((s) => s.claims);
  const claimOrder = useSession((s) => s.claimOrder);
  const edges = useSession((s) => s.edges);
  const selectedAxis = useSession((s) => s.selectedAxis);
  const selectedTurnId = useSession((s) => s.selectedTurnId);
  const selectedClaimId = useSession((s) => s.selectedClaimId);
  const selectClaim = useSession((s) => s.selectClaim);

  const highlightedClaims = useMemo<Set<string>>(() => {
    if (selectedAxis === "turn" && selectedTurnId) {
      return new Set(
        claimOrder.filter((cid) => claims[cid]?.source_turn_id === selectedTurnId),
      );
    }
    if (selectedAxis === "claim" && selectedClaimId) {
      return new Set([selectedClaimId]);
    }
    return new Set();
  }, [selectedAxis, selectedTurnId, selectedClaimId, claims, claimOrder]);

  const hasSelection = selectedAxis != null;

  const supersededBy: Record<string, string> = useMemo(() => {
    const out: Record<string, string> = {};
    for (const e of edges) out[e.old_claim_id] = e.new_claim_id;
    return out;
  }, [edges]);

  return (
    <section
      aria-labelledby="panel-claims-title"
      className="panel h-full min-h-0"
    >
      <div className="panel-header">
        <div className="flex flex-col">
          <span className="panel-eyebrow">Panel 02 · Claims</span>
          <h2 id="panel-claims-title" className="panel-title">
            lifecycle state
          </h2>
        </div>
        <span className="font-mono text-2xs text-fg-subtle">
          {claimOrder.length} total
        </span>
      </div>

      <div className="scroll-y flex-1 px-4 py-3">
        {claimOrder.length === 0 ? (
          <div className="flex h-full items-center justify-center text-sm text-fg-subtle">
            <span>no claims extracted yet</span>
          </div>
        ) : (
          <ul className="flex flex-col gap-2">
            {claimOrder.map((cid) => {
              const c = claims[cid];
              if (!c) return null;
              const isHL = highlightedClaims.has(cid);
              const dim = hasSelection && !isHL;
              const tone = TONE_BY_STATUS[c.status];
              const replacedBy = supersededBy[cid];
              return (
                <li key={cid}>
                  <button
                    type="button"
                    onClick={() =>
                      selectedClaimId === cid
                        ? selectClaim(null)
                        : selectClaim(cid)
                    }
                    aria-pressed={selectedClaimId === cid}
                    className={cn(
                      "group relative block w-full rounded border border-transparent px-3 py-2 text-left transition-all duration-200 ease-breathe",
                      "hover:border-border-strong hover:bg-surface-2",
                      "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/60",
                      isHL && "prov-highlight",
                      dim && "prov-dim",
                    )}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <div className="flex items-center gap-2">
                        <Badge tone={tone}>{c.status}</Badge>
                        <span className="font-mono text-2xs text-fg-subtle">
                          {cid}
                        </span>
                      </div>
                      <span className="font-mono text-2xs text-fg-subtle">
                        conf {c.confidence.toFixed(2)}
                      </span>
                    </div>
                    <div className="mt-1.5 flex flex-wrap items-baseline gap-x-1.5 gap-y-0.5">
                      <span className="panel-eyebrow text-fg-subtle">
                        {c.predicate}
                      </span>
                      <span
                        className={cn(
                          "font-mono text-xs",
                          c.status === "superseded"
                            ? "text-status-superseded line-through"
                            : "text-fg",
                        )}
                      >
                        {c.value}
                      </span>
                      {replacedBy ? (
                        <span className="font-mono text-2xs text-fg-subtle">
                          → {replacedBy}
                        </span>
                      ) : null}
                    </div>
                    <div className="mt-1 font-mono text-2xs text-fg-subtle">
                      from {c.source_turn_id}
                    </div>
                  </button>
                </li>
              );
            })}
          </ul>
        )}
      </div>
    </section>
  );
}
