import { useCallback, useEffect, useMemo, useRef } from "react";
import { cn } from "@/lib/cn";
import { useSession } from "@/store/session";
import { claimIdsForTurn, turnIdForSentence } from "@/lib/provenance";

/**
 * Panel 1 — Transcript (PRD.md §6.1).
 *
 * - Live, scrolling, diarised.
 * - Each turn shows speaker, relative time, text.
 * - Click a turn → highlights every downstream claim + note sentence
 *   (rules.md §4.5 provenance-is-clickable; the "hero" interaction).
 * - When a claim or sentence is selected elsewhere, this panel reflects
 *   that selection on its side (dim non-matching turns).
 *
 * First panel shipped end-to-end per CLAUDE.md §5.4 — validates
 * ASR → substrate → UI data flow before investing in other panels.
 */

function formatRelTime(tsNs: number, firstTsNs: number | null): string {
  if (firstTsNs == null) return "00:00";
  const deltaMs = Math.max(0, (tsNs - firstTsNs) / 1_000_000);
  const totalSec = Math.floor(deltaMs / 1000);
  const m = Math.floor(totalSec / 60);
  const s = totalSec % 60;
  return `${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`;
}

export function TranscriptPanel() {
  const turns = useSession((s) => s.turns);
  const turnOrder = useSession((s) => s.turnOrder);
  const claims = useSession((s) => s.claims);
  const claimOrder = useSession((s) => s.claimOrder);
  const sentences = useSession((s) => s.noteSentences);
  const selectedAxis = useSession((s) => s.selectedAxis);
  const selectedTurnId = useSession((s) => s.selectedTurnId);
  const selectedClaimId = useSession((s) => s.selectedClaimId);
  const selectedSentenceId = useSession((s) => s.selectedSentenceId);
  const selectTurn = useSession((s) => s.selectTurn);

  const scrollRef = useRef<HTMLDivElement>(null);
  // Auto-scroll pauses while the user is reading above; resumes at bottom.
  // userScrolledUp tracks whether the user has manually scrolled away from
  // the bottom so we don't fight them mid-read.
  const userScrolledUp = useRef(false);

  const handleScroll = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;
    // "at bottom" = within 40px of scrollHeight — accounts for rounding.
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
    userScrolledUp.current = !atBottom;
  }, []);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    // Resume auto-scroll only if user is at the bottom.
    if (!userScrolledUp.current) {
      el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
    }
  }, [turnOrder.length]);

  // Compute the highlighted turn set for any active selection axis.
  const highlightedTurns = useMemo<Set<string>>(() => {
    if (selectedAxis === "turn" && selectedTurnId) {
      return new Set([selectedTurnId]);
    }
    if (selectedAxis === "claim" && selectedClaimId) {
      const c = claims[selectedClaimId];
      return new Set(c ? [c.source_turn_id] : []);
    }
    if (selectedAxis === "sentence" && selectedSentenceId) {
      const tid = turnIdForSentence(claims, sentences, selectedSentenceId);
      return new Set(tid ? [tid] : []);
    }
    return new Set();
  }, [
    selectedAxis,
    selectedTurnId,
    selectedClaimId,
    selectedSentenceId,
    claims,
    sentences,
  ]);

  const firstTs = useMemo(() => {
    const first = turnOrder[0];
    if (!first) return null;
    return turns[first]?.ts ?? null;
  }, [turnOrder, turns]);

  const hasSelection = selectedAxis != null;

  const turnClaimsCount = (turnId: string): number =>
    claimIdsForTurn(claims, claimOrder, turnId).length;

  return (
    <section
      aria-labelledby="panel-transcript-title"
      className="panel h-full min-h-0"
    >
      <div className="panel-header">
        <div className="flex flex-col">
          <span className="panel-eyebrow">Panel 01 · Transcript</span>
          <h2 id="panel-transcript-title" className="panel-title">
            live dialogue
          </h2>
        </div>
        <span className="font-mono text-2xs text-fg-subtle">
          {turnOrder.length} turn{turnOrder.length === 1 ? "" : "s"}
        </span>
      </div>

      <div ref={scrollRef} onScroll={handleScroll} className="scroll-y flex-1 px-4 py-3">
        {turnOrder.length === 0 ? (
          <div className="flex h-full items-center justify-center text-sm text-fg-subtle">
            <span className="animate-pulse-soft">waiting for dialogue…</span>
          </div>
        ) : (
          <ul className="flex flex-col gap-2.5">
            {turnOrder.map((tid) => {
              const t = turns[tid];
              if (!t) return null;
              const isHighlighted = highlightedTurns.has(tid);
              const isSelected = selectedTurnId === tid;
              const dim = hasSelection && !isHighlighted;
              const nClaims = turnClaimsCount(tid);
              return (
                <li key={tid}>
                  <button
                    type="button"
                    onClick={() =>
                      isSelected ? selectTurn(null) : selectTurn(tid)
                    }
                    aria-pressed={isSelected}
                    className={cn(
                      "group relative block w-full rounded border border-transparent px-3 py-2 text-left transition-all duration-200 ease-breathe",
                      "hover:border-border-strong hover:bg-surface-2",
                      "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/60",
                      isHighlighted && "prov-highlight",
                      dim && "prov-dim",
                    )}
                  >
                    <div className="mb-1 flex items-baseline justify-between gap-2">
                      <span
                        className={cn(
                          "panel-eyebrow",
                          t.speaker === "physician"
                            ? "text-branch-pulmonary"
                            : t.speaker === "patient"
                              ? "text-branch-msk"
                              : "text-fg-subtle",
                        )}
                      >
                        {t.speaker}
                      </span>
                      <span className="font-mono text-2xs text-fg-subtle">
                        {formatRelTime(t.ts, firstTs)} · {tid}
                      </span>
                    </div>
                    <p className="text-[0.88rem] leading-relaxed text-fg">
                      {t.text}
                    </p>
                    {nClaims > 0 ? (
                      <div className="mt-1.5 flex items-center gap-1 text-2xs text-fg-subtle">
                        <span className="h-px w-3 bg-border-strong" />
                        <span>
                          {nClaims} claim{nClaims === 1 ? "" : "s"} extracted
                        </span>
                      </div>
                    ) : null}
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
