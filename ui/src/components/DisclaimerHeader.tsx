import { Button } from "@/components/ui/button";
import { useSession } from "@/store/session";

/**
 * Persistent disclaimer header — rules.md §3.7.
 *
 * The verbatim disclaimer text is load-bearing compliance content. Not
 * abbreviated, not collapsed, not hidden behind a hover.
 *
 * Layout: left — bright red vertical rule + disclaimer body. Right —
 * session id + end-encounter button. Full-bleed at 1920×1080 (Eng_doc.md §7).
 */
export function DisclaimerHeader({ onEndEncounter }: { onEndEncounter: () => void }) {
  const sessionId = useSession((s) => s.sessionId);

  return (
    <header
      role="banner"
      className="relative flex shrink-0 items-stretch border-b border-border bg-disclaimer-bg"
    >
      <div
        aria-hidden
        className="w-1 shrink-0 bg-disclaimer-border"
      />
      <div className="flex flex-1 items-center gap-6 px-6 py-3">
        <div className="flex flex-col">
          <span className="panel-eyebrow text-disclaimer-border">
            Research prototype — not a medical device
          </span>
          <span className="mt-0.5 text-[0.8rem] leading-snug text-fg-muted">
            Synthetic data. The physician makes every clinical decision.
          </span>
        </div>
        <div className="ml-auto flex items-center gap-4">
          <div className="flex flex-col items-end text-right">
            <span className="panel-eyebrow text-fg-subtle">session</span>
            <span className="font-mono text-xs text-fg-muted">
              {sessionId ?? "—"}
            </span>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={onEndEncounter}
            disabled={!sessionId}
          >
            End encounter
          </Button>
        </div>
      </div>
    </header>
  );
}
