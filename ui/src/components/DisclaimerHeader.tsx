import { Button } from "@/components/ui/button";
import { useSession } from "@/store/session";

/**
 * Persistent disclaimer header — rules.md §3.7.
 *
 * Verbatim text is load-bearing compliance content. Not abbreviated, not
 * collapsed, not hidden behind a hover. Always visible (rules.md §3.7).
 *
 * Colour: red rule bar for regulatory salience; body text stays in the
 * muted palette to avoid sensationalising (rules.md §9.1).
 */
export function DisclaimerHeader({ onEndEncounter }: { onEndEncounter: () => void }) {
  const sessionId = useSession((s) => s.sessionId);

  return (
    <header
      role="banner"
      aria-label="Regulatory disclaimer"
      className="relative flex shrink-0 items-stretch border-b border-border bg-disclaimer-bg"
    >
      {/* Left accent rule — colour is --disclaimer-border (cardinal red) */}
      <div aria-hidden className="w-1 shrink-0 bg-disclaimer-border" />

      <div className="flex flex-1 items-start gap-4 px-5 py-2.5">
        {/* Warning icon */}
        <svg
          aria-hidden
          viewBox="0 0 20 20"
          fill="currentColor"
          className="mt-0.5 h-4 w-4 shrink-0 text-disclaimer-border"
        >
          <path
            fillRule="evenodd"
            d="M8.485 2.495c.673-1.167 2.357-1.167 3.03 0l6.28 10.875c.673 1.167-.17
               2.625-1.516 2.625H3.72c-1.347 0-2.189-1.458-1.515-2.625L8.485 2.495zM10
               5a.75.75 0 01.75.75v3.5a.75.75 0 01-1.5 0v-3.5A.75.75 0 0110 5zm0 9a1
               1 0 100-2 1 1 0 000 2z"
            clipRule="evenodd"
          />
        </svg>

        {/* Verbatim text — rules.md §3.7. DO NOT abbreviate or collapse. */}
        <div className="flex flex-col gap-0.5">
          <p className="text-[0.8rem] font-semibold leading-snug text-disclaimer-border">
            Research prototype. Not a medical device.
          </p>
          <p className="max-w-5xl text-[0.74rem] leading-snug text-fg-muted">
            This software is a research demonstration. It does not diagnose, treat, or recommend
            treatment, and is not intended for use in patient care. All patients, conversations,
            and data shown are synthetic or from published research benchmarks. Clinical decisions
            are made by the physician. This system supports the physician&apos;s reasoning by
            tracking claims and differential hypotheses in real time; it does not direct clinical
            judgement.
          </p>
        </div>

        {/* Session controls — right-aligned */}
        <div className="ml-auto flex shrink-0 items-center gap-4">
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
