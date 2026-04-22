import * as React from "react";
import { cn } from "@/lib/cn";

type Tone = "neutral" | "active" | "superseded" | "confirmed" | "dismissed";

const toneClass: Record<Tone, string> = {
  neutral: "border-border-strong text-fg-muted bg-surface-2",
  active: "border-status-active/40 text-status-active bg-status-active/10",
  superseded:
    "border-status-superseded/40 text-status-superseded bg-status-superseded/10 line-through",
  confirmed:
    "border-status-confirmed/40 text-status-confirmed bg-status-confirmed/10",
  dismissed:
    "border-status-dismissed/40 text-status-dismissed bg-status-dismissed/10",
};

export function Badge({
  tone = "neutral",
  className,
  children,
  ...rest
}: { tone?: Tone } & React.HTMLAttributes<HTMLSpanElement>) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-sm border px-1.5 py-0.5 text-2xs uppercase tracking-eyebrow",
        toneClass[tone],
        className,
      )}
      {...rest}
    >
      {children}
    </span>
  );
}
