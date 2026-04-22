import { useSession } from "@/store/session";

/**
 * Auxiliary strip — PRD.md §6.5.
 *
 * Three slots: "why this moved" / "missing or contradicting" / "next
 * best question". Data comes from the verifier via `verifier.updated`
 * (the UI wire extension; server-side shape is `VerifierOutput` in
 * `src/verifier/verifier.py`).
 *
 * When no verifier output has landed yet, show three placeholders with
 * the event names so the data contract is visible to reviewers.
 */

function Slot({
  eyebrow,
  title,
  children,
  empty,
  emptyEventName,
}: {
  eyebrow: string;
  title: string;
  children?: React.ReactNode;
  empty: boolean;
  emptyEventName: string;
}) {
  return (
    <div className="flex min-h-[96px] flex-col gap-1.5 border-l border-border px-4 py-3 first:border-l-0">
      <div className="flex items-baseline gap-2">
        <span className="panel-eyebrow text-fg-subtle">{eyebrow}</span>
        <span className="font-display text-sm italic text-fg">{title}</span>
      </div>
      {empty ? (
        <p className="text-2xs text-fg-subtle">
          awaiting <span className="font-mono">{emptyEventName}</span>
        </p>
      ) : (
        <div className="text-[0.85rem] leading-relaxed text-fg">{children}</div>
      )}
    </div>
  );
}

export function AuxStrip() {
  const v = useSession((s) => s.verifier);
  const hasAny = v != null && (v.why_moved.length || v.missing_or_contradicting.length || v.next_best_question.length);

  return (
    <aside
      aria-label="Verifier strip"
      className="panel grid grid-cols-3 p-0"
    >
      <Slot
        eyebrow="Aux · why"
        title="why this moved"
        empty={!hasAny || v == null || v.why_moved.length === 0}
        emptyEventName="verifier.updated"
      >
        {v ? (
          <ul className="list-inside list-disc space-y-0.5">
            {v.why_moved.map((b, i) => (
              <li key={i}>{b}</li>
            ))}
          </ul>
        ) : null}
      </Slot>
      <Slot
        eyebrow="Aux · gap"
        title="missing or contradicting"
        empty={!hasAny || v == null || v.missing_or_contradicting.length === 0}
        emptyEventName="verifier.updated"
      >
        {v ? (
          <ul className="list-inside list-disc space-y-0.5">
            {v.missing_or_contradicting.map((b, i) => (
              <li key={i}>{b}</li>
            ))}
          </ul>
        ) : null}
      </Slot>
      <Slot
        eyebrow="Aux · ask"
        title="next best question"
        empty={!hasAny || v == null || v.next_best_question.length === 0}
        emptyEventName="verifier.updated"
      >
        {v ? (
          <div>
            <p className="text-fg">"{v.next_best_question}"</p>
            <p className="mt-1 text-2xs text-fg-subtle">
              {v.next_question_rationale}
            </p>
          </div>
        ) : null}
      </Slot>
    </aside>
  );
}
