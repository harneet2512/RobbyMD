import { useMemo } from "react";
import { cn } from "@/lib/cn";
import { useSession } from "@/store/session";
import type { NoteSection } from "@/types/substrate";

/**
 * Panel 4 — SOAP note (PRD.md §6.4).
 *
 * Phase 1: static scaffold with S/O/A/P section headers. Sentence
 * rendering lands once the note generator (wt-engine scope) emits
 * `note_sentence.added` + `note_sentence.full` events.
 *
 * Every rendered sentence must have non-empty `source_claim_ids`
 * (rules.md §4.2) — enforced server-side by the post-hoc validator; the
 * UI trusts and renders, never synthesises sentences.
 */

const SECTIONS: { id: NoteSection; label: string; help: string }[] = [
  { id: "S", label: "Subjective", help: "patient-reported history" },
  { id: "O", label: "Objective", help: "observed findings (not in demo scope)" },
  { id: "A", label: "Assessment", help: "physician's working differential" },
  { id: "P", label: "Plan", help: "proposed next steps" },
];

export function SoapNotePanel() {
  const sentences = useSession((s) => s.noteSentences);
  const noteOrder = useSession((s) => s.noteOrder);
  const selectedSentenceId = useSession((s) => s.selectedSentenceId);
  const selectedAxis = useSession((s) => s.selectedAxis);
  const selectSentence = useSession((s) => s.selectSentence);

  const bySection = useMemo(() => {
    const m: Record<NoteSection, string[]> = { S: [], O: [], A: [], P: [] };
    for (const sid of noteOrder) {
      const s = sentences[sid];
      if (s) m[s.section].push(sid);
    }
    return m;
  }, [sentences, noteOrder]);

  return (
    <section
      aria-labelledby="panel-soap-title"
      className="panel h-full min-h-0"
    >
      <div className="panel-header">
        <div className="flex flex-col">
          <span className="panel-eyebrow">Panel 04 · SOAP note</span>
          <h2 id="panel-soap-title" className="panel-title">
            with provenance
          </h2>
        </div>
        <span className="font-mono text-2xs text-fg-subtle">
          {noteOrder.length} sentence{noteOrder.length === 1 ? "" : "s"}
        </span>
      </div>

      <div className="scroll-y flex-1 px-5 py-4">
        {noteOrder.length === 0 ? (
          <div className="flex h-full flex-col items-center justify-center gap-2 text-center text-sm text-fg-subtle">
            <span>SOAP note will render here.</span>
            <span className="max-w-[28ch] text-2xs">
              sentences arrive from <span className="font-mono">note_sentence.full</span>{" "}
              after end-of-encounter
            </span>
          </div>
        ) : (
          <div className="flex flex-col gap-5">
            {SECTIONS.map((section) => {
              const ids = bySection[section.id];
              return (
                <div key={section.id}>
                  <div className="mb-2 flex items-baseline justify-between">
                    <h3 className="panel-eyebrow text-fg">
                      {section.id} · {section.label}
                    </h3>
                    <span className="text-2xs text-fg-subtle">
                      {section.help}
                    </span>
                  </div>
                  {ids.length === 0 ? (
                    <p className="text-2xs italic text-fg-subtle">
                      (empty)
                    </p>
                  ) : (
                    <ol className="flex flex-col gap-1.5">
                      {ids.map((sid) => {
                        const s = sentences[sid];
                        if (!s) return null;
                        const isSelected = selectedSentenceId === sid;
                        const dim = selectedAxis != null && !isSelected;
                        return (
                          <li key={sid}>
                            <button
                              type="button"
                              onClick={() =>
                                isSelected
                                  ? selectSentence(null)
                                  : selectSentence(sid)
                              }
                              className={cn(
                                "block w-full rounded border border-transparent px-2 py-1.5 text-left text-[0.88rem] leading-relaxed transition-all",
                                "hover:border-border-strong hover:bg-surface-2",
                                isSelected && "prov-highlight",
                                dim && "prov-dim",
                              )}
                            >
                              {s.text}
                            </button>
                          </li>
                        );
                      })}
                    </ol>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>
    </section>
  );
}
