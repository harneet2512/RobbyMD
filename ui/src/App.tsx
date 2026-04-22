import { useEffect, useRef } from "react";
import { DisclaimerHeader } from "@/components/DisclaimerHeader";
import { TranscriptPanel } from "@/components/panels/TranscriptPanel";
import { ClaimStatePanel } from "@/components/panels/ClaimStatePanel";
import { DifferentialTreesPanel } from "@/components/panels/DifferentialTreesPanel";
import { SoapNotePanel } from "@/components/panels/SoapNotePanel";
import { AuxStrip } from "@/components/panels/AuxStrip";
import { connectMock, type Connection } from "@/api/client";
import { useSession } from "@/store/session";

/**
 * App shell — 1920×1080 primary target; reflows to 1440×900 per Eng_doc.md §7.
 *
 * Layout (grid):
 *   row 1: disclaimer header (persistent, not collapsible — rules.md §3.7)
 *   row 2: [ transcript 25% | { claims 40% top / trees 50% bottom } 40% | soap 35% ]
 *   row 3: aux strip (spans full width under the trees)
 *
 * The exact column fractions from PRD.md §6 are honoured; we use fixed %
 * widths with `minmax(0, …fr)` so ReactFlow inside the trees panel can
 * measure correctly and the panels don't blow out at narrow widths.
 */
export default function App() {
  const conn = useRef<Connection | null>(null);
  const clearSelection = useSession((s) => s.clearSelection);

  useEffect(() => {
    // Connect to the MockServer on mount. Real API client lands once
    // `src/api/` is built — see `api/client.ts` header for the contract.
    conn.current = connectMock();
    return () => {
      conn.current?.disconnect();
      conn.current = null;
    };
  }, []);

  const handleEnd = () => {
    conn.current?.disconnect();
    conn.current = null;
    clearSelection();
  };

  return (
    <div className="flex h-full min-h-screen flex-col">
      <DisclaimerHeader onEndEncounter={handleEnd} />
      <main
        className="grid flex-1 min-h-0 gap-3 p-3"
        style={{
          gridTemplateColumns: "minmax(0, 25fr) minmax(0, 40fr) minmax(0, 35fr)",
          gridTemplateRows: "minmax(0, 2fr) minmax(0, 2fr) auto",
        }}
      >
        {/* Column 1 — transcript (rows 1–2) */}
        <div className="row-span-2 min-h-0">
          <TranscriptPanel />
        </div>

        {/* Column 2 — claims (row 1) + trees (row 2) */}
        <div className="min-h-0">
          <ClaimStatePanel />
        </div>
        <div className="col-start-2 row-start-2 min-h-0">
          <DifferentialTreesPanel />
        </div>

        {/* Column 3 — SOAP note (rows 1–2) */}
        <div className="col-start-3 row-span-2 min-h-0">
          <SoapNotePanel />
        </div>

        {/* Aux strip spans all three columns, row 3 */}
        <div className="col-span-3 row-start-3">
          <AuxStrip />
        </div>
      </main>
    </div>
  );
}
