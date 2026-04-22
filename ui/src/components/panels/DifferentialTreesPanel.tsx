import { useMemo } from "react";
import ReactFlow, {
  Background,
  BackgroundVariant,
  Controls,
  type Edge,
  type Node,
} from "reactflow";
import "reactflow/dist/style.css";
import { cn } from "@/lib/cn";
import { useSession } from "@/store/session";

/**
 * Panel 3 — Parallel differential trees (PRD.md §6.3).
 *
 * Four trees side-by-side via ReactFlow. In Phase 1 we scaffold a static
 * four-root layout keyed to `content/differentials/chest_pain/branches.json`
 * (we only need the branch roots + a placeholder subtree to prove the
 * ReactFlow plumbing). Full tree data + LR-weighted re-ranking lands in
 * Phase 2 once the differential engine streams `ranking.updated` events.
 *
 * "Trees must breathe" (Eng_doc.md §7) — 200ms transitions, no snapping.
 */

const BRANCHES = [
  { id: "cardiac", label: "Cardiac", hue: "branch-cardiac" },
  { id: "pulmonary", label: "Pulmonary", hue: "branch-pulmonary" },
  { id: "msk", label: "MSK", hue: "branch-msk" },
  { id: "gi", label: "GI", hue: "branch-gi" },
] as const;

export function DifferentialTreesPanel() {
  const ranking = useSession((s) => s.ranking);

  const scores = useMemo(() => {
    const map: Record<string, number> = {};
    if (ranking) {
      for (const s of ranking.scores) map[s.branch] = s.posterior;
    }
    return map;
  }, [ranking]);

  // Static scaffold — one node per branch with a placeholder child. Real
  // per-feature node graph comes in Phase 2.
  const { nodes, edges } = useMemo<{ nodes: Node[]; edges: Edge[] }>(() => {
    const n: Node[] = [];
    const e: Edge[] = [];
    const colW = 220;
    BRANCHES.forEach((b, idx) => {
      const posterior = scores[b.id] ?? 0.25;
      n.push({
        id: `root-${b.id}`,
        data: { label: `${b.label} · ${(posterior * 100).toFixed(0)}%` },
        position: { x: idx * colW, y: 0 },
        style: {
          background: "hsl(var(--surface-2))",
          color: "hsl(var(--fg))",
          border: `1px solid hsl(var(--${b.hue}) / 0.55)`,
          borderRadius: 4,
          fontSize: 12,
          padding: "8px 10px",
          width: 180,
        },
      });
      n.push({
        id: `child-${b.id}`,
        data: { label: "(subtree)" },
        position: { x: idx * colW, y: 90 },
        style: {
          background: "hsl(var(--surface))",
          color: "hsl(var(--fg-muted))",
          border: "1px dashed hsl(var(--border-strong))",
          borderRadius: 4,
          fontSize: 11,
          padding: "6px 8px",
          width: 180,
        },
      });
      e.push({
        id: `e-${b.id}`,
        source: `root-${b.id}`,
        target: `child-${b.id}`,
        style: { stroke: "hsl(var(--border-strong))" },
      });
    });
    return { nodes: n, edges: e };
  }, [scores]);

  return (
    <section
      aria-labelledby="panel-trees-title"
      className="panel h-full min-h-0"
    >
      <div className="panel-header">
        <div className="flex flex-col">
          <span className="panel-eyebrow">Panel 03 · Differential</span>
          <h2 id="panel-trees-title" className="panel-title">
            parallel trees
          </h2>
        </div>
        <div className="flex items-center gap-3 text-2xs text-fg-subtle">
          {BRANCHES.map((b) => (
            <span key={b.id} className="flex items-center gap-1.5">
              <span
                className={cn("h-1.5 w-1.5 rounded-full")}
                style={{ background: `hsl(var(--${b.hue}))` }}
              />
              {b.label}
            </span>
          ))}
        </div>
      </div>
      <div className="relative flex-1 min-h-0">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          fitView
          fitViewOptions={{ padding: 0.25 }}
          proOptions={{ hideAttribution: false }}
          nodesDraggable={false}
          nodesConnectable={false}
          zoomOnScroll={false}
          panOnScroll={false}
        >
          <Background
            variant={BackgroundVariant.Dots}
            gap={18}
            size={1}
            color="hsl(var(--border))"
          />
          <Controls
            showInteractive={false}
            className="!bg-surface-2 !border-border"
          />
        </ReactFlow>
        {ranking == null ? (
          <div className="pointer-events-none absolute inset-0 flex items-end justify-center pb-3 text-2xs text-fg-subtle">
            awaiting ranking stream from <span className="font-mono ml-1">ranking.updated</span>
          </div>
        ) : null}
      </div>
    </section>
  );
}
