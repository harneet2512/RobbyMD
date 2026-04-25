export type DirectionGlyph = "↑" | "↓" | "" | "†" | "⊕" | "×";

export function directionGlyph(
  prevPosterior: number | undefined,
  currentPosterior: number,
  status?: "deprioritized" | "workup_ordered" | "dismissed"
): DirectionGlyph {
  if (status === "deprioritized") return "†";
  if (status === "workup_ordered") return "⊕";
  if (status === "dismissed") return "×";
  if (prevPosterior === undefined) return "";
  const delta = currentPosterior - prevPosterior;
  if (delta > 0.02) return "↑";
  if (delta < -0.02) return "↓";
  return "";
}

export const DECISION_GLYPHS: Record<string, string> = {
  confirm: "✓",
  dismiss: "×",
  deprioritize: "†",
  workup_ordered: "⊕",
};
