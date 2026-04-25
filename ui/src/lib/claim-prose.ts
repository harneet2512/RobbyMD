/**
 * Transform raw substrate claims into human-readable prose for the trace.
 *
 * devil-wears-helvetica doctrine: every string a physician reads must be
 * authored, not defaulted. Raw predicate names never reach the screen.
 */

const PREDICATE_LABELS: Record<string, string> = {
  onset: "onset",
  character: "",
  severity: "severity",
  location: "",
  radiation: "radiation",
  aggravating_factor: "",
  alleviating_factor: "relieved by",
  associated_symptom: "",
  duration: "duration",
  medical_history: "hx",
  medication: "medication",
  family_history: "FHx",
  social_history: "SHx",
  risk_factor: "",
};

const VALUE_LABELS: Record<string, string> = {
  substernal: "substernal",
  burning: "burning",
  "post-prandial": "post-prandial",
  "negated:exertion": "exertional trigger · denied",
  "negated:immobilization": "immobilization · denied",
  "negated:leg_swelling": "leg swelling · denied",
  "Sun evening": "Sun evening",
  GERD: "GERD",
  "omeprazole · discontinued 2w": "omeprazole · off 2w",
};

export function claimToProse(predicate: string, value: string): string {
  const isNegated = value.startsWith("negated:");
  const bareValue = isNegated ? value.slice(8) : value;

  const mapped = VALUE_LABELS[value];
  if (mapped) return mapped;

  const label = PREDICATE_LABELS[predicate];
  if (label === undefined) {
    return isNegated ? `${bareValue} · denied` : value;
  }

  if (label === "") {
    return isNegated ? `${bareValue} · denied` : bareValue;
  }

  return isNegated
    ? `${label} ${bareValue} · denied`
    : `${label} ${value}`;
}

export interface TraceEntry {
  id: string;
  ts_ns: number;
  kind: "claims" | "supersession" | "decision" | "verifier" | "gap" | "ranking";
  fragments: string[];
  claimIds: string[];
  sourceTurnText?: string;
  sourceTurnSpeaker?: string;
  confidence: number;
  supersededClaimId?: string;
  decisionGlyph?: string;
  decisionInitials?: string;
}

/**
 * Group claims from the same turn into a single trace entry.
 * The spec shows: "14:22:04   chest pain · substernal · burning · post-prandial"
 * not four separate lines.
 */
export function groupClaimsByTurn(
  claims: Array<{
    claim_id: string;
    predicate: string;
    value: string;
    confidence: number;
    source_turn_id: string;
    source_turn_text?: string;
    source_turn_speaker?: string;
    created_ts_ns: number;
  }>
): TraceEntry[] {
  const byTurn = new Map<string, typeof claims>();

  for (const c of claims) {
    const key = c.source_turn_id;
    const group = byTurn.get(key) ?? [];
    group.push(c);
    byTurn.set(key, group);
  }

  const entries: TraceEntry[] = [];

  for (const [turnId, group] of byTurn) {
    const sorted = group.sort((a, b) => a.created_ts_ns - b.created_ts_ns);
    const fragments = sorted.map((c) => claimToProse(c.predicate, c.value));
    const minConf = Math.min(...sorted.map((c) => c.confidence));

    entries.push({
      id: `turn-${turnId}`,
      ts_ns: sorted[0]!.created_ts_ns,
      kind: "claims",
      fragments,
      claimIds: sorted.map((c) => c.claim_id),
      sourceTurnText: sorted[0]!.source_turn_text,
      sourceTurnSpeaker: sorted[0]!.source_turn_speaker,
      confidence: minConf,
    });
  }

  return entries.sort((a, b) => a.ts_ns - b.ts_ns);
}
