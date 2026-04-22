/**
 * TypeScript mirror of `src/substrate/schema.py` dataclasses and enums.
 *
 * This is the wire contract between the future `src/api/` server and the
 * UI. Keep it in lockstep with `src/substrate/schema.py` — when the Python
 * side changes a field, update here and bump the event-bus payload shapes
 * in `@/api/client.ts`.
 *
 * Reference: CLAUDE.md §5.4 ("wt-ui owns ui/"), Eng_doc.md §4.1 (tables),
 * PRD.md §6 (panel contracts), rules.md §4 (provenance invariants).
 */

// --- enums (match StrEnum values on the Python side) ---

export type Speaker = "patient" | "physician" | "system";

export type ClaimStatus = "active" | "superseded" | "confirmed" | "dismissed";

export type EdgeType =
  | "patient_correction"
  | "physician_confirm"
  | "semantic_replace"
  | "refines"
  | "contradicts"
  | "rules_out"
  | "dismissed_by_clinician";

export type NoteSection = "S" | "O" | "A" | "P";

// Closed 14-predicate chest-pain vocabulary (Eng_doc.md §4.2).
export const PREDICATE_FAMILIES = [
  "onset",
  "character",
  "severity",
  "location",
  "radiation",
  "aggravating_factor",
  "alleviating_factor",
  "associated_symptom",
  "duration",
  "medical_history",
  "medication",
  "family_history",
  "social_history",
  "risk_factor",
] as const;

export type PredicateFamily = (typeof PREDICATE_FAMILIES)[number];

// --- data model ---

export interface Turn {
  turn_id: string;
  session_id: string;
  speaker: Speaker;
  text: string;
  ts: number; // nanoseconds since epoch (per schema.py)
  asr_confidence: number | null;
}

export interface Claim {
  claim_id: string;
  session_id: string;
  subject: string;
  predicate: string;
  value: string;
  value_normalised: string | null;
  confidence: number;
  source_turn_id: string;
  status: ClaimStatus;
  created_ts: number;
  char_start: number | null;
  char_end: number | null;
}

export interface SupersessionEdge {
  edge_id: string;
  old_claim_id: string;
  new_claim_id: string;
  edge_type: EdgeType;
  identity_score: number | null;
  created_ts: number;
}

export interface NoteSentence {
  sentence_id: string;
  session_id: string;
  section: NoteSection;
  ordinal: number;
  text: string;
  source_claim_ids: string[];
}

// --- differential engine output (mirrors src/differential/engine.py) ---

export interface AppliedLR {
  claim_id: string;
  branch: string;
  feature: string;
  predicate_path: string;
  lr_value: number;
  log_lr: number;
  direction: "lr_plus" | "lr_minus";
  approximation: boolean;
}

export interface BranchScore {
  branch: string;
  log_score: number;
  posterior: number;
  applied: AppliedLR[];
}

export interface BranchRanking {
  scores: BranchScore[];
}

// --- verifier output (mirrors src/verifier/verifier.py VerifierOutput) ---

export interface VerifierOutput {
  why_moved: string[];
  missing_or_contradicting: string[];
  next_best_question: string;
  next_question_rationale: string;
  source_feature: string;
}

// --- event bus payloads (mirrors src/substrate/event_bus.py) ---

export const TURN_ADDED = "turn.added" as const;
export const CLAIM_CREATED = "claim.created" as const;
export const CLAIM_SUPERSEDED = "claim.superseded" as const;
export const CLAIM_STATUS_CHANGED = "claim.status_changed" as const;
export const PROJECTION_UPDATED = "projection.updated" as const;
export const NOTE_SENTENCE_ADDED = "note_sentence.added" as const;

// Extended events the UI *expects* once src/api/ is built. Not in the
// event_bus.py constants yet — these carry the full entity payload so the
// UI does not have to GET after every notification.
export const TURN_FULL = "turn.full" as const;
export const CLAIM_FULL = "claim.full" as const;
export const RANKING_UPDATED = "ranking.updated" as const;
export const VERIFIER_UPDATED = "verifier.updated" as const;
export const NOTE_SENTENCE_FULL = "note_sentence.full" as const;

export type SubstrateEvent =
  | { event: typeof TURN_ADDED; payload: { turn_id: string; session_id: string } }
  | {
      event: typeof CLAIM_CREATED;
      payload: {
        claim_id: string;
        session_id: string;
        predicate: string;
        status: ClaimStatus;
      };
    }
  | {
      event: typeof CLAIM_SUPERSEDED;
      payload: {
        old_claim_id: string;
        new_claim_id: string;
        edge_type: EdgeType;
        identity_score: number | null;
      };
    }
  | {
      event: typeof CLAIM_STATUS_CHANGED;
      payload: { claim_id: string; status: ClaimStatus };
    }
  | {
      event: typeof PROJECTION_UPDATED;
      payload: { session_id: string; active_count: number };
    }
  | {
      event: typeof NOTE_SENTENCE_ADDED;
      payload: {
        sentence_id: string;
        session_id: string;
        section: NoteSection;
        source_claim_ids: string[];
      };
    }
  // Extensions — full-entity events (documented; see client.ts).
  | { event: typeof TURN_FULL; payload: Turn }
  | { event: typeof CLAIM_FULL; payload: Claim }
  | { event: typeof RANKING_UPDATED; payload: BranchRanking }
  | { event: typeof VERIFIER_UPDATED; payload: VerifierOutput }
  | { event: typeof NOTE_SENTENCE_FULL; payload: NoteSentence };

export type SubstrateEventName = SubstrateEvent["event"];
