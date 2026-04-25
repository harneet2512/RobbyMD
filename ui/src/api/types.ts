export type Speaker = "patient" | "physician" | "system";
export type ClaimStatus = "active" | "superseded" | "confirmed" | "dismissed";

export interface EncounterMeta {
  session_id: string;
  patient_name: string;
  patient_age: number;
  patient_sex: string;
  chief_complaint: string;
  started_at_ns: number;
}

export interface ClaimEvent {
  claim_id: string;
  session_id: string;
  subject: string;
  predicate: string;
  value: string;
  confidence: number;
  source_turn_id: string;
  source_turn_text: string;
  source_turn_speaker: Speaker;
  char_start: number | null;
  char_end: number | null;
  status: ClaimStatus;
  created_ts_ns: number;
}

export interface SupersessionEvent {
  old_claim_id: string;
  new_claim_id: string;
  edge_type: string;
}

export interface HypothesisState {
  branch: string;
  label: string;
  posterior: number;
  log_score: number;
  claim_count: number;
}

export interface RankingEvent {
  hypotheses: HypothesisState[];
}

export interface VerifierEvent {
  why_moved: string[];
  missing_or_contradicting: string[];
  next_best_question: string;
  next_question_rationale: string;
  source_feature: string;
}

export interface DecisionEvent {
  claim_id: string;
  action: "confirm" | "dismiss" | "deprioritize" | "workup_ordered";
  hypothesis: string;
  initials: string;
  ts_ns: number;
}

export interface AsrGapEvent {
  duration_s: number;
  ts_ns: number;
}

export interface Snapshot {
  encounter: EncounterMeta;
  claims: ClaimEvent[];
  ranking: RankingEvent;
  verifier: VerifierEvent | null;
}

export type ServerMessage =
  | { type: "snapshot"; data: Snapshot }
  | { type: "claim.created"; data: ClaimEvent }
  | { type: "claim.superseded"; data: SupersessionEvent }
  | { type: "ranking.updated"; data: RankingEvent }
  | { type: "verifier.updated"; data: VerifierEvent }
  | { type: "decision.recorded"; data: DecisionEvent }
  | { type: "asr.gap"; data: AsrGapEvent };
