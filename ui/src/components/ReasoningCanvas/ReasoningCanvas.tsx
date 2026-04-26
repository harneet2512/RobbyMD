import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { motion, AnimatePresence, MotionConfig, useReducedMotion } from "framer-motion";
import { useSession } from "@/store/session";
import type { UserBranchDecision, UserClaimDecision } from "@/store/session";
import type {
  BranchScore,
  Claim,
  EdgeType,
  NoteSentence,
  SupersessionEdge,
  Turn,
  VerifierOutput,
} from "@/types/substrate";
import styles from "./ReasoningCanvas.module.css";

/* ── types ── */

type FocusTarget =
  | { type: "claim"; id: string }
  | { type: "hypothesis"; id: string }
  | { type: "sentence"; id: string }
  | { type: "verifier"; id: "current" };

type HoverTarget = FocusTarget | null;
type ClaimTreatment =
  | "normal"
  | "confirmed"
  | "systemSuperseded"
  | "systemDismissed"
  | "userSuppressed";
type BranchDecision = UserBranchDecision;
type ClaimDecision = UserClaimDecision;

interface CanvasData {
  turns: Turn[];
  claimsByTurn: Map<string, Claim[]>;
  claimsById: Map<string, Claim>;
  claimsByPredicate: Map<string, Claim[]>;
  hypotheses: BranchScore[];
  edges: SupersessionEdge[];
  sentences: NoteSentence[];
  strikeOrigin: Map<string, EdgeType>;
}

/* ── constants ── */

const statusRank: Record<Claim["status"], number> = {
  active: 0,
  confirmed: 1,
  superseded: 2,
  dismissed: 3,
};
const EXPO_OUT = [0.16, 1, 0.3, 1] as const;

interface LabItem {
  name: string;
  value: string;
  flag: "normal" | "high" | "low";
}

interface PatientDocument {
  type: "lab" | "imaging" | "procedure";
  date: string;
  title: string;
  items?: LabItem[];
  result?: string;
}

const DEMO_PATIENT = {
  name: "Otto Yuen",
  dob: "06/18/1969",
  age: 56,
  sex: "M",
  mrn: "MRN-2847103",
  allergies: ["Penicillin"],
  problems: ["CAD s/p stent", "HTN", "HLD", "Type 2 DM"],
  medications: [
    "Lisinopril 20mg daily",
    "Atorvastatin 40mg daily",
    "Metformin 1000mg BID",
    "Aspirin 81mg daily",
  ],
  priorVisits: [
    {
      date: "2026-03-15",
      chief: "Follow-up CAD",
      summary: "Stable angina, no new symptoms. Continued current medications.",
    },
    {
      date: "2025-11-02",
      chief: "Chest pain evaluation",
      summary: "Atypical chest pain, stress test negative. Reassured.",
    },
  ],
  documents: [
    {
      type: "lab" as const,
      date: "2026-04-25",
      title: "Basic Metabolic Panel",
      items: [
        { name: "Troponin I", value: "0.04 ng/mL", flag: "normal" as const },
        { name: "BNP", value: "89 pg/mL", flag: "normal" as const },
        { name: "Creatinine", value: "1.1 mg/dL", flag: "normal" as const },
        { name: "Glucose", value: "187 mg/dL", flag: "high" as const },
        { name: "HbA1c", value: "8.2%", flag: "high" as const },
      ],
    },
    {
      type: "lab" as const,
      date: "2026-03-15",
      title: "Lipid Panel",
      items: [
        { name: "Total Cholesterol", value: "242 mg/dL", flag: "high" as const },
        { name: "LDL", value: "168 mg/dL", flag: "high" as const },
        { name: "HDL", value: "38 mg/dL", flag: "low" as const },
        { name: "Triglycerides", value: "180 mg/dL", flag: "high" as const },
      ],
    },
    {
      type: "imaging" as const,
      date: "2025-11-02",
      title: "Stress Test",
      result: "Negative for inducible ischemia at 85% predicted HR.",
    },
    {
      type: "procedure" as const,
      date: "2020-06-14",
      title: "PCI with DES to LAD",
      result: "Successful revascularization. Single drug-eluting stent to mid-LAD. No residual stenosis.",
    },
  ] satisfies PatientDocument[],
};

const SECTION_NAMES: Record<string, string> = {
  S: "Subjective",
  O: "Objective",
  A: "Assessment",
  P: "Plan",
};

/* ── helpers ── */

const BRANCH_DISPLAY_NAMES: Record<string, string> = {
  cardiac: "Acute Coronary Syndrome",
  pulmonary_embolism: "Pulmonary Embolism",
  pulmonary: "Pulmonary Embolism",
  msk: "Musculoskeletal",
  gi: "Gastrointestinal",
};

function displayBranch(branch: string): string {
  return BRANCH_DISPLAY_NAMES[branch] ?? branch.replaceAll("_", " ");
}

function isClaim(c: Claim | undefined): c is Claim {
  return c !== undefined;
}
function isTurn(t: Turn | undefined): t is Turn {
  return t !== undefined;
}
function isSentence(s: NoteSentence | undefined): s is NoteSentence {
  return s !== undefined;
}

function fmtTime(ns: number) {
  const s = Math.max(0, Math.floor(ns / 1e9));
  return `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
}

function fmtValue(v: string) {
  return v.replace(/^negated:/, "denies ").replaceAll("_", " ");
}

function normStatus(
  s: Claim["status"],
): "active" | "superseded" | "dismissed" {
  return s === "superseded"
    ? "superseded"
    : s === "dismissed"
      ? "dismissed"
      : "active";
}

function strikeColorClass(origin: EdgeType | undefined): string {
  if (!origin) return "";
  switch (origin) {
    case "patient_correction":
      return styles.strikePatient!;
    case "dismissed_by_clinician":
    case "physician_confirm":
      return styles.strikePhysician!;
    default:
      return styles.strikeSystem!;
  }
}

function strikeOriginLabel(origin: EdgeType | undefined): string {
  switch (origin) {
    case "patient_correction":
      return "corrected";
    case "dismissed_by_clinician":
      return "clinician dismissed";
    case "contradicts":
      return "contradicted";
    case "semantic_replace":
      return "superseded";
    case "rules_out":
      return "ruled down";
    case "physician_confirm":
      return "clinician confirmed";
    case "refines":
      return "refined";
    default:
      return "superseded";
  }
}

function claimTreatment(claim: Claim, userSuppressedClaimIds: Set<string>): ClaimTreatment {
  if (userSuppressedClaimIds.has(claim.claim_id)) return "userSuppressed";
  if (claim.status === "superseded") return "systemSuperseded";
  if (claim.status === "dismissed") return "systemDismissed";
  if (claim.status === "confirmed") return "confirmed";
  return "normal";
}

function buildChains(edges: SupersessionEdge[]) {
  const next = new Map<string, string>();
  const prev = new Map<string, string>();
  for (const e of edges) {
    next.set(e.old_claim_id, e.new_claim_id);
    prev.set(e.new_claim_id, e.old_claim_id);
  }
  const chains = new Map<string, string[]>();
  for (const id of new Set([...next.keys(), ...prev.keys()])) {
    let h = id;
    while (prev.has(h)) h = prev.get(h)!;
    const c = [h];
    let cur = h;
    while (next.has(cur)) {
      cur = next.get(cur)!;
      c.push(cur);
    }
    for (const cid of c) chains.set(cid, c);
  }
  return chains;
}

function relClaimIds(h: BranchScore, claims: Claim[]): string[] {
  const a = h.applied.map((x) => x.claim_id);
  return a.length > 0
    ? a
    : claims
        .filter((c) => normStatus(c.status) === "active")
        .slice(0, 6)
        .map((c) => c.claim_id);
}

function restAssociationIds(
  hypotheses: BranchScore[],
  claims: Claim[],
): Set<string> {
  const ids = new Set<string>();
  if (hypotheses.length === 0) return ids;
  const focus = hypotheses[0];
  if (!focus) return ids;
  const influential = focus.applied.slice(0, 2).map((a) => a.claim_id);
  for (const id of influential) ids.add(id);
  const recent = [...claims]
    .filter((c) => normStatus(c.status) === "active")
    .sort((a, b) => b.created_ts - a.created_ts)
    .slice(0, 2)
    .map((c) => c.claim_id);
  for (const id of recent) ids.add(id);
  return ids;
}

/** Map claims to patient context items and return matched claim IDs (Fix 2c) */
const CONTEXT_CLAIM_MAP: Array<{ keywords: string[]; labels: string[] }> = [
  { keywords: ["diabetes", "dm", "glucose", "hba1c", "metformin"], labels: ["Type 2 DM", "Metformin 1000mg BID", "Glucose", "HbA1c"] },
  { keywords: ["hypertension", "htn", "blood_pressure", "lisinopril"], labels: ["HTN", "Lisinopril 20mg daily"] },
  { keywords: ["hyperlipidemia", "hld", "cholesterol", "ldl", "hdl", "statin", "atorvastatin", "lipid"], labels: ["HLD", "Atorvastatin 40mg daily", "Total Cholesterol", "LDL", "HDL", "Triglycerides"] },
  { keywords: ["cad", "coronary", "stent", "pci", "cardiac", "mi", "angina", "aspirin"], labels: ["CAD s/p stent", "Aspirin 81mg daily", "PCI with DES to LAD", "Stress Test"] },
  { keywords: ["chest_pain", "substernal", "pressure", "exertion"], labels: ["CAD s/p stent", "Stress Test"] },
  { keywords: ["allergy", "penicillin"], labels: ["Penicillin"] },
];

function buildPatientContextMatches(claims: Map<string, Claim>): { matchedClaimIds: Set<string>; matchedLabels: Set<string> } {
  const matchedClaimIds = new Set<string>();
  const matchedLabels = new Set<string>();
  for (const claim of claims.values()) {
    if (normStatus(claim.status) !== "active") continue;
    const haystack = `${claim.predicate} ${claim.value}`.toLowerCase();
    for (const entry of CONTEXT_CLAIM_MAP) {
      if (entry.keywords.some((kw) => haystack.includes(kw))) {
        matchedClaimIds.add(claim.claim_id);
        for (const label of entry.labels) matchedLabels.add(label);
      }
    }
  }
  return { matchedClaimIds, matchedLabels };
}

/* ── data hook ── */

function useCanvasData(): CanvasData {
  const tr = useSession((s) => s.turns);
  const to = useSession((s) => s.turnOrder);
  const cr = useSession((s) => s.claims);
  const co = useSession((s) => s.claimOrder);
  const rk = useSession((s) => s.ranking);
  const ed = useSession((s) => s.edges);
  const sr = useSession((s) => s.noteSentences);
  const no = useSession((s) => s.noteOrder);

  return useMemo(() => {
    const turns = to.map((id) => tr[id]).filter(isTurn);
    const claims = co
      .map((id) => cr[id])
      .filter(isClaim)
      .sort(
        (a, b) =>
          statusRank[a.status] - statusRank[b.status] ||
          a.created_ts - b.created_ts,
      );

    const claimsByTurn = new Map<string, Claim[]>();
    const claimsById = new Map<string, Claim>();
    const claimsByPredicate = new Map<string, Claim[]>();

    for (const c of claims) {
      claimsById.set(c.claim_id, c);

      const tg = claimsByTurn.get(c.source_turn_id) ?? [];
      tg.push(c);
      claimsByTurn.set(c.source_turn_id, tg);

      const pg = claimsByPredicate.get(c.predicate) ?? [];
      pg.push(c);
      claimsByPredicate.set(c.predicate, pg);
    }

    const strikeOrigin = new Map<string, EdgeType>();
    for (const e of ed) {
      strikeOrigin.set(e.old_claim_id, e.edge_type);
    }

    return {
      turns,
      claimsByTurn,
      claimsById,
      claimsByPredicate,
      hypotheses: [...(rk?.scores ?? [])].sort(
        (a, b) => b.log_score - a.log_score,
      ),
      edges: ed,
      sentences: no.map((id) => sr[id]).filter(isSentence),
      strikeOrigin,
    };
  }, [co, cr, ed, no, rk?.scores, sr, to, tr]);
}

function stateLabel(d: CanvasData) {
  if (d.claimsById.size === 0) return "";
  if (d.sentences.length > 0) return "SOAP available";
  if (d.hypotheses.length > 0) return "Reasoning updated";
  return "Evidence captured";
}

/* ── waveform ── */

const WAVE_PTS = 200;
const WAVE_H = 48;

function seededNoise(i: number): number {
  const x = Math.sin(i * 127.1 + 311.7) * 43758.5453;
  return x - Math.floor(x);
}

function buildWaveform(turns: Turn[], expectedDurationNs: number): number[] {
  if (expectedDurationNs === 0 || turns.length === 0)
    return new Array(WAVE_PTS).fill(0.03) as number[];
  const coarse = 60;
  const raw = new Array(coarse).fill(0) as number[];
  const bucketNs = expectedDurationNs / coarse;
  for (const t of turns) {
    const b = Math.min(coarse - 1, Math.floor(t.ts / bucketNs));
    raw[b]! += t.text.length;
  }
  const max = Math.max(...raw, 1);
  const norm = raw.map((v) => v / max);
  const up = new Array(WAVE_PTS).fill(0) as number[];
  for (let i = 0; i < WAVE_PTS; i++) {
    const srcIdx = (i / WAVE_PTS) * coarse;
    const lo = Math.floor(srcIdx);
    const hi = Math.min(coarse - 1, lo + 1);
    const frac = srcIdx - lo;
    up[i] = norm[lo]! * (1 - frac) + norm[hi]! * frac;
  }
  const out = new Array(WAVE_PTS).fill(0) as number[];
  for (let pass = 0; pass < 3; pass++) {
    const src = pass === 0 ? up : out;
    for (let i = 0; i < WAVE_PTS; i++) {
      const w = 8;
      let sum = 0,
        wt = 0;
      for (
        let j = Math.max(0, i - w);
        j <= Math.min(WAVE_PTS - 1, i + w);
        j++
      ) {
        const g = Math.exp(-0.5 * ((j - i) / (w * 0.45)) ** 2);
        sum += src[j]! * g;
        wt += g;
      }
      out[i] = sum / wt;
    }
  }
  for (let i = 0; i < WAVE_PTS; i++) {
    const noise = (seededNoise(i) - 0.5) * 0.12;
    const micro = (seededNoise(i * 3 + 77) - 0.5) * 0.06;
    out[i] = Math.max(0.03, Math.min(1, out[i]! + noise + micro + 0.04));
  }
  return out;
}

/* waveformPath removed — using vertical bar rendering */

/* ── focus overlay (Layer 3 — Fix 4) ── */

const FOCUS_LABEL_CLS: Record<string, string> = {
  utterance: styles.focusLabelUtterance!,
  evidence: styles.focusLabelEvidence!,
  reasoning: styles.focusLabelReasoning!,
  SOAP: styles.focusLabelSoap!,
  question: styles.focusLabelQuestion!,
  rail: styles.focusLabelRail!,
};

function buildFocusSteps(
  target: FocusTarget,
  data: CanvasData,
  verifier: VerifierOutput | null,
): Array<{ label: string; text: string }> {
  const steps: Array<{ label: string; text: string }> = [];

  if (target.type === "claim") {
    const claim = data.claimsById.get(target.id);
    if (!claim) return steps;
    const turn = data.turns.find((t) => t.turn_id === claim.source_turn_id);
    if (turn) steps.push({ label: "utterance", text: turn.text });
    steps.push({
      label: "evidence",
      text: `${claim.predicate.replaceAll("_", " ")}: ${fmtValue(claim.value)}`,
    });
    const hyp = data.hypotheses.find((h) =>
      h.applied.some((a) => a.claim_id === claim.claim_id),
    );
    if (hyp)
      steps.push({ label: "reasoning", text: displayBranch(hyp.branch) });
    if (verifier?.next_best_question)
      steps.push({ label: "question", text: verifier.next_best_question });
    const soap = data.sentences.find((s) =>
      s.source_claim_ids.includes(claim.claim_id),
    );
    if (soap) steps.push({ label: "SOAP", text: soap.text });
    if (turn) steps.push({ label: "rail", text: fmtTime(turn.ts) });
  } else if (target.type === "hypothesis") {
    const h = data.hypotheses.find((x) => x.branch === target.id);
    if (!h) return steps;
    steps.push({ label: "reasoning", text: displayBranch(h.branch) });
    for (const applied of h.applied) {
      const c = data.claimsById.get(applied.claim_id);
      if (c)
        steps.push({
          label: "evidence",
          text: `${c.predicate.replaceAll("_", " ")}: ${fmtValue(c.value)}`,
        });
    }
    const firstClaim = h.applied[0] ? data.claimsById.get(h.applied[0].claim_id) : undefined;
    if (firstClaim) {
      const turn = data.turns.find((t) => t.turn_id === firstClaim.source_turn_id);
      if (turn) steps.push({ label: "rail", text: fmtTime(turn.ts) });
    }
  } else if (target.type === "sentence") {
    const s = data.sentences.find((x) => x.sentence_id === target.id);
    if (!s) return steps;
    steps.push({ label: "SOAP", text: s.text });
    for (const cid of s.source_claim_ids) {
      const c = data.claimsById.get(cid);
      if (c) {
        steps.push({
          label: "evidence",
          text: `${c.predicate.replaceAll("_", " ")}: ${fmtValue(c.value)}`,
        });
        const turn = data.turns.find((t) => t.turn_id === c.source_turn_id);
        if (turn) steps.push({ label: "utterance", text: turn.text });
      }
    }
  }

  return steps;
}

function FocusOverlay({
  target,
  data,
  onClose,
  onPromote,
  onDemote,
  claimTreatments,
  restoreFocusTo,
}: {
  target: FocusTarget;
  data: CanvasData;
  onClose: () => void;
  onPromote: (target: FocusTarget, claimIds: string[]) => void;
  onDemote: (target: FocusTarget, claimIds: string[]) => void;
  claimTreatments: Map<string, ClaimTreatment>;
  restoreFocusTo: HTMLElement | null;
}) {
  const verifier = useSession((s) => s.verifier);
  const cardRef = useRef<HTMLDivElement>(null);
  const closeRef = useRef<HTMLButtonElement>(null);
  const reduceMotion = useReducedMotion();
  const steps = buildFocusSteps(target, data, verifier);
  const [threadExpanded, setThreadExpanded] = useState(false);

  const claimId = target.type === "claim" ? target.id : null;
  const claim = claimId ? data.claimsById.get(claimId) : null;
  const treatment = claimId ? claimTreatments.get(claimId) ?? "normal" : "normal";
  const targetClaimIds =
    target.type === "claim"
      ? [target.id]
      : target.type === "hypothesis"
        ? data.hypotheses.find((h) => h.branch === target.id)?.applied.map((a) => a.claim_id) ?? []
        : target.type === "sentence"
          ? data.sentences.find((s) => s.sentence_id === target.id)?.source_claim_ids ?? []
          : [];
  const canSteer = target.type === "hypothesis" || targetClaimIds.length > 0;
  const titleId = `focus-panel-title-${target.type}-${target.id}`;
  const summaryId = `focus-panel-summary-${target.type}-${target.id}`;

  const contextLabel = target.type === "claim" && claim
    ? `${claim.predicate.replaceAll("_", " ")}: ${fmtValue(claim.value)}`
    : target.type === "hypothesis"
      ? displayBranch(target.id)
      : target.type === "sentence"
        ? data.sentences.find(s => s.sentence_id === target.id)?.text?.slice(0, 80) ?? ""
        : "";

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  useEffect(() => {
    closeRef.current?.focus({ preventScroll: true });
    return () => { restoreFocusTo?.focus({ preventScroll: true }); };
  }, [restoreFocusTo]);

  if (steps.length === 0) return null;

  const cardInitial = reduceMotion ? { opacity: 0 } : { opacity: 0, y: 6, scale: 0.97 };
  const cardAnimate = reduceMotion ? { opacity: 1 } : { opacity: 1, y: 0, scale: 1 };
  const cardExit = reduceMotion ? { opacity: 0 } : { opacity: 0, y: 4, scale: 0.98 };

  const handleTab = (e: React.KeyboardEvent<HTMLDivElement>) => {
    if (e.key !== "Tab" || !cardRef.current) return;
    const focusable = Array.from(
      cardRef.current.querySelectorAll<HTMLElement>('button:not([disabled]), [href], [tabindex]:not([tabindex="-1"])'),
    );
    if (focusable.length === 0) return;
    const first = focusable[0]!;
    const last = focusable[focusable.length - 1]!;
    if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
    else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
  };

  if (typeof document === "undefined") return null;

  return createPortal(
    <>
      <motion.div
        className={styles.focusBackdrop}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        transition={{ duration: 0.12 }}
        onClick={onClose}
      />
      <motion.div
        ref={cardRef}
        className={styles.focusPanel}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        aria-describedby={summaryId}
        initial={cardInitial}
        animate={cardAnimate}
        exit={cardExit}
        transition={{ duration: reduceMotion ? 0.01 : 0.18, ease: EXPO_OUT }}
        onKeyDown={handleTab}
      >
        <div className={styles.focusPanelEyebrow}>Decision Point</div>
        <div id={titleId} className={styles.focusPanelContext}>{contextLabel}</div>
        {claim && treatment !== "normal" && treatment !== "confirmed" && (
          <div className={styles.focusPanelState}>
            {treatment === "userSuppressed"
              ? "Suppressed by user"
              : treatment === "systemDismissed"
                ? "Dismissed by system"
                : strikeOriginLabel(data.strikeOrigin.get(claim.claim_id))}
          </div>
        )}

        {/* Compact provenance summary — first 2 steps */}
        <div id={summaryId} className={styles.focusPanelSummary}>
          {steps.slice(0, 2).map((step, i) => (
            <span key={i} className={styles.focusPanelStep}>
              <span className={styles.focusPanelStepLabel}>{step.label}</span> {step.text}
            </span>
          ))}
        </div>

        {/* Expanded thread — full provenance */}
        {threadExpanded && (
          <div className={styles.focusPanelThread}>
            {steps.map((step, i) => (
              <div key={i} className={styles.focusOverlayStep}>
                <span className={`${styles.focusOverlayLabel} ${FOCUS_LABEL_CLS[step.label] ?? styles.focusLabelRail}`}>
                  {step.label}
                </span>
                <span className={styles.focusOverlayText}>{step.text}</span>
              </div>
            ))}
          </div>
        )}

        {/* 4 actions: Upvote / Downvote / Thread / Close */}
        <div className={styles.focusPanelActions}>
          {canSteer && (
            <>
              <button
                className={styles.focusActionUpvote}
                type="button"
                onClick={() => { onPromote(target, targetClaimIds); onClose(); }}
              >
                Upvote
              </button>
              <button
                className={styles.focusActionDownvote}
                type="button"
                onClick={() => { onDemote(target, targetClaimIds); onClose(); }}
              >
                Downvote
              </button>
            </>
          )}
          <button
            className={styles.focusActionThread}
            type="button"
            onClick={() => setThreadExpanded(!threadExpanded)}
          >
            {threadExpanded ? "Collapse" : "Thread"}
          </button>
          <button
            ref={closeRef}
            className={styles.focusActionClose}
            type="button"
            aria-label="Close"
            onClick={onClose}
          >
            Close
          </button>
        </div>
      </motion.div>
    </>,
    document.body,
  );
}

/* ── patient strip ── */

function PatientStrip({ state }: { state: string }) {
  return (
    <header className={styles.patientStrip}>
      <span className={styles.brandMark}>
        <img className={styles.brandLogo} src="/robby-md-wordmark.svg" alt="RobbyMD" />
      </span>
      <span className={styles.patientName}>Patient Name: {DEMO_PATIENT.name}</span>
      <div className={styles.patientMeta}>
        <span>{DEMO_PATIENT.dob}</span>
        <span>
          {DEMO_PATIENT.age}{DEMO_PATIENT.sex}
        </span>
        <span>{DEMO_PATIENT.mrn}</span>
      </div>
      <div className={styles.patientProblems}>
        {DEMO_PATIENT.problems.slice(0, 4).map((p) => (
          <span key={p} className={styles.problemPill}>
            {p}
          </span>
        ))}
      </div>
      <div className={styles.stripRight}>
        {state && <span className={styles.stripState}>{state}</span>}
        <span className={styles.stripDisclaimer}>
          Research prototype, not a medical device
        </span>
      </div>
    </header>
  );
}

/* ── transcript panel (left) ── */

function TranscriptPanel(props: {
  turns: Turn[];
  claimsByTurn: Map<string, Claim[]>;
  strikeOrigin: Map<string, EdgeType>;
  hoverTarget: HoverTarget;
  focusTarget: FocusTarget | null;
  activeTurnId: string | null;
  followTranscript: boolean;
  relatedClaims: Set<string>;
  restAssoc: Set<string>;
  onHover: (t: HoverTarget) => void;
  onFocus: (t: FocusTarget) => void;
  onManualScroll: () => void;
  onReturnToActive: () => void;
}) {
  const {
    turns,
    claimsByTurn,
    strikeOrigin,
    hoverTarget,
    focusTarget,
    activeTurnId,
    followTranscript,
    relatedClaims,
    restAssoc,
    onHover,
    onFocus,
    onManualScroll,
    onReturnToActive,
  } = props;
  const endRef = useRef<HTMLDivElement>(null);
  const transcriptRef = useRef<HTMLElement>(null);
  const isNearBottom = useRef(true);
  const followRaf = useRef<number | null>(null);
  const programmaticScrollRef = useRef(false);
  const reduceMotion = useReducedMotion();

  useEffect(() => {
    const el = transcriptRef.current;
    if (!el) return;
    const onScroll = () => {
      isNearBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
      if (!followTranscript && activeTurnId) {
        const row = el.querySelector<HTMLElement>(`[data-turn-id="${activeTurnId}"]`);
        if (!row) return;
        const rowTop = row.offsetTop - el.scrollTop;
        const rowBottom = rowTop + row.offsetHeight;
        if (rowTop >= 0 && rowBottom <= el.clientHeight) onReturnToActive();
      }
    };
    const onUserScrollIntent = () => {
      if (followRaf.current !== null) {
        cancelAnimationFrame(followRaf.current);
        followRaf.current = null;
      }
      if (!programmaticScrollRef.current) onManualScroll();
      isNearBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
    };
    el.addEventListener("scroll", onScroll, { passive: true });
    el.addEventListener("wheel", onUserScrollIntent, { passive: true });
    el.addEventListener("touchstart", onUserScrollIntent, { passive: true });
    el.addEventListener("pointerdown", onUserScrollIntent, { passive: true });
    return () => {
      el.removeEventListener("scroll", onScroll);
      el.removeEventListener("wheel", onUserScrollIntent);
      el.removeEventListener("touchstart", onUserScrollIntent);
      el.removeEventListener("pointerdown", onUserScrollIntent);
    };
  }, [activeTurnId, followTranscript, onManualScroll, onReturnToActive]);

  useEffect(() => {
    if (turns.length > 0 && isNearBottom.current && !activeTurnId) {
      const el = transcriptRef.current;
      if (!el) return;
      const targetTop = Math.max(0, el.scrollHeight - el.clientHeight);
      if (followRaf.current !== null) cancelAnimationFrame(followRaf.current);

      if (reduceMotion) {
        el.scrollTop = targetTop;
        return;
      }

      const tick = () => {
        const current = el.scrollTop;
        const next = current + (targetTop - current) * 0.18;
        if (Math.abs(targetTop - next) < 1) {
          el.scrollTop = targetTop;
          followRaf.current = null;
          return;
        }
        el.scrollTop = next;
        followRaf.current = requestAnimationFrame(tick);
      };

      followRaf.current = requestAnimationFrame(tick);
    }
    }, [turns.length, reduceMotion, activeTurnId]);

  useEffect(() => {
    if (!activeTurnId || !followTranscript) return;
    const el = transcriptRef.current;
    const row = el?.querySelector<HTMLElement>(`[data-turn-id="${activeTurnId}"]`);
    if (!el || !row) return;

    if (followRaf.current !== null) {
      cancelAnimationFrame(followRaf.current);
      followRaf.current = null;
    }

    const rowTop = row.offsetTop;
    const rowCenter = rowTop + row.offsetHeight / 2;
    const targetTop = Math.max(0, rowCenter - el.clientHeight / 2);

    programmaticScrollRef.current = true;
    if (reduceMotion) {
      el.scrollTop = targetTop;
      window.setTimeout(() => {
        programmaticScrollRef.current = false;
      }, 0);
      return;
    }

    const tick = () => {
      const current = el.scrollTop;
      const next = current + (targetTop - current) * 0.22;
      if (Math.abs(targetTop - next) < 1) {
        el.scrollTop = targetTop;
        followRaf.current = null;
        window.setTimeout(() => {
          programmaticScrollRef.current = false;
        }, 0);
        return;
      }
      el.scrollTop = next;
      followRaf.current = requestAnimationFrame(tick);
    };

    followRaf.current = requestAnimationFrame(tick);
  }, [activeTurnId, followTranscript, reduceMotion]);

  useEffect(() => {
    return () => {
      if (followRaf.current !== null) cancelAnimationFrame(followRaf.current);
    };
  }, []);

  return (
    <section ref={transcriptRef} className={styles.transcript} data-surface="transcript">
      <div className={styles.transcriptHeader}>Transcript</div>
      <AnimatePresence>
        {turns.map((turn) => {
          const isPhysician = turn.speaker === "physician";
          const turnClaims = claimsByTurn.get(turn.turn_id) ?? [];
          const isActive = activeTurnId === turn.turn_id;
          return (
            <motion.div
              className={`${styles.turn} ${isActive ? styles.turnActive : ""}`}
              key={turn.turn_id}
              data-turn-id={turn.turn_id}
              aria-current={isActive ? "true" : undefined}
              onFocus={() => {
                if (isActive) onReturnToActive();
              }}
              initial={reduceMotion ? { opacity: 1 } : { opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: reduceMotion ? 0.01 : 0.24, ease: EXPO_OUT }}
            >
              <div className={styles.turnHeader}>
                <span className={styles.turnTime}>{fmtTime(turn.ts)}</span>
                <span
                  className={`${styles.speakerLabel} ${isPhysician ? styles.speakerPhysician : styles.speakerPatient}`}
                >
                  {isPhysician ? "Physician" : "Patient"}
                </span>
              </div>
              <p
                className={
                  isPhysician ? styles.turnTextPhysician : styles.turnText
                }
              >
                {turn.text}
              </p>
              {turnClaims.length > 0 && (
                <div className={styles.claimPills}>
                  {turnClaims.map((claim) => {
                    const st = normStatus(claim.status);
                    const isFoc =
                      focusTarget?.type === "claim" &&
                      focusTarget.id === claim.claim_id;
                    const isHov =
                      hoverTarget?.type === "claim" &&
                      hoverTarget.id === claim.claim_id;
                    const isRel = relatedClaims.has(claim.claim_id);
                    const isRest = restAssoc.has(claim.claim_id);
                    const cls = [
                      st === "superseded"
                        ? styles.claimPillSuperseded
                        : st === "dismissed"
                          ? styles.claimPillDismissed
                          : styles.claimPill,
                      st === "superseded" || st === "dismissed"
                        ? strikeColorClass(strikeOrigin.get(claim.claim_id))
                        : "",
                      isFoc ? styles.focused : "",
                      isHov ? styles.hovered : "",
                      isRel ? styles.related : "",
                      isRest && !isFoc && !isHov ? styles.restThread : "",
                    ]
                      .filter(Boolean)
                      .join(" ");

                    return (
                      <button
                        className={`${cls} ${styles.claimPillEnter}`}
                        key={claim.claim_id}
                        data-claim-id={claim.claim_id}
                        type="button"
                        onMouseEnter={() =>
                          onHover({ type: "claim", id: claim.claim_id })
                        }
                        onMouseLeave={() => onHover(null)}
                        onClick={() =>
                          onFocus({ type: "claim", id: claim.claim_id })
                        }
                      >
                        <span className={styles.claimPredicate}>
                          {claim.predicate.replaceAll("_", " ")}
                        </span>
                        {fmtValue(claim.value)}
                      </button>
                    );
                  })}
                </div>
              )}
            </motion.div>
          );
        })}
      </AnimatePresence>
      <div ref={endRef} />
    </section>
  );
}

/* ── claims area (center top) ── */

function ClaimsArea(props: {
  claimsByPredicate: Map<string, Claim[]>;
  claimsById: Map<string, Claim>;
  strikeOrigin: Map<string, EdgeType>;
  claimTreatments: Map<string, ClaimTreatment>;
  claimDecisionsById: Record<string, ClaimDecision>;
  hoverTarget: HoverTarget;
  focusTarget: FocusTarget | null;
  relatedClaims: Set<string>;
  restAssoc: Set<string>;
  onHover: (t: HoverTarget) => void;
  onFocus: (t: FocusTarget) => void;
}) {
  const {
    claimsByPredicate,
    strikeOrigin,
    claimTreatments,
    claimDecisionsById,
    hoverTarget,
    focusTarget,
    relatedClaims,
    restAssoc,
    onHover,
    onFocus,
  } = props;
  const verifier = useSession((s) => s.verifier);

  const CLINICAL_SALIENCE = [
    "onset", "character", "severity", "location", "radiation",
    "aggravating_factor", "alleviating_factor", "associated_symptom",
    "duration", "medical_history", "family_history", "social_history",
    "risk_factor", "medication",
  ];

  const weakenedPredicates: string[] = [];
  for (const [pred, claims] of claimsByPredicate) {
    const hasReadableEvidence = claims.some((c) => {
      const t = claimTreatments.get(c.claim_id) ?? "normal";
      return t === "normal" || t === "confirmed" || t === "systemSuperseded";
    });
    if (!hasReadableEvidence) weakenedPredicates.push(pred);
  }

  const flaggedLabs: LabItem[] = [];
  for (const doc of DEMO_PATIENT.documents) {
    if (doc.items) for (const item of doc.items) {
      if (item.flag !== "normal") flaggedLabs.push(item);
    }
  }

  if (claimsByPredicate.size === 0) {
    return (
      <div className={styles.claimsArea} data-surface="claims">
        <div className={styles.claimsAreaHeader}>Evidence</div>
        <div className={styles.emptyState}>
          Waiting for clinical evidence...
        </div>
        <div className={styles.patientSignals}>
          <div className={styles.signalsLabel}>Patient baseline</div>
          <div className={styles.signalsRow}>
            {DEMO_PATIENT.problems.map((p) => (
              <span key={p} className={styles.signalPill}>{p}</span>
            ))}
            {flaggedLabs.map((l) => (
              <span key={l.name} className={`${styles.signalPill} ${styles.signalFlagged}`}>
                {l.name} {l.value}
              </span>
            ))}
          </div>
        </div>
      </div>
    );
  }

  const groups = [...claimsByPredicate.entries()].sort(([a], [b]) => {
    const ai = CLINICAL_SALIENCE.indexOf(a);
    const bi = CLINICAL_SALIENCE.indexOf(b);
    return (ai === -1 ? 99 : ai) - (bi === -1 ? 99 : bi);
  });

  const missingItems = verifier?.missing_or_contradicting ?? [];

  return (
    <div className={styles.claimsArea} data-surface="claims">
      <div className={styles.claimsAreaHeader}>Evidence</div>
      <div className={styles.evidenceFlow}>
        {groups.map(([predicate, claims]) => (
          <div key={predicate} className={styles.evidenceUnit}>
            <div className={styles.evidenceUnitLabel}>
              {predicate.replaceAll("_", " ")}
            </div>
            <div className={styles.evidenceUnitTokens}>
              {claims.map((claim) => {
                const isRel = relatedClaims.has(claim.claim_id);
                const isRest = restAssoc.has(claim.claim_id);
                const isFoc =
                  focusTarget?.type === "claim" &&
                  focusTarget.id === claim.claim_id;
                const isHov =
                  hoverTarget?.type === "claim" &&
                  hoverTarget.id === claim.claim_id;
                const treatment = claimTreatments.get(claim.claim_id) ?? "normal";
                const decision = claimDecisionsById[claim.claim_id] ?? null;
                const isSystemSup = treatment === "systemSuperseded";
                const isSystemDismissed = treatment === "systemDismissed";
                const isUserSuppressed = treatment === "userSuppressed";
                const stateLabel = isUserSuppressed
                  ? "user suppressed"
                  : isSystemDismissed
                    ? "system dismissed"
                    : isSystemSup
                      ? strikeOriginLabel(strikeOrigin.get(claim.claim_id))
                      : "";
                const cls = [
                  styles.claimToken,
                  decision === "promoted" ? styles.claimTokenPromoted : "",
                  isSystemSup ? styles.claimTokenSuperseded : "",
                  isSystemDismissed ? styles.claimTokenDismissed : "",
                  isUserSuppressed ? styles.claimTokenUserSuppressed : "",
                  isSystemSup ? strikeColorClass(strikeOrigin.get(claim.claim_id)) : "",
                  isRel ? styles.related : "",
                  isFoc ? styles.focused : "",
                  isHov ? styles.hovered : "",
                  isRest && !isFoc && !isHov ? styles.restThread : "",
                ]
                  .filter(Boolean)
                  .join(" ");

                return (
                  <button
                    key={claim.claim_id}
                    className={cls}
                    type="button"
                    data-claim-id={claim.claim_id}
                    onMouseEnter={() =>
                      onHover({ type: "claim", id: claim.claim_id })
                    }
                    onMouseLeave={() => onHover(null)}
                    onClick={() =>
                      onFocus({ type: "claim", id: claim.claim_id })
                    }
                  >
                    {decision === "promoted" && (
                      <span className={styles.decisionGlyph} aria-label="promoted by user">★</span>
                    )}
                    {fmtValue(claim.value)}
                    {stateLabel && <span className={styles.claimStateMarker}>{stateLabel}</span>}
                  </button>
                );
              })}
            </div>
          </div>
        ))}
      </div>
      {(weakenedPredicates.length > 0 || missingItems.length > 0) && (
        <div className={styles.coverageStrip}>
          {weakenedPredicates.length > 0 && (
            <>
              <span className={styles.coverageLabelWarn}>Unsupported</span>
              {weakenedPredicates.map((p) => (
                <span key={p} className={styles.coverageWeakened}>{p.replaceAll("_", " ")}</span>
              ))}
            </>
          )}
          {missingItems.length > 0 && (
            <>
              <span className={styles.coverageLabel}>Still needed</span>
              {missingItems.map((m, i) => (
                <span key={i} className={styles.coveragePending}>{m}</span>
              ))}
            </>
          )}
        </div>
      )}
      {claimsByPredicate.size <= 4 && flaggedLabs.length > 0 && (
        <div className={styles.patientSignals}>
          <div className={styles.signalsLabel}>Flagged values</div>
          <div className={styles.signalsRow}>
            {flaggedLabs.map((l) => (
              <span key={l.name} className={`${styles.signalPill} ${styles.signalFlagged}`}>
                {l.name} {l.value}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

/* ── differential focal space (center middle) ── */

type DifferentialRole =
  | "leading"
  | "risingChallenger"
  | "dangerousAlternative"
  | "falling"
  | "ruledDown"
  | "unresolved"
  | "peripheral";

type DifferentialZone =
  | "primary"
  | "upperLeft"
  | "upperRight"
  | "lowerLeft"
  | "lowerRight"
  | "sideLeft"
  | "sideRight"
  | "farUpperLeft"
  | "farUpperRight"
  | "farLowerLeft"
  | "farLowerRight"
  | "farSideLeft"
  | "farSideRight";

interface DifferentialSlot {
  zone: DifferentialZone;
  xPercent: number;
  yPercent: number;
  scale: number;
  opacity: number;
  fontWeight: number;
  zIndex: number;
  maxWidth: number;
  tokenDensity: number;
  parallaxDepth: number;
}

interface DifferentialLayout extends DifferentialSlot {
  id: string;
  label: string;
  rank: number;
  previousRank: number | null;
  semanticRole: DifferentialRole;
  previousSemanticRole: DifferentialRole | null;
  evidenceStrength: number;
  dangerLevel: number;
  suppressionState: ClaimTreatment;
  branchDecision: BranchDecision | null;
  destinationZone: DifferentialZone;
  previousZone: DifferentialZone | null;
  x: number;
  y: number;
}

const PRIMARY_SLOT: DifferentialSlot = {
  zone: "primary",
  xPercent: 50,
  yPercent: 45,
  scale: 1,
  opacity: 1,
  fontWeight: 650,
  zIndex: 8,
  maxWidth: 460,
  tokenDensity: 8,
  parallaxDepth: 3,
};

const SECONDARY_SLOTS: DifferentialSlot[] = [
  { zone: "upperLeft", xPercent: 18, yPercent: 15, scale: 0.88, opacity: 0.83, fontWeight: 510, zIndex: 5, maxWidth: 205, tokenDensity: 3, parallaxDepth: 5 },
  { zone: "upperRight", xPercent: 82, yPercent: 15, scale: 0.88, opacity: 0.83, fontWeight: 510, zIndex: 5, maxWidth: 205, tokenDensity: 3, parallaxDepth: 5 },
  { zone: "lowerLeft", xPercent: 24, yPercent: 72, scale: 0.86, opacity: 0.79, fontWeight: 490, zIndex: 4, maxWidth: 210, tokenDensity: 2, parallaxDepth: 4 },
  { zone: "lowerRight", xPercent: 76, yPercent: 72, scale: 0.86, opacity: 0.79, fontWeight: 490, zIndex: 4, maxWidth: 210, tokenDensity: 2, parallaxDepth: 4 },
  { zone: "sideLeft", xPercent: 13, yPercent: 48, scale: 0.84, opacity: 0.77, fontWeight: 480, zIndex: 3, maxWidth: 180, tokenDensity: 2, parallaxDepth: 3 },
  { zone: "sideRight", xPercent: 87, yPercent: 48, scale: 0.84, opacity: 0.77, fontWeight: 480, zIndex: 3, maxWidth: 180, tokenDensity: 2, parallaxDepth: 3 },
];

const TERTIARY_SLOTS: DifferentialSlot[] = [
  { zone: "farUpperLeft", xPercent: 12, yPercent: 16, scale: 0.76, opacity: 0.66, fontWeight: 420, zIndex: 2, maxWidth: 160, tokenDensity: 1, parallaxDepth: 2 },
  { zone: "farUpperRight", xPercent: 88, yPercent: 16, scale: 0.76, opacity: 0.66, fontWeight: 420, zIndex: 2, maxWidth: 160, tokenDensity: 1, parallaxDepth: 2 },
  { zone: "farLowerLeft", xPercent: 13, yPercent: 82, scale: 0.74, opacity: 0.62, fontWeight: 410, zIndex: 1, maxWidth: 155, tokenDensity: 0, parallaxDepth: 1 },
  { zone: "farLowerRight", xPercent: 87, yPercent: 82, scale: 0.74, opacity: 0.62, fontWeight: 410, zIndex: 1, maxWidth: 155, tokenDensity: 0, parallaxDepth: 1 },
  { zone: "farSideLeft", xPercent: 8, yPercent: 50, scale: 0.72, opacity: 0.60, fontWeight: 400, zIndex: 1, maxWidth: 145, tokenDensity: 0, parallaxDepth: 1 },
  { zone: "farSideRight", xPercent: 92, yPercent: 50, scale: 0.72, opacity: 0.60, fontWeight: 400, zIndex: 1, maxWidth: 145, tokenDensity: 0, parallaxDepth: 1 },
];

const DANGEROUS_BRANCH_PATTERNS = [
  "acute_coronary",
  "myocardial",
  "mi",
  "pulmonary_embolism",
  "pe",
  "aortic",
  "dissection",
  "pneumothorax",
  "stroke",
  "sepsis",
];

function dangerLevelForBranch(branch: string) {
  const normalized = branch.toLowerCase();
  return DANGEROUS_BRANCH_PATTERNS.some((pattern) => normalized.includes(pattern)) ? 2 : 0;
}

function strongestTreatment(claimIds: string[], claimsById: Map<string, Claim>, claimTreatments: Map<string, ClaimTreatment>): ClaimTreatment {
  let result: ClaimTreatment = "normal";
  for (const id of claimIds) {
    const claim = claimsById.get(id);
    if (!claim) continue;
    const treatment = claimTreatments.get(id) ?? "normal";
    if (treatment === "userSuppressed") return treatment;
    if (treatment === "systemDismissed") result = "systemDismissed";
    else if (treatment === "systemSuperseded" && result === "normal") result = "systemSuperseded";
  }
  return result;
}

function roleForHypothesis(
  h: BranchScore,
  rank: number,
  previousRank: number | null,
  dangerLevel: number,
  suppressionState: ClaimTreatment,
  branchDecision: BranchDecision | null,
): DifferentialRole {
  if (branchDecision === "demoted") return "ruledDown";
  if (rank === 0) return "leading";
  if (branchDecision === "promoted") return "risingChallenger";
  if (suppressionState === "userSuppressed" || suppressionState === "systemDismissed") return "ruledDown";
  if (dangerLevel > 0 && rank <= 5) return "dangerousAlternative";
  if (previousRank !== null && rank < previousRank) return "risingChallenger";
  if (previousRank !== null && rank > previousRank) return "falling";
  if (rank <= 2) return "risingChallenger";
  if (rank <= 5 || h.applied.length > 0) return "unresolved";
  return "peripheral";
}

const ZONE_FAMILIES: Record<DifferentialRole, DifferentialZone[]> = {
  leading: ["primary"],
  dangerousAlternative: ["upperLeft", "upperRight"],
  risingChallenger: ["upperRight", "sideRight", "upperLeft"],
  falling: ["lowerLeft", "lowerRight"],
  ruledDown: ["farLowerLeft", "farLowerRight", "farSideLeft", "farSideRight"],
  unresolved: ["sideLeft", "sideRight", "lowerLeft", "lowerRight"],
  peripheral: ["farUpperLeft", "farUpperRight", "farSideLeft", "farSideRight"],
};

const ALL_SLOTS: Map<DifferentialZone, DifferentialSlot> = new Map([
  ["primary" as DifferentialZone, PRIMARY_SLOT],
  ...SECONDARY_SLOTS.map((s) => [s.zone, s] as [DifferentialZone, DifferentialSlot]),
  ...TERTIARY_SLOTS.map((s) => [s.zone, s] as [DifferentialZone, DifferentialSlot]),
]);

function stableHash(s: string): number {
  let h = 0;
  for (let i = 0; i < s.length; i++) h = ((h << 5) - h + s.charCodeAt(i)) | 0;
  return Math.abs(h);
}

function buildDifferentialLayouts(
  hypotheses: BranchScore[],
  claimsById: Map<string, Claim>,
  claimTreatments: Map<string, ClaimTreatment>,
  branchDecisionsById: Record<string, BranchDecision>,
  previous: Map<string, DifferentialLayout>,
): DifferentialLayout[] {
  const occupied = new Set<DifferentialZone>();
  const layouts: DifferentialLayout[] = [];

  for (let rank = 0; rank < hypotheses.length; rank++) {
    const h = hypotheses[rank]!;
    const previousLayout = previous.get(h.branch) ?? null;
    const claimIds = h.applied.map((a) => a.claim_id);
    const suppressionState = strongestTreatment(claimIds, claimsById, claimTreatments);
    const evidenceStrength = h.applied.reduce((sum, applied) => sum + Math.abs(applied.log_lr), 0);
    const dangerLevel = dangerLevelForBranch(h.branch);
    const branchDecision = branchDecisionsById[h.branch] ?? null;
    const semanticRole = roleForHypothesis(h, rank, previousLayout?.rank ?? null, dangerLevel, suppressionState, branchDecision);

    let zone: DifferentialZone;

    if (previousLayout && previousLayout.semanticRole === semanticRole && !occupied.has(previousLayout.destinationZone)) {
      zone = previousLayout.destinationZone;
    } else {
      const family = ZONE_FAMILIES[semanticRole];
      const startIdx = stableHash(h.branch) % family.length;
      zone = family[startIdx]!;
      let attempts = 0;
      while (occupied.has(zone) && attempts < family.length) {
        attempts++;
        zone = family[(startIdx + attempts) % family.length]!;
      }
      if (occupied.has(zone)) {
        for (const [z] of ALL_SLOTS) {
          if (!occupied.has(z) && z !== "primary") { zone = z; break; }
        }
      }
    }

    occupied.add(zone);
    const slot = ALL_SLOTS.get(zone) ?? TERTIARY_SLOTS[TERTIARY_SLOTS.length - 1]!;

    layouts.push({
      ...slot,
      zone,
      id: h.branch,
      label: displayBranch(h.branch),
      rank,
      previousRank: previousLayout?.rank ?? null,
      semanticRole,
      previousSemanticRole: previousLayout?.semanticRole ?? null,
      evidenceStrength,
      dangerLevel,
      suppressionState,
      branchDecision,
      destinationZone: zone,
      previousZone: previousLayout?.destinationZone ?? null,
      x: slot.xPercent,
      y: slot.yPercent,
    });
  }

  if (import.meta.env.DEV) {
    console.table(layouts.map((l) => ({
      id: l.id, role: l.semanticRole, prevRole: l.previousSemanticRole,
      zone: l.destinationZone, prevZone: l.previousZone,
      moved: l.destinationZone !== l.previousZone,
      reason: l.previousSemanticRole == null ? "initial" : l.semanticRole !== l.previousSemanticRole ? "role change" : "stable",
    })));
  }

  return layouts;
}

function DifferentialFocal(props: {
  hypotheses: BranchScore[];
  allClaims: Claim[];
  claimsById: Map<string, Claim>;
  claimTreatments: Map<string, ClaimTreatment>;
  branchDecisionsById: Record<string, BranchDecision>;
  claimDecisionsById: Record<string, ClaimDecision>;
  hoverTarget: HoverTarget;
  focusTarget: FocusTarget | null;
  data: CanvasData;
  onHover: (t: HoverTarget) => void;
  onFocus: (t: FocusTarget) => void;
}) {
  const {
    hypotheses,
    allClaims,
    claimsById,
    claimTreatments,
    branchDecisionsById,
    claimDecisionsById,
    hoverTarget,
    focusTarget,
    onHover,
    onFocus,
  } = props;
  const verifier = useSession((s) => s.verifier);
  const reduceMotion = useReducedMotion();
  const containerRef = useRef<HTMLDivElement>(null);
  const rafId = useRef<number | null>(null);
  const previousLayoutsRef = useRef<Map<string, DifferentialLayout>>(new Map());

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (reduceMotion || rafId.current !== null) return;
    rafId.current = requestAnimationFrame(() => {
      const rect = containerRef.current?.getBoundingClientRect();
      if (!rect) { rafId.current = null; return; }
      const x = Math.max(-1, Math.min(1, (e.clientX - rect.left - rect.width / 2) / rect.width)).toFixed(3);
      const y = Math.max(-1, Math.min(1, (e.clientY - rect.top - rect.height / 2) / rect.height)).toFixed(3);
      containerRef.current?.style.setProperty("--mx", x);
      containerRef.current?.style.setProperty("--my", y);
      rafId.current = null;
    });
  }, [reduceMotion]);

  const handleMouseLeave = useCallback(() => {
    if (rafId.current !== null) { cancelAnimationFrame(rafId.current); rafId.current = null; }
    containerRef.current?.style.setProperty("--mx", "0");
    containerRef.current?.style.setProperty("--my", "0");
  }, []);

  useEffect(() => {
    return () => { if (rafId.current !== null) cancelAnimationFrame(rafId.current); };
  }, []);

  const visualHypotheses = useMemo(() => {
    const decisionWeight = (branch: string) =>
      branchDecisionsById[branch] === "promoted" ? 1000 :
      branchDecisionsById[branch] === "demoted" ? -1000 :
      0;
    return [...hypotheses].sort((a, b) => {
      const byDecision = decisionWeight(b.branch) - decisionWeight(a.branch);
      return byDecision || b.log_score - a.log_score;
    });
  }, [branchDecisionsById, hypotheses]);

  const layouts = useMemo(
    () => buildDifferentialLayouts(visualHypotheses, claimsById, claimTreatments, branchDecisionsById, previousLayoutsRef.current),
    [visualHypotheses, claimsById, claimTreatments, branchDecisionsById],
  );

  const maxEvidenceStrength = useMemo(
    () => Math.max(1, ...layouts.map((l) => l.evidenceStrength)),
    [layouts],
  );

  useEffect(() => {
    previousLayoutsRef.current = new Map(layouts.map((layout) => [layout.id, layout]));
  }, [layouts]);

  if (visualHypotheses.length === 0) {
    return (
      <div className={styles.differential} data-surface="differential">
        <div className={styles.emptyDifferential}>
          {allClaims.length > 2 ? "Building differential..." : ""}
        </div>
      </div>
    );
  }

  const primary = visualHypotheses[0]!;
  const rest = visualHypotheses.slice(1);
  const layoutById = new Map(layouts.map((layout) => [layout.id, layout]));

  const getEvidenceTokens = (h: BranchScore, tokenDensity: number) => {
    const ids = tokenDensity >= 6
      ? relClaimIds(h, allClaims)
      : h.applied.map((x) => x.claim_id);
    return ids
      .map((id) => claimsById.get(id))
      .filter(isClaim)
      .slice(0, tokenDensity)
      .map((c) => ({
        id: c.claim_id,
        text: fmtValue(c.value),
        treatment: claimTreatments.get(c.claim_id) ?? "normal",
        decision: claimDecisionsById[c.claim_id] ?? null,
        weakening: c.value.startsWith("negated:") || normStatus(c.status) === "superseded",
      }));
  };

  const breathe = (layout: DifferentialLayout) => {
    const t = maxEvidenceStrength > 0 ? layout.evidenceStrength / maxEvidenceStrength : 0;
    const promoted = layout.branchDecision === "promoted" ? 1 : 0;
    const demoted = layout.branchDecision === "demoted" ? 1 : 0;
    return {
      scale: Math.max(0.68, layout.scale + t * 0.05 + promoted * 0.06 - demoted * 0.05),
      opacity: Math.min(1, Math.max(0.48, layout.opacity + t * 0.10 + promoted * 0.10 - demoted * 0.12)),
      fontWeight: Math.round(layout.fontWeight + t * 50 + promoted * 55 - demoted * 45),
    };
  };

  const primaryLayout = layoutById.get(primary.branch);
  const primaryBreathe = primaryLayout ? breathe(primaryLayout) : { scale: 1, opacity: 1, fontWeight: 650 };
  const primaryTokens = getEvidenceTokens(primary, primaryLayout?.tokenDensity ?? 8);
  const primaryDecision = branchDecisionsById[primary.branch] ?? null;
  const isFocPrimary =
    focusTarget?.type === "hypothesis" && focusTarget.id === primary.branch;
  const isHovPrimary =
    hoverTarget?.type === "hypothesis" && hoverTarget.id === primary.branch;

  return (
    <div
      ref={containerRef}
      className={styles.differential}
      data-surface="differential"
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      style={{ "--mx": "0", "--my": "0" } as React.CSSProperties}
    >
      {/* Primary hypothesis — center, sharp, closest */}
      <motion.div
        className={[
          styles.differentialItem,
          styles.focalPrimary,
          primaryDecision === "promoted" ? styles.hypothesisPromoted : "",
          primaryDecision === "demoted" ? styles.hypothesisDemoted : "",
          isFocPrimary ? styles.focused : "",
          isHovPrimary ? styles.hovered : "",
          styles.related,
        ].filter(Boolean).join(" ")}
        style={{
          "--x": `${primaryLayout?.xPercent ?? 50}%`,
          "--y": `${primaryLayout?.yPercent ?? 45}%`,
          "--parallax": `${primaryLayout?.parallaxDepth ?? 3}px`,
          "--slot-scale": `${primaryBreathe.scale}`,
          "--z-depth": "0px",
          maxWidth: primaryLayout?.maxWidth ?? 460,
          zIndex: primaryLayout?.zIndex ?? 8,
        } as React.CSSProperties}
        animate={{ opacity: primaryBreathe.opacity }}
        transition={{ duration: reduceMotion ? 0.01 : 0.26, ease: EXPO_OUT }}
        onMouseEnter={() => onHover({ type: "hypothesis", id: primary.branch })}
        onMouseLeave={() => onHover(null)}
        onClick={() => onFocus({ type: "hypothesis", id: primary.branch })}
      >
        <h2 className={styles.focalPrimaryName} style={{ fontWeight: primaryBreathe.fontWeight }}>
          {primaryDecision === "promoted" && (
            <span className={styles.hypothesisDecisionGlyph} aria-label="promoted by user">★</span>
          )}
          {displayBranch(primary.branch)}
          {primaryDecision === "demoted" && (
            <span className={styles.hypothesisDecisionLabel}>downvoted</span>
          )}
        </h2>
        {primaryTokens.length > 0 && (
          <div className={styles.focalPrimaryEvidence}>
            {primaryTokens.map((t) => (
              <button
                key={t.id}
                className={[
                  styles.focalEvidenceToken,
                  t.decision === "promoted" ? styles.focalEvidencePromoted : "",
                  t.weakening || t.treatment === "systemSuperseded" ? styles.focalEvidenceWeakening : "",
                  t.treatment === "systemDismissed" ? styles.focalEvidenceDismissed : "",
                  t.treatment === "userSuppressed" ? styles.focalEvidenceUserSuppressed : "",
                ].filter(Boolean).join(" ")}
                type="button"
                onMouseEnter={() => onHover({ type: "claim", id: t.id })}
                onMouseLeave={() => onHover(null)}
                onClick={(e) => { e.stopPropagation(); onFocus({ type: "claim", id: t.id }); }}
              >
                {t.decision === "promoted" && (
                  <span className={styles.decisionGlyph} aria-label="promoted by user">★</span>
                )}
                {t.text}
              </button>
            ))}
          </div>
        )}
        {primaryTokens.length >= 3 && (
          <p className={styles.reasoningNarrative}>
            {displayBranch(primary.branch)} is the primary consideration based on{" "}
            {primaryTokens.filter((t) => !t.weakening && t.treatment === "normal").map((t) => t.text).join(", ")}
            {primaryTokens.some((t) => t.weakening || t.treatment !== "normal")
              ? `. ${primaryTokens.filter((t) => t.weakening || t.treatment !== "normal").map((t) => t.text).join(", ")} ${primaryTokens.filter((t) => t.weakening || t.treatment !== "normal").length > 1 ? "are" : "is"} less consistent.`
              : "."}
          </p>
        )}
        {verifier && verifier.why_moved.length > 0 && (
          <div className={styles.whyMoved}>
            <div className={styles.whyMovedLabel}>Reasoning shift</div>
            {verifier.why_moved.map((w, i) => <div key={i}>{w}</div>)}
          </div>
        )}
        {verifier && verifier.missing_or_contradicting.length > 0 && (
          <p className={styles.missingFeatures}>
            Still needed: {verifier.missing_or_contradicting.join(", ")}
          </p>
        )}
      </motion.div>

      {/* Non-primary hypotheses — spatially positioned around center */}
      {rest.map((h) => {
        const layout = layoutById.get(h.branch);
        if (!layout) return null;
        const tier = SECONDARY_SLOTS.some((slot) => slot.zone === layout.destinationZone) ? "secondary" : "tertiary";
        const tokens = getEvidenceTokens(h, layout.tokenDensity);
        const isFoc = focusTarget?.type === "hypothesis" && focusTarget.id === h.branch;
        const isHov = hoverTarget?.type === "hypothesis" && hoverTarget.id === h.branch;
        const tierCls = tier === "secondary" ? styles.spatialSecondary : styles.spatialTertiary;
        const b = breathe(layout);

        return (
          <motion.div
            key={h.branch}
            className={[
              styles.differentialItem,
              styles.spatialHypothesis,
              tierCls,
              layout.semanticRole === "dangerousAlternative" ? styles.dangerousAlternative : "",
              layout.semanticRole === "ruledDown" ? styles.ruledDownHypothesis : "",
              layout.branchDecision === "promoted" ? styles.hypothesisPromoted : "",
              layout.branchDecision === "demoted" ? styles.hypothesisDemoted : "",
              layout.suppressionState === "userSuppressed" ? styles.userSuppressedHypothesis : "",
              isFoc ? styles.focused : "",
              isHov ? styles.hovered : "",
            ].filter(Boolean).join(" ")}
            style={{
              "--x": `${layout.xPercent}%`,
              "--y": `${layout.yPercent}%`,
              "--parallax": `${layout.parallaxDepth}px`,
              "--slot-scale": `${b.scale}`,
              "--z-depth": "0px",
              "--blur": "0px",
              maxWidth: layout.maxWidth,
              zIndex: layout.zIndex,
            } as React.CSSProperties}
            animate={{ opacity: b.opacity }}
            transition={{ duration: reduceMotion ? 0.01 : 0.28, ease: EXPO_OUT }}
            data-role={layout.semanticRole}
            data-zone={layout.destinationZone}
            data-debug={import.meta.env.DEV ? JSON.stringify({
              id: h.branch, rank: layout.rank, prevRank: layout.previousRank,
              role: layout.semanticRole, prevRole: layout.previousSemanticRole,
              zone: layout.destinationZone, prevZone: layout.previousZone,
              decision: layout.branchDecision,
              moved: layout.destinationZone !== layout.previousZone,
              reason: layout.previousSemanticRole == null ? "initial"
                : layout.semanticRole !== layout.previousSemanticRole ? `role: ${layout.previousSemanticRole} → ${layout.semanticRole}`
                : "stable",
            }) : undefined}
            onMouseEnter={() => onHover({ type: "hypothesis", id: h.branch })}
            onMouseLeave={() => onHover(null)}
            onClick={() => onFocus({ type: "hypothesis", id: h.branch })}
          >
            <div
              className={tier === "secondary" ? styles.focalSecondaryName : styles.focalTertiaryName}
              style={{ fontWeight: b.fontWeight }}
            >
              {layout.branchDecision === "promoted" && (
                <span className={styles.hypothesisDecisionGlyph} aria-label="promoted by user">★</span>
              )}
              {displayBranch(h.branch)}
              {layout.branchDecision === "demoted" && (
                <span className={styles.hypothesisDecisionLabel}>downvoted</span>
              )}
            </div>
            {tokens.length > 0 && (
              <div className={styles.focalSecondaryEvidence}>
                {tokens.map((t) => (
                  <button
                    key={t.id}
                    className={[
                      styles.focalSecondaryToken,
                      t.decision === "promoted" ? styles.focalEvidencePromoted : "",
                      t.weakening || t.treatment === "systemSuperseded" ? styles.focalEvidenceWeakening : "",
                      t.treatment === "systemDismissed" ? styles.focalEvidenceDismissed : "",
                      t.treatment === "userSuppressed" ? styles.focalEvidenceUserSuppressed : "",
                    ].filter(Boolean).join(" ")}
                    type="button"
                    onMouseEnter={() => onHover({ type: "claim", id: t.id })}
                    onMouseLeave={() => onHover(null)}
                    onClick={(e) => { e.stopPropagation(); onFocus({ type: "claim", id: t.id }); }}
                  >
                    {t.decision === "promoted" && (
                      <span className={styles.decisionGlyph} aria-label="promoted by user">★</span>
                    )}
                    {t.text}
                  </button>
                ))}
              </div>
            )}
          </motion.div>
        );
      })}
    </div>
  );
}

/* ── next question (center bottom) ── */

function NextQuestion() {
  const verifier = useSession((s) => s.verifier);
  const ranking = useSession((s) => s.ranking);
  const sorted = useMemo(
    () => [...(ranking?.scores ?? [])].sort((a, b) => b.log_score - a.log_score),
    [ranking],
  );
  const rulesOutLabel = sorted[1] ? displayBranch(sorted[1].branch) : "";

  if (!verifier?.next_best_question) {
    return (
      <div className={styles.nextQuestion}>
        <div className={styles.nextQuestionHeader}>Next Question</div>
        <div className={styles.emptyQuestion}>
          {sorted.length > 0
            ? "Generating next question..."
            : "Waiting for differential..."}
        </div>
      </div>
    );
  }

  return (
    <div className={styles.nextQuestion}>
      <div className={styles.nextQuestionHeader}>Next Question</div>
      <div className={styles.questionRow}>
        <span className={`${styles.qLabel} ${styles.qLabelAsk}`}>
          Ask next
        </span>
        <span className={styles.qText}>
          {verifier.next_best_question}
        </span>
      </div>
      {rulesOutLabel && (
        <div className={styles.questionRow}>
          <span className={`${styles.qLabel} ${styles.qLabelRuleOut}`}>
            Rule out
          </span>
          <span className={styles.qTextMuted}>
            Does the presentation better fit {rulesOutLabel}?
          </span>
        </div>
      )}
      {verifier.next_question_rationale && (
        <div className={styles.questionRationale}>
          {verifier.next_question_rationale}
        </div>
      )}
    </div>
  );
}

/* ── context panel (right) — tabbed: Context | Documents | Note ── */

const SOAP_SECTION_COLOR: Record<string, string> = {
  S: styles.soapColorS!,
  O: styles.soapColorO!,
  A: styles.soapColorA!,
  P: styles.soapColorP!,
};

function ContextPanel(props: {
  sentences: NoteSentence[];
  hoverTarget: HoverTarget;
  focusTarget: FocusTarget | null;
  relatedSentences: Set<string>;
  restAssoc: Set<string>;
  matchedLabels: Set<string>;
  pulsingLabels: Set<string>;
  pulsingClaimIds: Set<string>;
  data: CanvasData;
  claimTreatments: Map<string, ClaimTreatment>;
  onHover: (t: HoverTarget) => void;
  onFocus: (t: FocusTarget) => void;
}) {
  const {
    sentences,
    hoverTarget,
    focusTarget,
    relatedSentences,
    restAssoc,
    matchedLabels,
    pulsingLabels,
    pulsingClaimIds,
    data,
    claimTreatments,
    onHover,
    onFocus,
  } = props;

  const [rightTab, setRightTab] = useState<"context" | "documents" | "note">("context");
  const [reviewed, setReviewed] = useState<Set<string>>(new Set());
  const prevSentenceCount = useRef(0);
  const rightTabs = ["context", "documents", "note"] as const;

  // Auto-switch to note tab when SOAP arrives
  useEffect(() => {
    if (sentences.length > 0 && prevSentenceCount.current === 0) {
      setRightTab("note");
    }
    prevSentenceCount.current = sentences.length;
  }, [sentences.length]);

  const toggleReviewed = useCallback((id: string) => {
    setReviewed((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  const allReviewed = sentences.length > 0 && reviewed.size === sentences.length;

  /* Context tab content */
  const contextContent = (
    <>
      <div className={styles.contextSection}>
        <div className={styles.contextSectionTitle}>
          <span className={`${styles.contextSectionIcon} ${styles.iconProblems}`}>P</span>
          Active Problems
        </div>
        {DEMO_PATIENT.problems.map((p) => (
          <div
            key={p}
            className={[
              styles.contextItem,
              matchedLabels.has(p) ? styles.restThread : "",
              pulsingLabels.has(p) ? styles.contextPulse : "",
            ].filter(Boolean).join(" ")}
          >
            {p}
          </div>
        ))}
      </div>

      <div className={styles.contextSection}>
        <div className={styles.contextSectionTitle}>
          <span className={`${styles.contextSectionIcon} ${styles.iconMeds}`}>M</span>
          Medications
        </div>
        {DEMO_PATIENT.medications.map((m) => (
          <div
            key={m}
            className={[
              styles.contextItem,
              matchedLabels.has(m) ? styles.restThread : "",
              pulsingLabels.has(m) ? styles.contextPulse : "",
            ].filter(Boolean).join(" ")}
          >
            {m}
          </div>
        ))}
      </div>

      <div className={styles.contextSection}>
        <div className={styles.contextSectionTitle}>
          <span className={`${styles.contextSectionIcon} ${styles.iconAllergy}`}>A</span>
          Allergies
        </div>
        {DEMO_PATIENT.allergies.map((a) => (
          <div
            key={a}
            className={[
              styles.contextItem,
              matchedLabels.has(a) ? styles.restThread : "",
              pulsingLabels.has(a) ? styles.contextPulse : "",
            ].filter(Boolean).join(" ")}
          >
            {a}
          </div>
        ))}
      </div>

      <div className={styles.contextSection}>
        <div className={styles.contextSectionTitle}>
          <span className={`${styles.contextSectionIcon} ${styles.iconVisits}`}>V</span>
          Prior Visits
        </div>
        {DEMO_PATIENT.priorVisits.map((v) => (
          <div key={v.date} className={styles.priorVisit}>
            <div className={styles.priorVisitDate}>{v.date}</div>
            <div className={styles.priorVisitChief}>{v.chief}</div>
            <div className={styles.priorVisitSummary}>{v.summary}</div>
          </div>
        ))}
      </div>
    </>
  );

  /* Documents tab content */
  const documentsContent = (
    <>
      {DEMO_PATIENT.documents.map((doc) => (
        <div key={`${doc.type}-${doc.date}`} className={styles.docCard}>
          <div className={styles.docCardHeader}>
            <span
              className={`${styles.docCardType} ${
                doc.type === "lab"
                  ? styles.docTypeLab
                  : doc.type === "imaging"
                    ? styles.docTypeImaging
                    : styles.docTypeProcedure
              }`}
            >
              {doc.type}
            </span>
            <span className={styles.docCardTitle}>{doc.title}</span>
            <span className={styles.docCardDate}>{doc.date}</span>
          </div>
          {doc.items && doc.items.map((item) => (
            <div
              key={item.name}
              className={[
                styles.labRow,
                matchedLabels.has(item.name) ? styles.restThread : "",
                pulsingLabels.has(item.name) ? styles.contextPulse : "",
              ].filter(Boolean).join(" ")}
            >
              <span className={styles.labName}>{item.name}</span>
              <span
                className={`${styles.labValue} ${
                  item.flag === "high"
                    ? styles.labFlagHigh
                    : item.flag === "low"
                      ? styles.labFlagLow
                      : styles.labFlagNormal
                }`}
              >
                {item.value}
                {item.flag !== "normal" && (
                  <> {item.flag === "high" ? "HIGH" : "LOW"}</>
                )}
              </span>
            </div>
          ))}
          {doc.result && (
            <div
              className={[
                styles.docResult,
                matchedLabels.has(doc.title) ? styles.restThread : "",
                pulsingLabels.has(doc.title) ? styles.contextPulse : "",
              ].filter(Boolean).join(" ")}
            >
              {doc.result}
            </div>
          )}
        </div>
      ))}
    </>
  );

  /* Note tab content — SOAP with review flow (Fix 3) */
  const grouped = new Map<string, NoteSentence[]>();
  for (const s of sentences) {
    const g = grouped.get(s.section) ?? [];
    g.push(s);
    grouped.set(s.section, g);
  }

  const noteContent = sentences.length === 0 ? (
    <div className={styles.emptyState}>Waiting for clinical note...</div>
  ) : (
    <>
      <div className={styles.soapTransition}>
        {["S", "O", "A", "P"].map((section) => {
          const sectionSentences = grouped.get(section);
          if (!sectionSentences || sectionSentences.length === 0) return null;
          return (
            <div key={section} className={styles.soapSection}>
              <div className={styles.soapSectionHeaderRow}>
                <span className={`${styles.soapSectionColorBar} ${SOAP_SECTION_COLOR[section] ?? ""}`} />
                <div className={styles.soapSectionLabel}>
                  {SECTION_NAMES[section] ?? section}
                </div>
                <span className={styles.soapSectionCount}>({sectionSentences.length})</span>
              </div>
              {sectionSentences.map((s) => {
                const hasRestAssoc = s.source_claim_ids.some((id) =>
                  restAssoc.has(id),
                );
                const hasPulse = s.source_claim_ids.some((id) =>
                  pulsingClaimIds.has(id),
                );
                const sentenceSuppression = s.source_claim_ids.some((id) => claimTreatments.get(id) === "userSuppressed")
                  ? "userSuppressed"
                  : s.source_claim_ids.some((id) => claimTreatments.get(id) === "systemDismissed")
                    ? "systemDismissed"
                    : s.source_claim_ids.some((id) => claimTreatments.get(id) === "systemSuperseded")
                      ? "systemSuperseded"
                      : "normal";
                const isFoc =
                  focusTarget?.type === "sentence" &&
                  focusTarget.id === s.sentence_id;
                const isHov =
                  hoverTarget?.type === "sentence" &&
                  hoverTarget.id === s.sentence_id;
                const isRel = relatedSentences.has(s.sentence_id);
                const isReviewed = reviewed.has(s.sentence_id);

                return (
                  <div key={s.sentence_id}>
                    <div
                      className={[
                        styles.soapSentenceRow,
                        hasRestAssoc ? styles.restThread : "",
                        hasPulse ? styles.contextPulse : "",
                      ].filter(Boolean).join(" ")}
                    >
                      <button
                        className={isReviewed ? styles.soapReviewBtnReviewed : styles.soapReviewBtn}
                        type="button"
                        aria-label={isReviewed ? "Sentence reviewed" : "Mark sentence as reviewed"}
                        aria-pressed={isReviewed}
                        onClick={() => toggleReviewed(s.sentence_id)}
                      >
                        <span>{isReviewed ? "Reviewed" : "Review"}</span>
                        {isReviewed && <span style={{ color: "#FFFFFF", fontSize: 10, lineHeight: 1 }}>✓</span>}
                      </button>
                      <button
                        className={[
                        styles.soapSentenceText,
                        sentenceSuppression === "userSuppressed" ? styles.soapUserSuppressed : "",
                        sentenceSuppression === "systemDismissed" ? styles.soapSystemDismissed : "",
                        sentenceSuppression === "systemSuperseded" ? styles.soapSystemSuperseded : "",
                        isFoc ? styles.focused : "",
                          isHov ? styles.hovered : "",
                          isRel ? styles.related : "",
                        ].filter(Boolean).join(" ")}
                        type="button"
                        onMouseEnter={() =>
                          onHover({ type: "sentence", id: s.sentence_id })
                        }
                        onMouseLeave={() => onHover(null)}
                        onClick={() =>
                          onFocus({ type: "sentence", id: s.sentence_id })
                        }
                      >
                        {s.text}
                      </button>
                    </div>
                    {s.source_claim_ids.length > 0 && (
                      <div className={styles.soapSourcePills}>
                        {s.source_claim_ids.map((cid) => {
                          const c = data.claimsById.get(cid);
                          if (!c) return null;
                          const treatment = claimTreatments.get(cid) ?? "normal";
                          return (
                            <button
                              key={cid}
                              className={[
                                styles.soapSourcePill,
                                treatment === "userSuppressed" ? styles.sourcePillUserSuppressed : "",
                                treatment === "systemDismissed" ? styles.sourcePillSystemDismissed : "",
                                treatment === "systemSuperseded" ? styles.sourcePillSystemSuperseded : "",
                              ].filter(Boolean).join(" ")}
                              type="button"
                              onClick={() => onFocus({ type: "claim", id: cid })}
                            >
                              {c.predicate.replaceAll("_", " ")}
                            </button>
                          );
                        })}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          );
        })}
      </div>
      <div className={styles.soapActions}>
        <div className={styles.soapProgress}>
          {reviewed.size}/{sentences.length} reviewed
        </div>
        <button
          className={allReviewed ? styles.sendToEhrBtn : styles.sendToEhrBtnDisabled}
          type="button"
          disabled={!allReviewed}
        >
          Send to EHR
        </button>
        <button className={styles.enableAftercareBtn} type="button">
          Enable Patient Aftercare
        </button>
      </div>
    </>
  );

  return (
    <aside className={styles.context} data-surface="context">
      <div
        className={styles.rightTabBar}
        role="tablist"
        onKeyDown={(e) => {
          if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
          e.preventDefault();
          const current = rightTabs.indexOf(rightTab);
          const next =
            e.key === "ArrowRight"
              ? (current + 1) % rightTabs.length
              : (current + rightTabs.length - 1) % rightTabs.length;
          setRightTab(rightTabs[next]!);
        }}
      >
        <button
          className={rightTab === "context" ? styles.rightTabActive : styles.rightTab}
          type="button"
          role="tab"
          aria-selected={rightTab === "context"}
          onClick={() => setRightTab("context")}
        >
          Context
        </button>
        <button
          className={rightTab === "documents" ? styles.rightTabActive : styles.rightTab}
          type="button"
          role="tab"
          aria-selected={rightTab === "documents"}
          onClick={() => setRightTab("documents")}
        >
          Documents
        </button>
        <button
          className={rightTab === "note" ? styles.rightTabActive : styles.rightTab}
          type="button"
          role="tab"
          aria-selected={rightTab === "note"}
          onClick={() => setRightTab("note")}
        >
          Note{sentences.length > 0 ? ` (${sentences.length})` : ""}
        </button>
      </div>
      <div className={styles.rightTabContent} role="tabpanel">
        {rightTab === "context" && contextContent}
        {rightTab === "documents" && documentsContent}
        {rightTab === "note" && noteContent}
      </div>
    </aside>
  );
}

/* ── conversation rail ── */

type RecordState = "idle" | "recording" | "paused";

interface SpeakerLane {
  speakerId: string;
  speakerLabel: string;
  speakerRole: "clinician" | "patient" | "caregiver" | "nurse" | "specialist" | "other";
}

interface SpeakerSegment {
  id: string;
  speakerId: string;
  startMs: number;
  endMs: number;
  transcriptTurnId?: string;
  estimatedEnd: boolean;
}

interface TimelineMarker {
  id: string;
  type: "claim" | "dismissal" | "supersession" | "soap" | "reasoning_shift" | "risk" | "contradiction";
  timeMs: number;
  endMs?: number;
  relatedSpeakerId?: string;
  turnId?: string;
}

function fmtTimeMs(ms: number) {
  return fmtTime(ms * 1_000_000);
}

function nsToMs(ns: number) {
  return Math.max(0, ns / 1_000_000);
}

function clampTime(ms: number, durationMs: number) {
  return Math.max(0, Math.min(durationMs, ms));
}

function speakerForTurn(turn: Turn): SpeakerLane {
  if (turn.speaker === "physician") {
    return { speakerId: "physician", speakerLabel: "Physician", speakerRole: "clinician" };
  }
  if (turn.speaker === "patient") {
    return { speakerId: "patient", speakerLabel: "Patient", speakerRole: "patient" };
  }
  return { speakerId: turn.speaker, speakerLabel: "System", speakerRole: "other" };
}

function arraysEqual(a: readonly string[], b: readonly string[]) {
  if (a.length !== b.length) return false;
  return a.every((item, index) => item === b[index]);
}

function activeTranscriptIdAt(timeMs: number, turns: Turn[]) {
  for (let i = turns.length - 1; i >= 0; i--) {
    const turn = turns[i]!;
    if (nsToMs(turn.ts) <= timeMs) return turn.turn_id;
  }
  return turns[0]?.turn_id ?? null;
}

function activeSpeakerIdsAt(timeMs: number, segments: SpeakerSegment[]) {
  return segments
    .filter((segment) => segment.startMs <= timeMs && timeMs < segment.endMs)
    .map((segment) => segment.speakerId)
    .sort();
}

function ConversationRail({
  data,
  followTranscript,
  onPlaybackTurnChange,
  onSeekToTurn,
}: {
  data: CanvasData;
  followTranscript: boolean;
  onPlaybackTurnChange: (turnId: string | null) => void;
  onSeekToTurn: (turnId: string | null) => void;
}) {
  const svgRef = useRef<SVGSVGElement>(null);
  const waveAreaRef = useRef<HTMLDivElement>(null);
  const rafRef = useRef<number | null>(null);
  const currentTimeRef = useRef(0);
  const lastPerfRef = useRef(0);
  const isPlayingRef = useRef(false);
  const playbackRateRef = useRef(1);
  const isScrubbingRef = useRef(false);
  const wasPlayingBeforeScrubRef = useRef(false);
  const activeTurnRef = useRef<string | null>(null);
  const activeSpeakerIdsRef = useRef<string[]>([]);
  const coarseDisplayBucketRef = useRef(-1);
  const [recordState, setRecordState] = useState<RecordState>("idle");
  const [selectedTimeMs, setSelectedTimeMs] = useState<number | null>(null);
  const [speedIndex, setSpeedIndex] = useState(1);
  const [currentTimeMs, setCurrentTimeMs] = useState(0);
  const [isScrubbing, setIsScrubbing] = useState(false);
  const [scrubTimeMs, setScrubTimeMs] = useState(0);
  const [activeSpeakerIds, setActiveSpeakerIds] = useState<string[]>([]);

  const isRecording = recordState === "recording";
  const hasData = data.turns.length > 0;

  useEffect(() => {
    if (selectedTimeMs === null) return;
    const t = setTimeout(() => setSelectedTimeMs(null), 1600);
    return () => clearTimeout(t);
  }, [selectedTimeMs]);

  const ENCOUNTER_DURATION_MS = 240_000;
  const latestInputTimeMs = useMemo(() => {
    const turnTimes = data.turns.map((turn) => nsToMs(turn.ts));
    const claimTimes = [...data.claimsById.values()].map((claim) => nsToMs(claim.created_ts));
    return Math.max(0, ...turnTimes, ...claimTimes);
  }, [data.turns, data.claimsById]);
  const durationMs = Math.max(ENCOUNTER_DURATION_MS, latestInputTimeMs * 1.1);
  const expectedDurationNs = durationMs * 1_000_000;
  const currentTime = fmtTimeMs(currentTimeMs);
  const totalTime = fmtTimeMs(durationMs);
  const livePct = durationMs > 0 ? Math.min(100, (currentTimeMs / durationMs) * 100) : 0;

  const waveHeights = useMemo(
    () => buildWaveform(data.turns, expectedDurationNs),
    [data.turns, expectedDurationNs],
  );
  const svgW = 960;
  const playbackRates = [0.5, 1, 1.5] as const;
  const playbackRate = playbackRates[speedIndex] ?? 1;

  useEffect(() => {
    playbackRateRef.current = playbackRate;
  }, [playbackRate]);

  const speakers = useMemo<SpeakerLane[]>(() => {
    const lanes = new Map<string, SpeakerLane>();
    lanes.set("physician", { speakerId: "physician", speakerLabel: "Physician", speakerRole: "clinician" });
    lanes.set("patient", { speakerId: "patient", speakerLabel: "Patient", speakerRole: "patient" });
    for (const turn of data.turns) {
      const speaker = speakerForTurn(turn);
      lanes.set(speaker.speakerId, speaker);
    }
    return [...lanes.values()];
  }, [data.turns]);

  const speakerSegments = useMemo<SpeakerSegment[]>(() => {
    return data.turns.map((turn, index) => {
      const startMs = nsToMs(turn.ts);
      const nextStartMs = data.turns[index + 1] ? nsToMs(data.turns[index + 1]!.ts) : durationMs;
      const estimatedSpeechMs = Math.min(10_000, Math.max(1_200, turn.text.length * 48));
      const estimatedEndMs = Math.min(durationMs, startMs + estimatedSpeechMs);
      const endMs = nextStartMs > startMs ? Math.min(nextStartMs, estimatedEndMs) : estimatedEndMs;
      return {
        id: turn.turn_id,
        speakerId: speakerForTurn(turn).speakerId,
        startMs,
        endMs: Math.max(startMs + 500, endMs),
        transcriptTurnId: turn.turn_id,
        estimatedEnd: true,
      };
    });
  }, [data.turns, durationMs]);

  const overlapWindows = useMemo(() => {
    const overlaps: Array<{ id: string; startMs: number; endMs: number }> = [];
    for (let i = 0; i < speakerSegments.length; i++) {
      for (let j = i + 1; j < speakerSegments.length; j++) {
        const a = speakerSegments[i]!;
        const b = speakerSegments[j]!;
        if (a.speakerId === b.speakerId) continue;
        const startMs = Math.max(a.startMs, b.startMs);
        const endMs = Math.min(a.endMs, b.endMs);
        if (endMs > startMs) overlaps.push({ id: `${a.id}-${b.id}`, startMs, endMs });
      }
    }
    return overlaps;
  }, [speakerSegments]);

  const markers = useMemo<TimelineMarker[]>(() => {
    const m: TimelineMarker[] = [];
    for (const c of data.claimsById.values()) {
      const st = normStatus(c.status);
      m.push({
        id: c.claim_id,
        timeMs: nsToMs(c.created_ts),
        type: st === "superseded" ? "supersession" : st === "dismissed" ? "dismissal" : "claim",
        turnId: c.source_turn_id,
        relatedSpeakerId: data.turns.find((turn) => turn.turn_id === c.source_turn_id)?.speaker,
      });
    }
    for (const s of data.sentences) {
      const srcClaim = s.source_claim_ids[0] ? data.claimsById.get(s.source_claim_ids[0]) : undefined;
      if (srcClaim) {
        m.push({
          id: s.sentence_id,
          timeMs: nsToMs(srcClaim.created_ts),
          type: "soap",
          turnId: srcClaim.source_turn_id,
        });
      }
    }
    return m;
  }, [data.turns, data.claimsById, data.sentences]);

  const applyVisualTime = useCallback(
    (timeMs: number) => {
      const pct = durationMs > 0 ? `${(timeMs / durationMs) * 100}%` : "0%";
      waveAreaRef.current?.style.setProperty("--rail-playhead-pct", pct);
    },
    [durationMs],
  );

  const syncPlaybackTime = useCallback(
    (timeMs: number, options?: { forceReact?: boolean; seekTranscript?: boolean }) => {
      const nextTime = clampTime(timeMs, durationMs);
      currentTimeRef.current = nextTime;
      applyVisualTime(nextTime);

      const nextTurnId = activeTranscriptIdAt(nextTime, data.turns);
      if (nextTurnId !== activeTurnRef.current) {
        activeTurnRef.current = nextTurnId;
        onPlaybackTurnChange(nextTurnId);
      }

      const nextSpeakerIds = activeSpeakerIdsAt(nextTime, speakerSegments);
      if (!arraysEqual(nextSpeakerIds, activeSpeakerIdsRef.current)) {
        activeSpeakerIdsRef.current = nextSpeakerIds;
        setActiveSpeakerIds(nextSpeakerIds);
      }

      const coarseBucket = Math.floor(nextTime / 250);
      if (options?.forceReact || coarseBucket !== coarseDisplayBucketRef.current) {
        coarseDisplayBucketRef.current = coarseBucket;
        setCurrentTimeMs(nextTime);
      }

      if (options?.seekTranscript) onSeekToTurn(nextTurnId);
    },
    [applyVisualTime, data.turns, durationMs, onPlaybackTurnChange, onSeekToTurn, speakerSegments],
  );

  useEffect(() => {
    if (!hasData || recordState !== "idle") return;
    setRecordState("recording");
    isPlayingRef.current = true;
    lastPerfRef.current = performance.now();
  }, [hasData, recordState]);

  useEffect(() => {
    syncPlaybackTime(clampTime(currentTimeRef.current, durationMs), { forceReact: true });
  }, [durationMs, syncPlaybackTime]);

  useEffect(() => {
    if (recordState !== "recording" || isScrubbing) {
      isPlayingRef.current = false;
      return;
    }
    isPlayingRef.current = true;
    lastPerfRef.current = performance.now();

    const tick = (now: number) => {
      if (!isPlayingRef.current || isScrubbingRef.current) return;
      const elapsedMs = Math.max(0, now - lastPerfRef.current) * playbackRateRef.current;
      lastPerfRef.current = now;
      const next = clampTime(currentTimeRef.current + elapsedMs, durationMs);
      syncPlaybackTime(next);
      if (next >= durationMs) {
        isPlayingRef.current = false;
        setRecordState("paused");
        syncPlaybackTime(durationMs, { forceReact: true });
        return;
      }
      rafRef.current = requestAnimationFrame(tick);
    };

    rafRef.current = requestAnimationFrame(tick);
    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    };
  }, [durationMs, isScrubbing, recordState, syncPlaybackTime]);

  const timeTicks = useMemo(() => {
    if (expectedDurationNs === 0) return [];
    const totalSec = expectedDurationNs / 1e9;
    const interval = totalSec > 300 ? 60 : totalSec > 120 ? 30 : 15;
    const ticks: Array<{ pct: number; label: string }> = [];
    for (let s = 0; s <= totalSec; s += interval) {
      ticks.push({
        pct: (s / totalSec) * 100,
        label: `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(Math.floor(s % 60)).padStart(2, "0")}`,
      });
    }
    return ticks;
  }, [expectedDurationNs]);

  const seekToTime = useCallback(
    (targetMs: number, options?: { commit?: boolean; preserveSelection?: boolean }) => {
      if (!hasData || durationMs === 0) return;
      const next = clampTime(targetMs, durationMs);
      syncPlaybackTime(next, { forceReact: true, seekTranscript: options?.commit });
      setScrubTimeMs(next);
      if (!options?.preserveSelection) setSelectedTimeMs(next);
    },
    [durationMs, hasData, syncPlaybackTime],
  );

  const timeFromClientX = useCallback(
    (clientX: number) => {
      const rect = svgRef.current?.getBoundingClientRect();
      if (!rect || durationMs === 0) return 0;
      const pct = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
      return pct * durationMs;
    },
    [durationMs],
  );

  const handleTimelinePointerDown = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      if (!hasData) return;
      e.preventDefault();
      e.currentTarget.setPointerCapture(e.pointerId);
      wasPlayingBeforeScrubRef.current = isPlayingRef.current;
      isScrubbingRef.current = true;
      setIsScrubbing(true);
      setRecordState("paused");
      seekToTime(timeFromClientX(e.clientX), { preserveSelection: true });
    },
    [hasData, seekToTime, timeFromClientX],
  );

  const handleTimelinePointerMove = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      if (!isScrubbingRef.current || (e.buttons & 1) !== 1) return;
      seekToTime(timeFromClientX(e.clientX), { preserveSelection: true });
    },
    [seekToTime, timeFromClientX],
  );

  const handleTimelinePointerUp = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      if (!isScrubbingRef.current) return;
      seekToTime(timeFromClientX(e.clientX), { commit: true });
      isScrubbingRef.current = false;
      setIsScrubbing(false);
      if (wasPlayingBeforeScrubRef.current) setRecordState("recording");
    },
    [seekToTime, timeFromClientX],
  );

  const handleTimelinePointerCancel = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      if (!isScrubbingRef.current) return;
      seekToTime(timeFromClientX(e.clientX), { commit: true });
      isScrubbingRef.current = false;
      setIsScrubbing(false);
      if (wasPlayingBeforeScrubRef.current) setRecordState("recording");
    },
    [seekToTime, timeFromClientX],
  );

  const handleRecordToggle = useCallback(() => {
    setRecordState((s) => {
      if (s === "recording") return "paused";
      if (currentTimeRef.current >= durationMs) syncPlaybackTime(0, { forceReact: true, seekTranscript: followTranscript });
      lastPerfRef.current = performance.now();
      return "recording";
    });
  }, [durationMs, followTranscript, syncPlaybackTime]);

  const handleSkip = useCallback(
    (deltaSeconds: number) => {
      seekToTime(currentTimeRef.current + deltaSeconds * 1000, { commit: true });
    },
    [seekToTime],
  );

  const handleFullscreen = useCallback(() => {
    const el = document.querySelector(`[data-surface="rail"]`) as HTMLElement | null;
    if (!el || !document.fullscreenEnabled) return;
    if (document.fullscreenElement) {
      void document.exitFullscreen();
      return;
    }
    void el.requestFullscreen();
  }, []);

  const speedOptions = ["0.5x", "1x", "1.5x"];

  const barCount = 200;
  const barW = svgW / barCount;
  const selectedPct = selectedTimeMs === null || durationMs === 0 ? null : (selectedTimeMs / durationMs) * 100;
  const scrubPct = durationMs > 0 ? (scrubTimeMs / durationMs) * 100 : 0;

  return (
    <footer className={styles.rail} data-surface="rail">
      <div className={styles.railTop}>
      {/* Left: label + time + transport */}
      <div className={styles.railLeft}>
        <span className={styles.railTitle}>
          Conversation Rail <span className={styles.railInfoIcon}>i</span>
        </span>
        <span className={styles.railTimeDisplay}>
          {currentTime} <span className={styles.railTimeTotal}>/ {totalTime}</span>
        </span>
        <div className={styles.railTransport}>
          <button className={styles.railBtn} type="button" aria-label="Jump back 15 seconds" onClick={() => handleSkip(-15)}>
            <span className={styles.railBtnSkipIcon}>◀</span>15
          </button>
          <button className={styles.railBtnPause} type="button" aria-label={isRecording ? "Pause" : "Play"} onClick={handleRecordToggle}>
            {isRecording ? "❚❚" : "▶"}
          </button>
          <button className={styles.railBtn} type="button" aria-label="Jump forward 15 seconds" onClick={() => handleSkip(15)}>
            15<span className={styles.railBtnSkipIcon}>▶</span>
          </button>
        </div>
      </div>

      {/* Center: waveform + markers */}
      <div
        className={styles.railCenter}
        onPointerDown={handleTimelinePointerDown}
        onPointerMove={handleTimelinePointerMove}
        onPointerUp={handleTimelinePointerUp}
        onPointerCancel={handleTimelinePointerCancel}
        onLostPointerCapture={handleTimelinePointerCancel}
      >
      <div
        ref={waveAreaRef}
        className={styles.railWaveArea}
        style={{ "--rail-playhead-pct": `${livePct}%` } as React.CSSProperties}
      >
        <svg
          ref={svgRef}
          className={styles.waveformSvg}
          role="button"
          aria-label="Conversation timeline"
          tabIndex={0}
          viewBox={`0 0 ${svgW} ${WAVE_H}`}
          preserveAspectRatio="none"
          onKeyDown={(e) => {
            if (e.key === "ArrowLeft") {
              e.preventDefault();
              handleSkip(-5);
            }
            if (e.key === "ArrowRight") {
              e.preventDefault();
              handleSkip(5);
            }
            if (e.key.toLowerCase() === "j") {
              e.preventDefault();
              handleSkip(-15);
            }
            if (e.key.toLowerCase() === "l") {
              e.preventDefault();
              handleSkip(15);
            }
            if (e.key === " " || e.key === "Enter") {
              e.preventDefault();
              handleRecordToggle();
            }
          }}
        >
          {waveHeights.map((h, i) => {
            const barPct = (i / barCount) * 100;
            if (barPct > livePct) return null;
            const barH = Math.max(1, h * (WAVE_H * 0.8));
            const x = i * barW;
            const y = (WAVE_H - barH) / 2;
            return (
              <rect
                key={i}
                x={x}
                y={y}
                width={Math.max(1, barW - 1)}
                height={barH}
                fill="#9a969f"
                opacity={0.45}
                rx={0.5}
              />
            );
          })}
        </svg>
        {hasData && (
          <>
            <div className={styles.railPlayhead} />
            {isRecording && (
              <div className={styles.railLiveDot} />
            )}
          </>
        )}
        {selectedPct !== null && (
          <span className={styles.scrubPreview} style={{ left: `${selectedPct}%` }}>
            {fmtTimeMs(selectedTimeMs ?? 0)}
          </span>
        )}
        {isScrubbing && (
          <span className={styles.scrubPreview} style={{ left: `${scrubPct}%` }}>
            {fmtTimeMs(scrubTimeMs)}
          </span>
        )}
      </div>

      <div className={styles.speakerLanes} aria-label="Speaker activity lanes">
        {speakers.map((speaker) => (
          <div
            key={speaker.speakerId}
            className={styles.speakerLane}
            data-speaker-role={speaker.speakerRole}
            aria-label={`${speaker.speakerLabel} activity`}
          >
            <span className={styles.speakerLaneLabel}>{speaker.speakerLabel}</span>
            <div className={styles.speakerLaneTrack}>
              {speakerSegments
                .filter((segment) => segment.speakerId === speaker.speakerId)
                .map((segment) => {
                  const left = durationMs > 0 ? (segment.startMs / durationMs) * 100 : 0;
                  const width = durationMs > 0 ? ((segment.endMs - segment.startMs) / durationMs) * 100 : 0;
                  const active =
                    activeSpeakerIds.includes(speaker.speakerId) &&
                    segment.startMs <= currentTimeMs &&
                    currentTimeMs < segment.endMs;
                  return (
                    <span
                      key={segment.id}
                      className={`${styles.speakerSegment} ${active ? styles.speakerSegmentActive : ""}`}
                      style={{ left: `${left}%`, width: `${Math.max(0.25, width)}%` }}
                      aria-label={`${speaker.speakerLabel} speech segment${segment.estimatedEnd ? ", estimated end" : ""}`}
                    />
                  );
                })}
            </div>
          </div>
        ))}
        <div className={styles.overlapLayer} aria-hidden="true">
          {overlapWindows.map((overlap) => {
            const left = durationMs > 0 ? (overlap.startMs / durationMs) * 100 : 0;
            const width = durationMs > 0 ? ((overlap.endMs - overlap.startMs) / durationMs) * 100 : 0;
            return (
              <span
                key={overlap.id}
                className={styles.overlapIndicator}
                style={{ left: `${left}%`, width: `${Math.max(0.2, width)}%` }}
              />
            );
          })}
        </div>
      </div>

      {/* Markers + time axis (inside railCenter) */}
      <div className={styles.railMarkersArea}>
        <div className={styles.railMarkers}>
          {markers.map((m, i) => {
            const cls =
              m.type === "claim" ? styles.markerDot :
              m.type === "dismissal" ? styles.markerCross :
              m.type === "supersession" ? styles.markerSlash :
              m.type === "soap" ? styles.markerBracket :
              styles.markerShift;
            const content =
              m.type === "supersession" ? "//" :
              m.type === "soap" ? "[ ]" :
              m.type === "reasoning_shift" ? "}" :
              m.type === "dismissal" ? "⊠" : null;
            const markerPct = durationMs > 0 ? (m.timeMs / durationMs) * 100 : 0;
            return (
              <button
                key={`${m.id}-${i}`}
                type="button"
                className={`${styles.railMarker} ${cls}`}
                style={{ left: `${markerPct}%` }}
                aria-label={`Seek to ${m.type.replace("_", " ")} marker at ${fmtTimeMs(m.timeMs)}`}
                onClick={(event) => {
                  event.stopPropagation();
                  seekToTime(m.timeMs, { commit: true });
                }}
              >
                {content}
              </button>
            );
          })}
        </div>
        <div className={styles.railTimeAxis}>
          {timeTicks.map((t) => (
            <span key={t.label} className={styles.railTimeTick} style={{ left: `${t.pct}%` }}>
              {t.label}
            </span>
          ))}
        </div>
      </div>
      </div>{/* close railCenter */}

      {/* Right: speed + fullscreen */}
      <div className={styles.railRight}>
        <div className={styles.railSpeedControls}>
          <button className={styles.railSpeedBtn} type="button" aria-label="Decrease playback speed" onClick={() => setSpeedIndex((i) => Math.max(0, i - 1))}>−</button>
          <span className={styles.railSpeedCurrent} aria-live="polite">{speedOptions[speedIndex]}</span>
          <button className={styles.railSpeedBtn} type="button" aria-label="Increase playback speed" onClick={() => setSpeedIndex((i) => Math.min(speedOptions.length - 1, i + 1))}>+</button>
        </div>
        <button className={styles.railFullscreen} type="button" aria-label="Toggle rail fullscreen" onClick={handleFullscreen}>⛶</button>
      </div>

      </div>{/* close railTop */}

      {/* Legend */}
      <div className={styles.railLegend}>
        <span className={styles.railLegendItem}><span className={styles.legendSpeech} /> speech</span>
        <span className={styles.railLegendItem}><span className={styles.legendDot} /> claim created</span>
        <span className={styles.railLegendItem}><span className={styles.legendCross}>⊠</span> claim dismissed</span>
        <span className={styles.railLegendItem}><span className={styles.legendSlash}>//</span> claim superseded</span>
        <span className={styles.railLegendItem}><span className={styles.legendBracket}>[ ]</span> soap sentence</span>
        <span className={styles.railLegendItem}><span className={styles.legendShift}>{"}"}</span> reasoning shift</span>
      </div>
    </footer>
  );
}

/* ── main canvas ── */

export function ReasoningCanvas() {
  const data = useCanvasData();
  const clearSelection = useSession((s) => s.clearSelection);
  const userSuppressedClaimIds = useSession((s) => s.userSuppressedClaimIds);
  const claimDecisionsById = useSession((s) => s.claimDecisionsById);
  const branchDecisionsById = useSession((s) => s.branchDecisionsById);
  const unsuppressClaim = useSession((s) => s.unsuppressClaim);
  const suppressClaim = useSession((s) => s.suppressClaim);
  const recordClaimDecision = useSession((s) => s.recordClaimDecision);
  const recordBranchDecision = useSession((s) => s.recordBranchDecision);
  const [hoverTarget, setHoverTarget] = useState<HoverTarget>(null);
  const [focusTarget, setFocusTarget] = useState<FocusTarget | null>(null);
  const [activeTurnId, setActiveTurnId] = useState<string | null>(null);
  const [followTranscript, setFollowTranscript] = useState(true);
  const restoreFocusRef = useRef<HTMLElement | null>(null);
  const suppressHoverRef = useRef(false);

  // Fix 5: pulse tracking
  const [pulsingLabels, setPulsingLabels] = useState<Set<string>>(new Set());
  const [pulsingClaimIds, setPulsingClaimIds] = useState<Set<string>>(new Set());
  const prevMatchedLabelsRef = useRef<Set<string>>(new Set());

  const closeFocus = useCallback(() => {
    suppressHoverRef.current = true;
    setFocusTarget(null);
    setHoverTarget(null);
    clearSelection();
  }, [clearSelection]);

  const handleHover = useCallback((target: HoverTarget) => {
    if (suppressHoverRef.current && target) return;
    setHoverTarget(target);
  }, []);

  const handleFocusTarget = useCallback((target: FocusTarget) => {
    restoreFocusRef.current =
      document.activeElement instanceof HTMLElement ? document.activeElement : null;
    suppressHoverRef.current = false;
    setFocusTarget(target);
  }, []);


  const handlePromoteTarget = useCallback((target: FocusTarget, claimIds: string[]) => {
    if (target.type === "hypothesis") {
      recordBranchDecision(target.id, "promoted");
      return;
    }
    for (const claimId of claimIds) {
      unsuppressClaim(claimId);
      recordClaimDecision(claimId, "promoted");
      useSession.getState().setClaimStatus(claimId, "confirmed");
    }
  }, [recordBranchDecision, recordClaimDecision, unsuppressClaim]);

  const handleDemoteTarget = useCallback((target: FocusTarget, claimIds: string[]) => {
    if (target.type === "hypothesis") {
      recordBranchDecision(target.id, "demoted");
      return;
    }
    for (const claimId of claimIds) {
      suppressClaim(claimId);
      recordClaimDecision(claimId, "suppressed");
    }
  }, [recordBranchDecision, recordClaimDecision, suppressClaim]);

  const handlePointerMove = useCallback(() => {
    if (suppressHoverRef.current) suppressHoverRef.current = false;
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") closeFocus();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [closeFocus]);

  const activeTarget = focusTarget ?? hoverTarget;
  const chains = useMemo(() => buildChains(data.edges), [data.edges]);
  const allClaims = useMemo(
    () => [...data.claimsById.values()],
    [data.claimsById],
  );
  const claimTreatments = useMemo(() => {
    const treatments = new Map<string, ClaimTreatment>();
    for (const claim of data.claimsById.values()) {
      treatments.set(claim.claim_id, claimTreatment(claim, userSuppressedClaimIds));
    }
    return treatments;
  }, [data.claimsById, userSuppressedClaimIds]);

  // Fix 2c: patient context matching
  const contextMatches = useMemo(
    () => buildPatientContextMatches(data.claimsById),
    [data.claimsById],
  );

  const restAssoc = useMemo(() => {
    const base = restAssociationIds(data.hypotheses, allClaims);
    // Merge context-matched claim IDs
    for (const id of contextMatches.matchedClaimIds) base.add(id);
    return base;
  }, [data.hypotheses, allClaims, contextMatches.matchedClaimIds]);

  // Fix 5: detect new matches and trigger pulse
  useEffect(() => {
    const prev = prevMatchedLabelsRef.current;
    const curr = contextMatches.matchedLabels;
    const newLabels = new Set<string>();
    for (const label of curr) {
      if (!prev.has(label)) newLabels.add(label);
    }
    if (newLabels.size > 0) {
      setPulsingLabels(newLabels);
      setPulsingClaimIds(new Set(contextMatches.matchedClaimIds));
      const timer = setTimeout(() => {
        setPulsingLabels(new Set());
        setPulsingClaimIds(new Set());
      }, 1000);
      prevMatchedLabelsRef.current = new Set(curr);
      return () => clearTimeout(timer);
    }
    prevMatchedLabelsRef.current = new Set(curr);
  }, [contextMatches]);

  const relatedClaims = useMemo(() => {
    const ids = new Set<string>();
    if (!activeTarget) return ids;
    if (activeTarget.type === "claim") {
      ids.add(activeTarget.id);
      for (const id of chains.get(activeTarget.id) ?? []) ids.add(id);
    }
    if (activeTarget.type === "hypothesis") {
      const h = data.hypotheses.find((x) => x.branch === activeTarget.id);
      if (h) for (const id of relClaimIds(h, allClaims)) ids.add(id);
    }
    if (activeTarget.type === "sentence") {
      const s = data.sentences.find(
        (x) => x.sentence_id === activeTarget.id,
      );
      for (const id of s?.source_claim_ids ?? []) ids.add(id);
    }
    return ids;
  }, [activeTarget, allClaims, chains, data.hypotheses, data.sentences]);

  const relatedSentences = useMemo(() => {
    const ids = new Set<string>();
    if (activeTarget?.type === "claim") {
      for (const s of data.sentences)
        if (s.source_claim_ids.includes(activeTarget.id))
          ids.add(s.sentence_id);
    }
    if (activeTarget?.type === "sentence") ids.add(activeTarget.id);
    return ids;
  }, [activeTarget, data.sentences]);

  const handlePlaybackTurnChange = useCallback((turnId: string | null) => {
    setActiveTurnId(turnId);
  }, []);

  const handleSeekToTurn = useCallback((turnId: string | null) => {
    setActiveTurnId(turnId);
    setFollowTranscript(true);
  }, []);

  const handleManualTranscriptScroll = useCallback(() => {
    setFollowTranscript(false);
  }, []);

  const handleReturnToActiveTranscript = useCallback(() => {
    setFollowTranscript(true);
  }, []);

  return (
    <MotionConfig reducedMotion="user">
      <main
        className={[
          styles.canvas,
          focusTarget ? styles.focusMode : "",
          !focusTarget && hoverTarget ? styles.hoverMode : "",
        ].filter(Boolean).join(" ")}
        onPointerMove={handlePointerMove}
      >
      <PatientStrip state={stateLabel(data)} />

      <TranscriptPanel
        turns={data.turns}
        claimsByTurn={data.claimsByTurn}
        strikeOrigin={data.strikeOrigin}
        hoverTarget={hoverTarget}
        focusTarget={focusTarget}
        activeTurnId={activeTurnId}
        followTranscript={followTranscript}
        relatedClaims={relatedClaims}
        restAssoc={restAssoc}
        onHover={handleHover}
        onFocus={handleFocusTarget}
        onManualScroll={handleManualTranscriptScroll}
        onReturnToActive={handleReturnToActiveTranscript}
      />

      <div className={styles.center}>
        <ClaimsArea
          claimsByPredicate={data.claimsByPredicate}
          claimsById={data.claimsById}
          strikeOrigin={data.strikeOrigin}
          claimTreatments={claimTreatments}
          claimDecisionsById={claimDecisionsById}
          hoverTarget={hoverTarget}
          focusTarget={focusTarget}
          relatedClaims={relatedClaims}
          restAssoc={restAssoc}
          onHover={handleHover}
          onFocus={handleFocusTarget}
        />

        <DifferentialFocal
          hypotheses={data.hypotheses}
          allClaims={allClaims}
          claimsById={data.claimsById}
          claimTreatments={claimTreatments}
          branchDecisionsById={branchDecisionsById}
          claimDecisionsById={claimDecisionsById}
          hoverTarget={hoverTarget}
          focusTarget={focusTarget}
          data={data}
          onHover={handleHover}
          onFocus={handleFocusTarget}
        />

        <NextQuestion />
      </div>

      <ContextPanel
        sentences={data.sentences}
        hoverTarget={hoverTarget}
        focusTarget={focusTarget}
        relatedSentences={relatedSentences}
        restAssoc={restAssoc}
        matchedLabels={contextMatches.matchedLabels}
        pulsingLabels={pulsingLabels}
        pulsingClaimIds={pulsingClaimIds}
        data={data}
        claimTreatments={claimTreatments}
        onHover={handleHover}
        onFocus={handleFocusTarget}
      />

      <ConversationRail
        data={data}
        followTranscript={followTranscript}
        onPlaybackTurnChange={handlePlaybackTurnChange}
        onSeekToTurn={handleSeekToTurn}
      />

      {/* Layer 3 — centered command panel */}
      <AnimatePresence>
        {focusTarget && (
          <FocusOverlay
            target={focusTarget}
            data={data}
            onClose={closeFocus}
            onPromote={handlePromoteTarget}
            onDemote={handleDemoteTarget}
            claimTreatments={claimTreatments}
            restoreFocusTo={restoreFocusRef.current}
          />
        )}
      </AnimatePresence>
      </main>
    </MotionConfig>
  );
}
