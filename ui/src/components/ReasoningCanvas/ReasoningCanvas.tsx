import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useSession } from "@/store/session";
import type {
  BranchScore,
  Claim,
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

interface CanvasData {
  turns: Turn[];
  claimsByTurn: Map<string, Claim[]>;
  claimsById: Map<string, Claim>;
  claimsByPredicate: Map<string, Claim[]>;
  hypotheses: BranchScore[];
  edges: SupersessionEdge[];
  sentences: NoteSentence[];
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

      if (normStatus(c.status) === "active") {
        const pg = claimsByPredicate.get(c.predicate) ?? [];
        pg.push(c);
        claimsByPredicate.set(c.predicate, pg);
      }
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
      steps.push({ label: "reasoning", text: hyp.branch.replaceAll("_", " ") });
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
    steps.push({ label: "reasoning", text: h.branch.replaceAll("_", " ") });
    for (const applied of h.applied) {
      const c = data.claimsById.get(applied.claim_id);
      if (c)
        steps.push({
          label: "evidence",
          text: `${c.predicate.replaceAll("_", " ")}: ${fmtValue(c.value)} (LR ${applied.lr_value.toFixed(1)})`,
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
}: {
  target: FocusTarget;
  data: CanvasData;
  onClose: () => void;
}) {
  const verifier = useSession((s) => s.verifier);
  const cardRef = useRef<HTMLDivElement>(null);
  const steps = buildFocusSteps(target, data, verifier);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  if (steps.length === 0) return null;

  return (
    <>
      <motion.div
        className={styles.focusOverlayBackdrop}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        transition={{ duration: 0.15 }}
        onClick={onClose}
      />
      <motion.div
        ref={cardRef}
        className={styles.focusOverlayCard}
        initial={{ opacity: 0, scale: 0.96 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={{ opacity: 0, scale: 0.96 }}
        transition={{ duration: 0.15, ease: EXPO_OUT }}
      >
        <button
          className={styles.focusOverlayClose}
          type="button"
          aria-label="Close provenance thread"
          onClick={onClose}
        >
          ×
        </button>
        <div className={styles.focusOverlayTitle}>Provenance Thread</div>
        {steps.map((step, i) => (
          <div key={i} className={styles.focusOverlayStep}>
            <span
              className={`${styles.focusOverlayLabel} ${FOCUS_LABEL_CLS[step.label] ?? styles.focusLabelRail}`}
            >
              {step.label}
            </span>
            <span className={styles.focusOverlayText}>{step.text}</span>
          </div>
        ))}
      </motion.div>
    </>
  );
}

/* ── patient strip ── */

function PatientStrip({ state }: { state: string }) {
  return (
    <header className={styles.patientStrip}>
      <span className={styles.brandMark}>RobbyMD</span>
      <span className={styles.patientName}>{DEMO_PATIENT.name}</span>
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
  hoverTarget: HoverTarget;
  focusTarget: FocusTarget | null;
  relatedClaims: Set<string>;
  restAssoc: Set<string>;
  onHover: (t: HoverTarget) => void;
  onFocus: (t: FocusTarget) => void;
}) {
  const {
    turns,
    claimsByTurn,
    hoverTarget,
    focusTarget,
    relatedClaims,
    restAssoc,
    onHover,
    onFocus,
  } = props;
  const endRef = useRef<HTMLDivElement>(null);
  const transcriptRef = useRef<HTMLElement>(null);
  const isNearBottom = useRef(true);

  useEffect(() => {
    const el = transcriptRef.current;
    if (!el) return;
    const onScroll = () => {
      isNearBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
    };
    el.addEventListener("scroll", onScroll, { passive: true });
    return () => el.removeEventListener("scroll", onScroll);
  }, []);

  useEffect(() => {
    if (isNearBottom.current) {
      endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
    }
  }, [turns.length]);

  return (
    <section ref={transcriptRef} className={styles.transcript} data-surface="transcript">
      <div className={styles.transcriptHeader}>Transcript</div>
      <AnimatePresence>
        {turns.map((turn) => {
          const isPhysician = turn.speaker === "physician";
          const turnClaims = claimsByTurn.get(turn.turn_id) ?? [];
          return (
            <motion.div
              className={styles.turn}
              key={turn.turn_id}
              data-turn-id={turn.turn_id}
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.35, ease: EXPO_OUT }}
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
                        style={{ opacity: st === "superseded" ? 0.55 : st === "dismissed" ? 0.45 : undefined }}
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
              {/* Focus overlay renders at canvas root */}
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
  hoverTarget: HoverTarget;
  focusTarget: FocusTarget | null;
  relatedClaims: Set<string>;
  restAssoc: Set<string>;
  onHover: (t: HoverTarget) => void;
  onFocus: (t: FocusTarget) => void;
}) {
  const {
    claimsByPredicate,
    hoverTarget,
    focusTarget,
    relatedClaims,
    restAssoc,
    onHover,
    onFocus,
  } = props;

  if (claimsByPredicate.size === 0) {
    return (
      <div className={styles.claimsArea} data-surface="claims">
        <div className={styles.claimsAreaHeader}>Evidence</div>
        <div className={styles.emptyState}>
          Waiting for clinical evidence...
        </div>
      </div>
    );
  }

  const groups = [...claimsByPredicate.entries()].sort(
    ([, a], [, b]) => b.length - a.length,
  );

  return (
    <div className={styles.claimsArea} data-surface="claims">
      <div className={styles.claimsAreaHeader}>Evidence</div>
      {groups.map(([predicate, claims]) => (
        <div key={predicate} className={styles.claimGroup}>
          <div className={styles.claimGroupLabel}>
            {predicate.replaceAll("_", " ")}
          </div>
          <div className={styles.claimGroupTokens}>
            {claims.map((claim) => {
              const isRel = relatedClaims.has(claim.claim_id);
              const isRest = restAssoc.has(claim.claim_id);
              const isFoc =
                focusTarget?.type === "claim" &&
                focusTarget.id === claim.claim_id;
              const isHov =
                hoverTarget?.type === "claim" &&
                hoverTarget.id === claim.claim_id;
              const cls = [
                normStatus(claim.status) === "superseded"
                  ? styles.claimTokenSuperseded
                  : styles.claimToken,
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
                  {fmtValue(claim.value)}
                </button>
              );
            })}
          </div>
        </div>
      ))}
    </div>
  );
}

/* ── differential focal space (center middle) ── */

function DifferentialFocal(props: {
  hypotheses: BranchScore[];
  allClaims: Claim[];
  claimsById: Map<string, Claim>;
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
    hoverTarget,
    focusTarget,
    onHover,
    onFocus,
  } = props;
  const verifier = useSession((s) => s.verifier);

  if (hypotheses.length === 0) {
    return (
      <div className={styles.differential} data-surface="differential">
        <div className={styles.emptyDifferential}>
          {allClaims.length > 2
            ? "Building differential..."
            : ""}
        </div>
      </div>
    );
  }

  const primary = hypotheses[0]!;
  const secondary = hypotheses.slice(1, 3);
  const tertiary = hypotheses.slice(3);

  const getEvidenceTokens = (h: BranchScore, isPrimary: boolean) => {
    const ids = isPrimary
      ? relClaimIds(h, allClaims)
      : h.applied.map((x) => x.claim_id);
    return ids
      .map((id) => claimsById.get(id))
      .filter(isClaim)
      .slice(0, isPrimary ? 8 : 4)
      .map((c) => ({
        id: c.claim_id,
        text: fmtValue(c.value),
        weakening: c.value.startsWith("negated:") || normStatus(c.status) === "superseded",
      }));
  };

  const primaryTokens = getEvidenceTokens(primary, true);
  const isFocPrimary =
    focusTarget?.type === "hypothesis" && focusTarget.id === primary.branch;
  const isHovPrimary =
    hoverTarget?.type === "hypothesis" && hoverTarget.id === primary.branch;

  return (
    <div className={styles.differential} data-surface="differential">
      {/* Primary hypothesis — center, large */}
      <motion.div
        className={[
          styles.focalPrimary,
          isFocPrimary ? styles.focused : "",
          isHovPrimary ? styles.hovered : "",
          styles.related,
        ]
          .filter(Boolean)
          .join(" ")}
        layout
        transition={{ duration: 0.3, ease: EXPO_OUT }}
        onMouseEnter={() =>
          onHover({ type: "hypothesis", id: primary.branch })
        }
        onMouseLeave={() => onHover(null)}
        onClick={() => onFocus({ type: "hypothesis", id: primary.branch })}
      >
        <span className={styles.focalPrimaryLabel}>Reasoning Focus</span>
        <h2 className={styles.focalPrimaryName}>
          {primary.branch.replaceAll("_", " ")}
        </h2>
        {primaryTokens.length > 0 && (
          <div className={styles.focalPrimaryEvidence}>
            {primaryTokens.map((t) => (
              <motion.button
                key={t.id}
                className={
                  t.weakening
                    ? styles.focalEvidenceWeakening
                    : styles.focalEvidenceToken
                }
                type="button"
                initial={{ opacity: 0, scale: 0.85, y: 4 }}
                animate={{ opacity: 1, scale: 1, y: 0 }}
                transition={{ duration: 0.35, ease: EXPO_OUT }}
                onMouseEnter={() =>
                  onHover({ type: "claim", id: t.id })
                }
                onMouseLeave={() => onHover(null)}
                onClick={(e) => {
                  e.stopPropagation();
                  onFocus({ type: "claim", id: t.id });
                }}
              >
                {t.text}
              </motion.button>
            ))}
          </div>
        )}
        {/* Reasoning narrative */}
        {primaryTokens.length >= 3 && (
          <p className={styles.reasoningNarrative}>
            {primary.branch.replaceAll("_", " ")} origin is the primary consideration based on{" "}
            {primaryTokens
              .filter((t) => !t.weakening)
              .map((t) => t.text)
              .join(", ")}
            {primaryTokens.some((t) => t.weakening)
              ? `. ${primaryTokens.filter((t) => t.weakening).map((t) => t.text).join(", ")} ${primaryTokens.filter((t) => t.weakening).length > 1 ? "are" : "is"} less consistent with this diagnosis.`
              : "."}
          </p>
        )}

        {/* Verifier: why moved */}
        {verifier && verifier.why_moved.length > 0 && (
          <div className={styles.whyMoved}>
            <div className={styles.whyMovedLabel}>Reasoning shift</div>
            {verifier.why_moved.map((w, i) => (
              <div key={i}>{w}</div>
            ))}
          </div>
        )}

        {/* Missing or contradicting */}
        {verifier && verifier.missing_or_contradicting.length > 0 && (
          <p className={styles.missingFeatures}>
            Still needed: {verifier.missing_or_contradicting.join(", ")}
          </p>
        )}

        {/* Focus overlay renders at canvas root */}
      </motion.div>

      {/* Secondary hypotheses */}
      {secondary.length > 0 && (
        <div className={styles.focalSecondaryRow}>
          {secondary.map((h) => {
            const tokens = getEvidenceTokens(h, false);
            const isFoc =
              focusTarget?.type === "hypothesis" &&
              focusTarget.id === h.branch;
            const isHov =
              hoverTarget?.type === "hypothesis" &&
              hoverTarget.id === h.branch;

            return (
              <motion.div
                key={h.branch}
                className={[
                  styles.focalSecondary,
                  isFoc ? styles.focused : "",
                  isHov ? styles.hovered : "",
                ]
                  .filter(Boolean)
                  .join(" ")}
                layout
                transition={{ duration: 0.3, ease: EXPO_OUT }}
                onMouseEnter={() =>
                  onHover({ type: "hypothesis", id: h.branch })
                }
                onMouseLeave={() => onHover(null)}
                onClick={() =>
                  onFocus({ type: "hypothesis", id: h.branch })
                }
              >
                <div className={styles.focalSecondaryName}>
                  {h.branch.replaceAll("_", " ")}
                </div>
                {tokens.length > 0 && (
                  <div className={styles.focalSecondaryEvidence}>
                    {tokens.map((t) => (
                      <span
                        key={t.id}
                        className={styles.focalSecondaryToken}
                      >
                        {t.text}
                      </span>
                    ))}
                  </div>
                )}
                {/* Focus overlay renders at canvas root */}
              </motion.div>
            );
          })}
        </div>
      )}

      {/* Tertiary hypotheses */}
      {tertiary.length > 0 && (
        <div className={styles.focalTertiaryRow}>
          {tertiary.map((h) => (
            <motion.div
              key={h.branch}
              className={styles.focalTertiary}
              layout
              transition={{ duration: 0.3, ease: EXPO_OUT }}
              onMouseEnter={() =>
                onHover({ type: "hypothesis", id: h.branch })
              }
              onMouseLeave={() => onHover(null)}
              onClick={() =>
                onFocus({ type: "hypothesis", id: h.branch })
              }
            >
              <div className={styles.focalTertiaryName}>
                {h.branch.replaceAll("_", " ")}
              </div>
            </motion.div>
          ))}
        </div>
      )}
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
  const rulesOutLabel = sorted[1]?.branch.replaceAll("_", " ") ?? "";

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
    onHover,
    onFocus,
  } = props;

  const [rightTab, setRightTab] = useState<"context" | "documents" | "note">("context");
  const [reviewed, setReviewed] = useState<Set<string>>(new Set());
  const prevSentenceCount = useRef(0);

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
                        className={isReviewed ? styles.soapCheckboxChecked : styles.soapCheckbox}
                        type="button"
                        onClick={() => toggleReviewed(s.sentence_id)}
                      >
                        {isReviewed && <span style={{ color: "#FFFFFF", fontSize: 10, lineHeight: 1 }}>✓</span>}
                      </button>
                      <button
                        className={[
                          styles.soapSentenceText,
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
                          return (
                            <button
                              key={cid}
                              className={styles.soapSourcePill}
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
      <div className={styles.rightTabBar} role="tablist">
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

interface RailEvent {
  pct: number;
  type: "speech" | "claim" | "dismissed" | "superseded" | "soap" | "shift";
  turnId?: string;
}

function ConversationRail({ data, onScrollToTurn }: { data: CanvasData; onScrollToTurn: (turnId: string) => void }) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [recordState, setRecordState] = useState<RecordState>("idle");
  const [selectedPct, setSelectedPct] = useState<number | null>(null);

  const isRecording = recordState === "recording";
  const hasData = data.turns.length > 0;

  useEffect(() => {
    if (data.turns.length > 0 && recordState === "idle")
      setRecordState("recording");
  }, [data.turns.length, recordState]);

  useEffect(() => {
    if (selectedPct === null) return;
    const t = setTimeout(() => setSelectedPct(null), 2000);
    return () => clearTimeout(t);
  }, [selectedPct]);

  const ENCOUNTER_DURATION_NS = 240_000_000_000;

  const currentTimeNs = hasData ? data.turns[data.turns.length - 1]!.ts : 0;
  const expectedDurationNs = Math.max(ENCOUNTER_DURATION_NS, currentTimeNs * 1.1);
  const currentTime = fmtTime(currentTimeNs);
  const totalTime = fmtTime(expectedDurationNs);
  const livePct = expectedDurationNs > 0 ? Math.min(100, (currentTimeNs / expectedDurationNs) * 100) : 0;

  const waveHeights = useMemo(
    () => buildWaveform(data.turns, expectedDurationNs),
    [data.turns, expectedDurationNs],
  );
  const svgW = 960;

  const markers = useMemo<RailEvent[]>(() => {
    if (expectedDurationNs === 0) return [];
    const m: RailEvent[] = [];
    for (const t of data.turns) {
      m.push({ pct: (t.ts / expectedDurationNs) * 100, type: "speech", turnId: t.turn_id });
    }
    for (const c of data.claimsById.values()) {
      const st = normStatus(c.status);
      m.push({
        pct: (c.created_ts / expectedDurationNs) * 100,
        type: st === "superseded" ? "superseded" : st === "dismissed" ? "dismissed" : "claim",
        turnId: c.source_turn_id,
      });
    }
    for (const s of data.sentences) {
      const srcClaim = s.source_claim_ids[0] ? data.claimsById.get(s.source_claim_ids[0]) : undefined;
      if (srcClaim) m.push({ pct: (srcClaim.created_ts / expectedDurationNs) * 100, type: "soap", turnId: srcClaim.source_turn_id });
    }
    return m;
  }, [expectedDurationNs, data.turns, data.claimsById, data.sentences]);

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

  const handleTimelineClick = useCallback(
    (e: React.MouseEvent<SVGSVGElement>) => {
      const rect = svgRef.current?.getBoundingClientRect();
      if (!rect || expectedDurationNs === 0) return;
      const pct = Math.max(0, Math.min(100, ((e.clientX - rect.left) / rect.width) * 100));
      setSelectedPct(pct);
      const targetNs = (pct / 100) * expectedDurationNs;
      const nearest = [...data.turns].reverse().find((t) => t.ts <= targetNs);
      if (nearest) onScrollToTurn(nearest.turn_id);
    },
    [expectedDurationNs, data.turns, onScrollToTurn],
  );

  const handleRecordToggle = useCallback(() => {
    setRecordState((s) => s === "recording" ? "paused" : "recording");
  }, []);

  const barCount = 200;
  const barW = svgW / barCount;

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
          <button className={styles.railBtn} type="button">
            <span className={styles.railBtnSkipIcon}>◀</span>15
          </button>
          <button className={styles.railBtnPause} type="button" aria-label={isRecording ? "Pause" : "Play"} onClick={handleRecordToggle}>
            {isRecording ? "❚❚" : "▶"}
          </button>
          <button className={styles.railBtn} type="button">
            15<span className={styles.railBtnSkipIcon}>▶</span>
          </button>
        </div>
      </div>

      {/* Center: waveform + markers */}
      <div className={styles.railCenter}>
      <div className={styles.railWaveArea}>
        <svg
          ref={svgRef}
          className={styles.waveformSvg}
          viewBox={`0 0 ${svgW} ${WAVE_H}`}
          preserveAspectRatio="none"
          onClick={handleTimelineClick}
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
            <div className={styles.railPlayhead} style={{ left: `${livePct}%` }} />
            {isRecording && (
              <div className={styles.railLiveDot} style={{ left: `${livePct}%` }} />
            )}
          </>
        )}
        {selectedPct !== null && (
          <span className={styles.scrubPreview} style={{ left: `${selectedPct}%` }}>
            {fmtTime((selectedPct / 100) * expectedDurationNs)}
          </span>
        )}
      </div>

      {/* Markers + time axis (inside railCenter) */}
      <div className={styles.railMarkersArea}>
        <div className={styles.railMarkers}>
          {markers.map((m, i) => {
            const cls =
              m.type === "speech" ? styles.markerSpeech :
              m.type === "claim" ? styles.markerDot :
              m.type === "dismissed" ? styles.markerCross :
              m.type === "superseded" ? styles.markerSlash :
              m.type === "soap" ? styles.markerBracket :
              styles.markerShift;
            const content =
              m.type === "superseded" ? "//" :
              m.type === "soap" ? "[ ]" :
              m.type === "shift" ? "}" :
              m.type === "dismissed" ? "⊠" : null;
            return (
              <div
                key={i}
                className={`${styles.railMarker} ${cls}`}
                style={{ left: `${m.pct}%` }}
                onClick={() => { if (m.turnId) onScrollToTurn(m.turnId); }}
              >
                {content}
              </div>
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
          <button className={styles.railSpeedBtn} type="button">−</button>
          <span className={styles.railSpeedCurrent}>1x</span>
          <button className={styles.railSpeedBtn} type="button">+</button>
        </div>
        <button className={styles.railFullscreen} type="button">⛶</button>
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
  const [hoverTarget, setHoverTarget] = useState<HoverTarget>(null);
  const [focusTarget, setFocusTarget] = useState<FocusTarget | null>(null);

  // Fix 5: pulse tracking
  const [pulsingLabels, setPulsingLabels] = useState<Set<string>>(new Set());
  const [pulsingClaimIds, setPulsingClaimIds] = useState<Set<string>>(new Set());
  const prevMatchedLabelsRef = useRef<Set<string>>(new Set());

  const closeFocus = useCallback(() => {
    setFocusTarget(null);
    setHoverTarget(null);
    clearSelection();
  }, [clearSelection]);

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

  const handleScrollToTurn = useCallback((turnId: string) => {
    const el = document.querySelector(`[data-turn-id="${turnId}"]`);
    if (el) el.scrollIntoView({ behavior: "smooth", block: "center" });
  }, []);

  return (
    <main
      className={[styles.canvas, activeTarget ? styles.revealing : ""]
        .filter(Boolean)
        .join(" ")}
    >
      <PatientStrip state={stateLabel(data)} />

      <TranscriptPanel
        turns={data.turns}
        claimsByTurn={data.claimsByTurn}
        hoverTarget={hoverTarget}
        focusTarget={focusTarget}
        relatedClaims={relatedClaims}
        restAssoc={restAssoc}
        onHover={setHoverTarget}
        onFocus={setFocusTarget}
      />

      <div className={styles.center}>
        <ClaimsArea
          claimsByPredicate={data.claimsByPredicate}
          claimsById={data.claimsById}
          hoverTarget={hoverTarget}
          focusTarget={focusTarget}
          relatedClaims={relatedClaims}
          restAssoc={restAssoc}
          onHover={setHoverTarget}
          onFocus={setFocusTarget}
        />

        <DifferentialFocal
          hypotheses={data.hypotheses}
          allClaims={allClaims}
          claimsById={data.claimsById}
          hoverTarget={hoverTarget}
          focusTarget={focusTarget}
          data={data}
          onHover={setHoverTarget}
          onFocus={setFocusTarget}
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
        onHover={setHoverTarget}
        onFocus={setFocusTarget}
      />

      <ConversationRail data={data} onScrollToTurn={handleScrollToTurn} />

      {/* Fix 4: Single focus overlay at canvas root */}
      <AnimatePresence>
        {focusTarget && (
          <FocusOverlay
            target={focusTarget}
            data={data}
            onClose={closeFocus}
          />
        )}
      </AnimatePresence>
    </main>
  );
}
