import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useSession } from "@/store/session";
import type { BranchScore, Claim, NoteSentence, SupersessionEdge, Turn } from "@/types/substrate";
import styles from "./ReasoningCanvas.module.css";

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
  hypotheses: BranchScore[];
  edges: SupersessionEdge[];
  sentences: NoteSentence[];
}

interface MotionState {
  newestClaimId: string | null;
  newestSentenceId: string | null;
  rankingRevision: number;
}

interface WebLine {
  d: string;
  claimId: string;
  branch: string;
}

const statusRank: Record<Claim["status"], number> = {
  active: 0,
  confirmed: 1,
  superseded: 2,
  dismissed: 3,
};

function isClaim(claim: Claim | undefined): claim is Claim {
  return claim !== undefined;
}

function isTurn(turn: Turn | undefined): turn is Turn {
  return turn !== undefined;
}

function isSentence(sentence: NoteSentence | undefined): sentence is NoteSentence {
  return sentence !== undefined;
}

function formatTime(ns: number): string {
  const totalSeconds = Math.max(0, Math.floor(ns / 1_000_000_000));
  return `${String(Math.floor(totalSeconds / 60)).padStart(2, "0")}:${String(totalSeconds % 60).padStart(2, "0")}`;
}

function formatValue(value: string): string {
  return value.replace(/^negated:/, "denies ").replaceAll("_", " ");
}

function claimText(claim: Claim): string {
  return `${claim.predicate.replaceAll("_", " ")}: ${formatValue(claim.value)}`;
}

function normalizeStatus(status: Claim["status"]): "active" | "superseded" | "dismissed" {
  if (status === "superseded") return "superseded";
  if (status === "dismissed") return "dismissed";
  return "active";
}

function buildChains(edges: SupersessionEdge[]): Map<string, string[]> {
  const next = new Map<string, string>();
  const previous = new Map<string, string>();
  for (const edge of edges) {
    next.set(edge.old_claim_id, edge.new_claim_id);
    previous.set(edge.new_claim_id, edge.old_claim_id);
  }

  const chains = new Map<string, string[]>();
  for (const id of new Set([...next.keys(), ...previous.keys()])) {
    let head = id;
    while (previous.has(head)) head = previous.get(head)!;
    const chain = [head];
    let current = head;
    while (next.has(current)) {
      current = next.get(current)!;
      chain.push(current);
    }
    for (const chainId of chain) chains.set(chainId, chain);
  }
  return chains;
}

function relatedClaimIdsForHypothesis(hypothesis: BranchScore, claims: Claim[]): string[] {
  const applied = hypothesis.applied.map((entry) => entry.claim_id);
  if (applied.length > 0) return applied;
  return claims
    .filter((claim) => normalizeStatus(claim.status) === "active")
    .slice(0, 6)
    .map((claim) => claim.claim_id);
}

function topHypotheses(hypotheses: BranchScore[], count: number): BranchScore[] {
  return [...hypotheses].sort((a, b) => b.log_score - a.log_score).slice(0, count);
}

function hypothesisPosition(index: number, total: number): { x: number; y: number } {
  if (total <= 1) return { x: 50, y: 45 };
  if (total === 2) return index === 0 ? { x: 48, y: 43 } : { x: 63, y: 56 };
  if (total === 3) return [{ x: 50, y: 43 }, { x: 67, y: 30 }, { x: 36, y: 61 }][index]!;
  if (total <= 5) {
    return [
      { x: 49, y: 42 },
      { x: 61, y: 53 },
      { x: 30, y: 30 },
      { x: 76, y: 26 },
      { x: 28, y: 72 },
    ][index]!;
  }
  const primary = [{ x: 49, y: 41 }, { x: 63, y: 51 }, { x: 38, y: 58 }];
  if (index < primary.length) return primary[index]!;
  const perimeter = [
    { x: 18, y: 24 },
    { x: 78, y: 20 },
    { x: 86, y: 45 },
    { x: 70, y: 74 },
    { x: 42, y: 78 },
    { x: 20, y: 58 },
    { x: 56, y: 22 },
    { x: 88, y: 68 },
    { x: 28, y: 82 },
  ];
  return perimeter[(index - primary.length) % perimeter.length]!;
}

function visualStrength(index: number, total: number, allWeak: boolean): { scale: number; opacity: number } {
  if (allWeak) return { scale: index < 3 ? 0.94 : 0.82, opacity: index < 3 ? 0.55 : 0.34 };
  if (index === 0) return { scale: total === 1 ? 1.08 : 1.14, opacity: 1 };
  if (index === 1) return { scale: 1.02, opacity: 0.76 };
  if (index === 2) return { scale: 0.96, opacity: 0.62 };
  return { scale: 0.86, opacity: total > 8 ? 0.38 : 0.48 };
}

function useCanvasData(): CanvasData {
  const turnsRecord = useSession((s) => s.turns);
  const turnOrder = useSession((s) => s.turnOrder);
  const claimsRecord = useSession((s) => s.claims);
  const claimOrder = useSession((s) => s.claimOrder);
  const ranking = useSession((s) => s.ranking);
  const edges = useSession((s) => s.edges);
  const sentencesRecord = useSession((s) => s.noteSentences);
  const noteOrder = useSession((s) => s.noteOrder);

  return useMemo(() => {
    const turns = turnOrder.map((id) => turnsRecord[id]).filter(isTurn);
    const claims = claimOrder
      .map((id) => claimsRecord[id])
      .filter(isClaim)
      .sort((a, b) => statusRank[a.status] - statusRank[b.status] || a.created_ts - b.created_ts);
    const claimsByTurn = new Map<string, Claim[]>();
    const claimsById = new Map<string, Claim>();
    for (const claim of claims) {
      claimsById.set(claim.claim_id, claim);
      const group = claimsByTurn.get(claim.source_turn_id) ?? [];
      group.push(claim);
      claimsByTurn.set(claim.source_turn_id, group);
    }
    return {
      turns,
      claimsByTurn,
      claimsById,
      hypotheses: [...(ranking?.scores ?? [])].sort((a, b) => b.log_score - a.log_score),
      edges,
      sentences: noteOrder.map((id) => sentencesRecord[id]).filter(isSentence),
    };
  }, [claimOrder, claimsRecord, edges, noteOrder, ranking?.scores, sentencesRecord, turnOrder, turnsRecord]);
}

function useMotionState(data: CanvasData): MotionState {
  const latestClaim = useMemo(() => {
    return [...data.claimsById.values()].sort((a, b) => b.created_ts - a.created_ts)[0]?.claim_id ?? null;
  }, [data.claimsById]);
  const latestSentence = data.sentences[data.sentences.length - 1]?.sentence_id ?? null;
  const rankingSignature = data.hypotheses.map((h) => `${h.branch}:${h.log_score}`).join("|");
  const [revision, setRevision] = useState(0);

  useEffect(() => {
    if (rankingSignature) setRevision((value) => value + 1);
  }, [rankingSignature]);

  return {
    newestClaimId: latestClaim,
    newestSentenceId: latestSentence,
    rankingRevision: revision,
  };
}

/* ── provenance web: SVG curves from claims to hypothesis nodes ── */

function ProvenanceWeb({
  hypotheses,
  hoverTarget,
}: {
  hypotheses: BranchScore[];
  hoverTarget: HoverTarget;
}) {
  const [paths, setPaths] = useState<WebLine[]>([]);
  const rafRef = useRef(0);

  const compute = useCallback(() => {
    cancelAnimationFrame(rafRef.current);
    rafRef.current = requestAnimationFrame(() => {
      const result: WebLine[] = [];

      for (const h of hypotheses) {
        const hEl = document.querySelector(`[data-hypothesis="${h.branch}"]`) as HTMLElement | null;
        if (!hEl) continue;
        const hRect = hEl.getBoundingClientRect();
        const hx = hRect.left;
        const hy = hRect.top + hRect.height / 2;

        for (const a of h.applied) {
          const cEl = document.querySelector(`[data-claim-id="${a.claim_id}"]`) as HTMLElement | null;
          if (!cEl) continue;
          const cRect = cEl.getBoundingClientRect();

          const conv = cEl.closest("[data-zone='conversation']");
          if (conv) {
            const convRect = conv.getBoundingClientRect();
            if (cRect.bottom < convRect.top || cRect.top > convRect.bottom) continue;
          }

          const cx = cRect.right + 4;
          const cy = cRect.top + cRect.height / 2;
          const midX = (cx + hx) / 2;
          const d = `M ${cx},${cy} C ${midX},${cy} ${midX},${hy} ${hx},${hy}`;
          result.push({ d, claimId: a.claim_id, branch: h.branch });
        }
      }

      setPaths(result);
    });
  }, [hypotheses]);

  useEffect(() => {
    compute();

    const ro = new ResizeObserver(compute);
    ro.observe(document.documentElement);

    const conv = document.querySelector("[data-zone='conversation']");
    conv?.addEventListener("scroll", compute, { passive: true });

    const interval = setInterval(compute, 600);

    return () => {
      cancelAnimationFrame(rafRef.current);
      ro.disconnect();
      conv?.removeEventListener("scroll", compute);
      clearInterval(interval);
    };
  }, [compute]);

  const activeHypothesis = hoverTarget?.type === "hypothesis" ? hoverTarget.id : null;
  const activeClaim = hoverTarget?.type === "claim" ? hoverTarget.id : null;

  return (
    <svg className={styles.provenanceWeb} aria-hidden="true">
      {paths.map((p) => {
        const isActive = p.branch === activeHypothesis || p.claimId === activeClaim;
        return (
          <path
            key={`${p.claimId}-${p.branch}`}
            d={p.d}
            className={[styles.webPath, isActive ? styles.webPathActive : ""].join(" ")}
          />
        );
      })}
    </svg>
  );
}

/* ── sub-components ── */

function DisclaimerBar() {
  return (
    <div className={styles.disclaimerBar} role="status">
      not a medical device — research prototype — not for clinical use
    </div>
  );
}

function SystemLine({ hasData }: { hasData: boolean }) {
  return (
    <div className={styles.systemLine} aria-label="RobbyMD system state">
      <span className={styles.systemName}>RobbyMD</span>
      <span>clinical reasoning flight recorder</span>
      <span className={styles.systemState}>{hasData ? "truth state stabilizing" : "neutral recording state"}</span>
    </div>
  );
}

function ConversationStream(props: {
  turns: Turn[];
  claimsByTurn: Map<string, Claim[]>;
  hoverTarget: HoverTarget;
  focusTarget: FocusTarget | null;
  relatedClaims: Set<string>;
  newestClaimId: string | null;
  onHover: (target: HoverTarget) => void;
  onFocus: (target: FocusTarget) => void;
}) {
  const { turns, claimsByTurn, hoverTarget, focusTarget, relatedClaims, newestClaimId, onHover, onFocus } = props;
  return (
    <section className={styles.conversation} data-zone="conversation" aria-label="Conversation with inline claims">
      {turns.length === 0 ? <div className={styles.emptyState}>awaiting claims</div> : null}
      {turns.map((turn) => (
        <div className={styles.turn} key={turn.turn_id}>
          <div className={styles.turnMeta}>
            <span>{formatTime(turn.ts)}</span>
            <span className={styles.speaker}>{turn.speaker}</span>
          </div>
          <p className={turn.speaker === "physician" ? styles.physicianText : styles.patientText}>{turn.text}</p>
          <div className={styles.claims}>
            {(claimsByTurn.get(turn.turn_id) ?? []).map((claim) => {
              const status = normalizeStatus(claim.status);
              const isFocused = focusTarget?.type === "claim" && focusTarget.id === claim.claim_id;
              const isHovered = hoverTarget?.type === "claim" && hoverTarget.id === claim.claim_id;
              return (
                <button
                  className={[
                    styles.claim,
                    styles[`claim_${status}`],
                    isFocused ? styles.focused : "",
                    isHovered ? styles.hovered : "",
                    relatedClaims.has(claim.claim_id) ? styles.related : "",
                    newestClaimId === claim.claim_id ? styles.justRecorded : "",
                  ].join(" ")}
                  key={claim.claim_id}
                  data-claim-id={claim.claim_id}
                  type="button"
                  aria-label={`${status} claim ${claimText(claim)}`}
                  onMouseEnter={() => onHover({ type: "claim", id: claim.claim_id })}
                  onMouseLeave={() => onHover(null)}
                  onFocus={() => onHover({ type: "claim", id: claim.claim_id })}
                  onBlur={() => onHover(null)}
                  onClick={() => onFocus({ type: "claim", id: claim.claim_id })}
                >
                  <span className={styles.claimPredicate}>{claim.predicate.replaceAll("_", " ")}</span>
                  <span>{formatValue(claim.value)}</span>
                </button>
              );
            })}
          </div>
        </div>
      ))}
    </section>
  );
}

function HypothesisField(props: {
  hypotheses: BranchScore[];
  allClaims: Claim[];
  hoverTarget: HoverTarget;
  focusTarget: FocusTarget | null;
  rankingRevision: number;
  onHover: (target: HoverTarget) => void;
  onFocus: (target: FocusTarget) => void;
}) {
  const { hypotheses, allClaims, hoverTarget, focusTarget, rankingRevision, onHover, onFocus } = props;
  const allWeak = hypotheses.length > 1 && hypotheses.every((h) => h.log_score < 1);
  return (
    <section className={styles.hypotheses} aria-label="Dynamic hypothesis field">
      {hypotheses.length === 0 ? <div className={styles.emptyHypotheses}>no dominant hypothesis</div> : null}
      {hypotheses.map((hypothesis, index) => {
        const pos = hypothesisPosition(index, hypotheses.length);
        const strength = visualStrength(index, hypotheses.length, allWeak);
        const related = relatedClaimIdsForHypothesis(hypothesis, allClaims);
        const pct = Math.round(hypothesis.posterior * 100);
        return (
          <button
            className={[
              styles.hypothesisNode,
              focusTarget?.type === "hypothesis" && focusTarget.id === hypothesis.branch ? styles.focused : "",
              hoverTarget?.type === "hypothesis" && hoverTarget.id === hypothesis.branch ? styles.hovered : "",
            ].join(" ")}
            key={hypothesis.branch}
            data-hypothesis={hypothesis.branch}
            data-revision={rankingRevision}
            style={{
              left: `${pos.x}%`,
              top: `${pos.y}%`,
              opacity: strength.opacity,
              transform: `translate(-50%, -50%) scale(${strength.scale})`,
            }}
            type="button"
            aria-label={`Hypothesis ${hypothesis.branch}, ${pct}% posterior, ${related.length} claims`}
            onMouseEnter={() => onHover({ type: "hypothesis", id: hypothesis.branch })}
            onMouseLeave={() => onHover(null)}
            onFocus={() => onHover({ type: "hypothesis", id: hypothesis.branch })}
            onBlur={() => onHover(null)}
            onClick={() => onFocus({ type: "hypothesis", id: hypothesis.branch })}
          >
            <span>{hypothesis.branch.replaceAll("_", " ")}</span>
            <span className={styles.posteriorBar}>
              <span className={styles.posteriorFill} style={{ width: `${pct}%` }} />
            </span>
            <span className={styles.evidenceCount}>{pct}% · {related.length} claims</span>
          </button>
        );
      })}
    </section>
  );
}

function VerifierStrip({
  hypotheses,
  onHover,
  onFocus,
}: {
  hypotheses: BranchScore[];
  onHover: (target: HoverTarget) => void;
  onFocus: (target: FocusTarget) => void;
}) {
  const verifier = useSession((s) => s.verifier);
  if (!verifier?.next_best_question) return null;
  const competitors = topHypotheses(hypotheses, 2);
  const confirms = competitors[0]?.branch ?? "leading branch";
  const rulesOut = competitors[1]?.branch ?? "nearest alternative";
  return (
    <button
      className={styles.verifier}
      type="button"
      aria-label={`Verifier question. Confirms ${confirms}. Rules out ${rulesOut}.`}
      onMouseEnter={() => onHover({ type: "verifier", id: "current" })}
      onMouseLeave={() => onHover(null)}
      onFocus={() => onHover({ type: "verifier", id: "current" })}
      onBlur={() => onHover(null)}
      onClick={() => onFocus({ type: "verifier", id: "current" })}
    >
      <span className={styles.verifierTag}>next disambiguation</span>
      <span className={styles.verifierQuestion}>{verifier.next_best_question}</span>
      <span className={styles.verifierLogic}>
        <span>confirms {confirms.replaceAll("_", " ")}</span>
        <span>rules out {rulesOut.replaceAll("_", " ")}</span>
      </span>
    </button>
  );
}

function SoapZone(props: {
  sentences: NoteSentence[];
  hoverTarget: HoverTarget;
  focusTarget: FocusTarget | null;
  relatedSentences: Set<string>;
  newestSentenceId: string | null;
  onHover: (target: HoverTarget) => void;
  onFocus: (target: FocusTarget) => void;
}) {
  const { sentences, hoverTarget, focusTarget, relatedSentences, newestSentenceId, onHover, onFocus } = props;
  return (
    <section className={styles.soap} aria-label="SOAP note with sentence provenance">
      {sentences.length === 0 ? <div className={styles.emptyState}>SOAP will emerge from claims</div> : null}
      {sentences.map((sentence) => (
        <button
          className={[
            styles.sentence,
            focusTarget?.type === "sentence" && focusTarget.id === sentence.sentence_id ? styles.focused : "",
            hoverTarget?.type === "sentence" && hoverTarget.id === sentence.sentence_id ? styles.hovered : "",
            relatedSentences.has(sentence.sentence_id) ? styles.related : "",
            newestSentenceId === sentence.sentence_id ? styles.justCompiled : "",
          ].join(" ")}
          key={sentence.sentence_id}
          type="button"
          aria-label={`SOAP ${sentence.section} sentence with ${sentence.source_claim_ids.length} source claims`}
          onMouseEnter={() => onHover({ type: "sentence", id: sentence.sentence_id })}
          onMouseLeave={() => onHover(null)}
          onFocus={() => onHover({ type: "sentence", id: sentence.sentence_id })}
          onBlur={() => onHover(null)}
          onClick={() => onFocus({ type: "sentence", id: sentence.sentence_id })}
        >
          <span className={styles.provenanceMark} aria-hidden="true" />
          <span className={styles.sentenceSection}>{sentence.section}</span>
          <span>{sentence.text}</span>
        </button>
      ))}
    </section>
  );
}

function FocusInspection({ target, data, onClose }: { target: FocusTarget | null; data: CanvasData; onClose: () => void }) {
  const verifier = useSession((s) => s.verifier);
  const chains = useMemo(() => buildChains(data.edges), [data.edges]);
  if (!target) return null;

  let title = "";
  let body: JSX.Element | null = null;

  if (target.type === "claim") {
    const claim = data.claimsById.get(target.id);
    if (!claim) return null;
    const sourceTurn = data.turns.find((turn) => turn.turn_id === claim.source_turn_id);
    const chain = chains.get(claim.claim_id) ?? [claim.claim_id];
    const noteUse = data.sentences.filter((sentence) => sentence.source_claim_ids.includes(claim.claim_id));
    title = "claim inspection";
    body = (
      <>
        <div className={styles.inspectionLine}>{claimText(claim)}</div>
        {sourceTurn ? <div className={styles.inspectionMuted}>{sourceTurn.speaker}: {sourceTurn.text}</div> : null}
        <div className={styles.inspectionGroup}>
          <span>lineage</span>
          <div className={styles.chain}>
            {chain.map((id, index) => (
              <span key={id}>{index > 0 ? <span className={styles.chainArrow}>-&gt;</span> : null}{id}</span>
            ))}
          </div>
        </div>
        <div className={styles.inspectionGroup}>
          <span>SOAP</span>
          <span>{noteUse.length > 0 ? noteUse.map((s) => s.sentence_id).join(", ") : "not yet compiled"}</span>
        </div>
      </>
    );
  } else if (target.type === "hypothesis") {
    const hypothesis = data.hypotheses.find((h) => h.branch === target.id);
    if (!hypothesis) return null;
    const claimIds = relatedClaimIdsForHypothesis(hypothesis, [...data.claimsById.values()]);
    title = "hypothesis evidence";
    body = (
      <>
        <div className={styles.inspectionLine}>{hypothesis.branch.replaceAll("_", " ")}</div>
        <div className={styles.inspectionGroup}>
          <span>posterior</span>
          <span>{Math.round(hypothesis.posterior * 100)}%</span>
        </div>
        <div className={styles.inspectionGroup}>
          <span>claims</span>
          <span>{claimIds.map((id) => data.claimsById.get(id)).filter(Boolean).map((claim) => claimText(claim!)).join(" | ")}</span>
        </div>
      </>
    );
  } else if (target.type === "sentence") {
    const sentence = data.sentences.find((s) => s.sentence_id === target.id);
    if (!sentence) return null;
    title = "SOAP provenance";
    body = (
      <>
        <div className={styles.inspectionLine}>{sentence.text}</div>
        <div className={styles.inspectionGroup}>
          <span>claims</span>
          <span>{sentence.source_claim_ids.join(", ")}</span>
        </div>
        <div className={styles.inspectionGroup}>
          <span>turns</span>
          <span>{sentence.source_claim_ids.map((id) => data.claimsById.get(id)?.source_turn_id).filter(Boolean).join(", ")}</span>
        </div>
      </>
    );
  } else {
    const competitors = topHypotheses(data.hypotheses, 2);
    title = "verifier logic";
    body = (
      <>
        <div className={styles.inspectionLine}>{verifier?.next_best_question ?? "No verifier question active"}</div>
        <div className={styles.inspectionGroup}>
          <span>confirms</span>
          <span>{competitors[0]?.branch.replaceAll("_", " ") ?? "leading branch"}</span>
        </div>
        <div className={styles.inspectionGroup}>
          <span>rules out</span>
          <span>{competitors[1]?.branch.replaceAll("_", " ") ?? "nearest alternative"}</span>
        </div>
      </>
    );
  }

  return (
    <aside className={styles.focusInspection} aria-label={title}>
      <div className={styles.inspectionHeader}>
        <span>{title}</span>
        <button className={styles.closeInspection} type="button" onClick={onClose} aria-label="Close inspection">x</button>
      </div>
      {body}
    </aside>
  );
}

export function ReasoningCanvas() {
  const data = useCanvasData();
  const clearSelection = useSession((s) => s.clearSelection);
  const [hoverTarget, setHoverTarget] = useState<HoverTarget>(null);
  const [focusTarget, setFocusTarget] = useState<FocusTarget | null>(null);
  const motion = useMotionState(data);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setFocusTarget(null);
        setHoverTarget(null);
        clearSelection();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [clearSelection]);

  const activeTarget = focusTarget ?? hoverTarget;
  const chains = useMemo(() => buildChains(data.edges), [data.edges]);
  const allClaims = useMemo(() => [...data.claimsById.values()], [data.claimsById]);

  const relatedClaims = useMemo(() => {
    const ids = new Set<string>();
    if (!activeTarget) return ids;
    if (activeTarget.type === "claim") {
      ids.add(activeTarget.id);
      for (const id of chains.get(activeTarget.id) ?? []) ids.add(id);
    }
    if (activeTarget.type === "hypothesis") {
      const hypothesis = data.hypotheses.find((h) => h.branch === activeTarget.id);
      if (hypothesis) for (const id of relatedClaimIdsForHypothesis(hypothesis, allClaims)) ids.add(id);
    }
    if (activeTarget.type === "sentence") {
      const sentence = data.sentences.find((s) => s.sentence_id === activeTarget.id);
      for (const id of sentence?.source_claim_ids ?? []) ids.add(id);
    }
    return ids;
  }, [activeTarget, allClaims, chains, data.hypotheses, data.sentences]);

  const relatedSentences = useMemo(() => {
    const ids = new Set<string>();
    if (activeTarget?.type === "claim") {
      for (const sentence of data.sentences) if (sentence.source_claim_ids.includes(activeTarget.id)) ids.add(sentence.sentence_id);
    }
    if (activeTarget?.type === "sentence") ids.add(activeTarget.id);
    return ids;
  }, [activeTarget, data.sentences]);

  return (
    <main className={[styles.canvas, activeTarget ? styles.revealing : ""].join(" ")}>
      <DisclaimerBar />
      <SystemLine hasData={data.turns.length > 0 || data.hypotheses.length > 0} />
      <ProvenanceWeb hypotheses={data.hypotheses} hoverTarget={hoverTarget} />
      <ConversationStream
        turns={data.turns}
        claimsByTurn={data.claimsByTurn}
        hoverTarget={hoverTarget}
        focusTarget={focusTarget}
        relatedClaims={relatedClaims}
        newestClaimId={motion.newestClaimId}
        onHover={setHoverTarget}
        onFocus={setFocusTarget}
      />
      <HypothesisField
        hypotheses={data.hypotheses}
        allClaims={allClaims}
        hoverTarget={hoverTarget}
        focusTarget={focusTarget}
        rankingRevision={motion.rankingRevision}
        onHover={setHoverTarget}
        onFocus={setFocusTarget}
      />
      <VerifierStrip hypotheses={data.hypotheses} onHover={setHoverTarget} onFocus={setFocusTarget} />
      <SoapZone
        sentences={data.sentences}
        hoverTarget={hoverTarget}
        focusTarget={focusTarget}
        relatedSentences={relatedSentences}
        newestSentenceId={motion.newestSentenceId}
        onHover={setHoverTarget}
        onFocus={setFocusTarget}
      />
      <FocusInspection target={focusTarget} data={data} onClose={() => setFocusTarget(null)} />
    </main>
  );
}
