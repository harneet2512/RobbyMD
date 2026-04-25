import { useEffect, useState, useCallback } from "react";
import { useEncounterStore, type FragmentState } from "@/store/encounter";
import { useTraceStore } from "@/store/trace";
import styles from "./Fragment.module.css";

export function Fragment() {
  const fragment = useEncounterStore((s) => s.activeFragment);
  const closeFragment = useEncounterStore((s) => s.closeFragment);
  const claims = useEncounterStore((s) => s.claims);
  const hypotheses = useEncounterStore((s) => s.hypotheses);
  const entries = useTraceStore((s) => s.entries);
  const [isOpen, setIsOpen] = useState(false);
  const [isClosing, setIsClosing] = useState(false);

  useEffect(() => {
    if (fragment) {
      requestAnimationFrame(() => setIsOpen(true));
      setIsClosing(false);
    } else {
      setIsOpen(false);
    }
  }, [fragment]);

  const handleClose = useCallback(() => {
    setIsClosing(true);
    setTimeout(() => {
      closeFragment();
      setIsClosing(false);
      setIsOpen(false);
    }, 120);
  }, [closeFragment]);

  if (!fragment && !isClosing) return null;

  const f = fragment as FragmentState;
  const overlayClass = [
    styles.overlay,
    isOpen && !isClosing && styles.open,
    isClosing && styles.closing,
  ]
    .filter(Boolean)
    .join(" ");

  const posStyle = {
    left: Math.min(f.anchorX, window.innerWidth - 440),
    top: Math.min(f.anchorY + 8, window.innerHeight - 300),
  };

  if (f.kind === "claim") {
    const claim = claims.get(f.id);
    const traceEntry = entries.find((e) => e.claimIds.includes(f.id));

    return (
      <>
        <div className={styles.backdrop} onClick={handleClose} />
        <div className={overlayClass} style={posStyle}>
          <div className={styles.heading}>provenance</div>
          {traceEntry?.sourceTurnText && (
            <div className={styles.source}>
              <span className={styles.speaker}>
                {traceEntry.sourceTurnSpeaker}:{" "}
              </span>
              &ldquo;{traceEntry.sourceTurnText}&rdquo;
            </div>
          )}
          {claim && (
            <div className={styles.section}>
              <div className={styles.sectionLabel}>claim</div>
              <div className={styles.item}>
                {claim.predicate} · {claim.value}
              </div>
              <div className={styles.item} style={{ color: "var(--text-tertiary)" }}>
                confidence: {(claim.confidence * 100).toFixed(0)}%
              </div>
            </div>
          )}
        </div>
      </>
    );
  }

  if (f.kind === "hypothesis") {
    const hyp = hypotheses.find((h) => h.branch === f.id);
    const branchClaims = Array.from(claims.values()).filter(
      (c) => c.status === "active"
    );

    return (
      <>
        <div className={styles.backdrop} onClick={handleClose} />
        <div className={overlayClass} style={posStyle}>
          <div className={styles.heading}>
            {hyp?.label ?? f.id}
          </div>
          {hyp && (
            <div className={styles.section}>
              <div className={styles.sectionLabel}>score</div>
              <div className={styles.item}>
                {hyp.claim_count} evidence items · posterior{" "}
                {(hyp.posterior * 100).toFixed(1)}%
              </div>
            </div>
          )}
          <div className={styles.section}>
            <div className={styles.sectionLabel}>evidence</div>
            {branchClaims.slice(0, 8).map((c) => (
              <div key={c.claim_id} className={styles.item}>
                {c.predicate} · {c.value}
              </div>
            ))}
            {branchClaims.length === 0 && (
              <div className={styles.item}>no evidence yet</div>
            )}
          </div>
        </div>
      </>
    );
  }

  return null;
}
