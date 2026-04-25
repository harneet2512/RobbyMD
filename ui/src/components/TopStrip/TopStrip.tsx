import { useEncounterStore } from "@/store/encounter";
import { useEncounterClock } from "@/hooks/useEncounterClock";
import { directionGlyph } from "@/lib/glyphs";
import { formatTimestamp } from "@/lib/format";
import styles from "./TopStrip.module.css";

export function TopStrip() {
  const session = useEncounterStore((s) => s.session);
  const hypotheses = useEncounterStore((s) => s.hypotheses);
  const previousHypotheses = useEncounterStore((s) => s.previousHypotheses);
  const openFragment = useEncounterStore((s) => s.openFragment);
  const elapsed = useEncounterClock(session?.started_at_ns ?? null);

  if (!session) {
    return (
      <div className={styles.strip}>
        <span className={styles.patient}>connecting...</span>
        <span className={styles.clock}>--:--</span>
      </div>
    );
  }

  const startTime = formatTimestamp(session.started_at_ns);
  const prevMap = new Map(previousHypotheses.map((h) => [h.branch, h]));

  return (
    <div className={styles.strip}>
      <span className={styles.patient}>
        {session.patient_name} · {session.patient_age}
        {session.patient_sex} · {startTime}
      </span>

      <div className={styles.ticker}>
        {hypotheses.map((h) => {
          const prev = prevMap.get(h.branch);
          const glyph = directionGlyph(prev?.posterior, h.posterior);

          return (
            <span
              key={h.branch}
              className={styles.token}
              onClick={(e) =>
                openFragment(
                  "hypothesis",
                  h.branch,
                  e.clientX,
                  e.clientY
                )
              }
            >
              <span className={styles.tokenName}>{h.label}</span>
              <span className={styles.tokenCount}>{h.claim_count}</span>
              {glyph && (
                <span className={styles.tokenGlyph}>{glyph}</span>
              )}
            </span>
          );
        })}

        <span className={styles.token}>
          <span className={styles.tokenName}>OTHER</span>
          <span className={styles.tokenCount}>1</span>
        </span>
      </div>

      <span className={styles.clock}>
        {elapsed} <span className={styles.active}>— active</span>
      </span>
    </div>
  );
}
