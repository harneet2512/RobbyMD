import type { TraceEntry } from "@/lib/claim-prose";
import { useEncounterStore } from "@/store/encounter";
import { formatTimestamp } from "@/lib/format";
import styles from "./Trace.module.css";

interface Props {
  entry: TraceEntry;
}

export function TraceLine({ entry }: Props) {
  const openFragment = useEncounterStore((s) => s.openFragment);
  const isSuperseded = !!entry.supersededClaimId;
  const isLowConf = entry.confidence < 0.7;

  const ts = formatTimestamp(entry.ts_ns);

  const handleClick = (e: React.MouseEvent) => {
    if (entry.claimIds.length > 0) {
      openFragment("claim", entry.claimIds[0]!, e.clientX, e.clientY);
    }
  };

  if (entry.kind === "gap") {
    return (
      <div className={styles.line}>
        <span className={styles.timestamp}>{ts}</span>
        <span className={styles.gapBody}>
          [{entry.fragments.join("")}]
        </span>
      </div>
    );
  }

  if (entry.kind === "verifier") {
    return (
      <div className={styles.line}>
        <span className={styles.timestamp}>{ts}</span>
        <span className={styles.verifierBody}>
          {entry.fragments.join("")}
        </span>
      </div>
    );
  }

  if (entry.kind === "decision") {
    return (
      <div className={styles.line} onClick={handleClick}>
        <span className={styles.timestamp}>{ts}</span>
        <span className={styles.body}>
          <span className={styles.glyph}>{entry.decisionGlyph}</span>
          {" "}
          {entry.fragments.join(" · ")}
          {entry.decisionInitials && (
            <>
              {" · "}
              <span className={styles.initials}>{entry.decisionInitials}</span>
            </>
          )}
        </span>
      </div>
    );
  }

  if (entry.kind === "ranking") {
    return (
      <div className={styles.line}>
        <span className={styles.timestamp}>{ts}</span>
        <span className={styles.rankingBody}>
          {entry.fragments.join(" — ")}
        </span>
      </div>
    );
  }

  // claims: joined with interpunct, the authored register
  const lineClass = [
    styles.line,
    isSuperseded ? styles.superseded : "",
    isLowConf ? styles.lowConf : "",
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <div className={lineClass} onClick={handleClick}>
      <span className={styles.timestamp}>
        {isSuperseded && <span className={styles.supersessionMark}>↺{"  "}</span>}
        {ts}
      </span>
      <span className={styles.body}>
        {entry.fragments.join(" · ")}
      </span>
    </div>
  );
}
