import { useEffect, useRef } from "react";
import { useTraceStore } from "@/store/trace";
import { TraceLine } from "./TraceLine";
import styles from "./Trace.module.css";

export function Trace() {
  const entries = useTraceStore((s) => s.entries);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [entries.length]);

  return (
    <div className={styles.container}>
      <div className={styles.fadeMask} />
      <div className={styles.column}>
        {entries.map((entry) => (
          <TraceLine key={entry.id} entry={entry} />
        ))}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
