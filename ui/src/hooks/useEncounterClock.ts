import { useState, useEffect } from "react";

export function useEncounterClock(startNs: number | null): string {
  const [now, setNow] = useState(Date.now());

  useEffect(() => {
    if (startNs === null) return;
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, [startNs]);

  if (startNs === null) return "--:--";

  const startMs = startNs / 1_000_000;
  const elapsed = Math.max(0, Math.floor((now - startMs) / 1000));
  const m = String(Math.floor(elapsed / 60)).padStart(2, "0");
  const s = String(elapsed % 60).padStart(2, "0");
  return `${m}:${s}`;
}
