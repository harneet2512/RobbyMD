export function nsToDate(ns: number): Date {
  return new Date(ns / 1_000_000);
}

export function formatTimestamp(ns: number): string {
  const d = nsToDate(ns);
  const h = String(d.getHours()).padStart(2, "0");
  const m = String(d.getMinutes()).padStart(2, "0");
  const s = String(d.getSeconds()).padStart(2, "0");
  return `${h}:${m}:${s}`;
}

export function formatElapsed(startNs: number, nowMs: number): string {
  const startMs = startNs / 1_000_000;
  const elapsed = Math.max(0, Math.floor((nowMs - startMs) / 1000));
  const h = String(Math.floor(elapsed / 3600)).padStart(2, "0");
  const m = String(Math.floor((elapsed % 3600) / 60)).padStart(2, "0");
  const s = String(elapsed % 60).padStart(2, "0");
  return `${h}:${m}:${s}`;
}
