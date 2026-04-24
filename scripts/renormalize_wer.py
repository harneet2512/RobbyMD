"""
Recompute WER on an existing per_clip_metrics.jsonl with case-insensitive,
punctuation-stripped normalization. For fair apples-to-apples comparison
between variant_a (default jiwer) and ship (default jiwer on
case-inconsistent Whisper output).

Usage:
    python scripts/renormalize_wer.py <per_clip_metrics.jsonl>

Writes alongside as per_clip_normalized.jsonl and prints aggregate.
"""
from __future__ import annotations

import json
import re
import statistics
import string
import sys
from pathlib import Path

import jiwer

_PUNCT_RE = re.compile("[" + re.escape(string.punctuation) + "]")


def _normalize(text: str) -> str:
    t = text.lower()
    t = _PUNCT_RE.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python scripts/renormalize_wer.py <per_clip_metrics.jsonl>", file=sys.stderr)
        return 1

    src = Path(sys.argv[1])
    rows = []
    with src.open() as f:
        for line in f:
            if not line.strip():
                continue
            m = json.loads(line)
            if "hypothesis_raw" not in m or "reference" not in m:
                continue
            ref = m["reference"]
            raw = m["hypothesis_raw"]
            # variant_a uses 'hypothesis_cleaned', ship uses 'hypothesis_corrected'
            post = m.get("hypothesis_cleaned") or m.get("hypothesis_corrected") or raw
            rows.append({
                "scenario": m["scenario"],
                "wer_raw_default": m.get("wer_raw"),
                "wer_raw_normalized": jiwer.wer(_normalize(ref), _normalize(raw)),
                "wer_post_default": m.get("wer_cleaned") or m.get("wer_corrected"),
                "wer_post_normalized": jiwer.wer(_normalize(ref), _normalize(post)),
            })

    dst = src.parent / "per_clip_normalized.jsonl"
    with dst.open("w") as o:
        for r in rows:
            o.write(json.dumps(r) + "\n")

    if rows:
        rawn = [r["wer_raw_normalized"] for r in rows]
        postn = [r["wer_post_normalized"] for r in rows]
        print(f"n={len(rows)} clips")
        print(f"mean WER raw  default    = {statistics.mean([r['wer_raw_default'] for r in rows]):.3%}")
        print(f"mean WER raw  normalized = {statistics.mean(rawn):.3%}")
        print(f"mean WER post default    = {statistics.mean([r['wer_post_default'] for r in rows]):.3%}")
        print(f"mean WER post normalized = {statistics.mean(postn):.3%}")
        print(f"correction delta (norm)  = {(statistics.mean(postn) - statistics.mean(rawn)) * 100:+.2f}pp")
        print(f"\nwrote {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
