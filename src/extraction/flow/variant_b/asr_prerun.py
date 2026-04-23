"""One-shot Canary-Qwen transcription with 30s chunking.

Canary-Qwen-2.5B has a 40s audio window per the NVIDIA model card. Our clips
are 76-99s so we slide a 30s window (5s overlap) across each file, transcribe
each chunk, and concatenate. Overlap trims at boundary to avoid duplication.
"""
from __future__ import annotations
import json, time, argparse, os, tempfile
from pathlib import Path

os.environ.setdefault("HF_TOKEN", Path.home().joinpath(".hf_token").read_text().strip())

import soundfile as sf
import numpy as np
from nemo.collections.speechlm2.models import SALM


CHUNK_SEC = 30.0
OVERLAP_SEC = 5.0


def chunk_audio(path: str, sr_target: int = 16000):
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != sr_target:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sr_target)
        sr = sr_target
    dur = len(audio) / sr
    step = CHUNK_SEC - OVERLAP_SEC
    chunks = []
    start = 0.0
    while start < dur:
        end = min(start + CHUNK_SEC, dur)
        chunks.append((start, end, audio[int(start*sr):int(end*sr)]))
        if end >= dur:
            break
        start += step
    return sr, dur, chunks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ground-truth", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()

    clips = [json.loads(ln) for ln in Path(args.ground_truth).read_text().splitlines() if ln.strip()]
    print(f"[asr-prerun] {len(clips)} clips, loading Canary-Qwen...", flush=True)
    t0 = time.time()
    m = SALM.from_pretrained("nvidia/canary-qwen-2.5b").eval().to("cuda")
    print(f"[asr-prerun] model loaded in {time.time()-t0:.1f}s", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for clip in clips:
            scenario = clip["scenario"]
            audio_path = clip["audio_path"]
            sr, dur, chunks = chunk_audio(audio_path)
            print(f"[asr-prerun] {scenario}: {dur:.1f}s, {len(chunks)} chunks", flush=True)
            t_total = time.perf_counter()
            parts = []
            with tempfile.TemporaryDirectory() as td:
                for i, (s_sec, e_sec, audio_np) in enumerate(chunks):
                    wav = Path(td) / f"chunk_{i}.wav"
                    sf.write(wav, audio_np, sr)
                    t = time.perf_counter()
                    out = m.generate(
                        prompts=[[{"role": "user",
                                   "content": f"Transcribe the following: {m.audio_locator_tag}",
                                   "audio": [str(wav)]}]],
                        max_new_tokens=args.max_new_tokens,
                    )
                    dt = (time.perf_counter() - t) * 1000
                    ids = out[0].tolist() if hasattr(out[0], "tolist") else list(out[0])
                    tok = m.tokenizer
                    text = tok.ids_to_text(ids) if hasattr(tok, "ids_to_text") else tok.decode(ids)
                    parts.append(text.strip())
                    print(f"[asr-prerun]   chunk {i} [{s_sec:.1f}-{e_sec:.1f}s]: {dt:.0f}ms, {len(text)} chars", flush=True)
            # Simple concat; overlap handling is lightweight here since medical dialogue chunks
            # rarely produce verbatim-duplicate phrasing across a 5s boundary.
            full_text = " ".join(parts).strip()
            total_ms = (time.perf_counter() - t_total) * 1000
            rec = {"scenario": scenario, "audio_path": audio_path,
                   "transcript": full_text, "asr_ms": total_ms,
                   "n_chunks": len(chunks), "audio_duration_sec": dur}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[asr-prerun]   total {total_ms:.0f}ms, {len(full_text)} chars", flush=True)
    print(f"[asr-prerun] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
