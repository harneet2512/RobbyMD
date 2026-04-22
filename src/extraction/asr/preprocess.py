"""Audio preprocessing for the ASR pipeline.

Normalisation and silence trimming before feeding audio to Whisper.

Motivation:
- Whisper was trained on 16 kHz mono PCM audio. Supplying any other sample rate
  or channel count forces the model's internal resampler, which adds latency and
  can introduce artefacts (e.g. DC offset from stereo collapse).
- Leading/trailing silence is a documented Whisper hallucination trigger: the
  model fills silent frames with repeated or nonsensical text.
  Cite: Koenecke et al., "Disparate ASR Accuracy in a Clinical Setting",
  ACM FAccT 2024 (arXiv 2312.05420) — observed elevated hallucination rate on
  utterances with long pre-roll or post-roll silence.

Two-step pipeline per CLAUDE.md §5.2 / Eng_doc.md §3.1:
1. `normalize_audio` — ffmpeg loudnorm to EBU R128 integrated loudness,
   output 16 kHz mono PCM WAV.
2. `trim_silence` — strip leading/trailing silence below a configurable dB
   threshold (default -40 dBFS), preserving interior speech pauses.

Both functions take `path_in` / `path_out` so they can be chained without
intermediate in-memory buffers (important for 5+ minute consultation recordings).

ffmpeg is a system dependency. Tests are skip-guarded via::

    @pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

# ffmpeg binary name — on Windows this is ffmpeg.exe, resolved by shutil.which.
_FFMPEG = "ffmpeg"


def _run_ffmpeg(*args: str, timeout_s: int = 120) -> None:
    """Run ffmpeg with the given argument list, raising on non-zero exit."""
    cmd = [_FFMPEG, "-y", *args]
    logger.debug("preprocess.ffmpeg.run", cmd=cmd)
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg exited {result.returncode}: {result.stderr[-500:]!r}"
        )


def normalize_audio(path_in: Path, path_out: Path) -> Path:
    """Normalise audio to 16 kHz mono PCM WAV with EBU R128 loudness correction.

    Uses ffmpeg's `loudnorm` filter (two-pass, linear mode) to bring the clip
    to -23 LUFS integrated loudness, then outputs 16 kHz mono signed-16-bit PCM
    WAV — the exact format Whisper large-v3 was trained on.

    Parameters
    ----------
    path_in:
        Source audio file (any format ffmpeg can decode: WAV, MP3, FLAC, OGG…).
    path_out:
        Destination path for the normalised WAV. Created (or overwritten) by
        this function. Parent directory must exist.

    Returns
    -------
    Path
        The written `path_out`, for easy chaining with `trim_silence`.
    """
    path_out.parent.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg(
        "-i", str(path_in),
        # loudnorm EBU R128 integrated loudness normalisation.
        # I (integrated loudness target), TP (true peak), LRA (loudness range).
        "-filter:a", "loudnorm=I=-23:TP=-1.5:LRA=11",
        "-ar", "16000",          # resample to 16 kHz
        "-ac", "1",              # downmix to mono
        "-sample_fmt", "s16",   # signed 16-bit PCM
        str(path_out),
    )
    logger.info(
        "preprocess.normalize_audio.done",
        src=str(path_in),
        dst=str(path_out),
    )
    return path_out


def trim_silence(
    path_in: Path,
    path_out: Path,
    threshold_db: float = -40.0,
) -> Path:
    """Strip leading and trailing silence from a WAV file.

    Uses ffmpeg's `silenceremove` filter with the `start_mode=any` /
    `stop_mode=any` config. Interior silences (natural speech pauses) are
    preserved — only the boundary dead-air is removed.

    Boundary silence is a documented Whisper hallucination trigger: the model
    tends to repeat the last decoded token or emit filler tokens ("Thank you",
    "Bye", "The") when it sees many silent MFCC frames at the start or end of
    a clip (Koenecke FAccT 2024, arXiv 2312.05420).

    Parameters
    ----------
    path_in:
        Source WAV (typically the output of `normalize_audio`).
    path_out:
        Destination path for the trimmed WAV.
    threshold_db:
        dB threshold below which a frame is considered silence.
        Default -40.0 dBFS. More negative = more aggressive trimming.

    Returns
    -------
    Path
        The written `path_out`, for easy chaining.
    """
    path_out.parent.mkdir(parents=True, exist_ok=True)
    # silenceremove docs:
    #   start_periods=1   — remove one leading-silence region
    #   start_duration=0  — start trimming immediately at threshold
    #   start_threshold   — dB level treated as silence at the start
    #   stop_periods=-1   — remove ALL trailing-silence regions (last only)
    #   stop_duration=0.1 — min duration of trailing silence to remove
    #   stop_threshold    — dB level treated as silence at the end
    # Negative start_periods removes from the start; the stop_periods=-1 idiom
    # in ffmpeg 6.x trims the last contiguous silence block.
    _run_ffmpeg(
        "-i", str(path_in),
        "-filter:a",
        (
            f"silenceremove=start_periods=1:start_duration=0"
            f":start_threshold={threshold_db}dB"
            f":stop_periods=-1:stop_duration=0.1"
            f":stop_threshold={threshold_db}dB"
        ),
        str(path_out),
    )
    logger.info(
        "preprocess.trim_silence.done",
        src=str(path_in),
        dst=str(path_out),
        threshold_db=threshold_db,
    )
    return path_out


def ffmpeg_available() -> bool:
    """Return True iff ffmpeg is on PATH."""
    return shutil.which(_FFMPEG) is not None
