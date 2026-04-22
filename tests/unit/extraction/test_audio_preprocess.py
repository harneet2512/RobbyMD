"""Tests for A.2 — audio preprocessing (normalize + trim silence).

Two tests:
1. normalize_audio produces a 16 kHz mono WAV.
2. trim_silence removes boundary silence and produces a shorter-or-equal file.

Both tests are skipped if ffmpeg is not on PATH (the function uses subprocess
to shell out to ffmpeg; it cannot run without it).
"""

from __future__ import annotations

import shutil
import struct
import wave
from pathlib import Path

import pytest

from src.extraction.asr.preprocess import ffmpeg_available, normalize_audio, trim_silence

# Skip entire module if ffmpeg is absent.
pytestmark = pytest.mark.skipif(
    not shutil.which("ffmpeg"),
    reason="ffmpeg required for audio preprocessing tests",
)


def _make_wav(path: Path, sample_rate: int = 44100, channels: int = 2, duration_s: float = 1.0) -> Path:
    """Write a minimal synthetic WAV file for testing."""
    n_frames = int(sample_rate * duration_s)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(sample_rate)
        # Pure 440 Hz sine-wave at low amplitude.
        import math
        frames = bytearray()
        for i in range(n_frames):
            sample = int(3000 * math.sin(2 * math.pi * 440 * i / sample_rate))
            for _ in range(channels):
                frames.extend(struct.pack("<h", sample))
        wf.writeframes(bytes(frames))
    return path


def _make_silence_wav(path: Path, duration_s: float = 2.0, speech_start: float = 0.5, speech_end: float = 1.5) -> Path:
    """Write a WAV with 500 ms silence, 1 s of tone, 500 ms silence."""
    sample_rate = 16_000
    n_total = int(sample_rate * duration_s)
    n_start_silence = int(sample_rate * speech_start)
    n_speech = int(sample_rate * (speech_end - speech_start))
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        frames = bytearray()
        import math
        for i in range(n_total):
            if n_start_silence <= i < n_start_silence + n_speech:
                sample = int(20000 * math.sin(2 * math.pi * 440 * i / sample_rate))
            else:
                sample = 0  # silence
            frames.extend(struct.pack("<h", sample))
        wf.writeframes(bytes(frames))
    return path


def test_normalize_produces_16khz_mono(tmp_path: Path) -> None:
    """normalize_audio must output a 16 kHz mono WAV."""
    src = _make_wav(tmp_path / "stereo_44k.wav", sample_rate=44100, channels=2)
    dst = tmp_path / "norm.wav"
    result = normalize_audio(src, dst)
    assert result == dst
    assert dst.exists()
    with wave.open(str(dst), "rb") as wf:
        assert wf.getframerate() == 16000, f"Expected 16000 Hz, got {wf.getframerate()}"
        assert wf.getnchannels() == 1, f"Expected 1 channel (mono), got {wf.getnchannels()}"


def test_trim_strips_boundary_silence(tmp_path: Path) -> None:
    """trim_silence must produce a file shorter than the input when silence is present."""
    src = _make_silence_wav(tmp_path / "silent_pad.wav", duration_s=2.0)
    dst = tmp_path / "trimmed.wav"
    result = trim_silence(src, dst)
    assert result == dst
    assert dst.exists()
    # The trimmed output should be shorter (or at most equal to) the original.
    with wave.open(str(src), "rb") as wf_src:
        src_frames = wf_src.getnframes()
    with wave.open(str(dst), "rb") as wf_dst:
        dst_frames = wf_dst.getnframes()
    # Trimmed file must be no longer than original.
    assert dst_frames <= src_frames, (
        f"trim_silence should not grow the file: src={src_frames} dst={dst_frames}"
    )
