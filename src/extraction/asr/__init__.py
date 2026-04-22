"""ASR — mic audio → diarised clinical turns.

Public surface:
- `Turn` — output dataclass emitted per utterance.
- `build_pipeline` — construct the VAD + ASR + alignment + diariser pipeline.
- `AsrPipeline.transcribe` — run the full pipeline on a WAV file, yield `Turn`s.

See `research/asr_stack.md` for the wiring plan and `docs/asr_benchmark.md`
for the measured RTF + medical-term-WER numbers.
"""

from src.extraction.asr.pipeline import AsrPipeline, PipelineConfig, Turn, build_pipeline
from src.extraction.asr.vocab import build_initial_prompt

__all__ = [
    "AsrPipeline",
    "PipelineConfig",
    "Turn",
    "build_initial_prompt",
    "build_pipeline",
]
