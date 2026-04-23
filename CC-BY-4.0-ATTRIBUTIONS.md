# CC-BY-4.0 Attributions

Model-weight and data-artifact attributions required under CC-BY-4.0 §3 and collected here for OSS-license compliance across the RobbyMD hackathon build. Listed per-artifact: name, authors/source, canonical URL, licence, and where it is used in this repository.

CC-BY-4.0 admissibility for model-weight data artifacts (distinct from source code) is governed by `docs/decisions/2026-04-21_pyannote-ccby40-model-weights.md` (accepted by user directive 2026-04-23). Source-code dependencies remain on the `rules.md §1.2` OSI allowlist (MIT / Apache-2.0 / BSD / MPL / ISC / LGPL) — this file covers only model weights and data.

---

## nvidia/canary-qwen-2.5b

- **Source**: NVIDIA NeMo team
- **URL**: https://huggingface.co/nvidia/canary-qwen-2.5b
- **Licence**: CC-BY-4.0 (model card: "This model is ready for commercial use")
- **Architecture**: FastConformer speech encoder + Qwen3-1.7B LLM (frozen during training) + LoRA adapters, SALM-style, 2.5B parameters total
- **Published baseline at release (July 2025)**: #1 on HuggingFace Open ASR Leaderboard at 5.63% avg WER; 458 RTFx on A100; ~1.6% WER on LibriSpeech clean
- **Training data**: 234K hours; AMI oversampled 15% for conversational-speech coverage
- **Use in this repo**: `src/extraction/flow/variant_b/pipeline.py` — ASR stage of the Variant B cascaded pipeline (measured against Variant A's Whisper-large-v3-turbo on identical audio clips)
- **Attribution required by CC-BY-4.0 §3**: "Canary-Qwen-2.5B (CC-BY-4.0), NVIDIA, https://huggingface.co/nvidia/canary-qwen-2.5b"

## pyannote/speaker-diarization-3.1

- **Source**: Hervé Bredin et al., pyannote.audio project
- **URL**: https://huggingface.co/pyannote/speaker-diarization-3.1
- **Licence**: CC-BY-4.0 (model weights); pyannote.audio library itself is MIT
- **Use in this repo**: `src/extraction/flow/variant_b/pipeline.py` — speaker diarisation stage of the Variant B cascaded pipeline; also used by Variant A (held constant across both variants so ASR choice is the only variable)
- **Attribution required by CC-BY-4.0 §3**: "pyannote/speaker-diarization-3.1 (CC-BY-4.0), Hervé Bredin et al., https://huggingface.co/pyannote/speaker-diarization-3.1"

## hexgrad/Kokoro-82M

- **Source**: Kokoro TTS project
- **URL**: https://huggingface.co/hexgrad/Kokoro-82M
- **Licence**: Apache-2.0 (included here for centralised model-credits visibility even though Apache-2.0 does not require CC-BY-style attribution)
- **Use in this repo**: Variant A (Bundle 4) renders six synthetic clinical dialogue clips used as the A/B evaluation audio. Variant B consumes those clips from `origin/flow/variant-a` — does not re-render.

---

## How this list is maintained

- Every model weight pulled via `scripts/download_and_verify_licenses.py` that resolves to a CC-BY-4.0 licence MUST have an entry here before code that invokes it can merge to `main`.
- The L4 operator (Bundle 5 Step 2) runs the license-verify script before download; if a new CC-BY model appears in the allowlist, extend this file in the same PR.
- Source-code licence compliance (MIT/Apache/etc. deps in `pyproject.toml` and `package.json`) is enforced by `tests/licensing/test_open_source.py` and is out of scope for this file.

## References

- `docs/decisions/2026-04-21_pyannote-ccby40-model-weights.md` — ADR extending `rules.md §1.2` to distinguish code-licence vs model-weight-licence allowlists
- `rules.md §1.2` — OSI-approved allowlist for source-code dependencies (unchanged by this work)
- `CLAUDE.md §2 rule #2` — hackathon OSI constraint
