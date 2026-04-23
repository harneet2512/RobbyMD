# 2026-04-21 — CC-BY-4.0 for model weights: extend rules.md §1.2 allowlist

**Status**: **accepted by user directive 2026-04-23** — `rules.md §1.2` patch pending human commit (CLAUDE.md §4 forbids agent edits of `rules.md`; the recommended-text patch in this ADR is the authoritative text the human will apply). Attribution obligations for CC-BY-4.0 model weights are consolidated in `CC-BY-4.0-ATTRIBUTIONS.md` at repo root.
**Driver**: hack_it operator
**Affected**: `rules.md §1.2`, `tests/licensing/test_open_source.py`, `research/asr_stack.md` (§R1 escalation), `wt-extraction` dispatch readiness

## Context

Researcher A's ASR stack brief (`research/asr_stack.md`) pins the speaker-diarisation model to **`pyannote/speaker-diarization-community-1`** — the current default of WhisperX 3.8.5 and the maintained successor to the gated `pyannote/speaker-diarization-3.1`. The model weights are released under **CC-BY-4.0**.

`rules.md §1.2` currently lists an OSI allowlist of **MIT, Apache-2.0, BSD (2/3/4-clause), MPL-2.0, ISC, LGPL** for every *dependency* (library, model, frontend component, backend component). CC-BY-4.0 is not on this list. A strict reading would forbid `pyannote/speaker-diarization-community-1`.

However, CC-BY-4.0 has a specific and well-understood meaning for **data artifacts** (including model weights): it permits redistribution and derivative use with attribution. The OSI has historically declined to accept CC licences as "open source" specifically because they are not designed for source code. For *model weights* — which are data, not code — the distinction is material.

The Validator report flagged this as WARN-R1; wt-extraction cannot commit diariser code until this is resolved.

## Decision (proposed)

**Extend `rules.md §1.2` to distinguish code licences from model-weight licences.** Recommended text:

> §1.2 (extension): For **model weights and data artifacts** (distinct from source code), the following additional licences are permitted alongside the OSI-approved set: **CC-BY-4.0** (attribution-preserving), **CC-BY-SA-4.0** (share-alike). Attribution is recorded in `SYNTHETIC_DATA.md` and surfaced in the app's model-credits display.
>
> For **source code, libraries, and frontend/backend dependencies**, the OSI-approved allowlist remains unchanged: MIT, Apache-2.0, BSD (2/3/4-clause), MPL-2.0, ISC, LGPL.
>
> `tests/licensing/test_open_source.py` inspects `pyproject.toml` and `package.json` (code dependencies only). A separate static check verifies that every model weight or large binary artifact referenced from code has an attribution line in `SYNTHETIC_DATA.md`.

Once this extension lands:
- `pyannote/speaker-diarization-community-1` is permitted with attribution in `SYNTHETIC_DATA.md` and the UI model-credits area.
- `wt-extraction` can dispatch and commit diariser code.
- No change to the code-dependency licence test.

## Alternatives considered

1. **Swap to NVIDIA NeMo Sortformer (Apache-2.0)**. Pure OSI, no ADR needed, no allowlist extension.
   - Cost: NeMo pulls a ~3 GB CUDA toolkit and the diarisation stack is heavier; may not hit our latency budget on a 16–24 GB GPU alongside Whisper large-v3. Needs a same-day benchmark.
   - Verdict: viable fallback if extension is rejected.

2. **Drop diarisation from the demo** — speaker labels come from heuristics (e.g., "odd-numbered turns = physician") on scripted audio.
   - Cost: loses the "diarisation + WhisperX alignment" quality claim from the ASR brief; scripted demo can fake it, but the UX and the ambient-scribe framing weaken.
   - Verdict: acceptable only as a last-ditch fallback.

3. **Keep strict §1.2 interpretation, block pyannote entirely.**
   - Cost: no available Apache-2.0/MIT diariser matches WhisperX's quality-at-latency. NeMo is the only viable alternative and has its own cost (above).
   - Verdict: strictest reading; operationally expensive.

**Recommended**: option 0 (this ADR — extend §1.2). Option 1 (NeMo) is the best fallback if option 0 is rejected.

## Consequences

### If accepted
- `rules.md §1.2` gains the data-vs-code distinction.
- `tests/licensing/test_open_source.py` stays scope-limited to `pyproject.toml`/`package.json` code deps (it already is).
- New static check `tests/licensing/test_model_attributions.py` added: for every model weight referenced in code (e.g., `pyannote/speaker-diarization-community-1`), there must be a corresponding attribution line in `SYNTHETIC_DATA.md` naming the licence.
- `wt-extraction` unblocks; diariser code commits to `feature/extraction`.
- `SYNTHETIC_DATA.md` gains a "Model weights" section listing pyannote + Whisper + Distil-Whisper + e5-small-v2 + silero-vad with licences.

### If rejected
- Swap to NeMo Sortformer per Alternative 1. `research/asr_stack.md` needs a §-update ADR-revision pass + NeMo latency benchmark on Day 2.
- No change to `rules.md`.

## References

- `research/asr_stack.md` §R1 (original escalation)
- `research/validation_report.md` §4 WARN #1
- `rules.md §1.2` (current text to amend)
- [pyannote/speaker-diarization-community-1 model card (CC-BY-4.0)](https://huggingface.co/pyannote/speaker-diarization-community-1)
- [WhisperX 3.8.5 README — community-1 default](https://github.com/m-bain/whisperX)
- [NVIDIA NeMo — diarisation (Apache-2.0)](https://github.com/NVIDIA/NeMo)
- [OSI position on CC licences](https://opensource.org/faq#cc)
- Cerebral Valley × Anthropic hackathon rule: "Everything shown in the demo must be fully open source…"

## Human decision needed

- [x] Accept extension (recommended) → user directive 2026-04-23. Human still to commit the `rules.md §1.2` patch per the recommended text above.
- [ ] Reject extension → swap to NeMo Sortformer per Alternative 1; I update `research/asr_stack.md` and run NeMo latency benchmark before wt-extraction dispatches.
- [ ] Defer decision → wt-extraction blocked on diariser; can dispatch transcription-only work on `feature/extraction` in the meantime.

## 2026-04-23 — user acceptance

User directive on 2026-04-23 accepts the proposed extension. Effective immediately for agent decision-making, with the `rules.md §1.2` patch itself still owed from the human operator.

Models unblocked under this acceptance (CC-BY-4.0 weights):
- `nvidia/canary-qwen-2.5b` — Variant B ASR on `flow/variant-b` (Bundle 5)
- `pyannote/speaker-diarization-3.1` — speaker diarisation across Variant A + Variant B (same model, held constant)
- `hexgrad/Kokoro-82M` — Apache-2.0, not strictly under this ADR but attributed alongside for centralised model-credits visibility

Attribution for all three is consolidated at repo root in `CC-BY-4.0-ATTRIBUTIONS.md`. The existing `tests/licensing/test_open_source.py` (code-dep allowlist, inspects `pyproject.toml` / `package.json`) is not impacted.
