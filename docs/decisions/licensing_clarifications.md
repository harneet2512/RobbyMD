# Licensing clarifications — Discord-question pending ADRs

**Status**: open (2 questions pending Discord reply)
**Driver**: hack_it operator
**Affected**: `rules.md §1.2`, `MODEL_ATTRIBUTIONS.md`, `SYNTHETIC_DATA.md`, `tests/licensing/*`

This document supersedes the earlier `2026-04-21_opus_api_compliance.md`. It records the two open licensing questions we want the hackathon organisers (Cerebral Valley × Anthropic) to confirm on the event Discord, and our working interpretation until they reply. Neither question blocks work — we proceed on the industry-standard reading — but a moderator answer supersedes our working interpretation.

---

## Q1 — Claude API (Opus 4.7) vs. the open-source rule

**Status**: PENDING — to be posted in #questions on Discord 2026-04-21.

**Question (copy-paste)**:

> Does using the Claude API (Opus 4.7) satisfy the hackathon open-source-everything rule, given the event is titled "Built with Opus 4.7"? Or does the rule require the LLM to be open-weight too?

**Working interpretation**:
Claude Opus 4.7 via the Anthropic API is the hackathon's **named sponsored platform tool** — the event is literally *Built with Opus 4.7*, and every participating team uses it. It is not a third-party proprietary component we ship alongside our code; it is the sponsored infrastructure, analogous to GitHub, Vercel, AWS, or the Cerebral Valley submission platform itself. Every *other* dependency in our repo is OSI-approved (code) or open-data licenced (model weights / datasets).

`rules.md §1.2` codifies this position. `tests/licensing/test_open_source.py` explicitly lists `anthropic` in `SPONSORED_TOOL_EXEMPT`.

**Decision on reply**:
- If the moderator confirms → we're done; close this entry.
- If the moderator requires open-weight LLM → **big blast radius**: we'd need a local medical LLM fallback as primary, with Opus 4.7 as optional enhancement. Revisit scope immediately.

---

## Q2 — CC-BY-4.0 for model weights (pyannote community-1)

**Status**: **RESOLVED 2026-04-21** — self-resolved against industry-standard reading; not posted to Discord.

### Resolution

User directive 2026-04-21 elected to proceed without Discord confirmation. The hackathon rule says *"approved open source license"*; the industry consensus (Linux Foundation's OpenMDW framework July 2025, OSI FAQ explicitly treating model weights as data not software, Hugging Face licensing guidance, OSI's own website using CC-BY-4.0 for its content, the vast majority of open-weight ML releases including pyannote and biomedical models) applies OSI to CODE and open-data licenses (CC-BY-4.0, CC-BY-SA-4.0, CDLA-Permissive-2.0, ODbL) to model weights. This is the standard reading, not a workaround.

**Binding outcome**:
- `rules.md §1.2` stands as amended 2026-04-21 (OSI for code; OSI or open-data for model weights; explicit denylist of CC-BY-NC / CC-BY-ND / Gemma ToU / HAI-DEF / Llama Community).
- `reasons.md` entry "Strict OSI-only licensing for model weights — rejected for industry-standard reading (2026-04-21)" documents the paper trail with citations (OpenMDW LF July 2025, OSI FAQ, OpenMDW-1.0, Model Openness Framework).
- `tests/licensing/test_model_attributions.py` enforces CC-BY-4.0 attribution in CI — attribution is load-bearing and cannot silently drift.
- `pyannote/speaker-diarization-community-1` (CC-BY-4.0) cleared for use.

If challenged at a later judging stage, the answer is: *"We applied the Linux Foundation OpenMDW framework (July 2025) reading — OSI for code, open-data (CC-BY-4.0) for model weights with mandatory attribution in MODEL_ATTRIBUTIONS.md, enforced by CI. Full paper trail in reasons.md."*

### Original question (kept for the submission paper trail)

> For model weights specifically, is CC-BY-4.0 acceptable? The Linux Foundation's OpenMDW framework (July 2025) recommends CC-BY-4.0 for model weights, with OSI licenses reserved for code. We're following that industry-standard split but want to confirm the hackathon interprets the rule the same way. Specific case: pyannote speaker-diarization-community-1 under CC-BY-4.0 for speaker diarisation.

### Revisit trigger

If a judge at Stage 1 or Stage 2 challenges the CC-BY-4.0 reading with a stricter interpretation **before** final submission (2026-04-26 20:00 EST), **swap** `pyannote/speaker-diarization-community-1` → **NVIDIA NeMo Sortformer** (Apache-2.0). Cost: same-day latency re-benchmark on a 24 GB GPU; ~1 day of wt-extraction delay. `research/asr_stack.md` §Alt-1 documents the fallback path.

---

## Protocol when a reply lands

1. Paste the moderator's verbatim reply under the Q that it answers, with a timestamp.
2. Flip the Status to **RESOLVED** (with the moderator's position as the ruling).
3. If the resolution changes any policy: open a new ADR in `docs/decisions/` and update `rules.md` + enforcement tests accordingly. Do NOT silently edit rules.md when a reply lands — every amendment gets its own dated ADR for traceability.
4. Append an entry to `progress.md` recording the reply and any downstream changes.

---

## References

- `rules.md §1.2` (current text, amended 2026-04-21 to include the code/weights split)
- `MODEL_ATTRIBUTIONS.md` — attribution registry
- `tests/licensing/test_model_attributions.py` — CI enforcement
- `reasons.md` — superseded-decision rationale with citations
- Cerebral Valley event page: *Built with Opus 4.7* (rules: "Everything shown in the demo must be fully open source…")
- [Linux Foundation / LF AI & Data, "Simplifying AI Model Licensing with OpenMDW" (July 2025)](https://lfaidata.foundation/blog/2025/07/22/simplifying-ai-model-licensing-with-openmdw/)
- [OSI FAQ — license categorisation](https://opensource.org/faq)
- [OpenMDW-1.0](https://lfaidata.foundation/projects/openmdw/)
- [Model Openness Framework (isitopen.ai)](https://isitopen.ai/)
