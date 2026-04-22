"""Claim extractor — Opus 4.7 prompt + validation.

**Status: Phase 2, deferred.** The prompt text lives in `prompt.py` as a
pre-publication draft per CLAUDE.md §5.2 scope ("Wait for wt-engine to
publish the substrate's `on_new_turn` API + claim schema"). Wiring this
module to the substrate is gated on wt-engine publishing its claim schema
+ admission signatures.

Public surface today:
- `CLAIM_EXTRACTOR_SYSTEM_PROMPT` — the system prompt (draft).
- `FEW_SHOT_EXAMPLES` — ≥5 few-shot examples covering the cases demanded
  by Eng_doc.md §5.2 and CLAUDE.md §5.2.
"""

from src.extraction.claim_extractor.prompt import (
    CLAIM_EXTRACTOR_SYSTEM_PROMPT,
    FEW_SHOT_EXAMPLES,
    PREDICATE_FAMILIES,
)

__all__ = [
    "CLAIM_EXTRACTOR_SYSTEM_PROMPT",
    "FEW_SHOT_EXAMPLES",
    "PREDICATE_FAMILIES",
]
