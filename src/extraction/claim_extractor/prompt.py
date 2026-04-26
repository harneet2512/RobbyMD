"""Opus 4.7 claim-extractor prompt — draft, not yet wired to the substrate.

Per `CLAUDE.md §5.2` Phase 2 scope: the prompt is drafted with ≥5 few-shot
examples; wire-up to wt-engine's `on_new_turn` API is blocked on that API
stabilising. When wt-engine publishes its claim schema, the only change
needed here is the output-schema JSON block below.

**Pack-aware (2026-04-21)**: `PREDICATE_FAMILIES` and `FEW_SHOT_EXAMPLES`
are loaded from the active `PredicatePack` via
`src/substrate/predicate_packs.py::active_pack`. Swapping packs (e.g.
`ACTIVE_PACK=personal_assistant` for LongMemEval-S) swaps the prompt's
closed vocabulary and in-context examples automatically — no code change.
Addresses audit finding #2 from commit `767d3e8`.

`Eng_doc.md §5.2` binds:
- Input: current turn + 2 prior turns + current active claim set + predicate family.
- Output: zero or more claim objects validated against the schema in §4.1.
- Target latency: ≤700 ms per call.
"""

from __future__ import annotations

from src.substrate.predicate_packs import FewShotExample, active_pack  # noqa: F401  # re-export

_pack = active_pack()

# Closed predicate set per the active pack (`Eng_doc.md §4.2`). Adding a
# predicate means editing the pack's `predicates.json`, not this module.
PREDICATE_FAMILIES: tuple[str, ...] = tuple(sorted(_pack.predicate_families))

# Few-shot examples per the active pack's `few_shot_examples.json`. Same
# invariants as `CLAUDE.md §5.2` — multi-claim, negation, self-correction /
# supersession, rare value, ambiguous phrasing — are satisfied by each pack's
# seeded example set (see `predicate_packs/<pack>/few_shot_examples.json`).
FEW_SHOT_EXAMPLES: tuple[FewShotExample, ...] = _pack.few_shot_examples


def _render_examples() -> str:
    """Render few-shot examples into the prompt body, in source order."""
    blocks: list[str] = []
    for i, ex in enumerate(FEW_SHOT_EXAMPLES, start=1):
        prior = "\n".join(f"    {spk}: {txt}" for spk, txt in ex.prior_turns)
        cur_spk, cur_txt = ex.current_turn
        blocks.append(
            f"### Example {i}: {ex.name}\n"
            f"Scenario: {ex.scenario}\n\n"
            f"Prior turns:\n{prior}\n"
            f"Current turn:\n    {cur_spk}: {cur_txt}\n"
            f"Active claims: {ex.active_claims_summary}\n\n"
            f"Expected output:\n{ex.expected_output}\n"
        )
    return "\n\n".join(blocks)


# The system prompt is composed once at import; `Eng_doc.md §5.2` says ≤700 ms
# latency end-to-end, which rules out per-call re-composition.
CLAIM_EXTRACTOR_SYSTEM_PROMPT: str = f"""\
You extract structured claims from one conversation turn. You emit **only**
JSON — a list of claim objects, possibly empty.

## Rules

1. **Predicate families (closed set, per active PredicatePack)**. Every claim's
   `predicate` MUST be one of:
   {", ".join(PREDICATE_FAMILIES)}.
   Emitting any other predicate is a failure. If a turn has content outside
   this set, drop it.

2. **One fact per claim**. If a turn contains multiple independent facts,
   emit multiple claims.

3. **Negations / denials**. When the speaker explicitly denies a feature,
   emit a claim with value `negated:<feature>` (clinical pack) or
   `dislikes:<feature>` (personal-assistant preferences). Follow the
   convention shown by the pack's few-shots.

4. **Honesty over fluency**. If the turn is ambiguous, either emit a claim
   with low `confidence` (≤ 0.5) and a vague `value`, or emit an empty list.
   NEVER fabricate a specific value you don't hear.

5. **No invented predicates**. If a speaker mentions something close to but
   not in the closed set, use the nearest listed family — do not invent a
   new predicate.

6. **Supersession is downstream**. You emit the new claim; the substrate
   decides what older claim it supersedes. You do not set status or
   supersession fields.

7. **Confidence scale** in [0, 1]. Use 0.9+ only when the speaker states
   the fact explicitly and unambiguously. Hedge words ("sort of", "kind of",
   "maybe") drop confidence below 0.7. Unheard or inferred content is never
   emitted.

8. **Exact fact preservation for memory benchmarks**. Preserve short scalar
   facts verbatim in at least one claim value: personal names and titles,
   locations, money amounts, durations, counts, dates/times, percentages,
   product/model names, course/degree names, and quoted titles. Do not round,
   paraphrase away, translate, or replace these values. If the turn says
   "$400,000", "8 days", "Dr. Arati Prabhakar", "the suburbs", "25:50",
   "Samsung Galaxy S22", or "Business Administration", at least one emitted
   claim value must contain that exact value or a numerically identical form.

## Output schema (one claim)

```
{{
  "subject":    str,   // what the claim is about, e.g. "chest_pain", "user"
  "predicate":  str,   // MUST be one of the predicate families above
  "value":      str,   // normalised value; for negations, "negated:<feature>"
  "confidence": float  // [0, 1]
}}
```

Return a JSON array (possibly empty). No commentary. No markdown.

## Few-shot examples (from active pack)

{_render_examples()}

## Context

You will receive: (a) up to 2 prior turns, (b) the current turn, (c) a
summary of active claims already in the substrate. Use the active-claims
summary only to avoid double-emitting facts already stated; do not cite them.
"""
