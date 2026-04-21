"""Opus 4.7 claim-extractor prompt — **draft**, not yet wired to the substrate.

Per CLAUDE.md §5.2 Phase 2 scope: the prompt is drafted with ≥5 few-shot
examples; wire-up to wt-engine's `on_new_turn` API is blocked on that API
stabilising. When wt-engine publishes its claim schema, the only change
needed here is the output-schema JSON in `CLAIM_EXTRACTOR_SYSTEM_PROMPT` —
the examples + predicate-family invariants are stable.

Eng_doc.md §5.2 binds:
- Input: current turn + 2 prior turns + current active claim set + predicate family.
- Output: zero or more claim objects validated against the schema in §4.1.
- Target latency: ≤700 ms per call.

Few-shot coverage (required by CLAUDE.md §5.2):
1. Multi-claim utterance.
2. Negative finding.
3. Patient self-correction (supersession).
4. Rare symptom.
5. Ambiguous phrasing (the model must express low confidence, not guess).
"""

from __future__ import annotations

from dataclasses import dataclass

# Closed predicate set per Eng_doc.md §4.2. Adding a predicate requires a PR.
PREDICATE_FAMILIES: tuple[str, ...] = (
    "onset",
    "character",
    "severity",
    "location",
    "radiation",
    "aggravating_factor",
    "alleviating_factor",
    "associated_symptom",
    "duration",
    "medical_history",
    "medication",
    "family_history",
    "social_history",
    "risk_factor",
)


@dataclass(frozen=True, slots=True)
class FewShotExample:
    """One canonical prompt-response pair for in-context learning."""

    name: str
    scenario: str
    prior_turns: tuple[tuple[str, str], ...]  # (speaker, text)
    current_turn: tuple[str, str]
    active_claims_summary: str  # what's already in the substrate
    expected_output: str  # JSON the model should emit


# NOTE: expected_output strings use illustrative claim_id placeholders
# (e.g. "c_001"). Production wiring will let the substrate assign real ids
# and persist them; the prompt shape stays identical.


FEW_SHOT_EXAMPLES: tuple[FewShotExample, ...] = (
    FewShotExample(
        name="multi_claim_utterance",
        scenario=(
            "One patient sentence packs multiple independent clinical facts. "
            "The extractor must emit one claim per fact, not bundle them."
        ),
        prior_turns=(
            ("physician", "Tell me about the chest pain."),
        ),
        current_turn=(
            "patient",
            "It started about an hour ago, feels like pressure, and radiates to my left arm.",
        ),
        active_claims_summary="(no active claims)",
        expected_output="""[
  {"subject":"chest_pain","predicate":"onset","value":"1 hour ago","confidence":0.92},
  {"subject":"chest_pain","predicate":"character","value":"pressure","confidence":0.95},
  {"subject":"chest_pain","predicate":"radiation","value":"left_arm","confidence":0.97}
]""",
    ),
    FewShotExample(
        name="negative_finding",
        scenario=(
            "Patient explicitly denies a feature. Encode as a claim with "
            "value prefixed 'negated:' so the differential engine can apply LR-."
        ),
        prior_turns=(
            ("physician", "Any shortness of breath?"),
        ),
        current_turn=("patient", "No, I can breathe fine."),
        active_claims_summary="(no active claims)",
        expected_output="""[
  {"subject":"chest_pain","predicate":"associated_symptom","value":"negated:dyspnea","confidence":0.93}
]""",
    ),
    FewShotExample(
        name="patient_self_correction_supersession",
        scenario=(
            "Patient corrects a prior statement. Emit the new claim; the "
            "substrate's deterministic supersession Pass 1 will link it to "
            "the superseded claim (same (subject, predicate), different value)."
        ),
        prior_turns=(
            ("patient", "The pain started about an hour ago."),
            ("physician", "You said an hour — can you be more precise?"),
        ),
        current_turn=(
            "patient",
            "Actually, thinking again, it was more like thirty minutes ago.",
        ),
        active_claims_summary=(
            "c_001: chest_pain.onset = '1 hour ago' (confidence 0.92)"
        ),
        expected_output="""[
  {"subject":"chest_pain","predicate":"onset","value":"30 minutes ago","confidence":0.94}
]""",
    ),
    FewShotExample(
        name="rare_symptom",
        scenario=(
            "A specific, low-base-rate symptom that matters diagnostically "
            "(here: radiation to jaw, suggestive of ACS). Encode crisply; "
            "don't normalise away the clinical specificity."
        ),
        prior_turns=(
            ("physician", "Does the pain travel anywhere?"),
        ),
        current_turn=(
            "patient",
            "It goes up into my jaw and makes my teeth ache.",
        ),
        active_claims_summary="(no active claims)",
        expected_output="""[
  {"subject":"chest_pain","predicate":"radiation","value":"jaw","confidence":0.95},
  {"subject":"chest_pain","predicate":"associated_symptom","value":"jaw_ache","confidence":0.80}
]""",
    ),
    FewShotExample(
        name="ambiguous_phrasing",
        scenario=(
            "Utterance is vague. The extractor must emit a low-confidence "
            "claim (never fabricate) OR emit nothing. Confidence must honestly "
            "reflect the ambiguity so the UI can dim the claim and the "
            "differential engine can under-weight it."
        ),
        prior_turns=(
            ("physician", "How would you describe the pain?"),
        ),
        current_turn=("patient", "It's just... weird. Like a strange feeling."),
        active_claims_summary="(no active claims)",
        expected_output="""[
  {"subject":"chest_pain","predicate":"character","value":"unspecified","confidence":0.30}
]""",
    ),
    FewShotExample(
        name="medication_history",
        scenario=(
            "Drug names are high-value, often mispronounced. Normalise to "
            "RxNorm-style lowercase ingredient names; keep the patient's "
            "surface form in `value` for provenance."
        ),
        prior_turns=(
            ("physician", "What medications do you take regularly?"),
        ),
        current_turn=(
            "patient",
            "I take metoprolol and aspirin every morning, and atorvastatin in the evening.",
        ),
        active_claims_summary="(no active claims)",
        expected_output="""[
  {"subject":"patient","predicate":"medication","value":"metoprolol","confidence":0.97},
  {"subject":"patient","predicate":"medication","value":"aspirin","confidence":0.98},
  {"subject":"patient","predicate":"medication","value":"atorvastatin","confidence":0.95}
]""",
    ),
)


def _render_examples() -> str:
    """Render few-shot examples into the prompt body, fixed order."""
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


# The system prompt is composed once at import; Eng_doc.md §5.2 says ≤700 ms
# latency end-to-end, which rules out per-call re-composition.
CLAIM_EXTRACTOR_SYSTEM_PROMPT: str = f"""\
You extract structured clinical claims from one physician-patient conversation
turn. You emit **only** JSON — a list of claim objects, possibly empty.

## Rules

1. **Predicate families (closed set)**. Every claim's `predicate` MUST be one of:
   {", ".join(PREDICATE_FAMILIES)}.
   Emitting any other predicate is a failure. If a turn has content outside
   this set, drop it.

2. **One fact per claim**. If a turn contains multiple independent facts,
   emit multiple claims.

3. **Negations**. When the patient explicitly denies a feature, emit a claim
   with value `negated:<feature>`. E.g. no dyspnea -> value `negated:dyspnea`.

4. **Honesty over fluency**. If the turn is ambiguous, either emit a claim
   with low `confidence` (<= 0.5) and `value` = `"unspecified"`, or emit an
   empty list. NEVER fabricate a specific value you don't hear.

5. **No invented predicates**. If the patient mentions a cough (common in
   pulmonary differentials) and the closest predicate family is
   `associated_symptom`, use that — do NOT invent `cough` as its own predicate.

6. **Supersession is downstream**. You emit the new claim; the substrate
   decides what older claim it supersedes. You do not set status or
   supersession fields.

7. **Confidence scale** in [0, 1]. Use 0.9+ only when the patient states the
   fact explicitly and unambiguously. Hedge words ("sort of", "kind of",
   "maybe") drop confidence below 0.7. Unheard or inferred content is never
   emitted.

## Output schema (one claim)

```
{{
  "subject":    str,   // what the claim is about, e.g. "chest_pain", "patient"
  "predicate":  str,   // MUST be one of the predicate families above
  "value":      str,   // normalised value; for negations, "negated:<feature>"
  "confidence": float  // [0, 1]
}}
```

Return a JSON array (possibly empty). No commentary. No markdown.

## Few-shot examples

{_render_examples()}

## Context

You will receive: (a) up to 2 prior turns, (b) the current turn, (c) a summary
of active claims already in the substrate. Use the active-claims summary only
to avoid double-emitting facts already stated; do not cite them.
"""
