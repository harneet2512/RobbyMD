# LongMemEval Capability Matrix

Date: 2026-04-25

This matrix replaces case-specific fixes with capability buckets. Each of the
20 diagnostic cases maps to exactly one bucket.

## Synthetic harness

Prompt for each synthetic case:

```text
Context (simulated conversation history):
[Insert 3-4 turn conversation with exact facts embedded]

Query:
What is [specific fact asked]?

Your task:
1. Extract all claims from context.
2. Return the exact fact with span provenance.
3. Confidence: high/medium/low.
4. If unsure, abstain.
```

Scoring:
- Correct fact plus correct span: 1 point.
- Correct fact with wrong span or confidence off: 0.5 points.
- Wrong fact or hallucinated answer: 0 points.
- Abstain on uncertain evidence: 0.5 points.

## 1. exact_fact_preservation

Contract: exact names, titles, locations, amounts, durations, model names, and
short scalar answers stated in a turn must survive extraction verbatim enough
for strict short-answer scoring.

Cases: `70b3e69b`, `e3fc4d6e`, `830ce83f`, `852ce960`

Synthetic invariant tests:
- Assistant states `Dr. Mira Patel`; extracted claim contains `Dr. Mira Patel`.
- Assistant states `$400,000`; extracted claim contains `$400,000` or `400000`, not another amount.
- User states `the suburbs`; extracted claim contains `the suburbs`.
- Assistant states `8 days`; extracted claim contains `8 days`, not only `days`.
- User states `Samsung Galaxy S22`; extracted claim contains the full model.
- Assistant states `Manolo Garcia`; extracted claim preserves the full name.
- User states `25:50`; extracted claim preserves `25:50`.
- User states `Business Administration`; extracted claim contains the full phrase.

Pass/fail: pass if the exact scalar/name phrase appears in at least one
extracted active claim and can be matched by the strict short-answer scorer.
Fail if the fact is omitted, rounded, generalized, or replaced by a different
value.

Minimal fix family: extractor prompt hardening for verbatim preservation of
short scalar facts and named entities.

## 2. event_slot_assembly

Contract: slot-bearing facts spread across a user goal, event, place, tool, or
preference must be assembled into evidence that answers the asked slot without
dropping the slot value.

Cases: `06878be2`, `0edc2aef`, `35a27287`, `8a2466db`

Synthetic invariant tests:
- Preference plus brand must retrieve brand-specific recommendations.
- Trip city plus hotel feature must keep both city and feature.
- Cultural-event interest plus languages must keep event type and languages.
- Tool interest plus product name must keep the product-specific resource.
- User says `near Miami` and later `rooftop pool`; both slots remain available.
- User says `Spanish and French`; both language slots remain present.

Pass/fail: pass if final evidence contains the relevant slot values needed by
the question. Fail if evidence contains adjacent topic facts but loses the slot
asked for.

Minimal fix family: event/slot bundling only.

## 3. composite_answer_synthesis

Contract: when the answer is a count, list, or aggregate derived from multiple
claims, the system must preserve all contributing items and synthesize the
requested aggregate.

Cases: `6aeb4375`, `0a995998`, `6d550036`, `b5ef892d`

Synthetic invariant tests:
- Three clothing pickup/return claims produce answer `3`.
- Two led-project claims produce answer `2`.
- Three restaurant-visit claims plus one update produce the current count.
- Two camping trips of 3 and 5 days produce `8 days`.
- Five model-kit claims produce a list of five kits.
- Superseded non-current items do not inflate the count.

Pass/fail: pass if all contributing items are in evidence and the answer
matches the required aggregate. Fail if one item is omitted, double-counted, or
ignored by the reader.

Minimal fix family: composite-answer synthesis only.

## 4. temporal_update_reasoning

Contract: temporal order, first/last status, and updated truth must be resolved
from dated or update-linked claims.

Cases: `gpt4_2655b836`, `gpt4_2487a7cb`, `gpt4_76048e76`, `gpt4_2312f94c`

Synthetic invariant tests:
- Earlier service issue beats later service issue for a first-event question.
- Earlier workshop/webinar date determines which came first.
- Vehicle maintenance events in February are ordered correctly.
- Device acquisition order is preserved.
- Later correction supersedes older value for current-truth questions.
- Historical questions can still use superseded values when asked.

Pass/fail: pass if the selected answer follows the requested temporal/update
semantics. Fail if the system picks a later event, current value for a history
question, or stale value for a current-truth question.

Minimal fix family: temporal/update reasoning only.

## 5. reader_under_clean_evidence

Contract: when clean direct evidence is already present, the reader must answer
from it without abstaining, hallucinating a contradiction, or being over-credited
by the scorer.

Cases: `75832dbd`, `352ab8bd`, `6a1eabeb`, `gpt4_59c863d7`

Synthetic invariant tests:
- Direct evidence about AI healthcare papers yields a relevant recommendation.
- Direct evidence says `20%`; reader includes `20%`.
- Direct evidence says `25:50`; reader includes `25:50`.
- Direct evidence lists five model kits; reader lists five.
- Clean evidence with no answer causes `I don't know`.
- Wrong scalar in reader output fails strict scoring.

Pass/fail: pass if reader output is entailed by clean evidence and passes the
scorer. Fail if reader abstains despite direct evidence or outputs a wrong
scalar/list.

Minimal fix family: reader-only prompt or answer formatting fix.
