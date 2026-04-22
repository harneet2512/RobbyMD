# predicate_packs/personal_assistant

Second seeded pack (commit `refactor: drop DDXPlus+MedQA ... seed personal_assistant pack ...`, 2026-04-21).

Scope: personal-assistant domain for the **LongMemEval-S** benchmark substrate variant. The pack covers the closed predicate vocabulary an assistant-memory agent needs to structure user-asserted facts, preferences, events, relationships, goals, and constraints across multi-session conversations.

## Predicate families (closed vocabulary)

| Family | Meaning |
|---|---|
| `user_fact` | Durable, verifiable facts the user asserts about themselves (occupation, location, history, demographics). |
| `user_preference` | Expressed likes / dislikes / aesthetic choices (food, music, travel, working style). |
| `user_event` | Discrete scheduled or past events (trips, appointments, deadlines, milestones). |
| `user_relationship` | Relations to other people (family, colleagues, friends) — includes names + roles. |
| `user_goal` | Active ambitions or multi-session efforts (learning a skill, fitness, career). |
| `user_constraint` | Limitations or hard requirements (dietary, scheduling, accessibility). |

## Why no sub-slots

Personal-assistant predicates are natural-language noun phrases. Sub-slot decomposition (à la `medication.{name, dose, route, ...}`) adds complexity without benefit for LongMemEval-S, which grades end-to-end QA accuracy, not structured-field extraction. If a future benchmark requires structured sub-slots (e.g., ICal-style event-date parsing), add them in a pack revision.

## Why no LR table

Personal-assistant has no differential hypotheses to rank. The `differentials/` subdir is absent by design; the differential engine reads `active_pack().lr_table_path` and no-ops when it is `None` (see `src/differential/lr_table.py` post-refactor).

## Few-shot examples

Six hand-authored 2-turn conversation snippets + expected claim outputs in `few_shot_examples.json`. Fresh work authored during this commit per `rules.md §1.1` — **no content copied from the LongMemEval dataset**. Examples demonstrate the same extraction patterns clinical_general teaches (multi-claim utterance, negation, self-correction / supersession, rare value, ambiguous phrasing, structured event).

## Usage

```bash
# Activate this pack (tests, benchmark harness):
ACTIVE_PACK=personal_assistant python -c "from src.substrate.predicate_packs import active_pack; print(active_pack().pack_id)"
```

Loaded automatically by `eval/smoke/run_smoke.py --benchmark longmemeval` and `eval/longmemeval/run.py`.

## Pack-registration schema

Same `PredicatePack` dataclass as `clinical_general`. See `Eng_doc.md §4.2` for the full schema.
