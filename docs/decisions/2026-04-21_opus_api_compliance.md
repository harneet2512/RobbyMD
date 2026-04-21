# 2026-04-21 — Claude Opus 4.7 via Anthropic API satisfies the open-source rule

**Status**: accepted
**Driver**: hack_it operator
**Affected**: `rules.md §1.2`, `tests/licensing/test_open_source.py`, `context.md §3`

## Context

Cerebral Valley × Anthropic's "Built with Opus 4.7" hackathon requires every demo component to be **OSI-approved open source** (context.md §3, verbatim rule: *"Everything shown in the demo must be fully open source. This includes every component — backend, frontend, models, and any other parts of the project — published under an approved open source license."*). Claude Opus 4.7 is a proprietary frontier model accessed via the Anthropic API under Anthropic's standard terms — not an OSI-approved licence in the traditional sense.

A naive reading would disqualify our entire stack.

## Decision

Claude Opus 4.7 via the Anthropic API is the hackathon's **named sponsored tool**. The event title is literally "Built with Opus 4.7"; the prize pool rewards Opus-4.7-specific features (25% of judging on "Opus 4.7 Use"); every participating team is expected to use it. It is not a third-party proprietary component shipped *alongside* our code — it is the sponsored infrastructure, analogous to GitHub, Vercel, AWS, or the hackathon platform itself.

All *other* dependencies in our repo (backend libs, frontend libs, ASR models, embedding models, clinical content) must remain OSI-approved: MIT, Apache-2.0, BSD (2/3/4-clause), MPL-2.0, ISC, LGPL. This is enforced by `tests/licensing/test_open_source.py` which explicitly lists `anthropic` in `SPONSORED_TOOL_EXEMPT = {"anthropic"}` and fails on any other non-OSI dependency.

Rules.md §1.2 codifies this exemption:
> *"Exception: Claude Opus 4.7 via the Anthropic API is the hackathon's named sponsored tool — the event is Built with Opus 4.7. It is available to every team. It is not considered a third-party proprietary component shipped with the project."*

## Alternatives considered

- **Strict interpretation — open-weight model only**: would force us to drop Opus 4.7 entirely, abandon the 25% Opus-4.7-Use judging axis, and require fine-tuning a local medical LLM in 5.5 days — infeasible and self-defeating given the event's framing. Rejected.
- **Hybrid stack with open-weight parity**: BioMistral-7B (Apache-2.0) is already spec'd as an offline rehearsal fallback (Eng_doc.md §3.2). Good belt-and-braces, but not a replacement for Opus 4.7 on the judging axis. Kept as fallback.
- **Post to Discord #questions for official confirmation**: drafted below as belt-and-braces; posting is optional. ADR is self-standing.

## Consequences

- `anthropic` SDK is flagged `SPONSORED_TOOL_EXEMPT` in the licensing test.
- Every non-Anthropic dependency must pass the OSI check (enforced in CI via `pytest tests/licensing/test_open_source.py`).
- If a judge raises the question, the canonical answer is context.md §3 + this ADR. No Discord confirmation required.
- No dependency on event organisers' reply time.

## Optional Discord post (draft; sending is not a gate)

> Hi — quick compliance check. Does using Claude Opus 4.7 via the Anthropic API satisfy the "everything open-source" rule given the event is "Built with Opus 4.7"? We're treating it as the sponsored tool (same category as GitHub / Vercel / the hackathon platform) with every other dependency OSI-approved (MIT / Apache-2.0 / BSD / etc.). Please confirm. Thanks!

## References

- `rules.md §1.2` — OSI-approved open source dependencies only (with sponsored exception)
- `context.md §3` — hackathon rules we must satisfy (verbatim quote)
- `tests/licensing/test_open_source.py` — `SPONSORED_TOOL_EXEMPT = {"anthropic"}`
- Event page: *Built with Opus 4.7: a Claude Code Hackathon* (Cerebral Valley × Anthropic, submission 2026-04-26)
