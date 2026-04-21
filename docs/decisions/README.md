# Architecture Decision Records

This directory captures decisions that deviate from `PRD.md` / `Eng_doc.md` / `rules.md`, or that resolve an open question those docs flag.

## Format

One file per decision. Filename: `YYYY-MM-DD_<short-topic>.md`. Template:

```markdown
# <YYYY-MM-DD> — <topic>

**Status**: proposed / accepted / superseded-by-<later ADR>
**Driver**: <name>
**Affected**: <module / spec section>

## Context
<Why the question came up. What was the constraint or trigger.>

## Decision
<What we chose.>

## Alternatives considered
<List — including 'do nothing' if that was an option.>

## Consequences
<What this makes easier / harder. What's now true about the system.>

## References
<Links to PRD/Eng_doc/rules sections, papers, prior ADRs.>
```

## Rules

- Per `CLAUDE.md §11` and `rules.md §10.2`: any deviation from a planning doc gets an ADR here. No silent deviations.
- Per `CLAUDE.md §4`: worktree agents write proposed ADRs but do not mark them accepted — the human operator drives the decision.
