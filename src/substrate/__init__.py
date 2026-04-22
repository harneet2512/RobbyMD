"""Clinical reasoning substrate — claim lifecycle, supersession, projections.

Scope (wt-engine worktree, see `CLAUDE.md` §5.1): fresh ~560 LOC implementation
of the minimum substrate defined in `Eng_doc.md` §2.1 with upgrades from
`docs/research_brief.md` §2.1 (typed supersession edges) and span-based
provenance from `docs/gt_v2_study_notes.md` §2.1.

Public API:

- `Claim`, `Turn`, `SupersessionEdge`, `NoteSentence`, `EdgeType`, `ClaimStatus`,
  `Speaker`, `NoteSection`, `PREDICATE_FAMILIES` — the typed data model.
- `open_database(path)` — initialise a SQLite connection with the required
  PRAGMAs and schema (idempotent).
- `claims.insert_claim`, `claims.list_active_claims`, `claims.get_claim`,
  `claims.set_claim_status`, etc. — CRUD.
- `supersession.detect_pass1(...)` — deterministic structural supersession.
- `supersession_semantic.SemanticSupersession` — Pass 2 with pluggable
  embedder (defaults to `NullEmbedder` so tests do not require a heavy
  model install).
- `projections.rebuild_active_projection`, `projections.per_branch_projection`.
- `admission.admit(...)` — noise-regex-only admission filter.
- `provenance.*` — forward/back-link utilities for the UI.
- `event_bus.EventBus` — simple synchronous in-memory pub/sub.
- `on_new_turn.on_new_turn(...)` — orchestrator entry point.
"""
from __future__ import annotations

from src.substrate.schema import (
    PREDICATE_FAMILIES,
    Claim,
    ClaimStatus,
    EdgeType,
    NoteSection,
    NoteSentence,
    Speaker,
    SupersessionEdge,
    Turn,
    open_database,
)

__all__ = [
    "PREDICATE_FAMILIES",
    "Claim",
    "ClaimStatus",
    "EdgeType",
    "NoteSection",
    "NoteSentence",
    "Speaker",
    "SupersessionEdge",
    "Turn",
    "open_database",
]
