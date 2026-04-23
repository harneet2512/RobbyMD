# Worker 2 — Substrate Core Schema (feature/substrate-core)

**Branch**: `feature/substrate-core`
**Status**: complete, 91/91 tests passing.
**Executed by**: orchestrator main thread (sub-agent stalled in plan mode).

## Files changed

| File | Change |
|------|--------|
| `src/substrate/schema.py` | Added `valid_from_ts INTEGER`, `valid_until_ts INTEGER` to `claims` DDL with `CHECK (valid_until_ts IS NULL OR valid_from_ts IS NULL OR valid_until_ts > valid_from_ts)`. Added `idx_claims_temporal` composite index. Extended `Claim` dataclass with two new optional fields (defaulted last for positional-arg compatibility). Citation block referencing Zep (arXiv:2501.13956) and Chronos (arXiv:2603.16862) embedded near the new columns. |
| `src/substrate/claims.py` | `row_to_claim()` hydrates new fields with `keys()`-based fallback for narrower SELECTs. `insert_claim()` accepts `valid_from`, `valid_until` kwargs; defaults `valid_from = created_ts`, `valid_until = None`. New `get_superseded_by(conn, claim_id) -> str | None` derived accessor over `supersession_edges` (NOT a stored field — per `architecture_validation.md` §4 design call). |
| `src/substrate/supersession.py` | Pass-1 `detect_pass1` now UPDATEs the superseded claim's `valid_until_ts` to `edge.created_ts` (single timestamp, two records — keeps the supersession event atomic). Switched `write_supersession_edge` from raw `time_ns()` to substrate's monotonic `now_ns()` to fix a pre-existing latent determinism bug. |
| `tests/property/test_temporal_validity.py` | NEW. Six tests: defaults, CHECK invariant (raw SQL), `insert_claim` rejection of inverted window, supersession-sets-valid_until, derived `get_superseded_by`, run-twice structural determinism. |

## Schema spec for downstream workers

**Workers 3, 4, 5 must build against these contracts.**

### `Claim` dataclass (src/substrate/schema.py:117–145)

```python
@dataclass(frozen=True, slots=True)
class Claim:
    claim_id: str
    session_id: str
    subject: str
    predicate: str
    value: str
    value_normalised: str | None
    confidence: float
    source_turn_id: str
    status: ClaimStatus
    created_ts: int
    char_start: int | None = None
    char_end: int | None = None
    valid_from_ts: int | None = None      # NEW
    valid_until_ts: int | None = None     # NEW
```

### `claims` SQL columns (DDL added, see schema.py:183–207)

```sql
valid_from_ts     INTEGER,        -- nullable; defaults to created_ts at insert time
valid_until_ts    INTEGER,        -- nullable; supersession sets this to edge.created_ts
CHECK (valid_until_ts IS NULL
       OR valid_from_ts IS NULL
       OR valid_until_ts > valid_from_ts)
-- Composite index for temporal-window queries:
CREATE INDEX idx_claims_temporal ON claims(valid_from_ts, valid_until_ts);
```

### `insert_claim()` new kwargs (src/substrate/claims.py:209–304)

```python
def insert_claim(
    conn,
    *,
    # ... existing kwargs ...
    valid_from: int | None = None,    # NEW; defaults to created_ts (now_ns())
    valid_until: int | None = None,   # NEW; defaults to None (unbounded)
) -> Claim
```

Validation: if `valid_until is not None and valid_until <= valid_from`, raises `ClaimValidationError`.

### Derived `get_superseded_by()` (src/substrate/claims.py)

```python
def get_superseded_by(conn: sqlite3.Connection, claim_id: str) -> str | None:
    """Most recent supersession edge's new_claim_id where this claim is the
    old_claim_id. None if not superseded. Deterministic tie-break by
    (created_ts DESC, edge_id DESC)."""
```

### When `valid_until_ts` becomes non-NULL

1. **Pass-1 supersession** (`src/substrate/supersession.py::detect_pass1`) — sets `valid_until_ts = edge.created_ts` on the superseded claim, atomically with `status → SUPERSEDED`. Guarded by `valid_from_ts < edge.created_ts` so a degenerate clock doesn't break the CHECK.
2. **Future workers may set it explicitly** via `insert_claim(..., valid_until=...)` — e.g. when extracting a claim with a known end date ("patient was on metformin from 2020 to 2023").

### Gotchas for downstream workers

- **Existing claims pre-migration** would have `NULL` for both columns. The repo doesn't yet have any persisted state, so this is theoretical, but if you query for "claims valid at time T", treat NULL `valid_from_ts` as "valid since the beginning of time" and NULL `valid_until_ts` as "valid until forever".
- **The `_last_ts` monotonic counter** (claims.py:48–58) is process-local. Tests that span processes can't assume timestamp ordering across them.
- **`write_supersession_edge` now uses `now_ns()` not `time_ns()`** — Worker 4/5 building on top should not re-introduce raw `time_ns()` for substrate writes; use `now_ns()` from `src.substrate.claims` for monotonic ordering.

## Test results

```
tests/property/test_temporal_validity.py ......        [ 6/91]
tests/property/test_determinism.py ...                 [ 9/91]
tests/unit/substrate/                                   [82/91]
============================= 91 passed in 0.46s ==============================
```

## Commit

To follow this artifact write — see `git log -1 feature/substrate-core` after commit lands.
