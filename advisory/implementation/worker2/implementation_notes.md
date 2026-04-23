# Worker 2 — Implementation Notes

## Decisions

### `superseded_by` as derived accessor, not stored field
Per `architecture_validation.md` §4: `supersession_edges` is canonical; `superseded_by` is a derived view. Stored field would (a) duplicate state with the edges table, (b) require careful sync on every supersession event, (c) make the data model lossier (storing only the latest pointer loses chain history that the edge table preserves). Derived accessor `get_superseded_by()` lives in `claims.py` (module cohesion with other read-side helpers).

### `write_supersession_edge` clock switch (`time_ns` → `now_ns`)
Latent pre-existing bug, surfaced by the new property test. The substrate's `now_ns()` is monotonically bumped (claims.py:48–58) to handle Windows' ~1 ms clock resolution. `write_supersession_edge` was using raw `time_ns()` from stdlib. After many `now_ns()` calls, `_last_ts` runs ahead of wall clock. The supersession edge would then get a `created_ts < old_claim.valid_from_ts`, and my temporal-validity update guard (`WHERE valid_from_ts < edge.created_ts`) would silently skip — non-deterministically across runs depending on warm-up state. Fix: switch the edge writer to `now_ns()` so all substrate timestamps share one monotonic clock. Existing tests still green; a determinism issue that wasn't being detected is now closed.

### Default `valid_from = created_ts`
A claim is valid from the moment it enters the substrate, by default. Downstream workers can override (e.g. event-tuple projection might extract historical validity windows from claim values: "patient moved to Boston in 2023" → `valid_from = 2023-01-01 epoch ns`). Keeping the default semantically meaningful (rather than NULL) means temporal queries don't need to special-case missing `valid_from`.

### Single CHECK constraint with two NULL escapes
```
CHECK (valid_until_ts IS NULL OR valid_from_ts IS NULL OR valid_until_ts > valid_from_ts)
```
Allows: both NULL (legacy/migration), only valid_from set (default — unbounded), both set (window). Disallows: valid_until set with no valid_from, or window inverted. The schema-level CHECK is verified by `test_valid_until_after_valid_from_invariant` (raw SQL insert).

## Surprises

1. **Test fixture safety**: existing `tests/unit/substrate/test_schema.py:55–75` does raw-SQL claim INSERTs that omit the new columns. Worked fine because they're nullable. Documented as the right migration path: nullable adds first, populate via writes second, never backfill via DDL.
2. **Pyright stale-cache noise**: after the `time_ns → now_ns` switch, Pyright reported `time_ns is not defined` for several seconds before refreshing. False alarm. Real test run confirmed clean.
3. **`row_to_claim` `.get()` workaround**: `sqlite3.Row` has no `.get()` method. Used `keys()` set membership check to detect column presence so callers passing narrower SELECTs (Worker 3's event-tuple projection might) don't trip a KeyError.

## Deviations from spec

- **None material.** The spec said "row_to_claim is in schema.py" — actually in claims.py. No structural change.
- **Skipped: `architecture_validation.md` §4 "Add: `tests/property/test_projection_invariants.py`"** — out of Worker 2 scope. Worker 3 (event tuples) should add it, since projection-invariant tests need the projection to exist.

## Citation pattern

In-code citation block embedded near the new schema columns (schema.py:198–202) and in the `Claim` docstring (schema.py:127–135). Per `cursor.md` Rule 1 not yet (rule is for file mirroring, not citations) but per `CLAUDE.md §8` ("Comments: why, not what. Reference PRD/Eng_doc section when implementing a spec'd behaviour"). The Zep / Chronos citations honor `architecture_validation.md` §6 ("Cite ancestors in every deliverable").

## Hand-off to Workers 3, 4, 5

Schema spec is in `worker_progress.md` — please read before implementing. Key contracts to NOT change:
- `Claim` field order (positional safety)
- `insert_claim` kwarg names (`valid_from`, `valid_until` — note: kwarg uses int seconds-equivalent ns; the column suffix `_ts` is internal naming)
- The `get_superseded_by()` accessor signature and return type
- `now_ns()` as the canonical substrate clock — do not introduce raw `time_ns()` calls in substrate writes
