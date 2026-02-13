# Store & Store-Derive Crates — Comprehensive Testing Plan

Testing plan for `store/` and `store-derive/` achieving high parity with Sui's `crates/typed-store/` and `crates/typed-store-derive/`. Covers file-by-file mapping, attribution requirements, test infrastructure, and every test needed for parity plus Soma-specific coverage.

**Sui reference**: `MystenLabs/sui` — `crates/typed-store/` and `crates/typed-store-derive/`
**Soma crates**: `store/` and `store-derive/`

---

## Audit Notes (Feb 2026)

**Priority Ranking**: #7 of 7 plans — lowest priority because it's already at near-parity with Sui. The store crate is well-tested.

**Accuracy Correction**: The executive summary claims **27 total test functions**, but the breakdown shows **24 rocks tests + 3 macro tests + 1 util test = 28**. However, verification against the codebase found **24 actual unit tests** in `rocks/tests.rs`, which matches Sui's 24 exactly. The 3 macro integration tests and 1 util test are separate. The discrepancy of "27" in the summary vs "28" in the breakdown is a minor arithmetic error.

**Corrected total: 28 existing tests (24 rocks + 3 macro + 1 util), not 27.**

**Key Concerns**:
1. **Already at near-parity** — 24/24 rocks tests match Sui exactly. This is the best-tested crate.
2. **2 missing Sui macro tests are intentionally omitted** — `test_sampling` and `test_sampling_time` depend on Prometheus metrics infrastructure that Soma removed. This is correctly documented as an intentional gap.
3. **InMemoryDB tests (Priority 3) are the highest value addition** — 10 new tests for the in-memory backend ensure it behaves identically to RocksDB for test infrastructure correctness.
4. **Low urgency** — this plan can be deferred until after consensus, authority, and types testing are addressed.

**Estimated Effort**: ~4 engineering days as planned.

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [File-by-File Cross-Reference](#file-by-file-cross-reference)
3. [Attribution Requirements](#attribution-requirements)
4. [Feature Gap Analysis](#feature-gap-analysis)
5. [Test Infrastructure](#test-infrastructure)
6. [Priority 1: Existing Test Verification (27 tests)](#priority-1-existing-test-verification)
7. [Priority 2: Missing Sui Parity Tests](#priority-2-missing-sui-parity-tests)
8. [Priority 3: InMemoryDB Tests](#priority-3-inmemorydb-tests)
9. [Priority 4: Options & Configuration Tests](#priority-4-options--configuration-tests)
10. [Priority 5: Error Handling Tests](#priority-5-error-handling-tests)
11. [Priority 6: SafeIter / SafeRevIter Tests](#priority-6-safeiter--safereviter-tests)
12. [Priority 7: Derive Macro Tests](#priority-7-derive-macro-tests)
13. [Priority 8: Soma-Specific Tests](#priority-8-soma-specific-tests)
14. [Implementation Order](#implementation-order)
15. [Build & Run Commands](#build--run-commands)

---

## Executive Summary

### Current State
- **27 total test functions** across the store crate
- 24 unit tests in `store/src/rocks/tests.rs` — all passing, **exact parity with Sui's 24 rocks tests**
- 3 integration tests in `store/tests/macro_tests.rs` — all passing (macro_test, rename_test, deprecate_test)
- 1 unit test in `store/src/util.rs` — `test_helpers` (big-endian arithmetic helpers)
- **0 tests** for `InMemoryDB`, `SafeIter`/`SafeRevIter`, `options.rs`, `error.rs`, `rocks_util.rs`
- No metrics module (intentionally removed from Sui's version)

### Target State
- **~55-65 total tests** covering all store modules
- Maintain existing 24 rocks test parity with Sui
- Add missing Sui macro tests where applicable (2 tests are metrics-dependent — document gap)
- Add InMemoryDB test coverage (new — Sui has no dedicated tests either, but we should)
- Add options validation tests
- Add SafeIter edge case tests
- Add derive macro edge case tests
- Complete attribution headers on all derived files

### Test Count Summary

| Category | Sui Tests | Soma Existing | Gap | Notes |
|----------|-----------|---------------|-----|-------|
| rocks/tests.rs (unit) | 24 | 24 | 0 | Full parity achieved |
| macro_tests.rs (integration) | 5 | 3 | 2 | `test_sampling` + `test_sampling_time` are metrics-dependent |
| util.rs (unit) | 1 | 1 | 0 | `test_helpers` exists in both |
| InMemoryDB tests | 0 | 0 | ~10 new | Soma-specific — good hygiene |
| Options tests | 0 | 0 | ~5 new | Validate configuration methods |
| SafeIter tests | 0 | 0 | ~4 new | Edge cases for iterator safety |
| Error tests | 0 | 0 | ~2 new | Error conversion coverage |
| Derive macro edge cases | 0 | 0 | ~3 new | Compile-time validation |
| nondeterministic! macro | 0 | 0 | 1 new | msim-specific behavior |
| **Total** | **30** | **28** | **~27 new** |

---

## File-by-File Cross-Reference

### Legend
- **Heavy** = Direct port/fork, needs full attribution
- **Moderate** = Significant shared patterns, needs attribution
- **Light** = Minor similarity only
- **Soma-only** = Original Soma code, no attribution needed

### store/ crate

| Soma File | Sui File | Derivation | Lines (Soma) | Inline Tests | Notes |
|-----------|----------|------------|-------------|--------------|-------|
| `store/src/lib.rs` | `typed-store/src/lib.rs` | Heavy | 14 | 0 | Soma removed: metrics module, `StorageType` enum, `MetricConf` re-export |
| `store/src/rocks/mod.rs` | `typed-store/src/rocks/mod.rs` | Heavy | 1031 | 0 (24 in submodule) | Core RocksDB wrapper. Soma removed: metrics fields, TideHunter, StagedBatch, fail_point macros, MetricConf params, write_sync |
| `store/src/rocks/tests.rs` | `typed-store/src/rocks/tests.rs` | Heavy | 582 | 24 | Exact parity with Sui's 24 tests. Helper fns differ (no MetricConf) |
| `store/src/rocks/errors.rs` | `typed-store/src/rocks/errors.rs` | Heavy | ~20 | 0 | Nearly identical error conversions |
| `store/src/rocks/options.rs` | `typed-store/src/rocks/options.rs` | Heavy | 353 | 0 | Soma has all key methods. Sui has additional: `optimize_for_no_deletion` |
| `store/src/rocks/safe_iter.rs` | `typed-store/src/rocks/safe_iter.rs` | Heavy | 92 | 0 | Soma removed: metrics (timer, perf_ctx, bytes_scanned, keys_scanned, db_metrics) |
| `store/src/rocks/rocks_util.rs` | N/A (inline in Sui) | Moderate | 14 | 0 | `apply_range_bounds` helper |
| `store/src/traits.rs` | `typed-store/src/traits.rs` | Heavy | 98 | 0 | Nearly identical. Soma commented out hdrhistogram fields in `TableSummary` |
| `store/src/memstore.rs` | `typed-store/src/memstore.rs` | Heavy | 119 | 0 | Nearly identical InMemoryDB implementation |
| `store/src/error.rs` | `typed-store/src/rocks/errors.rs` (partial) | Heavy | 17 | 0 | `TypedStoreError` enum |
| `store/src/util.rs` | `typed-store/src/util.rs` | Heavy | 121 | 1 | Sui has additional `be_fix_int_ser_into` and `ensure_database_type` functions |
| `store/tests/macro_tests.rs` | `typed-store/tests/macro_tests.rs` | Heavy | 216 | 3 | Missing: `test_sampling`, `test_sampling_time` (metrics-dependent) |

### store-derive/ crate

| Soma File | Sui File | Derivation | Lines (Soma) | Notes |
|-----------|----------|------------|-------------|-------|
| `store-derive/src/lib.rs` | `typed-store-derive/src/lib.rs` | Heavy | 446 | Soma removed: TideHunter storage backend support. Otherwise nearly identical proc macro generating `open_tables_read_write`, `get_read_only_handle`, `dump`, `describe_tables` |

---

## Attribution Requirements

All files below are derived from Sui's `typed-store` / `typed-store-derive` crates and **require attribution** with the Apache 2.0 / Mysten Labs copyright header.

### Files Requiring Attribution

| File | Sui Source | Derivation Level |
|------|-----------|-----------------|
| `store/src/lib.rs` | `crates/typed-store/src/lib.rs` | Heavy — simplified entry point |
| `store/src/rocks/mod.rs` | `crates/typed-store/src/rocks/mod.rs` | Heavy — core DB wrapper, metrics stripped |
| `store/src/rocks/tests.rs` | `crates/typed-store/src/rocks/tests.rs` | Heavy — 24 tests match exactly |
| `store/src/rocks/errors.rs` | `crates/typed-store/src/rocks/errors.rs` | Heavy — near-identical |
| `store/src/rocks/options.rs` | `crates/typed-store/src/rocks/options.rs` | Heavy — shared RocksDB tuning options |
| `store/src/rocks/safe_iter.rs` | `crates/typed-store/src/rocks/safe_iter.rs` | Heavy — metrics stripped |
| `store/src/rocks/rocks_util.rs` | `crates/typed-store/src/rocks/mod.rs` (inline) | Moderate — small helper extracted |
| `store/src/traits.rs` | `crates/typed-store/src/traits.rs` | Heavy — Map trait + TableSummary |
| `store/src/memstore.rs` | `crates/typed-store/src/memstore.rs` | Heavy — near-identical InMemoryDB |
| `store/src/error.rs` | `crates/typed-store/src/rocks/errors.rs` | Heavy — TypedStoreError enum |
| `store/src/util.rs` | `crates/typed-store/src/util.rs` | Heavy — serialization utilities |
| `store/tests/macro_tests.rs` | `crates/typed-store/tests/macro_tests.rs` | Heavy — 3 of 5 tests match |
| `store-derive/src/lib.rs` | `crates/typed-store-derive/src/lib.rs` | Heavy — proc macro, TideHunter stripped |

### Recommended Attribution Header

```rust
// Copyright (c) Mysten Labs, Inc.
// Portions Copyright (c) Soma Foundation
// SPDX-License-Identifier: Apache-2.0
```

---

## Feature Gap Analysis

Features present in Sui's `typed-store` that were **intentionally removed** from Soma's `store`:

| Feature | Sui Location | Soma Status | Impact on Testing |
|---------|-------------|-------------|-------------------|
| **Metrics module** (`DBMetrics`, `SamplingInterval`, `RocksDBPerfContext`, Prometheus integration) | `typed-store/src/metrics.rs` + fields in `DBMap`, `DBBatch`, `Database` | Removed entirely | `test_sampling` and `test_sampling_time` macro tests cannot be ported. No action needed. |
| **TideHunter storage backend** | Gated behind `cfg(msim)` in Sui | Removed | TideHunter-specific macro tests in Sui are not applicable. No action needed. |
| **StagedBatch** | `typed-store/src/rocks/mod.rs` | Removed | No tests to port (Sui has no dedicated StagedBatch tests). |
| **MetricConf parameter** | Throughout `open_cf_opts`, `Database::new`, `DBMap::reopen` | Removed | Test helpers already adapted (no MetricConf). No action needed. |
| **write_sync / `SUI_DB_SYNC_TO_DISK`** | `typed-store/src/rocks/mod.rs` | Removed | No tests to port. |
| **fail_point macros in DB methods** | `delete_cf`, `put_cf`, `batch-write` | Removed | Could add Soma-specific failpoints later if needed. Not critical for test parity. |
| **`ensure_database_type`** | `typed-store/src/util.rs` | Removed | Small utility for checking DB type on disk. Not needed without TideHunter. |
| **`be_fix_int_ser_into`** | `typed-store/src/util.rs` | Removed | Optimization variant. Not needed. |
| **`table_summary` method** | `typed-store/src/rocks/mod.rs` | Removed | Uses hdrhistogram (commented out in `TableSummary`). |
| **`optimize_for_no_deletion`** | `typed-store/src/rocks/options.rs` | Removed | Minor optimization helper. |

**Summary**: All removals are intentional simplifications. The 2 missing macro tests (`test_sampling`, `test_sampling_time`) directly test Prometheus metrics collection which Soma does not have. These are documented gaps, not bugs.

---

## Test Infrastructure

### Existing Test Files

| File | Framework | Tests | Status |
|------|-----------|-------|--------|
| `store/src/rocks/tests.rs` | `#[tokio::test]` | 24 | All passing |
| `store/tests/macro_tests.rs` | `#[tokio::test]` | 3 | All passing |
| `store/src/util.rs` | `#[test]` (inline) | 1 | Passing |

### Helper Functions (store/src/rocks/tests.rs)

```rust
fn open_map(path: PathBuf, opt_cf: Option<&str>) -> DBMap<String, String>
fn open_rocksdb(path: &Path, opt_cfs: &[&str]) -> Arc<Database>
```

Both are simplified from Sui's versions (no `MetricConf` parameter).

### Test Dependencies

- `tempfile` — temporary directories for test databases
- `tokio` — async test runtime
- `serde`, `serde_json`, `bcs` — serialization in tests
- `bincode` — for `be_fix_int_ser` utility
- `uint` — for big-endian arithmetic tests in `util.rs`

---

## Priority 1: Existing Test Verification

**Goal**: Verify all 28 existing tests pass and document their coverage.

### rocks/tests.rs — 24 Tests (Full Sui Parity)

| # | Test Name | Sui Equivalent | What It Tests |
|---|-----------|---------------|---------------|
| 1 | `test_open` | `test_open` | Open DB, insert, read back, verify |
| 2 | `test_reopen` | `test_reopen` | Open DB, insert, close, reopen, verify persistence |
| 3 | `test_contains_key` | `test_contains_key` | `contains_key` for existing and non-existing keys |
| 4 | `test_safe_drop_db` | `test_safe_drop_db` | `safe_drop_db` destroys DB on disk |
| 5 | `test_multi_contain` | `test_multi_contain` | `multi_contains_keys` bulk check |
| 6 | `test_get` | `test_get` | `get` returns `Some`/`None` correctly |
| 7 | `test_multi_get` | `test_multi_get` | `multi_get` for multiple keys at once |
| 8 | `test_skip` | `test_skip` | Iterator `skip` semantics |
| 9 | `test_reverse_iter_with_bounds` | `test_reverse_iter_with_bounds` | Reverse iteration with lower/upper bounds |
| 10 | `test_remove` | `test_remove` | `remove` deletes key, `get` returns `None` |
| 11 | `test_iter` | `test_iter` | Forward iteration over all entries |
| 12 | `test_iter_reverse` | `test_iter_reverse` | Reverse iteration over all entries |
| 13 | `test_insert_batch` | `test_insert_batch` | `DBBatch` insert + write atomicity |
| 14 | `test_insert_batch_across_cf` | `test_insert_batch_across_cf` | Batch across column families in same DB |
| 15 | `test_insert_batch_across_different_db` | `test_insert_batch_across_different_db` | Batch rejects cross-DB operations |
| 16 | `test_delete_batch` | `test_delete_batch` | Batch delete operations |
| 17 | `test_delete_range` | `test_delete_range` | `schedule_delete_all` removes all entries |
| 18 | `test_iter_with_bounds` | `test_iter_with_bounds` | Bounded forward iteration |
| 19 | `test_range_iter` | `test_range_iter` | Range-based iteration (inclusive/exclusive bounds) |
| 20 | `test_is_empty` | `test_is_empty` | `is_empty` before and after inserts |
| 21 | `test_multi_insert` | `test_multi_insert` | `multi_insert` bulk writes |
| 22 | `test_checkpoint` | `test_checkpoint` | RocksDB checkpoint creation + read from checkpoint |
| 23 | `test_multi_remove` | `test_multi_remove` | `multi_remove` bulk deletes |
| 24 | `open_as_secondary_test` | `open_as_secondary_test` | Secondary (read-only) DB access + `try_catch_up_with_primary` |

### macro_tests.rs — 3 Tests (Partial Sui Parity)

| # | Test Name | Sui Equivalent | What It Tests |
|---|-----------|---------------|---------------|
| 1 | `macro_test` | `macro_test` | `DBMapUtils` derive: open, insert, secondary read, dump, pagination |
| 2 | `rename_test` | `rename_test` | `#[rename = "..."]` attribute on column families |
| 3 | `deprecate_test` | `deprecate_test` | `#[deprecated]` attribute + `open_tables_read_write_with_deprecation_option` |

### util.rs — 1 Test

| # | Test Name | Sui Equivalent | What It Tests |
|---|-----------|---------------|---------------|
| 1 | `test_helpers` | `test_helpers` | `big_endian_saturating_add_one`, `is_max` edge cases |

**Action**: Run `cargo test -p store` and verify all 28 tests pass. Fix any failures before proceeding.

---

## Priority 2: Missing Sui Parity Tests

### 2A: Sui macro tests NOT portable (metrics-dependent)

These 2 Sui tests **cannot be ported** because they depend on the metrics infrastructure that Soma intentionally removed:

| Sui Test | Reason Not Portable |
|----------|-------------------|
| `test_sampling` | Tests `SamplingInterval` and `DBMetrics` — writes records and verifies Prometheus metric collection via `read_size_from_env("SAMPLING_INTERVAL")` |
| `test_sampling_time` | Tests time-based sampling with `Duration` intervals and metric counters |

**Action**: Document as intentional gap. No implementation needed.

### 2B: Sui util.rs functions NOT ported

Sui's `util.rs` contains additional functions not present in Soma:

| Function | Purpose | Action |
|----------|---------|--------|
| `be_fix_int_ser_into` | Optimization: serializes directly into existing buffer | Not needed — `be_fix_int_ser` covers all use cases |
| `ensure_database_type` | Validates on-disk DB type matches expected (RocksDB vs TideHunter) | Not needed — Soma only supports RocksDB |

**Action**: No tests needed. These are intentional omissions.

---

## Priority 3: InMemoryDB Tests

**File**: `store/src/memstore.rs`
**Sui status**: No dedicated tests in Sui either (InMemoryDB is only used behind the `Storage` enum)
**Rationale**: The `InMemoryDB` is used as the `Storage::InMemory` backend. While Sui doesn't test it directly, Soma should add coverage to ensure the in-memory backend behaves identically to RocksDB for all `Map` trait operations.

### New test file: `store/src/memstore_tests.rs` (or inline `#[cfg(test)] mod tests`)

| # | Test Name | What It Tests |
|---|-----------|---------------|
| 1 | `test_inmemory_put_get` | `put` + `get` round-trip |
| 2 | `test_inmemory_delete` | `delete` removes key, subsequent `get` returns `None` |
| 3 | `test_inmemory_multi_get` | `multi_get` returns correct values for multiple keys |
| 4 | `test_inmemory_batch_operations` | `InMemoryBatch`: `put_cf` + `delete_cf` applied atomically via `write` |
| 5 | `test_inmemory_iterator_forward` | Forward iteration returns entries in key order |
| 6 | `test_inmemory_iterator_reverse` | Reverse iteration returns entries in reverse key order |
| 7 | `test_inmemory_iterator_bounds` | Iteration with lower/upper bounds |
| 8 | `test_inmemory_drop_cf` | `drop_cf` removes all data for a column family |
| 9 | `test_inmemory_missing_cf` | `get` on non-existent column family returns `None` |
| 10 | `test_inmemory_overwrite` | `put` same key twice, second value overwrites first |

### Implementation Notes
- Tests should use raw `InMemoryDB` API (not through `DBMap`) since this is testing the storage layer directly
- Use `bincode` big-endian serialization for keys to match production usage
- Verify iterator ordering matches what RocksDB would return (BTreeMap guarantees this)

---

## Priority 4: Options & Configuration Tests

**File**: `store/src/rocks/options.rs`
**Sui status**: No dedicated options tests
**Rationale**: `default_db_options()` and the various `optimize_*` methods configure critical RocksDB parameters. Misconfiguration can cause performance degradation or data loss.

### New tests (inline in `options.rs` or new `options_tests.rs`)

| # | Test Name | What It Tests |
|---|-----------|---------------|
| 1 | `test_default_db_options` | `default_db_options()` returns valid `DBOptions` with non-default RocksDB options |
| 2 | `test_optimize_for_point_lookup` | `optimize_for_point_lookup` modifies block cache |
| 3 | `test_optimize_for_write_throughput` | `optimize_for_write_throughput` sets buffer sizes, compaction triggers |
| 4 | `test_optimize_for_large_values_no_scan` | `optimize_for_large_values_no_scan` enables blob storage |
| 5 | `test_read_write_options_builder` | `ReadWriteOptions` builder pattern (`set_ignore_range_deletions`, `set_log_value_hash`) |
| 6 | `test_read_size_from_env` | `read_size_from_env` parses env vars correctly, returns `None` for unset/invalid |
| 7 | `test_disable_blob_storage_env_var` | `DISABLE_BLOB_STORAGE` env var disables blob optimization |

### Implementation Notes
- For env var tests, use `std::env::set_var` / `remove_var` with unique var names or use a test mutex
- Options tests can be simple: call the method, verify a few key settings on the resulting `rocksdb::Options`
- Some options are "fire and forget" (RocksDB doesn't expose getters), so tests may need to just verify no panics

---

## Priority 5: Error Handling Tests

**File**: `store/src/error.rs` and `store/src/rocks/errors.rs`
**Sui status**: No dedicated error tests
**Rationale**: Ensure error conversions work correctly and display messages are useful.

### New tests (inline in `error.rs`)

| # | Test Name | What It Tests |
|---|-----------|---------------|
| 1 | `test_error_display` | All `TypedStoreError` variants produce expected `Display` output |
| 2 | `test_error_from_rocks` | `typed_store_err_from_rocks_err` converts RocksDB errors correctly |

---

## Priority 6: SafeIter / SafeRevIter Tests

**File**: `store/src/rocks/safe_iter.rs`
**Sui status**: No dedicated SafeIter tests (tested indirectly through rocks/tests.rs iterator tests)
**Rationale**: SafeIter has subtle initialization logic (`is_initialized` flag, lazy `seek_to_first`). Edge cases should be tested.

### New tests (can be added to rocks/tests.rs)

| # | Test Name | What It Tests |
|---|-----------|---------------|
| 1 | `test_safe_iter_empty_db` | `safe_iter` on empty DB returns no items |
| 2 | `test_safe_iter_deserialization_skip` | Iterator skips entries with invalid serialization (returns `None` for bad key/value) |
| 3 | `test_safe_rev_iter_empty_db` | Reverse iter on empty DB returns no items |
| 4 | `test_safe_iter_single_entry` | Iterator with exactly one entry works correctly in both directions |

### Implementation Notes
- SafeIter currently uses `.ok()` for deserialization (lines 48-49 of safe_iter.rs), which means invalid entries are **silently skipped**, not errored. Tests should verify this behavior.
- The `is_initialized` flag pattern ensures the first call to `next()` seeks to the first entry. Verify this works with multiple iterators on the same DBMap.

---

## Priority 7: Derive Macro Tests

**File**: `store/tests/macro_tests.rs` and `store-derive/src/lib.rs`
**Sui status**: 5 integration tests (3 ported, 2 metrics-dependent)

### Existing tests to keep

| # | Test Name | Status |
|---|-----------|--------|
| 1 | `macro_test` | Passing — comprehensive test of `DBMapUtils` |
| 2 | `rename_test` | Passing — `#[rename]` attribute |
| 3 | `deprecate_test` | Passing — `#[deprecated]` attribute with `open_tables_read_write_with_deprecation_option` |

### New derive macro tests

| # | Test Name | What It Tests |
|---|-----------|---------------|
| 1 | `test_custom_options_override` | `TablesCustomOptions` struct with `#[default_options_override_fn]` — verify custom functions are called (use the existing `TABLE1_OPTIONS_SET_FLAG`/`TABLE2_OPTIONS_SET_FLAG` statics) |
| 2 | `test_generics_derive` | `TablesGenerics<Q, W>` struct can be opened and used (compile + runtime check) |
| 3 | `test_describe_tables` | `describe_tables()` returns correct table name → (key_type, value_type) mapping |

### Implementation Notes
- `TablesCustomOptions`, `TablesGenerics`, and `TablesSingle` structs are already defined in macro_tests.rs but **not tested** — they just verify the derive compiles
- The `TABLE1_OPTIONS_SET_FLAG` and `TABLE2_OPTIONS_SET_FLAG` statics are defined but never asserted on
- `test_custom_options_override`: Open `TablesCustomOptions`, then check `TABLE1_OPTIONS_SET_FLAG.lock().unwrap().len()` and `TABLE2_OPTIONS_SET_FLAG.lock().unwrap().len()` match expected counts
- `test_generics_derive`: Instantiate `TablesGenerics::<String, i32>`, insert a `Generic<String, i32>` value, read it back

---

## Priority 8: Soma-Specific Tests

These tests cover Soma-specific behavior not present in Sui.

### 8A: nondeterministic! macro (msim compatibility)

**File**: `store/src/rocks/mod.rs` (lines 1015-1031)

| # | Test Name | Framework | What It Tests |
|---|-----------|-----------|---------------|
| 1 | `test_nondeterministic_macro` | `#[test]` | Under `cfg(not(msim))`: `nondeterministic!` simply evaluates the expression |

**Note**: Under `cfg(msim)`, the macro spawns a new OS thread to avoid intercepted system calls. Full msim testing is covered by e2e tests. This unit test just verifies the non-msim path.

### 8B: Database flush

**File**: `store/src/rocks/mod.rs`

| # | Test Name | What It Tests |
|---|-----------|---------------|
| 1 | `test_database_flush` | `Database::flush()` succeeds for both `Storage::Rocks` and `Storage::InMemory` |

### 8C: populate_missing_cfs

**File**: `store/src/rocks/mod.rs`

| # | Test Name | What It Tests |
|---|-----------|---------------|
| 1 | `test_populate_missing_cfs` | When reopening a DB that has CFs not in the input list, they are preserved |

### 8D: default_hash

**File**: `store/src/rocks/mod.rs`

| # | Test Name | What It Tests |
|---|-----------|---------------|
| 1 | `test_default_hash_deterministic` | `default_hash` produces consistent Blake2b256 digests |

---

## Implementation Order

### Phase 1: Verify & Attribute (Day 1)
1. Run all 28 existing tests, confirm passing: `cargo test -p store`
2. Add attribution headers to all 13 files listed in [Attribution Requirements](#attribution-requirements)
3. Fix any compilation or test issues found

### Phase 2: InMemoryDB Tests (Day 1-2)
4. Create `store/src/memstore_tests.rs` (or inline module) with 10 tests
5. Verify all pass

### Phase 3: Core Module Tests (Day 2-3)
6. Add options tests (7 tests) — inline in `options.rs` or new file
7. Add error handling tests (2 tests) — inline in `error.rs`
8. Add SafeIter edge case tests (4 tests) — in `rocks/tests.rs`
9. Verify all pass

### Phase 4: Derive Macro Tests (Day 3)
10. Add 3 new tests to `store/tests/macro_tests.rs`
11. Verify all pass

### Phase 5: Soma-Specific Tests (Day 3-4)
12. Add `nondeterministic!` macro test
13. Add `Database::flush` test
14. Add `populate_missing_cfs` test
15. Add `default_hash` determinism test
16. Verify all pass

### Phase 6: Final Verification (Day 4)
17. Run full test suite: `cargo test -p store`
18. Run under msim: `PYO3_PYTHON=python3 RUSTFLAGS="--cfg msim" cargo test -p store`
19. Document any remaining gaps

---

## Build & Run Commands

```bash
# Run all store unit tests
cargo test -p store

# Run specific test file
cargo test -p store --test macro_tests

# Run a single test by name
cargo test -p store test_open

# Run under msim (for nondeterministic! macro testing)
PYO3_PYTHON=python3 RUSTFLAGS="--cfg msim" cargo test -p store

# Check store-derive compiles
cargo check -p store-derive

# Run with output for debugging
cargo test -p store -- --nocapture

# Run only the new InMemoryDB tests (once added)
cargo test -p store inmemory
```

---

## Appendix: Cross-Reference File Paths

For the implementer's convenience, here are the exact file paths to cross-reference between Soma and Sui:

| Soma File (local) | Sui File (GitHub) |
|-------------------|-------------------|
| `store/src/lib.rs` | `crates/typed-store/src/lib.rs` |
| `store/src/rocks/mod.rs` | `crates/typed-store/src/rocks/mod.rs` |
| `store/src/rocks/tests.rs` | `crates/typed-store/src/rocks/tests.rs` |
| `store/src/rocks/errors.rs` | `crates/typed-store/src/rocks/errors.rs` |
| `store/src/rocks/options.rs` | `crates/typed-store/src/rocks/options.rs` |
| `store/src/rocks/safe_iter.rs` | `crates/typed-store/src/rocks/safe_iter.rs` |
| `store/src/rocks/rocks_util.rs` | (inline in `crates/typed-store/src/rocks/mod.rs`) |
| `store/src/traits.rs` | `crates/typed-store/src/traits.rs` |
| `store/src/memstore.rs` | `crates/typed-store/src/memstore.rs` |
| `store/src/error.rs` | `crates/typed-store/src/rocks/errors.rs` |
| `store/src/util.rs` | `crates/typed-store/src/util.rs` |
| `store/tests/macro_tests.rs` | `crates/typed-store/tests/macro_tests.rs` |
| `store-derive/src/lib.rs` | `crates/typed-store-derive/src/lib.rs` |

**Sui repo**: Use the `sui-repo` MCP tool to search/fetch any of these files:
```
mcp__sui-repo__search_sui_code(query="typed-store <function_name>")
mcp__sui-repo__fetch_generic_url_content(url="https://raw.githubusercontent.com/MystenLabs/sui/main/crates/typed-store/src/rocks/mod.rs")
```
