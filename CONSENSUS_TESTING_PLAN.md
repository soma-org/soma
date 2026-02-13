# Consensus Crate — Comprehensive Testing Plan

Testing plan for `consensus/src/` achieving high parity with Sui's `consensus/core/src/`. Covers file-by-file mapping, attribution requirements, test infrastructure gaps, and every test needed for parity.

**Sui reference**: `MystenLabs/sui` — `consensus/core/src/`
**Soma crate**: `consensus/src/`

---

## Audit Notes (Feb 2026)

**Priority Ranking**: #2 of 7 plans — critical for mainnet safety. BFT consensus is the foundation of chain liveness and safety; a consensus bug can cause forks, double-spends, or chain halts.

**Accuracy**: The claim of ZERO tests is **confirmed** — all test modules are commented out despite full test infrastructure existing (fixtures, builders, DAG parsers). The ~138 test gap estimate appears plausible based on Sui's consensus test structure. The main TESTING_PLAN.md incorrectly claims "~100+" consensus unit tests exist — this is false and has been corrected.

**Key Concerns**:
1. **ZERO tests on a BFT consensus fork is the single biggest mainnet risk.** This is a direct fork of Mysticeti (Sui's consensus) with all tests stripped. The code is complex (committers, DAG state, leader scheduling) and any divergence from Sui is unverified.
2. **`CommitTestFixture` is MISSING** — this is the critical infrastructure for randomized tests. Without it, the two most important consensus correctness tests (`test_randomized_dag_all_direct_commit`, `test_randomized_dag_and_decision_sequence`) cannot be implemented. This is the highest-priority infrastructure gap across all plans.
3. **Test modules are commented out but infrastructure exists** — uncommenting test modules in `base_committer.rs`, `universal_committer.rs`, and `network/mod.rs` is a quick win that may immediately enable ~40 committer tests.
4. **Missing: Byzantine behavior testing** — no tests for equivocating validators, censoring leaders, or network partition scenarios.
5. **Missing: msim integration tests** — Sui has `consensus/simtests/` with full-stack consensus tests under simulated conditions.

**Estimated Effort**: ~7 engineering days as planned, but porting `CommitTestFixture` could take 2-3 days alone. Recommend starting with uncommenting existing test modules as the fastest path to coverage.

**Mainnet Blocker**: YES — shipping without consensus tests is unacceptable. At minimum, the committer tests (Phase 2) and randomized tests (Phase 2, step 5) must be implemented before mainnet.

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [File-by-File Cross-Reference](#file-by-file-cross-reference)
3. [Attribution Requirements](#attribution-requirements)
4. [Test Infrastructure Gaps](#test-infrastructure-gaps)
5. [Dedicated Test Files (tests/ directory)](#dedicated-test-files)
6. [Inline Tests by Module](#inline-tests-by-module)
7. [Integration / Simtests](#integration--simtests)
8. [Implementation Order](#implementation-order)
9. [Build & Run Commands](#build--run-commands)

---

## Executive Summary

### Current State
- **Soma consensus crate has ZERO actual test functions** (`#[test]` or `#[tokio::test]`)
- Test modules are **commented out** in `base_committer.rs`, `universal_committer.rs`, and `network/mod.rs`
- Test infrastructure (fixtures, builders, DAG parsers) **exists but is unused**
- 42 `#[cfg(test)]` helper items are scattered across modules, ready for test use

### Target State
- **~138+ unit tests** matching Sui's consensus/core test coverage
- 5 dedicated test files in a `tests/` directory
- All test infrastructure wired up and functional

### Test Count Summary

| Category | Sui Tests | Soma Status | Gap |
|----------|-----------|-------------|-----|
| Dedicated committer tests (tests/) | 40 | 0 (commented out) | 40 |
| core.rs inline | 19 | 0 | 19 |
| dag_state.rs inline | 16 | 0 | 16 |
| leader_schedule.rs inline | 13 | 0 | 13 |
| block_manager.rs inline | 10 | 0 | 10 |
| transaction.rs inline | 7 | 0 | 7 |
| synchronizer.rs inline | 7 | 0 | 7 |
| linearizer.rs inline | 7 | 0 | 7 |
| authority_node.rs inline | 4 | 0 | 4 |
| threshold_clock.rs inline | 3 | 0 | 3 |
| commit_observer.rs inline | 2 | 0 | 2 |
| ancestor.rs inline | 2 | 0 | 2 |
| commit_syncer.rs inline | 1 | 0 | 1 |
| test_dag_parser.rs inline | 7 | 0 | 7 |
| **Total** | **~138** | **0** | **~138** |

---

## File-by-File Cross-Reference

### Legend
- **Match** = Soma file corresponds to Sui file
- **Partial** = Soma has the file but with modifications
- **Missing** = Sui has it, Soma doesn't (may be in types crate or not needed)
- **Soma-only** = Soma file with no Sui counterpart

### Core Consensus Logic

| Soma File | Sui File | Match | Has Tests in Sui | Notes |
|-----------|----------|-------|------------------|-------|
| `base_committer.rs` | `base_committer.rs` | Match | 8 (dedicated) + 7 (declarative) | Test modules commented out in Soma |
| `universal_committer.rs` | `universal_committer.rs` | Match | 11 (dedicated) | Test modules commented out in Soma |
| `core.rs` | `core.rs` | Match | 19 inline | CoreTextFixture present (cfg(test)) |
| `core_thread.rs` | `core_thread.rs` | Match | 0 (MockDispatcher only) | MockCoreThreadDispatcher present |
| `dag_state.rs` | `dag_state.rs` | Match | 16 inline | 3 cfg(test) helpers |
| `linearizer.rs` | `linearizer.rs` | Match | 7 inline | |
| `commit_observer.rs` | `commit_observer.rs` | Match | 2 inline | |
| `commit_finalizer.rs` | `commit_finalizer.rs` | Match | 0 (cfg(test) helpers only) | 1 cfg(test) helper |
| `leader_schedule.rs` | `leader_schedule.rs` | Partial | 13 inline | Soma may lack `leader_scoring.rs` |
| `block_manager.rs` | `block_manager.rs` | Match | 10 inline | 2 cfg(test) helpers |
| `threshold_clock.rs` | `threshold_clock.rs` | Match | 3 inline | 1 cfg(test) helper |
| `ancestor.rs` | `ancestor.rs` | Match | 2 inline | |
| `leader_timeout.rs` | `leader_timeout.rs` | Match | 0 | No inline tests in Sui either |

### Transactions & Block Processing

| Soma File | Sui File | Match | Has Tests in Sui | Notes |
|-----------|----------|-------|------------------|-------|
| `transaction.rs` | `transaction.rs` | Match | 7 inline | 2 cfg(test) helpers |
| `block_verifier.rs` | `block_verifier.rs` | Match | 0 | NoopBlockVerifier used in tests |
| `transaction_certifier.rs` | `transaction_certifier.rs` | Match | 0 | New in Sui, may diverge |
| `proposed_block_handler.rs` | `proposed_block_handler.rs` | Match | 0 | |

### Synchronization

| Soma File | Sui File | Match | Has Tests in Sui | Notes |
|-----------|----------|-------|------------------|-------|
| `synchronizer.rs` | `synchronizer.rs` | Match | 7 inline | 1 cfg(test) helper |
| `commit_syncer.rs` | `commit_syncer.rs` | Match | 1 inline | 5 cfg(test) helpers |
| `commit_consumer.rs` | `commit_consumer.rs` | Match | 0 | |
| `commit_vote_monitor.rs` | `commit_vote_monitor.rs` | Match | 0 | |
| `subscriber.rs` | `subscriber.rs` | Match | 0 | |
| `round_prober.rs` | `round_prober.rs` | Match | 0 | |
| `round_tracker.rs` | `round_tracker.rs` | Match | 0 | |

### Authority & Network

| Soma File | Sui File | Match | Has Tests in Sui | Notes |
|-----------|----------|-------|------------------|-------|
| `authority_node.rs` | `authority_node.rs` | Match | 4 inline | 1 cfg(test) helper |
| `authority_service.rs` | `authority_service.rs` | Match | 0 | |
| `network/mod.rs` | `network/mod.rs` | Match | 0 | Test modules commented out |
| `network/tonic_network.rs` | `network/tonic_network.rs` | Match | 0 | 1 cfg(test) helper |
| `network/tonic_tls.rs` | `network/tonic_tls.rs` | Match | 0 | |

### Test Infrastructure

| Soma File | Sui File | Match | Notes |
|-----------|----------|-------|-------|
| `test_dag.rs` | `test_dag.rs` | Match | build_dag(), create_random_dag() |
| `test_dag_builder.rs` | `test_dag_builder.rs` | Match | DagBuilder, LayerBuilder — NOT cfg(test) gated |
| `test_dag_parser.rs` | `test_dag_parser.rs` | Match | parse_dag() DSL — 7 tests in Sui |
| — | `commit_test_fixture.rs` | **Missing** | CommitTestFixture, RandomDag, RandomDagIterator |

### Files in Sui Not Present in Soma

| Sui File | Purpose | Needed? |
|----------|---------|---------|
| `context.rs` | `Context::new_for_test()`, metrics, committee | Types in Soma's types crate |
| `error.rs` | ConsensusError enum | Likely in Soma's types crate |
| `metrics.rs` | Metrics definitions | May be in Soma's types crate |
| `commit.rs` | Commit, CommittedSubDag, TrustedCommit | Types in Soma's types crate |
| `block.rs` | Block, TestBlock, VerifiedBlock | Types in Soma's types crate |
| `broadcaster.rs` | Block broadcasting | May not be needed |
| `leader_scoring.rs` | ReputationScores for leader election | May be integrated into leader_schedule.rs |
| `commit_test_fixture.rs` | **Critical test infrastructure** | **YES — needed for randomized tests** |
| `storage/rocksdb_store.rs` | RocksDB storage backend | Storage may differ |
| `storage/mem_store.rs` | In-memory store for tests | **YES — needed for tests** |
| `storage/store_tests.rs` | Storage unit tests | Needed if storage is similar |
| `network/test_network.rs` | Mock network for tests | Needed for network tests |
| `network/network_tests.rs` | Network integration tests | Needed for network tests |
| `network/metrics.rs` | Network metrics | May be in types crate |

---

## Attribution Requirements

**Rule**: Any file with code similarity to Sui requires Apache 2.0 attribution header.

All files below need the following header added:

```
// Portions of this file are derived from Mysticeti consensus (MystenLabs/sui).
// Original source: https://github.com/MystenLabs/sui/tree/main/consensus/core/src/
// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0
```

### Files Requiring Attribution (Heavy — Direct Fork)

Every `.rs` file in `consensus/src/` is a direct fork and needs attribution:

| File | Sui Counterpart |
|------|-----------------|
| `lib.rs` | `lib.rs` |
| `base_committer.rs` | `base_committer.rs` |
| `universal_committer.rs` | `universal_committer.rs` |
| `core.rs` | `core.rs` |
| `core_thread.rs` | `core_thread.rs` |
| `dag_state.rs` | `dag_state.rs` |
| `linearizer.rs` | `linearizer.rs` |
| `commit_observer.rs` | `commit_observer.rs` |
| `commit_finalizer.rs` | `commit_finalizer.rs` |
| `leader_schedule.rs` | `leader_schedule.rs` |
| `block_manager.rs` | `block_manager.rs` |
| `threshold_clock.rs` | `threshold_clock.rs` |
| `ancestor.rs` | `ancestor.rs` |
| `leader_timeout.rs` | `leader_timeout.rs` |
| `transaction.rs` | `transaction.rs` |
| `block_verifier.rs` | `block_verifier.rs` |
| `transaction_certifier.rs` | `transaction_certifier.rs` |
| `proposed_block_handler.rs` | `proposed_block_handler.rs` |
| `synchronizer.rs` | `synchronizer.rs` |
| `commit_syncer.rs` | `commit_syncer.rs` |
| `commit_consumer.rs` | `commit_consumer.rs` |
| `commit_vote_monitor.rs` | `commit_vote_monitor.rs` |
| `subscriber.rs` | `subscriber.rs` |
| `round_prober.rs` | `round_prober.rs` |
| `round_tracker.rs` | `round_tracker.rs` |
| `authority_node.rs` | `authority_node.rs` |
| `authority_service.rs` | `authority_service.rs` |
| `network/mod.rs` | `network/mod.rs` |
| `network/tonic_network.rs` | `network/tonic_network.rs` |
| `network/tonic_tls.rs` | `network/tonic_tls.rs` |
| `test_dag.rs` | `test_dag.rs` |
| `test_dag_builder.rs` | `test_dag_builder.rs` |
| `test_dag_parser.rs` | `test_dag_parser.rs` |

**Total: 33 files need attribution.**

---

## Test Infrastructure Gaps

Before implementing tests, the following infrastructure must be created or enabled.

### 1. Uncomment Existing Test Modules (Quick Win)

**`base_committer.rs`** (lines 15-21):
```rust
// Currently commented out:
// #[cfg(test)]
// #[path = "tests/base_committer_tests.rs"]
// mod base_committer_tests;

// Uncomment and create tests/base_committer_tests.rs
```

**`universal_committer.rs`** (lines 12-18):
```rust
// Currently commented out:
// #[cfg(test)]
// #[path = "tests/universal_committer_tests.rs"]
// mod universal_committer_tests;
```

**`network/mod.rs`** (lines 35-38):
```rust
// Currently commented out:
// #[cfg(test)]
// mod network_tests;
// #[cfg(test)]
// pub(crate) mod test_network;
```

### 2. Create `tests/` Directory

Create `consensus/src/tests/` with:
- `base_committer_tests.rs` — 8 tests
- `base_committer_declarative_tests.rs` — 7 tests (uses DAG DSL parser)
- `universal_committer_tests.rs` — 11 tests
- `pipelined_committer_tests.rs` — 12 tests (if Soma has pipelined committer)
- `randomized_tests.rs` — 2 tests (requires CommitTestFixture)

### 3. Create `commit_test_fixture.rs` (Critical)

This file is **required for randomized tests** and does not exist in Soma. Port from Sui's `consensus/core/src/commit_test_fixture.rs`:

```rust
// Key types to port:
pub struct CommitTestFixture { ... }
pub struct RandomDag { ... }
pub struct RandomDagIterator { ... }

// Key function:
pub fn assert_commit_sequences_match(commits_a: &[..], commits_b: &[..])
```

### 4. Ensure `Context::new_for_test()` Equivalent

Sui tests heavily use `Context::new_for_test(num_authorities)`. Soma needs an equivalent that creates:
- A test committee with the given number of authorities
- Default protocol config
- Test-compatible metrics/clock

### 5. Ensure `MemStore` Equivalent

Sui tests use `storage::mem_store::MemStore` as an in-memory store. Soma needs a compatible implementation if not already present.

### 6. Wire Up `BaseCommitterBuilder`

Already exists in `base_committer.rs` behind `#[cfg(test)]`. Verify it works and matches Sui's API:
```rust
pub(crate) struct BaseCommitterBuilder { ... }
impl BaseCommitterBuilder {
    fn new(context, dag_state) -> Self
    fn with_wave_length(self, wave_length) -> Self
    fn build(self) -> BaseCommitter
}
```

### 7. Wire Up `CoreTextFixture`

Already exists in `core.rs` behind `#[cfg(test)]`. Verify it provides:
- Committee setup with keypairs
- Transaction client/consumer
- DagState initialization
- Core instance ready for testing

---

## Dedicated Test Files

### `tests/base_committer_tests.rs` — 8 Tests

Uses `BaseCommitterBuilder` to construct committers and `DagBuilder` to create test DAGs.

| Test | Description | Sui Reference |
|------|-------------|---------------|
| `try_direct_commit` | Leader with 2f+1 votes is directly committed | base_committer_tests.rs |
| `idempotence` | Committing same leader twice returns same result | base_committer_tests.rs |
| `multiple_direct_commit` | Multiple leaders committed in sequence | base_committer_tests.rs |
| `direct_skip` | Leader with < 2f+1 votes and enough blame is skipped | base_committer_tests.rs |
| `indirect_commit` | Leader committed indirectly via later leader's certified link | base_committer_tests.rs |
| `indirect_skip` | Leader indirectly skipped via certified skip | base_committer_tests.rs |
| `undecided` | Leader with insufficient info remains undecided | base_committer_tests.rs |
| `test_byzantine_direct_commit` | Byzantine leader (equivocating blocks) still committed | base_committer_tests.rs |

### `tests/base_committer_declarative_tests.rs` — 7 Tests

Uses the DAG DSL parser (`parse_dag()`) for human-readable test DAG construction.

| Test | Description | Sui Reference |
|------|-------------|---------------|
| `direct_commit` | Declarative DAG: standard direct commit | base_committer_declarative_tests.rs |
| `direct_skip` | Declarative DAG: skip leader with no votes | base_committer_declarative_tests.rs |
| `direct_undecided` | Declarative DAG: insufficient votes = undecided | base_committer_declarative_tests.rs |
| `indirect_commit` | Declarative DAG: indirect commit path | base_committer_declarative_tests.rs |
| `indirect_skip` | Declarative DAG: indirect skip path | base_committer_declarative_tests.rs |
| `test_equivocating_direct_commit` | Equivocating authority, still commit | base_committer_declarative_tests.rs |
| `test_equivocating_direct_skip` | Equivocating authority, skip leader | base_committer_declarative_tests.rs |

### `tests/universal_committer_tests.rs` — 11 Tests

| Test | Description | Sui Reference |
|------|-------------|---------------|
| `direct_commit` | Universal: direct commit with quorum | universal_committer_tests.rs |
| `idempotence` | Universal: idempotent commits | universal_committer_tests.rs |
| `multiple_direct_commit` | Universal: multiple sequential commits | universal_committer_tests.rs |
| `direct_commit_late_call` | Universal: late commit still works | universal_committer_tests.rs |
| `no_genesis_commit` | Universal: genesis round never committed | universal_committer_tests.rs |
| `direct_skip_no_leader_votes` | Universal: skip when leader has no votes | universal_committer_tests.rs |
| `direct_skip_missing_leader_block` | Universal: skip when leader block is missing | universal_committer_tests.rs |
| `indirect_commit` | Universal: indirect commit path | universal_committer_tests.rs |
| `indirect_skip` | Universal: indirect skip path | universal_committer_tests.rs |
| `undecided` | Universal: insufficient info = undecided | universal_committer_tests.rs |
| `test_byzantine_direct_commit` | Universal: Byzantine validators | universal_committer_tests.rs |

### `tests/pipelined_committer_tests.rs` — 12 Tests

**Note**: Only needed if Soma implements pipelined committing. Check if Soma has a pipelined committer.

| Test | Description | Sui Reference |
|------|-------------|---------------|
| `direct_commit` | Pipelined: direct commit | pipelined_committer_tests.rs |
| `idempotence` | Pipelined: idempotent commits | pipelined_committer_tests.rs |
| `multiple_direct_commit` | Pipelined: multiple commits | pipelined_committer_tests.rs |
| `direct_commit_late_call` | Pipelined: late commit | pipelined_committer_tests.rs |
| `no_genesis_commit` | Pipelined: no genesis commit | pipelined_committer_tests.rs |
| `direct_skip_no_leader` | Pipelined: skip absent leader | pipelined_committer_tests.rs |
| `direct_skip_enough_blame` | Pipelined: skip with blame quorum | pipelined_committer_tests.rs |
| `indirect_commit` | Pipelined: indirect commit | pipelined_committer_tests.rs |
| `indirect_skip` | Pipelined: indirect skip | pipelined_committer_tests.rs |
| `undecided` | Pipelined: undecided | pipelined_committer_tests.rs |
| `test_byzantine_validator` | Pipelined: Byzantine node | pipelined_committer_tests.rs |
| (12th varies) | Additional pipelined scenario | pipelined_committer_tests.rs |

### `tests/randomized_tests.rs` — 2 Tests (Critical)

**Requires**: `commit_test_fixture.rs` (CommitTestFixture, RandomDag, RandomDagIterator)

| Test | Description | Sui Reference |
|------|-------------|---------------|
| `test_randomized_dag_all_direct_commit` | 100 iterations, 1000 rounds, 100% leader inclusion — every commit is direct | randomized_tests.rs |
| `test_randomized_dag_and_decision_sequence` | 100 iterations, 50% leader inclusion, uses RandomDagIterator to verify mixed commit/skip/undecided sequences match | randomized_tests.rs |

These are **the most important tests** for consensus correctness. They verify that the committer produces deterministic results across random DAG topologies.

---

## Inline Tests by Module

### `core.rs` — 19 Tests

| Test | Description |
|------|-------------|
| `test_core_recover_from_store_for_full_round` | Recovery when all blocks of a round are stored |
| `test_core_recover_from_store_for_partial_round` | Recovery with only partial round blocks stored |
| `test_core_propose_after_genesis` | First proposal after genesis blocks |
| `test_core_propose_once_receiving_a_quorum` | Proposal triggered by quorum of received blocks |
| `test_commit_and_notify_for_block_status` | Commit triggers block status notifications |
| `test_multiple_commits_advance_threshold_clock` | Multiple commits advance the threshold clock |
| `test_core_set_min_propose_round` | Setting minimum propose round prevents earlier proposals |
| `test_core_try_new_block_leader_timeout` | Block proposal on leader timeout |
| `test_core_try_new_block_with_leader_timeout_and_low_scoring_authority` | Leader timeout with low-scoring authority |
| `test_smart_ancestor_selection` | Smart ancestor selection for block proposals |
| `test_excluded_ancestor_limit` | Excluded ancestor count respects limits |
| `test_core_set_propagation_delay_per_authority` | Propagation delay per authority |
| `test_leader_schedule_change` | Schedule change after reputation scores update |
| `test_filter_new_commits` | Filtering already-known commits |
| `test_add_certified_commits` | Adding certified commits from commit sync |
| `test_commit_on_leader_schedule_change_boundary_without_multileader` | Schedule change boundary (single leader) |
| `test_commit_on_leader_schedule_change_boundary_with_multileader` | Schedule change boundary (multi-leader) |
| `test_core_signals` | Core signals (new block, new round) fire correctly |
| `test_core_compress_proposal_references` | Proposal reference compression |

### `dag_state.rs` — 16 Tests

| Test | Description |
|------|-------------|
| `test_get_blocks` | Basic block retrieval |
| `test_ancestors_at_uncommitted_round` | Ancestors at uncommitted round boundary |
| `test_link_causal_history` | Causal history linking |
| `test_contains_blocks_in_cache_or_store` | Block existence check across cache and store |
| `test_contains_cached_block_at_slot` | Cached block existence at specific slot |
| `test_contains_cached_block_at_slot_panics_when_ask_out_of_range` | Panics on out-of-range slot query |
| `test_get_blocks_in_cache_or_store` | Block retrieval from cache or store |
| `test_flush_and_recovery` | Flush to store and recovery |
| `test_block_info_as_committed` | Block info marking as committed |
| `test_get_cached_blocks` | Cached block retrieval |
| `test_get_last_cached_block` | Last cached block per authority |
| `test_get_cached_last_block_per_authority_requesting_out_of_round_range` | Out-of-range last cached block request |
| `test_last_quorum` | Last quorum round computation |
| `test_last_block_for_authority` | Last block for specific authority |
| `test_accept_block_not_panics_when_timestamp_is_ahead_and_median_timestamp` | Timestamp validation |
| `test_last_finalized_commit` | Last finalized commit tracking |

### `block_manager.rs` — 10 Tests

| Test | Description |
|------|-------------|
| `suspend_blocks_with_missing_ancestors` | Blocks suspended when ancestors missing |
| `try_accept_block_returns_missing_blocks` | Missing block refs returned on accept |
| `accept_blocks_with_complete_causal_history` | Blocks accepted when all ancestors present |
| `accept_blocks_with_causal_history_below_gc_round` | Acceptance with ancestors below GC round |
| `skip_accepting_blocks_below_gc_round` | Blocks below GC round are skipped |
| `accept_blocks_unsuspend_children_blocks` | Randomized: accepting blocks unsuspends children (100 seeds) |
| `unsuspend_blocks_for_latest_gc_round` | GC advancement unsuspends blocks (parametrized gc_depth) |
| `try_accept_committed_blocks` | Committed blocks bypass suspension |
| `try_find_blocks` | Finding blocks in dag state and suspended set |
| `test_verify_block_timestamps_and_accept` | Timestamp verification with median-based acceptance |

### `leader_schedule.rs` — 13 Tests

| Test | Description |
|------|-------------|
| `test_elect_leader` | Basic leader election (round-robin in tests) |
| `test_elect_leader_stake_based` | Stake-weighted leader election |
| `test_leader_schedule_from_store` | Schedule recovery from stored CommitInfo |
| `test_leader_schedule_from_store_no_commits` | Schedule recovery with no commits |
| `test_leader_schedule_from_store_no_commit_info` | Schedule recovery without CommitInfo |
| `test_leader_schedule_commits_until_leader_schedule_update` | Commits until next schedule update |
| `test_leader_schedule_update_leader_schedule` | Full schedule update flow |
| `test_leader_swap_table` | Good/bad node classification |
| `test_leader_swap_table_swap` | Leader swap mechanics |
| `test_leader_swap_table_retrieve_first_nodes` | Stake-threshold node retrieval |
| `test_leader_swap_table_swap_stake_threshold_out_of_bounds` | Out-of-bounds threshold panics |
| `test_update_leader_swap_table` | Updating swap table with new scores |
| `test_update_bad_leader_swap_table` | Invalid commit range panics |

### `linearizer.rs` — 7 Tests

| Test | Description |
|------|-------------|
| `test_handle_commit` | Basic commit handling with sub-dag collection |
| `test_handle_already_committed` | Handling blocks already committed in prior sub-dags |
| `test_handle_commit_with_gc_simple` | Commit with GC depth — blocks below GC excluded |
| `test_handle_commit_below_highest_committed_round` | Orphaned blocks committed below highest round |
| `test_calculate_commit_timestamp` | Median timestamp calculation by stake |
| `test_median_timestamps_by_stake` | Unit test for median timestamp with various stake distributions |
| `test_median_timestamps_by_stake_errors` | Error cases: no blocks, insufficient stake |

### `transaction.rs` — 7 Tests

| Test | Description |
|------|-------------|
| `basic_submit_and_consume` | Submit transactions and consume them |
| `block_status_update` | Block status notifications (Sequenced/GarbageCollected) |
| `submit_over_max_fetch_size_and_consume` | Transactions exceeding max block bytes |
| `submit_large_batch_and_ack` | Large batch submission with ack |
| `test_submit_over_max_block_size_and_validate_block_size` | Max block size limit enforcement |
| `submit_with_no_transactions` | Ping signal (empty transaction submission) |
| `ping_transaction_index_never_reached` | Reserved transaction index boundary |

### `synchronizer.rs` — 7 Tests

| Test | Description |
|------|-------------|
| `test_inflight_blocks_map` | Lock/unlock/swap mechanics for inflight blocks |
| `successful_fetch_blocks_from_peer` | Successful block fetch and processing |
| `saturate_fetch_blocks_from_peer` | Saturated synchronizer error handling |
| `synchronizer_periodic_task_fetch_blocks` | Periodic sync fetches missing blocks |
| `synchronizer_periodic_task_when_commit_lagging_gets_disabled` | Periodic sync disabled during commit lag |
| `synchronizer_fetch_own_last_block` | Amnesia recovery: fetch own last block |
| `test_process_fetched_blocks` | Processing fetched blocks end-to-end |

### `commit_observer.rs` — 2 Tests

| Test | Description |
|------|-------------|
| `test_handle_commit` | Commit observation with leader schedule updates |
| `test_recover_and_send_commits` | Recovery and replay of commits (multiple scenarios) |

### `threshold_clock.rs` — 3 Tests

| Test | Description |
|------|-------------|
| `test_threshold_clock_add_block` | Adding individual blocks advances clock |
| `test_threshold_clock_add_blocks` | Batch block addition |
| `test_threshold_clock_add_block_min_committee` | Single-authority committee behavior |

### `ancestor.rs` — 2 Tests

| Test | Description |
|------|-------------|
| `test_calculate_network_high_accepted_quorum_round` | Network high quorum round calculation |
| `test_update_all_ancestor_state_using_accepted_rounds` | Full state transition cycle: Include → Exclude → Include |

### `commit_syncer.rs` — 1 Test

| Test | Description |
|------|-------------|
| `commit_syncer_start_and_pause_scheduling` | Schedule fetch ranges, pause when commit consumer lags, resume |

### `test_dag_parser.rs` — 7 Tests

| Test | Description |
|------|-------------|
| `test_dag_parsing` | Full DAG DSL parsing |
| `test_genesis_round_parsing` | Genesis round specification |
| `test_slot_parsing` | Individual slot parsing |
| `test_all_round_parsing` | Wildcard round (`*`) parsing |
| `test_specific_round_parsing` | Named authority connections |
| `test_parse_author_and_connections` | Author and connection parsing |
| `test_str_to_authority_index` | Authority name to index mapping |

### `authority_node.rs` — 4 Tests

| Test | Description |
|------|-------------|
| `test_authority_start_and_stop` | Single authority start and clean stop |
| `test_authority_committee` | 4-node committee: submit transactions, verify commits, restart one node |
| `test_small_committee` | Parametrized (1, 2, 3 nodes): submit, verify, restart |
| `test_amnesia_recovery_success` | Amnesia recovery: wipe DB, restart, recover via peers |

---

## Integration / Simtests

Sui has integration-level tests in `consensus/simtests/tests/`:

### `consensus_tests.rs`
- Full consensus authority committee tests using `sim_test` macro
- Tests multi-authority consensus with real networking
- Verifies transaction commit across a committee

### `consensus_dag_tests.rs`
- Uses `CommitTestFixture` and `assert_commit_sequences_match`
- Randomized DAG construction with various topologies
- Verifies deterministic commit sequences

**Soma approach**: These simtests should be adapted as e2e-tests using Soma's msim framework in `e2e-tests/tests/`, or as standalone integration tests if the consensus crate supports `cfg(msim)`.

---

## Implementation Order

### Phase 1: Infrastructure (Day 1)

1. **Add attribution headers** to all 33 files
2. **Uncomment test modules** in `base_committer.rs`, `universal_committer.rs`, `network/mod.rs`
3. **Create `consensus/src/tests/` directory**
4. **Verify test infrastructure compiles**: `CoreTextFixture`, `BaseCommitterBuilder`, `DagBuilder`, `MockCoreThreadDispatcher`
5. **Verify `MemStore` or equivalent** is available for in-memory testing
6. **Verify `Context::new_for_test()`** or equivalent exists
7. **Port `commit_test_fixture.rs`** from Sui (needed for randomized tests)

### Phase 2: Committer Tests (Day 2-3) — 40 Tests

Priority: These are the **core consensus correctness tests**.

1. `tests/base_committer_tests.rs` — 8 tests
2. `tests/base_committer_declarative_tests.rs` — 7 tests
3. `tests/universal_committer_tests.rs` — 11 tests
4. `tests/pipelined_committer_tests.rs` — 12 tests (if applicable)
5. `tests/randomized_tests.rs` — 2 tests

### Phase 3: Core Module Tests (Day 3-4) — 35 Tests

1. `core.rs` — 19 inline tests (requires CoreTextFixture)
2. `dag_state.rs` — 16 inline tests (requires MemStore)

### Phase 4: Block & Leader Tests (Day 4-5) — 30 Tests

1. `block_manager.rs` — 10 inline tests
2. `leader_schedule.rs` — 13 inline tests
3. `linearizer.rs` — 7 inline tests

### Phase 5: Transaction & Sync Tests (Day 5-6) — 18 Tests

1. `transaction.rs` — 7 inline tests
2. `synchronizer.rs` — 7 inline tests (requires MockNetworkClient)
3. `authority_node.rs` — 4 inline tests

### Phase 6: Remaining Tests (Day 6-7) — 15 Tests

1. `threshold_clock.rs` — 3 inline tests
2. `ancestor.rs` — 2 inline tests
3. `commit_observer.rs` — 2 inline tests
4. `commit_syncer.rs` — 1 inline test
5. `test_dag_parser.rs` — 7 inline tests

### Phase 7: Verification

1. Run all tests: `cargo test -p consensus`
2. Verify test count matches target (~138)
3. Cross-check each test name against Sui's test list above
4. Run with msim if applicable: `PYO3_PYTHON=python3 RUSTFLAGS="--cfg msim" cargo test -p consensus`

---

## Build & Run Commands

```bash
# Run all consensus tests
cargo test -p consensus

# Run specific test file
cargo test -p consensus --test base_committer_tests

# Run specific test
cargo test -p consensus -- test_core_propose_after_genesis

# Run with msim (if tests use sim infrastructure)
PYO3_PYTHON=python3 RUSTFLAGS="--cfg msim" cargo test -p consensus

# Check compilation only
PYO3_PYTHON=python3 cargo check -p consensus

# Check with msim
PYO3_PYTHON=python3 RUSTFLAGS="--cfg msim" cargo check -p consensus
```
