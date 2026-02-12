# E2E Testing Guide

Reference for implementing, debugging, and expanding end-to-end tests in the Soma monorepo. Also covers using the `sui-repo` MCP to study Sui's upstream test patterns.

---

## Quick Start

```bash
# Build with msim (required for e2e tests)
RUSTFLAGS="--cfg msim" cargo build -p e2e-tests -p test-cluster

# Run all e2e tests
RUSTFLAGS="--cfg msim" cargo test -p e2e-tests

# Run a specific test file
RUSTFLAGS="--cfg msim" cargo test -p e2e-tests --test reconfiguration_tests

# Run a single test
RUSTFLAGS="--cfg msim" cargo test -p e2e-tests --test reconfiguration_tests basic_reconfig_end_to_end_test

# Control log level
RUST_LOG=info RUSTFLAGS="--cfg msim" cargo test -p e2e-tests --test target_tests
```

---

## Codebase Entry Points

### Test infrastructure

| File | Purpose |
|------|---------|
| `test-cluster/src/lib.rs` | **TestCluster** — main test orchestration. `TestClusterBuilder` for setup, methods for executing txs, triggering reconfig, waiting for epochs. |
| `test-cluster/src/swarm.rs` | **Swarm** — manages validator/fullnode topology. `SwarmBuilder` creates nodes from configs. |
| `test-cluster/src/swarm_node.rs` | **Node** — per-node lifecycle (`spawn`, `start`, `stop`, `is_running`, `get_node_handle`). |
| `test-cluster/src/container.rs` | Non-msim container (OS threads + tokio runtimes). |
| `test-cluster/src/container-sim.rs` | Msim container (simulated nodes via `msim::runtime::Handle`). |
| `e2e-tests/src/lib.rs` | Re-exports `SomaClient` and `integration_helpers`. |
| `e2e-tests/src/integration_helpers.rs` | Shared test utilities (currently minimal — expand as needed). |

### Core types and execution

| File | Purpose |
|------|---------|
| `types/src/transaction.rs` | All 24+ transaction types and their argument structs. |
| `types/src/object.rs` | Object model (Owner variants, ObjectRef, versioning). |
| `types/src/system_state/mod.rs` | SystemState — validators, emissions, parameters, model registry. |
| `authority/src/execution/mod.rs` | Execution pipeline — gas prep, input loading, dispatch, effects, commit. |
| `authority/src/consensus_handler.rs` | Consensus output handling — dedup, shared object versioning, scheduling. |
| `node/src/lib.rs` | Node startup — validator vs fullnode paths, reconfiguration monitor. |

### Existing test files

| File | Coverage |
|------|----------|
| `e2e-tests/tests/reconfiguration_tests.rs` | Epoch transitions, committee changes, voting power, validator lifecycle (14 tests) |
| `e2e-tests/tests/dynamic_committee_tests.rs` | Fuzz test for validator join/leave across epochs |
| `e2e-tests/tests/model_tests.rs` | Model commit-reveal lifecycle, genesis bootstrap |
| `e2e-tests/tests/target_tests.rs` | Target generation, submission, rewards, epoch boundaries |
| `e2e-tests/tests/challenge_tests.rs` | Challenge disputes, bond slashing, validator audit flows |
| `e2e-tests/tests/simulator_tests.rs` | msim determinism (FuturesOrdered/Unordered, select!, HashMap, full network) — 5 tests |
| `e2e-tests/tests/checkpoint_tests.rs` | Checkpoint integration, timestamp monotonicity, fork detection — 3 tests |
| `e2e-tests/tests/transaction_orchestrator_tests.rs` | Orchestrator execution, WAL/quorum, epoch boundaries, early validation — 8 tests |
| `e2e-tests/tests/multisig_tests.rs` | Ed25519 multisig: weighted thresholds, bitmap validation, signature verification — 1 test (6 sub-assertions) |
| `e2e-tests/tests/full_node_tests.rs` | Orchestrator presence, RunWithRange (checkpoint/epoch), stale object rejection, BCS round-trip — 7 tests |
| `e2e-tests/tests/protocol_version_tests.rs` | Version upgrade, quorum thresholds, laggard/shutdown validators, unsupported version panic — 6 tests |
| `e2e-tests/tests/rpc_tests.rs` | gRPC endpoints: get_object, get_transaction, get_checkpoint, get_epoch, get_service_info, get_balance, list_owned_objects — 7 tests |
| `e2e-tests/tests/shared_object_tests.rs` | Shared object mutation, conflicting owned txs, status transitions, version tracking, replay idempotency, racing miners, dependency tracking, concurrent conflicts — 8 tests |
| `e2e-tests/tests/failpoint_tests.rs` | Fault injection: delay injection (reconfig, epoch-change-tx), crash recovery (epoch store, certificate, consensus, checkpoint, DB batch, DB write), observation (checkpoint counter), crash + tx load — 10 tests |

---

## Test Anatomy

Every e2e test follows this pattern:

```rust
use test_cluster::TestClusterBuilder;
use tracing::info;
use types::{/* relevant types */};
use utils::logging::init_tracing;

#[cfg(msim)]
#[msim::sim_test]
async fn test_name() {
    init_tracing();

    // 1. Build cluster
    let test_cluster = TestClusterBuilder::new()
        .with_num_validators(4)       // default is 4
        .with_epoch_duration_ms(1000) // short epochs for testing
        .with_accounts(vec![...])     // funded test accounts
        .with_genesis_models(vec![...]) // seed models
        .build()
        .await;

    // 2. Execute transactions
    let tx_data = TransactionData::new(
        sender, kind, gas_object, gas_budget, gas_price,
    );
    let response = test_cluster.sign_and_execute_transaction(&tx_data).await;
    assert!(response.effects.status().is_ok());

    // 3. Query state
    let system_state = test_cluster.fullnode_handle.soma_node.with(|node| {
        node.state().get_system_state_object_for_testing().unwrap()
    });

    // 4. Assert
    assert_eq!(system_state.epoch(), 0);
}
```

### Key TestCluster methods

```rust
// Transaction execution
test_cluster.sign_and_execute_transaction(&tx_data).await;
test_cluster.execute_transaction(signed_tx).await;

// Epoch management
test_cluster.trigger_reconfiguration().await;
test_cluster.wait_for_epoch(Some(target_epoch)).await;
test_cluster.wait_for_epoch_all_nodes(epoch).await;

// Node control
test_cluster.stop_node(&name);
test_cluster.start_node(&name).await;
test_cluster.stop_all_validators();
test_cluster.start_all_validators().await;
test_cluster.spawn_new_validator(genesis_config).await;

// Access
test_cluster.all_node_handles();
test_cluster.all_validator_handles();
test_cluster.get_validator_pubkeys();
test_cluster.fullnode_handle.soma_node.with(|node| { ... });
```

### Determinism checking

```rust
// Runs the test twice and asserts identical execution
#[msim::sim_test(check_determinism)]
async fn test_determinism() {
    // ...
}
```

---

## Debugging with Logs

### Log files

When you call `init_tracing()` in a test, logs are written to `e2e-tests/logs/`:

| File | Contents |
|------|----------|
| `combined.log` | **All nodes, all events.** Start here. Can be 20+ MB. |
| `node_0.log` | Logs from node 0 only (typically first validator). |
| `node_1.log` | Logs from node 1, etc. |
| `node_N.log` | Created lazily — one file per simulated node. |

Node IDs are extracted from tracing spans via the regex `node{... id=N ...}`. Validators are typically nodes 0–3 (with 4 validators), fullnodes are higher IDs.

### How to read logs

```bash
# Tail combined log during a test
tail -f e2e-tests/logs/combined.log

# Search for a specific transaction
grep "tx_digest" e2e-tests/logs/combined.log

# Look at a specific node's activity
cat e2e-tests/logs/node_3.log | grep "ERROR\|WARN"

# Find epoch transitions
grep -i "epoch\|reconfig" e2e-tests/logs/combined.log

# Find consensus events for node 2
grep "consensus" e2e-tests/logs/node_2.log

# Check checkpoint progress
grep "checkpoint" e2e-tests/logs/combined.log | grep -i "certified\|created"
```

### Log format

```
2022-05-25T04:18:06.000000Z DEBUG node{name="main" id=0}: target_module: message
```

Fields: `timestamp`, `level`, `span_context` (with node ID), `target`, `message`.

### Common debugging patterns

**Test hangs?** Check if consensus is making progress:
```bash
grep "commit_round\|new_block" e2e-tests/logs/combined.log | tail -20
```

**Transaction failed?** Find the execution error:
```bash
grep "execution_error\|ExecutionFailure\|status.*Failure" e2e-tests/logs/combined.log
```

**Epoch not advancing?** Check if validators closed the epoch:
```bash
grep "close_epoch\|ChangeEpoch\|reconfig" e2e-tests/logs/combined.log
```

**State sync stalled?** Check sync progress on the fullnode:
```bash
grep "state_sync\|sync_to_checkpoint" e2e-tests/logs/node_5.log
```

### Controlling log verbosity

```bash
# Only warnings and errors (fast, small logs)
RUST_LOG=warn RUSTFLAGS="--cfg msim" cargo test -p e2e-tests ...

# Info level (good balance)
RUST_LOG=info RUSTFLAGS="--cfg msim" cargo test -p e2e-tests ...

# Debug for a specific module
RUST_LOG=info,authority::execution=debug RUSTFLAGS="--cfg msim" cargo test ...
```

### Important: logs are overwritten

Each test run recreates `combined.log` and all `node_N.log` files (the `File::create` in `init_tracing()` truncates). If you need to preserve logs from a run, copy the `logs/` directory before running again.

---

## Using the sui-repo MCP

The `sui-repo` MCP provides tools to search and read the upstream Sui codebase (MystenLabs/sui on GitHub). This is invaluable for studying test patterns, understanding how Sui implements features we've forked, and porting tests.

### Available tools

| Tool | Purpose | When to use |
|------|---------|-------------|
| `fetch_sui_documentation` | Fetches the repo README | General orientation |
| `search_sui_documentation` | Semantic search in docs | Finding design docs, architecture notes |
| `search_sui_code` | GitHub code search (exact match) | Finding test files, function definitions, type usage |
| `fetch_generic_url_content` | Fetch any URL | Reading specific files via raw GitHub URLs |

### Searching for tests

```
# Find all e2e test files
search_sui_code("filename:test path:crates/sui-e2e-tests/tests")

# Find tests using a specific feature
search_sui_code("sim_test path:crates/sui-e2e-tests")

# Find how Sui tests transaction orchestrator
search_sui_code("TransactionOrchestrator path:crates/sui-e2e-tests")

# Paginate (30 results per page)
search_sui_code("sim_test path:crates/sui-e2e-tests", page=2)
```

### Reading test file contents

```
# Fetch a specific file via raw GitHub URL
fetch_generic_url_content(
    "https://raw.githubusercontent.com/MystenLabs/sui/main/crates/sui-e2e-tests/tests/checkpoint_tests.rs"
)
```

### Searching for implementation patterns

```
# How does Sui's TestCluster work?
search_sui_code("TestClusterBuilder path:crates/sui-test-utils")
search_sui_code("TestCluster path:crates/test-cluster")

# How does Sui handle shared object versioning?
search_sui_code("assign_shared_object_versions path:crates/sui-core")

# Find Sui's transaction orchestrator
search_sui_code("TransactionOrchestrator path:crates/sui-core")
```

### Tips

- **GitHub code search is exact match** — use specific identifiers, not fuzzy queries.
- **Results are paginated** at 30 per page. Always check `Found N matches` and fetch page 2 if needed.
- **Raw URLs** follow the pattern: `https://raw.githubusercontent.com/MystenLabs/sui/main/<path>`.
- **Soma diverges from Sui** — no MoveVM, no ZKLogin, no events, no dynamic fields. When porting, strip Move-specific logic and substitute Soma's native transaction types (coin transfers, staking, model ops) as test workloads.

---

## Test Porting Plan

Tests to port from Sui's `crates/sui-e2e-tests/tests/`, organized by priority.

### Tier 1: Direct ports (no/minimal rewriting) — COMPLETE

All Tier 1 tests have been ported and pass consistently. **17 tests total** (see also Tier 2 below for 20 more).

| Source File | Tests | Status | Notes |
|-------------|-------|--------|-------|
| `simulator_tests.rs` | 5/5 | **PASS** | FuturesOrdered, FuturesUnordered, select!, HashMap, full network. Must use `msim::collections::{HashMap, HashSet}` for deterministic hashing (not `std::collections`). `test_net_determinism` runs without `check_determinism` due to non-deterministic RPC wallet path. |
| `transaction_orchestrator_tests.rs` | 8/8 | **PASS** | Blocking execution, WAL/quorum loss+recovery, epoch boundary tx, reconfig, execute+staking, early validation (no side effects + stale object rejection). Uses coin transfers and staking as workloads instead of Move. |
| `checkpoint_tests.rs` | 3/3 | **PASS** | Basic checkpoint integration, timestamp monotonicity, fork detection storage. Skipped split-brain test (requires `fail_point` infrastructure not in Soma) and alias test (Move-dependent). |
| `multisig_tests.rs` | 1/1 | **PASS** | Ed25519-only multisig with 6 sub-assertions: 2-of-3 threshold (two key combos), below-threshold rejection, empty sigs, duplicate sigs, wrong sender/key mismatch. Soma only supports Ed25519 (no Secp256k1/Secp256r1/ZkLogin/Passkey). |
| `state_sync_resilience_tests.rs` | — | **NOT FEASIBLE** | Requires `SimConfig`, `NetSim`, `InterNodeLatencyMap` for network latency injection. These exist in Sui's `sui-simulator` wrapper but not in Soma's direct msim usage. Would need significant msim extension work. |

#### Lessons learned during Tier 1 porting

1. **Deterministic collections**: `std::collections::HashMap/HashSet` use randomized hashing. For `check_determinism` tests, use `msim::collections::{HashMap, HashSet}` which seed ahash deterministically.

2. **TransactionOrchestrator architecture difference**: Sui spawns inner execution via `spawn_monitored_task!` (WAL guard survives caller timeout, background retry continues). Soma runs execution inline (WAL guard cleaned on drop, client must re-submit after timeout). WAL test adapted accordingly.

3. **ValidatorSet access**: Use `system_state.validators.validators[i].metadata.soma_address` (not `.active_validators`).

4. **Build command**: Always set `PYO3_PYTHON=python3` if python-sdk is in the workspace: `PYO3_PYTHON=python3 RUSTFLAGS="--cfg msim" cargo test -p e2e-tests --test <name>`

5. **Failpoint infrastructure**: Implemented in `utils/src/fp.rs` with macros in `utils/src/lib.rs`. Under `cfg(msim)`, the failpoint map is `thread_local!` — since msim runs all simulated nodes on the same OS thread, failpoints registered in the test context are visible to all validators. Under `cfg(not(msim))`, a process-wide `Mutex` is used instead. Use `AtomicBool` guards for crash-once patterns. Note: files with a local `mod utils` (e.g., `checkpoint_executor/mod.rs`) must use `::utils::fail_point!()` to avoid shadowing.

6. **No network simulation**: Soma's msim lacks `SimConfig`/`NetSim` for latency injection. State sync resilience tests that depend on degraded network simulation are not feasible.

### Tier 2: Selective ports (skip Move-dependent tests) — COMPLETE

All Tier 2 tests have been ported and pass consistently. **20 tests total across 3 files.**

| Source File | Tests | Status | Notes |
|-------------|-------|--------|-------|
| `full_node_tests.rs` | 7/7 | **PASS** | Orchestrator presence on validators (None) and fullnodes (Some), RunWithRange shutdown for checkpoint and epoch modes, stale object version rejection, orchestrator-based coin transfer, BCS serialize/deserialize round-trip. Skipped: follows_txes (no TransactionFilter), cold_sync/bootstrap_from_snapshot (no DB checkpoint infra), sponsored_transaction (Sui-specific), indexes (no index infra). |
| `protocol_version_tests.rs` | 6/6 | **PASS** | Unsupported version panic (`MAX_ALLOWED+1`), all-validator upgrade (v1→v2), no-quorum (50% insufficient), one-laggard (75% sufficient), shutdown+restart catch-up, insufficient support (25%). Skipped 19/26 Sui tests that require Move framework or framework-specific types. |
| `rpc_tests.rs` | 7/7 | **PASS** | gRPC endpoint coverage: get_service_info (chain_id, server_version), get_object (by ID), get_object_with_version (after mutation), get_transaction (by digest, with effects/checkpoint/timestamp), get_checkpoint (genesis + latest), get_epoch (epoch 0 + latest), get_balance + list_owned_objects (with and without type filter). Skipped ~36 Move-specific, coin registry, and ZkLogin tests. |

#### Infrastructure added for Tier 2

- **`RunWithRange` threading**: `FullnodeConfigBuilder.with_run_with_range()` → `SwarmBuilder.with_fullnode_run_with_range()` → `TestClusterBuilder.with_fullnode_run_with_range()`. Required for `run_with_range_checkpoint` and `run_with_range_epoch` tests.
- **SDK methods**: `SomaClient.get_balance()`, `get_latest_checkpoint()`, `get_checkpoint_summary()`. Required for RPC tests.
- **Protocol version 2 (msim)**: Uncommented `#[cfg(msim)]` block in `protocol-config/src/lib.rs` to generate a fake version 2 config (copies v1 with `base_fee += 1000`). Required for all upgrade tests.
- **Buffer stake override (msim)**: Set `buffer_stake_for_protocol_upgrade_bps=0` in msim protocol config so protocol upgrades need only 2/3 quorum (standard BFT threshold). Without this, the default 50% buffer made 3/4 validators insufficient for upgrades.

#### Bugs found and fixed during Tier 2 porting

1. **`ObjectVersionUnavailableForConsumption` was retriable** (`types/src/error.rs`): This error fell into the catch-all `_ => ErrorCategory::Aborted` which is retriable. The quorum driver retried the stale-ref transaction 17 times over 90 seconds before returning `TimeoutBeforeFinality`. Fixed by explicitly categorizing it as `ErrorCategory::InvalidTransaction` (non-retriable). Sui does the same — `ObjectVersionUnavailableForConsumption` is a `UserInputError` variant, and all `UserInputError` variants except `ObjectNotFound` and `DependentPackageNotFound` map to `InvalidTransaction`.

2. **`list_owned_objects` panicked on `None` object_type** (`authority/src/rpc_index.rs:731`): `owner_iter()` accepts `Option<ObjectType>` but calls `.unwrap()` on it when constructing the lower bound cursor. Fixed by defaulting to `ObjectType::SystemState` (the smallest enum variant) when no filter is provided, so iteration starts at the beginning of the owner's key space.

#### Lessons learned during Tier 2 porting

7. **Error categorization matters**: The transaction driver retries on `Aborted` errors. If a permanent error like stale object version is categorized as `Aborted`, the driver retries for 90 seconds before returning `TimeoutBeforeFinality` — hiding the real error. Always explicitly categorize new error variants.

8. **Use logs to debug flaky tests**: When a test is intermittently failing, check `e2e-tests/logs/combined.log`. The log revealed that validators were correctly detecting the stale version but the error was being retried. The key log line: `TransactionDriver error: TimeoutBeforeFinalityWithErrors { last_error: "...not available for consumption..." }`.

9. **Protocol version upgrade thresholds**: The effective upgrade threshold = quorum threshold (2/3) + buffer stake. With 4 equal-stake validators and `buffer_stake_bps=5000`, the threshold is ~83.3%, making 75% (3/4) insufficient. In msim we override to 0; in production the buffer provides safety margin against hasty upgrades.

10. **Pruned objects vs stale objects**: After pruning, the newer object version is removed from both cache and store. The early validation path (`check_transaction_validity`) only detects stale refs when the newer version IS in cache. After pruning, validation passes and the transaction goes to the quorum driver. This is a known architectural gap — not tested, but documented here for future work.

### Tier 3: Shared object rewrites — COMPLETE

All Tier 3 tests have been ported and pass consistently. **8 tests total in 1 file.**

| Source File | Tests | Status | Notes |
|-------------|-------|--------|-------|
| `shared_object_tests.rs` | 8/8 | **PASS** | **Core (4):** Shared object mutation via SubmitData (verifies Target appears in effects.mutated(), Owner::Shared preserved, input_shared_objects includes both Target and SystemState), conflicting owned transactions (same coin ObjectRef used twice — first succeeds, second rejected with stale version), shared object status transition via ClaimRewards (Target mutated to Claimed, not deleted; subsequent ClaimRewards and SubmitData correctly rejected), version tracking invariants (Lamport versioning: version strictly increases across SubmitData → ReportSubmission → second ReportSubmission; all mutated objects in same tx share effects version). **Advanced (4):** Transaction replay idempotency (resubmit same signed SubmitData tx via orchestrator — identical effects digest, version, mutated/created counts), racing miners (3 concurrent SubmitData to same Target via orchestrator spawns — exactly 1 winner, 2 losers with TargetNotOpen), shared object dependency tracking (SubmitData → ReportSubmission → second ReportSubmission chain verifies effects.dependencies() links each mutation to its predecessor), concurrent conflicting owned transactions (2 TransferCoin of same coin submitted concurrently via orchestrator — exactly 1 succeeds, 1 gets ObjectsDoubleUsed). |

#### Approach: Sui→Soma rewrite

Sui's `shared_objects_tests.rs` and `shared_objects_version_tests.rs` use Move counters as shared objects. Since Soma has no MoveVM, we rewrote these patterns using Soma's native shared objects:

- **Move counter creation → Genesis Target**: Targets are created as shared objects at genesis (via `GenesisModelConfig`), replacing the need for Move `create_shared_counter()`.
- **Move counter increment → SubmitData/ReportSubmission**: These transactions mutate the Target shared object (status change, report recording), replacing `increment_counter()`.
- **Move counter deletion → ClaimRewards**: ClaimRewards transitions Target to `Claimed` status. Soma uses status transitions rather than true object deletion for shared objects — the object persists but becomes unusable. This is verified by asserting Target appears in `mutated()` (not `deleted()`) and subsequent operations fail.
- **Owned object equivocation → TransferCoin with stale ObjectRef**: Two transfers using the same coin ObjectRef validate that the consensus layer prevents double-spending.

#### Lessons learned during Tier 3 porting

11. **Shared object mutation vs deletion**: Soma's Target lifecycle uses status transitions (Open → Filled → Claimed) rather than object deletion. `ClaimRewards` mutates the target, it doesn't delete it. Tests should assert `mutated()` not `deleted()` for status transitions.

12. **Effects version invariants**: All mutated objects in a single transaction share the same effects version (Lamport timestamp). This is a useful invariant to assert: `effects.version() == mutated_object_version` for every mutated object.

13. **Conflicting owned transactions**: When a transaction spends a coin, subsequent transactions using the stale ObjectRef may fail either at the orchestrator level (returned as `Err`) or at the effects level (returned as `Ok` with error status). Tests should handle both paths.

14. **InputSharedObject tracking**: `effects.input_shared_objects()` lists all shared objects that were inputs to the transaction, classified as `Mutate`, `ReadOnly`, `ReadDeleted`, or `MutateDeleted`. For SubmitData, both Target and SystemState appear as `Mutate`.

15. **Transaction replay idempotency**: Resubmitting the exact same signed transaction returns identical effects (same digest, version, mutated/created objects). The orchestrator detects already-executed transactions via `is_tx_already_executed()` and returns cached results. This is important for client retry safety.

16. **Concurrent shared object mutations**: When multiple transactions target the same shared object concurrently, consensus sequences them. Exactly one succeeds; the rest fail at execution time (e.g., TargetNotOpen). Both successful and failed transactions produce effects — check `effects.status().is_ok()` to distinguish winners from losers.

17. **Orchestrator for concurrent tests**: Use `transaction_orchestrator()` (returns `Arc<TransactionOrchestrator>`) with `tokio::task::spawn` for concurrent submission. The orchestrator is cloneable via Arc. `SomaClient.execute_transaction()` serializes through a write lock and cannot be used for true concurrency.

18. **Dependency tracking via effects**: `effects.dependencies()` contains the digests of transactions that last modified any of the current transaction's inputs. When two transactions sequentially mutate the same shared object, the second's dependencies include the first's digest. This forms a causal dependency chain verifiable in tests.

### Tier 4: Expand existing tests — COMPLETE

All 6 TODO'd reconfiguration tests implemented and passing. Dynamic committee tests assessed — no additional stress scenarios needed (reconfiguration tests now cover validator lifecycle comprehensively).

| Source File | Tests | Status | Notes |
|-------------|-------|--------|-------|
| `reconfiguration_tests.rs` | 6/6 new | **PASS** | **`test_inactive_validator_pool_read`**: 5 validators, remove one, trigger reconfig — verifies removed validator appears in `inactive_validators` with `deactivation_epoch` set, committee shrinks, node reports as fullnode. **`test_validator_candidate_pool_read`**: 4 validators + 1 candidate via `spawn_new_validator` — verifies candidate in `pending_validators` before reconfig, promoted to active committee after reconfig. **`test_reconfig_with_failing_validator`**: 7 validators, 5s epochs — stops and restarts validators in rotation while network progresses to epoch 4; verifies liveness under validator churn. **`test_create_advance_epoch_tx_race`**: 4 validators, 2s epochs — continuously submits `AddStake` txs while epochs advance to 3; verifies txs execute across epoch boundaries without corruption. **`test_expired_locks`**: Executes `AddStake` in epoch 0, triggers reconfig, attempts stale object ref in epoch 1 (must fail), then fresh tx in epoch 1 (must succeed). **`test_passive_reconfig_with_tx_load`**: 4 validators, 3s epochs — continuous `AddStake` workload until epoch 4; verifies passive epoch transitions under sustained load. |
| `dynamic_committee_tests.rs` | — | **No changes** | Existing `fuzz_dynamic_committee` (12 validators, 20 random stake ops, voting power verification, full unstake cycle) is comprehensive for stake-based committee dynamics. The `StressTestRunner` framework has unused fields (`preactive_validators`, `removed_validators`, `reports`) for future validator add/remove actions, but the new reconfiguration tests already cover those lifecycle scenarios with targeted assertions. |

#### Lessons learned during Tier 4 porting

19. **`TransferCoin` as gas object bug**: When the coin being transferred is also the gas object, `deduct_gas_fee` may delete the gas object (balance becomes 0), causing `CoinExecutor::execute_transfer_coin` to hit `debug_assert!(!self.execution_results.deleted_object_ids.contains(id))` in `TemporaryStore::read_object`. Use `AddStake` as a reliable test workload instead of `TransferCoin` when the coin is also the gas payment.

20. **`AddStake` as universal test workload**: `AddStake` with a validator address is the most reliable transaction workload for testing. It doesn't require extra account setup (works with default genesis accounts), doesn't conflict with gas payment handling, and exercises the staking pipeline. Prefer it over `TransferCoin` for epoch boundary and concurrent transaction tests.

21. **Validator vs fullnode detection**: `AuthorityState` exposes `is_validator(&epoch_store)` and `is_fullnode(&epoch_store)`. After removing a validator and reconfiguring, the node's epoch store reflects its new role. This is useful for asserting role transitions in tests.

22. **`spawn_new_validator` for candidate tests**: `TestCluster::spawn_new_validator(genesis_config)` creates a new validator node with independent keypairs and submits an `AddValidator` transaction. The new validator appears in `pending_validators` immediately but only joins the active committee after `trigger_reconfiguration()`.

23. **Validator crash tolerance**: With 7 validators (can tolerate 2 Byzantine), stopping/restarting individual validators while the network runs demonstrates liveness. Use `tokio::select!` to race a "chaos" task (stop/restart cycle) against `wait_for_epoch(target)` to bound test duration.

### Tier 5: Failpoint-based fault injection tests

Failpoint infrastructure implemented in `utils/src/fp.rs` with macros in `utils/src/lib.rs`. Macros (`fail_point!`, `fail_point_async!`, `fail_point_if!`, `fail_point_arg!`) compile to no-ops unless `cfg(msim)` or `cfg(fail_points)` is set. **10 tests implemented** in `e2e-tests/tests/failpoint_tests.rs`, covering all 9 failpoint tags.

#### Failpoint tags inserted in production code

| Tag | Crate | File | Location |
|-----|-------|------|----------|
| `crash-before-commit-certificate` | authority | `authority.rs` | `try_execute_immediately()` — before `commit_certificate()` |
| `crash-after-consensus-commit` | authority | `consensus_handler.rs` | `handle_consensus_commit()` — after quarantine write |
| `before-open-new-epoch-store` | authority | `authority.rs` | `reopen_epoch_db()` — before `new_at_next_epoch()` |
| `reconfig_delay` | node | `lib.rs` | `monitor_reconfiguration()` — before `reconfigure_state()` |
| `change_epoch_tx_delay` | authority | `authority.rs` | `create_and_execute_advance_epoch_tx()` — before `acquire_tx_lock()` |
| `crash-after-accumulate-epoch` | authority | `checkpoint_executor/mod.rs` | `execute_checkpoint()` — after `accumulate_epoch()` |
| `crash-after-build-batch` | authority | `authority_store.rs` | `build_db_batch()` — before return |
| `crash-after-db-write` | authority | `writeback_cache.rs` | `commit_transaction_outputs()` — after `db_batch.write()` |
| `highest-executed-checkpoint` | authority | `backpressure_manager.rs` | `update_highest_executed_checkpoint()` — before watermark update |

#### Tests in `failpoint_tests.rs` (10 total, all passing)

| # | Test | Failpoint Tags | Type |
|---|------|----------------|------|
| 1 | `test_reconfig_with_delay` | `reconfig_delay` | Delay injection |
| 2 | `test_change_epoch_tx_delay` | `change_epoch_tx_delay` | Delay injection |
| 3 | `test_crash_before_open_new_epoch_store` | `before-open-new-epoch-store` | Crash-once + BFT recovery |
| 4 | `test_crash_before_commit_certificate` | `crash-before-commit-certificate` | Crash-once + BFT recovery |
| 5 | `test_crash_after_consensus_commit` | `crash-after-consensus-commit` | Crash-once + BFT recovery |
| 6 | `test_crash_after_accumulate_epoch` | `crash-after-accumulate-epoch` | Crash-once + BFT recovery |
| 7 | `test_crash_after_db_write` | `crash-after-db-write` | Crash (5th write) + BFT recovery |
| 8 | `test_crash_after_build_batch` | `crash-after-build-batch` | Crash (4th batch) + BFT recovery |
| 9 | `test_highest_executed_checkpoint_failpoint` | `highest-executed-checkpoint` | Observation (hit counter) |
| 10 | `test_crash_during_reconfig_with_tx_load` | `before-open-new-epoch-store` | Crash + concurrent tx load |

#### Future failpoint tests

| Test | Failpoint Tags | Status |
|------|----------------|--------|
| `test_checkpoint_split_brain` | `simulate_fork_during_execution` | Needs execution fork injection tag |
| `test_fork_recovery_transaction_effects` | `simulate_fork_during_execution` + fork recovery | Needs execution fork injection tag |

#### Not porting (failpoint tests)

| Test | Reason |
|------|--------|
| `safe_mode_reconfig_test` | Move framework injection — no safe mode in Soma |
| `test_randomness_*` (3 tests) | No DKG/randomness beacon |
| `test_simulated_load_*` (4 tests) | No benchmark/load harness |
| `test_epoch_flag_upgrade` | No EpochFlag system |

### Not porting

| File | Reason |
|------|--------|
| `object_deletion_tests.rs` | Fully Move-dependent (wrap/unwrap/delete). |
| `overload_monitor_tests.rs` | No overload monitor in Soma currently. |
| `sdk_stream_tests.rs` | No coin metadata or streaming pagination yet. |
| `zklogin_tests.rs` | No ZKLogin. |
| `passkey_e2e_tests.rs` | No passkeys. |
| `alias_tests.rs` | No address aliases. |
| `coin_registry_tests.rs` | Single coin type, no registry. |
| `rpc/v2/move_package_service/*` | No Move. |

---

## Writing New Tests: Checklist

1. **File location**: `e2e-tests/tests/<name>_tests.rs`
2. **Gate with msim**: Every test needs `#[cfg(msim)]` and `#[msim::sim_test]`
3. **Call `init_tracing()`**: First line of every test for log output
4. **Use `TestClusterBuilder`**: Configure validators, accounts, epoch duration
5. **Use native Soma transactions as workloads**: Prefer `AddStake` for most test workloads (reliable, no gas-object conflicts). Also: `PayCoins`, `TransferCoin` (avoid when coin is also gas payment), or model ops — not Move calls
6. **Assert via effects**: `response.effects.status().is_ok()`, check created/mutated/deleted objects
7. **Assert via system state**: `test_cluster.fullnode_handle.soma_node.with(|node| { ... })`
8. **Add to `e2e-tests/Cargo.toml`** if you need new dependencies
9. **Check logs if a test fails**: `e2e-tests/logs/combined.log` and per-node logs
