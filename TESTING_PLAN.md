# Soma Testing Plan — Sui Parity & Beyond

Comprehensive plan for achieving high test parity with Sui (MystenLabs/sui) across all adopted crates, plus Soma-specific test coverage for mining, models, targets, challenges, and epoch transitions.

---

## Audit Notes (Feb 2026)

> **Audited by:** Cross-referencing all 7 sub-plans against the Soma codebase and upstream Sui (MystenLabs/sui).

### Corrected Test Counts

The original summary statistics contained several inaccuracies. Corrected figures:

| Category | Claimed | Actual | Delta |
|----------|---------|--------|-------|
| Consensus Unit Tests | ~100+ | **0** | All test modules are commented out (`#[cfg(test)] mod tests`) |
| Types Unit Tests | ~50+ | **89** | Higher than claimed — strong Soma-specific coverage |
| Store/DB Tests | 27+ | **28** | 24 rocks + 3 macro + 1 util |
| SDK/RPC Tests | 30+ | **29** | Minor overcount |
| **Total** | **~400+** | **~293** | ~27% inflation, primarily from phantom consensus tests |

The most critical inaccuracy is the consensus crate: it has **zero** compiling tests despite the plan claiming "~100+". The test infrastructure (`CommitTestFixture`, `RandomDag`, etc.) was not forked from Sui.

### Priority Ordering (All 7 Sub-Plans)

Ranked by mainnet criticality and effort-to-impact ratio:

| Priority | Plan | Mainnet Blocker? | Est. Effort | Rationale |
|----------|------|-----------------|-------------|-----------|
| **#1** | [Authority Testing](AUTHORITY_TESTING_PLAN.md) | YES | **COMPLETE** | 201 tests passing (was 54). 10 failing pay_coin fixed, 16 test files + consensus test infra. All priorities done except checkpoint builder (deferred to E2E). |
| **#2** | [Consensus Testing](CONSENSUS_TESTING_PLAN.md) | **YES — CRITICAL** | ~7 days | **Zero tests** on the BFT consensus fork. Cannot ship consensus without test coverage. |
| **#3** | [Types Testing](TYPES_TESTING_PLAN.md) | YES | ~14 days | Serialization and crypto are consensus-critical. Zero tests on Sui-derived infra (digests, effects, envelopes). |
| **#4** | [Protocol Config Testing](PROTOCOL_CONFIG_TESTING_PLAN.md) | Soft | ~3 days | Best effort-to-impact ratio. Snapshot regression tests prevent silent config drift. |
| **#5** | [Sync Testing](SYNC_TESTING_PLAN.md) | Soft | ~10 days | Operational resilience. Blocked on `CommitteeFixture` (shared blocker with consensus). |
| **#6** | [CLI/SDK/RPC Testing](CLI_SDK_RPC_TESTING_PLAN.md) | No | ~17 days | User-facing but not consensus-critical. Most ambitious plan (163 new tests proposed). |
| **#7** | [Store Testing](STORE_TESTING_PLAN.md) | No | ~4 days | Already at near-parity with Sui. Lowest urgency. |

**Total estimated effort: ~70 engineering days** (one engineer, sequential). Parallelizable to ~5-6 weeks with 2-3 engineers.

### Mainnet Readiness Assessment

**Verdict: NOT ready for mainnet with current test coverage.**

Critical gaps that must be closed before mainnet:

1. **Consensus has zero tests.** The BFT consensus layer (Mysticeti fork) has no compiling unit tests and no integration tests. This is the single largest risk — a consensus bug means chain halt or fork.

2. ~~**10 failing authority tests.**~~ **FIXED** — All `pay_coin_tests` now pass. Root cause: intent scope mismatch + 0-balance gas coin handling. 4 bugs found and fixed across `types/src/transaction.rs`, `execution/prepare_gas.rs`, `execution/staking.rs`.

3. **No serialization regression tests.** BCS serialization for consensus-critical types (transactions, effects, digests, checkpoints) has no roundtrip or snapshot tests. A serialization change could cause network splits.

4. ~~**Epoch transitions tested only at E2E level.**~~ **FIXED** — 9 unit tests now cover `advance_epoch()` arithmetic, safe mode fallback/recovery, emission pool, difficulty adjustment, and u128 overflow protection.

**What IS strong:**
- E2E test suite (93 tests) covers the happy path comprehensively
- Failpoint infrastructure enables crash/delay simulation
- Safe mode provides liveness guarantee during epoch failures
- Soma-specific domain logic (challenges, targets, submissions) has good coverage

**Recommended mainnet gate:**
- All Priority #1-#3 plans implemented (authority, consensus, types)
- Priority #4 snapshot tests in place
- Zero failing tests
- All E2E tests passing

### Gaps Not Covered by Any Sub-Plan

The following testing categories are absent from all 7 sub-plans and should be considered for a comprehensive mainnet testing strategy:

1. **Load/Stress Testing** — No plan tests throughput limits, memory pressure, or degradation under sustained load. Critical for understanding mainnet capacity.

2. **Adversarial/Fuzz Testing** — No property-based testing (proptest/quickcheck) or fuzzing for parsing, deserialization, or transaction validation. Important for security hardening.

3. **Economic Invariant Testing** — No tests verify that total token supply is conserved across epoch transitions, that emission pool + staked + circulating = genesis supply, or that BPS allocations always sum to ≤ 10000.

4. **Long-Running Soak Tests** — No multi-hour tests that verify the system doesn't leak memory, accumulate state, or degrade over hundreds of epochs.

5. **Network Partition / Byzantine Behavior** — While failpoint tests cover crash recovery, no tests simulate sustained network partitions, message reordering, or Byzantine validator behavior beyond simple crashes.

6. **Upgrade Path Testing** — Protocol version tests exist in E2E, but no tests verify that a running network can upgrade from version N to N+1 without state corruption or downtime (rolling upgrade simulation).

7. **Determinism Verification** — No tests run the same workload twice and verify identical state roots, which is essential for a deterministic blockchain.

---

## Table of Contents
0. [Audit Notes (Feb 2026)](#audit-notes-feb-2026) — **START HERE** — Corrected counts, priority ordering, mainnet readiness, gaps
1. [Current Soma Test Coverage](#current-soma-test-coverage)
2. [Priority 1: Execution Engine Tests](#priority-1-execution-engine-tests)
3. [Priority 2: Authority Unit Tests](#priority-2-authority-unit-tests)
4. [Priority 3: Genesis Tests](#priority-3-genesis-tests)
5. [Priority 4: Consensus Tests (msim)](#priority-4-consensus-tests-msim)
6. [Priority 5: State Sync & Discovery Tests](#priority-5-state-sync--discovery-tests)
7. [Priority 6: Types Crate Tests](#priority-6-types-crate-tests)
8. [Priority 7: Supporting Crate Tests](#priority-7-supporting-crate-tests)
9. [Soma-Specific Tests (Non-Sui)](#soma-specific-tests-non-sui)
10. [Licensing Attribution](#licensing-attribution)
11. [Build & Run Commands](#build--run-commands)
12. [Implementation Order (Revised)](#implementation-order-revised)

---

## Current Soma Test Coverage

### Summary Statistics (as of Feb 2026)

| Category | Test Count | Notes |
|----------|-----------|-------|
| E2E Tests (e2e-tests) | 93 across 14 files | Comprehensive msim tests |
| Authority Unit Tests | **201** | **ALL PASSING** — 16 test files + consensus test infra + 52 pre-existing (pay_coin fixed, gas, transfer, staking, authority, validator, model, epoch, tx validation, submission, server, batch, epoch store, execution driver, batch verification, consensus) |
| Types Unit Tests | **89** | Rewards, challenge, target, tensor, grpc, multiaddr *(corrected from ~50+)* |
| Store/DB Tests | **28** | 24 RocksDB + 3 macro + 1 util *(corrected from 27+)* |
| SDK/RPC Tests | **29** | Proxy client, field masks, serde *(corrected from 30+)* |
| Network/TLS Tests | 7 | TLS verification, HTTP server |
| Utils Tests | 8 | Failpoints, notify |
| Protocol Config Tests | 8 | Tensor operations |
| CLI Tests | 2 | Formatting |
| Blobs Tests | 11+ | Download, transfer, engine |
| Consensus Unit Tests | **0** | **All test modules commented out** *(corrected from ~100+)* |
| **Total** | **~414** | *(was ~293 before authority test expansion; consensus crate still has 0 tests)* |

### Existing E2E Test Files

| File | Tests | Coverage |
|------|-------|----------|
| challenge_tests.rs | 14 | Challenge initiation, bonds, tally resolution, quorum |
| failpoint_tests.rs | 13 | 6 crash points, delays, safe mode (3), race conditions |
| reconfiguration_tests.rs | 13 | Epoch transitions, validator join/leave, voting power |
| shared_object_tests.rs | 8 | Target/SystemState mutations, racing miners, versioning |
| transaction_orchestrator_tests.rs | 8 | Submission, WAL, epoch boundaries, stale objects |
| full_node_tests.rs | 7 | Fullnode sync, orchestrator, RunWithRange, BCS |
| rpc_tests.rs | 7 | gRPC endpoints, system state, balance, objects |
| target_tests.rs | 6 | Generation, submission, challenge window, expiration |
| protocol_version_tests.rs | 6 | Upgrades, quorum, laggard shutdown |
| simulator_tests.rs | 5 | Determinism, ordering, HashMap iteration |
| model_tests.rs | 2 | Commit-reveal, staking |
| checkpoint_tests.rs | 2 | Timestamp monotonicity, fork detection |
| multisig_tests.rs | 2 | Ed25519, weighted thresholds |

### Authority Unit Tests (Updated Feb 2026)

| File | Tests | Coverage |
|------|-------|----------|
| unit_tests/pay_coin_tests.rs | 10 | **ALL PASSING** — coin payment success/failure (was 10 failing, fixed) |
| unit_tests/authority_tests.rs | 16 | Transfer, double-spend, signatures, effects, objects, system state, shared objects |
| unit_tests/gas_tests.rs | 14 | Base fee, insufficient gas, smashing, operation/value fees, zero-balance |
| unit_tests/model_tests.rs | 10 | CommitModel success/failure, fees, coin deduction, management |
| unit_tests/staking_tests.rs | 9 | AddStake/WithdrawStake, half value fee, gas coin awareness |
| unit_tests/epoch_tests.rs | 9 | advance_epoch arithmetic, safe mode, emissions, overflow |
| unit_tests/submission_tests.rs | 8 | SubmitData validation, ClaimRewards edge cases, report authorization |
| unit_tests/epoch_store_tests.rs | 8 | Epoch properties, signed tx storage, effects signatures, reconfig |
| unit_tests/transfer_coin_tests.rs | 7 | TransferCoin success/failure, gas-is-transfer-coin, pay-all |
| unit_tests/validator_tests.rs | 7 | SetCommissionRate, UpdateValidatorMetadata, ReportValidator |
| unit_tests/transaction_validation_tests.rs | 7 | System tx behavior, gas edge cases, BCS, digest determinism |
| unit_tests/execution_driver_tests.rs | 6 | Scheduler enqueue, shared object versioning, dependency ordering, idempotency |
| unit_tests/batch_verification_tests.rs | 5 | Batch cert verify, caching, async verify, multi-verify, sender sig |
| unit_tests/consensus_tests.rs | 5 | Handler processing, deduplication, shared obj versions, checkpoints, multi-tx |
| unit_tests/server_tests.rs | 5 | ValidatorService handle_transaction, object/tx info |
| unit_tests/batch_transaction_tests.rs | 4 | Sequential transfers, rollback, effects accumulation, versioning |
| execution/challenge.rs | 16 | Bond calculation, reward distribution |
| audit_service.rs | 6 | Mock API, tolerance, fraud detection |
| proxy_server.rs | 20 | Config, parsing, error display |
| checkpoints/causal_order.rs | 2 | Causal ordering |
| transaction_driver/moving_window.rs | 4 | Moving window stats |
| transaction_driver/backoff.rs | 3 | Exponential backoff |
| fullnode_proxy.rs | 2 | Object ID parsing |
| authority_per_epoch_store_pruner.rs | 1 | Epoch pruning |

---

## Priority 1: Execution Engine Tests

### 1A. Fix Failing pay_coin_tests (10 tests) — CRITICAL

**Soma file:** `authority/src/unit_tests/pay_coin_tests.rs`
**Sui equivalent:** `crates/sui-core/src/unit_tests/pay_sui_tests.rs`

The 10 failing tests mirror Sui's `pay_sui_tests.rs` which contains 12 tests:

| Sui Test | Soma Equivalent | Status |
|----------|----------------|--------|
| `test_pay_sui_failure_empty_recipients` | — | Missing |
| `test_pay_sui_failure_insufficient_gas_balance_one_input_coin` | `test_pay_coin_failure_insufficient_gas_one_input_coin` | FAILING |
| `test_pay_sui_failure_insufficient_total_balance_one_input_coin` | `test_pay_coin_failure_insufficient_total_balance_one_input_coin` | FAILING |
| `test_pay_sui_failure_insufficient_gas_balance_multiple_input_coins` | `test_pay_coin_failure_insufficient_gas_multiple_input_coins` | FAILING |
| `test_pay_sui_failure_insufficient_total_balance_multiple_input_coins` | — | Missing |
| `test_pay_sui_success_one_input_coin` | `test_pay_coin_success_one_input_coin` | FAILING |
| `test_pay_sui_success_multiple_input_coins` | `test_pay_coin_success_multiple_input_coins` | FAILING |
| `test_pay_all_sui_failure_insufficient_gas_one_input_coin` | `test_pay_all_coins_failure_insufficient_gas_one_input_coin` | FAILING |
| `test_pay_all_sui_failure_insufficient_gas_budget_multiple_input_coins` | `test_pay_all_coins_failure_insufficient_gas_multiple_input_coins` | FAILING |
| `test_pay_all_sui_success_one_input_coin` | `test_pay_all_coins_success_one_input_coin` | FAILING |
| `test_pay_all_sui_success_multiple_input_coins` | `test_pay_all_coins_success_multiple_input_coins` | FAILING |

**Key implementation details from Sui's tests:**
- Tests use `TestAuthorityBuilder::new().build().await` for standalone authority state
- `ProgrammableTransactionBuilder` with `pay_sui()` and `pay_all_sui()` methods
- Gas budget validation happens via `UserInputError::GasBalanceTooLow`
- Insufficient balance during execution returns `ExecutionFailureStatus::InsufficientCoinBalance`
- Success tests verify: created objects count, recipient ownership, gas deduction, coin deletion (for merge)
- `execute_pay_all_sui` uses `init_state_with_committee()` with genesis objects

**Action items:**
1. Debug why all 10 tests fail — likely a gas model or coin smashing difference
2. Verify Soma's `TransferCoin` and `PayCoins` executor logic matches Sui's `pay_sui` PTB behavior
3. Ensure value fees are computed correctly (check `prepare_gas.rs`)
4. Add missing `test_pay_coin_failure_empty_recipients` test
5. Add missing `test_pay_coin_failure_insufficient_total_balance_multiple_input_coins` test

### 1B. Gas Tests — NEW

**Sui file:** `crates/sui-core/src/unit_tests/gas_tests.rs`
**Soma file to create:** `authority/src/unit_tests/gas_tests.rs`

Sui's gas tests cover:
- Gas price validation (below/above reference gas price)
- Gas budget limits (too low, too high)
- Gas coin object handling (smashing, splitting)
- Gas cost summary correctness
- Storage rebate calculations
- Gas coin as transfer coin edge cases

**Tests to implement:**

| Test | Description |
|------|------------|
| `test_gas_price_below_rgp` | Reject tx with gas price below reference |
| `test_gas_price_above_rgp` | Accept tx with higher gas price |
| `test_gas_budget_too_low` | Reject tx with insufficient gas budget |
| `test_gas_budget_too_high` | Reject tx with excessive gas budget |
| `test_gas_smashing_multiple_coins` | Verify multiple gas coins are merged |
| `test_gas_coin_is_transfer_coin` | Handle case where gas coin == transfer coin |
| `test_gas_balance_after_failed_execution` | Verify gas deducted on failure |
| `test_gas_refund_on_success` | Verify unused gas is refunded |
| `test_value_fee_deduction` | Verify operation-specific value fees |
| `test_base_fee_dos_protection` | Verify base fee is always deducted |

### 1C. Staking Execution Tests — NEW

**Sui file:** `crates/sui-core/src/unit_tests/` (staking-related tests in authority_tests.rs)
**Soma file to create:** `authority/src/unit_tests/staking_tests.rs`

| Test | Description |
|------|------------|
| `test_add_stake_basic` | Stake tokens with a validator |
| `test_add_stake_insufficient_balance` | Insufficient funds for staking |
| `test_withdraw_stake_basic` | Withdraw staked tokens |
| `test_withdraw_stake_before_epoch_end` | Cannot withdraw in same epoch |
| `test_stake_rewards_calculation` | Verify reward distribution after epoch |
| `test_model_stake_basic` | Stake tokens to a model |
| `test_model_stake_commission` | Verify model commission rate |
| `test_validator_slashing_on_report_quorum` | Rewards slashed when reported |

### 1D. Advance Epoch & Rewards Tests — NEW

**Soma file to create:** `authority/src/unit_tests/epoch_tests.rs`

| Test | Description |
|------|------------|
| `test_advance_epoch_basic` | Normal epoch transition |
| `test_advance_epoch_distributes_rewards` | Validator rewards distributed correctly |
| `test_advance_epoch_emissions_allocation` | Emission pool allocation per epoch |
| `test_advance_epoch_validator_reward_bps` | BPS allocation to validators |
| `test_advance_epoch_model_registry_processing` | Model reveals, deactivations processed |
| `test_advance_epoch_target_generation` | New targets generated at boundary |
| `test_advance_epoch_difficulty_adjustment` | Hit rate EMA -> distance threshold |
| `test_advance_epoch_safe_mode_fallback` | Safe mode on advance_epoch failure |
| `test_advance_epoch_safe_mode_recovery` | Recovery from safe mode drains accumulators |
| `test_advance_epoch_u128_overflow_protection` | BPS calculations use u128 intermediates |
| `test_advance_epoch_reported_validator_slashing` | Reported validators get rewards slashed |
| `test_advance_epoch_model_report_quorum_slash` | Model deactivated on report quorum |

### 1E. Validator Registration/Deregistration Tests — NEW

**Soma file to create:** `authority/src/unit_tests/validator_tests.rs`

| Test | Description |
|------|------------|
| `test_add_validator` | Register new validator |
| `test_remove_validator` | Remove validator from set |
| `test_update_validator_metadata` | Update validator info |
| `test_set_commission_rate` | Change validator commission |
| `test_report_validator` | Report misbehaving validator |
| `test_undo_report_validator` | Retract validator report |
| `test_validator_cannot_remove_below_threshold` | Minimum validator count |

---

## Priority 2: Authority Unit Tests

### 2A. Authority State Tests — NEW

**Sui file:** `crates/sui-core/src/unit_tests/authority_tests.rs` (~5000+ lines, ~100+ tests)
**Soma file to create:** `authority/src/unit_tests/authority_tests.rs`

Sui's `authority_tests.rs` is their largest test file. Key test categories to implement for Soma:

| Category | Tests to Implement |
|----------|-------------------|
| **Transaction submission** | `test_handle_transfer_coin_with_max_gas`, `test_handle_transfer_transaction_unknown_sender`, `test_handle_transfer_transaction_bad_signature` |
| **Object handling** | `test_handle_transfer_transaction_double_spend`, `test_object_not_found`, `test_get_latest_parent_entry` |
| **Certificate processing** | `test_handle_confirmation_transaction_ok`, `test_handle_confirmation_transaction_bad_sig` |
| **Shared objects** | `test_handle_shared_object_transaction`, `test_shared_object_version_assignment` |
| **Epoch boundary** | `test_change_epoch_transaction`, `test_epoch_store_isolation` |
| **Error cases** | `test_transaction_gas_errors`, `test_immutable_object_mutation_rejected` |

**Sui's pattern:** Uses `TestAuthorityBuilder::new().build().await` for standalone authority state. Tests produce snapshot files (`.snap`) for execution error outputs using the `insta` crate. Snapshot names like `sui_core__authority__authority_tests__clever_abort_with_code_abort_error.snap` indicate error categorization tests.

**Key Sui authority test categories (from snapshot analysis):**
- Abort error classification (clever_only_abort, clever_abort_with_const, clever_abort_with_code)
- Execution error source tracking
- Dry run / dev inspect (dynamic field edge cases)

### 2B. Transaction Tests — NEW

**Sui file:** `crates/sui-core/src/unit_tests/transaction_tests.rs`
**Soma file to create:** `authority/src/unit_tests/transaction_tests.rs`

| Test | Description |
|------|------------|
| `test_transaction_data_serialization` | BCS round-trip for all tx types |
| `test_transaction_digest_determinism` | Same inputs -> same digest |
| `test_sender_signed_data_verification` | Signature verification |
| `test_certificate_aggregation` | Quorum signature assembly |
| `test_effects_creation` | TransactionEffects from execution results |
| `test_all_31_transaction_kinds` | Each TransactionKind variant round-trips |

### 2C. Consensus Handler Tests — NEW

**Sui file:** `crates/sui-core/src/consensus_handler.rs` (inline tests)
**Soma file:** `authority/src/consensus_handler.rs`

| Test | Description |
|------|------------|
| `test_consensus_handler_deduplication` | Same tx not executed twice |
| `test_shared_object_version_assignment` | Shared objects get correct versions |
| `test_consensus_commit_prologue_creation` | Prologue tx created per commit |
| `test_consensus_handler_epoch_boundary` | Handler stops at epoch end |

---

## Priority 3: Genesis Tests

### 3A. Genesis Builder Tests — NEW

**Soma file to create:** `types/src/unit_tests/genesis_tests.rs`

| Test | Description |
|------|------------|
| `test_genesis_creates_system_state` | SystemState object exists after genesis |
| `test_genesis_creates_validators` | All validators in ValidatorSet |
| `test_genesis_creates_initial_coins` | Account allocations correct |
| `test_genesis_creates_seed_models` | Seed models in ModelRegistry |
| `test_genesis_creates_initial_targets` | Initial targets generated |
| `test_genesis_emission_pool` | EmissionPool initialized correctly |
| `test_genesis_protocol_version` | Protocol version set correctly |
| `test_genesis_deterministic` | Same config -> same genesis |
| `test_genesis_builder_custom_parameters` | SystemParameters respected |
| `test_genesis_builder_multiple_validators` | 4, 7, 10 validator configs |

**Cross-reference files:**
- Soma: `types/src/genesis_builder.rs` ← Sui: `crates/sui-genesis-builder/src/lib.rs`
- Soma: `types/src/config/genesis_config.rs` ← Sui: `crates/sui-swarm-config/src/genesis_config.rs`

---

## Priority 4: Consensus Tests (msim)

### 4A. Consensus Simtests — ADAPT FROM SUI

**Sui files:**
- `consensus/simtests/tests/consensus_tests.rs`
- `consensus/simtests/tests/consensus_dag_tests.rs`

**Soma equivalent:** `consensus/` crate already has extensive inline unit tests. Need to add msim-level integration tests.

Sui's consensus simtests include:

| Test | Description | Soma Priority |
|------|-------------|---------------|
| `test_consensus_basic` | Basic 4-validator consensus | High |
| `test_consensus_with_crashes` | Consensus survives validator crashes | High |
| `test_consensus_with_network_partitions` | Handles network splits | Medium |
| `test_consensus_commit_ordering` | Commit ordering guarantees | High |
| `test_consensus_dag_recovery` | DAG rebuilds after crash | Medium |
| `test_consensus_leader_rotation` | Leader schedule works | Medium |
| `test_consensus_byzantine_validators` | Handles Byzantine behavior | Low |

**Soma currently has:** `#[cfg(test)] mod tests` blocks exist in many consensus source files (`core.rs`, `core_thread.rs`, `block_manager.rs`, `universal_committer.rs`, `base_committer.rs`, `ancestor.rs`, `authority_node.rs`, `synchronizer.rs`, `leader_schedule.rs`, `commit_finalizer.rs`, `dag_state.rs`, `commit_syncer.rs`, `threshold_clock.rs`) but **all are commented out or empty**. Zero tests compile.

**Critical blocker:** The `CommitTestFixture` and `RandomDag` test infrastructure from Sui's `consensus/core/src/tests/` was **not forked**. This must be ported before most consensus tests can be enabled.

**Gap:** Need both (1) unit tests re-enabled by porting test fixtures, and (2) msim integration tests for the full consensus stack under simulated conditions (crashes, delays, partitions).

### 4C. Consensus Randomized Tests — NEW

**Sui file:** `consensus/core/src/tests/randomized_tests.rs`
**Soma file to create:** `consensus/src/tests/randomized_tests.rs`

Sui has 2 randomized DAG tests that are critical for confidence in consensus correctness:

| Test | Description | Priority |
|------|-------------|----------|
| `test_randomized_dag_all_direct_commit` | Builds randomized DAG with 2f+1 ancestors + 100% leader inclusion across 1000 rounds. Verifies all rounds produce direct commits. Runs 100 iterations per seed. | High |
| `test_randomized_dag_and_decision_sequence` | Builds randomized DAG with 50% leader inclusion. Delivers blocks via `RandomDagIterator` in constrained random order (simulating realistic block arrival). Verifies commit sequences match across different delivery orders. Tests incrementally through Linearizer and CommitFinalizer. | High |

**Key infrastructure from Sui:**
- `CommitTestFixture` — Test harness for commit logic
- `RandomDag` / `RandomDagIterator` — Random block delivery simulation
- `create_random_dag()` — Deterministic DAG generation with seed
- `assert_commit_sequences_match()` — Cross-run consistency verification
- Uses `DAG_TEST_SEED` env var for reproducibility

### 4D. Consensus Handler Test Utilities ✅ DONE

**Sui file:** `crates/sui-core/src/unit_tests/consensus_test_utils.rs`
**Soma file:** `authority/src/consensus_test_utils.rs` — **IMPLEMENTED**

Soma now provides full consensus handler test infrastructure:
- `TestConsensusCommit` — Mock consensus commit implementing `ConsensusCommitAPI`
- `NoopConsensusClient` — No-op `ConsensusClient` that returns `BlockStatus::Sequenced` immediately
- `setup_consensus_handler_for_testing()` — Full handler setup with captured transactions
- `CapturedTransactions` — Thread-safe capture of `(Vec<Schedulable>, AssignedTxAndVersions, SchedulingSource)` tuples
- 5 consensus handler tests in `unit_tests/consensus_tests.rs` exercise: processing, deduplication, shared object versioning, pending checkpoints, and multi-tx commits

### 4B. Consensus Core Unit Tests — VERIFY PARITY

Cross-reference Soma's existing consensus unit tests against Sui's:

| Sui Module | Soma Module | Status |
|-----------|-------------|--------|
| `consensus/core/src/core.rs` tests | `consensus/src/core.rs` | Verify parity |
| `consensus/core/src/block_manager.rs` tests | `consensus/src/block_manager.rs` | Verify parity |
| `consensus/core/src/dag_state.rs` tests | `consensus/src/dag_state.rs` | Verify parity |
| `consensus/core/src/universal_committer.rs` tests | `consensus/src/universal_committer.rs` | Verify parity |
| `consensus/core/src/base_committer.rs` tests | `consensus/src/base_committer.rs` | Verify parity |
| `consensus/core/src/leader_schedule.rs` tests | `consensus/src/leader_schedule.rs` | Verify parity |
| `consensus/core/src/synchronizer.rs` tests | `consensus/src/synchronizer.rs` | Verify parity |
| `consensus/core/src/commit_syncer.rs` tests | `consensus/src/commit_syncer.rs` | Verify parity |
| `consensus/core/src/ancestor.rs` tests | `consensus/src/ancestor.rs` | Verify parity |

---

## Priority 5: State Sync & Discovery Tests

### 5A. State Sync Tests — ADAPT FROM SUI

**Sui files:**
- `crates/sui-network/src/state_sync/tests.rs` — Unit tests for state sync protocol
- `crates/sui-e2e-tests/tests/state_sync_resilience_tests.rs` — E2E resilience tests

**Soma equivalent:** `sync/` crate

| Test | Description | Priority |
|------|-------------|----------|
| `test_state_sync_basic` | Fullnode syncs with validators | High |
| `test_state_sync_checkpoint_download` | Download checkpoints from peers | High |
| `test_state_sync_catch_up` | Catch up after falling behind | High |
| `test_state_sync_peer_selection` | Select best peers for sync | Medium |
| `test_state_sync_resilience_slow_peer` | Handle slow sync peers | Medium |
| `test_state_sync_resilience_offline_peer` | Handle offline peers | Medium |
| `test_state_sync_epoch_boundary` | Sync across epoch transitions | High |

### 5B. Discovery Tests — ADAPT FROM SUI

**Sui file:** `crates/sui-network/src/discovery/` module tests
**Soma equivalent:** `sync/` crate (peer discovery)

| Test | Description | Priority |
|------|-------------|----------|
| `test_peer_discovery_basic` | Discover peers from seed | High |
| `test_peer_discovery_reconnect` | Reconnect to lost peers | Medium |
| `test_peer_discovery_committee_change` | Update peers on epoch boundary | High |

---

## Priority 6: Types Crate Tests

### 6A. Existing Sui Unit Test Files to Match

**Sui directory:** `crates/sui-types/src/unit_tests/`

| Sui Test File | Soma Equivalent | Status |
|--------------|-----------------|--------|
| `base_types_tests.rs` | `types/src/unit_tests/` | Need to add |
| `messages_tests.rs` | `types/src/unit_tests/` | Need to add |
| `multisig_tests.rs` | `types/src/unit_tests/multisig_tests.rs` | EXISTS |
| `intent_tests.rs` | — | Need to add |
| `effects_tests.rs` | — | Need to add |
| `transaction_serialization_tests.rs` | — | Need to add |
| `transaction_claims_tests.rs` | — | Need to add (adapt for Soma tx types) |
| `passkey_authenticator_test.rs` | — | N/A (Sui-specific) |
| `address_balance_gas_tests.rs` | — | Need to add |
| `event_filter_tests.rs` | — | N/A (Sui-specific) |
| `balance_withdraw_tests.rs` | — | Adapt for Soma staking |

**Tests to implement in `types/src/unit_tests/`:**

| Test File | Tests | Description |
|-----------|-------|-------------|
| `base_types_tests.rs` | 10+ | SomaAddress, ObjectID, Digest serialization/hashing |
| `transaction_tests.rs` | 15+ | All 31 TransactionKind variants: BCS round-trip, digest determinism |
| `effects_tests.rs` | 5+ | TransactionEffects creation, status codes, object changes |
| `intent_tests.rs` | 5+ | Intent signing, scope validation |
| `checkpoint_tests.rs` | 5+ | CheckpointSummary, inclusion proofs, content hashing |

### 6B. Existing Soma Types Tests (verify completeness)

| File | Tests | Status |
|------|-------|--------|
| `system_state/unit_tests/rewards_distribution_tests.rs` | 9 | Complete |
| `challenge.rs` inline tests | 3 | Good |
| `target.rs` inline tests | 4 | Good |
| `system_state/target_state.rs` inline tests | 6 | Good |
| `grpc_timeout.rs` inline tests | 10 | Good |
| `multiaddr.rs` inline tests | 6 | Good |
| `tls/mod.rs` inline tests | 5 | Good |

---

## Priority 7: Supporting Crate Tests

### 7A. Protocol Config Tests

**Sui file:** `crates/sui-protocol-config/src/lib.rs` (inline tests)
**Soma file:** `protocol-config/src/lib.rs`

Sui's protocol config crate has **126+ snapshot files** for regression testing across Mainnet, Testnet, and generic versions (74-111+). This ensures config changes are intentional and backwards-compatible.

| Test | Description | Priority |
|------|-------------|----------|
| `test_protocol_config_version_1` | Version 1 params are correct | High |
| `test_protocol_config_snapshot` | Snapshot test for backwards compat (use `insta` crate) | High |
| `test_protocol_config_all_fields_set` | No `None` fields in active version | Medium |
| `test_protocol_config_feature_flags` | Feature flag toggling | Medium |
| `test_protocol_config_snapshot_per_version` | One snapshot per protocol version for regression | Medium |

### 7B. Store Tests — VERIFY PARITY

**Soma:** `store/src/rocks/tests.rs` (24 tests) — likely good coverage
**Sui:** `crates/sui-storage/` tests

Existing Soma store tests cover: open, reopen, get, multi_get, insert_batch, delete_batch, delete_range, iter, checkpoint. This appears comprehensive.

**Gap:** Authority store integration tests (object locking, certificate storage, effects storage).

### 7C. RPC Tests — EXTEND

**Soma:** `rpc/` crate has some inline tests for field masks and serde
**Sui:** `crates/sui-json-rpc/src/` (5 files with inline `mod tests`: `move_utils.rs`, `governance_api.rs`, `coin_api.rs`, `read_api.rs`, `error.rs`), `crates/sui-json-rpc-api/` (5 files), `crates/sui-e2e-tests/tests/rpc/`

| Test | Description | Priority |
|------|-------------|----------|
| `test_rpc_get_system_state` | Verify system state RPC response | Medium |
| `test_rpc_execute_transaction` | Submit tx via RPC and verify | Medium |
| `test_rpc_simulate_transaction` | Simulate without execution | Medium |
| `test_rpc_list_targets` | List targets with filters | Low |
| `test_rpc_get_challenge` | Get challenge by ID | Low |

### 7D. CLI Tests — EXTEND

**Soma:** `cli/src/response.rs` (2 tests)
**Sui:** `crates/sui/tests/cli_tests.rs` (extensive), `crates/sui/tests/ptb_files_tests.rs`, plus 212+ shell test scripts in `crates/sui/tests/shell_tests/` (publish, upgrade, PTB, ephemeral, chain ID caching, etc.)

| Test | Description | Priority |
|------|-------------|----------|
| `test_cli_balance_command` | `soma balance` output formatting | Medium |
| `test_cli_send_command_parsing` | `soma send` argument parsing | Medium |
| `test_cli_stake_command` | `soma stake` with validator | Medium |
| `test_cli_model_commands` | `soma model commit/reveal` parsing | Low |
| `test_cli_target_commands` | `soma target list` formatting | Low |

### 7E. SDK Tests — VERIFY

**Soma:** `sdk/src/proxy_client.rs` (19 tests) — good coverage for proxy client

**Gap:** Need SDK integration tests for transaction building and submission.

### 7F. soma-tls / soma-keys / soma-http

**Soma:** `soma-tls/src/lib.rs` (5 tests), `soma-http/src/lib.rs` (2 tests)
**Sui:** Corresponding crates in `crates/sui-tls/`, `crates/sui-keys/`

These appear to have reasonable coverage. Verify parity with Sui's corresponding tests.

### 7G. Data Ingestion Tests

**Soma:** `data-ingestion/` crate
**Sui:** `crates/sui-data-ingestion/`

| Test | Description | Priority |
|------|-------------|----------|
| `test_data_ingestion_basic` | Process checkpoint data | Low |
| `test_data_ingestion_missing_checkpoint` | Handle gaps | Low |

---

## Soma-Specific Tests (Non-Sui)

These tests cover Soma-unique functionality that has no Sui equivalent.

### Targets

| Test | Description | File |
|------|-------------|------|
| `test_target_expiration_edge_cases` | Target expiration exactly at epoch boundary | e2e-tests |
| `test_target_generation_empty_model_registry` | No targets when no models exist | e2e-tests |
| `test_target_generation_single_model` | All targets assign to single model | e2e-tests |
| `test_target_reward_per_target_calculation` | Correct emission division | unit test |
| `test_target_distance_threshold_bounds` | Min/max clamping | unit test |
| `test_target_spawn_on_fill_emission_depleted` | No replacement when pool empty | e2e-tests |
| `test_target_deterministic_embedding` | Same seed -> same embedding | unit test |
| `test_target_model_selection_stake_weighted` | Higher stake -> more targets | unit test |

### Submissions

| Test | Description | File |
|------|-------------|------|
| `test_concurrent_submissions_same_target` | Only first submission succeeds | e2e-tests |
| `test_submission_bond_calculation` | Bond = bond_per_byte * data_size | unit test |
| `test_submission_gas_edge_cases` | Gas + bond exhaustion | e2e-tests |
| `test_submission_wrong_model` | Reject submission with wrong model | e2e-tests |
| `test_submission_distance_exceeds_threshold` | Reject when distance too large | unit test |
| `test_submission_expired_target` | Cannot submit to expired target | e2e-tests |
| `test_claim_rewards_distribution_bps` | 50% miner, 30% model, 1% claimer | unit test |
| `test_claim_rewards_fraud_quorum` | Bond forfeited on quorum | e2e-tests |

### Epoch Boundary (Soma-specific)

| Test | Description | File |
|------|-------------|------|
| `test_model_deactivation_at_epoch_boundary` | Model deactivated via report quorum | e2e-tests |
| `test_model_reveal_timeout` | Unrevealed model cleaned up | e2e-tests |
| `test_target_cleanup_at_epoch_boundary` | Expired targets returned to pool | e2e-tests |
| `test_hit_rate_ema_update` | EMA calculation correctness | unit test |
| `test_difficulty_increase_on_high_hit_rate` | Threshold decreases | unit test |
| `test_difficulty_decrease_on_low_hit_rate` | Threshold increases | unit test |
| `test_emission_pool_depletion` | Chain behavior when emissions run out | e2e-tests |

---

## Licensing Attribution

### Files Requiring Sui/Mysten Labs License Headers

The following Soma source files are adopted or heavily derived from Sui (MystenLabs/sui) and require proper Apache 2.0 license attribution:

```
// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Modified for the Soma project.
```

#### Types Crate (`types/src/`)

| Soma File | Sui Equivalent | Derivation Level |
|-----------|---------------|-----------------|
| `types/src/checkpoints.rs` | `sui-types/src/messages_checkpoint.rs` | Heavy |
| `types/src/digests.rs` | `sui-types/src/digests.rs` | Heavy |
| `types/src/effects/mod.rs` | `sui-types/src/effects/mod.rs` | Heavy |
| `types/src/effects/object_change.rs` | `sui-types/src/effects/object_change.rs` | Heavy |
| `types/src/signature_verification.rs` | `sui-types/src/signature_verification.rs` | Heavy |
| `types/src/multisig.rs` | `sui-types/src/multisig.rs` | Heavy |
| `types/src/serde.rs` | `sui-types/src/serde.rs` | Heavy |
| `types/src/full_checkpoint_content.rs` | `sui-types/src/full_checkpoint_content.rs` | Heavy |
| `types/src/transaction.rs` | `sui-types/src/transaction.rs` | Moderate (extended with Soma tx types) |
| `types/src/object.rs` | `sui-types/src/object.rs` | Moderate (extended with Soma object types) |
| `types/src/envelope.rs` | `sui-types/src/envelope.rs` | Heavy |
| `types/src/intent.rs` | `sui-types/src/intent.rs` | Heavy |
| `types/src/crypto/` | `sui-types/src/crypto.rs` | Heavy |
| `types/src/base/` | `sui-types/src/base_types.rs` | Heavy |
| `types/src/committee.rs` | `sui-types/src/committee.rs` | Heavy |
| `types/src/error.rs` | `sui-types/src/error.rs` | Heavy |
| `types/src/messages_grpc.rs` | `sui-types/src/messages_grpc.rs` | Heavy |
| `types/src/balance_change.rs` | `sui-types/src/balance_change.rs` | Direct copy |
| `types/src/client.rs` | `sui-types/src/client.rs` | Heavy |
| `types/src/config/` (all files) | `sui-swarm-config/` + `sui-config/` | Heavy |
| `types/src/consensus/` (all files) | `consensus/core/src/` types | Heavy |
| `types/src/system_state/validator.rs` | `sui-types/src/sui_system_state/` | Moderate |
| `types/src/system_state/staking.rs` | `sui-types/src/sui_system_state/` | Moderate |
| `types/src/unit_tests/multisig_tests.rs` | `sui-types/src/unit_tests/multisig_tests.rs` | Heavy |
| `types/src/unit_tests/utils.rs` | `sui-types/src/unit_tests/utils.rs` | Heavy |

#### Authority Crate (`authority/src/`)

| Soma File | Sui Equivalent | Derivation Level |
|-----------|---------------|-----------------|
| `authority/src/authority.rs` | `sui-core/src/authority.rs` | Heavy |
| `authority/src/authority_server.rs` | `sui-core/src/authority_server.rs` | Heavy |
| `authority/src/consensus_handler.rs` | `sui-core/src/consensus_handler.rs` | Heavy |
| `authority/src/execution/mod.rs` | `sui-core/src/execution_engine.rs` | Moderate |
| `authority/src/execution/prepare_gas.rs` | Sui gas logic | Moderate |
| `authority/src/execution/coin.rs` | Sui PTB pay logic | Moderate |
| `authority/src/checkpoints/` | `sui-core/src/checkpoints/` | Heavy |
| `authority/src/transaction_driver/` | `sui-core/src/transaction_manager.rs` | Moderate |
| `authority/src/epoch_store.rs` | `sui-core/src/authority_per_epoch_store.rs` | Heavy |
| `authority/src/authority_store.rs` | `sui-core/src/authority_store.rs` | Heavy |
| `authority/src/unit_tests/pay_coin_tests.rs` | `sui-core/src/unit_tests/pay_sui_tests.rs` | Heavy |

#### Consensus Crate (`consensus/src/`)

| Soma File | Sui Equivalent | Derivation Level |
|-----------|---------------|-----------------|
| All files in `consensus/src/` | `consensus/core/src/` | Heavy (direct fork) |

#### Other Crates

| Soma Crate | Sui Equivalent | Derivation Level |
|-----------|---------------|-----------------|
| `store/src/` | `sui-storage/` (typed-store) | Heavy |
| `store-derive/` | `typed-store-derive/` | Direct copy |
| `sync/` | `sui-network/` | Heavy |
| `node/src/` | `sui-node/src/` | Heavy |
| `test-cluster/src/` | `test-cluster/src/` | Heavy |
| `soma-tls/` | `sui-tls/` | Heavy |
| `soma-keys/` | `sui-keys/` | Heavy |
| `soma-http/` | `sui-http/` | Heavy (if used) |
| `utils/src/` | `sui-macros/` + utilities | Moderate |
| `data-ingestion/` | `sui-data-ingestion/` | Heavy |
| `protocol-config/` | `sui-protocol-config/` | Heavy |
| `protocol-config-macros/` | `sui-protocol-config-macros/` | Direct copy |
| `e2e-tests/` | `sui-e2e-tests/` | Moderate (test structure adopted) |

**Soma-original files (no Sui attribution needed):**
- `types/src/model.rs`, `types/src/model_selection.rs`
- `types/src/target.rs`, `types/src/submission.rs`, `types/src/challenge.rs`
- `types/src/tensor.rs`
- `types/src/system_state/model_registry.rs`, `types/src/system_state/emission.rs`
- `types/src/system_state/target_state.rs`
- `authority/src/execution/submission.rs`, `authority/src/execution/challenge.rs`
- `authority/src/execution/model.rs`
- `authority/src/execution/change_epoch.rs` (heavily customized)
- `authority/src/audit_service.rs`, `authority/src/proxy_server.rs`
- `blobs/` (entire crate)
- `models/` (entire crate)
- `sdk/src/proxy_client.rs`
- `cli/` (mostly custom commands)
- `rpc/` (custom proto definitions and services)

---

## Build & Run Commands

```bash
# Build all crates
PYO3_PYTHON=python3 cargo build --release

# Run specific unit tests
cargo test -p authority -- pay_coin_tests
cargo test -p types -- rewards_distribution
cargo test -p store -- rocks::tests

# Build for msim (required for e2e/consensus integration tests)
PYO3_PYTHON=python3 RUSTFLAGS="--cfg msim" cargo build -p e2e-tests -p test-cluster

# Run specific e2e test
PYO3_PYTHON=python3 RUSTFLAGS="--cfg msim" cargo test -p e2e-tests --test target_tests

# Run all e2e tests
PYO3_PYTHON=python3 RUSTFLAGS="--cfg msim" cargo test -p e2e-tests

# Check compilation under msim without running
PYO3_PYTHON=python3 RUSTFLAGS="--cfg msim" cargo check -p authority
```

---

## Implementation Order (Revised)

Ordered by mainnet criticality. See [Audit Notes](#audit-notes-feb-2026) for full rationale.

### Phase 1: Mainnet Blockers (Weeks 1-3)
1. **Week 1:** Fix 10 failing `pay_coin_tests` + add gas tests (Authority Plan 1A, 1B)
2. **Week 1-2:** Consensus test infrastructure — fork `CommitTestFixture`, uncomment test modules, add randomized DAG tests (Consensus Plan)
3. **Week 2-3:** Staking execution tests + advance epoch unit tests + validator tests (Authority Plan 1C, 1D, 1E)

### Phase 2: Consensus-Critical Foundation (Weeks 3-5)
4. **Week 3-4:** Types serialization tests — BCS roundtrip for all 31 tx kinds, effects, digests, checkpoints (Types Plan)
5. **Week 4:** Protocol config snapshot tests with `insta` crate (Protocol Config Plan)
6. **Week 4-5:** Authority state tests + genesis tests (Authority Plan 2A, 3A)

### Phase 3: Operational Resilience (Weeks 5-7)
7. **Week 5-6:** State sync integration tests (requires `CommitteeFixture` from Phase 1) (Sync Plan)
8. **Week 6-7:** RPC proto roundtrip tests + SDK integration (CLI/SDK/RPC Plan, high-value subset)
9. **Week 7:** Store — InMemoryDB tests, authority store integration (Store Plan)

### Phase 4: Hardening (Week 8+)
10. **Week 8:** Soma-specific edge case tests (targets, submissions, challenges)
11. **Ongoing:** Licensing attribution audit across all files
12. **Ongoing:** Consider load testing, fuzz testing, economic invariant testing (see Gaps section)

---

## Sui Cross-Reference URLs

Key Sui files referenced in this plan (commit `e5b13459e4`):

| Category | Sui File Path |
|----------|--------------|
| Pay tests | `crates/sui-core/src/unit_tests/pay_sui_tests.rs` |
| Gas tests | `crates/sui-core/src/unit_tests/gas_tests.rs` |
| Authority tests | `crates/sui-core/src/unit_tests/authority_tests.rs` |
| Authority state | `crates/sui-core/src/authority.rs` |
| Consensus handler | `crates/sui-core/src/consensus_handler.rs` |
| Consensus test utils | `crates/sui-core/src/unit_tests/consensus_test_utils.rs` |
| Consensus simtests | `consensus/simtests/tests/consensus_tests.rs` |
| Consensus DAG simtests | `consensus/simtests/tests/consensus_dag_tests.rs` |
| Randomized tests | `consensus/core/src/tests/randomized_tests.rs` |
| State sync tests | `crates/sui-network/src/state_sync/tests.rs` |
| State sync resilience | `crates/sui-e2e-tests/tests/state_sync_resilience_tests.rs` |
| Checkpoint tests (e2e) | `crates/sui-e2e-tests/tests/checkpoint_tests.rs` |
| Types unit tests | `crates/sui-types/src/unit_tests/` (15+ files) |
| Protocol config | `crates/sui-protocol-config/src/lib.rs` + 126 snapshots |
| Genesis builder | `crates/sui-genesis-builder/src/lib.rs` |
| CLI tests | `crates/sui/tests/cli_tests.rs` |
| TLS tests | `crates/sui-tls/src/lib.rs` |
| Reconfiguration tests | `crates/sui-e2e-tests/tests/reconfiguration_tests.rs` |
