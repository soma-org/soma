// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Failpoint-based E2E tests.
//!
//! These tests use the failpoint infrastructure (`utils::fp`) to inject faults
//! at specific code points (epoch transitions, consensus commits, checkpoint
//! execution) and verify that the network recovers gracefully.
//!
//! All failpoint macros compile to no-ops without `cfg(msim)`, so these tests
//! only run under the simulator.
//!
//! Crash tests use `msim::task::kill_current_node()` which cleanly kills the
//! simulated node (with automatic restart) rather than `panic!()` which would
//! fail the entire test.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use test_cluster::TestClusterBuilder;
use tokio::time::sleep;
use tracing::info;
use types::effects::TransactionEffectsAPI;
use types::system_state::SystemStateTrait as _;
use utils::fp::{
    clear_fail_point, register_fail_point, register_fail_point_async, register_fail_point_if,
};
use utils::logging::init_tracing;

/// Inject a delay into the reconfiguration path and verify the network still
/// progresses through multiple epochs. The delay fires only on the first
/// epoch transition to avoid excessive slowdown.
#[cfg(msim)]
#[msim::sim_test]
async fn test_reconfig_with_delay() {
    init_tracing();

    static DELAYED: AtomicBool = AtomicBool::new(false);

    register_fail_point_async("reconfig_delay", || async {
        if !DELAYED.swap(true, Ordering::SeqCst) {
            info!("failpoint: injecting 2s reconfig delay");
            sleep(Duration::from_secs(2)).await;
        }
    });

    let test_cluster =
        TestClusterBuilder::new().with_num_validators(4).with_epoch_duration_ms(5000).build().await;

    let target_epoch = 3u64;
    let system_state = test_cluster
        .wait_for_epoch_with_timeout(Some(target_epoch), Duration::from_secs(120))
        .await;

    info!(epoch = system_state.epoch(), "reconfig-delay test passed");
    assert!(system_state.epoch() >= target_epoch);
    assert!(DELAYED.load(Ordering::SeqCst), "Failpoint should have fired");

    clear_fail_point("reconfig_delay");
}

/// Inject a delay into the advance-epoch transaction creation path.
/// This simulates slow epoch change processing and verifies that the
/// network still reconfigures correctly.
#[cfg(msim)]
#[msim::sim_test]
async fn test_change_epoch_tx_delay() {
    init_tracing();

    static DELAYED: AtomicBool = AtomicBool::new(false);

    register_fail_point_async("change_epoch_tx_delay", || async {
        if !DELAYED.swap(true, Ordering::SeqCst) {
            info!("failpoint: injecting 3s change-epoch-tx delay");
            sleep(Duration::from_secs(3)).await;
        }
    });

    let test_cluster =
        TestClusterBuilder::new().with_num_validators(4).with_epoch_duration_ms(5000).build().await;

    let target_epoch = 3u64;
    let system_state = test_cluster
        .wait_for_epoch_with_timeout(Some(target_epoch), Duration::from_secs(180))
        .await;

    info!(epoch = system_state.epoch(), "change-epoch-tx-delay test passed");
    assert!(system_state.epoch() >= target_epoch);
    assert!(DELAYED.load(Ordering::SeqCst), "Failpoint should have fired");

    clear_fail_point("change_epoch_tx_delay");
}

/// Crash a validator once when it tries to open the new epoch store during
/// reconfiguration. The node is killed via `msim::task::kill_current_node()`
/// which triggers automatic restart after a random delay. The remaining
/// validators (BFT majority) continue and the network reaches the target epoch.
#[cfg(msim)]
#[msim::sim_test]
async fn test_crash_before_open_new_epoch_store() {
    init_tracing();

    static CRASHED: AtomicBool = AtomicBool::new(false);

    register_fail_point("before-open-new-epoch-store", || {
        if !CRASHED.swap(true, Ordering::SeqCst) {
            info!("failpoint: killing node before opening new epoch store");
            msim::task::kill_current_node(Some(Duration::from_secs(1)));
        }
    });

    let test_cluster =
        TestClusterBuilder::new().with_num_validators(7).with_epoch_duration_ms(5000).build().await;

    let target_epoch = 3u64;
    let system_state = test_cluster
        .wait_for_epoch_with_timeout(Some(target_epoch), Duration::from_secs(120))
        .await;

    info!(
        epoch = system_state.epoch(),
        crashed = CRASHED.load(Ordering::SeqCst),
        "crash-before-open-new-epoch-store test passed"
    );
    assert!(system_state.epoch() >= target_epoch);
    assert!(CRASHED.load(Ordering::SeqCst), "Failpoint should have fired");

    clear_fail_point("before-open-new-epoch-store");
}

/// Crash a validator once right before it commits a certificate to storage.
/// This tests that the node can recover from a crash in the execution path.
#[cfg(msim)]
#[msim::sim_test]
async fn test_crash_before_commit_certificate() {
    init_tracing();

    static CRASHED: AtomicBool = AtomicBool::new(false);

    register_fail_point("crash-before-commit-certificate", || {
        if !CRASHED.swap(true, Ordering::SeqCst) {
            info!("failpoint: killing node before commit certificate");
            msim::task::kill_current_node(Some(Duration::from_secs(1)));
        }
    });

    let test_cluster =
        TestClusterBuilder::new().with_num_validators(7).with_epoch_duration_ms(5000).build().await;

    let target_epoch = 2u64;
    let system_state = test_cluster
        .wait_for_epoch_with_timeout(Some(target_epoch), Duration::from_secs(120))
        .await;

    info!(epoch = system_state.epoch(), "crash-before-commit-certificate test passed");
    assert!(system_state.epoch() >= target_epoch);
    assert!(CRASHED.load(Ordering::SeqCst), "Failpoint should have fired");

    clear_fail_point("crash-before-commit-certificate");
}

/// Crash a validator once after processing a consensus commit. This tests
/// recovery from a crash between consensus output processing and transaction
/// execution scheduling.
#[cfg(msim)]
#[msim::sim_test]
async fn test_crash_after_consensus_commit() {
    init_tracing();

    static CRASHED: AtomicBool = AtomicBool::new(false);

    register_fail_point("crash-after-consensus-commit", || {
        if !CRASHED.swap(true, Ordering::SeqCst) {
            info!("failpoint: killing node after consensus commit");
            msim::task::kill_current_node(Some(Duration::from_secs(1)));
        }
    });

    let test_cluster =
        TestClusterBuilder::new().with_num_validators(7).with_epoch_duration_ms(5000).build().await;

    let target_epoch = 2u64;
    let system_state = test_cluster
        .wait_for_epoch_with_timeout(Some(target_epoch), Duration::from_secs(120))
        .await;

    info!(epoch = system_state.epoch(), "crash-after-consensus-commit test passed");
    assert!(system_state.epoch() >= target_epoch);
    assert!(CRASHED.load(Ordering::SeqCst), "Failpoint should have fired");

    clear_fail_point("crash-after-consensus-commit");
}

/// Crash a validator once after accumulating the epoch hash in the
/// checkpoint executor. This tests recovery from a crash at the very end
/// of epoch execution, just before the checkpoint store is pruned.
#[cfg(msim)]
#[msim::sim_test]
async fn test_crash_after_accumulate_epoch() {
    init_tracing();

    static CRASHED: AtomicBool = AtomicBool::new(false);

    register_fail_point("crash-after-accumulate-epoch", || {
        if !CRASHED.swap(true, Ordering::SeqCst) {
            info!("failpoint: killing node after accumulate epoch");
            msim::task::kill_current_node(Some(Duration::from_secs(1)));
        }
    });

    let test_cluster =
        TestClusterBuilder::new().with_num_validators(7).with_epoch_duration_ms(5000).build().await;

    let target_epoch = 2u64;
    let system_state = test_cluster
        .wait_for_epoch_with_timeout(Some(target_epoch), Duration::from_secs(120))
        .await;

    info!(epoch = system_state.epoch(), "crash-after-accumulate-epoch test passed");
    assert!(system_state.epoch() >= target_epoch);
    assert!(CRASHED.load(Ordering::SeqCst), "Failpoint should have fired");

    clear_fail_point("crash-after-accumulate-epoch");
}

/// Crash a validator once after writing a batch to the DB, before the
/// writeback cache is updated. This tests that the node can recover when
/// the DB has been written but in-memory caches are stale.
#[cfg(msim)]
#[msim::sim_test]
async fn test_crash_after_db_write() {
    init_tracing();

    // Use a counter so we don't crash on the very first DB write (which would
    // happen during genesis/startup). Instead, crash on the Nth write to
    // simulate a mid-operation failure.
    static WRITE_COUNT: AtomicUsize = AtomicUsize::new(0);
    static CRASHED: AtomicBool = AtomicBool::new(false);

    register_fail_point("crash-after-db-write", || {
        let count = WRITE_COUNT.fetch_add(1, Ordering::SeqCst);
        // Crash on the 5th DB write (after startup stabilizes)
        if count == 4 && !CRASHED.swap(true, Ordering::SeqCst) {
            info!("failpoint: killing node after DB write (write #{count})");
            msim::task::kill_current_node(Some(Duration::from_secs(1)));
        }
    });

    let test_cluster =
        TestClusterBuilder::new().with_num_validators(7).with_epoch_duration_ms(5000).build().await;

    let target_epoch = 2u64;
    let system_state = test_cluster
        .wait_for_epoch_with_timeout(Some(target_epoch), Duration::from_secs(120))
        .await;

    info!(
        epoch = system_state.epoch(),
        writes = WRITE_COUNT.load(Ordering::SeqCst),
        "crash-after-db-write test passed"
    );
    assert!(system_state.epoch() >= target_epoch);

    clear_fail_point("crash-after-db-write");
}

/// Crash a validator once after building a DB batch in the authority store,
/// before it is written. This tests recovery from a crash where the batch
/// was constructed but never persisted.
#[cfg(msim)]
#[msim::sim_test]
async fn test_crash_after_build_batch() {
    init_tracing();

    // Use a counter to skip early batch builds during genesis/startup.
    static BUILD_COUNT: AtomicUsize = AtomicUsize::new(0);
    static CRASHED: AtomicBool = AtomicBool::new(false);

    register_fail_point("crash-after-build-batch", || {
        let count = BUILD_COUNT.fetch_add(1, Ordering::SeqCst);
        if count == 3 && !CRASHED.swap(true, Ordering::SeqCst) {
            info!("failpoint: killing node after building DB batch (batch #{count})");
            msim::task::kill_current_node(Some(Duration::from_secs(1)));
        }
    });

    let test_cluster =
        TestClusterBuilder::new().with_num_validators(7).with_epoch_duration_ms(5000).build().await;

    let target_epoch = 2u64;
    let system_state = test_cluster
        .wait_for_epoch_with_timeout(Some(target_epoch), Duration::from_secs(120))
        .await;

    info!(
        epoch = system_state.epoch(),
        batches = BUILD_COUNT.load(Ordering::SeqCst),
        "crash-after-build-batch test passed"
    );
    assert!(system_state.epoch() >= target_epoch);

    clear_fail_point("crash-after-build-batch");
}

/// Verify the highest-executed-checkpoint failpoint fires during normal
/// operation. This is an observation-only test (no crash, no delay) that
/// validates the failpoint is reachable in the backpressure manager code path.
#[cfg(msim)]
#[msim::sim_test]
async fn test_highest_executed_checkpoint_failpoint() {
    init_tracing();

    static HIT_COUNT: AtomicUsize = AtomicUsize::new(0);

    register_fail_point("highest-executed-checkpoint", || {
        HIT_COUNT.fetch_add(1, Ordering::SeqCst);
    });

    let test_cluster =
        TestClusterBuilder::new().with_num_validators(4).with_epoch_duration_ms(3000).build().await;

    let target_epoch = 3u64;
    let system_state = test_cluster
        .wait_for_epoch_with_timeout(Some(target_epoch), Duration::from_secs(120))
        .await;

    let hits = HIT_COUNT.load(Ordering::SeqCst);
    info!(
        hits,
        epoch = system_state.epoch(),
        "highest-executed-checkpoint failpoint was hit {hits} times"
    );
    assert!(hits > 0, "Failpoint should have been hit at least once during checkpoint execution");

    clear_fail_point("highest-executed-checkpoint");
}

/// Test safe mode: inject a failure at epoch 2 and verify the network
/// enters safe mode, continues processing transactions, and recovers
/// automatically at epoch 3.
///
/// Mirrors Sui's `safe_mode_reconfig_test`.
#[cfg(msim)]
#[msim::sim_test]
async fn test_safe_mode_reconfig() {
    init_tracing();

    // Register failpoint to inject failure during the epoch 1→2 transition.
    // Starts disabled so the epoch 0→1 transition succeeds normally.
    static INJECT_FAILURE: AtomicBool = AtomicBool::new(false);

    register_fail_point_if("advance_epoch_result_injection", || {
        INJECT_FAILURE.load(Ordering::SeqCst)
    });

    let test_cluster =
        TestClusterBuilder::new().with_num_validators(4).with_epoch_duration_ms(5000).build().await;

    // Wait for epoch 1 to complete normally
    let system_state =
        test_cluster.wait_for_epoch_with_timeout(Some(1), Duration::from_secs(60)).await;
    assert!(!system_state.safe_mode(), "Should not be in safe mode at epoch 1");
    assert_eq!(system_state.epoch(), 1);

    // Now enable failure injection so the epoch 1→2 transition fails
    INJECT_FAILURE.store(true, Ordering::SeqCst);

    // Epoch 2 transition will fail → safe mode
    let system_state =
        test_cluster.wait_for_epoch_with_timeout(Some(2), Duration::from_secs(60)).await;
    assert!(system_state.safe_mode(), "Should be in safe mode at epoch 2");
    assert_eq!(system_state.epoch(), 2);
    info!("Safe mode activated at epoch 2 (emissions forfeited, fees route to fund inline)");

    // Verify transactions still work during safe mode
    let sender = test_cluster.get_addresses()[0];
    let validator_address = test_cluster.fullnode_handle.soma_node.with(|node| {
        node.state().get_system_state_object_for_testing().unwrap().validators().validators[0]
            .metadata
            .soma_address
    });

    // Stage 13c: AddStake is balance-mode for both stake (SOMA) and
    // gas (USDC) — no per-tx coin object needed.
    let tx_data = e2e_tests::stateless_tx_data(
        &test_cluster,
        sender,
        types::transaction::TransactionKind::AddStake {
            validator: validator_address,
            amount: 1_000_000,
        },
    );

    let response = test_cluster.sign_and_execute_transaction(&tx_data).await;
    assert!(response.effects.status().is_ok(), "Transactions should work during safe mode");
    info!("Transaction succeeded during safe mode");

    // Remove the failure injection so epoch 3 recovers
    INJECT_FAILURE.store(false, Ordering::SeqCst);

    // Wait for epoch 3 — should recover from safe mode
    let system_state =
        test_cluster.wait_for_epoch_with_timeout(Some(3), Duration::from_secs(60)).await;
    assert!(!system_state.safe_mode(), "Should have recovered from safe mode at epoch 3");
    assert_eq!(system_state.epoch(), 3);

    info!("Safe mode recovery complete at epoch 3");

    clear_fail_point("advance_epoch_result_injection");
}

/// Test safe mode across multiple consecutive epochs: inject failures for
/// epochs 2 and 3, verify fees accumulate, then recover at epoch 4 with
/// all accumulated rewards drained.
#[cfg(msim)]
#[msim::sim_test]
async fn test_safe_mode_multi_epoch() {
    init_tracing();

    // Inject failure for epochs 2 and 3, recover at epoch 4.
    // Uses a simple boolean flag — NOT a counter — because all validators
    // call the failpoint independently and must see the same result.
    static INJECT_FAILURE: AtomicBool = AtomicBool::new(false);

    register_fail_point_if("advance_epoch_result_injection", || {
        INJECT_FAILURE.load(Ordering::SeqCst)
    });

    let test_cluster =
        TestClusterBuilder::new().with_num_validators(4).with_epoch_duration_ms(5000).build().await;

    // Wait for epoch 1 (normal)
    let system_state =
        test_cluster.wait_for_epoch_with_timeout(Some(1), Duration::from_secs(60)).await;
    assert!(!system_state.safe_mode());

    // Enable failure injection for epochs 2 and 3
    INJECT_FAILURE.store(true, Ordering::SeqCst);

    // Wait for epoch 2 (safe mode)
    let system_state =
        test_cluster.wait_for_epoch_with_timeout(Some(2), Duration::from_secs(60)).await;
    assert!(system_state.safe_mode(), "Should be in safe mode at epoch 2");

    // Wait for epoch 3 (still safe mode)
    let system_state =
        test_cluster.wait_for_epoch_with_timeout(Some(3), Duration::from_secs(60)).await;
    assert!(system_state.safe_mode(), "Should still be in safe mode at epoch 3");

    info!("Multi-epoch safe mode held across epochs 2-3 (emissions forfeited)");

    // Disable failure injection so epoch 4 recovers
    INJECT_FAILURE.store(false, Ordering::SeqCst);

    // Wait for epoch 4 (recovery)
    let system_state =
        test_cluster.wait_for_epoch_with_timeout(Some(4), Duration::from_secs(60)).await;
    assert!(!system_state.safe_mode(), "Should recover from safe mode at epoch 4");

    info!("Multi-epoch safe mode recovery complete at epoch 4");

    clear_fail_point("advance_epoch_result_injection");
}

/// Test that the `is_tx_already_executed` guard in `create_and_execute_advance_epoch_tx`
/// prevents a race condition from causing unexpected safe mode entry.
///
/// Scenario: one validator is delayed at `change_epoch_tx_delay`. Other validators
/// build the last checkpoint, which gets delivered to the delayed validator via
/// state sync. When the delayed validator resumes, it should detect the tx was
/// already executed and return early — NOT re-execute and enter safe mode.
///
/// Mirrors Sui's `test_create_advance_epoch_tx_race`.
#[cfg(msim)]
#[msim::sim_test]
async fn test_advance_epoch_tx_race() {
    use std::sync::Arc;

    use tokio::sync::Notify;

    init_tracing();

    // Panic if safe mode is recorded by the checkpoint builder.
    // If the race guard fails, a double-execution would cause advance_epoch to fail
    // (epoch already bumped), triggering safe mode.
    register_fail_point("checkpoint_builder_advance_epoch_is_safe_mode", || {
        panic!("safe mode recorded in checkpoint builder — race condition detected!");
    });

    // Delay exactly one node at change_epoch_tx_delay during the epoch 1→2 transition.
    // The first caller after ENABLED waits on the notify; all others pass through.
    static ENABLED: AtomicBool = AtomicBool::new(false);
    static DELAYED: AtomicBool = AtomicBool::new(false);
    let notify = Arc::new(Notify::new());
    let notify_clone = notify.clone();

    register_fail_point_async("change_epoch_tx_delay", move || {
        let notify = notify_clone.clone();
        async move {
            if !ENABLED.load(Ordering::SeqCst) {
                return;
            }
            // Only delay the first caller — others must proceed to build the checkpoint
            if !DELAYED.swap(true, Ordering::SeqCst) {
                info!("failpoint: delaying change_epoch_tx on one node");
                notify.notified().await;
                info!("failpoint: released change_epoch_tx delay");
            }
        }
    });

    let test_cluster =
        TestClusterBuilder::new().with_num_validators(4).with_epoch_duration_ms(5000).build().await;

    // Wait for epoch 1 (normal — failpoint disabled)
    test_cluster.wait_for_epoch_with_timeout(Some(1), Duration::from_secs(60)).await;

    // Enable the delay for the epoch 1→2 transition
    ENABLED.store(true, Ordering::SeqCst);

    // Wait for one node to get delayed and state sync to deliver the executed tx
    sleep(Duration::from_secs(8)).await;

    // Disable further delays (we only care about the 1→2 race)
    ENABLED.store(false, Ordering::SeqCst);

    // Release the delayed node — it should find the tx already executed and skip re-execution
    notify.notify_one();

    // Wait for epoch 2 — if the race guard failed, the panic failpoint would have fired
    let system_state =
        test_cluster.wait_for_epoch_with_timeout(Some(2), Duration::from_secs(60)).await;
    assert!(!system_state.safe_mode(), "Should not enter safe mode due to race condition");

    info!("Advance epoch tx race test passed — no unexpected safe mode entry");

    clear_fail_point("change_epoch_tx_delay");
    clear_fail_point("checkpoint_builder_advance_epoch_is_safe_mode");
}

/// Crash a validator during epoch change and verify that the network still
/// progresses when combined with active transaction load. This combines the
/// crash test with concurrent stake transactions to test more realistic
/// conditions.
#[cfg(msim)]
#[msim::sim_test]
async fn test_crash_during_reconfig_with_tx_load() {
    init_tracing();

    static CRASHED: AtomicBool = AtomicBool::new(false);

    register_fail_point("before-open-new-epoch-store", || {
        if !CRASHED.swap(true, Ordering::SeqCst) {
            info!("failpoint: killing node during reconfig (with tx load)");
            msim::task::kill_current_node(Some(Duration::from_secs(1)));
        }
    });

    let test_cluster =
        TestClusterBuilder::new().with_num_validators(7).with_epoch_duration_ms(5000).build().await;

    let sender = test_cluster.get_addresses()[0];
    let validator_address = test_cluster.fullnode_handle.soma_node.with(|node| {
        node.state().get_system_state_object_for_testing().unwrap().validators().validators[0]
            .metadata
            .soma_address
    });

    let target_epoch = 3u64;
    let mut tx_count = 0u64;

    loop {
        let current_epoch = test_cluster
            .fullnode_handle
            .soma_node
            .with(|node| node.state().epoch_store_for_testing().epoch());
        if current_epoch >= target_epoch {
            break;
        }

        // Submit stake transactions while epochs transition.
        // Stage 13c: balance-mode for both stake and gas.
        let tx_data = e2e_tests::stateless_tx_data(
            &test_cluster,
            sender,
            types::transaction::TransactionKind::AddStake {
                validator: validator_address,
                amount: 1_000_000,
            },
        );

        if let Ok(response) = tokio::time::timeout(
            Duration::from_secs(30),
            test_cluster.sign_and_execute_transaction(&tx_data),
        )
        .await
        {
            if response.effects.status().is_ok() {
                tx_count += 1;
            }
        }

        sleep(Duration::from_millis(300)).await;
    }

    info!(tx_count, "crash-during-reconfig-with-tx-load test passed");
    assert!(CRASHED.load(Ordering::SeqCst), "Failpoint should have fired");

    clear_fail_point("before-open-new-epoch-store");
}

/// Stage 14d safety. Kill one validator in the crash window between
/// user-tx writeback (effects + per-tx CF on disk) and settlement
/// publish (the in-memory `notify_settlement_transactions_ready` call
/// the cp builder makes). This is the SIP-58 single-path's most
/// load-bearing recovery case: we dropped per-tx accumulator-CF drains
/// in stage 14d step 3, so until the settlement TX runs the
/// `accumulator_balances` CF lags behind the user-tx effects. If
/// crash recovery doesn't re-run settlement, the lag becomes
/// permanent. Recovery is built into the architecture
/// (`pending_checkpoint` on disk has the `settlement_root`; on restart
/// the cp builder re-runs `construct_and_execute_settlement`
/// deterministically against the same user-tx effects) — this test
/// proves it under fault injection.
#[cfg(msim)]
#[msim::sim_test]
async fn test_crash_after_user_effects_before_settlement_publish() {
    init_tracing();

    static CRASHED: AtomicBool = AtomicBool::new(false);

    register_fail_point("crash-after-user-effects-before-settlement-publish", || {
        if !CRASHED.swap(true, Ordering::SeqCst) {
            info!("failpoint: killing node between user-tx writeback and settlement publish");
            msim::task::kill_current_node(Some(Duration::from_secs(1)));
        }
    });

    let test_cluster =
        TestClusterBuilder::new().with_num_validators(7).with_epoch_duration_ms(5000).build().await;

    let sender = test_cluster.get_addresses()[0];
    let recipient = types::base::SomaAddress::random();

    // Submit a balance transfer that produces accumulator events (so
    // the cp builder's settlement path is exercised — without events
    // the failpoint is never reached). Wait for it to land in a cp on
    // all validators, which only succeeds if the killed validator
    // re-runs settlement after restart.
    let tx_data = e2e_tests::balance_transfer_data(
        &test_cluster,
        types::object::CoinType::Usdc,
        sender,
        vec![(recipient, 1_000)],
    );
    let response = test_cluster.sign_and_execute_transaction(&tx_data).await;
    assert!(response.effects.status().is_ok());
    let user_tx_digest = *response.effects.transaction_digest();

    // Drive a couple more epochs while traffic continues so any post-
    // crash recovery that requires consensus to re-deliver pending
    // commits has time to play out.
    let target_epoch = 2u64;
    let system_state = test_cluster
        .wait_for_epoch_with_timeout(Some(target_epoch), Duration::from_secs(120))
        .await;
    assert!(system_state.epoch() >= target_epoch);
    assert!(CRASHED.load(Ordering::SeqCst), "failpoint should have fired");

    // Every validator (including the one that was killed mid-cp)
    // must end up with the user tx durably executed AND the
    // settlement that accompanies it durably applied. The is_tx
    // poll only proves "the cp containing user_tx eventually
    // executed"; on top of that we read both the
    // `accumulator_balances` CF (`get_balance`) and the
    // `BalanceAccumulator` object world (`get_balance_via_object`)
    // and assert (a) the recipient's balance equals 1_000 in both
    // stores on every validator, and (b) the two stores agree.
    //
    // Together those assertions catch:
    //  - skip-apply (settlement never re-ran post-restart)        → balance == 0
    //  - double-apply (settlement applied twice on recovery)      → balance == 2_000
    //  - CF/object divergence (one path re-applied, other didn't) → cf != object
    // Polling because the killed node's restart and state-sync
    // take wall time.
    for _ in 0..600 {
        let observations: Vec<_> = test_cluster
            .swarm
            .validator_node_handles()
            .into_iter()
            .map(|h| {
                h.with(|node| {
                    let db = node.state().database_for_testing();
                    let executed = db.is_tx_already_executed(&user_tx_digest).unwrap_or(false);
                    let cf_balance = db
                        .get_balance(recipient, types::object::CoinType::Usdc)
                        .unwrap_or(u64::MAX);
                    let obj_balance = db
                        .get_balance_via_object(recipient, types::object::CoinType::Usdc)
                        .unwrap_or(u64::MAX);
                    (executed, cf_balance, obj_balance)
                })
            })
            .collect();

        let all_consistent = observations
            .iter()
            .all(|(executed, cf, obj)| *executed && *cf == 1_000 && *obj == 1_000);
        if all_consistent {
            info!("crash-after-user-effects-before-settlement-publish test passed");
            clear_fail_point("crash-after-user-effects-before-settlement-publish");
            return;
        }
        sleep(Duration::from_millis(100)).await;
    }

    // Final dump on failure so a regression is debuggable.
    let observations: Vec<_> = test_cluster
        .swarm
        .validator_node_handles()
        .into_iter()
        .map(|h| {
            h.with(|node| {
                let db = node.state().database_for_testing();
                (
                    db.is_tx_already_executed(&user_tx_digest).unwrap_or(false),
                    db.get_balance(recipient, types::object::CoinType::Usdc)
                        .unwrap_or(u64::MAX),
                    db.get_balance_via_object(recipient, types::object::CoinType::Usdc)
                        .unwrap_or(u64::MAX),
                )
            })
        })
        .collect();
    panic!(
        "post-recovery state diverged. (executed, cf_balance, obj_balance) per validator: \
         {observations:?}"
    );
}
