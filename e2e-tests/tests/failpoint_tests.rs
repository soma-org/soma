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
use types::system_state::SystemStateTrait;
use utils::fp::{clear_fail_point, register_fail_point, register_fail_point_async};
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

    let test_cluster = TestClusterBuilder::new()
        .with_num_validators(4)
        .with_epoch_duration_ms(5000)
        .build()
        .await;

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

    let test_cluster = TestClusterBuilder::new()
        .with_num_validators(4)
        .with_epoch_duration_ms(5000)
        .build()
        .await;

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

    let test_cluster = TestClusterBuilder::new()
        .with_num_validators(7)
        .with_epoch_duration_ms(5000)
        .build()
        .await;

    let target_epoch = 3u64;
    let system_state = test_cluster
        .wait_for_epoch_with_timeout(Some(target_epoch), Duration::from_secs(120))
        .await;

    info!(epoch = system_state.epoch(), crashed = CRASHED.load(Ordering::SeqCst), "crash-before-open-new-epoch-store test passed");
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

    let test_cluster = TestClusterBuilder::new()
        .with_num_validators(7)
        .with_epoch_duration_ms(5000)
        .build()
        .await;

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

    let test_cluster = TestClusterBuilder::new()
        .with_num_validators(7)
        .with_epoch_duration_ms(5000)
        .build()
        .await;

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

    let test_cluster = TestClusterBuilder::new()
        .with_num_validators(7)
        .with_epoch_duration_ms(5000)
        .build()
        .await;

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

    let test_cluster = TestClusterBuilder::new()
        .with_num_validators(7)
        .with_epoch_duration_ms(5000)
        .build()
        .await;

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

    let test_cluster = TestClusterBuilder::new()
        .with_num_validators(7)
        .with_epoch_duration_ms(5000)
        .build()
        .await;

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

    let test_cluster = TestClusterBuilder::new()
        .with_num_validators(4)
        .with_epoch_duration_ms(3000)
        .build()
        .await;

    let target_epoch = 3u64;
    let system_state = test_cluster
        .wait_for_epoch_with_timeout(Some(target_epoch), Duration::from_secs(120))
        .await;

    let hits = HIT_COUNT.load(Ordering::SeqCst);
    info!(hits, epoch = system_state.epoch(), "highest-executed-checkpoint failpoint was hit {hits} times");
    assert!(hits > 0, "Failpoint should have been hit at least once during checkpoint execution");

    clear_fail_point("highest-executed-checkpoint");
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

    let test_cluster = TestClusterBuilder::new()
        .with_num_validators(7)
        .with_epoch_duration_ms(5000)
        .build()
        .await;

    let sender = test_cluster.get_addresses()[0];
    let validator_address = test_cluster.fullnode_handle.soma_node.with(|node| {
        node.state()
            .get_system_state_object_for_testing()
            .unwrap()
            .validators
            .validators[0]
            .metadata
            .soma_address
    });

    let target_epoch = 3u64;
    let mut tx_count = 0u64;

    loop {
        let current_epoch = test_cluster.fullnode_handle.soma_node.with(|node| {
            node.state().epoch_store_for_testing().epoch()
        });
        if current_epoch >= target_epoch {
            break;
        }

        // Submit stake transactions while epochs transition
        if let Some(gas_object) = test_cluster
            .wallet
            .get_one_gas_object_owned_by_address(sender)
            .await
            .unwrap()
        {
            let tx_data = types::transaction::TransactionData::new(
                types::transaction::TransactionKind::AddStake {
                    address: validator_address,
                    coin_ref: gas_object,
                    amount: Some(1_000_000),
                },
                sender,
                vec![gas_object],
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
        }

        sleep(Duration::from_millis(300)).await;
    }

    info!(tx_count, "crash-during-reconfig-with-tx-load test passed");
    assert!(CRASHED.load(Ordering::SeqCst), "Failpoint should have fired");

    clear_fail_point("before-open-new-epoch-store");
}
