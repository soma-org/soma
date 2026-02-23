//! Checkpoint integration tests.
//!
//! Tests:
//! 1. basic_checkpoints_integration_test — Transaction included in checkpoint across all validators
//! 2. test_checkpoint_timestamps_non_decreasing — Checkpoint timestamps never decrease
//! 3. test_checkpoint_fork_detection_storage — Fork detection storage API correctness
//!
//! Ported from Sui's `checkpoint_tests.rs`. Skipped:
//! - test_checkpoint_split_brain (requires fail_point infrastructure)
//! - test_checkpoint_contents_v2_alias_versions (Move/alias dependent)

use std::time::Duration;
use test_cluster::TestClusterBuilder;
use tokio::time::sleep;
use tracing::info;
use types::{
    effects::TransactionEffectsAPI,
    transaction::{TransactionData, TransactionKind},
};
use utils::logging::init_tracing;

/// Execute a coin transfer and verify it gets included in a checkpoint on all validators.
#[cfg(msim)]
#[msim::sim_test]
async fn basic_checkpoints_integration_test() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new().build().await;

    // Create and execute a coin transfer
    let addresses = test_cluster.wallet.get_addresses();
    let sender = addresses[0];
    let recipient = addresses[1];

    let gas =
        test_cluster.wallet.get_one_gas_object_owned_by_address(sender).await.unwrap().unwrap();

    let tx_data = TransactionData::new(
        TransactionKind::TransferCoin { coin: gas, amount: Some(1000), recipient },
        sender,
        vec![gas],
    );

    let response = test_cluster.sign_and_execute_transaction(&tx_data).await;
    assert!(response.effects.status().is_ok());
    let digest = response.effects.transaction_digest();

    // Poll until all validators include the transaction in a checkpoint
    for _ in 0..600 {
        let all_included = test_cluster.swarm.validator_node_handles().into_iter().all(|handle| {
            handle.with(|node| {
                node.state()
                    .epoch_store_for_testing()
                    .is_transaction_executed_in_checkpoint(&digest)
                    .unwrap()
            })
        });
        if all_included {
            info!("Transaction included in checkpoint on all validators");
            return;
        }
        sleep(Duration::from_millis(100)).await;
    }

    panic!("Did not include transaction in checkpoint on all validators in 60 seconds");
}

/// Verify that checkpoint timestamps are non-decreasing across all checkpoints,
/// even across epoch boundaries.
#[cfg(msim)]
#[msim::sim_test]
async fn test_checkpoint_timestamps_non_decreasing() {
    init_tracing();

    let epoch_duration_ms = 10_000; // 10 seconds
    let num_epochs_to_run = 3;

    let test_cluster =
        TestClusterBuilder::new().with_epoch_duration_ms(epoch_duration_ms).build().await;

    // Wait for multiple epochs to pass
    sleep(Duration::from_millis(epoch_duration_ms * num_epochs_to_run + epoch_duration_ms / 2))
        .await;

    // Retrieve checkpoints from the first fullnode
    let fullnode_handle = &test_cluster.fullnode_handle.soma_node;

    let checkpoint_store = fullnode_handle.with(|node| node.state().get_checkpoint_store().clone());

    let highest_executed_checkpoint = checkpoint_store
        .get_highest_executed_checkpoint()
        .expect("Failed to get highest executed checkpoint")
        .expect("No executed checkpoints found in store");

    assert!(
        highest_executed_checkpoint.epoch() > 0,
        "Test did not run long enough to cross epochs"
    );

    let mut current_seq = *highest_executed_checkpoint.sequence_number();
    let mut prev_timestamp = highest_executed_checkpoint.timestamp();
    let mut checkpoints_checked = 0;

    // Iterate backwards from the highest checkpoint
    loop {
        if current_seq == 0 {
            info!("Reached checkpoint 0.");
            break;
        }
        current_seq -= 1;

        let current_checkpoint = checkpoint_store
            .get_checkpoint_by_sequence_number(current_seq)
            .expect("DB error getting current checkpoint")
            .unwrap_or_else(|| panic!("checkpoint missing for seq {}", current_seq));

        let current_timestamp = current_checkpoint.timestamp();
        assert!(
            current_timestamp <= prev_timestamp,
            "Timestamp decreased! current seq {}, {:?} vs {:?}",
            current_seq,
            current_timestamp,
            prev_timestamp,
        );
        prev_timestamp = current_timestamp;
        checkpoints_checked += 1;
    }

    assert!(checkpoints_checked > 0, "Test created only 1 checkpoint");
    info!("Verified {} checkpoints with non-decreasing timestamps", checkpoints_checked);
}

/// Test the checkpoint fork detection storage API: record, retrieve, and clear.
#[cfg(msim)]
#[msim::sim_test]
async fn test_checkpoint_fork_detection_storage() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new().with_num_validators(4).build().await;

    // Get the first validator for testing
    let validator_handle =
        test_cluster.swarm.validator_node_handles().into_iter().next().expect("No validator found");

    // Test: Basic fork detection storage functionality
    validator_handle.with(|node| {
        let checkpoint_store = node.state().get_checkpoint_store().clone();

        // Initially, no fork detected
        assert!(
            checkpoint_store.get_checkpoint_fork_detected().unwrap().is_none(),
            "No fork should be detected initially"
        );

        // Record a fork
        let fork_seq = 42;
        let fork_digest = types::digests::CheckpointDigest::random();

        checkpoint_store
            .record_checkpoint_fork_detected(fork_seq, fork_digest)
            .expect("Failed to record checkpoint fork");

        // Retrieve the recorded fork
        let retrieved = checkpoint_store.get_checkpoint_fork_detected().unwrap();
        assert!(retrieved.is_some(), "Fork should have been recorded");
        let (retrieved_seq, retrieved_digest) = retrieved.unwrap();
        assert_eq!(retrieved_seq, fork_seq);
        assert_eq!(retrieved_digest, fork_digest);

        // Clear the fork
        checkpoint_store.clear_checkpoint_fork_detected().unwrap();
        let retrieved_after_clear = checkpoint_store.get_checkpoint_fork_detected().unwrap();
        assert!(retrieved_after_clear.is_none(), "Fork state should be cleared");

        info!("Fork detection storage API works correctly");
    });
}
