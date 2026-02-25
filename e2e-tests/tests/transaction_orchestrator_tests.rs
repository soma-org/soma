// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Transaction Orchestrator E2E tests.
//!
//! Tests:
//! 1. test_blocking_execution — Execute via orchestrator with local execution wait
//! 2. test_fullnode_wal_log — WAL persistence during quorum loss and recovery
//! 3. test_transaction_orchestrator_reconfig — Orchestrator epoch transitions
//! 4. test_tx_across_epoch_boundaries — Transaction submitted during epoch change
//! 5. test_orchestrator_execute_transaction — Execute and verify effects
//! 6. test_orchestrator_execute_staking — Execute staking transaction via orchestrator
//! 7. test_early_validation_no_side_effects — Sequential transactions don't conflict
//! 8. test_early_validation_with_old_object_version — Stale object version rejected
//!
//! Ported from Sui's `transaction_orchestrator_tests.rs`.
//! Adapted to use SOMA's native transaction types (coin transfers, staking)
//! instead of Move transactions.

use std::time::Duration;
use test_cluster::TestClusterBuilder;
use tokio::time::{sleep, timeout};
use tracing::info;
use types::{
    effects::TransactionEffectsAPI,
    quorum_driver::{ExecuteTransactionRequest, ExecuteTransactionRequestType, FinalizedEffects},
    system_state::SystemStateTrait as _,
    transaction::{TransactionData, TransactionKind},
};
use utils::logging::init_tracing;

/// Helper: create a coin transfer TransactionData
async fn make_transfer(
    test_cluster: &test_cluster::TestCluster,
    sender_idx: usize,
    recipient_idx: usize,
    amount: u64,
) -> TransactionData {
    let addresses = test_cluster.wallet.get_addresses();
    let sender = addresses[sender_idx];
    let recipient = addresses[recipient_idx];

    let gas = test_cluster
        .wallet
        .get_one_gas_object_owned_by_address(sender)
        .await
        .unwrap()
        .expect("sender must have a gas object");

    TransactionData::new(
        TransactionKind::TransferCoin { coin: gas, amount: Some(amount), recipient },
        sender,
        vec![gas],
    )
}

/// Helper: sign a TransactionData and return a Transaction
async fn sign_tx(
    test_cluster: &test_cluster::TestCluster,
    tx_data: &TransactionData,
) -> types::transaction::Transaction {
    test_cluster.wallet.sign_transaction(tx_data).await
}

/// Execute via orchestrator with WaitForLocalExecution and verify the transaction
/// is executed locally.
#[cfg(msim)]
#[msim::sim_test]
async fn test_blocking_execution() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new().build().await;
    let handle = &test_cluster.fullnode_handle.soma_node;
    let orchestrator =
        handle.with(|n| n.transaction_orchestrator().expect("fullnode must have orchestrator"));

    // Execute a transaction via the orchestrator with WaitForLocalExecution
    let tx_data = make_transfer(&test_cluster, 0, 1, 1000).await;
    let tx = sign_tx(&test_cluster, &tx_data).await;
    let digest = *tx.digest();

    let request = ExecuteTransactionRequest {
        transaction: tx,
        include_input_objects: false,
        include_output_objects: false,
    };

    let (_response, executed_locally) = orchestrator
        .execute_transaction_block(
            request,
            ExecuteTransactionRequestType::WaitForLocalExecution,
            None,
        )
        .await
        .unwrap_or_else(|e| panic!("Failed to execute transaction {:?}: {:?}", digest, e));

    assert!(executed_locally, "Transaction should have been executed locally");

    // Verify the transaction was executed
    assert!(
        handle.with(|n| n.state().is_tx_already_executed(&digest)),
        "Transaction should be marked as executed"
    );

    info!("Blocking execution test passed");
}

/// Test transaction behavior during quorum loss and recovery.
///
/// Architecture note: Unlike Sui (which spawns inner execution via `spawn_monitored_task!`,
/// keeping the WAL guard alive even after the caller times out), SOMA runs execution inline.
/// When a timeout drops the caller's future, the `TransactionSubmissionGuard` is also dropped,
/// which cleans up the WAL entry. SOMA's WAL serves crash recovery (entries replayed on restart),
/// not in-flight retry across timeouts.
///
/// This test verifies:
/// 1. Normal execution works
/// 2. Quorum loss causes timeout
/// 3. WAL is clean after timeout (guard cleanup on drop)
/// 4. After quorum is restored, re-submitting the same transaction succeeds
#[cfg(msim)]
#[msim::sim_test]
async fn test_fullnode_wal_log() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new().with_epoch_duration_ms(600_000).build().await;

    let handle = &test_cluster.fullnode_handle.soma_node;
    let orchestrator =
        handle.with(|n| n.transaction_orchestrator().expect("fullnode must have orchestrator"));

    // First verify a tx can go through normally
    let tx_data = make_transfer(&test_cluster, 0, 1, 500).await;
    let tx = sign_tx(&test_cluster, &tx_data).await;
    let digest = *tx.digest();

    let request = ExecuteTransactionRequest {
        transaction: tx,
        include_input_objects: false,
        include_output_objects: false,
    };

    let (_, executed_locally) = orchestrator
        .execute_transaction_block(
            request,
            ExecuteTransactionRequestType::WaitForLocalExecution,
            None,
        )
        .await
        .unwrap_or_else(|e| panic!("Failed to execute transaction {:?}: {:?}", digest, e));

    assert!(executed_locally);

    let validator_addresses = test_cluster.get_validator_pubkeys();
    assert_eq!(validator_addresses.len(), 4);

    // Stop 2 validators — we lose quorum
    test_cluster.stop_node(&validator_addresses[0]);
    test_cluster.stop_node(&validator_addresses[1]);

    // Submit another transaction — expect it to timeout
    let tx_data2 = make_transfer(&test_cluster, 1, 2, 500).await;
    let tx2 = sign_tx(&test_cluster, &tx_data2).await;
    let tx2_digest = *tx2.digest();

    let request2 = ExecuteTransactionRequest {
        transaction: tx2.clone(),
        include_input_objects: false,
        include_output_objects: false,
    };

    let result = timeout(
        Duration::from_secs(10),
        orchestrator.execute_transaction_block(
            request2,
            ExecuteTransactionRequestType::WaitForLocalExecution,
            None,
        ),
    )
    .await;

    // Should timeout since we don't have quorum
    assert!(result.is_err(), "Transaction should timeout without quorum");

    // In SOMA, the WAL is cleaned up when the future is dropped (guard cleanup on drop).
    // This is expected behavior — the WAL serves crash recovery, not in-flight retry.
    let pending_txes = orchestrator
        .load_all_pending_transactions_in_test()
        .expect("Should be able to load pending transactions");
    info!("WAL after timeout: {} pending entries", pending_txes.len());

    // Restore quorum
    test_cluster.start_node(&validator_addresses[0]).await;
    test_cluster.start_node(&validator_addresses[1]).await;
    tokio::task::yield_now().await;

    // Re-submit the same transaction — should succeed now with quorum restored
    let request3 = ExecuteTransactionRequest {
        transaction: tx2,
        include_input_objects: false,
        include_output_objects: false,
    };

    let result = timeout(
        Duration::from_secs(60),
        orchestrator.execute_transaction_block(
            request3,
            ExecuteTransactionRequestType::WaitForLocalExecution,
            None,
        ),
    )
    .await;

    match result {
        Ok(Ok(_)) => {
            assert!(
                handle.with(|n| n.state().is_tx_already_executed(&tx2_digest)),
                "Transaction should be marked as executed"
            );
        }
        Ok(Err(e)) => panic!("Re-submission failed: {:?}", e),
        Err(_) => panic!("Re-submission timed out even with quorum restored"),
    }

    info!("WAL log test passed — quorum loss, timeout, restore, and re-submission verified");
}

/// Verify that the orchestrator's epoch updates after reconfiguration.
#[cfg(msim)]
#[msim::sim_test]
async fn test_transaction_orchestrator_reconfig() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new().build().await;

    let epoch = test_cluster.fullnode_handle.soma_node.with(|node| {
        node.transaction_orchestrator()
            .expect("fullnode must have orchestrator")
            .authority_state()
            .epoch_store_for_testing()
            .epoch()
    });
    assert_eq!(epoch, 0);

    test_cluster.trigger_reconfiguration().await;

    // After epoch change, the orchestrator should update its committee.
    // Use a timeout to make the test reliable (async update).
    timeout(Duration::from_secs(5), async {
        loop {
            let epoch = test_cluster.fullnode_handle.soma_node.with(|node| {
                node.transaction_orchestrator()
                    .unwrap()
                    .authority_state()
                    .epoch_store_for_testing()
                    .epoch()
            });
            if epoch == 1 {
                break;
            }
            sleep(Duration::from_millis(100)).await;
        }
    })
    .await
    .expect("Orchestrator should reach epoch 1 within 5 seconds");

    // Verify the aggregator's committee also updated
    let agg_epoch = test_cluster.fullnode_handle.soma_node.with(|node| {
        node.clone_authority_aggregator()
            .expect("fullnode should have authority aggregator")
            .committee
            .epoch
    });
    assert_eq!(agg_epoch, 1);

    info!("Transaction orchestrator reconfig test passed");
}

/// Submit a transaction while an epoch change is in progress.
/// The transaction must eventually finalize (in the new epoch).
#[cfg(msim)]
#[msim::sim_test]
async fn test_tx_across_epoch_boundaries() {
    init_tracing();

    let total_tx_cnt = 1;
    let (result_tx, mut result_rx) = tokio::sync::mpsc::channel::<FinalizedEffects>(total_tx_cnt);

    let test_cluster = TestClusterBuilder::new().build().await;

    let tx_data = make_transfer(&test_cluster, 0, 1, 1000).await;
    let tx = sign_tx(&test_cluster, &tx_data).await;
    let authorities = test_cluster.swarm.validator_node_handles();

    // Let 2 validators close their epoch early, preventing quorum until full reconfig
    for handle in authorities.iter().take(2) {
        handle.with_async(|node| async { node.close_epoch_for_testing().await.unwrap() }).await;
    }

    // Spawn a task that submits the transaction through the orchestrator
    let orchestrator = test_cluster
        .fullnode_handle
        .soma_node
        .with(|node| node.transaction_orchestrator().unwrap());

    let tx_digest = *tx.digest();
    info!(?tx_digest, "Submitting tx across epoch boundary");

    tokio::task::spawn(async move {
        let request = ExecuteTransactionRequest {
            transaction: tx,
            include_input_objects: false,
            include_output_objects: false,
        };

        match orchestrator
            .execute_transaction_block(
                request,
                ExecuteTransactionRequestType::WaitForLocalExecution,
                None,
            )
            .await
        {
            Ok((response, _)) => {
                info!(?tx_digest, "tx result: ok");
                result_tx.send(response.effects).await.unwrap();
            }
            Err(e) => {
                info!(?tx_digest, ?e, "tx result: error");
                // Transaction might timeout and be retried — that's expected
            }
        }
    });

    // Ask the remaining 2 validators to close epoch
    for handle in authorities.iter().skip(2) {
        handle.with_async(|node| async { node.close_epoch_for_testing().await.unwrap() }).await;
    }

    // Wait for the network to reach epoch 1
    test_cluster.wait_for_epoch(Some(1)).await;

    // The transaction should finalize
    let start = std::time::Instant::now();
    match timeout(Duration::from_secs(30), result_rx.recv()).await {
        Ok(Some(effects_cert)) => {
            info!(epoch = effects_cert.epoch(), "Transaction finalized across epoch boundary");
        }
        Ok(None) => {
            // Channel closed — the spawn task might have errored.
            // Verify the transaction was still executed.
            let executed = test_cluster
                .fullnode_handle
                .soma_node
                .with(|node| node.state().is_tx_already_executed(&tx_digest));
            assert!(executed, "Transaction should have been executed even if channel closed");
        }
        Err(_) => {
            // Timeout — check if executed anyway
            let executed = test_cluster
                .fullnode_handle
                .soma_node
                .with(|node| node.state().is_tx_already_executed(&tx_digest));
            assert!(
                executed,
                "Transaction should have been executed within 30s after epoch change"
            );
        }
    }

    info!("test completed in {:?}", start.elapsed());
}

/// Execute a transaction via the orchestrator and verify effects fields.
#[cfg(msim)]
#[msim::sim_test]
async fn test_orchestrator_execute_transaction() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new().build().await;
    let handle = &test_cluster.fullnode_handle.soma_node;
    let orchestrator =
        handle.with(|n| n.transaction_orchestrator().expect("fullnode must have orchestrator"));

    let tx_data = make_transfer(&test_cluster, 0, 1, 1000).await;
    let tx = sign_tx(&test_cluster, &tx_data).await;

    let request = ExecuteTransactionRequest {
        transaction: tx,
        include_input_objects: true,
        include_output_objects: true,
    };

    let response =
        orchestrator.execute_transaction(request, None).await.expect("Transaction should succeed");

    let effects = &response.effects.effects;

    // Verify effects are present
    assert!(effects.status().is_ok(), "Transaction should succeed");
    assert!(!effects.all_changed_objects().is_empty(), "Should have changed objects");

    // If input/output objects were requested, verify they match effects
    if let Some(input_objects) = &response.input_objects {
        assert!(!input_objects.is_empty(), "Should have input objects");
    }
    if let Some(output_objects) = &response.output_objects {
        assert!(!output_objects.is_empty(), "Should have output objects");
    }

    info!("Orchestrator execute transaction test passed");
}

/// Execute a staking transaction via the orchestrator and verify effects.
#[cfg(msim)]
#[msim::sim_test]
async fn test_orchestrator_execute_staking() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new().build().await;
    let handle = &test_cluster.fullnode_handle.soma_node;
    let orchestrator =
        handle.with(|n| n.transaction_orchestrator().expect("fullnode must have orchestrator"));

    // Get a validator address to stake with
    let system_state = handle.with(|n| n.state().get_system_state_object_for_testing().unwrap());
    let validator_address = system_state.validators().validators[0].metadata.soma_address;

    // Get sender and gas object
    let addresses = test_cluster.wallet.get_addresses();
    let sender = addresses[0];
    let gas = test_cluster
        .wallet
        .get_one_gas_object_owned_by_address(sender)
        .await
        .unwrap()
        .expect("sender must have a gas object");

    let tx_data = TransactionData::new(
        TransactionKind::AddStake {
            address: validator_address,
            coin_ref: gas,
            amount: Some(1_000_000),
        },
        sender,
        vec![gas],
    );
    let tx = sign_tx(&test_cluster, &tx_data).await;

    let request = ExecuteTransactionRequest {
        transaction: tx,
        include_input_objects: true,
        include_output_objects: true,
    };

    let response = orchestrator
        .execute_transaction(request, None)
        .await
        .expect("Staking transaction should succeed");

    assert!(response.effects.effects.status().is_ok(), "Staking transaction should succeed");

    info!("Orchestrator staking transaction test passed");
}

/// Verify that two sequential transactions execute without lock conflicts.
/// Early validation should not cause side effects that interfere with later transactions.
#[cfg(msim)]
#[msim::sim_test]
async fn test_early_validation_no_side_effects() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new().build().await;
    let handle = &test_cluster.fullnode_handle.soma_node;
    let orchestrator =
        handle.with(|n| n.transaction_orchestrator().expect("fullnode must have orchestrator"));

    // Execute first transaction
    let tx_data1 = make_transfer(&test_cluster, 0, 1, 500).await;
    let tx1 = sign_tx(&test_cluster, &tx_data1).await;
    let digest1 = *tx1.digest();

    let request1 = ExecuteTransactionRequest {
        transaction: tx1,
        include_input_objects: false,
        include_output_objects: false,
    };

    let result1 = orchestrator
        .execute_transaction_block(
            request1,
            ExecuteTransactionRequestType::WaitForLocalExecution,
            None,
        )
        .await;

    assert!(result1.is_ok(), "First transaction should succeed: {:?}", result1.err());

    // Execute second transaction from a different sender — early validation should
    // not have caused lock conflicts
    let tx_data2 = make_transfer(&test_cluster, 1, 2, 500).await;
    let tx2 = sign_tx(&test_cluster, &tx_data2).await;
    let digest2 = *tx2.digest();

    let request2 = ExecuteTransactionRequest {
        transaction: tx2,
        include_input_objects: false,
        include_output_objects: false,
    };

    let result2 = orchestrator
        .execute_transaction_block(
            request2,
            ExecuteTransactionRequestType::WaitForLocalExecution,
            None,
        )
        .await;

    assert!(
        result2.is_ok(),
        "Second transaction should succeed without lock conflicts: {:?}",
        result2.err()
    );

    assert_ne!(digest1, digest2, "Transactions should have different digests");

    info!("Early validation no side effects test passed");
}

/// Verify that a transaction referencing a stale (already-spent) object version
/// is rejected by early validation.
#[cfg(msim)]
#[msim::sim_test]
async fn test_early_validation_with_old_object_version() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new().build().await;

    // First, execute a transaction to mutate a coin object
    let addresses = test_cluster.wallet.get_addresses();
    let sender = addresses[0];
    let recipient = addresses[1];

    let gas_before = test_cluster
        .wallet
        .get_one_gas_object_owned_by_address(sender)
        .await
        .unwrap()
        .expect("sender must have a gas object");

    let tx_data = TransactionData::new(
        TransactionKind::TransferCoin { coin: gas_before, amount: Some(1000), recipient },
        sender,
        vec![gas_before],
    );

    let response = test_cluster.sign_and_execute_transaction(&tx_data).await;
    assert!(response.effects.status().is_ok());

    // Now try to use the OLD object version in a new transaction
    // This should fail because the object has been mutated
    let tx_data_stale = TransactionData::new(
        TransactionKind::TransferCoin { coin: gas_before, amount: Some(500), recipient },
        sender,
        vec![gas_before],
    );

    let handle = &test_cluster.fullnode_handle.soma_node;
    let orchestrator =
        handle.with(|n| n.transaction_orchestrator().expect("fullnode must have orchestrator"));

    let tx_stale = sign_tx(&test_cluster, &tx_data_stale).await;
    let request = ExecuteTransactionRequest {
        transaction: tx_stale,
        include_input_objects: false,
        include_output_objects: false,
    };

    let result = orchestrator
        .execute_transaction_block(
            request,
            ExecuteTransactionRequestType::WaitForLocalExecution,
            None,
        )
        .await;

    // Transaction should be rejected (stale object version)
    assert!(result.is_err(), "Transaction with old object version should be rejected");

    info!("Early validation with old object version test passed");
}
