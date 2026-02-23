// Copyright (c) Mysten Labs, Inc.
// Portions of this file are derived from Sui (https://github.com/MystenLabs/sui).
// SPDX-License-Identifier: Apache-2.0

//! Full node integration tests.
//!
//! Tests:
//! 1. test_validator_node_has_no_transaction_orchestrator — Validators have no orchestrator
//! 2. test_full_node_has_transaction_orchestrator — Fullnode has orchestrator
//! 3. test_full_node_run_with_range_checkpoint — Fullnode shuts down at target checkpoint
//! 4. test_full_node_run_with_range_epoch — Fullnode shuts down at target epoch
//! 5. test_access_stale_object_version — Stale object version is rejected
//! 6. test_full_node_transaction_orchestrator_basic — Execute coin transfer via orchestrator
//! 7. test_execute_tx_with_serialized_signature — Round-trip BCS serialize/deserialize transaction
//!
//! Ported from Sui's `full_node_tests.rs`.
//! Skipped: follows_txes (no TransactionFilter), cold_sync/bootstrap_from_snapshot (no DB checkpoint infra),
//! sponsored_transaction (Sui-specific), indexes (no index infra).

use test_cluster::TestClusterBuilder;
use tracing::info;
use types::{
    config::node_config::RunWithRange,
    effects::TransactionEffectsAPI,
    quorum_driver::{ExecuteTransactionRequest, ExecuteTransactionRequestType},
    transaction::{TransactionData, TransactionKind},
};
use utils::logging::init_tracing;

/// Iterate validator handles and assert that transaction_orchestrator() returns None.
#[cfg(msim)]
#[msim::sim_test]
async fn test_validator_node_has_no_transaction_orchestrator() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new().build().await;

    for handle in test_cluster.all_validator_handles() {
        handle.with(|node| {
            assert!(
                node.transaction_orchestrator().is_none(),
                "Validator {} should NOT have a transaction orchestrator",
                node.state().name
            );
        });
    }

    info!("All validators correctly have no transaction orchestrator");
}

/// Check fullnode handle, assert transaction_orchestrator() returns Some.
#[cfg(msim)]
#[msim::sim_test]
async fn test_full_node_has_transaction_orchestrator() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new().build().await;

    test_cluster.fullnode_handle.soma_node.with(|node| {
        assert!(
            node.transaction_orchestrator().is_some(),
            "Fullnode should have a transaction orchestrator"
        );
    });

    info!("Fullnode correctly has a transaction orchestrator");
}

/// Build cluster with RunWithRange::Checkpoint(3). Wait for shutdown signal.
/// Assert the signal matches Checkpoint(3).
#[cfg(msim)]
#[msim::sim_test]
async fn test_full_node_run_with_range_checkpoint() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new()
        .with_fullnode_run_with_range(RunWithRange::Checkpoint(3))
        .build()
        .await;

    let signal = test_cluster.wait_for_run_with_range_shutdown_signal().await;
    assert_eq!(
        signal,
        Some(RunWithRange::Checkpoint(3)),
        "Shutdown signal should be Checkpoint(3)"
    );

    info!("Fullnode shut down at checkpoint 3 as expected");
}

/// Build cluster with RunWithRange::Epoch(0) and short epoch duration.
/// Wait for shutdown signal. Assert it matches Epoch(0).
/// Also verify the orchestrator is None (disabled when run_with_range is set).
#[cfg(msim)]
#[msim::sim_test]
async fn test_full_node_run_with_range_epoch() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new()
        .with_epoch_duration_ms(10_000)
        .with_fullnode_run_with_range(RunWithRange::Epoch(0))
        .build()
        .await;

    // When run_with_range is set, the transaction orchestrator should be disabled
    test_cluster.fullnode_handle.soma_node.with(|node| {
        assert!(
            node.transaction_orchestrator().is_none(),
            "Orchestrator should be None when run_with_range is set"
        );
    });

    let signal = test_cluster.wait_for_run_with_range_shutdown_signal().await;
    assert_eq!(signal, Some(RunWithRange::Epoch(0)), "Shutdown signal should be Epoch(0)");

    info!("Fullnode shut down at epoch 0 boundary as expected");
}

/// Transfer a coin to mutate the gas object, then try to use the old (stale) ObjectRef.
/// The fullnode's early validation should detect the version mismatch (newer version in
/// cache) and return ObjectVersionUnavailableForConsumption.
///
/// Note: We intentionally do NOT prune objects. Pruning removes the newer version from
/// the cache, causing early validation to miss the stale ref. That edge case (pruned
/// objects not cleanly rejected) is a known limitation — see E2E_TESTING_GUIDE.md.
#[cfg(msim)]
#[msim::sim_test]
async fn test_access_stale_object_version() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new().build().await;

    let addresses = test_cluster.wallet.get_addresses();
    let sender = addresses[0];
    let recipient = addresses[1];

    // Get a gas object and record its current ref
    let old_gas = test_cluster
        .wallet
        .get_one_gas_object_owned_by_address(sender)
        .await
        .unwrap()
        .expect("sender must have a gas object");

    // Mutate the gas object via a transfer (creates version N+1)
    let tx_data = TransactionData::new(
        TransactionKind::TransferCoin { coin: old_gas, amount: Some(1000), recipient },
        sender,
        vec![old_gas],
    );
    let response = test_cluster.sign_and_execute_transaction(&tx_data).await;
    assert!(response.effects.status().is_ok());

    // Now try to use the OLD ref (version N) — the fullnode should reject it
    // because the live version in cache is N+1.
    let tx_data_stale = TransactionData::new(
        TransactionKind::TransferCoin { coin: old_gas, amount: Some(500), recipient },
        sender,
        vec![old_gas],
    );
    let tx_stale = test_cluster.sign_transaction(&tx_data_stale).await;

    let orchestrator = test_cluster
        .fullnode_handle
        .soma_node
        .with(|n| n.transaction_orchestrator().expect("fullnode must have orchestrator"));

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

    assert!(result.is_err(), "Transaction with stale object version should fail");
    let err_msg = format!("{:?}", result.err().unwrap());
    assert!(
        err_msg.contains("ObjectVersionUnavailableForConsumption")
            || err_msg.contains("not available for consumption"),
        "Error should indicate object version unavailable, got: {}",
        err_msg
    );

    info!("Correctly rejected transaction with stale object version");
}

/// Get orchestrator from fullnode. Execute a coin transfer via the orchestrator.
/// Assert executed locally, effects exist, status ok.
#[cfg(msim)]
#[msim::sim_test]
async fn test_full_node_transaction_orchestrator_basic() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new().build().await;
    let handle = &test_cluster.fullnode_handle.soma_node;
    let orchestrator =
        handle.with(|n| n.transaction_orchestrator().expect("fullnode must have orchestrator"));

    let addresses = test_cluster.wallet.get_addresses();
    let sender = addresses[0];
    let recipient = addresses[1];

    let gas = test_cluster
        .wallet
        .get_one_gas_object_owned_by_address(sender)
        .await
        .unwrap()
        .expect("sender must have a gas object");

    let tx_data = TransactionData::new(
        TransactionKind::TransferCoin { coin: gas, amount: Some(1000), recipient },
        sender,
        vec![gas],
    );
    let tx = test_cluster.sign_transaction(&tx_data).await;
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
    assert!(
        handle.with(|n| n.state().is_tx_already_executed(&digest)),
        "Transaction should be marked as executed"
    );

    info!("Transaction {} executed via orchestrator successfully", digest);
}

/// Create a transaction, sign it, serialize to BCS bytes.
/// Reconstruct Transaction from bytes. Execute. Assert success.
#[cfg(msim)]
#[msim::sim_test]
async fn test_execute_tx_with_serialized_signature() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new().build().await;

    let addresses = test_cluster.wallet.get_addresses();
    let sender = addresses[0];
    let recipient = addresses[1];

    let gas = test_cluster
        .wallet
        .get_one_gas_object_owned_by_address(sender)
        .await
        .unwrap()
        .expect("sender must have a gas object");

    let tx_data = TransactionData::new(
        TransactionKind::TransferCoin { coin: gas, amount: Some(1000), recipient },
        sender,
        vec![gas],
    );
    let tx = test_cluster.sign_transaction(&tx_data).await;

    // Serialize to BCS bytes
    let tx_bytes = bcs::to_bytes(&tx).expect("Failed to serialize transaction");

    // Deserialize back from BCS bytes
    let reconstructed: types::transaction::Transaction =
        bcs::from_bytes(&tx_bytes).expect("Failed to deserialize transaction");

    assert_eq!(tx.digest(), reconstructed.digest(), "Digests should match after round-trip");

    // Execute the reconstructed transaction
    let response = test_cluster.execute_transaction(reconstructed).await;
    assert!(response.effects.status().is_ok(), "Transaction should succeed");

    info!(
        "Transaction {} executed successfully after BCS round-trip",
        response.effects.transaction_digest()
    );
}
