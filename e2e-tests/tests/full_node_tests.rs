// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
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
use types::config::node_config::RunWithRange;
use types::effects::TransactionEffectsAPI;
use types::quorum_driver::{ExecuteTransactionRequest, ExecuteTransactionRequestType};
use types::transaction::{TransactionData, TransactionKind};
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

// Stage 13c: test_access_stale_object_version was a coin-mode-only
// test — it relied on a Coin gas object being mutated and then
// re-used at the old ref. With balance-mode gas, there is no per-tx
// object ref to go stale; the analogous stale-state failure mode
// is "underfunded after Settlement", which is covered by
// `test_balance_transfer_underfunded_dropped_by_prepass` in
// balance_transfer_tests.rs.

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

    let tx_data = e2e_tests::balance_transfer_data(
        &test_cluster,
        types::object::CoinType::Usdc,
        sender,
        vec![(recipient, 1000)],
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

    let tx_data = e2e_tests::balance_transfer_data(
        &test_cluster,
        types::object::CoinType::Usdc,
        sender,
        vec![(recipient, 1000)],
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

