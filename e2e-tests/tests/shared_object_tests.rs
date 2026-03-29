// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Tier 3: Shared object E2E tests.
//!
//! Rewrites of Sui's shared_objects_tests.rs and shared_objects_version_tests.rs
//! using SOMA's native shared objects and owned-object conflict scenarios.
//!
//! Tests:
//! 1. test_conflicting_owned_transactions_same_coin — Two transfers spending the same coin
//! 2. test_concurrent_conflicting_owned_transactions — Concurrent spends of same coin via orchestrator

use test_cluster::TestClusterBuilder;
use tracing::info;
use types::effects::TransactionEffectsAPI;
use types::object::Owner;
use types::quorum_driver::{ExecuteTransactionRequest, ExecuteTransactionRequestType};
use types::transaction::{TransactionData, TransactionKind};
use utils::logging::init_tracing;

// ===================================================================
// Test 1: Conflicting owned transactions — same coin
//
// Two transfers spending the same coin (same ObjectRef as input).
// The first transaction succeeds; the second fails because the coin's
// version has changed (ObjectVersionUnavailableForConsumption).
// ===================================================================

#[cfg(msim)]
#[msim::sim_test]
async fn test_conflicting_owned_transactions_same_coin() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new().build().await;

    let addresses = test_cluster.get_addresses();
    let sender = addresses[0];
    let recipient_a = addresses[1];
    let recipient_b = addresses[2];

    // Get a gas coin — this ObjectRef will be used by BOTH transactions
    let coin_ref = test_cluster
        .wallet
        .get_one_gas_object_owned_by_address(sender)
        .await
        .unwrap()
        .expect("Sender should have a gas object");

    info!("Using coin {} version {} for both transactions", coin_ref.0, coin_ref.1.value());

    // Create two TransferCoin transactions using the SAME ObjectRef
    let tx1_data = TransactionData::new(
        TransactionKind::Transfer {
            coins: vec![coin_ref],
            amounts: Some(1_000_000).map(|a| vec![a]),
            recipients: vec![recipient_a],
        },
        sender,
        vec![coin_ref],
    );

    let tx2_data = TransactionData::new(
        TransactionKind::Transfer {
            coins: vec![coin_ref],
            amounts: Some(1_000_000).map(|a| vec![a]),
            recipients: vec![recipient_b],
        },
        sender,
        vec![coin_ref],
    );

    // Execute first transaction — should succeed
    let response1 = test_cluster.sign_and_execute_transaction(&tx1_data).await;
    assert!(
        response1.effects.status().is_ok(),
        "First transfer should succeed: {:?}",
        response1.effects.status()
    );
    info!("First transfer succeeded, coin version now {}", response1.effects.version().value());

    // Execute second transaction with the STALE ObjectRef — should fail
    // The coin's version has changed, so the ObjectRef is now invalid
    let tx2 = test_cluster.wallet.sign_transaction(&tx2_data).await;
    let result2 = test_cluster.wallet.execute_transaction_may_fail(tx2).await;

    match result2 {
        Ok(response2) => {
            // If we got a response, the effects should show a failure
            assert!(
                response2.effects.status().is_err(),
                "Second transfer should fail with stale object version"
            );
            info!(
                "Second transfer correctly rejected with error in effects: {:?}",
                response2.effects.status()
            );
        }
        Err(e) => {
            // The transaction may fail at the orchestrator level (before execution)
            info!("Second transfer correctly rejected at submission: {}", e);
        }
    }

    // Verify the coin was only spent once — check that recipient_a received funds
    // but recipient_b did not get a new coin from this specific transfer
    let created_by_tx1 = response1.effects.created();
    let recipient_a_coin = created_by_tx1
        .iter()
        .any(|(_, owner)| matches!(owner, Owner::AddressOwner(addr) if *addr == recipient_a));
    assert!(recipient_a_coin, "recipient_a should have received a coin from tx1");

    info!("test_conflicting_owned_transactions_same_coin passed: equivocation correctly prevented");
}

// ===================================================================
// Test 2: Concurrent conflicting owned transactions
//
// Two transfers spending the same owned coin submitted concurrently
// via the orchestrator. Only one should succeed; the other gets
// ObjectsDoubleUsed from the quorum driver (validators lock the
// object to one transaction).
// Adapted from Sui's test_conflicting_owned_transactions (soft bundle).
// ===================================================================

#[cfg(msim)]
#[msim::sim_test]
async fn test_concurrent_conflicting_owned_transactions() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new().build().await;

    let addresses = test_cluster.get_addresses();
    let sender = addresses[0];
    let recipient_a = addresses[1];
    let recipient_b = addresses[2];

    // Get three coins: one contested coin + separate gas for each tx
    let coins =
        test_cluster.wallet.get_gas_objects_owned_by_address(sender, Some(3)).await.unwrap();
    assert!(coins.len() >= 3, "Sender needs at least 3 coins");
    let coin_ref = coins[0];
    let gas_ref1 = coins[1];
    let gas_ref2 = coins[2];

    info!("Concurrent spend of coin {} version {}", coin_ref.0, coin_ref.1.value());

    // Create and sign two conflicting TransferCoin transactions
    // Each uses a separate gas coin so only the transfer coin conflicts
    let tx1_data = TransactionData::new(
        TransactionKind::Transfer {
            coins: vec![coin_ref],
            amounts: Some(1_000_000).map(|a| vec![a]),
            recipients: vec![recipient_a],
        },
        sender,
        vec![gas_ref1],
    );

    let tx2_data = TransactionData::new(
        TransactionKind::Transfer {
            coins: vec![coin_ref],
            amounts: Some(1_000_000).map(|a| vec![a]),
            recipients: vec![recipient_b],
        },
        sender,
        vec![gas_ref2],
    );

    let signed_tx1 = test_cluster.wallet.sign_transaction(&tx1_data).await;
    let signed_tx2 = test_cluster.wallet.sign_transaction(&tx2_data).await;
    let digest1 = *signed_tx1.digest();
    let digest2 = *signed_tx2.digest();

    info!("TX1 (to recipient_a) digest: {}", digest1);
    info!("TX2 (to recipient_b) digest: {}", digest2);

    // Submit both concurrently via the orchestrator
    let orchestrator = test_cluster
        .fullnode_handle
        .soma_node
        .with(|n| n.transaction_orchestrator().expect("fullnode must have orchestrator"));

    let orch1 = orchestrator.clone();
    let orch2 = orchestrator.clone();

    let handle1 = tokio::task::spawn(async move {
        let request = ExecuteTransactionRequest {
            transaction: signed_tx1,
            include_input_objects: false,
            include_output_objects: false,
        };
        orch1
            .execute_transaction_block(
                request,
                ExecuteTransactionRequestType::WaitForLocalExecution,
                None,
            )
            .await
    });

    let handle2 = tokio::task::spawn(async move {
        let request = ExecuteTransactionRequest {
            transaction: signed_tx2,
            include_input_objects: false,
            include_output_objects: false,
        };
        orch2
            .execute_transaction_block(
                request,
                ExecuteTransactionRequestType::WaitForLocalExecution,
                None,
            )
            .await
    });

    let (result1, result2) = tokio::join!(handle1, handle2);
    let result1 = result1.unwrap();
    let result2 = result2.unwrap();

    // The safety property: at most one conflicting transaction succeeds (no double-spend).
    // With concurrent submission, it's possible for both to fail due to split object locks
    // (each tx locks on a different subset of validators, neither reaching quorum).
    let (successes, failures) =
        [("TX1", &result1), ("TX2", &result2)].iter().fold((0, 0), |(s, f), (name, result)| {
            match result {
                Ok((response, _)) => {
                    let effects = &response.effects.effects;
                    if effects.status().is_ok() {
                        info!("{} succeeded", name);
                        (s + 1, f)
                    } else {
                        info!("{} failed in effects: {:?}", name, effects.status());
                        (s, f + 1)
                    }
                }
                Err(e) => {
                    info!("{} failed at submission: {:?}", name, e);
                    (s, f + 1)
                }
            }
        });

    assert!(
        successes <= 1,
        "At most one conflicting owned transaction should succeed (got {})",
        successes
    );

    // Verify the coin was spent at most once
    let executed_count = [digest1, digest2]
        .iter()
        .filter(|d| {
            test_cluster.fullnode_handle.soma_node.with(|n| n.state().is_tx_already_executed(d))
        })
        .count();

    assert!(
        executed_count <= 1,
        "At most one conflicting transaction should be marked as executed (got {})",
        executed_count
    );

    info!(
        "test_concurrent_conflicting_owned_transactions passed: {} succeeded, {} failed",
        successes, failures
    );
}
