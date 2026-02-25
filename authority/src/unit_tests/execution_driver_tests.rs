// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

// Tests for execution scheduling and the execution driver.
// These exercise the execution scheduler's enqueue/execute flow,
// shared object version assignment, and dependency ordering.
//
// Adapted from Sui's execution_driver_tests.rs patterns, but simplified
// because SOMA lacks LocalAuthorityClient and multi-authority unit test
// infrastructure. Multi-authority execution tests are covered by E2E tests.
//
// Portions of this file are derived from Sui (MystenLabs/sui).
// Original source: https://github.com/MystenLabs/sui/tree/main/crates/sui-core/src/unit_tests/execution_driver_tests.rs
// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

use fastcrypto::ed25519::Ed25519KeyPair;
use types::{
    base::dbg_addr,
    crypto::{SomaKeyPair, get_key_pair},
    effects::{ExecutionStatus, TransactionEffectsAPI},
    object::{Object, ObjectID},
    transaction::{TransactionData, TransactionKind},
    unit_tests::utils::to_sender_signed_transaction,
};

use crate::{
    authority::ExecutionEnv,
    authority_test_utils::{
        certify_transaction, enqueue_all_and_execute_all, execute_sequenced_certificate_to_effects,
        send_and_confirm_transaction, send_consensus_no_execution,
    },
    test_authority_builder::TestAuthorityBuilder,
};

// =============================================================================
// Basic execution scheduling
// =============================================================================

#[tokio::test]
async fn test_execution_scheduler_basic_enqueue() {
    // Enqueue a single owned-object transaction through the execution scheduler
    // and verify it executes successfully.
    let (sender, sender_key): (_, Ed25519KeyPair) = get_key_pair();
    let recipient = dbg_addr(1);
    let coin_id = ObjectID::random();
    let coin = Object::with_id_owner_coin_for_testing(coin_id, sender, 50_000_000);

    let authority_state = TestAuthorityBuilder::new().build().await;
    authority_state.insert_genesis_object(coin.clone()).await;

    let data = TransactionData::new_transfer_coin(
        recipient,
        sender,
        Some(1000),
        coin.compute_object_reference(),
    );
    let tx = to_sender_signed_transaction(data, &sender_key);
    let cert = certify_transaction(&authority_state, tx).await.unwrap();

    // Execute via the scheduler path
    let results =
        enqueue_all_and_execute_all(&authority_state, vec![(cert.clone(), ExecutionEnv::new())])
            .await
            .unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(*results[0].status(), ExecutionStatus::Success);
}

#[tokio::test]
async fn test_execution_scheduler_multiple_independent_txns() {
    // Enqueue multiple independent transactions (different gas objects)
    // through the execution scheduler and verify all execute.
    let (sender, sender_key): (_, Ed25519KeyPair) = get_key_pair();

    let authority_state = TestAuthorityBuilder::new().build().await;

    let mut certs_and_envs = Vec::new();
    for i in 0..5u8 {
        let coin_id = ObjectID::random();
        let coin = Object::with_id_owner_coin_for_testing(coin_id, sender, 50_000_000);
        authority_state.insert_genesis_object(coin.clone()).await;

        let data = TransactionData::new_transfer_coin(
            dbg_addr(i + 1),
            sender,
            Some(100),
            coin.compute_object_reference(),
        );
        let tx = to_sender_signed_transaction(data, &sender_key);
        let cert = certify_transaction(&authority_state, tx).await.unwrap();
        certs_and_envs.push((cert, ExecutionEnv::new()));
    }

    let results = enqueue_all_and_execute_all(&authority_state, certs_and_envs).await.unwrap();

    assert_eq!(results.len(), 5);
    for (i, effects) in results.iter().enumerate() {
        assert_eq!(*effects.status(), ExecutionStatus::Success, "Transaction {} should succeed", i);
    }
}

// =============================================================================
// Shared object version assignment
// =============================================================================

#[tokio::test]
async fn test_shared_object_version_assignment() {
    // Verify that shared object version assignment works correctly for
    // AddStake transactions (which use the SystemState shared object).
    let (sender, sender_key): (_, Ed25519KeyPair) = get_key_pair();
    let coin_id = ObjectID::random();
    let coin = Object::with_id_owner_coin_for_testing(coin_id, sender, 50_000_000);

    let authority_state = TestAuthorityBuilder::new().build().await;
    authority_state.insert_genesis_object(coin.clone()).await;

    // Get the first validator's address from the system state
    let validator_address = {
        let system_state = authority_state.get_system_state_object_for_testing().unwrap();
        system_state.validators().validators[0].metadata.soma_address
    };

    // AddStake uses SystemState (shared object)
    let coin_ref = coin.compute_object_reference();
    let data = TransactionData::new(
        TransactionKind::AddStake { address: validator_address, coin_ref, amount: Some(1_000_000) },
        sender,
        vec![coin_ref],
    );
    let tx = to_sender_signed_transaction(data, &sender_key);
    let cert = certify_transaction(&authority_state, tx).await.unwrap();

    // Assign versions through the consensus path
    let assigned_versions = send_consensus_no_execution(&authority_state, &cert).await;

    // The assigned versions should contain at least the SystemState object
    assert!(
        !assigned_versions.shared_object_versions.is_empty(),
        "Assigned versions should contain SystemState shared object version"
    );
}

#[tokio::test]
async fn test_execute_sequenced_shared_object_transaction() {
    // Execute a shared-object transaction through the sequenced path
    // (assign versions then execute), verifying correct execution.
    let (sender, sender_key): (_, Ed25519KeyPair) = get_key_pair();
    let coin_id = ObjectID::random();
    let coin = Object::with_id_owner_coin_for_testing(coin_id, sender, 50_000_000);

    let authority_state = TestAuthorityBuilder::new().build().await;
    authority_state.insert_genesis_object(coin.clone()).await;

    // Get the first validator's address from the system state
    let validator_address = {
        let system_state = authority_state.get_system_state_object_for_testing().unwrap();
        system_state.validators().validators[0].metadata.soma_address
    };

    let coin_ref = coin.compute_object_reference();
    let data = TransactionData::new(
        TransactionKind::AddStake { address: validator_address, coin_ref, amount: Some(1_000_000) },
        sender,
        vec![coin_ref],
    );
    let tx = to_sender_signed_transaction(data, &sender_key);
    let cert = certify_transaction(&authority_state, tx).await.unwrap();

    // Assign versions and execute
    let assigned_versions = send_consensus_no_execution(&authority_state, &cert).await;
    let (effects, exec_error) =
        execute_sequenced_certificate_to_effects(&authority_state, cert, assigned_versions).await;

    assert_eq!(*effects.status(), ExecutionStatus::Success);
    assert!(exec_error.is_none(), "Should have no execution error");
}

// =============================================================================
// Sequential dependent transactions through scheduler
// =============================================================================

#[tokio::test]
async fn test_dependent_transactions_execute_in_order() {
    // Execute two transactions that depend on the same object (coin)
    // sequentially, verifying the second sees the updated version.
    let (sender, sender_key): (_, Ed25519KeyPair) = get_key_pair();
    let coin_id = ObjectID::random();
    let coin = Object::with_id_owner_coin_for_testing(coin_id, sender, 50_000_000);

    let authority_state = TestAuthorityBuilder::new().build().await;
    authority_state.insert_genesis_object(coin.clone()).await;

    // First transfer
    let data1 = TransactionData::new_transfer_coin(
        dbg_addr(1),
        sender,
        Some(100),
        coin.compute_object_reference(),
    );
    let tx1 = to_sender_signed_transaction(data1, &sender_key);
    let (_, effects1) = send_and_confirm_transaction(&authority_state, tx1).await.unwrap();
    assert_eq!(*effects1.status(), ExecutionStatus::Success);

    // Get updated coin ref after first tx
    let updated_coin = authority_state.get_object(&coin_id).await.unwrap();
    let updated_ref = updated_coin.compute_object_reference();

    // Second transfer using updated ref
    let data2 = TransactionData::new_transfer_coin(dbg_addr(2), sender, Some(100), updated_ref);
    let tx2 = to_sender_signed_transaction(data2, &sender_key);
    let (_, effects2) = send_and_confirm_transaction(&authority_state, tx2).await.unwrap();
    assert_eq!(*effects2.status(), ExecutionStatus::Success);

    // Verify the second transaction's effects reference the correct version
    assert_ne!(
        effects1.transaction_digest(),
        effects2.transaction_digest(),
        "Should be different transactions"
    );

    // Coin version should have increased twice
    let final_coin = authority_state.get_object(&coin_id).await.unwrap();
    assert!(
        final_coin.version() > updated_coin.version(),
        "Version should increase after second transaction"
    );
}

#[tokio::test]
async fn test_effects_idempotent_reexecution() {
    // Re-executing a certificate should return the same effects (idempotency).
    let (sender, sender_key): (_, Ed25519KeyPair) = get_key_pair();
    let coin_id = ObjectID::random();
    let coin = Object::with_id_owner_coin_for_testing(coin_id, sender, 50_000_000);

    let authority_state = TestAuthorityBuilder::new().build().await;
    authority_state.insert_genesis_object(coin.clone()).await;

    let data = TransactionData::new_transfer_coin(
        dbg_addr(1),
        sender,
        Some(1000),
        coin.compute_object_reference(),
    );
    let tx = to_sender_signed_transaction(data, &sender_key);
    let cert = certify_transaction(&authority_state, tx).await.unwrap();

    // Execute the first time
    let (effects1, _) = authority_state.try_execute_for_test(&cert, ExecutionEnv::new()).await;
    assert_eq!(*effects1.status(), ExecutionStatus::Success);

    // Execute again — should return same effects
    let (effects2, _) = authority_state.try_execute_for_test(&cert, ExecutionEnv::new()).await;

    assert_eq!(
        effects1.digest(),
        effects2.digest(),
        "Re-executing a certificate should produce identical effects"
    );
}
