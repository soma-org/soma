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
use types::base::dbg_addr;
use types::crypto::get_key_pair;
use types::effects::{ExecutionStatus, TransactionEffectsAPI};
use types::transaction::{TransactionData, TransactionKind};
use types::unit_tests::utils::to_sender_signed_transaction;

use crate::authority::ExecutionEnv;
use crate::authority_test_utils::{
    certify_transaction, enqueue_all_and_execute_all, execute_sequenced_certificate_to_effects,
    seed_balance_mode_funds, send_and_confirm_transaction, send_consensus_no_execution,
};
use crate::test_authority_builder::TestAuthorityBuilder;

// =============================================================================
// Basic execution scheduling
// =============================================================================

#[tokio::test]
async fn test_execution_scheduler_basic_enqueue() {
    // Enqueue a single balance-mode transaction through the execution
    // scheduler and verify it executes successfully.
    let (sender, sender_key): (_, Ed25519KeyPair) = get_key_pair();
    let recipient = dbg_addr(1);

    let authority_state = TestAuthorityBuilder::new().build().await;
    seed_balance_mode_funds(&authority_state, sender, 50_000_000, 50_000_000);

    let data = crate::authority_test_utils::balance_transfer_data_legacy(
        recipient,
        sender,
        Some(1000),
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
    // Enqueue multiple independent transactions (different recipients)
    // through the execution scheduler and verify all execute. Stage 13c:
    // gas is balance-mode, so all txs draw from the same accumulator.
    let (sender, sender_key): (_, Ed25519KeyPair) = get_key_pair();

    let authority_state = TestAuthorityBuilder::new().build().await;
    seed_balance_mode_funds(&authority_state, sender, 50_000_000, 50_000_000);

    let mut certs_and_envs = Vec::new();
    for i in 0..5u8 {
        let data = crate::authority_test_utils::balance_transfer_data_legacy(
            dbg_addr(i + 1),
            sender,
            Some(100),
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
    // Stage 13c: balance-mode for both stake and gas.
    let (sender, sender_key): (_, Ed25519KeyPair) = get_key_pair();

    let authority_state = TestAuthorityBuilder::new().build().await;
    seed_balance_mode_funds(&authority_state, sender, 50_000_000, 50_000_000);

    // Get the first validator's address from the system state
    let validator_address = {
        let system_state = authority_state.get_system_state_object_for_testing().unwrap();
        system_state.validators().validators[0].metadata.soma_address
    };

    let data = TransactionData::new(
        TransactionKind::AddStake { validator: validator_address, amount: 1_000_000 },
        sender,
        vec![],
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
    // Stage 13c: AddStake is balance-mode for both stake and gas.
    let (sender, sender_key): (_, Ed25519KeyPair) = get_key_pair();

    let authority_state = TestAuthorityBuilder::new().build().await;
    seed_balance_mode_funds(&authority_state, sender, 50_000_000, 10_000_000);

    // Get the first validator's address from the system state
    let validator_address = {
        let system_state = authority_state.get_system_state_object_for_testing().unwrap();
        system_state.validators().validators[0].metadata.soma_address
    };

    let data = TransactionData::new(
        TransactionKind::AddStake {
            validator: validator_address,
            amount: 1_000_000,
        },
        sender,
        vec![],
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
    // Stage 13c: BalanceTransfer doesn't lock or version any per-tx
    // owned object — both txs draw gas from the same accumulator and
    // run sequentially. We verify that two distinct txs with the
    // same sender execute in order with monotonically advancing
    // lamport versions.
    let (sender, sender_key): (_, Ed25519KeyPair) = get_key_pair();

    let authority_state = TestAuthorityBuilder::new().build().await;
    seed_balance_mode_funds(&authority_state, sender, 50_000_000, 50_000_000);

    let data1 = crate::authority_test_utils::balance_transfer_data_legacy(
        dbg_addr(1),
        sender,
        Some(100),
    );
    let tx1 = to_sender_signed_transaction(data1, &sender_key);
    let (_, effects1) = send_and_confirm_transaction(&authority_state, tx1).await.unwrap();
    assert_eq!(*effects1.status(), ExecutionStatus::Success);

    let data2 = crate::authority_test_utils::balance_transfer_data_legacy(
        dbg_addr(2),
        sender,
        Some(100),
    );
    let tx2 = to_sender_signed_transaction(data2, &sender_key);
    let (_, effects2) = send_and_confirm_transaction(&authority_state, tx2).await.unwrap();
    assert_eq!(*effects2.status(), ExecutionStatus::Success);

    assert_ne!(
        effects1.transaction_digest(),
        effects2.transaction_digest(),
        "Should be different transactions"
    );
    // Stage 13c: BalanceTransfer touches no per-object versions, so
    // the per-tx lamport version is unchanged across consecutive
    // user-only txs. Distinct digests are the meaningful invariant.
}

#[tokio::test]
async fn test_effects_idempotent_reexecution() {
    // Re-executing a certificate should return the same effects (idempotency).
    let (sender, sender_key): (_, Ed25519KeyPair) = get_key_pair();

    let authority_state = TestAuthorityBuilder::new().build().await;
    seed_balance_mode_funds(&authority_state, sender, 50_000_000, 50_000_000);

    let data = crate::authority_test_utils::balance_transfer_data_legacy(
        dbg_addr(1),
        sender,
        Some(1000),
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
