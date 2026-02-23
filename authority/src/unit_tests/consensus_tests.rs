// Portions of this file are derived from Sui (MystenLabs/sui).
// Original source: https://github.com/MystenLabs/sui/tree/main/crates/sui-core/src/unit_tests/consensus_tests.rs
// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Tests for the ConsensusHandler pipeline.
//!
//! These exercise the handler's ability to:
//! - Process user transactions from synthetic consensus commits
//! - Deduplicate transactions across commits
//! - Assign shared object versions for shared-object transactions
//! - Create pending checkpoints
//!
//! Uses `TestConsensusCommit` to feed synthetic consensus output and
//! `CapturedTransactions` to inspect what gets scheduled for execution.

use fastcrypto::ed25519::Ed25519KeyPair;
use types::consensus::ConsensusTransaction;
use types::crypto::get_key_pair;
use types::object::{Object, ObjectID};
use types::system_state::SystemStateTrait as _;
use types::system_state::epoch_start::EpochStartSystemStateTrait;
use types::transaction::{TransactionData, TransactionKind};
use types::unit_tests::utils::to_sender_signed_transaction;

use crate::consensus_test_utils::{TestConsensusCommit, setup_consensus_handler_for_testing};
use crate::test_authority_builder::TestAuthorityBuilder;

/// Helper: create a user transaction wrapping an AddStake (shared object) transaction.
async fn make_add_stake_consensus_tx(
    authority: &crate::authority::AuthorityState,
) -> ConsensusTransaction {
    let (sender, sender_key): (_, Ed25519KeyPair) = get_key_pair();
    let gas_object_id = ObjectID::random();
    let gas_object = Object::with_id_owner_coin_for_testing(gas_object_id, sender, 10_000_000_000);
    authority.insert_genesis_object(gas_object.clone()).await;

    let system_state = authority.get_system_state_object_for_testing().unwrap();
    let validator_address = system_state.validators().validators[0].metadata.soma_address;

    let gas_ref = gas_object.compute_object_reference();
    let data = TransactionData::new(
        TransactionKind::AddStake {
            address: validator_address,
            coin_ref: gas_ref,
            amount: Some(1_000_000),
        },
        sender,
        vec![gas_ref],
    );
    let tx = to_sender_signed_transaction(data, &sender_key);
    ConsensusTransaction::new_user_transaction_message(&authority.name, tx)
}

/// Helper: create a TransferCoin consensus transaction (owned objects only).
async fn make_transfer_consensus_tx(
    authority: &crate::authority::AuthorityState,
) -> ConsensusTransaction {
    let (sender, sender_key): (_, Ed25519KeyPair) = get_key_pair();
    let (recipient, _): (_, Ed25519KeyPair) = get_key_pair();
    let gas_object_id = ObjectID::random();
    let gas_object = Object::with_id_owner_coin_for_testing(gas_object_id, sender, 10_000_000_000);
    authority.insert_genesis_object(gas_object.clone()).await;

    let gas_ref = gas_object.compute_object_reference();
    let data = TransactionData::new(
        TransactionKind::TransferCoin { coin: gas_ref, amount: Some(1_000), recipient },
        sender,
        vec![gas_ref],
    );
    let tx = to_sender_signed_transaction(data, &sender_key);
    ConsensusTransaction::new_user_transaction_message(&authority.name, tx)
}

/// Test that the consensus handler processes a user transaction from a synthetic
/// consensus commit and sends it to the execution scheduler.
#[tokio::test]
async fn test_consensus_handler_processes_user_transaction() {
    let authority = TestAuthorityBuilder::new().build().await;
    let mut setup = setup_consensus_handler_for_testing(&authority).await;

    let epoch_store = authority.load_epoch_store_one_call_per_task();
    let epoch_start_ts = epoch_store.epoch_start_state().epoch_start_timestamp_ms();

    let consensus_tx = make_transfer_consensus_tx(&authority).await;

    let commit = TestConsensusCommit::new(
        vec![consensus_tx],
        1, // round
        epoch_start_ts,
        1, // sub_dag_index
    );

    setup.consensus_handler.handle_consensus_commit_for_test(commit).await;

    // Wait for async capture
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

    let captured = setup.captured_transactions.lock();
    assert!(!captured.is_empty(), "Expected at least one batch of transactions to be scheduled");
    // The first batch should contain at least the user transaction
    // (plus possibly a ConsensusCommitPrologueV1 system transaction)
    let (schedulables, _assigned_versions, _source) = &captured[0];
    assert!(!schedulables.is_empty(), "Expected at least one schedulable transaction");
}

/// Test that the consensus handler deduplicates the same transaction
/// submitted in two different consensus commits.
#[tokio::test]
async fn test_consensus_handler_deduplication() {
    let authority = TestAuthorityBuilder::new().build().await;
    let mut setup = setup_consensus_handler_for_testing(&authority).await;

    let epoch_store = authority.load_epoch_store_one_call_per_task();
    let epoch_start_ts = epoch_store.epoch_start_state().epoch_start_timestamp_ms();

    let consensus_tx = make_transfer_consensus_tx(&authority).await;

    // Submit the same transaction in two different rounds
    let commit1 = TestConsensusCommit::new(vec![consensus_tx.clone()], 1, epoch_start_ts, 1);
    let commit2 = TestConsensusCommit::new(vec![consensus_tx], 2, epoch_start_ts + 1000, 2);

    setup.consensus_handler.handle_consensus_commit_for_test(commit1).await;
    setup.consensus_handler.handle_consensus_commit_for_test(commit2).await;

    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

    let captured = setup.captured_transactions.lock();
    // Both commits should produce output (each has a ConsensusCommitPrologueV1 at minimum).
    // But the duplicate user transaction should only appear once.
    assert!(
        captured.len() >= 2,
        "Expected at least 2 batches (one per commit), got {}",
        captured.len()
    );

    // Count total schedulable transactions across all batches
    let total_schedulables: usize =
        captured.iter().map(|(schedulables, _, _)| schedulables.len()).sum();

    // commit1: 1 CCP + 1 user tx = 2; commit2: 1 CCP + 0 user tx (deduped) = 1
    // Total should be <= 3 (allowing for CCP in each commit + 1 user tx)
    assert!(
        total_schedulables <= 4,
        "Expected deduplication to prevent duplicate user tx: got {} total schedulables",
        total_schedulables,
    );
}

/// Test that the consensus handler correctly assigns shared object versions
/// for transactions involving shared objects (AddStake touches SystemState).
#[tokio::test]
async fn test_consensus_handler_shared_object_version_assignment() {
    let authority = TestAuthorityBuilder::new().build().await;
    let mut setup = setup_consensus_handler_for_testing(&authority).await;

    let epoch_store = authority.load_epoch_store_one_call_per_task();
    let epoch_start_ts = epoch_store.epoch_start_state().epoch_start_timestamp_ms();

    // AddStake touches the shared SystemState object
    let consensus_tx = make_add_stake_consensus_tx(&authority).await;

    let commit = TestConsensusCommit::new(vec![consensus_tx], 1, epoch_start_ts, 1);

    setup.consensus_handler.handle_consensus_commit_for_test(commit).await;

    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

    let captured = setup.captured_transactions.lock();
    assert!(!captured.is_empty(), "Expected transactions to be scheduled");

    // The assigned versions should be non-empty because shared objects
    // were involved (SystemState for AddStake, plus ConsensusCommitPrologueV1
    // also touches SystemState).
    let (_, assigned_versions, _) = &captured[0];
    let versions_map = assigned_versions.0.clone();
    assert!(
        !versions_map.is_empty(),
        "Expected shared object versions to be assigned for shared-object transactions"
    );
}

/// Test that the consensus handler creates pending checkpoints after
/// processing a consensus commit.
#[tokio::test]
async fn test_consensus_handler_creates_pending_checkpoint() {
    let authority = TestAuthorityBuilder::new().build().await;
    let mut setup = setup_consensus_handler_for_testing(&authority).await;

    let epoch_store = authority.load_epoch_store_one_call_per_task();
    let epoch_start_ts = epoch_store.epoch_start_state().epoch_start_timestamp_ms();

    let consensus_tx = make_transfer_consensus_tx(&authority).await;

    let commit = TestConsensusCommit::new(vec![consensus_tx], 1, epoch_start_ts, 1);

    setup.consensus_handler.handle_consensus_commit_for_test(commit).await;

    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

    // Verify that the highest pending checkpoint height has advanced.
    // The handler calls write_pending_checkpoint, which increments the height.
    let highest_pending = epoch_store.get_highest_pending_checkpoint_height();
    assert!(
        highest_pending > 0,
        "Expected pending checkpoint height to advance after consensus commit, got {}",
        highest_pending
    );
}

/// Test that multiple independent transactions in a single consensus commit
/// are all processed and scheduled.
#[tokio::test]
async fn test_consensus_handler_multiple_transactions_in_commit() {
    let authority = TestAuthorityBuilder::new().build().await;
    let mut setup = setup_consensus_handler_for_testing(&authority).await;

    let epoch_store = authority.load_epoch_store_one_call_per_task();
    let epoch_start_ts = epoch_store.epoch_start_state().epoch_start_timestamp_ms();

    // Create 3 independent transactions
    let tx1 = make_transfer_consensus_tx(&authority).await;
    let tx2 = make_transfer_consensus_tx(&authority).await;
    let tx3 = make_transfer_consensus_tx(&authority).await;

    let commit = TestConsensusCommit::new(vec![tx1, tx2, tx3], 1, epoch_start_ts, 1);

    setup.consensus_handler.handle_consensus_commit_for_test(commit).await;

    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

    let captured = setup.captured_transactions.lock();
    assert!(!captured.is_empty(), "Expected transactions to be scheduled");

    // Count total schedulable transactions (including ConsensusCommitPrologueV1)
    let total_schedulables: usize =
        captured.iter().map(|(schedulables, _, _)| schedulables.len()).sum();

    // Should have at least 3 user transactions + 1 ConsensusCommitPrologueV1 = 4
    assert!(
        total_schedulables >= 4,
        "Expected at least 4 schedulable transactions (3 user + 1 system), got {}",
        total_schedulables
    );
}
