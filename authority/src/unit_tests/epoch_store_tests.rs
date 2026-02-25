// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use fastcrypto::ed25519::Ed25519KeyPair;
use types::base::dbg_addr;
use types::crypto::{SomaKeyPair, get_key_pair};
use types::effects::{ExecutionStatus, TransactionEffectsAPI};
use types::object::{Object, ObjectID};
use types::system_state::epoch_start::EpochStartSystemStateTrait;
use types::transaction::TransactionData;
use types::unit_tests::utils::to_sender_signed_transaction;

use crate::authority_test_utils::send_and_confirm_transaction;
use crate::test_authority_builder::TestAuthorityBuilder;

// =============================================================================
// Basic epoch store properties
// =============================================================================

#[tokio::test]
async fn test_epoch_store_basic_properties() {
    // Verify the epoch store has the correct epoch number and committee info
    // after genesis initialization.
    let authority_state = TestAuthorityBuilder::new().build().await;
    let epoch_store = authority_state.epoch_store_for_testing();

    // Epoch should be 0 after genesis
    assert_eq!(epoch_store.epoch(), 0, "Epoch should be 0 after genesis");

    // Committee should contain at least one validator
    let committee = epoch_store.committee();
    assert!(!committee.voting_rights.is_empty(), "Committee should have voting members");

    // Protocol config should be accessible
    let protocol_config = epoch_store.protocol_config();
    assert!(protocol_config.version.as_u64() >= 1, "Protocol version should be at least 1");

    // This node should be a validator in this epoch
    assert!(epoch_store.is_validator(), "Test authority should be a validator in its own epoch");
}

#[tokio::test]
async fn test_epoch_store_epoch_start_state() {
    // Verify epoch start configuration is properly accessible.
    let authority_state = TestAuthorityBuilder::new().build().await;
    let epoch_store = authority_state.epoch_store_for_testing();

    let epoch_start_state = epoch_store.epoch_start_state();
    assert_eq!(epoch_start_state.epoch(), 0, "Epoch start state should report epoch 0");
}

// =============================================================================
// Signed transaction storage
// =============================================================================

#[tokio::test]
async fn test_epoch_store_signed_transaction_storage() {
    // After executing a transaction, the epoch store should have the signed
    // transaction retrievable by digest.
    let (sender, sender_key): (_, Ed25519KeyPair) = get_key_pair();
    let coin_id = ObjectID::random();
    let coin = Object::with_id_owner_coin_for_testing(coin_id, sender, 10_000_000);

    let authority_state = TestAuthorityBuilder::new().build().await;
    authority_state.insert_genesis_object(coin.clone()).await;

    let data = TransactionData::new_transfer_coin(
        dbg_addr(1),
        sender,
        Some(1000),
        coin.compute_object_reference(),
    );
    let tx = to_sender_signed_transaction(data, &sender_key);
    let tx_digest = *tx.digest();

    // Execute the transaction (this will sign and store)
    let (_, effects) = send_and_confirm_transaction(&authority_state, tx).await.unwrap();
    assert_eq!(*effects.status(), ExecutionStatus::Success);

    // Retrieve the signed transaction from epoch store
    let epoch_store = authority_state.epoch_store_for_testing();
    let signed_tx = epoch_store.get_signed_transaction(&tx_digest);
    assert!(signed_tx.is_ok(), "get_signed_transaction should not error: {:?}", signed_tx.err());
    assert!(
        signed_tx.unwrap().is_some(),
        "Signed transaction should exist in epoch store after execution"
    );
}

#[tokio::test]
async fn test_epoch_store_signed_transaction_not_found() {
    // Querying for a non-existent signed transaction should return None.
    let authority_state = TestAuthorityBuilder::new().build().await;
    let epoch_store = authority_state.epoch_store_for_testing();

    let random_digest = types::digests::TransactionDigest::random();
    let result = epoch_store.get_signed_transaction(&random_digest);
    assert!(result.is_ok(), "Should not error");
    assert!(result.unwrap().is_none(), "Non-existent transaction should return None");
}

// =============================================================================
// Effects signatures storage
// =============================================================================

#[tokio::test]
async fn test_epoch_store_effects_signatures() {
    // After executing a transaction, the epoch store should have an effects
    // signature stored that is retrievable by the transaction digest.
    let (sender, sender_key): (_, Ed25519KeyPair) = get_key_pair();
    let coin_id = ObjectID::random();
    let coin = Object::with_id_owner_coin_for_testing(coin_id, sender, 10_000_000);

    let authority_state = TestAuthorityBuilder::new().build().await;
    authority_state.insert_genesis_object(coin.clone()).await;

    let data = TransactionData::new_transfer_coin(
        dbg_addr(1),
        sender,
        Some(1000),
        coin.compute_object_reference(),
    );
    let tx = to_sender_signed_transaction(data, &sender_key);
    let tx_digest = *tx.digest();

    let (_, effects) = send_and_confirm_transaction(&authority_state, tx).await.unwrap();
    assert_eq!(*effects.status(), ExecutionStatus::Success);

    let epoch_store = authority_state.epoch_store_for_testing();

    // Effects signature should exist
    let effects_sig = epoch_store.get_effects_signature(&tx_digest);
    assert!(effects_sig.is_ok(), "get_effects_signature should not error: {:?}", effects_sig.err());
    assert!(
        effects_sig.unwrap().is_some(),
        "Effects signature should exist after transaction execution"
    );
}

#[tokio::test]
async fn test_epoch_store_effects_signature_not_found() {
    // Querying effects signature for a non-existent transaction should return None.
    let authority_state = TestAuthorityBuilder::new().build().await;
    let epoch_store = authority_state.epoch_store_for_testing();

    let random_digest = types::digests::TransactionDigest::random();
    let result = epoch_store.get_effects_signature(&random_digest);
    assert!(result.is_ok(), "Should not error");
    assert!(result.unwrap().is_none(), "Non-existent effects signature should return None");
}

// =============================================================================
// Reconfig state
// =============================================================================

#[tokio::test]
async fn test_epoch_store_reconfig_state_allows_user_certs() {
    // At genesis, the reconfig state should allow user certificates.
    let authority_state = TestAuthorityBuilder::new().build().await;
    let epoch_store = authority_state.epoch_store_for_testing();

    let reconfig_guard = epoch_store.get_reconfig_state_read_lock_guard();
    assert!(
        reconfig_guard.should_accept_user_certs(),
        "Reconfig state should accept user certs at genesis"
    );
}
