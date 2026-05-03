// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use fastcrypto::ed25519::Ed25519KeyPair;
use types::SYSTEM_STATE_OBJECT_ID;
use types::base::dbg_addr;
use types::crypto::{SomaKeyPair, get_key_pair};
use types::effects::{ExecutionStatus, TransactionEffectsAPI};
use types::object::{Object, ObjectID, Owner};
use types::system_state::SystemStateTrait as _;
use types::transaction::{
    SenderSignedData, TransactionData, TransactionKind,
    verify_sender_signed_data_message_signatures,
};
use types::unit_tests::utils::to_sender_signed_transaction;

use crate::authority_test_utils::{
    certify_transaction, execute_certificate_with_execution_error, init_state_with_ids,
    send_and_confirm_transaction, send_and_confirm_transaction_,
};
use crate::test_authority_builder::TestAuthorityBuilder;

// =============================================================================
// Transfer transaction tests
// =============================================================================

#[tokio::test]
async fn test_handle_transfer_transaction_ok() {
    // Basic object transfer should succeed
    let (sender, sender_key): (_, Ed25519KeyPair) = get_key_pair();
    let recipient = dbg_addr(1);
    let object_id = ObjectID::random();

    let authority_state = init_state_with_ids(vec![(sender, object_id)]).await;
    crate::authority_test_utils::seed_balance_mode_funds(
        &authority_state,
        sender,
        0,
        10_000_000,
    );

    let obj = authority_state.get_object(&object_id).await.unwrap();

    let data = TransactionData::new_transfer(
        recipient,
        obj.compute_object_reference(),
        sender,
        vec![],
    );
    let tx = to_sender_signed_transaction(data, &sender_key);
    let (_, effects) = send_and_confirm_transaction(&authority_state, tx).await.unwrap();
    assert_eq!(*effects.status(), ExecutionStatus::Success);

    // Object should now be owned by recipient
    let transferred = authority_state.get_object(&object_id).await.unwrap();
    assert_eq!(transferred.owner, Owner::AddressOwner(recipient));
}

#[tokio::test]
async fn test_handle_transfer_receiver_equal_sender() {
    // Self-transfer should succeed
    let (sender, sender_key): (_, Ed25519KeyPair) = get_key_pair();
    let object_id = ObjectID::random();

    let authority_state = init_state_with_ids(vec![(sender, object_id)]).await;
    crate::authority_test_utils::seed_balance_mode_funds(
        &authority_state,
        sender,
        0,
        10_000_000,
    );

    let obj = authority_state.get_object(&object_id).await.unwrap();

    let data = TransactionData::new_transfer(
        sender, // recipient == sender
        obj.compute_object_reference(),
        sender,
        vec![],
    );
    let tx = to_sender_signed_transaction(data, &sender_key);
    let result = send_and_confirm_transaction(&authority_state, tx).await;
    assert!(result.is_ok(), "Self-transfer should succeed");

    let (_, effects) = result.unwrap();
    assert_eq!(*effects.status(), ExecutionStatus::Success);

    // Object should still be owned by sender
    let obj_after = authority_state.get_object(&object_id).await.unwrap();
    assert_eq!(obj_after.owner, Owner::AddressOwner(sender));
}

#[tokio::test]
async fn test_handle_transfer_double_spend() {
    // Transferring the same object twice should fail on the second attempt
    let (sender, sender_key): (_, Ed25519KeyPair) = get_key_pair();
    let recipient1 = dbg_addr(1);
    let recipient2 = dbg_addr(2);
    let object_id = ObjectID::random();

    let authority_state = init_state_with_ids(vec![(sender, object_id)]).await;
    crate::authority_test_utils::seed_balance_mode_funds(
        &authority_state,
        sender,
        0,
        10_000_000,
    );

    let obj = authority_state.get_object(&object_id).await.unwrap();

    // First transfer succeeds
    let data1 = TransactionData::new_transfer(
        recipient1,
        obj.compute_object_reference(),
        sender,
        vec![],
    );
    let tx1 = to_sender_signed_transaction(data1, &sender_key);
    let result1 = send_and_confirm_transaction(&authority_state, tx1).await;
    assert!(result1.is_ok(), "First transfer should succeed");

    // Second transfer with stale object ref should fail
    let data2 = TransactionData::new_transfer(
        recipient2,
        obj.compute_object_reference(), // stale ref
        sender,
        vec![],
    );
    let tx2 = to_sender_signed_transaction(data2, &sender_key);
    let result2 = send_and_confirm_transaction(&authority_state, tx2).await;
    assert!(result2.is_err(), "Double-spend should be rejected");
}

#[tokio::test]
async fn test_object_not_found() {
    // Requesting a non-existent object should return None
    let authority_state = TestAuthorityBuilder::new().build().await;
    let nonexistent = ObjectID::random();

    let result = authority_state.get_object(&nonexistent).await;
    assert!(result.is_none(), "Non-existent object should return None");
}

// =============================================================================
// Effects consistency tests
// =============================================================================

#[tokio::test]
async fn test_effects_internal_consistency() {
    // Verify that effects have consistent created/mutated/deleted sets
    let (sender, sender_key): (_, Ed25519KeyPair) = get_key_pair();
    let recipient = dbg_addr(1);

    let authority_state = TestAuthorityBuilder::new().build().await;
    crate::authority_test_utils::seed_balance_mode_funds(
        &authority_state,
        sender,
        10_000_000,
        10_000_000,
    );

    let data = crate::authority_test_utils::balance_transfer_data_legacy(
        recipient,
        sender,
        Some(1000),
    );
    let tx = to_sender_signed_transaction(data, &sender_key);
    let (_, effects) = send_and_confirm_transaction(&authority_state, tx).await.unwrap();
    let effects = effects.into_data();

    assert_eq!(*effects.status(), ExecutionStatus::Success);

    // Verify no object appears in multiple sets
    let created_ids: Vec<ObjectID> = effects.created().iter().map(|c| c.0.0).collect();
    let mutated_ids: Vec<ObjectID> = effects.mutated().iter().map(|m| m.0.0).collect();
    let deleted_ids: Vec<ObjectID> = effects.deleted().iter().map(|d| d.0).collect();

    for id in &created_ids {
        assert!(!mutated_ids.contains(id), "Object {:?} in both created and mutated", id);
        assert!(!deleted_ids.contains(id), "Object {:?} in both created and deleted", id);
    }
    for id in &mutated_ids {
        assert!(!deleted_ids.contains(id), "Object {:?} in both mutated and deleted", id);
    }

    // Stage 13c: BalanceTransfer creates / mutates / deletes
    // nothing — both gas (USDC) and the SOMA debit/credit live
    // in accumulators only.
    assert!(
        created_ids.is_empty(),
        "Stage 13c BalanceTransfer must create no objects; got {:?}",
        created_ids,
    );
    assert!(
        mutated_ids.is_empty(),
        "Stage 13c BalanceTransfer must mutate no objects; got {:?}",
        mutated_ids,
    );
    assert!(
        deleted_ids.is_empty(),
        "Stage 13c BalanceTransfer must delete no objects; got {:?}",
        deleted_ids,
    );
}

#[tokio::test]
async fn test_effects_retrievable_after_execution() {
    // After executing a certificate, effects should be retrievable by digest
    let (sender, sender_key): (_, Ed25519KeyPair) = get_key_pair();
    let recipient = dbg_addr(1);

    let authority_state = TestAuthorityBuilder::new().build().await;
    crate::authority_test_utils::seed_balance_mode_funds(
        &authority_state,
        sender,
        10_000_000,
        10_000_000,
    );

    let data = crate::authority_test_utils::balance_transfer_data_legacy(
        recipient,
        sender,
        Some(1000),
    );
    let tx = to_sender_signed_transaction(data, &sender_key);

    let (cert, effects) = send_and_confirm_transaction(&authority_state, tx).await.unwrap();
    assert_eq!(*effects.status(), ExecutionStatus::Success);

    // Effects should be readable via notify_read_effects
    let read_effects = authority_state.notify_read_effects(*cert.digest()).await.unwrap();
    assert_eq!(
        effects.transaction_digest(),
        read_effects.transaction_digest(),
        "Read effects should match executed effects by transaction digest"
    );
}

// =============================================================================
// Object version tracking
// =============================================================================

#[tokio::test]
async fn test_object_version_increments_after_mutation() {
    // Object version (lamport timestamp) should increment after each
    // mutation. Stage 13c: BalanceTransfer mutates no objects, so
    // we exercise this invariant with AddStake — which mutates the
    // shared SystemState — and assert that the SystemState's
    // version strictly increases.
    let (sender, sender_key): (_, Ed25519KeyPair) = get_key_pair();

    let authority_state = TestAuthorityBuilder::new().build().await;
    crate::authority_test_utils::seed_balance_mode_funds(
        &authority_state,
        sender,
        50_000_000,
        10_000_000,
    );

    let initial_system_state_version = authority_state
        .get_object(&types::SYSTEM_STATE_OBJECT_ID)
        .await
        .expect("SystemState exists")
        .version();

    let validator_address = authority_state
        .get_system_state_object_for_testing()
        .unwrap()
        .validators()
        .validators[0]
        .metadata
        .soma_address;

    let data = TransactionData::new(
        TransactionKind::AddStake { validator: validator_address, amount: 1_000_000 },
        sender,
        vec![],
    );
    let tx = to_sender_signed_transaction(data, &sender_key);
    let (_, effects) =
        send_and_confirm_transaction_(&authority_state, None, tx, true).await.unwrap();
    assert_eq!(*effects.status(), ExecutionStatus::Success);

    let after_system_state_version = authority_state
        .get_object(&types::SYSTEM_STATE_OBJECT_ID)
        .await
        .expect("SystemState exists")
        .version();
    assert!(
        after_system_state_version > initial_system_state_version,
        "SystemState version must strictly increase after AddStake mutation: \
         was {:?}, now {:?}",
        initial_system_state_version,
        after_system_state_version,
    );
}

// =============================================================================
// Signature validation tests
// =============================================================================

#[tokio::test]
async fn test_bad_signature_rejected() {
    // Transaction signed with wrong key should be rejected
    let (sender, _sender_key): (_, Ed25519KeyPair) = get_key_pair();
    let (_, wrong_key): (_, Ed25519KeyPair) = get_key_pair();

    let data = crate::authority_test_utils::balance_transfer_data_legacy(
        dbg_addr(1),
        sender,
        Some(1000),
    );
    // Sign with wrong key
    let tx = to_sender_signed_transaction(data, &wrong_key);

    // Signature verification should fail
    let result = verify_sender_signed_data_message_signatures(tx.data());
    assert!(result.is_err(), "Wrong key should fail signature verification");
}

#[tokio::test]
async fn test_no_signature_rejected() {
    // Transaction with missing signature should be rejected
    use types::crypto::GenericSignature;

    let (sender, _sender_key): (_, Ed25519KeyPair) = get_key_pair();

    let data = crate::authority_test_utils::balance_transfer_data_legacy(
        dbg_addr(1),
        sender,
        Some(1000),
    );

    // Create a transaction with no signatures
    let signed_data = SenderSignedData::new(data, vec![] as Vec<GenericSignature>);
    let result = verify_sender_signed_data_message_signatures(&signed_data);
    assert!(result.is_err(), "No signature should fail verification");
    let err = result.unwrap_err();
    assert!(
        err.to_string().contains("signer signatures"),
        "Error should mention signature count mismatch, got: {}",
        err
    );
}

// =============================================================================
// System state tests
// =============================================================================

#[tokio::test]
async fn test_system_state_exists_after_genesis() {
    let authority_state = TestAuthorityBuilder::new().build().await;

    let system_state = authority_state.get_system_state_object_for_testing();
    assert!(system_state.is_ok(), "System state should exist after genesis");

    let state = system_state.unwrap();
    assert_eq!(state.epoch(), 0, "Initial epoch should be 0");
    assert!(!state.validators().validators.is_empty(), "Should have validators");
}

#[tokio::test]
async fn test_system_state_has_emission_pool() {
    let authority_state = TestAuthorityBuilder::new().build().await;
    let state = authority_state.get_system_state_object_for_testing().unwrap();

    assert!(
        state.emission_pool().balance > 0,
        "Emission pool should have a positive balance after genesis"
    );
}

#[tokio::test]
async fn test_system_state_has_protocol_version() {
    let authority_state = TestAuthorityBuilder::new().build().await;
    let state = authority_state.get_system_state_object_for_testing().unwrap();

    assert!(state.protocol_version() >= 1, "Protocol version should be at least 1");
}

// =============================================================================
// TransferObjects tests
// =============================================================================

#[tokio::test]
async fn test_transfer_objects_success() {
    // TransferObjects transfers ownership of arbitrary objects
    let (sender, sender_key): (_, Ed25519KeyPair) = get_key_pair();
    let recipient = dbg_addr(1);
    let object_id = ObjectID::random();

    let obj = Object::with_id_owner_for_testing(object_id, sender);

    let authority_state = TestAuthorityBuilder::new().build().await;
    authority_state.insert_genesis_object(obj.clone()).await;
    crate::authority_test_utils::seed_balance_mode_funds(
        &authority_state,
        sender,
        0,
        10_000_000,
    );

    let data = TransactionData::new(
        TransactionKind::TransferObjects {
            objects: vec![obj.compute_object_reference()],
            recipient,
        },
        sender,
        vec![],
    );
    let tx = to_sender_signed_transaction(data, &sender_key);
    let result = send_and_confirm_transaction(&authority_state, tx).await;
    assert!(result.is_ok(), "TransferObjects should succeed");

    let (_, effects) = result.unwrap();
    assert_eq!(*effects.status(), ExecutionStatus::Success);

    // Object should now be owned by recipient
    let transferred = authority_state.get_object(&object_id).await.unwrap();
    assert_eq!(transferred.owner, Owner::AddressOwner(recipient));
}

#[tokio::test]
async fn test_transfer_objects_wrong_owner() {
    // Cannot transfer objects you don't own
    let (sender, sender_key): (_, Ed25519KeyPair) = get_key_pair();
    let other_owner = dbg_addr(99);
    let recipient = dbg_addr(1);
    let object_id = ObjectID::random();

    let obj = Object::with_id_owner_for_testing(object_id, other_owner); // owned by someone else

    let authority_state = TestAuthorityBuilder::new().build().await;
    authority_state.insert_genesis_object(obj.clone()).await;
    crate::authority_test_utils::seed_balance_mode_funds(
        &authority_state,
        sender,
        0,
        10_000_000,
    );

    let data = TransactionData::new(
        TransactionKind::TransferObjects {
            objects: vec![obj.compute_object_reference()],
            recipient,
        },
        sender,
        vec![],
    );
    let tx = to_sender_signed_transaction(data, &sender_key);
    let result = send_and_confirm_transaction(&authority_state, tx).await;

    // Should fail - sender doesn't own the object
    match result {
        Ok((_, effects)) => {
            assert!(!effects.status().is_ok(), "Should fail: wrong owner");
        }
        Err(_) => {
            // Also acceptable - may fail at input validation
        }
    }
}

#[tokio::test]
async fn test_transfer_multiple_objects() {
    // Transfer multiple objects in a single TransferObjects
    let (sender, sender_key): (_, Ed25519KeyPair) = get_key_pair();
    let recipient = dbg_addr(1);
    let obj_id1 = ObjectID::random();
    let obj_id2 = ObjectID::random();

    let obj1 = Object::with_id_owner_for_testing(obj_id1, sender);
    let obj2 = Object::with_id_owner_for_testing(obj_id2, sender);

    let authority_state = TestAuthorityBuilder::new().build().await;
    authority_state.insert_genesis_object(obj1.clone()).await;
    authority_state.insert_genesis_object(obj2.clone()).await;
    crate::authority_test_utils::seed_balance_mode_funds(
        &authority_state,
        sender,
        0,
        10_000_000,
    );

    let data = TransactionData::new(
        TransactionKind::TransferObjects {
            objects: vec![obj1.compute_object_reference(), obj2.compute_object_reference()],
            recipient,
        },
        sender,
        vec![],
    );
    let tx = to_sender_signed_transaction(data, &sender_key);
    let result = send_and_confirm_transaction(&authority_state, tx).await;
    assert!(result.is_ok(), "Multi-object transfer should succeed");

    let (_, effects) = result.unwrap();
    assert_eq!(*effects.status(), ExecutionStatus::Success);

    // Both objects should now be owned by recipient
    let t1 = authority_state.get_object(&obj_id1).await.unwrap();
    let t2 = authority_state.get_object(&obj_id2).await.unwrap();
    assert_eq!(t1.owner, Owner::AddressOwner(recipient));
    assert_eq!(t2.owner, Owner::AddressOwner(recipient));
}

// =============================================================================
// Shared object tests
// =============================================================================

#[tokio::test]
async fn test_staking_creates_shared_object_mutation() {
    // AddStake mutates SystemState (shared object) — verify shared
    // object versioning works. Stage 13c: AddStake is balance-mode
    // for both stake (SOMA) and gas (USDC).
    let (sender, sender_key): (_, Ed25519KeyPair) = get_key_pair();

    let authority_state = TestAuthorityBuilder::new().build().await;
    crate::authority_test_utils::seed_balance_mode_funds(
        &authority_state,
        sender,
        50_000_000,
        10_000_000,
    );

    let system_state = authority_state.get_system_state_object_for_testing().unwrap();
    let validator_address = system_state.validators().validators[0].metadata.soma_address;

    let data = TransactionData::new(
        TransactionKind::AddStake {
            validator: validator_address,
            amount: 1_000_000,
        },
        sender,
        vec![],
    );
    let tx = to_sender_signed_transaction(data, &sender_key);
    let result = send_and_confirm_transaction_(&authority_state, None, tx, true).await;
    assert!(result.is_ok(), "AddStake should succeed");

    let (_, effects) = result.unwrap();
    assert_eq!(*effects.status(), ExecutionStatus::Success);

    // SystemState should be in the mutated set (shared object was modified)
    let mutated_ids: Vec<ObjectID> = effects.mutated().iter().map(|m| m.0.0).collect();
    assert!(
        mutated_ids.iter().any(|id| *id == SYSTEM_STATE_OBJECT_ID),
        "SystemState should be mutated during staking"
    );
}
