// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use fastcrypto::ed25519::Ed25519KeyPair;
use types::{
    base::dbg_addr,
    crypto::{SomaKeyPair, get_key_pair},
    effects::{ExecutionFailureStatus, ExecutionStatus, TransactionEffectsAPI},
    object::{Object, ObjectID, Owner},
    transaction::TransactionData,
    unit_tests::utils::to_sender_signed_transaction,
};

use crate::{
    authority_test_utils::send_and_confirm_transaction,
    test_authority_builder::TestAuthorityBuilder,
};

// =============================================================================
// Multiple sequential transfers
// =============================================================================

#[tokio::test]
async fn test_multiple_sequential_transfers() {
    // Execute 3 TransferCoin transactions in sequence, each using updated object refs.
    let (sender, sender_key): (_, Ed25519KeyPair) = get_key_pair();
    let coin_id = ObjectID::random();
    let coin = Object::with_id_owner_coin_for_testing(coin_id, sender, 50_000_000);

    let authority_state = TestAuthorityBuilder::new().build().await;
    authority_state.insert_genesis_object(coin.clone()).await;

    let recipients = [dbg_addr(1), dbg_addr(2), dbg_addr(3)];

    let mut current_coin_ref = coin.compute_object_reference();

    for (i, recipient) in recipients.iter().enumerate() {
        let data =
            TransactionData::new_transfer_coin(*recipient, sender, Some(1000), current_coin_ref);
        let tx = to_sender_signed_transaction(data, &sender_key);
        let (_, effects) = send_and_confirm_transaction(&authority_state, tx)
            .await
            .unwrap_or_else(|e| panic!("Transfer {} should succeed: {:?}", i + 1, e));
        assert_eq!(
            *effects.status(),
            ExecutionStatus::Success,
            "Transfer {} should succeed",
            i + 1
        );

        // Update the coin ref for the next iteration (gas coin was mutated)
        let coin_obj = authority_state.get_object(&coin_id).await.unwrap();
        current_coin_ref = coin_obj.compute_object_reference();
    }

    // Verify final coin balance: original - 3 * 1000 - total fees
    let final_coin = authority_state.get_object(&coin_id).await.unwrap();
    let final_balance = final_coin.as_coin().unwrap();
    assert!(
        final_balance < 50_000_000 - 3000,
        "Balance should reflect 3 transfers plus fees: got {}",
        final_balance
    );
    assert!(final_balance > 0, "Balance should still be positive after 3 small transfers");
}

// =============================================================================
// Failed execution reverts non-gas changes
// =============================================================================

#[tokio::test]
async fn test_failed_execution_reverts_non_gas() {
    // Execute a transaction that fails during execution.
    // The gas coin should still be mutated (fee deducted), but the
    // transfer target should not be created.
    let (sender, sender_key): (_, Ed25519KeyPair) = get_key_pair();
    let coin_id = ObjectID::random();
    // Balance just enough for base fee but not for transfer + operation fee + value fee
    let coin = Object::with_id_owner_coin_for_testing(coin_id, sender, 5000);

    let authority_state = TestAuthorityBuilder::new().build().await;
    authority_state.insert_genesis_object(coin.clone()).await;

    let recipient = dbg_addr(1);
    // Try to transfer more than available after fees
    let data = TransactionData::new_transfer_coin(
        recipient,
        sender,
        Some(4500),
        coin.compute_object_reference(),
    );
    let tx = to_sender_signed_transaction(data, &sender_key);
    let result = send_and_confirm_transaction(&authority_state, tx).await;

    let (_, effects) = result.unwrap();
    let effects = effects.into_data();

    // Transaction should fail
    assert!(!effects.status().is_ok(), "Transaction should fail: transfer + fees > balance");

    // Gas coin should still exist (fee was deducted from it)
    let gas_obj = authority_state.get_object(&coin_id).await;
    assert!(gas_obj.is_some(), "Gas coin should still exist after failed tx");

    // No new objects should be created for recipient
    assert!(effects.created().is_empty(), "Failed execution should not create objects");

    // Gas should have been deducted
    let fee = effects.transaction_fee();
    assert!(fee.total_fee > 0, "Some fee should be charged even on failure");
}

// =============================================================================
// Effects accumulate correctly across transactions
// =============================================================================

#[tokio::test]
async fn test_effects_accumulate_correctly() {
    // Execute multiple transactions and verify each set of effects is stored
    // and retrievable independently.
    let (sender, sender_key): (_, Ed25519KeyPair) = get_key_pair();
    let coin_id = ObjectID::random();
    let coin = Object::with_id_owner_coin_for_testing(coin_id, sender, 50_000_000);

    let authority_state = TestAuthorityBuilder::new().build().await;
    authority_state.insert_genesis_object(coin.clone()).await;

    let mut digests = Vec::new();
    let mut current_coin_ref = coin.compute_object_reference();

    // Execute 3 transactions
    for i in 0..3 {
        let recipient = dbg_addr((i + 1) as u8);
        let data =
            TransactionData::new_transfer_coin(recipient, sender, Some(1000), current_coin_ref);
        let tx = to_sender_signed_transaction(data, &sender_key);
        let tx_digest = *tx.digest();

        let (_, effects) = send_and_confirm_transaction(&authority_state, tx)
            .await
            .unwrap_or_else(|e| panic!("Transaction {} should succeed: {:?}", i, e));
        assert_eq!(*effects.status(), ExecutionStatus::Success);

        digests.push(tx_digest);

        // Update coin ref
        let coin_obj = authority_state.get_object(&coin_id).await.unwrap();
        current_coin_ref = coin_obj.compute_object_reference();
    }

    // Verify each transaction's effects are retrievable
    for (i, digest) in digests.iter().enumerate() {
        let effects = authority_state.notify_read_effects(*digest).await;
        assert!(effects.is_ok(), "Effects for transaction {} should be readable", i);
        let effects = effects.unwrap();
        assert_eq!(
            effects.transaction_digest(),
            digest,
            "Effects digest should match transaction {} digest",
            i
        );
    }

    // All three digests should be distinct
    assert_ne!(digests[0], digests[1]);
    assert_ne!(digests[1], digests[2]);
    assert_ne!(digests[0], digests[2]);
}

// =============================================================================
// Version tracking across sequential mutations
// =============================================================================

#[tokio::test]
async fn test_version_monotonically_increases() {
    // Object version should increase with each mutation.
    let (sender, sender_key): (_, Ed25519KeyPair) = get_key_pair();
    let coin_id = ObjectID::random();
    let coin = Object::with_id_owner_coin_for_testing(coin_id, sender, 50_000_000);

    let authority_state = TestAuthorityBuilder::new().build().await;
    authority_state.insert_genesis_object(coin.clone()).await;

    let mut prev_version = authority_state.get_object(&coin_id).await.unwrap().version();

    for i in 0..3 {
        let coin_obj = authority_state.get_object(&coin_id).await.unwrap();
        let data = TransactionData::new_transfer_coin(
            dbg_addr((i + 1) as u8),
            sender,
            Some(100),
            coin_obj.compute_object_reference(),
        );
        let tx = to_sender_signed_transaction(data, &sender_key);
        let (_, effects) = send_and_confirm_transaction(&authority_state, tx).await.unwrap();
        assert_eq!(*effects.status(), ExecutionStatus::Success);

        let new_version = authority_state.get_object(&coin_id).await.unwrap().version();
        assert!(
            new_version > prev_version,
            "Version should increase: {:?} vs {:?} at iteration {}",
            new_version,
            prev_version,
            i
        );
        prev_version = new_version;
    }
}
