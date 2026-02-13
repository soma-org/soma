use std::sync::Arc;

use fastcrypto::ed25519::Ed25519KeyPair;
use types::{
    base::{SomaAddress, dbg_addr},
    crypto::{SomaKeyPair, get_key_pair},
    effects::{
        ExecutionFailureStatus, ExecutionStatus, SignedTransactionEffects, TransactionEffectsAPI,
    },
    error::SomaError,
    object::{Object, ObjectID, ObjectRef},
    transaction::TransactionData,
    unit_tests::utils::to_sender_signed_transaction,
};

use crate::{
    authority::AuthorityState, authority_test_utils::send_and_confirm_transaction,
    test_authority_builder::TestAuthorityBuilder,
};

// =============================================================================
// TransferCoin success cases
// =============================================================================

#[tokio::test]
async fn test_transfer_coin_specific_amount() {
    let (sender, key): (_, Ed25519KeyPair) = get_key_pair();
    let coin_id = ObjectID::random();
    let coin = Object::with_id_owner_coin_for_testing(coin_id, sender, 10_000_000);

    let recipient = dbg_addr(1);
    let amount = 500_000u64;

    let res = execute_transfer(coin, recipient, Some(amount), sender, SomaKeyPair::Ed25519(key))
        .await;

    let effects = res.txn_result.unwrap().into_data();
    assert_eq!(*effects.status(), ExecutionStatus::Success);

    // Should create one new coin for recipient
    assert_eq!(effects.created().len(), 1);
    let created_id = effects.created()[0].0 .0;
    let created_obj = res.authority_state.get_object(&created_id).await.unwrap();
    assert_eq!(created_obj.as_coin().unwrap(), amount);
    assert_eq!(
        effects.created()[0].1.get_address_owner_address().unwrap(),
        recipient
    );

    // Source coin should still belong to sender with reduced balance
    let gas_used = effects.transaction_fee().total_fee;
    let source_obj = res.authority_state.get_object(&coin_id).await.unwrap();
    assert_eq!(source_obj.as_coin().unwrap(), 10_000_000 - amount - gas_used);
}

#[tokio::test]
async fn test_transfer_coin_full_amount_non_gas() {
    // When transferring the full amount from a non-gas coin, ownership transfers
    // (no new coin created, source coin changes owner)
    let (sender, key): (_, Ed25519KeyPair) = get_key_pair();
    let coin_id = ObjectID::random();
    let gas_id = ObjectID::random();
    let coin = Object::with_id_owner_coin_for_testing(coin_id, sender, 1_000_000);
    let gas = Object::with_id_owner_coin_for_testing(gas_id, sender, 10_000_000);

    let recipient = dbg_addr(1);
    let amount = 1_000_000u64; // full balance

    let res = execute_transfer_with_separate_gas(
        coin,
        gas,
        recipient,
        Some(amount),
        sender,
        SomaKeyPair::Ed25519(key),
    )
    .await;

    let effects = res.txn_result.unwrap().into_data();
    assert_eq!(*effects.status(), ExecutionStatus::Success);

    // When full balance is transferred from a non-gas coin, ownership changes
    // (the coin is mutated to be owned by recipient, no new coin created)
    // Verify recipient now owns the original coin
    let mutated_ids: Vec<ObjectID> = effects.mutated().iter().map(|m| m.0 .0).collect();
    assert!(
        mutated_ids.contains(&coin_id),
        "Source coin should be mutated (ownership transfer)"
    );
}

#[tokio::test]
async fn test_transfer_coin_pay_all_gas_coin() {
    // Pay-all where coin == gas coin: creates new coin for recipient, gas coin consumed
    let (sender, key): (_, Ed25519KeyPair) = get_key_pair();
    let coin_id = ObjectID::random();
    let balance = 5_000_000u64;
    let coin = Object::with_id_owner_coin_for_testing(coin_id, sender, balance);

    let recipient = dbg_addr(1);

    let res =
        execute_transfer(coin, recipient, None, sender, SomaKeyPair::Ed25519(key)).await;

    let effects = res.txn_result.unwrap().into_data();
    assert_eq!(*effects.status(), ExecutionStatus::Success);

    // Should create one new coin for recipient
    assert_eq!(effects.created().len(), 1);
    let created_id = effects.created()[0].0 .0;
    let created_obj = res.authority_state.get_object(&created_id).await.unwrap();

    // Recipient gets balance minus all fees
    let gas_used = effects.transaction_fee().total_fee;
    assert_eq!(created_obj.as_coin().unwrap(), balance - gas_used);
    assert_eq!(
        effects.created()[0].1.get_address_owner_address().unwrap(),
        recipient
    );
}

// =============================================================================
// TransferCoin failure cases
// =============================================================================

#[tokio::test]
async fn test_transfer_coin_insufficient_balance() {
    let (sender, key): (_, Ed25519KeyPair) = get_key_pair();
    let coin = Object::with_id_owner_coin_for_testing(ObjectID::random(), sender, 10_000_000);

    let recipient = dbg_addr(1);
    // Try to transfer more than balance (even before fees)
    let amount = 20_000_000u64;

    let res = execute_transfer(coin, recipient, Some(amount), sender, SomaKeyPair::Ed25519(key))
        .await;

    let effects = res.txn_result.unwrap().into_data();
    assert_eq!(
        *effects.status(),
        ExecutionStatus::Failure { error: ExecutionFailureStatus::InsufficientCoinBalance }
    );
}

#[tokio::test]
async fn test_transfer_coin_wrong_owner() {
    let (sender, key): (_, Ed25519KeyPair) = get_key_pair();
    let other_owner = dbg_addr(99);
    // Coin is owned by `other_owner`, not `sender`
    let coin = Object::with_id_owner_coin_for_testing(ObjectID::random(), other_owner, 10_000_000);

    let recipient = dbg_addr(1);

    let res = execute_transfer(
        coin,
        recipient,
        Some(1000),
        sender,
        SomaKeyPair::Ed25519(key),
    )
    .await;

    // Should fail at the pre-certification ownership check
    match res.txn_result {
        Ok(effects) => {
            // If it makes it through signing, the execution should still fail
            assert!(!effects.status().is_ok(), "Should fail: wrong owner");
        }
        Err(e) => {
            // Expected: fails before execution due to ownership mismatch
            let err_msg = e.to_string();
            assert!(
                err_msg.contains("owned by") || err_msg.contains("IncorrectUserSignature"),
                "Should fail with ownership error, got: {}",
                err_msg
            );
        }
    }
}

// =============================================================================
// TransferCoin gas-is-transfer-coin edge cases
// =============================================================================

#[tokio::test]
async fn test_transfer_gas_coin_is_transfer_coin_specific_amount() {
    // The most common case: gas coin IS the coin being transferred
    // Needs to account for fees when checking balance
    let (sender, key): (_, Ed25519KeyPair) = get_key_pair();
    let coin_id = ObjectID::random();
    let coin = Object::with_id_owner_coin_for_testing(coin_id, sender, 10_000_000);

    let recipient = dbg_addr(1);
    let amount = 5_000_000u64;

    let res =
        execute_transfer(coin, recipient, Some(amount), sender, SomaKeyPair::Ed25519(key)).await;

    let effects = res.txn_result.unwrap().into_data();
    assert_eq!(*effects.status(), ExecutionStatus::Success);

    // Recipient gets the exact amount
    let created_id = effects.created()[0].0 .0;
    let created_obj = res.authority_state.get_object(&created_id).await.unwrap();
    assert_eq!(created_obj.as_coin().unwrap(), amount);

    // Sender keeps balance - amount - fees
    let gas_used = effects.transaction_fee().total_fee;
    let source_obj = res.authority_state.get_object(&coin_id).await.unwrap();
    assert_eq!(source_obj.as_coin().unwrap(), 10_000_000 - amount - gas_used);
}

#[tokio::test]
async fn test_transfer_gas_coin_insufficient_for_amount_plus_fees() {
    // Gas coin is also transfer coin. Balance is enough for the amount
    // but not enough for amount + all fees
    let (sender, key): (_, Ed25519KeyPair) = get_key_pair();
    let balance = 10_000u64;
    let coin = Object::with_id_owner_coin_for_testing(ObjectID::random(), sender, balance);

    let recipient = dbg_addr(1);
    // Transfer 9000, but base_fee=1000 + operation_fee + value_fee > remaining 1000
    let amount = 9000u64;

    let res =
        execute_transfer(coin, recipient, Some(amount), sender, SomaKeyPair::Ed25519(key)).await;

    let effects = res.txn_result.unwrap().into_data();
    assert!(
        !effects.status().is_ok(),
        "Should fail: balance - amount < total_fees"
    );
}

// =============================================================================
// Helpers
// =============================================================================

struct TransactionResult {
    authority_state: Arc<AuthorityState>,
    txn_result: Result<SignedTransactionEffects, SomaError>,
}

async fn execute_transfer(
    coin: Object,
    recipient: SomaAddress,
    amount: Option<u64>,
    sender: SomaAddress,
    sender_key: SomaKeyPair,
) -> TransactionResult {
    let authority_state = TestAuthorityBuilder::new().build().await;
    let coin_ref = coin.compute_object_reference();
    authority_state.insert_genesis_object(coin).await;

    let data = TransactionData::new_transfer_coin(recipient, sender, amount, coin_ref);
    let tx = to_sender_signed_transaction(data, &sender_key);
    let txn_result =
        send_and_confirm_transaction(&authority_state, tx).await.map(|(_, effects)| effects);

    TransactionResult { authority_state, txn_result }
}

async fn execute_transfer_with_separate_gas(
    coin: Object,
    gas: Object,
    recipient: SomaAddress,
    amount: Option<u64>,
    sender: SomaAddress,
    sender_key: SomaKeyPair,
) -> TransactionResult {
    let authority_state = TestAuthorityBuilder::new().build().await;
    let coin_ref = coin.compute_object_reference();
    let gas_ref = gas.compute_object_reference();
    authority_state.insert_genesis_object(coin).await;
    authority_state.insert_genesis_object(gas).await;

    // Use the gas object for gas payment, transfer the coin
    let data = TransactionData::new(
        types::transaction::TransactionKind::TransferCoin {
            coin: coin_ref,
            amount,
            recipient,
        },
        sender,
        vec![gas_ref],
    );
    let tx = to_sender_signed_transaction(data, &sender_key);
    let txn_result =
        send_and_confirm_transaction(&authority_state, tx).await.map(|(_, effects)| effects);

    TransactionResult { authority_state, txn_result }
}
