// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use fastcrypto::ed25519::Ed25519KeyPair;
use futures::future::join_all;
use tracing::info;
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
// Gas fee structure tests
// =============================================================================

/// Default fee parameters from protocol config v1:
/// - base_fee = 1000
/// - write_object_fee = 300 per written object
/// - value_fee_bps = 10 (0.1%)
const BASE_FEE: u64 = 1000;
const WRITE_FEE: u64 = 300;
const VALUE_FEE_BPS: u64 = 10;
const BPS_DENOMINATOR: u64 = 10000;

// =============================================================================
// Base fee deduction tests
// =============================================================================

#[tokio::test]
async fn test_base_fee_deducted_on_success() {
    // A simple pay-coins transaction should always deduct at least the base fee.
    let (sender, sender_key): (_, Ed25519KeyPair) = get_key_pair();
    let coin = Object::with_id_owner_coin_for_testing(ObjectID::random(), sender, 10_000_000);
    let coin_ref = coin.compute_object_reference();

    let recipient = dbg_addr(1);
    let transfer_amount = 1000u64;

    let res = execute_transfer_coin(
        coin,
        recipient,
        Some(transfer_amount),
        sender,
        SomaKeyPair::Ed25519(sender_key),
    )
    .await;

    let effects = res.txn_result.unwrap().into_data();
    assert_eq!(*effects.status(), ExecutionStatus::Success);

    let fee = effects.transaction_fee();
    // Base fee should always be present
    assert!(fee.base_fee >= BASE_FEE, "Base fee {} < expected {}", fee.base_fee, BASE_FEE);
    assert!(fee.total_fee > 0, "Total fee should be non-zero");

    // Verify gas object balance reflects deductions
    let gas_obj = res.authority_state.get_object(&coin_ref.0).await.unwrap();
    let remaining = gas_obj.as_coin().unwrap();
    assert_eq!(remaining, 10_000_000 - transfer_amount - fee.total_fee);
}

#[tokio::test]
async fn test_base_fee_deducted_on_insufficient_balance_for_transfer() {
    // If the user has enough for base fee but not enough for the transfer amount + fees,
    // execution should fail but base fee should still be charged.
    let (sender, sender_key): (_, Ed25519KeyPair) = get_key_pair();
    let coin_id = ObjectID::random();
    // Just barely enough for base fee + operation fee, but not enough for large transfer
    let coin = Object::with_id_owner_coin_for_testing(coin_id, sender, 5000);

    let recipient = dbg_addr(1);
    // Try to transfer more than available (after fees)
    let transfer_amount = 4500u64;

    let res = execute_transfer_coin(
        coin,
        recipient,
        Some(transfer_amount),
        sender,
        SomaKeyPair::Ed25519(sender_key),
    )
    .await;

    let effects = res.txn_result.unwrap().into_data();
    // Should fail because transfer + fees > balance
    assert!(!effects.status().is_ok(), "Should fail: transfer amount + fees > balance");

    // Base fee should still be charged even on failure
    let fee = effects.transaction_fee();
    assert!(fee.base_fee >= BASE_FEE, "Base fee should still be charged on execution failure");
}

#[tokio::test]
async fn test_insufficient_gas_below_base_fee() {
    // With balance below the base fee, prepare_gas should fail
    let (sender, sender_key): (_, Ed25519KeyPair) = get_key_pair();
    let coin_id = ObjectID::random();
    let coin = Object::with_id_owner_coin_for_testing(coin_id, sender, 500); // < 1000 base fee

    let recipient = dbg_addr(1);
    let res =
        execute_transfer_coin(coin, recipient, Some(100), sender, SomaKeyPair::Ed25519(sender_key))
            .await;

    let effects = res.txn_result.unwrap().into_data();
    assert_eq!(
        *effects.status(),
        ExecutionStatus::Failure { error: ExecutionFailureStatus::InsufficientGas },
    );

    // Partial fee should be deducted (whatever was available)
    let fee = effects.transaction_fee();
    assert!(fee.total_fee <= 500, "Should only charge up to balance");
}

#[tokio::test]
async fn test_zero_balance_coin_gas() {
    // A coin with 0 balance should fail with InsufficientGas
    let (sender, sender_key): (_, Ed25519KeyPair) = get_key_pair();
    let coin = Object::with_id_owner_coin_for_testing(ObjectID::random(), sender, 0);

    let recipient = dbg_addr(1);
    let res =
        execute_transfer_coin(coin, recipient, Some(0), sender, SomaKeyPair::Ed25519(sender_key))
            .await;

    let effects = res.txn_result.unwrap().into_data();
    assert_eq!(
        *effects.status(),
        ExecutionStatus::Failure { error: ExecutionFailureStatus::InsufficientGas },
    );
}

// =============================================================================
// Gas smashing tests
// =============================================================================

#[tokio::test]
async fn test_gas_smashing_merges_balances() {
    // Using PayCoins with multiple input coins should smash them into one
    let (sender, sender_key): (_, Ed25519KeyPair) = get_key_pair();
    let id1 = ObjectID::random();
    let id2 = ObjectID::random();
    let id3 = ObjectID::random();
    let coin1 = Object::with_id_owner_coin_for_testing(id1, sender, 5_000_000);
    let coin2 = Object::with_id_owner_coin_for_testing(id2, sender, 3_000);
    let coin3 = Object::with_id_owner_coin_for_testing(id3, sender, 7_000);

    let recipient = dbg_addr(1);

    let res = execute_pay_coin(
        vec![coin1, coin2, coin3],
        vec![recipient],
        Some(vec![100]),
        sender,
        SomaKeyPair::Ed25519(sender_key),
    )
    .await;

    let effects = res.txn_result.unwrap().into_data();
    assert_eq!(*effects.status(), ExecutionStatus::Success);

    // Gas coins 2 and 3 should be deleted (smashed into coin 1)
    let deleted_ids: Vec<ObjectID> = effects.deleted().iter().map(|d| d.0).collect();
    assert!(deleted_ids.contains(&id2), "Second gas coin should be deleted");
    assert!(deleted_ids.contains(&id3), "Third gas coin should be deleted");

    // Primary gas coin should have the merged balance minus fees and payment
    let gas_used = effects.transaction_fee().total_fee;
    let gas_obj = res.authority_state.get_object(&id1).await.unwrap();
    let total_original = 5_000_000 + 3_000 + 7_000;
    assert_eq!(gas_obj.as_coin().unwrap(), total_original - 100 - gas_used);
}

#[tokio::test]
async fn test_gas_smashing_single_coin_no_deletion() {
    // With a single gas coin, no coins should be deleted from smashing
    let (sender, sender_key): (_, Ed25519KeyPair) = get_key_pair();
    let id = ObjectID::random();
    let coin = Object::with_id_owner_coin_for_testing(id, sender, 5_000_000);

    let recipient = dbg_addr(1);

    let res = execute_pay_coin(
        vec![coin],
        vec![recipient],
        Some(vec![100]),
        sender,
        SomaKeyPair::Ed25519(sender_key),
    )
    .await;

    let effects = res.txn_result.unwrap().into_data();
    assert_eq!(*effects.status(), ExecutionStatus::Success);

    // No coins should be deleted (single coin, not pay-all)
    assert_eq!(effects.deleted().len(), 0, "No coins should be deleted for single-coin pay");

    // Gas coin should be in mutated
    assert!(
        effects.mutated().iter().any(|(r, _)| r.0 == id),
        "Gas coin should be mutated, not deleted"
    );
}

// =============================================================================
// Operation fee tests
// =============================================================================

#[tokio::test]
async fn test_operation_fee_scales_with_created_objects() {
    // PayCoins with more recipients should have higher operation fees
    let (sender, sender_key1): (_, Ed25519KeyPair) = get_key_pair();
    let coin1 = Object::with_id_owner_coin_for_testing(ObjectID::random(), sender, 50_000_000);

    // Pay to 1 recipient
    let res1 = execute_pay_coin(
        vec![coin1],
        vec![dbg_addr(1)],
        Some(vec![100]),
        sender,
        SomaKeyPair::Ed25519(sender_key1),
    )
    .await;
    let effects1 = res1.txn_result.unwrap().into_data();
    assert_eq!(*effects1.status(), ExecutionStatus::Success);
    let fee1 = effects1.transaction_fee().total_fee;

    // Pay to 3 recipients (more created objects → higher write fee)
    let (sender2, sender_key2): (_, Ed25519KeyPair) = get_key_pair();
    let coin2 = Object::with_id_owner_coin_for_testing(ObjectID::random(), sender2, 50_000_000);

    let res2 = execute_pay_coin(
        vec![coin2],
        vec![dbg_addr(1), dbg_addr(2), dbg_addr(3)],
        Some(vec![100, 100, 100]),
        sender2,
        SomaKeyPair::Ed25519(sender_key2),
    )
    .await;
    let effects2 = res2.txn_result.unwrap().into_data();
    assert_eq!(*effects2.status(), ExecutionStatus::Success);
    let fee2 = effects2.transaction_fee().total_fee;

    // More recipients → more created objects → higher total fee
    assert!(
        fee2 > fee1,
        "Fee for 3 recipients ({}) should be greater than 1 recipient ({})",
        fee2,
        fee1
    );
}

// =============================================================================
// Value fee tests
// =============================================================================

#[tokio::test]
async fn test_value_fee_proportional_to_amount() {
    // Larger transfer amounts should incur higher value fees
    let (sender1, key1): (_, Ed25519KeyPair) = get_key_pair();
    let coin1 = Object::with_id_owner_coin_for_testing(ObjectID::random(), sender1, 50_000_000);

    // Small transfer: 1000
    let res1 = execute_pay_coin(
        vec![coin1],
        vec![dbg_addr(1)],
        Some(vec![1000]),
        sender1,
        SomaKeyPair::Ed25519(key1),
    )
    .await;
    let effects1 = res1.txn_result.unwrap().into_data();
    assert_eq!(*effects1.status(), ExecutionStatus::Success);
    let fee1 = effects1.transaction_fee();

    // Large transfer: 10_000_000
    let (sender2, key2): (_, Ed25519KeyPair) = get_key_pair();
    let coin2 = Object::with_id_owner_coin_for_testing(ObjectID::random(), sender2, 50_000_000);

    let res2 = execute_pay_coin(
        vec![coin2],
        vec![dbg_addr(1)],
        Some(vec![10_000_000]),
        sender2,
        SomaKeyPair::Ed25519(key2),
    )
    .await;
    let effects2 = res2.txn_result.unwrap().into_data();
    assert_eq!(*effects2.status(), ExecutionStatus::Success);
    let fee2 = effects2.transaction_fee();

    // Value fee should be proportional
    assert!(
        fee2.value_fee > fee1.value_fee,
        "Value fee for 10M ({}) should be greater than for 1K ({})",
        fee2.value_fee,
        fee1.value_fee
    );

    // Check approximate correctness: value_fee ≈ amount * 10 / 10000
    let expected_vfee_small = (1000 * VALUE_FEE_BPS) / BPS_DENOMINATOR;
    let expected_vfee_large = (10_000_000 * VALUE_FEE_BPS) / BPS_DENOMINATOR;
    assert_eq!(fee1.value_fee, expected_vfee_small, "Small transfer value fee mismatch");
    assert_eq!(fee2.value_fee, expected_vfee_large, "Large transfer value fee mismatch");
}

#[tokio::test]
async fn test_value_fee_zero_for_zero_transfer() {
    // A zero-amount transfer should have zero value fee
    // (though operation fee and base fee still apply)
    let (sender, key): (_, Ed25519KeyPair) = get_key_pair();
    let coin_id = ObjectID::random();
    let coin = Object::with_id_owner_coin_for_testing(coin_id, sender, 50_000_000);

    // Transfer 0 amount
    let res = execute_pay_coin(
        vec![coin],
        vec![dbg_addr(1)],
        Some(vec![0]),
        sender,
        SomaKeyPair::Ed25519(key),
    )
    .await;
    let effects = res.txn_result.unwrap().into_data();
    assert_eq!(*effects.status(), ExecutionStatus::Success);
    let fee = effects.transaction_fee();
    assert_eq!(fee.value_fee, 0, "Zero transfer should have zero value fee");
    assert!(fee.base_fee > 0, "Base fee should still be charged");
}

// =============================================================================
// Gas coin consumed to zero tests
// =============================================================================

#[tokio::test]
async fn test_gas_coin_deleted_when_balance_reaches_zero() {
    // When a pay-all operation consumes the gas coin to exactly 0, it should be deleted
    let (sender, key): (_, Ed25519KeyPair) = get_key_pair();
    let coin_id = ObjectID::random();
    let coin = Object::with_id_owner_coin_for_testing(coin_id, sender, 5_000_000);

    let recipient = dbg_addr(1);

    // Pay-all: None amounts means transfer everything
    let res = execute_pay_coin(
        vec![coin],
        vec![recipient],
        None, // pay-all
        sender,
        SomaKeyPair::Ed25519(key),
    )
    .await;

    let effects = res.txn_result.unwrap().into_data();
    assert_eq!(*effects.status(), ExecutionStatus::Success);

    // The original coin should be deleted (consumed fully)
    let deleted_ids: Vec<ObjectID> = effects.deleted().iter().map(|d| d.0).collect();
    assert!(
        deleted_ids.contains(&coin_id),
        "Gas coin should be deleted when fully consumed in pay-all"
    );

    // Recipient should have a new coin with balance = original - total_fee
    assert_eq!(effects.created().len(), 1);
    let created_id = effects.created()[0].0.0;
    let created_obj = res.authority_state.get_object(&created_id).await.unwrap();
    let gas_used = effects.transaction_fee().total_fee;
    assert_eq!(created_obj.as_coin().unwrap(), 5_000_000 - gas_used);
}

// =============================================================================
// Fee breakdown verification
// =============================================================================

#[tokio::test]
async fn test_fee_components_sum_to_total() {
    let (sender, key): (_, Ed25519KeyPair) = get_key_pair();
    let coin = Object::with_id_owner_coin_for_testing(ObjectID::random(), sender, 50_000_000);

    let res = execute_pay_coin(
        vec![coin],
        vec![dbg_addr(1), dbg_addr(2)],
        Some(vec![1000, 2000]),
        sender,
        SomaKeyPair::Ed25519(key),
    )
    .await;

    let effects = res.txn_result.unwrap().into_data();
    assert_eq!(*effects.status(), ExecutionStatus::Success);

    let fee = effects.transaction_fee();
    assert_eq!(
        fee.total_fee,
        fee.base_fee + fee.operation_fee + fee.value_fee,
        "Fee components should sum to total: {} != {} + {} + {}",
        fee.total_fee,
        fee.base_fee,
        fee.operation_fee,
        fee.value_fee
    );

    // Verify individual components
    assert_eq!(fee.base_fee, BASE_FEE, "Base fee mismatch");

    // Value fee: (1000 + 2000) * 10 / 10000 = 0 (integer division)
    let expected_value_fee = (3000u64 * VALUE_FEE_BPS) / BPS_DENOMINATOR;
    assert_eq!(fee.value_fee, expected_value_fee, "Value fee mismatch");
}

#[tokio::test]
async fn test_fee_components_large_transfer() {
    // Test with a large enough transfer to see all fee components
    let (sender, key): (_, Ed25519KeyPair) = get_key_pair();
    let coin = Object::with_id_owner_coin_for_testing(ObjectID::random(), sender, 100_000_000);

    let transfer_amount = 50_000_000u64;
    let res = execute_pay_coin(
        vec![coin],
        vec![dbg_addr(1)],
        Some(vec![transfer_amount]),
        sender,
        SomaKeyPair::Ed25519(key),
    )
    .await;

    let effects = res.txn_result.unwrap().into_data();
    assert_eq!(*effects.status(), ExecutionStatus::Success);

    let fee = effects.transaction_fee();
    // Value fee: 50_000_000 * 10 / 10_000 = 50_000
    let expected_value_fee = (transfer_amount * VALUE_FEE_BPS) / BPS_DENOMINATOR;
    assert_eq!(fee.value_fee, expected_value_fee, "Large transfer value fee mismatch");
    assert_eq!(fee.base_fee, BASE_FEE);
    assert!(fee.operation_fee > 0, "Operation fee should be non-zero");
    assert_eq!(fee.total_fee, fee.base_fee + fee.operation_fee + fee.value_fee);
}

// =============================================================================
// Edge case: exact balance for fees
// =============================================================================

#[tokio::test]
async fn test_balance_barely_above_base_fee() {
    // Balance just above base_fee: prepare_gas succeeds but execution may fail
    // because there's not enough for operation fees after base fee deduction
    let (sender, key): (_, Ed25519KeyPair) = get_key_pair();
    // 1001 = base_fee + 1, not enough for operation_fee (300 * num_objects)
    let coin = Object::with_id_owner_coin_for_testing(ObjectID::random(), sender, BASE_FEE + 1);

    let res =
        execute_transfer_coin(coin, dbg_addr(1), Some(0), sender, SomaKeyPair::Ed25519(key)).await;

    let effects = res.txn_result.unwrap().into_data();
    // After base_fee deduction, only 1 left. Operation fee for 2 objects = 600.
    // So remaining fee deduction fails.
    assert!(
        !effects.status().is_ok(),
        "Should fail: after base fee there's not enough for operation fee"
    );
}

// =============================================================================
// Helpers
// =============================================================================

struct TransactionResult {
    authority_state: Arc<AuthorityState>,
    txn_result: Result<SignedTransactionEffects, SomaError>,
}

async fn execute_transfer_coin(
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

async fn execute_pay_coin(
    input_coin_objects: Vec<Object>,
    recipients: Vec<SomaAddress>,
    amounts: Option<Vec<u64>>,
    sender: SomaAddress,
    sender_key: SomaKeyPair,
) -> TransactionResult {
    let authority_state = TestAuthorityBuilder::new().build().await;

    let input_coin_refs: Vec<ObjectRef> =
        input_coin_objects.iter().map(|coin_obj| coin_obj.compute_object_reference()).collect();
    let handles: Vec<_> = input_coin_objects
        .into_iter()
        .map(|obj| authority_state.insert_genesis_object(obj))
        .collect();
    join_all(handles).await;

    let data = TransactionData::new_pay_coins(input_coin_refs, amounts, recipients, sender);
    let tx = to_sender_signed_transaction(data, &sender_key);
    let txn_result =
        send_and_confirm_transaction(&authority_state, tx).await.map(|(_, effects)| effects);

    TransactionResult { authority_state, txn_result }
}
