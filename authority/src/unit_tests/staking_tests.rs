// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use fastcrypto::ed25519::Ed25519KeyPair;
use types::{
    base::SomaAddress,
    crypto::{SomaKeyPair, get_key_pair},
    effects::{
        ExecutionFailureStatus, ExecutionStatus, SignedTransactionEffects, TransactionEffectsAPI,
    },
    error::SomaError,
    object::{Object, ObjectID, ObjectRef},
    transaction::{TransactionData, TransactionKind},
    unit_tests::utils::to_sender_signed_transaction,
};

use crate::{
    authority::AuthorityState, authority_test_utils::send_and_confirm_transaction_,
    test_authority_builder::TestAuthorityBuilder,
};

// Default fees from protocol config v1
const BASE_FEE: u64 = 1000;
const WRITE_FEE: u64 = 300;
const VALUE_FEE_BPS: u64 = 10;
const BPS_DENOMINATOR: u64 = 10000;
// Staking gets half the normal value fee
const STAKING_VALUE_FEE_BPS: u64 = VALUE_FEE_BPS / 2; // = 5

// =============================================================================
// AddStake success cases
// =============================================================================

#[tokio::test]
async fn test_add_stake_specific_amount() {
    let (sender, key): (_, Ed25519KeyPair) = get_key_pair();
    let coin_id = ObjectID::random();
    let coin = Object::with_id_owner_coin_for_testing(coin_id, sender, 50_000_000);

    let res = execute_add_stake(coin, Some(10_000_000), sender, SomaKeyPair::Ed25519(key)).await;

    let effects = res.txn_result.unwrap().into_data();
    assert_eq!(*effects.status(), ExecutionStatus::Success);

    // Should create a StakedSoma object
    assert!(effects.created().len() >= 1, "Should create at least one StakedSoma object");

    // Source coin should still exist with reduced balance
    let gas_used = effects.transaction_fee().total_fee;
    let source_obj = res.authority_state.get_object(&coin_id).await.unwrap();
    assert_eq!(
        source_obj.as_coin().unwrap(),
        50_000_000 - 10_000_000 - gas_used,
        "Source coin balance should be original - stake_amount - fees"
    );
}

#[tokio::test]
async fn test_add_stake_entire_coin_as_gas() {
    // Stake all (amount=None) when coin is also gas coin
    // Should deduct fees first, then stake the remainder
    let (sender, key): (_, Ed25519KeyPair) = get_key_pair();
    let coin_id = ObjectID::random();
    let balance = 50_000_000u64;
    let coin = Object::with_id_owner_coin_for_testing(coin_id, sender, balance);

    let res = execute_add_stake(
        coin,
        None, // stake all
        sender,
        SomaKeyPair::Ed25519(key),
    )
    .await;

    let effects = res.txn_result.unwrap().into_data();
    assert_eq!(*effects.status(), ExecutionStatus::Success);

    // Coin should be deleted (entire balance staked minus fees)
    let deleted_ids: Vec<ObjectID> = effects.deleted().iter().map(|d| d.0).collect();
    assert!(deleted_ids.contains(&coin_id), "Coin should be deleted when staking all");

    // Should have created a StakedSoma object
    assert!(effects.created().len() >= 1, "Should create StakedSoma");
}

// =============================================================================
// AddStake fee handling (half value fee)
// =============================================================================

#[tokio::test]
async fn test_add_stake_half_value_fee() {
    // Staking should get half the normal value fee rate
    let (sender, key): (_, Ed25519KeyPair) = get_key_pair();
    let stake_amount = 10_000_000u64;
    let coin = Object::with_id_owner_coin_for_testing(ObjectID::random(), sender, 50_000_000);

    let res = execute_add_stake(coin, Some(stake_amount), sender, SomaKeyPair::Ed25519(key)).await;

    let effects = res.txn_result.unwrap().into_data();
    assert_eq!(*effects.status(), ExecutionStatus::Success);

    let fee = effects.transaction_fee();
    // Value fee should be half the normal rate: amount * (value_fee_bps / 2) / BPS_DENOMINATOR
    let expected_value_fee = (stake_amount * STAKING_VALUE_FEE_BPS) / BPS_DENOMINATOR;
    assert_eq!(
        fee.value_fee, expected_value_fee,
        "Staking value fee should be half normal rate: got {}, expected {}",
        fee.value_fee, expected_value_fee
    );

    assert_eq!(fee.base_fee, BASE_FEE, "Base fee should be standard");
    assert_eq!(fee.total_fee, fee.base_fee + fee.operation_fee + fee.value_fee);
}

// =============================================================================
// AddStake failure cases
// =============================================================================

#[tokio::test]
async fn test_add_stake_insufficient_balance_specific_amount() {
    let (sender, key): (_, Ed25519KeyPair) = get_key_pair();
    // Balance that covers base fee but not stake amount + remaining fees
    let coin = Object::with_id_owner_coin_for_testing(ObjectID::random(), sender, 5000);

    let res = execute_add_stake(
        coin,
        Some(4500), // + base_fee(1000) + operation_fee + value_fee > 5000
        sender,
        SomaKeyPair::Ed25519(key),
    )
    .await;

    let effects = res.txn_result.unwrap().into_data();
    assert!(!effects.status().is_ok(), "Should fail: stake amount + fees > balance");
}

#[tokio::test]
async fn test_add_stake_insufficient_gas() {
    // Balance below base fee
    let (sender, key): (_, Ed25519KeyPair) = get_key_pair();
    let coin = Object::with_id_owner_coin_for_testing(ObjectID::random(), sender, 500);

    let res = execute_add_stake(coin, Some(100), sender, SomaKeyPair::Ed25519(key)).await;

    let effects = res.txn_result.unwrap().into_data();
    assert_eq!(
        *effects.status(),
        ExecutionStatus::Failure { error: ExecutionFailureStatus::InsufficientGas },
    );
}

#[tokio::test]
async fn test_add_stake_entire_coin_insufficient_for_fees() {
    // Stake-all where the balance is barely above base_fee but not enough for
    // value + operation fees
    let (sender, key): (_, Ed25519KeyPair) = get_key_pair();
    let coin = Object::with_id_owner_coin_for_testing(ObjectID::random(), sender, 1100);

    let res = execute_add_stake(
        coin,
        None, // stake all
        sender,
        SomaKeyPair::Ed25519(key),
    )
    .await;

    let effects = res.txn_result.unwrap().into_data();
    // After base fee (1000), only 100 left. Operation fee for 1 object = 300.
    // So total remaining fee > 100. Should fail.
    assert!(
        !effects.status().is_ok(),
        "Should fail: after base_fee, not enough for stake-all fees"
    );
}

// =============================================================================
// AddStake gas coin awareness
// =============================================================================

#[tokio::test]
async fn test_add_stake_gas_coin_reserves_for_fees() {
    // When gas coin == stake coin with specific amount, executor should check
    // that balance >= stake_amount + fees (not just stake_amount)
    let (sender, key): (_, Ed25519KeyPair) = get_key_pair();
    let balance = 10_000_000u64;
    let stake_amount = 9_990_000u64;
    let coin = Object::with_id_owner_coin_for_testing(ObjectID::random(), sender, balance);

    // After base_fee (1000), balance = 9_999_000
    // stake_amount (9_990_000) + operation_fee (2*300=600) + value_fee > 9_999_000?
    // Let's calculate: value_fee = 9_990_000 * 5 / 10000 = 4995
    // Total needed = 9_990_000 + 600 + 4995 = 9_995_595
    // Available after base fee = 9_999_000
    // This should succeed (9_999_000 > 9_995_595)

    let res = execute_add_stake(coin, Some(stake_amount), sender, SomaKeyPair::Ed25519(key)).await;

    let effects = res.txn_result.unwrap().into_data();
    assert_eq!(*effects.status(), ExecutionStatus::Success);
}

// =============================================================================
// WithdrawStake tests (basic validation only — requires active epoch with staked objects)
// Note: Full withdraw testing requires the staking pool to accept the withdrawal
// request, which needs a proper epoch context. These tests verify the execution
// path handles basic validation.
// =============================================================================

#[tokio::test]
async fn test_withdraw_stake_nonexistent_object() {
    let (sender, key): (_, Ed25519KeyPair) = get_key_pair();
    let gas_id = ObjectID::random();
    let gas = Object::with_id_owner_coin_for_testing(gas_id, sender, 10_000_000);

    let authority_state = TestAuthorityBuilder::new().build().await;
    authority_state.insert_genesis_object(gas.clone()).await;

    let fake_staked_ref = (ObjectID::random(), (0u64).into(), types::digests::ObjectDigest::MIN);
    let gas_ref = gas.compute_object_reference();

    let data = TransactionData::new(
        TransactionKind::WithdrawStake { staked_soma: fake_staked_ref },
        sender,
        vec![gas_ref],
    );
    let tx = to_sender_signed_transaction(data, &key);
    let result = send_and_confirm_transaction_(&authority_state, None, tx, true).await;

    // Should fail because the staked soma object doesn't exist
    // The exact error depends on input loading phase
    match result {
        Ok((_, effects)) => {
            assert!(!effects.status().is_ok(), "Should fail: nonexistent staked object");
        }
        Err(_) => {
            // Also acceptable — input loading may reject before execution
        }
    }
}

// =============================================================================
// Helpers
// =============================================================================

struct TransactionResult {
    authority_state: Arc<AuthorityState>,
    txn_result: Result<SignedTransactionEffects, SomaError>,
}

async fn execute_add_stake(
    coin: Object,
    amount: Option<u64>,
    sender: SomaAddress,
    sender_key: SomaKeyPair,
) -> TransactionResult {
    let authority_state = TestAuthorityBuilder::new().build().await;
    let coin_ref = coin.compute_object_reference();
    authority_state.insert_genesis_object(coin).await;

    // Get the first validator's soma_address from the system state
    let validator_address = {
        let system_state = authority_state.get_system_state_object_for_testing().unwrap();
        system_state.validators().validators[0].metadata.soma_address
    };

    let data = TransactionData::new(
        TransactionKind::AddStake { address: validator_address, coin_ref, amount },
        sender,
        vec![coin_ref],
    );
    let tx = to_sender_signed_transaction(data, &sender_key);
    // AddStake requires SystemState (shared object) — must use with_shared: true
    let txn_result = send_and_confirm_transaction_(&authority_state, None, tx, true)
        .await
        .map(|(_, effects)| effects);

    TransactionResult { authority_state, txn_result }
}
