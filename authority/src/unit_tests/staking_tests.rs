// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Tests for AddStake and WithdrawStake under the unit-fee USDC-gas model.
//!
//! Stake principal is SOMA (because validators are staked in SOMA), gas is USDC.
//! Each test thus needs two coins: a USDC gas coin and a SOMA stake coin.

use std::sync::Arc;

use fastcrypto::ed25519::Ed25519KeyPair;
use types::base::SomaAddress;
use types::crypto::{SomaKeyPair, get_key_pair};
use types::effects::{
    ExecutionFailureStatus, ExecutionStatus, SignedTransactionEffects, TransactionEffectsAPI,
};
use types::error::SomaError;
use types::object::{Object, ObjectID, ObjectRef};
use types::transaction::{TransactionData, TransactionKind};
use types::unit_tests::utils::to_sender_signed_transaction;

use crate::authority::AuthorityState;
use crate::authority_test_utils::send_and_confirm_transaction_;
use crate::test_authority_builder::TestAuthorityBuilder;

// Default `unit_fee` from protocol config v1.
const UNIT_FEE: u64 = 1000;
// Staking ops cost a fixed 2 units (see StakingExecutor::fee_units).
const STAKING_FEE_UNITS: u64 = 2;
const STAKING_FEE: u64 = UNIT_FEE * STAKING_FEE_UNITS;

#[tokio::test]
async fn test_add_stake_specific_amount() {
    let (sender, key): (_, Ed25519KeyPair) = get_key_pair();
    let stake_id = ObjectID::random();
    let stake_coin = Object::with_id_owner_soma_coin_for_testing(stake_id, sender, 50_000_000);

    let res = execute_add_stake(stake_coin, Some(10_000_000), sender, SomaKeyPair::Ed25519(key))
        .await;

    let effects = res.txn_result.unwrap().into_data();
    assert_eq!(*effects.status(), ExecutionStatus::Success);

    // Should create a StakedSoma object.
    assert!(effects.created().len() >= 1, "Should create StakedSoma");

    // SOMA stake coin should still exist with reduced balance (gas is paid in USDC).
    let stake_obj = res.authority_state.get_object(&stake_id).await.unwrap();
    assert_eq!(
        stake_obj.as_coin().unwrap(),
        50_000_000 - 10_000_000,
        "Stake coin balance = original - stake_amount (no fee deducted from SOMA)"
    );
}

#[tokio::test]
async fn test_add_stake_entire_coin() {
    // amount = None stakes the entire SOMA coin balance.
    let (sender, key): (_, Ed25519KeyPair) = get_key_pair();
    let stake_id = ObjectID::random();
    let balance = 50_000_000u64;
    let stake_coin = Object::with_id_owner_soma_coin_for_testing(stake_id, sender, balance);

    let res = execute_add_stake(stake_coin, None, sender, SomaKeyPair::Ed25519(key)).await;

    let effects = res.txn_result.unwrap().into_data();
    assert_eq!(*effects.status(), ExecutionStatus::Success);

    // Stake coin should be deleted (entire balance staked).
    let deleted_ids: Vec<ObjectID> = effects.deleted().iter().map(|d| d.0).collect();
    assert!(deleted_ids.contains(&stake_id), "Stake coin should be deleted when staking all");

    assert!(effects.created().len() >= 1, "Should create StakedSoma");
}

#[tokio::test]
async fn test_add_stake_charges_fixed_fee() {
    let (sender, key): (_, Ed25519KeyPair) = get_key_pair();
    let stake_amount = 10_000_000u64;
    let stake_coin =
        Object::with_id_owner_soma_coin_for_testing(ObjectID::random(), sender, 50_000_000);

    let res = execute_add_stake(stake_coin, Some(stake_amount), sender, SomaKeyPair::Ed25519(key))
        .await;

    let effects = res.txn_result.unwrap().into_data();
    assert_eq!(*effects.status(), ExecutionStatus::Success);

    let fee = effects.transaction_fee();
    assert_eq!(fee.total_fee, STAKING_FEE);
}

#[tokio::test]
async fn test_add_stake_insufficient_balance_specific_amount() {
    // SOMA stake coin doesn't have enough for the requested stake amount.
    // Gas is USDC and is charged separately, so this fails on InsufficientCoinBalance
    // (the SOMA coin), not InsufficientGas.
    let (sender, key): (_, Ed25519KeyPair) = get_key_pair();
    let stake_coin =
        Object::with_id_owner_soma_coin_for_testing(ObjectID::random(), sender, 5000);

    let res = execute_add_stake(stake_coin, Some(6000), sender, SomaKeyPair::Ed25519(key)).await;

    let effects = res.txn_result.unwrap().into_data();
    assert!(!effects.status().is_ok(), "Should fail: stake amount > SOMA coin balance");
}

#[tokio::test]
async fn test_add_stake_insufficient_gas() {
    // Helper creates a small USDC gas coin (configurable). Force a failure by
    // overriding the gas amount below the staking fee.
    let (sender, key): (_, Ed25519KeyPair) = get_key_pair();
    let stake_coin =
        Object::with_id_owner_soma_coin_for_testing(ObjectID::random(), sender, 50_000_000);

    let res = execute_add_stake_with_gas(
        stake_coin,
        Some(1000),
        500, // USDC gas coin balance < STAKING_FEE (2000)
        sender,
        SomaKeyPair::Ed25519(key),
    )
    .await;

    let effects = res.txn_result.unwrap().into_data();
    assert_eq!(
        *effects.status(),
        ExecutionStatus::Failure { error: ExecutionFailureStatus::InsufficientGas },
    );
}

// =============================================================================
// WithdrawStake tests
// =============================================================================

#[tokio::test]
async fn test_withdraw_stake_nonexistent_object() {
    let (sender, key): (_, Ed25519KeyPair) = get_key_pair();
    let gas_id = ObjectID::random();
    // Gas coin is USDC.
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

    match result {
        Ok((_, effects)) => {
            assert!(!effects.status().is_ok(), "Should fail: nonexistent staked object");
        }
        Err(_) => {
            // Also acceptable — input loading may reject before execution.
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

/// Submit an AddStake with the given SOMA stake coin. A separate USDC gas
/// coin (with default funding) is generated automatically and inserted at
/// genesis.
async fn execute_add_stake(
    stake_coin: Object,
    amount: Option<u64>,
    sender: SomaAddress,
    sender_key: SomaKeyPair,
) -> TransactionResult {
    execute_add_stake_with_gas(stake_coin, amount, 10_000_000, sender, sender_key).await
}

/// Same as [`execute_add_stake`] but with a custom USDC gas coin balance.
async fn execute_add_stake_with_gas(
    stake_coin: Object,
    amount: Option<u64>,
    gas_balance: u64,
    sender: SomaAddress,
    sender_key: SomaKeyPair,
) -> TransactionResult {
    let authority_state = TestAuthorityBuilder::new().build().await;

    let stake_ref = stake_coin.compute_object_reference();
    authority_state.insert_genesis_object(stake_coin).await;

    // USDC gas coin (separate from the SOMA stake coin).
    let gas_coin =
        Object::with_id_owner_coin_for_testing(ObjectID::random(), sender, gas_balance);
    let gas_ref = gas_coin.compute_object_reference();
    authority_state.insert_genesis_object(gas_coin).await;

    let validator_address = {
        let system_state = authority_state.get_system_state_object_for_testing().unwrap();
        system_state.validators().validators[0].metadata.soma_address
    };

    let data = TransactionData::new(
        TransactionKind::AddStake { address: validator_address, coin_ref: stake_ref, amount },
        sender,
        vec![gas_ref],
    );
    let tx = to_sender_signed_transaction(data, &sender_key);
    let txn_result = send_and_confirm_transaction_(&authority_state, None, tx, true)
        .await
        .map(|(_, effects)| effects);

    TransactionResult { authority_state, txn_result }
}
