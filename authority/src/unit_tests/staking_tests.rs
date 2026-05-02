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

/// Stage 9b: a successful AddStake writes both a StakedSomaV1 object
/// **and** a matching row in the `delegations` column family. The two
/// must agree on principal — they're the dual-write Stage 9d will
/// later collapse into one.
#[tokio::test]
async fn test_add_stake_dual_writes_delegation_row() {
    let (sender, key): (_, Ed25519KeyPair) = get_key_pair();
    let stake_id = ObjectID::random();
    let stake_amount = 7_500_000u64;
    let stake_coin = Object::with_id_owner_soma_coin_for_testing(stake_id, sender, 50_000_000);

    let res = execute_add_stake(
        stake_coin,
        Some(stake_amount),
        sender,
        SomaKeyPair::Ed25519(key),
    )
    .await;
    let effects = res.txn_result.unwrap().into_data();
    assert_eq!(*effects.status(), ExecutionStatus::Success);

    // The AddStake's write_transaction_outputs lands in the writeback
    // cache's dirty state but isn't flushed to the perpetual store
    // until a checkpoint executor commits it. Unit tests skip that
    // path, so flush manually here so `database_for_testing()` reads
    // the delegations table from the underlying RocksDB. Production
    // (e2e) flow flushes via the checkpoint executor; this is purely
    // a unit-test plumbing concern.
    let tx_digest = effects.transaction_digest();
    let epoch = res.authority_state.epoch_store_for_testing().epoch();
    let batch = res.authority_state.get_cache_commit().build_db_batch(epoch, &[*tx_digest]);
    res.authority_state.get_cache_commit().commit_transaction_outputs(epoch, batch, &[*tx_digest]);

    // Find the created StakedSoma so we can compare against the
    // delegation row by the same key.
    let store = res.authority_state.database_for_testing();
    let mut staked_soma_pool: Option<types::object::ObjectID> = None;
    let mut staked_soma_principal: Option<u64> = None;
    let mut staked_soma_activation_epoch: Option<u64> = None;

    for created in effects.created() {
        let obj = res.authority_state.get_object(&created.0.0).await.unwrap();
        if let Some(staked) = types::object::Object::as_staked_soma(&obj) {
            staked_soma_pool = Some(staked.pool_id);
            staked_soma_principal = Some(staked.principal);
            staked_soma_activation_epoch = Some(staked.stake_activation_epoch);
            break;
        }
    }
    let pool = staked_soma_pool.expect("a StakedSomaV1 must have been created");
    let principal = staked_soma_principal.unwrap();
    let activation_epoch = staked_soma_activation_epoch.unwrap();
    assert_eq!(
        principal, stake_amount,
        "StakedSoma.principal must equal the requested stake amount",
    );

    // Delegation row must mirror the object exactly.
    let delegation_principal = store.get_delegation(pool, sender, activation_epoch).unwrap();
    assert_eq!(
        delegation_principal, principal,
        "delegations[(pool, staker, activation_epoch)] must equal StakedSoma.principal",
    );
    let listed = store.iter_delegations_for_staker(sender).unwrap();
    assert_eq!(listed.len(), 1, "exactly one delegation row should exist for this staker");
    assert_eq!(listed[0], (pool, activation_epoch, principal));
}

/// Stage 9b: a successful WithdrawStake removes both the StakedSomaV1
/// object **and** the matching delegations table row. Verifies the
/// negative side of the dual-write — the executor emits
/// `-principal`, and the post-execution write path drains the row to
/// zero (which `apply_delegation_events` then deletes outright per
/// the row-deletion contract on `set_delegation`).
///
/// Inlines AddStake here rather than calling `execute_add_stake` so
/// the test keeps ownership of the keypair across both transactions
/// (SomaKeyPair isn't `Clone`).
#[tokio::test]
async fn test_withdraw_stake_dual_writes_delegation_removal() {
    use types::object::Owner;

    let (sender, key): (_, Ed25519KeyPair) = get_key_pair();
    let sender_key = SomaKeyPair::Ed25519(key);
    let stake_amount = 4_500_000u64;

    let authority_state = TestAuthorityBuilder::new().build().await;

    let stake_coin =
        Object::with_id_owner_soma_coin_for_testing(ObjectID::random(), sender, 50_000_000);
    let stake_ref = stake_coin.compute_object_reference();
    authority_state.insert_genesis_object(stake_coin).await;

    let gas_coin =
        Object::with_id_owner_coin_for_testing(ObjectID::random(), sender, 10_000_000);
    let gas_ref = gas_coin.compute_object_reference();
    authority_state.insert_genesis_object(gas_coin).await;

    let validator_address = {
        let system_state = authority_state.get_system_state_object_for_testing().unwrap();
        system_state.validators().validators[0].metadata.soma_address
    };

    // Step 1: AddStake.
    let add_data = TransactionData::new(
        TransactionKind::AddStake {
            address: validator_address,
            coin_ref: stake_ref,
            amount: Some(stake_amount),
        },
        sender,
        vec![gas_ref],
    );
    let add_tx = to_sender_signed_transaction(add_data, &sender_key);
    let (_, add_effects) =
        send_and_confirm_transaction_(&authority_state, None, add_tx, true).await.unwrap();
    let add_effects = add_effects.into_data();
    assert_eq!(*add_effects.status(), ExecutionStatus::Success);

    // Flush so the delegation row lands in perpetual_tables before we
    // read it.
    let add_tx_digest = add_effects.transaction_digest();
    let epoch = authority_state.epoch_store_for_testing().epoch();
    let batch = authority_state.get_cache_commit().build_db_batch(epoch, &[*add_tx_digest]);
    authority_state.get_cache_commit().commit_transaction_outputs(
        epoch,
        batch,
        &[*add_tx_digest],
    );

    let store = authority_state.database_for_testing();
    let mut staked_oref: Option<ObjectRef> = None;
    let mut staked_pool: Option<ObjectID> = None;
    let mut staked_activation_epoch: Option<u64> = None;
    for created in add_effects.created() {
        let obj = authority_state.get_object(&created.0.0).await.unwrap();
        if let Some(staked) = types::object::Object::as_staked_soma(&obj) {
            // The owner check matters here — genesis stakes also exist
            // and live under different addresses; we only want the one
            // we just created.
            if matches!(obj.owner(), Owner::AddressOwner(addr) if *addr == sender) {
                staked_oref = Some(obj.compute_object_reference());
                staked_pool = Some(staked.pool_id);
                staked_activation_epoch = Some(staked.stake_activation_epoch);
                break;
            }
        }
    }
    let staked_oref = staked_oref.expect("StakedSomaV1 must exist after AddStake");
    let pool = staked_pool.unwrap();
    let activation_epoch = staked_activation_epoch.unwrap();

    // Sanity: the row is there before WithdrawStake.
    assert_eq!(
        store.get_delegation(pool, sender, activation_epoch).unwrap(),
        stake_amount,
        "delegation row must mirror the StakedSoma post-AddStake",
    );

    // Step 2: WithdrawStake — produces a Coin output and removes the
    // StakedSomaV1. The dual-write should also clear the delegations
    // row.
    let gas_coin2 =
        Object::with_id_owner_coin_for_testing(ObjectID::random(), sender, 10_000_000);
    let gas_ref2 = gas_coin2.compute_object_reference();
    authority_state.insert_genesis_object(gas_coin2).await;

    let withdraw_data = TransactionData::new(
        TransactionKind::WithdrawStake { staked_soma: staked_oref },
        sender,
        vec![gas_ref2],
    );
    let withdraw_tx = to_sender_signed_transaction(withdraw_data, &sender_key);
    let (_, withdraw_effects) =
        send_and_confirm_transaction_(&authority_state, None, withdraw_tx, true)
            .await
            .unwrap();
    let withdraw_effects = withdraw_effects.into_data();
    assert_eq!(
        *withdraw_effects.status(),
        ExecutionStatus::Success,
        "WithdrawStake must succeed",
    );

    // Flush so the perpetual store reflects the WithdrawStake's writes.
    let withdraw_tx_digest = withdraw_effects.transaction_digest();
    let batch = authority_state
        .get_cache_commit()
        .build_db_batch(epoch, &[*withdraw_tx_digest]);
    authority_state.get_cache_commit().commit_transaction_outputs(
        epoch,
        batch,
        &[*withdraw_tx_digest],
    );

    // The StakedSomaV1 object is deleted (existing behavior) — so is
    // its delegations row.
    assert_eq!(
        store.get_delegation(pool, sender, activation_epoch).unwrap(),
        0,
        "delegation row must drain to zero after WithdrawStake",
    );
    let listed = store.iter_delegations_for_staker(sender).unwrap();
    assert!(
        listed.is_empty(),
        "no delegation rows should remain for `sender` after the only stake is withdrawn \
         (got {:?})",
        listed,
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
