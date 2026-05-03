// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Tests for AddStake (Stage 9d-C2: balance-mode) and WithdrawStake.
//!
//! Stake principal is SOMA, debited from the sender's accumulator
//! balance — no SOMA coin input. Gas is paid in USDC (still
//! coin-mode here for simplicity). The F1 fold-to-balance path is
//! verified separately via the dual-write delegation tests.

use std::sync::Arc;

use fastcrypto::ed25519::Ed25519KeyPair;
use types::base::SomaAddress;
use types::crypto::{SomaKeyPair, get_key_pair};
use types::effects::{
    ExecutionFailureStatus, ExecutionStatus, SignedTransactionEffects, TransactionEffectsAPI,
};
use types::error::SomaError;
use types::object::{CoinType, Object, ObjectID, ObjectRef};
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
async fn test_add_stake_balance_mode_succeeds() {
    let (sender, key): (_, Ed25519KeyPair) = get_key_pair();
    let stake_amount = 10_000_000u64;
    let starting_balance = 50_000_000u64;

    let res = execute_add_stake(
        starting_balance,
        stake_amount,
        sender,
        SomaKeyPair::Ed25519(key),
    )
    .await;

    let effects = res.txn_result.unwrap().into_data();
    assert_eq!(*effects.status(), ExecutionStatus::Success);

    // Should create a StakedSomaV1 object (still dual-written until C5).
    assert!(effects.created().len() >= 1, "Should create StakedSomaV1");

    // Flush so the writeback cache's settlement events land in the
    // perpetual store (unit tests skip the checkpoint executor that
    // does this in production — same plumbing concern as
    // delegation_dual_write tests).
    let tx_digest = effects.transaction_digest();
    let epoch = res.authority_state.epoch_store_for_testing().epoch();
    let batch = res.authority_state.get_cache_commit().build_db_batch(epoch, &[*tx_digest]);
    res.authority_state.get_cache_commit().commit_transaction_outputs(epoch, batch, &[*tx_digest]);

    let store = res.authority_state.database_for_testing();
    let post_balance = store.get_balance(sender, CoinType::Soma).unwrap();
    assert_eq!(
        post_balance,
        starting_balance - stake_amount,
        "stake_amount must be debited from the SOMA accumulator",
    );
}

#[tokio::test]
async fn test_add_stake_zero_amount_rejected() {
    // Stage 9d-C2: zero is now an explicit error in the executor.
    let (sender, key): (_, Ed25519KeyPair) = get_key_pair();
    let res = execute_add_stake(
        50_000_000,
        0,
        sender,
        SomaKeyPair::Ed25519(key),
    )
    .await;

    let effects = res.txn_result.unwrap().into_data();
    assert!(!effects.status().is_ok(), "zero stake amount must fail");
}

#[tokio::test]
async fn test_add_stake_charges_fixed_fee() {
    let (sender, key): (_, Ed25519KeyPair) = get_key_pair();
    let res = execute_add_stake(
        50_000_000,
        10_000_000,
        sender,
        SomaKeyPair::Ed25519(key),
    )
    .await;

    let effects = res.txn_result.unwrap().into_data();
    assert_eq!(*effects.status(), ExecutionStatus::Success);

    let fee = effects.transaction_fee();
    assert_eq!(fee.total_fee, STAKING_FEE);
}

#[tokio::test]
async fn test_add_stake_insufficient_gas() {
    // Force a failure by giving a USDC gas coin smaller than STAKING_FEE.
    let (sender, key): (_, Ed25519KeyPair) = get_key_pair();
    let res = execute_add_stake_with_gas(
        50_000_000,
        1_000,
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

/// A successful AddStake writes the F1 delegation row alongside the
/// StakedSomaV1 object until Stage 9d-C5 collapses to one source.
/// Verifies the row's principal matches the StakedSomaV1's principal,
/// and that `last_collected_period` advanced from 0 to the pool's
/// current period.
#[tokio::test]
async fn test_add_stake_dual_writes_delegation_row() {
    let (sender, key): (_, Ed25519KeyPair) = get_key_pair();
    let stake_amount = 7_500_000u64;

    let res = execute_add_stake(
        50_000_000,
        stake_amount,
        sender,
        SomaKeyPair::Ed25519(key),
    )
    .await;
    let effects = res.txn_result.unwrap().into_data();
    assert_eq!(*effects.status(), ExecutionStatus::Success);

    // Flush so the delegation row lands in perpetual_tables before we
    // read it.
    let tx_digest = effects.transaction_digest();
    let epoch = res.authority_state.epoch_store_for_testing().epoch();
    let batch = res.authority_state.get_cache_commit().build_db_batch(epoch, &[*tx_digest]);
    res.authority_state.get_cache_commit().commit_transaction_outputs(epoch, batch, &[*tx_digest]);

    // Find the created StakedSomaV1 to compare against the delegation row.
    let store = res.authority_state.database_for_testing();
    let mut staked_pool: Option<types::object::ObjectID> = None;
    let mut staked_principal: Option<u64> = None;

    for created in effects.created() {
        let obj = res.authority_state.get_object(&created.0.0).await.unwrap();
        if let Some(staked) = types::object::Object::as_staked_soma(&obj) {
            staked_pool = Some(staked.pool_id);
            staked_principal = Some(staked.principal);
            break;
        }
    }
    let pool = staked_pool.expect("a StakedSomaV1 must have been created");
    let principal = staked_principal.unwrap();
    assert_eq!(
        principal, stake_amount,
        "StakedSomaV1.principal must equal the requested stake amount",
    );

    let delegation = store.get_delegation(pool, sender).unwrap();
    assert_eq!(
        delegation.principal, principal,
        "delegations[(pool, staker)].principal must equal StakedSomaV1.principal",
    );

    // F1 fold semantics: AddStake advances last_collected_period to the
    // pool's current period.
    let system_state = res.authority_state.get_system_state_object_for_testing().unwrap();
    let pool_state = &system_state
        .validators()
        .validators
        .iter()
        .find(|v| v.staking_pool.id == pool)
        .expect("pool must exist on a validator")
        .staking_pool;
    assert_eq!(
        delegation.last_collected_period, pool_state.current_period,
        "AddStake must advance last_collected_period to the pool's current period",
    );

    let listed = store.iter_delegations_for_staker(sender).unwrap();
    assert_eq!(listed.len(), 1, "exactly one delegation row should exist for this staker");
    assert_eq!(listed[0].0, pool);
    assert_eq!(listed[0].1.principal, principal);
}

/// A successful WithdrawStake removes both the StakedSomaV1 object
/// **and** the matching delegation row. Verifies the negative side of
/// the dual-write — the executor emits `-principal`, and the
/// post-execution write path drains the row to zero (which
/// `apply_delegation_events` then deletes outright per the row-deletion
/// contract).
#[tokio::test]
async fn test_withdraw_stake_dual_writes_delegation_removal() {
    use types::object::Owner;

    let (sender, key): (_, Ed25519KeyPair) = get_key_pair();
    let sender_key = SomaKeyPair::Ed25519(key);
    let stake_amount = 4_500_000u64;

    let authority_state = TestAuthorityBuilder::new().build().await;

    // Seed sender's SOMA balance so balance-mode AddStake can debit it.
    authority_state
        .database_for_testing()
        .set_balance(sender, CoinType::Soma, 50_000_000)
        .unwrap();

    let gas_coin =
        Object::with_id_owner_coin_for_testing(ObjectID::random(), sender, 10_000_000);
    let gas_ref = gas_coin.compute_object_reference();
    authority_state.insert_genesis_object(gas_coin).await;

    let validator_address = {
        let system_state = authority_state.get_system_state_object_for_testing().unwrap();
        system_state.validators().validators[0].metadata.soma_address
    };

    // Step 1: AddStake (Stage 9d-C2: balance-mode).
    let add_data = TransactionData::new(
        TransactionKind::AddStake { validator: validator_address, amount: stake_amount },
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
    for created in add_effects.created() {
        let obj = authority_state.get_object(&created.0.0).await.unwrap();
        if let Some(_staked) = types::object::Object::as_staked_soma(&obj) {
            // Owner check: genesis stakes also exist; we want the new one.
            if matches!(obj.owner(), Owner::AddressOwner(addr) if *addr == sender) {
                staked_oref = Some(obj.compute_object_reference());
                staked_pool = Some(_staked.pool_id);
                break;
            }
        }
    }
    let staked_oref = staked_oref.expect("StakedSomaV1 must exist after AddStake");
    let pool = staked_pool.unwrap();

    assert_eq!(
        store.get_delegation(pool, sender).unwrap().principal,
        stake_amount,
        "delegation row must mirror StakedSomaV1 post-AddStake",
    );

    // Step 2: WithdrawStake — produces a Coin output and removes the
    // StakedSomaV1. The dual-write should also clear the delegation row.
    let gas_coin2 =
        Object::with_id_owner_coin_for_testing(ObjectID::random(), sender, 10_000_000);
    let gas_ref2 = gas_coin2.compute_object_reference();
    authority_state.insert_genesis_object(gas_coin2).await;

    // Stage 9d-C3: WithdrawStake is balance-mode and keys off
    // (pool_id, sender). Use `amount: None` to drain the entire row.
    let _ = staked_oref;
    let withdraw_data = TransactionData::new(
        TransactionKind::WithdrawStake { pool_id: pool, amount: None },
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

    let withdraw_tx_digest = withdraw_effects.transaction_digest();
    let batch = authority_state
        .get_cache_commit()
        .build_db_batch(epoch, &[*withdraw_tx_digest]);
    authority_state.get_cache_commit().commit_transaction_outputs(
        epoch,
        batch,
        &[*withdraw_tx_digest],
    );

    assert_eq!(
        store.get_delegation(pool, sender).unwrap().principal,
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

    let gas_ref = gas.compute_object_reference();

    // Stage 9d-C3: WithdrawStake against a pool the sender has no
    // stake in. Executor reads the prefetched (pool, sender) row,
    // finds none, errors out.
    let data = TransactionData::new(
        TransactionKind::WithdrawStake {
            pool_id: ObjectID::random(),
            amount: None,
        },
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

/// Submit an AddStake (balance-mode) where the sender's SOMA
/// accumulator is pre-funded with `soma_balance`. A USDC gas coin
/// (with default funding) is created automatically.
async fn execute_add_stake(
    soma_balance: u64,
    amount: u64,
    sender: SomaAddress,
    sender_key: SomaKeyPair,
) -> TransactionResult {
    execute_add_stake_with_gas(soma_balance, amount, 10_000_000, sender, sender_key).await
}

async fn execute_add_stake_with_gas(
    soma_balance: u64,
    amount: u64,
    gas_balance: u64,
    sender: SomaAddress,
    sender_key: SomaKeyPair,
) -> TransactionResult {
    let authority_state = TestAuthorityBuilder::new().build().await;

    // Stage 9d-C2: AddStake debits the sender's SOMA balance directly.
    authority_state
        .database_for_testing()
        .set_balance(sender, CoinType::Soma, soma_balance)
        .unwrap();

    // USDC gas coin.
    let gas_coin =
        Object::with_id_owner_coin_for_testing(ObjectID::random(), sender, gas_balance);
    let gas_ref = gas_coin.compute_object_reference();
    authority_state.insert_genesis_object(gas_coin).await;

    let validator_address = {
        let system_state = authority_state.get_system_state_object_for_testing().unwrap();
        system_state.validators().validators[0].metadata.soma_address
    };

    let data = TransactionData::new(
        TransactionKind::AddStake { validator: validator_address, amount },
        sender,
        vec![gas_ref],
    );
    let tx = to_sender_signed_transaction(data, &sender_key);
    let txn_result = send_and_confirm_transaction_(&authority_state, None, tx, true)
        .await
        .map(|(_, effects)| effects);

    TransactionResult { authority_state, txn_result }
}
