// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Tests for model management transactions:
//! CommitModel, DeactivateModel, SetModelCommissionRate, ReportModel.
//!
//! Note: RevealModel requires a different epoch than the commit epoch,
//! which is not testable in single-authority unit tests without epoch
//! advancement. These tests focus on CommitModel parameter validation
//! and model lifecycle operations that work within a single epoch.

use std::sync::Arc;

use fastcrypto::ed25519::Ed25519KeyPair;
use types::{
    SYSTEM_STATE_OBJECT_ID,
    base::SomaAddress,
    crypto::{SomaKeyPair, get_key_pair},
    digests::{ModelWeightsCommitment, ModelWeightsUrlCommitment},
    effects::{ExecutionStatus, TransactionEffectsAPI},
    error::SomaError,
    object::{Object, ObjectID},
    transaction::{CommitModelArgs, TransactionData, TransactionKind},
    unit_tests::utils::to_sender_signed_transaction,
};

use crate::{
    authority::AuthorityState, authority_test_utils::send_and_confirm_transaction_,
    test_authority_builder::TestAuthorityBuilder,
};

// Default fees from protocol config v1
const BPS_DENOMINATOR: u64 = 10000;

// =============================================================================
// Helpers
// =============================================================================

struct TransactionResult {
    authority_state: Arc<AuthorityState>,
    txn_result: Result<types::effects::SignedTransactionEffects, SomaError>,
    coin_id: ObjectID,
}

async fn execute_commit_model(
    coin: Object,
    stake_amount: u64,
    architecture_version: Option<u64>,
    commission_rate: Option<u64>,
    sender: SomaAddress,
    sender_key: SomaKeyPair,
) -> TransactionResult {
    let authority_state = TestAuthorityBuilder::new().build().await;
    let coin_ref = coin.compute_object_reference();
    let coin_id = coin_ref.0;
    authority_state.insert_genesis_object(coin).await;

    // Get architecture version from system state if not overridden
    let system_state = authority_state.get_system_state_object_for_testing().unwrap();
    let arch_version =
        architecture_version.unwrap_or(system_state.parameters().model_architecture_version);
    let comm_rate = commission_rate.unwrap_or(1000); // 10% default

    let data = TransactionData::new(
        TransactionKind::CommitModel(CommitModelArgs {
            model_id: ObjectID::random(),
            weights_url_commitment: ModelWeightsUrlCommitment::new([1u8; 32]),
            weights_commitment: ModelWeightsCommitment::new([2u8; 32]),
            architecture_version: arch_version,
            stake_amount,
            commission_rate: comm_rate,
            staking_pool_id: ObjectID::random(),
        }),
        sender,
        vec![coin_ref],
    );
    let tx = to_sender_signed_transaction(data, &sender_key);
    // CommitModel modifies SystemState (shared object)
    let txn_result = send_and_confirm_transaction_(&authority_state, None, tx, true)
        .await
        .map(|(_, effects)| effects);

    TransactionResult { authority_state, txn_result, coin_id }
}

// =============================================================================
// CommitModel success
// =============================================================================

#[tokio::test]
async fn test_commit_model_success() {
    let (sender, key): (_, Ed25519KeyPair) = get_key_pair();
    // Need enough for min stake (1 SOMA = 1_000_000_000) + fees
    let coin = Object::with_id_owner_coin_for_testing(ObjectID::random(), sender, 2_000_000_000);

    let res = execute_commit_model(
        coin,
        1_000_000_000, // min stake
        None,          // use default architecture version
        Some(1000),    // 10% commission
        sender,
        SomaKeyPair::Ed25519(key),
    )
    .await;

    let effects = res.txn_result.unwrap().into_data();
    assert_eq!(*effects.status(), ExecutionStatus::Success);

    // Should create a StakedSoma object (for the model's staking pool)
    assert!(!effects.created().is_empty(), "Should create at least one object (StakedSoma)");

    // SystemState should be mutated (model added to pending)
    let mutated_ids: Vec<ObjectID> = effects.mutated().iter().map(|m| m.0.0).collect();
    assert!(
        mutated_ids.contains(&SYSTEM_STATE_OBJECT_ID),
        "SystemState should be mutated during CommitModel"
    );

    // Verify the model was added to pending_models in system state
    let system_state = res.authority_state.get_system_state_object_for_testing().unwrap();
    assert!(
        !system_state.model_registry().pending_models.is_empty(),
        "Model should be in pending_models after commit"
    );

    // Verify the gas coin balance was reduced by stake_amount + fees
    let gas_obj = res.authority_state.get_object(&res.coin_id).await.unwrap();
    let remaining = gas_obj.as_coin().unwrap();
    let total_fee = effects.transaction_fee().total_fee;
    let expected_remaining = 2_000_000_000u64 - 1_000_000_000 - total_fee;
    assert_eq!(
        remaining, expected_remaining,
        "Gas coin should be reduced by stake_amount ({}) + fees ({}): got {}, expected {}",
        1_000_000_000u64, total_fee, remaining, expected_remaining
    );
}

// =============================================================================
// CommitModel parameter validation failures
// =============================================================================

#[tokio::test]
async fn test_commit_model_bad_architecture_version() {
    let (sender, key): (_, Ed25519KeyPair) = get_key_pair();
    let coin = Object::with_id_owner_coin_for_testing(ObjectID::random(), sender, 2_000_000_000);

    let res = execute_commit_model(
        coin,
        1_000_000_000,
        Some(999), // wrong version
        Some(1000),
        sender,
        SomaKeyPair::Ed25519(key),
    )
    .await;

    let effects = res.txn_result.unwrap().into_data();
    assert!(!effects.status().is_ok(), "Should fail: architecture version mismatch");
}

#[tokio::test]
async fn test_commit_model_min_stake_not_met() {
    let (sender, key): (_, Ed25519KeyPair) = get_key_pair();
    let coin = Object::with_id_owner_coin_for_testing(ObjectID::random(), sender, 50_000_000);

    let res = execute_commit_model(
        coin,
        100, // way below min stake (1_000_000_000)
        None,
        Some(1000),
        sender,
        SomaKeyPair::Ed25519(key),
    )
    .await;

    let effects = res.txn_result.unwrap().into_data();
    assert!(!effects.status().is_ok(), "Should fail: stake amount below minimum");
}

#[tokio::test]
async fn test_commit_model_commission_rate_too_high() {
    let (sender, key): (_, Ed25519KeyPair) = get_key_pair();
    let coin = Object::with_id_owner_coin_for_testing(ObjectID::random(), sender, 2_000_000_000);

    let res = execute_commit_model(
        coin,
        1_000_000_000,
        None,
        Some(BPS_DENOMINATOR + 1), // > 100%
        sender,
        SomaKeyPair::Ed25519(key),
    )
    .await;

    let effects = res.txn_result.unwrap().into_data();
    assert!(!effects.status().is_ok(), "Should fail: commission rate exceeds BPS_DENOMINATOR");
}

#[tokio::test]
async fn test_commit_model_insufficient_gas() {
    // Balance below base fee should fail with InsufficientGas
    let (sender, key): (_, Ed25519KeyPair) = get_key_pair();
    let coin = Object::with_id_owner_coin_for_testing(
        ObjectID::random(),
        sender,
        500, // below base fee (1000)
    );

    let res = execute_commit_model(
        coin,
        100, // below min_stake but gas check happens first
        None,
        Some(1000),
        sender,
        SomaKeyPair::Ed25519(key),
    )
    .await;

    let effects = res.txn_result.unwrap().into_data();
    assert!(!effects.status().is_ok(), "Should fail: balance insufficient for gas");
}

#[tokio::test]
async fn test_commit_model_insufficient_balance_for_stake() {
    // Balance covers base fee and min_stake validation passes, but not enough
    // for stake_amount + remaining fees after base fee deduction.
    let (sender, key): (_, Ed25519KeyPair) = get_key_pair();
    // 1_000_010_000 > min_stake (1B) but after base fee (1000), remaining = 1_000_009_000
    // which is < stake (1B) + value_fee + write_fees
    let coin = Object::with_id_owner_coin_for_testing(ObjectID::random(), sender, 1_000_010_000);

    let res = execute_commit_model(
        coin,
        1_000_000_000, // min stake — consumes nearly all balance
        None,
        Some(1000),
        sender,
        SomaKeyPair::Ed25519(key),
    )
    .await;

    let effects = res.txn_result.unwrap().into_data();
    assert!(!effects.status().is_ok(), "Should fail: balance insufficient for stake_amount + fees");
}

// =============================================================================
// CommitModel fee handling (half value fee, same as staking)
// =============================================================================

#[tokio::test]
async fn test_commit_model_half_value_fee() {
    let (sender, key): (_, Ed25519KeyPair) = get_key_pair();
    let stake_amount = 1_000_000_000u64;
    let coin = Object::with_id_owner_coin_for_testing(ObjectID::random(), sender, 2_000_000_000);

    let res = execute_commit_model(
        coin,
        stake_amount,
        None,
        Some(1000),
        sender,
        SomaKeyPair::Ed25519(key),
    )
    .await;

    let effects = res.txn_result.unwrap().into_data();
    assert_eq!(*effects.status(), ExecutionStatus::Success);

    let fee = effects.transaction_fee();
    // Model staking gets half value fee (same as validator staking)
    // value_fee_bps = 10 / 2 = 5
    let expected_value_fee = (stake_amount * 5) / BPS_DENOMINATOR;
    assert_eq!(
        fee.value_fee, expected_value_fee,
        "Model commit should have half value fee: got {}, expected {}",
        fee.value_fee, expected_value_fee
    );
}

// =============================================================================
// ReportModel tests (requires validator signer)
// =============================================================================

#[tokio::test]
async fn test_report_model_not_a_validator() {
    // Non-validator trying to report a model should fail
    let (sender, key): (_, Ed25519KeyPair) = get_key_pair();
    let gas = Object::with_id_owner_coin_for_testing(ObjectID::random(), sender, 10_000_000);
    let gas_ref = gas.compute_object_reference();

    let authority_state = TestAuthorityBuilder::new().build().await;
    authority_state.insert_genesis_object(gas).await;

    let data = TransactionData::new(
        TransactionKind::ReportModel { model_id: ObjectID::random() },
        sender,
        vec![gas_ref],
    );
    let tx = to_sender_signed_transaction(data, &key);
    let result = send_and_confirm_transaction_(&authority_state, None, tx, true).await;

    match result {
        Ok((_, effects)) => {
            assert!(!effects.status().is_ok(), "Non-validator should not be able to report model");
        }
        Err(_) => {
            // Also acceptable
        }
    }
}

// =============================================================================
// SetModelCommissionRate tests
// =============================================================================

#[tokio::test]
async fn test_set_model_commission_rate_nonexistent_model() {
    // Setting commission on a non-existent model should fail
    let (sender, key): (_, Ed25519KeyPair) = get_key_pair();
    let gas = Object::with_id_owner_coin_for_testing(ObjectID::random(), sender, 10_000_000);
    let gas_ref = gas.compute_object_reference();

    let authority_state = TestAuthorityBuilder::new().build().await;
    authority_state.insert_genesis_object(gas).await;

    let data = TransactionData::new(
        TransactionKind::SetModelCommissionRate { model_id: ObjectID::random(), new_rate: 500 },
        sender,
        vec![gas_ref],
    );
    let tx = to_sender_signed_transaction(data, &key);
    let result = send_and_confirm_transaction_(&authority_state, None, tx, true).await;

    match result {
        Ok((_, effects)) => {
            assert!(!effects.status().is_ok(), "Should fail: model doesn't exist");
        }
        Err(_) => {
            // Also acceptable
        }
    }
}

// =============================================================================
// DeactivateModel tests
// =============================================================================

#[tokio::test]
async fn test_deactivate_model_nonexistent() {
    // Deactivating a non-existent model should fail
    let (sender, key): (_, Ed25519KeyPair) = get_key_pair();
    let gas = Object::with_id_owner_coin_for_testing(ObjectID::random(), sender, 10_000_000);
    let gas_ref = gas.compute_object_reference();

    let authority_state = TestAuthorityBuilder::new().build().await;
    authority_state.insert_genesis_object(gas).await;

    let data = TransactionData::new(
        TransactionKind::DeactivateModel { model_id: ObjectID::random() },
        sender,
        vec![gas_ref],
    );
    let tx = to_sender_signed_transaction(data, &key);
    let result = send_and_confirm_transaction_(&authority_state, None, tx, true).await;

    match result {
        Ok((_, effects)) => {
            assert!(!effects.status().is_ok(), "Should fail: model doesn't exist");
        }
        Err(_) => {
            // Also acceptable
        }
    }
}
