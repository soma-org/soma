// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Tests for the unit-fee gas model.
//!
//! Each tx pays `unit_fee × executor.fee_units(...)`. For Transfer ops, the
//! executor charges `coins.len() + recipients.len()` units; this means more
//! input coins or more recipients raises the fee linearly.

use std::sync::Arc;

use fastcrypto::ed25519::Ed25519KeyPair;
use futures::future::join_all;
use types::base::{SomaAddress, dbg_addr};
use types::crypto::{SomaKeyPair, get_key_pair};
use types::effects::{
    ExecutionFailureStatus, ExecutionStatus, SignedTransactionEffects, TransactionEffectsAPI,
};
use types::error::SomaError;
use types::object::{Object, ObjectID, ObjectRef};
use types::transaction::TransactionData;
use types::unit_tests::utils::to_sender_signed_transaction;

use crate::authority::AuthorityState;
use crate::authority_test_utils::send_and_confirm_transaction;
use crate::test_authority_builder::TestAuthorityBuilder;

// Default `unit_fee` from protocol config v1.
const UNIT_FEE: u64 = 1000;

#[tokio::test]
async fn test_fee_deducted_on_success() {
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
    // BalanceTransfer fee_units = 1 + transfers.len() = 2 with one recipient.
    assert_eq!(fee.total_fee, 2 * UNIT_FEE);

    // Stage 13b: gas coin gets debited only by the fee. The transfer
    // amount comes out of the sender's SOMA balance accumulator.
    let gas_obj = res.authority_state.get_object(&coin_ref.0).await.unwrap();
    assert_eq!(gas_obj.as_coin().unwrap(), 10_000_000 - fee.total_fee);
}

#[tokio::test]
async fn test_insufficient_gas_drains_what_it_can() {
    // Balance below the required unit fee: prepare_gas takes whatever is
    // available and reports InsufficientGas.
    let (sender, sender_key): (_, Ed25519KeyPair) = get_key_pair();
    let coin = Object::with_id_owner_coin_for_testing(ObjectID::random(), sender, 500); // < UNIT_FEE

    let recipient = dbg_addr(1);
    let res =
        execute_transfer_coin(coin, recipient, Some(100), sender, SomaKeyPair::Ed25519(sender_key))
            .await;

    let effects = res.txn_result.unwrap().into_data();
    assert_eq!(
        *effects.status(),
        ExecutionStatus::Failure { error: ExecutionFailureStatus::InsufficientGas },
    );

    // Whatever was available is deducted.
    let fee = effects.transaction_fee();
    assert!(fee.total_fee <= 500);
}

#[tokio::test]
async fn test_zero_balance_coin_fails() {
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

// Stage 13b: test_gas_smashing_merges_balances deleted — gas
// smashing was a coin-mode-only feature (multiple gas coins
// merged into one to fund larger fees). Balance-mode txs source
// gas from the USDC accumulator, so there are no coins to smash.

#[tokio::test]
async fn test_fee_scales_with_recipients() {
    // More recipients → more fee_units → higher total_fee.
    let (sender, sender_key1): (_, Ed25519KeyPair) = get_key_pair();
    let coin1 = Object::with_id_owner_coin_for_testing(ObjectID::random(), sender, 50_000_000);

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
    // 1 coin + 1 recipient = 2 units
    assert_eq!(effects1.transaction_fee().total_fee, 2 * UNIT_FEE);

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
    // 1 coin + 3 recipients = 4 units
    assert_eq!(effects2.transaction_fee().total_fee, 4 * UNIT_FEE);
}

// Stage 13b: test_pay_all_drains_gas_coin_when_consumed deleted —
// "pay-all" was a coin-mode-only operation (fully drain a coin
// and produce a new one for the recipient). Balance-mode has no
// such semantics; the sender always retains a balance accumulator
// row, drained or not.

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
    let coin_balance = coin.as_coin().unwrap_or(0);
    authority_state.insert_genesis_object(coin).await;

    // Stage 13b: BalanceTransfer debits the SOMA accumulator. Seed
    // the sender's SOMA balance to mirror the gas-coin's balance —
    // tests that previously deducted "transfer_amount" out of the
    // gas coin now deduct from the accumulator.
    authority_state
        .database_for_testing()
        .set_balance(sender, types::object::CoinType::Soma, coin_balance)
        .unwrap();

    let data = crate::authority_test_utils::balance_transfer_data_legacy(recipient, sender, amount, coin_ref);
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

    // Stage 13b: balance-mode multi-recipient transfer.
    use types::object::CoinType;
    use types::transaction::{BalanceTransferArgs, TransactionKind};
    let amount_vec = amounts.unwrap_or_else(|| vec![1; recipients.len()]);
    let transfers: Vec<_> = recipients.into_iter().zip(amount_vec.into_iter()).collect();
    let data = TransactionData::new(
        TransactionKind::BalanceTransfer(BalanceTransferArgs {
            coin_type: CoinType::Soma,
            transfers,
        }),
        sender,
        vec![input_coin_refs[0]],
    );
    let tx = to_sender_signed_transaction(data, &sender_key);
    let txn_result =
        send_and_confirm_transaction(&authority_state, tx).await.map(|(_, effects)| effects);

    TransactionResult { authority_state, txn_result }
}
