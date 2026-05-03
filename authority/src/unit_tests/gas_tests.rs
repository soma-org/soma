// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Tests for the unit-fee gas model in balance-mode (Stage 13c).
//!
//! Each tx pays `unit_fee × executor.fee_units(...)` out of the
//! sender's USDC accumulator. For BalanceTransfer ops, the executor
//! charges `1 + transfers.len()` units; more recipients raise the
//! fee linearly.

use std::sync::Arc;

use fastcrypto::ed25519::Ed25519KeyPair;
use types::base::{SomaAddress, dbg_addr};
use types::crypto::{SomaKeyPair, get_key_pair};
use types::effects::{
    ExecutionFailureStatus, ExecutionStatus, SignedTransactionEffects, TransactionEffectsAPI,
};
use types::error::SomaError;
use types::object::CoinType;
use types::transaction::{BalanceTransferArgs, TransactionData, TransactionKind};
use types::unit_tests::utils::to_sender_signed_transaction;

use crate::authority::AuthorityState;
use crate::authority_test_utils::{seed_balance_mode_funds, send_and_confirm_transaction};
use crate::test_authority_builder::TestAuthorityBuilder;

// Default `unit_fee` from protocol config v1.
const UNIT_FEE: u64 = 1000;

#[tokio::test]
async fn test_fee_deducted_on_success() {
    let (sender, sender_key): (_, Ed25519KeyPair) = get_key_pair();
    let starting_usdc = 10_000_000u64;
    let starting_soma = 10_000_000u64;
    let recipient = dbg_addr(1);
    let transfer_amount = 1000u64;

    let res = execute_balance_transfer(
        starting_soma,
        starting_usdc,
        vec![(recipient, transfer_amount)],
        sender,
        SomaKeyPair::Ed25519(sender_key),
    )
    .await;

    let effects = res.txn_result.unwrap().into_data();
    assert_eq!(*effects.status(), ExecutionStatus::Success);

    let fee = effects.transaction_fee();
    // BalanceTransfer fee_units = 1 + transfers.len() = 2 with one recipient.
    assert_eq!(fee.total_fee, 2 * UNIT_FEE);

    // Stage 13c: gas comes from the USDC accumulator, transfer
    // amount comes from the SOMA accumulator. Flush settlement and
    // verify both debits landed.
    let tx_digest = effects.transaction_digest();
    let epoch = res.authority_state.epoch_store_for_testing().epoch();
    let batch = res.authority_state.get_cache_commit().build_db_batch(epoch, &[*tx_digest]);
    res.authority_state
        .get_cache_commit()
        .commit_transaction_outputs(epoch, batch, &[*tx_digest]);

    let store = res.authority_state.database_for_testing();
    let post_usdc = store.get_balance(sender, CoinType::Usdc).unwrap();
    let post_soma = store.get_balance(sender, CoinType::Soma).unwrap();
    assert_eq!(post_usdc, starting_usdc - fee.total_fee, "USDC must drop by fee");
    assert_eq!(
        post_soma,
        starting_soma - transfer_amount,
        "SOMA must drop by transfer amount",
    );
}

#[tokio::test]
async fn test_insufficient_gas_drains_what_it_can() {
    // Sender USDC < required unit fee → tx fails with InsufficientGas
    // and no fee is debited (prepare_gas returns Err before emitting
    // the Withdraw event).
    let (sender, sender_key): (_, Ed25519KeyPair) = get_key_pair();
    let recipient = dbg_addr(1);

    let res = execute_balance_transfer(
        10_000_000,
        500, // USDC balance < UNIT_FEE × 2 = 2000
        vec![(recipient, 100)],
        sender,
        SomaKeyPair::Ed25519(sender_key),
    )
    .await;

    let effects = res.txn_result.unwrap().into_data();
    assert_eq!(
        *effects.status(),
        ExecutionStatus::Failure { error: ExecutionFailureStatus::InsufficientGas },
    );

    // Stage 13c: no partial drain — prepare_gas refuses to emit
    // a Withdraw event when balance < total_fee, so the fee on
    // effects is zero.
    assert_eq!(effects.transaction_fee().total_fee, 0);
}

#[tokio::test]
async fn test_zero_balance_coin_fails() {
    let (sender, sender_key): (_, Ed25519KeyPair) = get_key_pair();
    let recipient = dbg_addr(1);

    let res = execute_balance_transfer(
        10_000_000,
        0,
        vec![(recipient, 0)],
        sender,
        SomaKeyPair::Ed25519(sender_key),
    )
    .await;

    let effects = res.txn_result.unwrap().into_data();
    assert_eq!(
        *effects.status(),
        ExecutionStatus::Failure { error: ExecutionFailureStatus::InsufficientGas },
    );
}

#[tokio::test]
async fn test_fee_scales_with_recipients() {
    // More recipients → more fee_units → higher total_fee.
    let (sender1, sender_key1): (_, Ed25519KeyPair) = get_key_pair();
    let res1 = execute_balance_transfer(
        50_000_000,
        50_000_000,
        vec![(dbg_addr(1), 100)],
        sender1,
        SomaKeyPair::Ed25519(sender_key1),
    )
    .await;
    let effects1 = res1.txn_result.unwrap().into_data();
    assert_eq!(*effects1.status(), ExecutionStatus::Success);
    // 1 base + 1 recipient = 2 units
    assert_eq!(effects1.transaction_fee().total_fee, 2 * UNIT_FEE);

    let (sender2, sender_key2): (_, Ed25519KeyPair) = get_key_pair();
    let res2 = execute_balance_transfer(
        50_000_000,
        50_000_000,
        vec![(dbg_addr(1), 100), (dbg_addr(2), 100), (dbg_addr(3), 100)],
        sender2,
        SomaKeyPair::Ed25519(sender_key2),
    )
    .await;
    let effects2 = res2.txn_result.unwrap().into_data();
    assert_eq!(*effects2.status(), ExecutionStatus::Success);
    // 1 base + 3 recipients = 4 units
    assert_eq!(effects2.transaction_fee().total_fee, 4 * UNIT_FEE);
}

// =============================================================================
// Helpers
// =============================================================================

struct TransactionResult {
    authority_state: Arc<AuthorityState>,
    txn_result: Result<SignedTransactionEffects, SomaError>,
}

async fn execute_balance_transfer(
    soma_balance: u64,
    usdc_balance: u64,
    transfers: Vec<(SomaAddress, u64)>,
    sender: SomaAddress,
    sender_key: SomaKeyPair,
) -> TransactionResult {
    let authority_state = TestAuthorityBuilder::new().build().await;
    seed_balance_mode_funds(&authority_state, sender, soma_balance, usdc_balance);

    let data = TransactionData::new(
        TransactionKind::BalanceTransfer(BalanceTransferArgs {
            coin_type: CoinType::Soma,
            transfers,
        }),
        sender,
        vec![],
    );
    let tx = to_sender_signed_transaction(data, &sender_key);
    let txn_result =
        send_and_confirm_transaction(&authority_state, tx).await.map(|(_, effects)| effects);

    TransactionResult { authority_state, txn_result }
}
