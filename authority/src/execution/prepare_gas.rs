// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use types::balance::BalanceEvent;
use types::base::SomaAddress;
use types::effects::ExecutionFailureStatus;
use types::error::SomaError;
use types::object::{CoinType, ObjectID};
use types::temporary_store::TemporaryStore;
use types::transaction::TransactionKind;
use types::tx_fee::TransactionFee;

use super::TransactionExecutor;

/// Result of gas preparation.
pub(crate) struct GasPreparationResult {
    /// Stage 13c: always `None` — gas no longer routes through a
    /// coin object. The field is preserved because it is part of
    /// `TransactionEffectsV1::gas_object_index`'s wire layout (the
    /// effects struct still serializes it for backward-compatibility
    /// with existing checkpoint data). A future effects-schema bump
    /// (e.g. `TransactionEffectsV2`) can drop the field cleanly;
    /// changing it now would alter BCS encoding and break the digest.
    pub primary_gas_id: Option<ObjectID>,
    /// Transaction fee that was deducted from the sender's USDC
    /// accumulator via a `BalanceEvent::Withdraw`.
    pub transaction_fee: TransactionFee,
}

/// Prepare gas for a transaction (Stage 13c: balance-mode only).
///
/// All non-system txs MUST submit with an empty `gas_payment` and
/// rely on the sender's USDC accumulator. The reservation pre-pass
/// (Stage 4) gates the tx's admittance based on a pre-read
/// USDC balance; this function emits the matching
/// `BalanceEvent::Withdraw` for the per-tx fee, which the per-commit
/// settlement applies atomically.
///
/// System txs pay no fee and skip the entire flow.
///
/// `sender_usdc_balance` must be `Some` for non-system txs — the
/// caller pre-reads it from the accumulator. Reaching here with a
/// non-empty `gas_payment` (a left-over from the pre-Stage-13c
/// coin-mode path) is now a hard error: the validator should have
/// rejected the tx upstream.
pub fn prepare_gas(
    temporary_store: &mut TemporaryStore,
    kind: &TransactionKind,
    signer: &SomaAddress,
    gas_payment: Vec<types::object::ObjectRef>,
    executor: &dyn TransactionExecutor,
    sender_usdc_balance: Option<u64>,
) -> Result<GasPreparationResult, (ExecutionFailureStatus, TransactionFee)> {
    if kind.is_system_tx() {
        return Ok(GasPreparationResult {
            primary_gas_id: None,
            transaction_fee: TransactionFee::default(),
        });
    }

    if !gas_payment.is_empty() {
        // Stage 13c: coin-mode gas is gone. A tx that submits a
        // non-empty `gas_payment` predates the migration — reject it.
        return Err((
            ExecutionFailureStatus::SomaError(SomaError::from(
                "Stage 13c: gas_payment must be empty (balance-mode gas only)".to_string(),
            )),
            TransactionFee::default(),
        ));
    }

    // Compute total fee = unit_fee × fee_units.
    let unit_fee = temporary_store.fee_parameters.unit_fee;
    let units = executor.fee_units(temporary_store, kind) as u64;
    let total_fee = unit_fee.saturating_mul(units);

    let balance = sender_usdc_balance.ok_or((
        ExecutionFailureStatus::SomaError(SomaError::from(
            "Balance-mode gas requires pre-computed sender USDC balance".to_string(),
        )),
        TransactionFee::default(),
    ))?;

    if balance < total_fee {
        // Underfunded. The reservation pre-pass should have caught
        // this; reaching here indicates a race or missing pre-pass.
        // Don't emit the Withdraw event so settlement doesn't try
        // to debit.
        return Err((ExecutionFailureStatus::InsufficientGas, TransactionFee::default()));
    }

    // Stage 14c.6 (SIP-58 cutover): user-tx executors emit ONLY
    // `AccumulatorWriteV1`. The per-cp SettlementScheduler aggregates
    // these and is the sole driver of CF apply.
    temporary_store.emit_accumulator_event(
        types::effects::object_change::AccumulatorAddress::balance(*signer, CoinType::Usdc),
        types::effects::object_change::AccumulatorOperation::Split,
        total_fee,
    );

    Ok(GasPreparationResult {
        primary_gas_id: None,
        transaction_fee: TransactionFee::new(total_fee),
    })
}
