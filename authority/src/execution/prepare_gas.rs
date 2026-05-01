// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use types::balance::BalanceEvent;
use types::base::SomaAddress;
use types::effects::ExecutionFailureStatus;
use types::error::{ExecutionResult, SomaError};
use types::object::{CoinType, ObjectID, ObjectRef};
use types::temporary_store::TemporaryStore;
use types::transaction::TransactionKind;
use types::tx_fee::TransactionFee;

use super::TransactionExecutor;

/// Result of gas preparation
pub(crate) struct GasPreparationResult {
    /// Primary gas object ID (if gas handling is enabled)
    pub primary_gas_id: Option<ObjectID>,
    /// Transaction fee that was deducted
    pub transaction_fee: TransactionFee,
}

/// Prepare gas for a transaction.
///
/// Two gas modes are supported:
///
/// **Coin mode** (`!gas_payment.is_empty()`): the legacy path — smash
/// the provided owned USDC coin objects into one primary, deduct the
/// fee in-place, mutate-or-delete the gas coin object.
///
/// **Balance mode** (`gas_payment.is_empty()`, Stage 6c): no owned
/// gas coin. The caller must have pre-read the sender's USDC balance
/// from the accumulator and pass it as `sender_usdc_balance`. The
/// scheduler's reservation pre-pass (Stage 4) and the validator's
/// `is_replay_protected()` check (Stage 5.5c) together guarantee
/// the tx has both the funds and a valid `TransactionExpiration`
/// before this function runs. We compute the fee, verify the
/// pre-read balance covers it, and emit a `BalanceEvent::Withdraw`
/// for the fee amount. Settlement (Stage 6a) actually applies the
/// debit to the on-chain accumulator at commit boundary.
///
/// For system transactions both branches are skipped — system txs
/// pay no fee and have no gas object.
///
/// On insufficient gas, both modes return `InsufficientGas`. Coin
/// mode drains the partial balance to record a failed effect; balance
/// mode emits no Withdraw event (settlement won't see it) so the
/// debit doesn't happen.
pub fn prepare_gas(
    temporary_store: &mut TemporaryStore,
    kind: &TransactionKind,
    signer: &SomaAddress,
    gas_payment: Vec<ObjectRef>,
    executor: &dyn TransactionExecutor,
    sender_usdc_balance: Option<u64>,
) -> Result<GasPreparationResult, (ExecutionFailureStatus, TransactionFee)> {
    // Skip gas handling for system transactions
    if kind.is_system_tx() {
        return Ok(GasPreparationResult {
            primary_gas_id: None,
            transaction_fee: TransactionFee::default(),
        });
    }

    // Compute total fee = unit_fee × fee_units (same calculation in
    // both modes — the protocol charges identically regardless of how
    // the sender funds it).
    let unit_fee = temporary_store.fee_parameters.unit_fee;
    let units = executor.fee_units(temporary_store, kind) as u64;
    let total_fee = unit_fee.saturating_mul(units);

    // Stage 6c: branch on gas mode.
    if gas_payment.is_empty() {
        // Balance mode. Verify the caller pre-read a balance.
        let balance = match sender_usdc_balance {
            Some(b) => b,
            None => {
                // Stateless tx with no balance pre-read is a caller bug —
                // balance-mode prepare_gas must always be invoked with a
                // pre-computed balance. Surface as a structured error
                // rather than panic so the failed-effect path runs.
                return Err((
                    ExecutionFailureStatus::SomaError(SomaError::from(
                        "Balance-mode gas requires pre-computed sender USDC balance".to_string(),
                    )),
                    TransactionFee::default(),
                ));
            }
        };

        if balance < total_fee {
            // Underfunded. The reservation pre-pass should have caught
            // this before execution; reaching here indicates either a
            // race (sender's balance dropped between reservation and
            // execution) or a missing pre-pass. Fail without emitting
            // the Withdraw event so settlement doesn't try to debit.
            return Err((ExecutionFailureStatus::InsufficientGas, TransactionFee::default()));
        }

        // Emit a Withdraw event for the fee amount. Settlement
        // aggregates these per (owner, coin_type) and applies the net
        // delta atomically at commit boundary.
        temporary_store.emit_balance_event(BalanceEvent::withdraw(
            *signer,
            CoinType::Usdc,
            total_fee,
        ));

        return Ok(GasPreparationResult {
            primary_gas_id: None,
            transaction_fee: TransactionFee::new(total_fee),
        });
    }

    // Coin mode (legacy). Smash gas coins and get primary gas ID.
    let gas_id = match smash_gas_coins(temporary_store, signer, gas_payment) {
        Ok(id) => id,
        Err(err) => {
            return Err((err, TransactionFee::default()));
        }
    };

    // Set gas object ID in temporary store
    let primary_gas_id = Some(gas_id);
    temporary_store.gas_object_id = primary_gas_id;

    // Get gas object with merged balance
    let gas_obj = temporary_store.read_object(&gas_id).unwrap();
    let gas_balance = gas_obj.as_coin().unwrap();

    if gas_balance < total_fee {
        // Not enough — take what we can and report InsufficientGas.
        if gas_balance > 0 {
            let partial_fee = TransactionFee::new(gas_balance);
            if deduct_gas_fee(temporary_store, &partial_fee).is_ok() {
                return Err((ExecutionFailureStatus::InsufficientGas, partial_fee));
            }
        }
        return Err((ExecutionFailureStatus::InsufficientGas, TransactionFee::default()));
    }

    let fee = TransactionFee::new(total_fee);
    match deduct_gas_fee(temporary_store, &fee) {
        Ok(_) => Ok(GasPreparationResult { primary_gas_id, transaction_fee: fee }),
        Err(err) => Err((err, TransactionFee::default())),
    }
}

fn smash_gas_coins(
    store: &mut TemporaryStore,
    signer: &SomaAddress,
    gas_payment: Vec<ObjectRef>,
) -> ExecutionResult<ObjectID> {
    if gas_payment.is_empty() {
        return Err(ExecutionFailureStatus::SomaError(SomaError::from("No gas payment provided")));
    }

    let primary_gas_id = gas_payment[0].0;

    // Skip if only one gas coin
    if gas_payment.len() == 1 {
        // Still need to check ownership and verify it's a USDC coin
        let primary_gas_obj = store
            .read_object(&primary_gas_id)
            .ok_or_else(|| ExecutionFailureStatus::ObjectNotFound { object_id: primary_gas_id })?;

        // Verify ownership of primary gas
        if primary_gas_obj.owner().get_owner_address()? != *signer {
            return Err(ExecutionFailureStatus::InvalidOwnership {
                object_id: primary_gas_id,
                expected_owner: *signer,
                actual_owner: primary_gas_obj.owner().get_owner_address().ok(),
            });
        }

        verify_usdc_coin(&primary_gas_obj)?;

        return Ok(primary_gas_id);
    }

    let primary_gas_obj = store
        .read_object(&primary_gas_id)
        .ok_or_else(|| ExecutionFailureStatus::ObjectNotFound { object_id: primary_gas_id })?;

    // Verify ownership of primary gas
    if primary_gas_obj.owner().get_owner_address()? != *signer {
        return Err(ExecutionFailureStatus::InvalidOwnership {
            object_id: primary_gas_id,
            expected_owner: *signer,
            actual_owner: primary_gas_obj.owner().get_owner_address().ok(),
        });
    }

    // Get balance of primary gas object (must be USDC)
    let primary_balance = verify_usdc_coin(&primary_gas_obj)?;

    let mut total_balance = primary_balance;

    // Process additional gas coins
    for gas_ref in gas_payment.iter().skip(1) {
        let gas_id = gas_ref.0;
        let gas_obj = store
            .read_object(&gas_id)
            .ok_or_else(|| ExecutionFailureStatus::ObjectNotFound { object_id: gas_id })?;

        // Verify ownership
        if gas_obj.owner().get_owner_address()? != *signer {
            return Err(ExecutionFailureStatus::InvalidOwnership {
                object_id: gas_id,
                expected_owner: *signer,
                actual_owner: gas_obj.owner().get_owner_address().ok(),
            });
        }

        // Verify it's a USDC coin and add balance
        let balance = verify_usdc_coin(&gas_obj)?;

        total_balance =
            total_balance.checked_add(balance).ok_or(ExecutionFailureStatus::ArithmeticOverflow)?;

        // Delete this gas coin (we'll merge into the first)
        store.delete_input_object(&gas_id);
    }

    // Update the primary gas coin with total balance
    let mut updated_gas = primary_gas_obj.clone();
    updated_gas.update_coin_balance(total_balance);
    store.mutate_input_object(updated_gas);

    Ok(primary_gas_id)
}

/// Verify a gas-payment object is a USDC coin and return its balance.
/// All fees on Soma are paid in USDC.
fn verify_usdc_coin(obj: &types::object::Object) -> ExecutionResult<u64> {
    let balance = obj.as_coin().ok_or_else(|| {
        ExecutionFailureStatus::SomaError(SomaError::from("Gas object is not a coin"))
    })?;
    match obj.coin_type() {
        Some(CoinType::Usdc) => Ok(balance),
        _ => Err(ExecutionFailureStatus::InvalidGasCoinType { object_id: obj.id() }),
    }
}

fn deduct_gas_fee(store: &mut TemporaryStore, fee: &TransactionFee) -> ExecutionResult<u64> {
    let gas_id = store
        .gas_object_id
        .ok_or_else(|| ExecutionFailureStatus::SomaError(SomaError::from("No gas object set")))?;

    let gas_obj = store
        .read_object(&gas_id)
        .ok_or_else(|| ExecutionFailureStatus::ObjectNotFound { object_id: gas_id })?;

    let current_balance = gas_obj.as_coin().ok_or_else(|| {
        ExecutionFailureStatus::SomaError(SomaError::from("Gas object is not a coin"))
    })?;

    if current_balance < fee.total_fee {
        return Err(ExecutionFailureStatus::InsufficientGas);
    }

    let new_balance = current_balance - fee.total_fee;

    if new_balance == 0 {
        // Gas coin fully consumed (e.g. pay-all) — delete it so it appears in
        // effects.deleted() rather than leaving a 0-balance coin on chain.
        store.delete_input_object(&gas_id);
    } else {
        let mut updated_gas = gas_obj.clone();
        updated_gas.update_coin_balance(new_balance);
        store.mutate_input_object(updated_gas);
    }

    Ok(fee.total_fee)
}
