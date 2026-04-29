// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use types::base::SomaAddress;
use types::effects::ExecutionFailureStatus;
use types::error::{ExecutionResult, SomaError};
use types::object::{ObjectID, ObjectRef};
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
/// For system transactions, this does nothing.
/// For user transactions:
///   1. Smashes gas coins into one primary coin.
///   2. Computes total fee = `unit_fee * executor.fee_units(...)`.
///   3. Deducts the fee from the gas coin in one shot.
///
/// On insufficient balance, drains whatever's available and returns
/// `InsufficientGas` so the caller can record the failed effect.
pub fn prepare_gas(
    temporary_store: &mut TemporaryStore,
    kind: &TransactionKind,
    signer: &SomaAddress,
    gas_payment: Vec<ObjectRef>,
    executor: &dyn TransactionExecutor,
) -> Result<GasPreparationResult, (ExecutionFailureStatus, TransactionFee)> {
    // Skip gas handling for system transactions
    if kind.is_system_tx() {
        return Ok(GasPreparationResult {
            primary_gas_id: None,
            transaction_fee: TransactionFee::default(),
        });
    }

    // Smash gas coins and get primary gas ID
    let gas_id = match smash_gas_coins(temporary_store, signer, gas_payment) {
        Ok(id) => id,
        Err(err) => {
            return Err((err, TransactionFee::default()));
        }
    };

    // Set gas object ID in temporary store
    let primary_gas_id = Some(gas_id);
    temporary_store.gas_object_id = primary_gas_id;

    // Compute total fee = unit_fee × fee_units
    let unit_fee = temporary_store.fee_parameters.unit_fee;
    let units = executor.fee_units(temporary_store, kind) as u64;
    let total_fee = unit_fee.saturating_mul(units);

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
        // Still need to check ownership and verify it's a coin
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

        // Verify it's a coin
        let _balance = primary_gas_obj.as_coin().ok_or_else(|| {
            ExecutionFailureStatus::SomaError(SomaError::from("Gas object is not a coin"))
        })?;

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

    // Get balance of primary gas object
    let primary_balance = primary_gas_obj.as_coin().ok_or_else(|| {
        ExecutionFailureStatus::SomaError(SomaError::from("Gas object is not a coin"))
    })?;

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

        // Verify it's a coin and add balance
        let balance = gas_obj.as_coin().ok_or_else(|| {
            ExecutionFailureStatus::SomaError(SomaError::from("Gas object is not a coin"))
        })?;

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
