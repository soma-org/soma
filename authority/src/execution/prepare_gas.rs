use tracing::info;
use types::{
    base::SomaAddress,
    effects::ExecutionFailureStatus,
    error::{ExecutionResult, SomaError},
    object::{ObjectID, ObjectRef},
    temporary_store::TemporaryStore,
    transaction::TransactionKind,
    tx_fee::TransactionFee,
};

use super::TransactionExecutor;

/// Result of gas preparation
pub(crate) struct GasPreparationResult {
    /// Primary gas object ID (if gas handling is enabled)
    primary_gas_id: Option<ObjectID>,
    /// Transaction fee information (if base fee was deducted)
    pub transaction_fee: Option<TransactionFee>,
    /// Amount of base fee that was deducted
    base_fee_deducted: u64,
    /// Pre-calculated value fee to ensure consistency
    pub value_fee: u64,
}

/// Prepares gas for a transaction
///
/// For system transactions, this does nothing.
/// For regular transactions, it smashes gas coins and attempts to deduct base fee.
pub fn prepare_gas(
    temporary_store: &mut TemporaryStore,
    kind: &TransactionKind,
    signer: &SomaAddress,
    gas_payment: Vec<ObjectRef>,
    executor: &dyn TransactionExecutor,
) -> Result<GasPreparationResult, (ExecutionFailureStatus, Option<TransactionFee>)> {
    // Skip gas handling for system transactions
    if kind.is_system_tx() {
        return Ok(GasPreparationResult {
            primary_gas_id: None,
            transaction_fee: None,
            base_fee_deducted: 0,
            value_fee: 0,
        });
    }

    // Smash gas coins and get primary gas ID
    let gas_id = match smash_gas_coins(temporary_store, signer, gas_payment) {
        Ok(id) => id,
        Err(err) => {
            return Err((err, None));
        }
    };

    // Set gas object ID in temporary store
    let primary_gas_id = Some(gas_id);
    temporary_store.gas_object_id = primary_gas_id;

    // Deduct base fee for DOS protection
    let base_fee = executor.base_fee();

    // Get gas object with merged balance
    let gas_obj = temporary_store.read_object(&gas_id).unwrap();
    let gas_balance = gas_obj.as_coin().unwrap();

    // Check if there's enough for base fee
    if gas_balance < base_fee {
        // Not enough for base fee - take what we can and fail
        if gas_balance > 0 {
            // Deduct whatever is available
            let gas_ref = gas_obj.compute_object_reference();
            let partial_fee_obj = TransactionFee::new(gas_balance, 0, 0, gas_ref);

            // This should always succeed since we're taking at most the available balance
            if let Ok(_) = deduct_gas_fee(temporary_store, &partial_fee_obj) {
                // Create transaction fee with what we could deduct
                let updated_gas_ref = if gas_balance == gas_obj.as_coin().unwrap() {
                    // Gas object was deleted, use the original reference
                    gas_ref
                } else if let Some(updated_obj) = temporary_store.read_object(&gas_id) {
                    // Gas object was updated, get new reference
                    updated_obj.compute_object_reference()
                } else {
                    // Gas object was deleted, use the original reference
                    gas_ref
                };

                let partial_fee = Some(TransactionFee::new(gas_balance, 0, 0, updated_gas_ref));

                return Err((ExecutionFailureStatus::InsufficientGas, partial_fee));
            }
        }

        return Err((ExecutionFailureStatus::InsufficientGas, None));
    }

    // Sufficient gas for base fee - deduct it
    let gas_ref = gas_obj.compute_object_reference();
    let base_fee_obj = TransactionFee::new(base_fee, 0, 0, gas_ref);

    // Calculate value fee before deducting any gas
    let value_fee = executor.calculate_value_fee(temporary_store, kind);

    match deduct_gas_fee(temporary_store, &base_fee_obj) {
        Ok(_) => {
            // Base fee deducted successfully
            // Get updated gas object reference
            let updated_gas_obj = temporary_store.read_object(&gas_id).unwrap();
            let updated_gas_ref = updated_gas_obj.compute_object_reference();

            let transaction_fee = Some(TransactionFee::new(base_fee, 0, 0, updated_gas_ref));

            Ok(GasPreparationResult {
                primary_gas_id,
                transaction_fee,
                base_fee_deducted: base_fee,
                value_fee,
            })
        }
        Err(err) => {
            // This shouldn't happen since we checked the balance
            Err((err, None))
        }
    }
}

fn smash_gas_coins(
    store: &mut TemporaryStore,
    signer: &SomaAddress,
    gas_payment: Vec<ObjectRef>,
) -> ExecutionResult<ObjectID> {
    if gas_payment.is_empty() {
        return Err(ExecutionFailureStatus::SomaError(SomaError::from(
            "No gas payment provided",
        )));
    }

    let primary_gas_id = gas_payment[0].0;

    // Skip if only one gas coin
    if gas_payment.len() == 1 {
        // Still need to check ownership and verify it's a coin
        let primary_gas_obj = store.read_object(&primary_gas_id).ok_or_else(|| {
            ExecutionFailureStatus::ObjectNotFound {
                object_id: primary_gas_id,
            }
        })?;

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

    let primary_gas_obj = store.read_object(&primary_gas_id).ok_or_else(|| {
        ExecutionFailureStatus::ObjectNotFound {
            object_id: primary_gas_id,
        }
    })?;

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

        total_balance += balance;

        // Delete this gas coin (we'll merge into the first)
        store.delete_input_object(&gas_id);
    }

    // Update the primary gas coin with total balance
    let mut updated_gas = primary_gas_obj.clone();
    updated_gas.update_coin_balance(total_balance);
    store.mutate_input_object(updated_gas);

    Ok(primary_gas_id)
}

/// Calculates and deducts operation and value fees after successful execution
///
/// Returns the final transaction fee that includes both base fee and remaining fees.
pub fn calculate_and_deduct_remaining_fees(
    temporary_store: &mut TemporaryStore,
    kind: &TransactionKind,
    executor: &dyn TransactionExecutor,
    gas_result: &GasPreparationResult,
) -> Result<Option<TransactionFee>, ExecutionFailureStatus> {
    // Skip for system transactions or if no gas ID
    if kind.is_system_tx() || gas_result.primary_gas_id.is_none() {
        return Ok(gas_result.transaction_fee.clone());
    }

    let gas_id = gas_result.primary_gas_id.unwrap();

    // Use pre-calculated value fee instead of recalculating
    let value_fee = gas_result.value_fee;

    // Calculate operation fee (without recalculating value fee)
    let operation_fee = temporary_store.execution_results.written_objects.len() as u64
        * executor.write_fee_per_object();

    // Get gas object
    let gas_obj = match temporary_store.read_object(&gas_id) {
        Some(obj) => obj,
        None => return Err(ExecutionFailureStatus::ObjectNotFound { object_id: gas_id }),
    };

    let gas_object_ref = gas_obj.compute_object_reference();

    // Create TransactionFee with pre-calculated value fee
    let remaining_fee = TransactionFee::new(
        0, // Base fee already deducted
        operation_fee,
        value_fee,
        gas_object_ref,
    );

    // Attempt to deduct the remaining fee
    match deduct_gas_fee(temporary_store, &remaining_fee) {
        Ok(_) => {
            // Combine base fee with operation and value fees for reporting
            let final_fee = merge_fee_components(gas_result.base_fee_deducted, remaining_fee);
            Ok(Some(final_fee))
        }
        Err(err) => Err(err),
    }
}

fn deduct_gas_fee(store: &mut TemporaryStore, fee: &TransactionFee) -> ExecutionResult<u64> {
    let gas_id = fee.gas_object_ref.0;
    let gas_obj = store
        .read_object(&gas_id)
        .ok_or_else(|| ExecutionFailureStatus::ObjectNotFound { object_id: gas_id })?;

    let current_balance = gas_obj.as_coin().ok_or_else(|| {
        ExecutionFailureStatus::SomaError(SomaError::from("Gas object is not a coin"))
    })?;

    // Check sufficient balance
    if current_balance < fee.total_fee {
        return Err(ExecutionFailureStatus::InsufficientGas);
    }

    // Deduct fee from gas object
    let new_balance = current_balance - fee.total_fee;

    // Delete object if balance is zero, otherwise update it
    if new_balance == 0 {
        store.delete_input_object(&gas_id);
    } else {
        let mut updated_gas = gas_obj.clone();
        updated_gas.update_coin_balance(new_balance);
        store.mutate_input_object(updated_gas);
    }

    Ok(fee.total_fee)
}

// Helper function to merge fee components for reporting
fn merge_fee_components(base_fee: u64, remaining_fee: TransactionFee) -> TransactionFee {
    TransactionFee::new(
        base_fee,
        remaining_fee.operation_fee,
        remaining_fee.value_fee,
        remaining_fee.gas_object_ref,
    )
}
