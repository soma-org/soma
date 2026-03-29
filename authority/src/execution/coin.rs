// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use types::base::SomaAddress;
use types::digests::TransactionDigest;
use types::effects::ExecutionFailureStatus;
use types::error::ExecutionResult;
use types::object::{CoinType, Object, ObjectID, ObjectRef, ObjectType, Owner};
use types::temporary_store::TemporaryStore;
use types::transaction::TransactionKind;

use super::object::check_ownership;
use super::{FeeCalculator, TransactionExecutor, bps_mul, checked_add, checked_sub, checked_sum};

/// Executor for coin-related transactions
pub struct CoinExecutor;

impl CoinExecutor {
    pub fn new() -> Self {
        Self {}
    }

    /// Execute a Transfer transaction (unified from TransferCoin and PayCoins)
    fn execute_transfer(
        &self,
        store: &mut TemporaryStore,
        coin_refs: Vec<ObjectRef>,
        amounts: Option<Vec<u64>>,
        recipients: Vec<SomaAddress>,
        signer: SomaAddress,
        tx_digest: TransactionDigest,
        value_fee: u64,
    ) -> ExecutionResult<()> {
        if coin_refs.is_empty() {
            return Err(ExecutionFailureStatus::InvalidArguments {
                reason: "Must provide at least one coin".to_string(),
            }
            .into());
        }

        if recipients.is_empty() {
            return Err(ExecutionFailureStatus::InvalidArguments {
                reason: "Must provide at least one recipient".to_string(),
            }
            .into());
        }

        // Check for primary gas coin
        let gas_object_id = store.gas_object_id;
        let has_gas_coin = gas_object_id.is_some()
            && coin_refs.iter().any(|coin_ref| coin_ref.0 == gas_object_id.unwrap());

        match amounts {
            // Specific amounts provided
            Some(specific_amounts) => {
                // Validate args
                if specific_amounts.len() != recipients.len() {
                    return Err(ExecutionFailureStatus::InvalidArguments {
                        reason: format!(
                            "Amounts and recipients must match. Got {} amounts and {} recipients",
                            specific_amounts.len(),
                            recipients.len()
                        ),
                    }
                    .into());
                }

                // STEP 1: Calculate total available balance
                let mut total_available: u64 = 0;
                let mut available_coins = Vec::new();
                let mut gas_coin_data = None;
                let mut resolved_coin_type: Option<CoinType> = None;

                for coin_ref in &coin_refs {
                    let coin_id = coin_ref.0;

                    // Check if the id was already merged by smash_gas
                    if store.is_deleted(&coin_id) {
                        continue;
                    }

                    let coin_object = store.read_object(&coin_id).ok_or_else(|| {
                        ExecutionFailureStatus::ObjectNotFound { object_id: coin_id }
                    })?;

                    // Check ownership
                    check_ownership(&coin_object, signer)?;

                    // Check this is a coin object
                    let balance = verify_coin(&coin_object)?;

                    // Validate all coins share the same CoinType
                    let ct = coin_object.coin_type().unwrap();
                    if let Some(expected) = &resolved_coin_type {
                        if ct != *expected {
                            return Err(ExecutionFailureStatus::InvalidArguments {
                                reason: format!(
                                    "All coins must be the same type. Expected {:?}, got {:?}",
                                    expected, ct
                                ),
                            }
                            .into());
                        }
                    } else {
                        resolved_coin_type = Some(ct);
                    }

                    // If this is the gas coin, keep track of it separately
                    if Some(coin_id) == store.gas_object_id {
                        gas_coin_data = Some((coin_id, balance));
                    }

                    total_available = checked_add(total_available, balance)?;
                    available_coins.push((coin_id, balance));
                }

                let coin_type = resolved_coin_type.unwrap_or(CoinType::Soma);

                // Calculate total needed
                let total_payments: u64 = checked_sum(specific_amounts.iter().copied())?;

                // Calculate estimated remaining fees (base fee already deducted)

                // Write fee: 1 update for gas coin + 1 for each recipient
                let write_fee = self.calculate_operation_fee(store, 1 + recipients.len() as u64);

                // Total remaining fee
                let remaining_fee = checked_add(value_fee, write_fee)?;

                // Check sufficient balance for payments (+ fees if gas coin is among transfer coins)
                // When the gas coin is separate, remaining fees are deducted from it in
                // post-execution, not from the transfer coins.
                let required = if has_gas_coin {
                    checked_add(total_payments, remaining_fee)?
                } else {
                    total_payments
                };
                if total_available < required {
                    return Err(ExecutionFailureStatus::InsufficientCoinBalance.into());
                }

                // STEP 2: Create new coins for each recipient
                for (amount, recipient) in specific_amounts.iter().zip(recipients.iter()) {
                    let new_coin = Object::new_coin(
                        ObjectID::derive_id(tx_digest, store.next_creation_num()),
                        coin_type,
                        *amount,
                        Owner::AddressOwner(*recipient),
                        tx_digest,
                    );
                    store.create_object(new_coin);
                }

                // STEP 3: Update primary coin
                let remaining_balance = checked_sub(total_available, total_payments)?;

                // Distribute the remaining balance, prioritizing the gas coin
                if let Some((gas_id, _)) = gas_coin_data {
                    // Update the gas coin with remaining balance
                    let gas_obj = store.read_object(&gas_id).unwrap();
                    let mut updated_gas = gas_obj.clone();
                    updated_gas.update_coin_balance(remaining_balance);
                    store.mutate_input_object(updated_gas);

                    // Delete all other coins
                    for (coin_id, _) in available_coins {
                        if coin_id != gas_id {
                            store.delete_input_object(&coin_id);
                        }
                    }
                } else if !available_coins.is_empty() {
                    // No gas coin, use the first available coin for remaining balance
                    let (first_coin_id, _) = available_coins[0];
                    let first_obj = store.read_object(&first_coin_id).unwrap();
                    let mut updated_coin = first_obj.clone();
                    updated_coin.update_coin_balance(remaining_balance);
                    store.mutate_input_object(updated_coin);

                    // Delete all other coins
                    for (coin_id, _) in available_coins.iter().skip(1) {
                        store.delete_input_object(coin_id);
                    }
                } else {
                    // Should never happen as we already checked for available coins above
                    return Err(ExecutionFailureStatus::InsufficientCoinBalance.into());
                }
            }

            // Pay all coins to a single recipient
            None => {
                // Only allow a single recipient for pay-all
                if recipients.len() != 1 {
                    return Err(ExecutionFailureStatus::InvalidArguments {
                        reason: "Pay-all operation only supports a single recipient".to_string(),
                    }
                    .into());
                }

                let recipient = recipients[0];

                if has_gas_coin {
                    // Note: Base fee has already been deducted during gas preparation
                    // We only need to account for value and operation fees

                    // Calculate total available balance across all coins
                    let mut total_available = 0;
                    let gas_id = gas_object_id.unwrap();

                    // First handle the gas coin (which should have already had base fee deducted)
                    let gas_coin = store.read_object(&gas_id).unwrap();
                    check_ownership(&gas_coin, signer)?;
                    let gas_balance = verify_coin(&gas_coin)?;
                    let coin_type = gas_coin.coin_type().unwrap();
                    total_available = checked_add(total_available, gas_balance)?;

                    // Process other coins
                    let mut other_coins = Vec::new();
                    for coin_ref in coin_refs.iter() {
                        if coin_ref.0 != gas_id && !store.is_deleted(&coin_ref.0) {
                            let coin_obj = store.read_object(&coin_ref.0).unwrap();
                            check_ownership(&coin_obj, signer)?;
                            let balance = verify_coin(&coin_obj)?;
                            // Validate matching CoinType
                            let ct = coin_obj.coin_type().unwrap();
                            if ct != coin_type {
                                return Err(ExecutionFailureStatus::InvalidArguments {
                                    reason: format!(
                                        "All coins must be the same type. Expected {:?}, got {:?}",
                                        coin_type, ct
                                    ),
                                }
                                .into());
                            }
                            total_available = checked_add(total_available, balance)?;
                            other_coins.push((coin_ref.0, balance));
                        }
                    }

                    // Calculate value fee using the trait method

                    // For write fee, we know we'll create one new object and keep gas coin
                    let write_fee = self.calculate_operation_fee(store, 2);

                    // Total remaining fee (base fee already deducted)
                    let remaining_fee = checked_add(value_fee, write_fee)?;

                    // Ensure there's enough balance to pay the fee
                    if total_available <= remaining_fee {
                        return Err(ExecutionFailureStatus::InsufficientCoinBalance.into());
                    }

                    // Calculate amount to transfer (total - fee)
                    let transfer_amount = checked_sub(total_available, remaining_fee)?;

                    // Create new coin for recipient with (total balance - fee)
                    let new_coin = Object::new_coin(
                        ObjectID::derive_id(tx_digest, store.next_creation_num()),
                        coin_type,
                        transfer_amount,
                        Owner::AddressOwner(recipient),
                        tx_digest,
                    );
                    store.create_object(new_coin);

                    // Update gas coin to just keep the fee amount
                    let mut updated_gas = gas_coin.clone();
                    updated_gas.update_coin_balance(remaining_fee);
                    store.mutate_input_object(updated_gas);

                    // Delete all other coins
                    for (coin_id, _) in other_coins {
                        store.delete_input_object(&coin_id);
                    }
                } else {
                    // No gas coin involved, just merge and transfer all coins

                    if coin_refs.len() == 1 {
                        // For a single coin without gas responsibility, just change ownership
                        let coin_id = coin_refs[0].0;
                        let coin_object = store.read_object(&coin_id).ok_or_else(|| {
                            ExecutionFailureStatus::ObjectNotFound { object_id: coin_id }
                        })?;

                        // Check ownership
                        check_ownership(&coin_object, signer)?;

                        // Verify it's a coin
                        verify_coin(&coin_object)?;

                        // Transfer ownership
                        let mut updated_coin = coin_object.clone();
                        updated_coin.owner = Owner::AddressOwner(recipient);
                        store.mutate_input_object(updated_coin);
                    } else {
                        // For multiple coins, merge them all into the first coin and transfer
                        let first_coin_id = coin_refs[0].0;
                        let first_coin_object =
                            store.read_object(&first_coin_id).ok_or_else(|| {
                                ExecutionFailureStatus::ObjectNotFound { object_id: first_coin_id }
                            })?;

                        // Check ownership of first coin
                        check_ownership(&first_coin_object, signer)?;

                        // Get balance of first coin
                        let mut total_balance = verify_coin(&first_coin_object)?;
                        let coin_type = first_coin_object.coin_type().unwrap();

                        // Process all other coins
                        for coin_ref in coin_refs.iter().skip(1) {
                            let coin_id = coin_ref.0;

                            // Check if the id was already merged by smash_gas
                            if store.is_deleted(&coin_id) {
                                continue;
                            }

                            let coin_object = store.read_object(&coin_id).ok_or_else(|| {
                                ExecutionFailureStatus::ObjectNotFound { object_id: coin_id }
                            })?;

                            // Check ownership
                            check_ownership(&coin_object, signer)?;

                            // Verify it's a coin and add balance
                            let balance = verify_coin(&coin_object)?;

                            // Validate matching CoinType
                            let ct = coin_object.coin_type().unwrap();
                            if ct != coin_type {
                                return Err(ExecutionFailureStatus::InvalidArguments {
                                    reason: format!(
                                        "All coins must be the same type. Expected {:?}, got {:?}",
                                        coin_type, ct
                                    ),
                                }
                                .into());
                            }

                            total_balance = checked_add(total_balance, balance)?;

                            // Delete this coin (we'll merge into the first)
                            store.delete_input_object(&coin_id);
                        }

                        // Update the first coin with total balance and change ownership
                        let mut updated_first_coin = first_coin_object.clone();
                        updated_first_coin.update_coin_balance(total_balance);
                        updated_first_coin.owner = Owner::AddressOwner(recipient);
                        store.mutate_input_object(updated_first_coin);
                    }
                }
            }
        }

        Ok(())
    }

    /// Execute a MergeCoins transaction — merges all coins into the first one
    fn execute_merge_coins(
        &self,
        store: &mut TemporaryStore,
        coin_refs: Vec<ObjectRef>,
        signer: SomaAddress,
        _tx_digest: TransactionDigest,
    ) -> ExecutionResult<()> {
        if coin_refs.is_empty() {
            return Err(ExecutionFailureStatus::InvalidArguments {
                reason: "Must provide at least one coin".to_string(),
            }
            .into());
        }

        // Get the first coin as the target
        let first_coin_id = coin_refs[0].0;

        // Check if the first coin was already merged by smash_gas
        if store.is_deleted(&first_coin_id) {
            return Err(ExecutionFailureStatus::ObjectNotFound { object_id: first_coin_id }.into());
        }

        let first_coin_object = store.read_object(&first_coin_id).ok_or_else(|| {
            ExecutionFailureStatus::ObjectNotFound { object_id: first_coin_id }
        })?;

        check_ownership(&first_coin_object, signer)?;
        let mut total_balance = verify_coin(&first_coin_object)?;
        let coin_type = first_coin_object.coin_type().unwrap();

        // Merge all other coins into the first
        for coin_ref in coin_refs.iter().skip(1) {
            let coin_id = coin_ref.0;

            if store.is_deleted(&coin_id) {
                continue;
            }

            let coin_object = store.read_object(&coin_id).ok_or_else(|| {
                ExecutionFailureStatus::ObjectNotFound { object_id: coin_id }
            })?;

            check_ownership(&coin_object, signer)?;
            let balance = verify_coin(&coin_object)?;

            // Validate matching CoinType
            let ct = coin_object.coin_type().unwrap();
            if ct != coin_type {
                return Err(ExecutionFailureStatus::InvalidArguments {
                    reason: format!(
                        "All coins must be the same type. Expected {:?}, got {:?}",
                        coin_type, ct
                    ),
                }
                .into());
            }

            total_balance = checked_add(total_balance, balance)?;

            store.delete_input_object(&coin_id);
        }

        // Update the first coin with total balance
        let mut updated_first_coin = first_coin_object.clone();
        updated_first_coin.update_coin_balance(total_balance);
        store.mutate_input_object(updated_first_coin);

        Ok(())
    }
}

impl TransactionExecutor for CoinExecutor {
    fn execute(
        &mut self,
        store: &mut TemporaryStore,
        signer: SomaAddress,
        kind: TransactionKind,
        tx_digest: TransactionDigest,
        value_fee: u64,
    ) -> ExecutionResult<()> {
        match kind {
            TransactionKind::Transfer { coins, amounts, recipients } => {
                self.execute_transfer(
                    store, coins, amounts, recipients, signer, tx_digest, value_fee,
                )
            }
            TransactionKind::MergeCoins { coins } => {
                self.execute_merge_coins(store, coins, signer, tx_digest)
            }
            _ => Err(ExecutionFailureStatus::InvalidTransactionType),
        }
    }
}

impl FeeCalculator for CoinExecutor {
    fn calculate_value_fee(&self, store: &TemporaryStore, kind: &TransactionKind) -> u64 {
        let value_fee_bps = store.fee_parameters.value_fee_bps;

        match kind {
            TransactionKind::Transfer { coins, amounts, .. } => {
                let total_amount: u64 = if let Some(specific_amounts) = amounts {
                    specific_amounts.iter().copied().sum::<u64>()
                } else {
                    // For pay_all, calculate total of actual balances
                    coins
                        .iter()
                        .filter_map(|coin_ref| store.read_object(&coin_ref.0))
                        .filter_map(|obj| obj.as_coin())
                        .sum::<u64>()
                };

                if total_amount == 0 {
                    return 0;
                }

                bps_mul(total_amount, value_fee_bps)
            }

            _ => 0,
        }
    }
}

/// Verifies an object is a coin and returns its balance
fn verify_coin(object: &Object) -> Result<u64, ExecutionFailureStatus> {
    object.as_coin().ok_or_else(|| ExecutionFailureStatus::InvalidObjectType {
        object_id: object.id(),
        expected_type: ObjectType::Coin(CoinType::Soma),
        actual_type: object.type_().clone(),
    })
}
