use types::{
    base::SomaAddress,
    digests::TransactionDigest,
    effects::ExecutionFailureStatus,
    error::ExecutionResult,
    object::{Object, ObjectID, ObjectRef, ObjectType, Owner},
    temporary_store::TemporaryStore,
    transaction::TransactionKind,
};

use crate::execution::BPS_DENOMINATOR;

use super::{object::check_ownership, FeeCalculator, TransactionExecutor};

/// Executor for coin-related transactions
pub struct CoinExecutor;

impl CoinExecutor {
    pub fn new() -> Self {
        Self {}
    }

    /// Execute a TransferCoin transaction
    fn execute_transfer_coin(
        &self,
        store: &mut TemporaryStore,
        coin_ref: ObjectRef,
        amount: Option<u64>,
        recipient: SomaAddress,
        signer: SomaAddress,
        tx_digest: TransactionDigest,
        value_fee: u64,
    ) -> ExecutionResult<()> {
        let coin_id = coin_ref.0;

        // Get source coin
        let source_object = store
            .read_object(&coin_id)
            .ok_or_else(|| ExecutionFailureStatus::ObjectNotFound { object_id: coin_id })?;

        // Check ownership
        check_ownership(&source_object, signer)?;

        // Check this is a coin object and get balance
        let source_balance = verify_coin(&source_object)?;

        // Check if this coin is the primary gas coin
        let is_gas_coin = store.gas_object_id == Some(coin_id);

        match amount {
            // Pay specific amount
            Some(specific_amount) => {
                // If this is a gas coin, we need to ensure there's enough left for fees
                if is_gas_coin {
                    // For write fee, we'll create one new coin and update the source
                    let write_fee = self.calculate_operation_fee(store, 2);

                    // Total fee needed
                    let total_fee = value_fee + write_fee;

                    // Check sufficient balance for both the transfer and the fee
                    if source_balance < specific_amount + total_fee {
                        return Err(ExecutionFailureStatus::InsufficientCoinBalance.into());
                    }

                    // Calculate remaining balance after transfer and fees
                    let remaining_balance = source_balance - specific_amount;

                    // Create new coin for recipient
                    let new_coin = Object::new_coin(
                        ObjectID::derive_id(tx_digest, store.next_creation_num()),
                        specific_amount,
                        Owner::AddressOwner(recipient),
                        tx_digest,
                    );
                    store.create_object(new_coin);

                    // Update source coin with remaining balance (which includes the fee)
                    let mut updated_source = source_object.clone();
                    updated_source.update_coin_balance(remaining_balance);
                    store.mutate_input_object(updated_source);
                } else {
                    // Not a gas coin, proceed normally
                    // Check sufficient balance
                    if source_balance < specific_amount {
                        return Err(ExecutionFailureStatus::InsufficientCoinBalance.into());
                    }

                    // Calculate remaining balance after transfer
                    let remaining_balance = source_balance - specific_amount;

                    // If transferring the entire balance, just change ownership
                    if specific_amount == source_balance {
                        let mut updated_source = source_object.clone();
                        updated_source.owner = Owner::AddressOwner(recipient);
                        store.mutate_input_object(updated_source);
                        return Ok(());
                    }

                    // Create new coin for recipient
                    let new_coin = Object::new_coin(
                        ObjectID::derive_id(tx_digest, store.next_creation_num()),
                        specific_amount,
                        Owner::AddressOwner(recipient),
                        tx_digest,
                    );
                    store.create_object(new_coin);

                    // Update source coin
                    let mut updated_source = source_object.clone();
                    updated_source.update_coin_balance(remaining_balance);
                    store.mutate_input_object(updated_source);
                }
            }

            // Pay all (transfer the entire coin)
            None => {
                if is_gas_coin {
                    // Note: Base fee has already been deducted during gas preparation
                    // We only need to account for value and operation fees

                    // Use the FeeCalculator trait methods to calculate fees

                    // For write fee, we know we'll create one new object and update the original
                    let write_fee = self.calculate_operation_fee(store, 2);

                    // Total fee is value_fee + write_fee (base fee already deducted)
                    let remaining_fee = write_fee + value_fee;

                    // Ensure there's enough balance after previous fee deduction
                    if source_balance <= remaining_fee {
                        return Err(ExecutionFailureStatus::InsufficientCoinBalance.into());
                    }

                    // Calculate amount to transfer (total - remaining fee)
                    let transfer_amount = source_balance - remaining_fee;

                    // Create new coin for recipient with (balance - fee)
                    let new_coin = Object::new_coin(
                        ObjectID::derive_id(tx_digest, store.next_creation_num()),
                        transfer_amount,
                        Owner::AddressOwner(recipient),
                        tx_digest,
                    );
                    store.create_object(new_coin);

                    // Update source coin to just keep the fee amount
                    let mut updated_source = source_object.clone();
                    updated_source.update_coin_balance(remaining_fee);
                    store.mutate_input_object(updated_source);
                } else {
                    // Not a gas coin, just change ownership of entire coin
                    let mut updated_source = source_object.clone();
                    updated_source.owner = Owner::AddressOwner(recipient);
                    store.mutate_input_object(updated_source);
                }
            }
        }

        Ok(())
    }

    /// Execute a PayCoins transaction
    fn execute_pay_coins(
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
            && coin_refs
                .iter()
                .any(|coin_ref| coin_ref.0 == gas_object_id.unwrap());

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

                    // If this is the gas coin, keep track of it separately
                    if Some(coin_id) == store.gas_object_id {
                        gas_coin_data = Some((coin_id, balance));
                    }

                    total_available += balance;
                    available_coins.push((coin_id, balance));
                }

                // Calculate total needed
                let total_payments: u64 = specific_amounts.iter().sum();

                // Calculate estimated remaining fees (base fee already deducted)

                // Write fee: 1 update for gas coin + 1 for each recipient
                let write_fee = self.calculate_operation_fee(store, 1 + recipients.len() as u64);

                // Total remaining fee
                let remaining_fee = value_fee + write_fee;

                // Check sufficient balance for payments + fees
                if total_available < total_payments + remaining_fee {
                    return Err(ExecutionFailureStatus::InsufficientCoinBalance.into());
                }

                // STEP 2: Create new coins for each recipient
                for (amount, recipient) in specific_amounts.iter().zip(recipients.iter()) {
                    let new_coin = Object::new_coin(
                        ObjectID::derive_id(tx_digest, store.next_creation_num()),
                        *amount,
                        Owner::AddressOwner(*recipient),
                        tx_digest,
                    );
                    store.create_object(new_coin);
                }

                // STEP 3: Update primary coin
                let remaining_balance = total_available - total_payments;

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
                    total_available += gas_balance;

                    // Process other coins
                    let mut other_coins = Vec::new();
                    for coin_ref in coin_refs.iter() {
                        if coin_ref.0 != gas_id && !store.is_deleted(&coin_ref.0) {
                            let coin_obj = store.read_object(&coin_ref.0).unwrap();
                            check_ownership(&coin_obj, signer)?;
                            let balance = verify_coin(&coin_obj)?;
                            total_available += balance;
                            other_coins.push((coin_ref.0, balance));
                        }
                    }

                    // Calculate value fee using the trait method

                    // For write fee, we know we'll create one new object and keep gas coin
                    let write_fee = self.calculate_operation_fee(store, 2);

                    // Total remaining fee (base fee already deducted)
                    let remaining_fee = value_fee + write_fee;

                    // Ensure there's enough balance to pay the fee
                    if total_available <= remaining_fee {
                        return Err(ExecutionFailureStatus::InsufficientCoinBalance.into());
                    }

                    // Calculate amount to transfer (total - fee)
                    let transfer_amount = total_available - remaining_fee;

                    // Create new coin for recipient with (total balance - fee)
                    let new_coin = Object::new_coin(
                        ObjectID::derive_id(tx_digest, store.next_creation_num()),
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
                                ExecutionFailureStatus::ObjectNotFound {
                                    object_id: first_coin_id,
                                }
                            })?;

                        // Check ownership of first coin
                        check_ownership(&first_coin_object, signer)?;

                        // Get balance of first coin
                        let mut total_balance = verify_coin(&first_coin_object)?;

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
                            total_balance += balance;

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
            TransactionKind::TransferCoin {
                coin,
                amount,
                recipient,
            } => self.execute_transfer_coin(
                store, coin, amount, recipient, signer, tx_digest, value_fee,
            ),
            TransactionKind::PayCoins {
                coins,
                amounts,
                recipients,
            } => self.execute_pay_coins(
                store, coins, amounts, recipients, signer, tx_digest, value_fee,
            ),
            _ => Err(ExecutionFailureStatus::InvalidTransactionType),
        }
    }
}

impl FeeCalculator for CoinExecutor {
    fn calculate_value_fee(&self, store: &TemporaryStore, kind: &TransactionKind) -> u64 {
        let value_fee_bps = store.fee_parameters.value_fee_bps;

        match kind {
            TransactionKind::TransferCoin { coin, amount, .. } => {
                let transfer_amount = if let Some(specific_amount) = amount {
                    *specific_amount
                } else {
                    // For full transfer, get the actual coin balance
                    store
                        .read_object(&coin.0)
                        .and_then(|obj| obj.as_coin())
                        .unwrap_or(0)
                };

                if transfer_amount == 0 {
                    return 0;
                }

                (transfer_amount * value_fee_bps) / BPS_DENOMINATOR
            }

            TransactionKind::PayCoins { coins, amounts, .. } => {
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

                (total_amount * value_fee_bps) / BPS_DENOMINATOR
            }

            _ => 0,
        }
    }
}

/// Verifies an object is a coin and returns its balance
fn verify_coin(object: &Object) -> Result<u64, ExecutionFailureStatus> {
    object
        .as_coin()
        .ok_or_else(|| ExecutionFailureStatus::InvalidObjectType {
            object_id: object.id(),
            expected_type: ObjectType::Coin,
            actual_type: object.type_().clone(),
        })
}
