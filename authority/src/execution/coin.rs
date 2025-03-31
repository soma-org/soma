use types::{
    base::SomaAddress,
    digests::TransactionDigest,
    effects::ExecutionFailureStatus,
    error::ExecutionResult,
    object::{Object, ObjectID, ObjectRef, ObjectType, Owner},
    temporary_store::TemporaryStore,
    transaction::TransactionKind,
};

use super::{object::check_ownership, TransactionExecutor};

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

        match amount {
            // Pay specific amount
            Some(specific_amount) => {
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
                let new_coin =
                    Object::new_coin(specific_amount, Owner::AddressOwner(recipient), tx_digest);
                store.create_object(new_coin);

                // Update source coin
                let mut updated_source = source_object.clone();
                updated_source.update_coin_balance(remaining_balance);
                store.mutate_input_object(updated_source);
            }

            // Pay all (transfer the entire coin)
            None => {
                // Just change the ownership of the existing coin
                let mut updated_source = source_object.clone();
                updated_source.owner = Owner::AddressOwner(recipient);
                store.mutate_input_object(updated_source);
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
                let mut primary_coin_id: Option<ObjectID> = None;
                let mut coins_to_delete = Vec::new();

                for coin_ref in &coin_refs {
                    let coin_id = coin_ref.0;
                    let coin_object = store.read_object(&coin_id).ok_or_else(|| {
                        ExecutionFailureStatus::ObjectNotFound { object_id: coin_id }
                    })?;

                    // Check ownership
                    check_ownership(&coin_object, signer)?;

                    // Check this is a coin object
                    let balance = verify_coin(&coin_object)?;

                    total_available += balance;

                    // Set first coin as primary coin
                    if primary_coin_id.is_none() {
                        primary_coin_id = Some(coin_id);
                    } else {
                        // Mark all non-primary coins for deletion
                        coins_to_delete.push(coin_id);
                    }
                }

                // Calculate total needed
                let total_payments: u64 = specific_amounts.iter().sum();

                // Check sufficient balance
                if total_available < total_payments {
                    return Err(ExecutionFailureStatus::InsufficientCoinBalance.into());
                }

                // STEP 2: Create new coins for each recipient
                for (amount, recipient) in specific_amounts.iter().zip(recipients.iter()) {
                    let new_coin =
                        Object::new_coin(*amount, Owner::AddressOwner(*recipient), tx_digest);
                    store.create_object(new_coin);
                }

                // STEP 3: Update primary coin
                let remaining_balance = total_available - total_payments;

                if let Some(primary_id) = primary_coin_id {
                    if remaining_balance > 0 {
                        // Update primary coin with remaining balance
                        let primary_object = store.read_object(&primary_id).unwrap();
                        let mut updated_primary = primary_object.clone();
                        updated_primary.update_coin_balance(remaining_balance);
                        store.mutate_input_object(updated_primary);
                    } else {
                        // Delete primary coin if empty
                        store.delete_input_object(&primary_id);
                    }
                }

                // STEP 4: Delete all other coins
                for coin_id in coins_to_delete {
                    store.delete_input_object(&coin_id);
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

                // Pay all coins to this recipient
                if coin_refs.len() == 1 {
                    // For a single coin, just change ownership
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
                    let first_coin_object = store.read_object(&first_coin_id).ok_or_else(|| {
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

        Ok(())
    }
}

impl TransactionExecutor for CoinExecutor {
    fn execute(
        &self,
        store: &mut TemporaryStore,
        signer: SomaAddress,
        kind: TransactionKind,
        tx_digest: TransactionDigest,
    ) -> ExecutionResult<()> {
        match kind {
            TransactionKind::TransferCoin {
                coin,
                amount,
                recipient,
            } => self.execute_transfer_coin(store, coin, amount, recipient, signer, tx_digest),
            TransactionKind::PayCoins {
                coins,
                amounts,
                recipients,
            } => self.execute_pay_coins(store, coins, amounts, recipients, signer, tx_digest),
            _ => Err(ExecutionFailureStatus::InvalidTransactionType),
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
