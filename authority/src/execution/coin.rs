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
        amount: u64,
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

        // Check sufficient balance for amount and gas fee
        let required_balance = amount; // + GAS_FEE;

        if source_balance < required_balance {
            return Err(ExecutionFailureStatus::InsufficientCoinBalance.into());
        }

        // Calculate remaining balance after transfer and fee
        let remaining_balance = source_balance - amount; // - GAS_FEE;

        // OPTIMIZATION 1: If transferring the entire balance (minus gas),
        // don't create a new object, just change ownership
        if amount == source_balance {
            // if amount == source_balance - GAS_FEE {
            let mut updated_source = source_object.clone();
            updated_source.update_coin_balance(amount);
            updated_source.owner = Owner::AddressOwner(recipient);
            store.mutate_input_object(updated_source);
            return Ok(());
        }

        // TODO: OPTIMIZATION 2: Handle dust amounts by including them in the transferred amount
        // if remaining_balance > 0 && remaining_balance < MIN_COIN_AMOUNT && amount > MIN_COIN_AMOUNT
        // {
        //     // Adjust the transfer to include the dust in the new coin
        //     let adjusted_amount = amount + remaining_balance;
        //     let new_coin =
        //         Object::new_coin(adjusted_amount, Owner::AddressOwner(recipient), tx_digest);
        //     store.create_object(new_coin);

        //     // Delete the original coin
        //     store.delete_input_object(&coin_id);
        //     return Ok(());
        // }

        // STANDARD CASE: Create new coin for recipient and update or delete source
        let new_coin = Object::new_coin(amount, Owner::AddressOwner(recipient), tx_digest);
        store.create_object(new_coin);

        // Update source coin or delete if empty
        if remaining_balance > 0 {
            let mut updated_source = source_object.clone();
            updated_source.update_coin_balance(remaining_balance);
            store.mutate_input_object(updated_source);
        } else {
            store.delete_input_object(&coin_id);
        }

        Ok(())
    }

    /// Execute a PayCoins transaction
    fn execute_pay_coins(
        &self,
        store: &mut TemporaryStore,
        coin_refs: Vec<ObjectRef>,
        amounts: Vec<u64>,
        recipients: Vec<SomaAddress>,
        signer: SomaAddress,
        tx_digest: TransactionDigest,
    ) -> ExecutionResult<()> {
        // Validate args
        if amounts.len() != recipients.len() {
            return Err(ExecutionFailureStatus::InvalidArguments {
                reason: format!(
                    "Amounts and recipients must match. Got {} amounts and {} recipients",
                    amounts.len(),
                    recipients.len()
                ),
            }
            .into());
        }

        if coin_refs.is_empty() {
            return Err(ExecutionFailureStatus::InvalidArguments {
                reason: "Must provide at least one coin".to_string(),
            }
            .into());
        }

        // STEP 1: MERGE (conceptually) - Calculate total available balance
        let mut total_available: u64 = 0;
        let mut primary_coin_id: Option<ObjectID> = None;

        // Track which coins to delete
        let mut coins_to_delete = Vec::new();

        for coin_ref in &coin_refs {
            let coin_id = coin_ref.0;
            let coin_object = store
                .read_object(&coin_id)
                .ok_or_else(|| ExecutionFailureStatus::ObjectNotFound { object_id: coin_id })?;

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
        let total_payments: u64 = amounts.iter().sum();
        let total_needed: u64 = total_payments; // + GAS_FEE;

        // Check sufficient balance
        if total_available < total_needed {
            return Err(ExecutionFailureStatus::InsufficientCoinBalance.into());
        }

        // STEP 2: SPLIT & TRANSFER - Create new coins for each recipient
        for (amount, recipient) in amounts.iter().zip(recipients.iter()) {
            let new_coin = Object::new_coin(*amount, Owner::AddressOwner(*recipient), tx_digest);
            store.create_object(new_coin);
        }

        // STEP 3: UPDATE PRIMARY - Update or delete primary coin
        let remaining_balance = total_available - total_needed;

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
