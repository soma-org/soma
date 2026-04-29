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
use super::{TransactionExecutor, checked_add, checked_sub, checked_sum};

/// Executor for coin-related transactions
pub struct CoinExecutor;

impl CoinExecutor {
    pub fn new() -> Self {
        Self {}
    }

    /// Execute a Transfer transaction (unified from TransferCoin and PayCoins).
    ///
    /// Gas fee was already fully deducted from the gas coin in `prepare_gas`,
    /// so this routine just moves balances around.
    fn execute_transfer(
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

                // Calculate total needed (fee already taken from gas coin in prepare_gas).
                let total_payments: u64 = checked_sum(specific_amounts.iter().copied())?;

                if total_available < total_payments {
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
                    // Gas fee already deducted from the gas coin by prepare_gas, so
                    // pay-all just moves the remaining balance to the recipient.
                    let mut total_available = 0;
                    let gas_id = gas_object_id.unwrap();

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

                    if total_available == 0 {
                        return Err(ExecutionFailureStatus::InsufficientCoinBalance.into());
                    }

                    // New coin for recipient with the full remaining balance.
                    let new_coin = Object::new_coin(
                        ObjectID::derive_id(tx_digest, store.next_creation_num()),
                        coin_type,
                        total_available,
                        Owner::AddressOwner(recipient),
                        tx_digest,
                    );
                    store.create_object(new_coin);

                    // Drain the gas coin completely.
                    store.delete_input_object(&gas_id);

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
    fn fee_units(&self, _store: &TemporaryStore, kind: &TransactionKind) -> u32 {
        match kind {
            // Transfer: cost grows with the number of input coins consumed and output
            // coins created. 1 unit per side, minimum 2.
            TransactionKind::Transfer { coins, recipients, .. } => {
                let n = coins.len().saturating_add(recipients.len());
                n.try_into().unwrap_or(u32::MAX)
            }
            // MergeCoins: cost grows with the merge size.
            TransactionKind::MergeCoins { coins } => {
                coins.len().try_into().unwrap_or(u32::MAX)
            }
            _ => 1,
        }
    }

    fn execute(
        &mut self,
        store: &mut TemporaryStore,
        signer: SomaAddress,
        kind: TransactionKind,
        tx_digest: TransactionDigest,
    ) -> ExecutionResult<()> {
        match kind {
            TransactionKind::Transfer { coins, amounts, recipients } => {
                self.execute_transfer(store, coins, amounts, recipients, signer, tx_digest)
            }
            TransactionKind::MergeCoins { coins } => {
                self.execute_merge_coins(store, coins, signer, tx_digest)
            }
            _ => Err(ExecutionFailureStatus::InvalidTransactionType),
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
