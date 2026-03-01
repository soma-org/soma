// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use types::SYSTEM_STATE_OBJECT_ID;
use types::base::SomaAddress;
use types::digests::TransactionDigest;
use types::effects::ExecutionFailureStatus;
use types::error::{ExecutionResult, SomaError};
use types::object::{Object, ObjectID, ObjectRef, ObjectType, Owner};
use types::system_state::SystemState;
use types::temporary_store::TemporaryStore;
use types::transaction::TransactionKind;

use super::object::check_ownership;
use super::{FeeCalculator, TransactionExecutor, bps_mul, checked_add, checked_sub};

pub struct StakingExecutor;

impl StakingExecutor {
    pub fn new() -> Self {
        Self {}
    }

    /// Execute AddStake transaction with coin handling
    fn execute_add_stake(
        &self,
        store: &mut TemporaryStore,
        signer: SomaAddress,
        address: SomaAddress,
        coin_ref: ObjectRef,
        amount: Option<u64>,
        tx_digest: TransactionDigest,
        value_fee: u64,
    ) -> ExecutionResult<()> {
        let coin_id = coin_ref.0;
        let is_gas_coin = store.gas_object_id == Some(coin_id);

        // Get source coin
        let source_object = store
            .read_object(&coin_id)
            .ok_or_else(|| ExecutionFailureStatus::ObjectNotFound { object_id: coin_id })?;

        // Check ownership
        check_ownership(&source_object, signer)?;

        // Check this is a coin object and get balance
        let source_balance = verify_coin(&source_object)?;

        // Get system state object
        let state_object = store
            .read_object(&SYSTEM_STATE_OBJECT_ID)
            .ok_or_else(|| ExecutionFailureStatus::ObjectNotFound {
                object_id: SYSTEM_STATE_OBJECT_ID,
            })?
            .clone();

        // Deserialize system state
        let mut state = bcs::from_bytes::<SystemState>(state_object.as_inner().data.contents())
            .map_err(|e| {
                ExecutionFailureStatus::SomaError(SomaError::from(format!(
                    "Failed to deserialize system state: {}",
                    e
                )))
            })?;

        match amount {
            // Stake specific amount
            Some(stake_amount) => {
                // If this is a gas coin, we need to ensure there's enough left for fees
                if is_gas_coin {
                    // Write fee accounts for all written objects the pipeline will see:
                    // StakedSoma (created) + gas coin (mutated) + SystemState (mutated) = 3
                    let write_fee = self.calculate_operation_fee(store, 3);

                    // Total fee needed
                    let total_fee = checked_add(value_fee, write_fee)?;

                    // Check sufficient balance for both staking and the fee
                    if source_balance < checked_add(stake_amount, total_fee)? {
                        return Err(ExecutionFailureStatus::InsufficientCoinBalance.into());
                    }

                    // Request to add stake
                    let staked_soma = state.request_add_stake(signer, address, stake_amount)?;

                    // Create StakedSoma object
                    let staked_soma_object = Object::new_staked_soma_object(
                        ObjectID::derive_id(tx_digest, store.next_creation_num()),
                        staked_soma,
                        Owner::AddressOwner(signer),
                        tx_digest,
                    );
                    store.create_object(staked_soma_object);

                    // Calculate remaining balance after staking and fees
                    let remaining_balance = checked_sub(source_balance, stake_amount)?;

                    // Update source coin with remaining balance
                    let mut updated_source = source_object.clone();
                    updated_source.update_coin_balance(remaining_balance);
                    store.mutate_input_object(updated_source);
                } else {
                    // Not a gas coin, proceed normally
                    // Check sufficient balance
                    if source_balance < stake_amount {
                        return Err(ExecutionFailureStatus::InsufficientCoinBalance.into());
                    }

                    // Request to add stake
                    let staked_soma = state.request_add_stake(signer, address, stake_amount)?;

                    // Create StakedSoma object
                    let staked_soma_object = Object::new_staked_soma_object(
                        ObjectID::derive_id(tx_digest, store.next_creation_num()),
                        staked_soma,
                        Owner::AddressOwner(signer),
                        tx_digest,
                    );
                    store.create_object(staked_soma_object);

                    // If staking the entire balance, delete the coin
                    if stake_amount == source_balance {
                        store.delete_input_object(&coin_id);
                    } else {
                        // Otherwise update coin with remaining balance
                        let remaining_balance = checked_sub(source_balance, stake_amount)?;
                        let mut updated_source = source_object.clone();
                        updated_source.update_coin_balance(remaining_balance);
                        store.mutate_input_object(updated_source);
                    }
                }
            }

            // Stake entire coin
            None => {
                let stake_amount;

                if is_gas_coin {
                    // For gas coin, we need to account for fees
                    // Write fee: StakedSoma (created) + gas coin (mutated) + SystemState (mutated) = 3
                    let write_fee = self.calculate_operation_fee(store, 3);

                    // Total fee needed
                    let total_fee = checked_add(value_fee, write_fee)?;

                    // Check sufficient balance
                    if source_balance <= total_fee {
                        return Err(ExecutionFailureStatus::InsufficientCoinBalance.into());
                    }

                    // Calculate amount to stake (total - fee)
                    stake_amount = checked_sub(source_balance, total_fee)?;

                    // Request to add stake
                    let staked_soma = state.request_add_stake(signer, address, stake_amount)?;

                    // Create StakedSoma object
                    let staked_soma_object = Object::new_staked_soma_object(
                        ObjectID::derive_id(tx_digest, store.next_creation_num()),
                        staked_soma,
                        Owner::AddressOwner(signer),
                        tx_digest,
                    );
                    store.create_object(staked_soma_object);

                    // Keep the coin with remaining fee amount — pipeline will deduct fees
                    // from the gas object via calculate_and_deduct_remaining_fees.
                    // Do NOT delete: the gas coin must remain readable for fee deduction.
                    let mut updated_source = source_object.clone();
                    updated_source.update_coin_balance(total_fee);
                    store.mutate_input_object(updated_source);
                } else {
                    // Not a gas coin, stake entire amount
                    stake_amount = source_balance;

                    // Request to add stake
                    let staked_soma = state.request_add_stake(signer, address, stake_amount)?;

                    // Create StakedSoma object
                    let staked_soma_object = Object::new_staked_soma_object(
                        ObjectID::derive_id(tx_digest, store.next_creation_num()),
                        staked_soma,
                        Owner::AddressOwner(signer),
                        tx_digest,
                    );
                    store.create_object(staked_soma_object);

                    // Delete the coin since we're staking all of it
                    store.delete_input_object(&coin_id);
                }
            }
        }

        // Update system state
        let state_bytes = bcs::to_bytes(&state).map_err(|e| {
            ExecutionFailureStatus::SomaError(SomaError::from(format!(
                "Failed to serialize updated system state: {}",
                e
            )))
        })?;

        let mut updated_state_object = state_object;
        updated_state_object.data.update_contents(state_bytes);
        store.mutate_input_object(updated_state_object);

        Ok(())
    }

    /// Execute WithdrawStake transaction
    fn execute_withdraw_stake(
        &self,
        store: &mut TemporaryStore,
        signer: SomaAddress,
        staked_soma_ref: ObjectRef,
        tx_digest: TransactionDigest,
    ) -> ExecutionResult<()> {
        // Get StakedSoma object
        let staked_soma_object = store
            .read_object(&staked_soma_ref.0)
            .ok_or_else(|| ExecutionFailureStatus::ObjectNotFound { object_id: staked_soma_ref.0 })?
            .clone();

        // Check ownership
        check_ownership(&staked_soma_object, signer)?;

        // Extract StakedSoma data
        let staked_soma = Object::as_staked_soma(&staked_soma_object).ok_or_else(|| {
            ExecutionFailureStatus::InvalidObjectType {
                object_id: staked_soma_object.id(),
                expected_type: ObjectType::StakedSoma,
                actual_type: staked_soma_object.type_().clone(),
            }
        })?;

        // Get system state object
        let state_object = store
            .read_object(&SYSTEM_STATE_OBJECT_ID)
            .ok_or_else(|| ExecutionFailureStatus::ObjectNotFound {
                object_id: SYSTEM_STATE_OBJECT_ID,
            })?
            .clone();

        // Deserialize system state
        let mut state = bcs::from_bytes::<SystemState>(state_object.as_inner().data.contents())
            .map_err(|e| {
                ExecutionFailureStatus::SomaError(SomaError::from(format!(
                    "Failed to deserialize system state: {}",
                    e
                )))
            })?;

        // Process withdrawal
        let withdrawn_amount = state.request_withdraw_stake(staked_soma)?;

        // Delete StakedSoma object
        store.delete_input_object(&staked_soma_ref.0);

        // Create a Coin object with the withdrawn amount
        let new_coin = Object::new_coin(
            ObjectID::derive_id(tx_digest, store.next_creation_num()),
            withdrawn_amount,
            Owner::AddressOwner(signer),
            tx_digest,
        );
        store.create_object(new_coin);

        // Update system state
        let state_bytes = bcs::to_bytes(&state).map_err(|e| {
            ExecutionFailureStatus::SomaError(SomaError::from(format!(
                "Failed to serialize updated system state: {}",
                e
            )))
        })?;

        let mut updated_state_object = state_object;
        updated_state_object.data.update_contents(state_bytes);
        store.mutate_input_object(updated_state_object);

        Ok(())
    }
}

impl TransactionExecutor for StakingExecutor {
    fn execute(
        &mut self,
        store: &mut TemporaryStore,
        signer: SomaAddress,
        kind: TransactionKind,
        tx_digest: TransactionDigest,
        value_fee: u64,
    ) -> ExecutionResult<()> {
        match kind {
            TransactionKind::AddStake { address, coin_ref, amount } => self
                .execute_add_stake(store, signer, address, coin_ref, amount, tx_digest, value_fee),
            TransactionKind::WithdrawStake { staked_soma } => {
                self.execute_withdraw_stake(store, signer, staked_soma, tx_digest)
            }
            _ => Err(ExecutionFailureStatus::InvalidTransactionType),
        }
    }
}

impl FeeCalculator for StakingExecutor {
    fn calculate_value_fee(&self, store: &TemporaryStore, kind: &TransactionKind) -> u64 {
        // Staking gets half the normal value fee to incentivize participation
        let value_fee_bps = store.fee_parameters.value_fee_bps / 2;

        match kind {
            TransactionKind::AddStake { coin_ref, amount, .. } => {
                let stake_amount = if let Some(specific_amount) = amount {
                    *specific_amount
                } else {
                    store.read_object(&coin_ref.0).and_then(|obj| obj.as_coin()).unwrap_or(0)
                };

                if stake_amount == 0 {
                    return 0;
                }

                bps_mul(stake_amount, value_fee_bps)
            }

            TransactionKind::WithdrawStake { staked_soma } => {
                if let Some(staked_obj) = store.read_object(&staked_soma.0) {
                    if let Some(staked_soma_data) = Object::as_staked_soma(&staked_obj) {
                        let principal = staked_soma_data.principal;
                        return bps_mul(principal, value_fee_bps);
                    }
                }
                // Fallback to base fee if we can't read the object
                store.fee_parameters.base_fee
            }

            _ => 0,
        }
    }
}

/// Verifies an object is a coin and returns its balance
fn verify_coin(object: &Object) -> Result<u64, ExecutionFailureStatus> {
    object.as_coin().ok_or_else(|| ExecutionFailureStatus::InvalidObjectType {
        object_id: object.id(),
        expected_type: ObjectType::Coin,
        actual_type: object.type_().clone(),
    })
}
