// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use types::SYSTEM_STATE_OBJECT_ID;
use types::base::SomaAddress;
use types::digests::TransactionDigest;
use types::effects::ExecutionFailureStatus;
use types::error::{ExecutionResult, SomaError};
use types::object::{CoinType, Object, ObjectID, ObjectRef, ObjectType, Owner};
use types::system_state::SystemState;
use types::temporary_store::TemporaryStore;
use types::transaction::TransactionKind;

use super::object::check_ownership;
use super::{TransactionExecutor, checked_sub};

pub struct StakingExecutor;

impl StakingExecutor {
    pub fn new() -> Self {
        Self {}
    }

    /// Execute AddStake. Gas fee was already deducted from the gas coin in
    /// `prepare_gas`, so this just moves the stake amount.
    fn execute_add_stake(
        &self,
        store: &mut TemporaryStore,
        signer: SomaAddress,
        address: SomaAddress,
        coin_ref: ObjectRef,
        amount: Option<u64>,
        tx_digest: TransactionDigest,
    ) -> ExecutionResult<()> {
        let coin_id = coin_ref.0;
        let is_gas_coin = store.gas_object_id == Some(coin_id);

        let source_object = store
            .read_object(&coin_id)
            .ok_or_else(|| ExecutionFailureStatus::ObjectNotFound { object_id: coin_id })?;

        check_ownership(&source_object, signer)?;
        let source_balance = verify_coin(&source_object)?;

        let state_object = store
            .read_object(&SYSTEM_STATE_OBJECT_ID)
            .ok_or_else(|| ExecutionFailureStatus::ObjectNotFound {
                object_id: SYSTEM_STATE_OBJECT_ID,
            })?
            .clone();

        let mut state = bcs::from_bytes::<SystemState>(state_object.as_inner().data.contents())
            .map_err(|e| {
                ExecutionFailureStatus::SomaError(SomaError::from(format!(
                    "Failed to deserialize system state: {}",
                    e
                )))
            })?;

        // Resolve actual stake amount.
        let stake_amount = match amount {
            Some(n) => n,
            None => source_balance, // stake the full coin
        };

        if source_balance < stake_amount {
            return Err(ExecutionFailureStatus::InsufficientCoinBalance.into());
        }

        let staked_soma = state.request_add_stake(signer, address, stake_amount)?;
        let staked_soma_object = Object::new_staked_soma_object(
            ObjectID::derive_id(tx_digest, store.next_creation_num()),
            staked_soma,
            Owner::AddressOwner(signer),
            tx_digest,
        );
        store.create_object(staked_soma_object);

        // Update or delete the source coin. Gas fee was already taken in prepare_gas,
        // so the gas-coin distinction no longer matters here.
        let _ = is_gas_coin;
        let remaining = checked_sub(source_balance, stake_amount)?;
        if remaining == 0 {
            store.delete_input_object(&coin_id);
        } else {
            let mut updated_source = source_object.clone();
            updated_source.update_coin_balance(remaining);
            store.mutate_input_object(updated_source);
        }

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
            CoinType::Soma,
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
    fn fee_units(&self, _store: &TemporaryStore, kind: &TransactionKind) -> u32 {
        match kind {
            // AddStake / WithdrawStake each touch SystemState, source/StakedSoma, and
            // create or delete a coin/StakedSoma. Charge a small flat amount.
            TransactionKind::AddStake { .. } | TransactionKind::WithdrawStake { .. } => 2,
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
            TransactionKind::AddStake { address, coin_ref, amount } => {
                self.execute_add_stake(store, signer, address, coin_ref, amount, tx_digest)
            }
            TransactionKind::WithdrawStake { staked_soma } => {
                self.execute_withdraw_stake(store, signer, staked_soma, tx_digest)
            }
            _ => Err(ExecutionFailureStatus::InvalidTransactionType),
        }
    }
}

/// Verifies an object is a SOMA coin and returns its balance
fn verify_coin(object: &Object) -> Result<u64, ExecutionFailureStatus> {
    // Staking requires SOMA coins
    let balance = object.as_coin().ok_or_else(|| ExecutionFailureStatus::InvalidObjectType {
        object_id: object.id(),
        expected_type: ObjectType::Coin(CoinType::Soma),
        actual_type: object.type_().clone(),
    })?;
    match object.coin_type() {
        Some(CoinType::Soma) => Ok(balance),
        _ => Err(ExecutionFailureStatus::InvalidObjectType {
            object_id: object.id(),
            expected_type: ObjectType::Coin(CoinType::Soma),
            actual_type: object.type_().clone(),
        }),
    }
}
