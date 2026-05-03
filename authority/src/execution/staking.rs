// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use types::SYSTEM_STATE_OBJECT_ID;
use types::balance::BalanceEvent;
use types::base::SomaAddress;
use types::digests::TransactionDigest;
use types::effects::ExecutionFailureStatus;
use types::error::{ExecutionResult, SomaError};
use types::object::{CoinType, Object, ObjectID, ObjectRef, ObjectType, Owner};
use types::system_state::SystemState;
use types::temporary_store::TemporaryStore;
use types::transaction::TransactionKind;

use super::object::check_ownership;
use super::TransactionExecutor;

pub struct StakingExecutor;

impl StakingExecutor {
    pub fn new() -> Self {
        Self {}
    }

    /// Execute AddStake (Stage 9d-C2: balance-mode + F1 fold-to-balance).
    ///
    /// Flow:
    /// 1. Look up the validator's StakingPool and its current F1
    ///    period + cumulative index.
    /// 2. Read the pre-fetched delegation row for (pool, signer). If
    ///    a non-empty row exists, compute pending reward via F1 and
    ///    emit a Deposit balance event paying it to the signer's SOMA
    ///    balance — that's the "fold to balance" semantics.
    /// 3. Emit a Withdraw balance event for `amount` SOMA against the
    ///    signer's accumulator. The reservation pre-pass already
    ///    confirmed the sender has enough; underflow at apply time
    ///    indicates a race we propagate as a hard error.
    /// 4. Run the pool-token side via `state.request_add_stake` so the
    ///    StakingPool's pending_stake / soma_balance / pool_token_balance
    ///    stay in sync (Stage 9d-C5 deletes those fields).
    /// 5. Emit a delegation event with `delta = +amount` and
    ///    `set_period = current_period` to advance the row's fold
    ///    mark. The principal grows by `amount` only; pending
    ///    rewards have already been paid out separately.
    /// 6. Create a StakedSomaV1 object as a safety net — Stage 9d-C5
    ///    deletes it once the F1 path is sole truth.
    #[allow(clippy::too_many_arguments)]
    fn execute_add_stake(
        &self,
        store: &mut TemporaryStore,
        signer: SomaAddress,
        validator: SomaAddress,
        amount: u64,
        tx_digest: TransactionDigest,
    ) -> ExecutionResult<()> {
        if amount == 0 {
            return Err(ExecutionFailureStatus::InvalidArguments {
                reason: "Stake amount cannot be 0".to_string(),
            }
            .into());
        }

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

        // Locate the validator's pool to get the current F1 period.
        // request_add_stake itself will validate the validator
        // exists, but we need the pool data first for the fold.
        let (pool_id, current_period) = {
            let v = state
                .validators()
                .find_validator(validator)
                .or_else(|| {
                    state.validators().pending_validators.iter().find(|v| {
                        v.metadata.soma_address == validator
                    })
                })
                .ok_or(ExecutionFailureStatus::ValidatorNotFound)?;
            (v.staking_pool.id, v.staking_pool.current_period)
        };

        // F1 fold-to-balance: read pre-fetched row, compute pending,
        // pay to balance. A new staker (no row) gets nothing.
        if let Some(existing) = store.prefetched_delegations.get(&pool_id).copied() {
            if existing.principal > 0 {
                let v = state
                    .validators()
                    .find_validator(validator)
                    .expect("validator existence checked above");
                let pending = v
                    .staking_pool
                    .f1_pending_reward(existing.principal, existing.last_collected_period);
                if pending > 0 {
                    store.emit_balance_event(BalanceEvent::Deposit {
                        owner: signer,
                        coin_type: CoinType::Soma,
                        amount: pending,
                    });
                }
            }
        }

        // Debit the principal from the staker's SOMA balance.
        store.emit_balance_event(BalanceEvent::Withdraw {
            owner: signer,
            coin_type: CoinType::Soma,
            amount,
        });

        // Run the pool-token side. Stage 9d-C5 deletes the StakedSomaV1
        // creation and the pool-token math; today they remain as a
        // safety net.
        let staked_soma = state.request_add_stake(signer, validator, amount)?;

        // F1 row update: bump principal by `amount`, advance the
        // collection mark to current_period (the period from which
        // any future rewards will count).
        store.emit_delegation_event(
            pool_id,
            signer,
            amount as i128,
            Some(current_period),
        );

        let staked_soma_object = Object::new_staked_soma_object(
            ObjectID::derive_id(tx_digest, store.next_creation_num()),
            staked_soma,
            Owner::AddressOwner(signer),
            tx_digest,
        );
        store.create_object(staked_soma_object);

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

        // Stage 9b: capture the delegation row identity before the
        // StakedSomaV1 is consumed by `request_withdraw_stake` so we
        // can emit the matching negative delegation delta below.
        let pool_id = staked_soma.pool_id;
        let activation_epoch = staked_soma.stake_activation_epoch;
        let principal = staked_soma.principal;

        // Process withdrawal
        let withdrawn_amount = state.request_withdraw_stake(staked_soma)?;

        // Stage 9d-C1: clear the delegation row. With ONE row per
        // (pool, staker), withdrawing the StakedSomaV1's full
        // principal drains that row to zero — `apply_delegation_events`
        // deletes it outright per the row-deletion contract.
        // `set_period: None` because no fold yet (Stage 9d-C3 wires
        // the F1 fold-to-balance into WithdrawStake).
        let _ = activation_epoch; // unused under the new schema
        store.emit_delegation_event(
            pool_id,
            signer,
            -(principal as i128),
            None,
        );

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
            TransactionKind::AddStake { validator, amount } => {
                self.execute_add_stake(store, signer, validator, amount, tx_digest)
            }
            TransactionKind::WithdrawStake { staked_soma } => {
                self.execute_withdraw_stake(store, signer, staked_soma, tx_digest)
            }
            _ => Err(ExecutionFailureStatus::InvalidTransactionType),
        }
    }
}
