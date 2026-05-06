// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use types::SYSTEM_STATE_OBJECT_ID;
use types::base::SomaAddress;
use types::digests::TransactionDigest;
use types::effects::ExecutionFailureStatus;
use types::error::{ExecutionResult, SomaError};
use types::object::{CoinType, ObjectID};
use types::system_state::staking::{Delegation, auto_settle};
use types::system_state::{SystemState, SystemStateTrait};
use types::temporary_store::TemporaryStore;
use types::transaction::TransactionKind;

use super::TransactionExecutor;

pub struct StakingExecutor;

impl StakingExecutor {
    pub fn new() -> Self {
        Self {}
    }

    /// Execute AddStake under the auto-compound F1 model.
    ///
    /// Flow:
    /// 1. Read the prefetched `(pool, signer)` delegation row (or
    ///    default-empty for a first-time staker).
    /// 2. `auto_settle` against the validator's pool: any accrued
    ///    rewards on the existing principal compound into the row's
    ///    principal; matured pending stake promotes alongside its
    ///    accrued share.
    /// 3. Add `amount` to the appropriate bucket on the pool — direct
    ///    `active_stake` for preactive pools (so the first
    ///    `deposit_staker_rewards` has a divisor), or
    ///    `pending_active_stake` for active pools (so the new stake
    ///    doesn't earn current-epoch rewards).
    /// 4. Mirror the bucket choice on the delegation row's
    ///    `principal` / `pending_principal`.
    /// 5. Withdraw `amount` SOMA from the staker's balance accumulator.
    /// 6. Emit the post-mutation row as a `DelegationEvent` so the
    ///    dual-write step persists it atomically with the rest of
    ///    this commit.
    fn execute_add_stake(
        &self,
        store: &mut TemporaryStore,
        signer: SomaAddress,
        validator: SomaAddress,
        amount: u64,
        _tx_digest: TransactionDigest,
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

        let current_epoch = state.epoch();

        // Look up the pool (active or pending) to read the current
        // cumulative_index for auto_settle, plus the pool_id and
        // preactive flag for downstream routing.
        let (pool_id, is_preactive, current_index) = {
            let v = state
                .validators()
                .find_validator(validator)
                .or_else(|| {
                    state
                        .validators()
                        .pending_validators
                        .iter()
                        .find(|v| v.metadata.soma_address == validator)
                })
                .ok_or(ExecutionFailureStatus::ValidatorNotFound)?;
            (
                v.staking_pool.id,
                v.staking_pool.is_preactive(),
                v.staking_pool.cumulative_index,
            )
        };

        // Auto-compound any prior accrual into the row's principal,
        // then bump the appropriate bucket by `amount`.
        let mut row = store
            .prefetched_delegations
            .get(&pool_id)
            .copied()
            .unwrap_or_default();
        {
            let pool_view = state
                .validators()
                .find_validator(validator)
                .or_else(|| {
                    state
                        .validators()
                        .pending_validators
                        .iter()
                        .find(|v| v.metadata.soma_address == validator)
                })
                .map(|v| &v.staking_pool)
                .ok_or(ExecutionFailureStatus::ValidatorNotFound)?;
            auto_settle(&mut row, pool_view, current_epoch);
        }

        if is_preactive {
            row.principal = row.principal.saturating_add(amount);
            row.index_at_last_collect = current_index;
        } else {
            row.pending_principal = row.pending_principal.saturating_add(amount);
            row.pending_added_at_epoch = current_epoch;
        }

        // Bump the pool aggregate (preactive → active, active →
        // pending) and refresh the staking_pool_mappings index.
        state.add_stake_to_validator(validator, amount)?;

        // Debit the principal from the staker's SOMA balance.
        store.emit_accumulator_event(
            types::effects::object_change::AccumulatorAddress::balance(signer, CoinType::Soma),
            types::effects::object_change::AccumulatorOperation::Split,
            amount,
        );

        // Persist the post-mutation row.
        store.emit_delegation_event(pool_id, signer, Some(row));

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

    /// Execute WithdrawStake under the auto-compound F1 model.
    ///
    /// Flow:
    /// 1. Read the prefetched `(pool_id, signer)` row. Missing or
    ///    fully-empty row → error.
    /// 2. `auto_settle` against the pool to compound prior accrual
    ///    into `principal` (so the user can withdraw from a fully
    ///    up-to-date balance).
    /// 3. Resolve `amount` (None = full balance, both buckets).
    ///    Reject withdrawals exceeding `principal + pending_principal`.
    /// 4. Drain pending first (a same-epoch deposit reversal — that
    ///    stake never earned anything), then active. The active
    ///    decrement at the next boundary's smaller fold divisor
    ///    redistributes the withdrawer's current-epoch share to
    ///    remaining stakers (matches Sui's pool-token semantics).
    /// 5. Credit `amount` SOMA to the staker's balance accumulator.
    /// 6. Emit the post-mutation row (or a drain event if both
    ///    buckets are now zero).
    fn execute_withdraw_stake(
        &self,
        store: &mut TemporaryStore,
        signer: SomaAddress,
        pool_id: ObjectID,
        amount: Option<u64>,
        _tx_digest: TransactionDigest,
    ) -> ExecutionResult<()> {
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

        let current_epoch = state.epoch();

        let mut row = store
            .prefetched_delegations
            .get(&pool_id)
            .copied()
            .ok_or_else(|| {
                ExecutionFailureStatus::SomaError(SomaError::from(format!(
                    "No active stake by {} in pool {}",
                    signer, pool_id
                )))
            })?;
        if row.is_empty() {
            return Err(ExecutionFailureStatus::SomaError(SomaError::from(format!(
                "No active stake by {} in pool {}",
                signer, pool_id
            )))
            .into());
        }

        // Locate the validator owning this pool (any of the three
        // collections). We need a pool view for auto_settle and the
        // current_index snapshot for the post-settle row.
        let validator_addr = state
            .validators()
            .staking_pool_mappings
            .get(&pool_id)
            .copied()
            .ok_or_else(|| {
                ExecutionFailureStatus::SomaError(SomaError::from(format!(
                    "StakingPool not found: {}",
                    pool_id
                )))
            })?;
        let current_index = {
            let pool_view = state
                .validators()
                .find_validator(validator_addr)
                .or_else(|| {
                    state
                        .validators()
                        .pending_validators
                        .iter()
                        .find(|v| v.metadata.soma_address == validator_addr)
                })
                .map(|v| &v.staking_pool)
                .or_else(|| {
                    state.validators().inactive_validators.get(&pool_id).map(|v| &v.staking_pool)
                })
                .ok_or(ExecutionFailureStatus::ValidatorNotFound)?;
            auto_settle(&mut row, pool_view, current_epoch);
            pool_view.cumulative_index
        };

        let total_stake = row.total();
        let withdraw_amount = amount.unwrap_or(total_stake);
        if withdraw_amount == 0 {
            return Err(ExecutionFailureStatus::InvalidArguments {
                reason: "Withdraw amount cannot be 0".to_string(),
            }
            .into());
        }
        if withdraw_amount > total_stake {
            return Err(ExecutionFailureStatus::InvalidArguments {
                reason: format!(
                    "Withdraw amount {} exceeds delegation balance {} (active {} + pending {})",
                    withdraw_amount, total_stake, row.principal, row.pending_principal,
                ),
            }
            .into());
        }

        // Drain pending first, then active.
        let from_pending = std::cmp::min(withdraw_amount, row.pending_principal);
        let from_active = withdraw_amount - from_pending;
        row.pending_principal = row.pending_principal.saturating_sub(from_pending);
        row.principal = row.principal.saturating_sub(from_active);
        if row.pending_principal == 0 {
            row.pending_added_at_epoch = 0;
        }
        row.index_at_last_collect = current_index;

        // Pool aggregate: pass the same active/pending split we
        // applied to the row, so pool aggregates stay in sync with
        // the staker's row when other stakers also have pending
        // stake on this pool.
        state.remove_stake_from_validator(pool_id, from_active, from_pending)?;

        // Credit the principal to the staker's SOMA balance.
        store.emit_accumulator_event(
            types::effects::object_change::AccumulatorAddress::balance(signer, CoinType::Soma),
            types::effects::object_change::AccumulatorOperation::Merge,
            withdraw_amount,
        );

        // Persist the post-mutation row, or signal a full drain.
        let new_state = if row.is_empty() { None } else { Some(row) };
        store.emit_delegation_event(pool_id, signer, new_state);

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
            TransactionKind::WithdrawStake { pool_id, amount } => {
                self.execute_withdraw_stake(store, signer, pool_id, amount, tx_digest)
            }
            _ => Err(ExecutionFailureStatus::InvalidTransactionType),
        }
    }
}
