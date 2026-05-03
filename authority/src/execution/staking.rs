// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use types::SYSTEM_STATE_OBJECT_ID;
use types::balance::BalanceEvent;
use types::base::SomaAddress;
use types::digests::TransactionDigest;
use types::effects::ExecutionFailureStatus;
use types::error::{ExecutionResult, SomaError};
use types::object::{CoinType, ObjectID};
use types::system_state::{SystemState, SystemStateTrait};
use types::temporary_store::TemporaryStore;
use types::transaction::TransactionKind;

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

        // Run the pool-token side. Stage 9d-C5 deletes the
        // pool-token math entirely; until then, `request_add_stake`
        // still mutates pool's pending_stake / next_epoch_stake which
        // drive voting-power calculations at the next epoch boundary.
        // Stage 9d-C4: the returned StakedSomaV1 is discarded — no
        // object output. The F1 row is the sole user-visible record
        // of the stake.
        let _ = tx_digest;
        let _staked_soma = state.request_add_stake(signer, validator, amount)?;

        // F1 row update: bump principal by `amount`, advance the
        // collection mark to current_period (the period from which
        // any future rewards will count).
        store.emit_delegation_event(
            pool_id,
            signer,
            amount as i128,
            Some(current_period),
        );

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

    /// Execute WithdrawStake (Stage 9d-C3: balance-mode + F1 fold-to-balance).
    ///
    /// Flow:
    /// 1. Read the pre-fetched (pool_id, signer) delegation row.
    ///    Missing or zero-principal row → error.
    /// 2. Resolve `amount` (None = entire row). Reject withdrawals
    ///    larger than the row's principal.
    /// 3. F1 fold pending rewards via the pool's cumulative index;
    ///    emit a Deposit balance event paying them to the signer's
    ///    SOMA balance.
    /// 4. Run the pool-token side via `state.request_withdraw_stake`
    ///    with a synthesized StakedSomaV1 whose `stake_activation_epoch`
    ///    equals the current epoch — that path returns 0 rewards
    ///    (rate at "activation" == rate now), so the only state
    ///    change is the bookkeeping for `pending_total_soma_withdraw`
    ///    and `next_epoch_stake`. No reward double-count: F1 already
    ///    paid them in step 3.
    /// 5. Emit a Deposit for the withdrawn principal — the staker's
    ///    SOMA balance gets credited atomically with the principal
    ///    reduction.
    /// 6. Emit a delegation event with `delta = -amount` and
    ///    `set_period = current_period`. A full drain deletes the
    ///    row outright per the `apply_delegation_events` contract.
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

        // Read the F1 row.
        let existing = store.prefetched_delegations.get(&pool_id).copied().ok_or_else(|| {
            ExecutionFailureStatus::SomaError(SomaError::from(format!(
                "No active stake by {} in pool {}",
                signer, pool_id
            )))
        })?;
        if existing.principal == 0 {
            return Err(ExecutionFailureStatus::SomaError(SomaError::from(format!(
                "No active stake by {} in pool {}",
                signer, pool_id
            )))
            .into());
        }

        let withdraw_amount = amount.unwrap_or(existing.principal);
        if withdraw_amount == 0 {
            return Err(ExecutionFailureStatus::InvalidArguments {
                reason: "Withdraw amount cannot be 0".to_string(),
            }
            .into());
        }
        if withdraw_amount > existing.principal {
            return Err(ExecutionFailureStatus::InvalidArguments {
                reason: format!(
                    "Withdraw amount {} exceeds delegation principal {}",
                    withdraw_amount, existing.principal,
                ),
            }
            .into());
        }

        // Look up the pool's current F1 period for the fold mark.
        // Also confirm the pool exists; without it, the pool-token
        // side below will error and we want to fail fast here.
        let current_period = {
            let mappings = &state.validators().staking_pool_mappings;
            let validator_addr = mappings.get(&pool_id).copied().ok_or_else(|| {
                ExecutionFailureStatus::SomaError(SomaError::from(format!(
                    "StakingPool not found: {}",
                    pool_id
                )))
            })?;
            let validator = state
                .validators()
                .find_validator(validator_addr)
                .or_else(|| {
                    state
                        .validators()
                        .pending_validators
                        .iter()
                        .find(|v| v.metadata.soma_address == validator_addr)
                })
                .or_else(|| state.validators().inactive_validators.get(&pool_id))
                .ok_or(ExecutionFailureStatus::ValidatorNotFound)?;
            let pool = &validator.staking_pool;
            // F1 fold-to-balance: pay pending rewards on the pre-
            // mutation principal. Uses the current cumulative index
            // and the row's last_collected_period.
            let pending = pool.f1_pending_reward(
                existing.principal,
                existing.last_collected_period,
            );
            if pending > 0 {
                store.emit_balance_event(BalanceEvent::Deposit {
                    owner: signer,
                    coin_type: CoinType::Soma,
                    amount: pending,
                });
            }
            pool.current_period
        };

        // Pool-token side: synthesize a StakedSomaV1 whose
        // stake_activation_epoch == self.epoch so the rate-at-
        // activation matches the current rate and `withdraw_rewards`
        // returns 0. The only effect is updating
        // `pending_total_soma_withdraw` and `next_epoch_stake`. F1
        // owns reward payout, so this stays in sync with no
        // double-counting.
        let synthetic = types::system_state::staking::StakedSomaV1::new(
            pool_id,
            state.epoch(),
            withdraw_amount,
        );
        let withdrawn_principal = state.request_withdraw_stake(synthetic)?;
        debug_assert_eq!(
            withdrawn_principal, withdraw_amount,
            "fresh-activation withdraw must return exactly principal",
        );

        // Credit the principal to the staker's SOMA balance.
        store.emit_balance_event(BalanceEvent::Deposit {
            owner: signer,
            coin_type: CoinType::Soma,
            amount: withdrawn_principal,
        });

        // Drain the F1 row by `withdraw_amount`; advance the fold
        // mark to current_period so any future rewards on the
        // remaining principal start from this period.
        store.emit_delegation_event(
            pool_id,
            signer,
            -(withdraw_amount as i128),
            Some(current_period),
        );

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
            TransactionKind::WithdrawStake { pool_id, amount } => {
                self.execute_withdraw_stake(store, signer, pool_id, amount, tx_digest)
            }
            _ => Err(ExecutionFailureStatus::InvalidTransactionType),
        }
    }
}
