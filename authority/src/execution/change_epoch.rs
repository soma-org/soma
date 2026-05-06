// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use types::SYSTEM_STATE_OBJECT_ID;
use types::base::SomaAddress;
use types::digests::TransactionDigest;
use types::effects::ExecutionFailureStatus;
use types::error::{ExecutionResult, SomaError};
// Object/Owner no longer needed once Stage 9d-C4 removed StakedSomaV1
// creation from the validator-reward path.
use types::system_state::{SystemState, SystemStateTrait};
use types::temporary_store::TemporaryStore;
use types::transaction::TransactionKind;

use super::TransactionExecutor;

/// Executor for system state transactions (validators)
pub struct ChangeEpochExecutor;

impl ChangeEpochExecutor {
    pub(crate) fn new() -> Self {
        Self {}
    }
}


/// Under msim, optionally inject a failure for specific epochs.
/// Tests register a `fail_point_if` callback for "advance_epoch_result_injection"
/// that returns `true` when the epoch should fail.
#[cfg(msim)]
fn maybe_inject_advance_epoch_failure(
    result: ExecutionResult<
        std::collections::BTreeMap<
            types::base::SomaAddress,
            types::system_state::validator::ValidatorRewardCredit,
        >,
    >,
    new_epoch: u64,
) -> ExecutionResult<
    std::collections::BTreeMap<
        types::base::SomaAddress,
        types::system_state::validator::ValidatorRewardCredit,
    >,
> {
    let should_fail = utils::fp::handle_fail_point_if("advance_epoch_result_injection");
    if should_fail {
        tracing::warn!("Failpoint: injecting advance_epoch failure for epoch {}", new_epoch);
        return Err(ExecutionFailureStatus::SomaError(SomaError::from(format!(
            "Injected advance_epoch failure for epoch {}",
            new_epoch
        ))));
    }
    result
}

impl TransactionExecutor for ChangeEpochExecutor {
    fn fee_units(&self, _store: &TemporaryStore, _kind: &TransactionKind) -> u32 {
        0
    }

    fn execute(
        &mut self,
        store: &mut TemporaryStore,
        _signer: SomaAddress,
        kind: TransactionKind,
        tx_digest: TransactionDigest,
    ) -> ExecutionResult<()> {
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

        // Process the transaction - extract ChangeEpoch data
        let TransactionKind::ChangeEpoch(change_epoch) = kind else {
            return Err(ExecutionFailureStatus::InvalidTransactionType);
        };

        let next_protocol_config = protocol_config::ProtocolConfig::get_for_version(
            change_epoch.protocol_version,
            store.chain,
        );

        let epoch_start_timestamp_ms = change_epoch.epoch_start_timestamp_ms;

        // Clone state before attempting advance_epoch so we can restore on failure
        let state_backup = state.clone();

        let result = state.advance_epoch(
            change_epoch.epoch,
            &next_protocol_config,
            change_epoch.fees,
            epoch_start_timestamp_ms,
            change_epoch.epoch_randomness,
        );

        // Under msim, optionally inject failure for testing safe mode
        #[cfg(msim)]
        let result = maybe_inject_advance_epoch_failure(result, change_epoch.epoch);

        match result {
            Ok(validator_rewards) => {
                // Stage 9d-C4 + F1/F9 audit fix: validator commission
                // credits flow through the F1 row — ONE row per
                // (pool, validator), successive epochs accumulate.
                //
                // Two halves to keep the CF and the
                // `DelegationAccumulator` object in sync (audit F1):
                //
                // 1. Mutate the object directly via `mutate_input_object`.
                //    The object was pre-loaded by `execute_transaction`'s
                //    `resolved_accumulators` block for ChangeEpoch.
                // 2. Emit a `DelegationEvent` so `apply_delegation_events`
                //    drains the matching delta into the `delegations`
                //    CF in the same atomic write batch.
                //
                // `set_period` is the validator's pool's NEW
                // `current_period` (post-fold). This advances the row's
                // `last_collected_period` to current so the validator's
                // next AddStake/WithdrawStake doesn't compute
                // `f1_pending_reward` over a period range that
                // pre-dates the commission credit (audit F9).
                for (validator, reward) in validator_rewards {
                    let pool_id = reward.pool_id;
                    let principal_delta = reward.principal;
                    if principal_delta == 0 {
                        continue;
                    }

                    // Look up post-fold current_period for this pool.
                    let new_current_period = state
                        .validators()
                        .find_validator(validator)
                        .map(|v| v.staking_pool.current_period)
                        .or_else(|| {
                            state
                                .validators()
                                .pending_validators
                                .iter()
                                .find(|v| v.metadata.soma_address == validator)
                                .map(|v| v.staking_pool.current_period)
                        })
                        .or_else(|| {
                            state
                                .validators()
                                .inactive_validators
                                .get(&pool_id)
                                .map(|v| v.staking_pool.current_period)
                        });

                    let acc_id = types::accumulator::DelegationAccumulator::derive_id(
                        pool_id, validator,
                    );

                    // Mutate the DelegationAccumulator object so the
                    // object world stays in sync with the CF.
                    if let Some(existing) = store.read_object(&acc_id) {
                        if let Some(mut acc) = existing.as_delegation_accumulator() {
                            acc.principal = acc.principal.saturating_add(principal_delta);
                            if let Some(p) = new_current_period {
                                acc.last_collected_period = p;
                            }
                            let mut new_obj = existing.clone();
                            new_obj.set_delegation_accumulator(&acc);
                            store.mutate_input_object(new_obj);
                        }
                    } else {
                        // First-touch: validator's row doesn't exist yet
                        // (e.g., fresh validator at activation epoch).
                        // Create the object so the next mutation finds it.
                        let acc = types::accumulator::DelegationAccumulator::new(
                            pool_id,
                            validator,
                            principal_delta,
                            new_current_period.unwrap_or(0),
                        );
                        let new_obj = types::object::Object::new_delegation_accumulator(
                            acc, tx_digest,
                        );
                        store.create_object(new_obj);
                    }

                    // Emit the matching CF event.
                    store.emit_delegation_event(
                        pool_id,
                        validator,
                        principal_delta as i128,
                        new_current_period,
                    );
                }
            }
            Err(e) => {
                // Safe mode: restore state from backup and do minimal epoch bump.
                tracing::error!(
                    "advance_epoch FAILED, entering safe mode: {:?}. \
                     The network will continue operating in degraded mode.",
                    e
                );

                state = state_backup;
                state.advance_epoch_safe_mode(
                    change_epoch.epoch,
                    change_epoch.protocol_version.as_u64(),
                    change_epoch.fees,
                    epoch_start_timestamp_ms,
                );
            }
        }

        // Serialize and commit state
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

