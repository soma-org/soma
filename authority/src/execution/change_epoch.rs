// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use types::SYSTEM_STATE_OBJECT_ID;
use types::base::SomaAddress;
use types::digests::TransactionDigest;
use types::effects::ExecutionFailureStatus;
use types::error::{ExecutionResult, SomaError};
use types::object::{Object, ObjectID, Owner};
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
            types::system_state::staking::StakedSomaV1,
        >,
    >,
    new_epoch: u64,
) -> ExecutionResult<
    std::collections::BTreeMap<
        types::base::SomaAddress,
        types::system_state::staking::StakedSomaV1,
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
                // Normal path: create reward objects for validators
                for (validator, reward) in validator_rewards {
                    let staked_soma_object = Object::new_staked_soma_object(
                        ObjectID::derive_id(tx_digest, store.next_creation_num()),
                        reward,
                        Owner::AddressOwner(validator),
                        tx_digest,
                    );
                    store.create_object(staked_soma_object);
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
