use types::{
    SYSTEM_STATE_OBJECT_ID,
    base::SomaAddress,
    digests::TransactionDigest,
    effects::ExecutionFailureStatus,
    error::{ExecutionResult, SomaError},
    object::{Object, ObjectID, Owner},
    system_state::{SystemState, SystemStateTrait},
    target::{generate_target, make_target_seed},
    temporary_store::TemporaryStore,
    transaction::TransactionKind,
};

use super::{FeeCalculator, TransactionExecutor};

/// Executor for system state transactions (validators)
pub struct ChangeEpochExecutor;

impl ChangeEpochExecutor {
    pub(crate) fn new() -> Self {
        Self {}
    }
}

impl FeeCalculator for ChangeEpochExecutor {}

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
    fn execute(
        &mut self,
        store: &mut TemporaryStore,
        signer: SomaAddress,
        kind: TransactionKind,
        tx_digest: TransactionDigest,
        _value_fee: u64,
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

        // Store epoch_start_timestamp_ms for target generation (before moving change_epoch)
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
                // Normal path: create reward objects and generate targets

                for (validator, reward) in validator_rewards {
                    // Create StakedSoma object
                    let staked_soma_object = Object::new_staked_soma_object(
                        ObjectID::derive_id(tx_digest, store.next_creation_num()),
                        reward,
                        Owner::AddressOwner(validator),
                        tx_digest,
                    );
                    store.create_object(staked_soma_object);
                }

                // Create initial targets for the new epoch if active models exist
                if !state.model_registry().active_models.is_empty() {
                    let initial_targets = state.parameters().target_initial_targets_per_epoch;
                    let reward_per_target = state.target_state().reward_per_target;
                    let models_per_target = state.parameters().target_models_per_target;
                    let embedding_dim = state.parameters().target_embedding_dim;
                    let new_epoch = state.epoch();

                    let mut targets_created = 0u64;
                    for _ in 0..initial_targets {
                        // Check emission pool has sufficient funds
                        if state.emission_pool().balance < reward_per_target {
                            tracing::warn!(
                                "Emission pool depleted at epoch {}, stopping target generation at {} targets",
                                new_epoch,
                                targets_created
                            );
                            break;
                        }

                        let creation_num = store.next_creation_num();
                        let seed = make_target_seed(&tx_digest, creation_num);

                        match generate_target(
                            seed,
                            &state.model_registry(),
                            &state.target_state(),
                            models_per_target,
                            embedding_dim,
                            new_epoch,
                        ) {
                            Ok(target) => {
                                // Deduct reward from emission pool
                                state.emission_pool_mut().balance -= reward_per_target;

                                // Record that a target was generated (for difficulty adjustment)
                                state.target_state_mut().record_target_generated();

                                // Create target as shared object
                                let target_object = Object::new_target_object(
                                    ObjectID::derive_id(tx_digest, creation_num),
                                    target,
                                    tx_digest,
                                );
                                store.create_object(target_object);
                                targets_created += 1;
                            }
                            Err(e) => {
                                tracing::warn!(
                                    "Failed to generate target at epoch {}: {:?}",
                                    new_epoch,
                                    e
                                );
                                break;
                            }
                        }
                    }

                    tracing::info!(
                        "Created {} targets for epoch {} with reward_per_target={}",
                        targets_created,
                        new_epoch,
                        reward_per_target
                    );
                }
            }
            Err(e) => {
                // Safe mode: restore state from backup and do minimal epoch bump.
                // This path is intentionally simple and cannot fail.
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

                // No validator rewards are created, no targets generated.
                // Everything is accumulated for the next successful epoch transition.
            }
        }

        // Serialize and commit state — this path is shared for both normal and safe mode
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
