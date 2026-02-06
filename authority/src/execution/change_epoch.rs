use types::{
    SYSTEM_STATE_OBJECT_ID,
    base::SomaAddress,
    digests::TransactionDigest,
    effects::ExecutionFailureStatus,
    error::{ExecutionResult, SomaError},
    object::{Object, ObjectID, Owner},
    system_state::SystemState,
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
            protocol_config::Chain::Mainnet, // TODO: detect which chain to use here
        );

        // Store epoch_start_timestamp_ms for target generation (before moving change_epoch)
        let epoch_start_timestamp_ms = change_epoch.epoch_start_timestamp_ms;

        let validator_rewards = state.advance_epoch(
            change_epoch.epoch,
            &next_protocol_config,
            change_epoch.fees,
            epoch_start_timestamp_ms,
            change_epoch.epoch_randomness,
        )?;

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
        // Note: advance_epoch() already called advance_epoch_targets() which set reward_per_target
        if !state.model_registry.active_models.is_empty() {
            let initial_targets = state.parameters.target_initial_targets_per_epoch;
            let reward_per_target = state.target_state.reward_per_target;
            let models_per_target = state.parameters.target_models_per_target;
            let embedding_dim = state.parameters.target_embedding_dim;
            let new_epoch = state.epoch;

            let mut targets_created = 0u64;
            for _ in 0..initial_targets {
                // Check emission pool has sufficient funds
                if state.emission_pool.balance < reward_per_target {
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
                    &state.model_registry,
                    &state.target_state,
                    models_per_target,
                    embedding_dim,
                    new_epoch,
                ) {
                    Ok(target) => {
                        // Deduct reward from emission pool
                        state.emission_pool.balance -= reward_per_target;

                        // Record that a target was generated (for difficulty adjustment)
                        state.target_state.record_target_generated();

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
                        tracing::warn!("Failed to generate target at epoch {}: {:?}", new_epoch, e);
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

        // Update state object with new state
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
