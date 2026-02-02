use std::collections::HashMap;

use types::{
    SYSTEM_STATE_OBJECT_ID,
    base::SomaAddress,
    digests::TransactionDigest,
    effects::ExecutionFailureStatus,
    error::{ExecutionResult, SomaError},
    object::{Object, ObjectID, Owner},
    system_state::SystemState,
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

        // Process the transaction
        let validator_rewards = match kind {
            TransactionKind::ChangeEpoch(change_epoch) => {
                let next_protocol_config = protocol_config::ProtocolConfig::get_for_version(
                    change_epoch.protocol_version,
                    protocol_config::Chain::Mainnet, // TODO: detect which chain to use here
                );

                let validator_rewards = state.advance_epoch(
                    change_epoch.epoch,
                    &next_protocol_config,
                    change_epoch.fees,
                    change_epoch.epoch_start_timestamp_ms,
                    change_epoch.epoch_randomness,
                )?;

                Ok(validator_rewards)
            }
            _ => Err(ExecutionFailureStatus::InvalidTransactionType),
        }?;

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
