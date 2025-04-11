use types::{
    base::SomaAddress,
    digests::TransactionDigest,
    effects::ExecutionFailureStatus,
    error::{ExecutionResult, SomaError},
    object::{Object, Owner},
    system_state::SystemState,
    temporary_store::TemporaryStore,
    transaction::TransactionKind,
    SYSTEM_STATE_OBJECT_ID,
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
        &self,
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
                // TODO: Pass in cumulative epoch transaction fees to split rewards
                // TODO: Fixed reward slashing rate at 50%
                let reward_slashing_rate: u64 = 5000; // 50% in basis points
                state.advance_epoch(
                    change_epoch.epoch,
                    0,
                    change_epoch.epoch_start_timestamp_ms,
                    reward_slashing_rate,
                )
            }
            _ => Err(ExecutionFailureStatus::InvalidTransactionType),
        }?;

        for (validator, reward) in validator_rewards {
            // Create StakedSoma object
            let staked_soma_object =
                Object::new_staked_soma_object(reward, Owner::AddressOwner(validator), tx_digest);
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
