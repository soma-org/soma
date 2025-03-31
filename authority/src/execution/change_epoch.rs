use types::{
    base::SomaAddress,
    digests::TransactionDigest,
    effects::ExecutionFailureStatus,
    error::{ExecutionResult, SomaError},
    system_state::SystemState,
    temporary_store::TemporaryStore,
    transaction::TransactionKind,
    SYSTEM_STATE_OBJECT_ID,
};

use super::TransactionExecutor;

/// Executor for system state transactions (validators)
pub struct ChangeEpochExecutor;

impl ChangeEpochExecutor {
    pub(crate) fn new() -> Self {
        Self {}
    }
}

impl TransactionExecutor for ChangeEpochExecutor {
    fn execute(
        &self,
        store: &mut TemporaryStore,
        signer: SomaAddress,
        kind: TransactionKind,
        tx_digest: TransactionDigest,
        // _gas_object_id: Option<ObjectID>,
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
        let result = match kind {
            TransactionKind::ChangeEpoch(change_epoch) => {
                state.advance_epoch(change_epoch.epoch, change_epoch.epoch_start_timestamp_ms)
            }
            _ => Err(ExecutionFailureStatus::InvalidTransactionType),
        };

        // Early return on error
        result?;

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
