use types::{
    base::SomaAddress,
    digests::TransactionDigest,
    error::{SomaError, SomaResult},
    object::ObjectID,
    system_state::SystemState,
    temporary_store::TemporaryStore,
    transaction::TransactionKind,
    SYSTEM_STATE_OBJECT_ID,
};

use super::TransactionExecutor;

/// Executor for system state transactions (validators)
pub struct ValidatorExecutor;

impl ValidatorExecutor {
    pub(crate) fn new() -> Self {
        Self {}
    }

    fn process_system_state(
        &self,
        state: &mut SystemState,
        tx_kind: &TransactionKind,
        signer: SomaAddress,
    ) -> SomaResult<()> {
        match tx_kind {
            TransactionKind::AddValidator(args) => state.request_add_validator(
                signer,
                args.pubkey_bytes.clone(),
                args.network_pubkey_bytes.clone(),
                args.worker_pubkey_bytes.clone(),
                args.net_address.clone(),
                args.p2p_address.clone(),
                args.primary_address.clone(),
            ),
            TransactionKind::RemoveValidator(args) => {
                state.request_remove_validator(signer, args.pubkey_bytes.clone())
            }
            _ => Err(SomaError::from(format!(
                "Invalid transaction type for validator executor"
            ))),
        }
    }
}

impl TransactionExecutor for ValidatorExecutor {
    fn execute(
        &self,
        store: &mut TemporaryStore,
        signer: SomaAddress,
        kind: TransactionKind,
        tx_digest: TransactionDigest,
        // _gas_object_id: Option<ObjectID>,
    ) -> SomaResult<()> {
        // Get system state object
        let state_object = store
            .read_object(&SYSTEM_STATE_OBJECT_ID)
            .ok_or_else(|| {
                SomaError::from(format!(
                    "System state object not found in the temporary store"
                ))
            })?
            .clone();

        // Deserialize system state
        let mut state = bcs::from_bytes::<SystemState>(state_object.as_inner().data.contents())
            .map_err(|e| SomaError::from(format!("Failed to deserialize system state: {}", e)))?;

        // Process the transaction
        let result = self.process_system_state(&mut state, &kind, signer);

        // Early return on error
        result?;

        // Update state object with new state
        let state_bytes = bcs::to_bytes(&state).map_err(|e| {
            SomaError::from(format!("Failed to serialize updated system state: {}", e))
        })?;

        let mut updated_state_object = state_object;
        updated_state_object.data.update_contents(state_bytes);
        store.mutate_input_object(updated_state_object);

        Ok(())
    }
}
