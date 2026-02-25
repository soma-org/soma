// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use tracing::info;
use types::{
    SYSTEM_STATE_OBJECT_ID,
    base::SomaAddress,
    digests::TransactionDigest,
    effects::ExecutionFailureStatus,
    error::{ExecutionResult, SomaError, SomaResult},
    object::ObjectID,
    system_state::SystemState,
    temporary_store::TemporaryStore,
    transaction::TransactionKind,
};

use super::{FeeCalculator, TransactionExecutor};

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
        tx_digest: TransactionDigest,
        store: &mut TemporaryStore,
    ) -> ExecutionResult<()> {
        match tx_kind {
            TransactionKind::AddValidator(args) => state.request_add_validator(
                signer,
                args.pubkey_bytes.clone(),
                args.network_pubkey_bytes.clone(),
                args.worker_pubkey_bytes.clone(),
                args.proof_of_possession.clone(),
                args.net_address.clone(),
                args.p2p_address.clone(),
                args.primary_address.clone(),
                args.proxy_address.clone(),
                ObjectID::derive_id(tx_digest, store.next_creation_num()),
            ),
            TransactionKind::RemoveValidator(args) => {
                state.request_remove_validator(signer, args.pubkey_bytes.clone())
            }
            TransactionKind::ReportValidator { reportee } => {
                state.report_validator(signer, *reportee)
            }
            TransactionKind::UndoReportValidator { reportee } => {
                state.undo_report_validator(signer, *reportee)
            }
            TransactionKind::SetCommissionRate { new_rate } => {
                state.request_set_commission_rate(signer, *new_rate)
            }
            TransactionKind::UpdateValidatorMetadata(args) => {
                state.request_update_validator_metadata(signer, args)
            }
            _ => Err(ExecutionFailureStatus::InvalidTransactionType),
        }
    }
}

impl TransactionExecutor for ValidatorExecutor {
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
        let result = self.process_system_state(&mut state, &kind, signer, tx_digest, store);

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

impl FeeCalculator for ValidatorExecutor {}
