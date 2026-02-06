use std::sync::Arc;

use consensus::{TransactionVerifier, ValidationError};
use tap::TapFallible;
use tracing::{debug, info, instrument, warn};
use types::error::SomaResult;
use types::transaction::Transaction;
use types::{
    consensus::{
        ConsensusPosition, ConsensusTransaction, ConsensusTransactionKind,
        block::{BlockRef, TransactionIndex},
    },
    error::SomaError,
};

use crate::{
    authority::AuthorityState, authority_per_epoch_store::AuthorityPerEpochStore,
    checkpoints::CheckpointServiceNotify,
};

/// Allows verifying the validity of transactions
#[derive(Clone)]
pub struct TxValidator {
    authority_state: Arc<AuthorityState>,

    checkpoint_service: Arc<dyn CheckpointServiceNotify + Send + Sync>,
}

impl TxValidator {
    pub fn new(
        authority_state: Arc<AuthorityState>,

        checkpoint_service: Arc<dyn CheckpointServiceNotify + Send + Sync>,
    ) -> Self {
        let epoch_store = authority_state.load_epoch_store_one_call_per_task().clone();
        info!("TxValidator constructed for epoch {}", epoch_store.epoch());
        Self { authority_state, checkpoint_service }
    }

    fn validate_transactions(&self, txs: &[ConsensusTransactionKind]) -> Result<(), SomaError> {
        let epoch_store = self.authority_state.load_epoch_store_one_call_per_task();

        let mut cert_batch = Vec::new();
        let mut ckpt_messages = Vec::new();
        let mut ckpt_batch = Vec::new();
        for tx in txs.iter() {
            match tx {
                ConsensusTransactionKind::CertifiedTransaction(certificate) => {
                    cert_batch.push(certificate.as_ref());
                }

                ConsensusTransactionKind::CheckpointSignature(signature) => {
                    ckpt_messages.push(signature.as_ref());
                    ckpt_batch.push(&signature.summary);
                }

                ConsensusTransactionKind::EndOfPublish(_) => {}

                ConsensusTransactionKind::CapabilityNotification(_) => {}

                ConsensusTransactionKind::UserTransaction(_tx) => {

                    // TODO(fastpath): move deterministic verifications of user transactions here,
                    // for example verify_transaction().
                }
            }
        }

        epoch_store
            .signature_verifier
            .verify_certs_and_checkpoints(cert_batch, ckpt_batch)
            .tap_err(|e| warn!("batch verification error: {}", e))?;

        // All checkpoint sigs have been verified, forward them to the checkpoint service
        for ckpt in ckpt_messages {
            self.checkpoint_service.notify_checkpoint_signature(&epoch_store, ckpt)?;
        }

        Ok(())
    }

    #[instrument(level = "debug", skip_all, fields(block_ref))]
    fn vote_transactions(
        &self,
        block_ref: &BlockRef,
        txs: Vec<ConsensusTransactionKind>,
    ) -> Vec<TransactionIndex> {
        let epoch_store = self.authority_state.load_epoch_store_one_call_per_task();

        let mut result = Vec::new();
        for (i, tx) in txs.into_iter().enumerate() {
            let ConsensusTransactionKind::UserTransaction(tx) = tx else {
                continue;
            };

            let tx_digest = *tx.digest();
            if let Err(error) = self.vote_transaction(&epoch_store, tx) {
                debug!(?tx_digest, "Voting to reject transaction: {error}");

                result.push(i as TransactionIndex);
                // Cache the rejection vote reason (error) for the transaction
                epoch_store.set_rejection_vote_reason(
                    ConsensusPosition {
                        epoch: epoch_store.epoch(),
                        block: *block_ref,
                        index: i as TransactionIndex,
                    },
                    &error,
                );
            } else {
                debug!(?tx_digest, "Voting to accept transaction");
            }
        }

        result
    }

    #[instrument(level = "debug", skip_all, err(level = "debug"), fields(tx_digest = ?tx.digest()))]
    fn vote_transaction(
        &self,
        epoch_store: &Arc<AuthorityPerEpochStore>,
        tx: Box<Transaction>,
    ) -> SomaResult<()> {
        let tx = epoch_store.verify_transaction(*tx)?;

        self.authority_state.handle_vote_transaction(epoch_store, tx)?;

        Ok(())
    }
}

fn tx_kind_from_bytes(tx: &[u8]) -> Result<ConsensusTransactionKind, ValidationError> {
    bcs::from_bytes::<ConsensusTransaction>(tx)
        .map_err(|e| {
            ValidationError::InvalidTransaction(format!(
                "Failed to parse transaction bytes: {:?}",
                e
            ))
        })
        .map(|tx| tx.kind)
}

impl TransactionVerifier for TxValidator {
    fn verify_batch(&self, batch: &[&[u8]]) -> Result<(), ValidationError> {
        let txs: Vec<_> =
            batch.iter().map(|tx| tx_kind_from_bytes(tx)).collect::<Result<Vec<_>, _>>()?;

        self.validate_transactions(&txs)
            .map_err(|e| ValidationError::InvalidTransaction(e.to_string()))
    }

    fn verify_and_vote_batch(
        &self,
        block_ref: &BlockRef,
        batch: &[&[u8]],
    ) -> Result<Vec<TransactionIndex>, ValidationError> {
        let txs: Vec<_> =
            batch.iter().map(|tx| tx_kind_from_bytes(tx)).collect::<Result<Vec<_>, _>>()?;

        self.validate_transactions(&txs)
            .map_err(|e| ValidationError::InvalidTransaction(e.to_string()))?;

        Ok(self.vote_transactions(block_ref, txs))
    }
}
