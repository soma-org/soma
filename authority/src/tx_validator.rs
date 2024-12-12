use std::sync::Arc;

use eyre::Context;
use tap::TapFallible;
use tracing::{info, warn};
use types::consensus::{ConsensusTransaction, ConsensusTransactionKind};

use types::consensus::transaction::{TransactionVerifier, ValidationError};

use crate::epoch_store::AuthorityPerEpochStore;

/// Allows verifying the validity of transactions
#[derive(Clone)]
pub struct TxValidator {
    epoch_store: Arc<AuthorityPerEpochStore>,
}

impl TxValidator {
    pub fn new(epoch_store: Arc<AuthorityPerEpochStore>) -> Self {
        info!(
            "SuiTxValidator constructed for epoch {}",
            epoch_store.epoch()
        );
        Self { epoch_store }
    }

    fn validate_transactions(
        &self,
        txs: Vec<ConsensusTransactionKind>,
    ) -> Result<(), eyre::Report> {
        let mut cert_batch = Vec::new();
        for tx in txs.into_iter() {
            match tx {
                ConsensusTransactionKind::UserTransaction(certificate) => {
                    cert_batch.push(*certificate);

                    // if !certificate.contains_shared_object() {
                    //     // new_unchecked safety: we do not use the certs in this list until all
                    //     // have had their signatures verified.
                    //     owned_tx_certs.push(VerifiedCertificate::new_unchecked(*certificate));
                    // }
                }
                // ConsensusTransactionKind::CheckpointSignature(signature) => {
                //     ckpt_messages.push(signature.clone());
                //     ckpt_batch.push(signature.summary);
                // }
                ConsensusTransactionKind::EndOfPublish(_) => {}
            }
        }

        // verify the certificate signatures as a batch
        let cert_count = cert_batch.len();

        self.epoch_store
            .signature_verifier
            .verify_certs(cert_batch)
            .tap_err(|e| warn!("batch verification error: {}", e))
            .wrap_err("Malformed batch (failed to verify)")?;

        // All checkpoint sigs have been verified, forward them to the checkpoint service
        // for ckpt in ckpt_messages {
        //     self.checkpoint_service
        //         .notify_checkpoint_signature(&self.epoch_store, &ckpt)?;
        // }

        Ok(())

        // todo - we should un-comment line below once we have a way to revert those transactions at the end of epoch
        // all certificates had valid signatures, schedule them for execution prior to sequencing
        // which is unnecessary for owned object transactions.
        // It is unnecessary to write to pending_certificates table because the certs will be written
        // via consensus output.
        // self.transaction_manager
        //     .enqueue_certificates(owned_tx_certs, &self.epoch_store)
        //     .wrap_err("Failed to schedule certificates for execution")
    }
}

fn tx_from_bytes(tx: &[u8]) -> Result<ConsensusTransaction, eyre::Report> {
    bcs::from_bytes::<ConsensusTransaction>(tx)
        .wrap_err("Malformed transaction (failed to deserialize)")
}

impl TransactionVerifier for TxValidator {
    fn verify_batch(&self, batch: &[&[u8]]) -> Result<(), ValidationError> {
        let txs = batch
            .iter()
            .map(|tx| {
                tx_from_bytes(tx)
                    .map(|tx| tx.kind)
                    .map_err(|e| ValidationError::InvalidTransaction(e.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?;

        self.validate_transactions(txs)
            .map_err(|e| ValidationError::InvalidTransaction(e.to_string()))
    }
}
