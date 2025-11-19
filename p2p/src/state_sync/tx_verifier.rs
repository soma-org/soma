use eyre::Context;
use std::sync::Arc;
use tap::TapFallible;
use tracing::warn;
use types::committee::Epoch;
use types::consensus::{
    transaction::{TransactionVerifier, ValidationError},
    ConsensusTransaction, ConsensusTransactionKind,
}; // Assuming this is where SignatureVerifier is defined

pub struct TxVerifier {
    signature_verifier: Arc<SignatureVerifier>,
}

impl TxVerifier {
    pub fn new(signature_verifier: Arc<SignatureVerifier>) -> Self {
        Self { signature_verifier }
    }

    fn validate_transactions(
        &self,
        txs: Vec<ConsensusTransactionKind>,
        epoch: Option<Epoch>,
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

                ConsensusTransactionKind::EndOfPublish(_) => {}
            }
        }

        self.signature_verifier
            .verify_certs(cert_batch, epoch)
            .tap_err(|e| warn!("batch verification error: {}", e))
            .wrap_err("Malformed batch (failed to verify)")?;

        Ok(())
    }
}

impl TransactionVerifier for TxVerifier {
    fn verify_batch(&self, batch: &[&[u8]], epoch: Option<Epoch>) -> Result<(), ValidationError> {
        let txs = batch
            .iter()
            .map(|tx| {
                tx_from_bytes(tx)
                    .map(|tx| tx.kind)
                    .map_err(|e| ValidationError::InvalidTransaction(e.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?;

        self.validate_transactions(txs, epoch)
            .map_err(|e| ValidationError::InvalidTransaction(e.to_string()))
    }
}

fn tx_from_bytes(tx: &[u8]) -> Result<ConsensusTransaction, eyre::Report> {
    bcs::from_bytes::<ConsensusTransaction>(tx)
        .wrap_err("Malformed transaction (failed to deserialize)")
}
