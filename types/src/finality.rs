use crate::checkpoints::{
    CertifiedCheckpointSummary, CheckpointContents, CheckpointSequenceNumber,
};
use crate::committee::Committee;
use crate::digests::{CheckpointDigest, TransactionDigest};
use crate::effects::{TransactionEffects, TransactionEffectsAPI};
use crate::envelope::Message as _;
use crate::error::{SomaError, SomaResult};
use crate::transaction::Transaction;
use serde::{Deserialize, Serialize};

/// Proof that a transaction has achieved finality through checkpoint inclusion.
///
/// This proof is verifiable by any party with access to the validator committee,
/// and provides the entropy source for deterministic shard selection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinalityProof {
    /// The original user-signed transaction
    pub transaction: Transaction,
    /// The transaction effects (proves execution outcome)
    pub effects: TransactionEffects,
    /// The certified checkpoint containing this transaction
    pub checkpoint: CertifiedCheckpointSummary,
    /// The checkpoint contents (for verifying transaction inclusion)
    ///
    /// Note: In the future, this could be replaced with a merkle proof
    /// for more efficient verification of large checkpoints.
    pub checkpoint_contents: CheckpointContents,
}

impl FinalityProof {
    pub fn new(
        transaction: Transaction,
        effects: TransactionEffects,
        checkpoint: CertifiedCheckpointSummary,
        checkpoint_contents: CheckpointContents,
    ) -> Self {
        Self {
            transaction,
            effects,
            checkpoint,
            checkpoint_contents,
        }
    }

    /// Get the transaction digest
    pub fn transaction_digest(&self) -> &TransactionDigest {
        self.transaction.digest()
    }

    /// Get the checkpoint digest - used as VDF entropy source
    pub fn checkpoint_digest(&self) -> &CheckpointDigest {
        self.checkpoint.digest()
    }

    /// Get the checkpoint sequence number
    pub fn checkpoint_sequence_number(&self) -> CheckpointSequenceNumber {
        self.checkpoint.data().sequence_number
    }

    /// Get the epoch this finality proof is for
    pub fn epoch(&self) -> u64 {
        self.checkpoint.data().epoch
    }

    /// Verify the finality proof against the authority committee.
    ///
    /// This verifies:
    /// 1. The checkpoint is properly certified by 2f+1 validators
    /// 2. The checkpoint contents match the certified content digest
    /// 3. The transaction is included in the checkpoint contents
    /// 4. The transaction digest matches the effects
    /// 5. The transaction executed successfully
    pub fn verify(&self, committee: &Committee) -> SomaResult<()> {
        // 1. Verify checkpoint is properly certified by the committee
        self.checkpoint.verify_authority_signatures(committee)?;

        // 2. Verify checkpoint contents match the certified content digest
        let computed_content_digest = *self.checkpoint_contents.digest();
        if computed_content_digest != self.checkpoint.data().content_digest {
            return Err(SomaError::InvalidFinalityProof(format!(
                "checkpoint contents digest mismatch: expected {}, got {}",
                self.checkpoint.data().content_digest,
                computed_content_digest
            )));
        }

        // 3. Verify transaction is included in checkpoint contents
        let tx_digest = self.transaction.digest();
        let effects_digest = self.effects.digest();

        let found = self.checkpoint_contents.iter().any(|exec_digests| {
            &exec_digests.transaction == tx_digest && exec_digests.effects == effects_digest
        });

        if !found {
            return Err(SomaError::InvalidFinalityProof(
                "transaction not found in checkpoint contents".to_string(),
            ));
        }

        // 4. Verify transaction digest matches effects
        if tx_digest != self.effects.transaction_digest() {
            return Err(SomaError::InvalidFinalityProof(
                "transaction digest does not match effects".to_string(),
            ));
        }

        // 5. Verify execution was successful
        if !self.effects.status().is_ok() {
            return Err(SomaError::InvalidFinalityProof(
                "transaction execution was not successful".to_string(),
            ));
        }

        // 6. Verify epoch consistency
        if self.checkpoint.data().epoch != self.effects.executed_epoch() {
            return Err(SomaError::InvalidFinalityProof(
                "checkpoint epoch does not match effects epoch".to_string(),
            ));
        }

        Ok(())
    }
}
