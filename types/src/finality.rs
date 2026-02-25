// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use crate::checkpoints::{
    CertifiedCheckpointSummary, CheckpointContents, CheckpointInclusionProof,
    CheckpointSequenceNumber,
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
    /// Merkle proof of transaction inclusion in checkpoint contents
    pub inclusion_proof: CheckpointInclusionProof,
}

impl FinalityProof {
    pub fn new(
        transaction: Transaction,
        effects: TransactionEffects,
        checkpoint: CertifiedCheckpointSummary,
        inclusion_proof: CheckpointInclusionProof,
    ) -> Self {
        Self { transaction, effects, checkpoint, inclusion_proof }
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
    /// 2. The Merkle inclusion proof is valid against the checkpoint's content_digest
    /// 3. The proven leaf matches our transaction and effects digests
    /// 4. The transaction digest matches the effects
    /// 5. The transaction executed successfully
    /// 6. Epoch consistency between checkpoint and effects
    pub fn verify(&self, committee: &Committee) -> SomaResult<()> {
        // 1. Verify checkpoint is properly certified by the committee
        self.checkpoint.verify_authority_signatures(committee)?;

        // 2. Verify the Merkle inclusion proof against the checkpoint's content_digest
        self.inclusion_proof.verify(&self.checkpoint.data().content_digest)?;

        // 3. Verify the proven leaf matches our transaction and effects
        let tx_digest = self.transaction.digest();
        let effects_digest = self.effects.digest();

        if &self.inclusion_proof.leaf.transaction != tx_digest {
            return Err(SomaError::InvalidFinalityProof(
                "Inclusion proof leaf transaction doesn't match provided transaction".to_string(),
            ));
        }

        if self.inclusion_proof.leaf.effects != effects_digest {
            return Err(SomaError::InvalidFinalityProof(
                "Inclusion proof leaf effects doesn't match provided effects".to_string(),
            ));
        }

        // 4. Verify transaction digest matches effects
        if tx_digest != self.effects.transaction_digest() {
            return Err(SomaError::InvalidFinalityProof(
                "Transaction digest does not match effects".to_string(),
            ));
        }

        // 5. Verify execution was successful
        if !self.effects.status().is_ok() {
            return Err(SomaError::InvalidFinalityProof(
                "Transaction execution was not successful".to_string(),
            ));
        }

        // 6. Verify epoch consistency
        if self.checkpoint.data().epoch != self.effects.executed_epoch() {
            return Err(SomaError::InvalidFinalityProof(
                "Checkpoint epoch does not match effects epoch".to_string(),
            ));
        }

        Ok(())
    }
}
