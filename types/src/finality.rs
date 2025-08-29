use crate::{
    committee::Committee,
    consensus::block::BlockRef,
    crypto::{AuthoritySignInfo, AuthoritySignInfoTrait, AuthorityStrongQuorumSignInfo},
    digests::TransactionDigest,
    effects::ExecutionStatus,
    envelope::{Envelope, Message, VerifiedEnvelope},
    error::SomaResult,
    intent::{Intent, IntentScope},
    transaction::Transaction,
};
use serde::{Deserialize, Serialize};
use shared::entropy::{BlockEntropy, BlockEntropyProof};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ConsensusFinality {
    pub tx_digest: TransactionDigest,
    pub leader_block: BlockRef,
    pub execution_status: ExecutionStatus,
}

impl Message for ConsensusFinality {
    type DigestType = TransactionDigest;
    const SCOPE: IntentScope = IntentScope::ConsensusFinality;

    fn digest(&self) -> Self::DigestType {
        self.tx_digest
    }
}

pub type ConsensusFinalityEnvelope<S> = Envelope<ConsensusFinality, S>;
pub type SignedConsensusFinality = ConsensusFinalityEnvelope<AuthoritySignInfo>;
pub type CertifiedConsensusFinality = ConsensusFinalityEnvelope<AuthorityStrongQuorumSignInfo>;

pub type VerifiedConsensusFinalityEnvelope<S> = VerifiedEnvelope<ConsensusFinality, S>;
pub type VerifiedSignedConsensusFinality = VerifiedConsensusFinalityEnvelope<AuthoritySignInfo>;
pub type VerifiedCertifiedConsensusFinality =
    VerifiedConsensusFinalityEnvelope<AuthorityStrongQuorumSignInfo>;

impl CertifiedConsensusFinality {
    pub fn verify_authority_signatures(&self, committee: &Committee) -> SomaResult {
        self.auth_sig()
            .verify_secure(self.data(), Intent::soma_transaction(), committee)
    }

    pub fn verify(self, committee: &Committee) -> SomaResult<VerifiedCertifiedConsensusFinality> {
        self.verify_authority_signatures(committee)?;
        Ok(VerifiedCertifiedConsensusFinality::new_from_verified(self))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinalityProof {
    /// The original user-signed transaction with inputs
    pub transaction: Transaction,

    /// The certified consensus finality containing the leader block reference
    pub consensus_finality: CertifiedConsensusFinality,

    /// The entropy generated from the leader block
    pub block_entropy: BlockEntropy,

    /// The proof of the entropy generation
    pub block_entropy_proof: BlockEntropyProof,
}

impl FinalityProof {
    pub fn new(
        transaction: Transaction,
        consensus_finality: CertifiedConsensusFinality,
        block_entropy: BlockEntropy,
        block_entropy_proof: BlockEntropyProof,
    ) -> Self {
        Self {
            transaction,
            consensus_finality,
            block_entropy,
            block_entropy_proof,
        }
    }

    /// Get the transaction digest for this finality proof
    pub fn transaction_digest(&self) -> &TransactionDigest {
        self.transaction.digest()
    }

    /// Get the block reference from the consensus finality
    pub fn block_ref(&self) -> &BlockRef {
        &self.consensus_finality.data().leader_block
    }
}
