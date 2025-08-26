use crate::{
    accumulator::CommitIndex,
    base::AuthorityName,
    committee::{Committee, EpochId},
    consensus::block::BlockRef,
    crypto::{AuthoritySignInfo, AuthoritySignInfoTrait, AuthorityStrongQuorumSignInfo},
    digests::TransactionDigest,
    effects::ExecutionStatus,
    envelope::{Envelope, Message, VerifiedEnvelope},
    error::SomaResult,
    intent::{Intent, IntentScope},
};
use serde::{Deserialize, Serialize};

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
