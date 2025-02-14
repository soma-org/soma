use crate::{
    accumulator::Accumulator,
    base::{AuthorityName, ConciseableName},
    crypto::AuthorityPublicKeyBytes,
    digests::{ConsensusCommitDigest, ECMHLiveObjectSetDigest, TransactionDigest},
    state_sync::CommitTimestamp,
    transaction::CertifiedTransaction,
};
use serde::{Deserialize, Serialize};
use std::fmt::{Debug, Formatter};
use validator_set::ValidatorSet;

pub mod block;
pub mod block_verifier;
pub mod commit;
pub mod committee;
pub mod context;
pub mod stake_aggregator;
pub mod transaction;
pub mod validator_set;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ConsensusTransaction {
    pub kind: ConsensusTransactionKind,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum ConsensusTransactionKind {
    UserTransaction(Box<CertifiedTransaction>),
    // CheckpointSignature(Box<CheckpointSignatureMessage>),
    EndOfPublish(AuthorityName),
}

#[derive(Serialize, Deserialize, Clone, Hash, PartialEq, Eq, Ord, PartialOrd)]
pub enum ConsensusTransactionKey {
    Certificate(TransactionDigest),
    // CheckpointSignature(AuthorityName, CheckpointSequenceNumber),
    EndOfPublish(AuthorityName),
}

impl Debug for ConsensusTransactionKey {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Certificate(digest) => write!(f, "Certificate({:?})", digest),
            // Self::CheckpointSignature(name, seq) => {
            //     write!(f, "CheckpointSignature({:?}, {:?})", name.concise(), seq)
            // }
            Self::EndOfPublish(name) => write!(f, "EndOfPublish({:?})", name.concise()),
        }
    }
}

impl ConsensusTransaction {
    pub fn new_certificate_message(
        authority: &AuthorityName,
        certificate: CertifiedTransaction,
    ) -> Self {
        Self {
            kind: ConsensusTransactionKind::UserTransaction(Box::new(certificate)),
        }
    }

    // pub fn new_checkpoint_signature_message(data: CheckpointSignatureMessage) -> Self {
    //     Self {
    //         kind: ConsensusTransactionKind::CheckpointSignature(Box::new(data)),
    //     }
    // }

    pub fn new_end_of_publish(authority: AuthorityName) -> Self {
        Self {
            kind: ConsensusTransactionKind::EndOfPublish(authority),
        }
    }

    pub fn new_mysticeti_certificate(
        round: u64,
        offset: u64,
        certificate: CertifiedTransaction,
    ) -> Self {
        Self {
            kind: ConsensusTransactionKind::UserTransaction(Box::new(certificate)),
        }
    }

    pub fn key(&self) -> ConsensusTransactionKey {
        match &self.kind {
            ConsensusTransactionKind::UserTransaction(cert) => {
                ConsensusTransactionKey::Certificate(*cert.digest())
            }
            // ConsensusTransactionKind::CheckpointSignature(data) => {
            //     ConsensusTransactionKey::CheckpointSignature(
            //         data.summary.auth_sig().authority,
            //         data.summary.sequence_number,
            //     )
            // }
            ConsensusTransactionKind::EndOfPublish(authority) => {
                ConsensusTransactionKey::EndOfPublish(*authority)
            }
        }
    }

    pub fn is_user_certificate(&self) -> bool {
        matches!(self.kind, ConsensusTransactionKind::UserTransaction(_))
    }

    pub fn is_end_of_publish(&self) -> bool {
        matches!(self.kind, ConsensusTransactionKind::EndOfPublish(_))
    }
}

#[derive(Debug, PartialEq, Eq, Hash, Clone, Serialize, Deserialize)]
pub struct ConsensusCommitPrologue {
    /// Epoch of the commit prologue transaction
    pub epoch: u64,
    /// Consensus round of the commit
    pub round: u64,
    /// The sub DAG index of the consensus commit. This field will be populated if there
    /// are multiple consensus commits per round.
    pub sub_dag_index: Option<u64>,
    /// Unix timestamp from consensus
    pub commit_timestamp_ms: CommitTimestamp,
    /// Digest of consensus output
    pub consensus_commit_digest: ConsensusCommitDigest,
}

pub trait EndOfEpochAPI: Send + Sync + 'static {
    /// Returns the committee for the next epoch if one has been computed, and epoch state digest
    fn get_next_epoch_state(&self) -> Option<(ValidatorSet, ECMHLiveObjectSetDigest)>;
}

pub struct TestEpochStore {
    pub next_epoch_state: Option<(ValidatorSet, ECMHLiveObjectSetDigest)>,
}

impl TestEpochStore {
    pub fn new() -> Self {
        Self {
            next_epoch_state: None,
        }
    }

    pub fn set_next_epoch_state(&mut self, state: (ValidatorSet, ECMHLiveObjectSetDigest)) {
        self.next_epoch_state = Some(state);
    }
}

impl EndOfEpochAPI for TestEpochStore {
    fn get_next_epoch_state(&self) -> Option<(ValidatorSet, ECMHLiveObjectSetDigest)> {
        self.next_epoch_state.clone()
    }
}
