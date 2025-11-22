use crate::{
    base::{AuthorityName, ConciseableName, TimestampMs},
    checkpoints::{CheckpointSequenceNumber, CheckpointSignatureMessage, ECMHLiveObjectSetDigest},
    committee::{EpochId, NetworkingCommittee},
    consensus::block::{BlockRef, TransactionIndex, PING_TRANSACTION_INDEX},
    crypto::AuthorityPublicKeyBytes,
    digests::{
        AdditionalConsensusStateDigest, CheckpointDigest, ConsensusCommitDigest, TransactionDigest,
    },
    encoder_committee::EncoderCommittee,
    error::SomaError,
    transaction::{CertifiedTransaction, Transaction},
};
use byteorder::{BigEndian, ReadBytesExt};
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use std::fmt::{Debug, Formatter};
use std::{collections::hash_map::DefaultHasher, hash::Hash as _, hash::Hasher as _};
use validator_set::ValidatorSet;

pub mod block;
pub mod commit;
pub mod context;
pub mod leader_scoring;
pub mod stake_aggregator;
pub mod validator_set;

// TODO: Switch to using consensus_types::block::Round?
/// Consensus round number in u64 instead of u32.
pub type Round = u64;

/// The position of a transaction in consensus.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ConsensusPosition {
    // Epoch of the consensus instance.
    pub epoch: EpochId,
    // Block containing a transaction.
    pub block: BlockRef,
    // Index of the transaction in the block.
    pub index: TransactionIndex,
}

impl ConsensusPosition {
    pub fn into_raw(self) -> Result<Bytes, SomaError> {
        bcs::to_bytes(&self)
            .map_err(|e| {
                SomaError::GrpcMessageSerializeError {
                    type_info: "ConsensusPosition".to_string(),
                    error: e.to_string(),
                }
                .into()
            })
            .map(Bytes::from)
    }

    // We reserve the max index for the "ping" transaction. This transaction is not included in the block, but we are
    // simulating by assuming its position in the block as the max index.
    pub fn ping(epoch: EpochId, block: BlockRef) -> Self {
        Self {
            epoch,
            block,
            index: PING_TRANSACTION_INDEX,
        }
    }
}

impl TryFrom<&[u8]> for ConsensusPosition {
    type Error = SomaError;

    fn try_from(bytes: &[u8]) -> Result<Self, Self::Error> {
        bcs::from_bytes(bytes).map_err(|e| {
            SomaError::GrpcMessageDeserializeError {
                type_info: "ConsensusPosition".to_string(),
                error: e.to_string(),
            }
            .into()
        })
    }
}

impl std::fmt::Display for ConsensusPosition {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "P(E{}, {}, {})", self.epoch, self.block, self.index)
    }
}

impl std::fmt::Debug for ConsensusPosition {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "P(E{}, {:?}, {})", self.epoch, self.block, self.index)
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ConsensusTransaction {
    /// Encodes an u64 unique tracking id to allow us trace a message between Sui and consensus.
    /// Use an byte array instead of u64 to ensure stable serialization.
    pub tracking_id: [u8; 8],
    /// The specific type of consensus transaction
    pub kind: ConsensusTransactionKind,
}

impl ConsensusTransaction {
    /// Displays a ConsensusTransaction created locally by the validator, for example during submission to consensus.
    pub fn local_display(&self) -> String {
        match &self.kind {
            ConsensusTransactionKind::CertifiedTransaction(cert) => {
                format!("Certified({})", cert.digest())
            }
            ConsensusTransactionKind::CheckpointSignature(data) => {
                format!(
                    "CkptSig({}, {})",
                    data.summary.sequence_number,
                    data.summary.digest()
                )
            }
            ConsensusTransactionKind::EndOfPublish(..) => "EOP".to_string(),
            ConsensusTransactionKind::UserTransaction(tx) => {
                format!("User({})", tx.digest())
            }
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum ConsensusTransactionKind {
    /// A user-submitted transaction that has been certified by a quorum of validators
    CertifiedTransaction(Box<CertifiedTransaction>),

    UserTransaction(Box<Transaction>),

    /// A message indicating that an authority has no more transactions to publish in this epoch
    /// Used for consensus liveness and epoch boundary detection
    EndOfPublish(AuthorityName),

    CheckpointSignature(Box<CheckpointSignatureMessage>),
}

#[derive(Serialize, Deserialize, Clone, Hash, PartialEq, Eq, Ord, PartialOrd)]
pub enum ConsensusTransactionKey {
    /// Key for a user transaction certificate, identified by its digest
    Certificate(TransactionDigest),

    CheckpointSignature(AuthorityName, CheckpointSequenceNumber, CheckpointDigest),

    /// Key for an end-of-publish message, identified by the authority that sent it
    EndOfPublish(AuthorityName),
}

impl Debug for ConsensusTransactionKey {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Certificate(digest) => write!(f, "Certificate({:?})", digest),
            Self::CheckpointSignature(name, seq, digest) => write!(
                f,
                "CheckpointSignature({:?}, {:?}, {:?})",
                name.concise(),
                seq,
                digest
            ),
            Self::EndOfPublish(name) => write!(f, "EndOfPublish({:?})", name.concise()),
        }
    }
}

impl ConsensusTransaction {
    pub fn new_certificate_message(
        authority: &AuthorityName,
        certificate: CertifiedTransaction,
    ) -> Self {
        let mut hasher = DefaultHasher::new();
        let tx_digest = certificate.digest();
        tx_digest.hash(&mut hasher);
        authority.hash(&mut hasher);
        let tracking_id = hasher.finish().to_le_bytes();
        Self {
            tracking_id,
            kind: ConsensusTransactionKind::CertifiedTransaction(Box::new(certificate)),
        }
    }

    pub fn new_user_transaction_message(authority: &AuthorityName, tx: Transaction) -> Self {
        let mut hasher = DefaultHasher::new();
        let tx_digest = tx.digest();
        tx_digest.hash(&mut hasher);
        authority.hash(&mut hasher);
        let tracking_id = hasher.finish().to_le_bytes();
        Self {
            tracking_id,
            kind: ConsensusTransactionKind::UserTransaction(Box::new(tx)),
        }
    }

    pub fn new_checkpoint_signature_message(data: CheckpointSignatureMessage) -> Self {
        let mut hasher = DefaultHasher::new();
        data.summary.auth_sig().signature.hash(&mut hasher);
        let tracking_id = hasher.finish().to_le_bytes();
        Self {
            tracking_id,
            kind: ConsensusTransactionKind::CheckpointSignature(Box::new(data)),
        }
    }

    pub fn new_end_of_publish(authority: AuthorityName) -> Self {
        let mut hasher = DefaultHasher::new();
        authority.hash(&mut hasher);
        let tracking_id = hasher.finish().to_le_bytes();
        Self {
            tracking_id,
            kind: ConsensusTransactionKind::EndOfPublish(authority),
        }
    }

    pub fn new_mysticeti_certificate(
        round: u64,
        offset: u64,
        certificate: CertifiedTransaction,
    ) -> Self {
        let mut hasher = DefaultHasher::new();
        let tx_digest = certificate.digest();
        tx_digest.hash(&mut hasher);
        round.hash(&mut hasher);
        offset.hash(&mut hasher);
        let tracking_id = hasher.finish().to_le_bytes();
        Self {
            tracking_id,
            kind: ConsensusTransactionKind::CertifiedTransaction(Box::new(certificate)),
        }
    }

    pub fn get_tracking_id(&self) -> u64 {
        (&self.tracking_id[..])
            .read_u64::<BigEndian>()
            .unwrap_or_default()
    }

    pub fn key(&self) -> ConsensusTransactionKey {
        match &self.kind {
            ConsensusTransactionKind::CertifiedTransaction(cert) => {
                ConsensusTransactionKey::Certificate(*cert.digest())
            }
            ConsensusTransactionKind::CheckpointSignature(data) => {
                ConsensusTransactionKey::CheckpointSignature(
                    data.summary.auth_sig().authority,
                    data.summary.sequence_number,
                    *data.summary.digest(),
                )
            }
            ConsensusTransactionKind::EndOfPublish(authority) => {
                ConsensusTransactionKey::EndOfPublish(*authority)
            }
            ConsensusTransactionKind::UserTransaction(tx) => {
                // Use the same key format as ConsensusTransactionKind::CertifiedTransaction,
                // because existing usages of ConsensusTransactionKey should not differentiate
                // between CertifiedTransaction and UserTransaction.
                ConsensusTransactionKey::Certificate(*tx.digest())
            }
        }
    }

    pub fn is_user_transaction(&self) -> bool {
        matches!(
            self.kind,
            ConsensusTransactionKind::UserTransaction(_)
                | ConsensusTransactionKind::CertifiedTransaction(_)
        )
    }

    pub fn is_mfp_transaction(&self) -> bool {
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

    /// The sub DAG index of the consensus commit
    /// This field will be populated if there are multiple consensus commits per round
    pub sub_dag_index: Option<u64>,

    /// Unix timestamp from consensus (in milliseconds)
    pub commit_timestamp_ms: TimestampMs,

    /// Digest of consensus output for verification
    pub consensus_commit_digest: ConsensusCommitDigest,

    /// Digest of any additional state computed by the consensus handler.
    /// Used to detect forking bugs as early as possible.
    pub additional_state_digest: AdditionalConsensusStateDigest,
}

pub trait EndOfEpochAPI: Send + Sync + 'static {
    fn get_next_epoch_state(
        &self,
    ) -> Option<(
        ValidatorSet,
        EncoderCommittee,
        NetworkingCommittee,
        ECMHLiveObjectSetDigest,
        u64,
    )>;
}

pub struct TestEpochStore {
    /// The next epoch state, if one has been computed
    pub next_epoch_state: Option<(
        ValidatorSet,
        EncoderCommittee,
        NetworkingCommittee,
        ECMHLiveObjectSetDigest,
        u64,
    )>,
}

impl TestEpochStore {
    pub fn new() -> Self {
        Self {
            next_epoch_state: None,
        }
    }

    pub fn set_next_epoch_state(
        &mut self,
        state: (
            ValidatorSet,
            EncoderCommittee,
            NetworkingCommittee,
            ECMHLiveObjectSetDigest,
            u64,
        ),
    ) {
        self.next_epoch_state = Some(state);
    }
}

impl EndOfEpochAPI for TestEpochStore {
    fn get_next_epoch_state(
        &self,
    ) -> Option<(
        ValidatorSet,
        EncoderCommittee,
        NetworkingCommittee,
        ECMHLiveObjectSetDigest,
        u64,
    )> {
        self.next_epoch_state.clone()
    }
}
