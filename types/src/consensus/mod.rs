use crate::{
    base::{AuthorityName, ConciseableName, TimestampMs},
    checkpoints::{CheckpointSequenceNumber, CheckpointSignatureMessage, ECMHLiveObjectSetDigest},
    committee::EpochId,
    consensus::block::{BlockRef, PING_TRANSACTION_INDEX, TransactionIndex},
    crypto::AuthorityPublicKeyBytes,
    digests::{
        AdditionalConsensusStateDigest, CheckpointDigest, ConsensusCommitDigest, TransactionDigest,
    },
    error::SomaError,
    supported_protocol_versions::{SupportedProtocolVersions, SupportedProtocolVersionsWithHashes},
    transaction::{CertifiedTransaction, Transaction},
};
use byteorder::{BigEndian, ReadBytesExt};
use bytes::Bytes;
use protocol_config::Chain;
use serde::{Deserialize, Serialize};
use std::{collections::hash_map::DefaultHasher, hash::Hash as _, hash::Hasher as _};
use std::{
    fmt::{Debug, Formatter},
    time::{SystemTime, UNIX_EPOCH},
};
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
            .map_err(|e| SomaError::GrpcMessageSerializeError {
                type_info: "ConsensusPosition".to_string(),
                error: e.to_string(),
            })
            .map(Bytes::from)
    }

    // We reserve the max index for the "ping" transaction. This transaction is not included in the block, but we are
    // simulating by assuming its position in the block as the max index.
    pub fn ping(epoch: EpochId, block: BlockRef) -> Self {
        Self { epoch, block, index: PING_TRANSACTION_INDEX }
    }
}

impl TryFrom<&[u8]> for ConsensusPosition {
    type Error = SomaError;

    fn try_from(bytes: &[u8]) -> Result<Self, Self::Error> {
        bcs::from_bytes(bytes).map_err(|e| SomaError::GrpcMessageDeserializeError {
            type_info: "ConsensusPosition".to_string(),
            error: e.to_string(),
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
                format!("CkptSig({}, {})", data.summary.sequence_number, data.summary.digest())
            }
            ConsensusTransactionKind::EndOfPublish(..) => "EOP".to_string(),
            ConsensusTransactionKind::UserTransaction(tx) => {
                format!("User({})", tx.digest())
            }
            ConsensusTransactionKind::CapabilityNotification(..) => "Cap".to_string(),
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

    CapabilityNotification(AuthorityCapabilities),
}

#[derive(Serialize, Deserialize, Clone, Hash, PartialEq, Eq, Ord, PartialOrd)]
pub enum ConsensusTransactionKey {
    /// Key for a user transaction certificate, identified by its digest
    Certificate(TransactionDigest),

    CheckpointSignature(AuthorityName, CheckpointSequenceNumber, CheckpointDigest),

    /// Key for an end-of-publish message, identified by the authority that sent it
    EndOfPublish(AuthorityName),

    CapabilityNotification(AuthorityName, u64 /* generation */),
}

impl Debug for ConsensusTransactionKey {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Certificate(digest) => write!(f, "Certificate({:?})", digest),
            Self::CheckpointSignature(name, seq, digest) => {
                write!(f, "CheckpointSignature({:?}, {:?}, {:?})", name.concise(), seq, digest)
            }
            Self::EndOfPublish(name) => write!(f, "EndOfPublish({:?})", name.concise()),
            Self::CapabilityNotification(name, generation) => {
                write!(f, "CapabilityNotification({:?}, {:?})", name.concise(), generation)
            }
        }
    }
}

/// Used to advertise capabilities of each authority via consensus. This allows validators to
/// negotiate the creation of the ChangeEpoch transaction.
#[derive(Serialize, Deserialize, Clone, Hash)]
pub struct AuthorityCapabilities {
    /// Originating authority - must match transaction source authority from consensus.
    pub authority: AuthorityName,
    /// Generation number set by sending authority. Used to determine which of multiple
    /// AuthorityCapabilities messages from the same authority is the most recent.
    ///
    /// (Currently, we just set this to the current time in milliseconds since the epoch, but this
    /// should not be interpreted as a timestamp.)
    pub generation: u64,

    /// ProtocolVersions that the authority supports.
    pub supported_protocol_versions: SupportedProtocolVersionsWithHashes,
}

impl Debug for AuthorityCapabilities {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AuthorityCapabilities")
            .field("authority", &self.authority.concise())
            .field("generation", &self.generation)
            .field("supported_protocol_versions", &self.supported_protocol_versions)
            .finish()
    }
}

impl AuthorityCapabilities {
    pub fn new(
        authority: AuthorityName,
        chain: Chain,
        supported_protocol_versions: SupportedProtocolVersions,
    ) -> Self {
        let generation = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Soma did not exist prior to 1970")
            .as_millis()
            .try_into()
            .expect("This build of soma is not supported in the year 500,000,000");
        Self {
            authority,
            generation,
            supported_protocol_versions:
                SupportedProtocolVersionsWithHashes::from_supported_versions(
                    supported_protocol_versions,
                    chain,
                ),
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
        Self { tracking_id, kind: ConsensusTransactionKind::UserTransaction(Box::new(tx)) }
    }

    pub fn new_checkpoint_signature_message(data: CheckpointSignatureMessage) -> Self {
        let mut hasher = DefaultHasher::new();
        data.summary.auth_sig().signature.hash(&mut hasher);
        let tracking_id = hasher.finish().to_le_bytes();
        Self { tracking_id, kind: ConsensusTransactionKind::CheckpointSignature(Box::new(data)) }
    }

    pub fn new_capability_notification(capabilities: AuthorityCapabilities) -> Self {
        let mut hasher = DefaultHasher::new();
        capabilities.hash(&mut hasher);
        let tracking_id = hasher.finish().to_le_bytes();
        Self { tracking_id, kind: ConsensusTransactionKind::CapabilityNotification(capabilities) }
    }

    pub fn new_end_of_publish(authority: AuthorityName) -> Self {
        let mut hasher = DefaultHasher::new();
        authority.hash(&mut hasher);
        let tracking_id = hasher.finish().to_le_bytes();
        Self { tracking_id, kind: ConsensusTransactionKind::EndOfPublish(authority) }
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
        (&self.tracking_id[..]).read_u64::<BigEndian>().unwrap_or_default()
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
            ConsensusTransactionKind::CapabilityNotification(cap) => {
                ConsensusTransactionKey::CapabilityNotification(cap.authority, cap.generation)
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
    fn get_next_epoch_state(&self) -> Option<(ValidatorSet, ECMHLiveObjectSetDigest, u64)>;
}

#[derive(Default)]
pub struct TestEpochStore {
    /// The next epoch state, if one has been computed
    pub next_epoch_state: Option<(ValidatorSet, ECMHLiveObjectSetDigest, u64)>,
}

impl TestEpochStore {
    pub fn new() -> Self {
        Self { next_epoch_state: None }
    }

    pub fn set_next_epoch_state(&mut self, state: (ValidatorSet, ECMHLiveObjectSetDigest, u64)) {
        self.next_epoch_state = Some(state);
    }
}

impl EndOfEpochAPI for TestEpochStore {
    fn get_next_epoch_state(&self) -> Option<(ValidatorSet, ECMHLiveObjectSetDigest, u64)> {
        self.next_epoch_state.clone()
    }
}
