use std::collections::BTreeMap;

#[cfg(feature = "ml")]
use burn::store::SafetensorsStoreError;
use fastcrypto::error;
use fastcrypto::{error::FastCryptoError, hash::Digest};
use serde::{Deserialize, Serialize};
#[cfg(feature = "storage")]
use store::TypedStoreError;
use strum::IntoStaticStr;
use thiserror::Error;
use tonic::Status;

use crate::checkpoints::CheckpointSequenceNumber;
use crate::committee::{AuthorityIndex, Epoch, Stake};
use crate::consensus::{
    block::{BlockRef, Round},
    commit::{Commit, CommitIndex},
};

use crate::crypto::NetworkPublicKey;
use crate::digests::CheckpointContentsDigest;
use crate::transaction::TransactionKind;
use crate::{
    base::AuthorityName,
    committee::{Committee, EpochId, VotingPower},
    digests::{ObjectDigest, TransactionDigest, TransactionEffectsDigest},
    effects::ExecutionFailureStatus,
    object::{ObjectID, ObjectRef, Version},
    peer_id::PeerId,
};

pub const TRANSACTION_NOT_FOUND_MSG_PREFIX: &str = "Could not find the referenced transaction";
pub const TRANSACTIONS_NOT_FOUND_MSG_PREFIX: &str = "Could not find the referenced transactions";

pub type SomaResult<T = ()> = Result<T, SomaError>;

#[derive(Eq, PartialEq, Clone, Debug, Serialize, Deserialize, Error, Hash)]
pub enum SomaError {
    /// Error when the committee configuration is invalid
    #[error("Invalid committee composition")]
    InvalidCommittee(String),

    // Cryptography errors.
    #[error("Signature key generation error: {0}")]
    SignatureKeyGenError(String),

    #[error("Invalid Private Key provided")]
    InvalidPrivateKey,

    /// Error when converting between key formats
    #[error("Key Conversion Error: {0}")]
    KeyConversionError(String),

    /// Error when a cryptographic signature is invalid
    #[error("Signature is not valid: {}", error)]
    InvalidSignature { error: String },

    /// Error when an address format is invalid
    #[error("Invalid address")]
    InvalidAddress,

    /// Error when a value wasn't signed by the expected signer
    #[error("Value was not signed by the correct sender: {}", error)]
    IncorrectSigner { error: String },

    /// Error from a failed RPC call
    ///
    /// These errors occur when an RPC fails and is simply the utf8 message sent in a
    /// Tonic::Status
    #[error("{1} - {0}")]
    RpcError(String, String),

    /// Error when a validator has stopped processing transactions due to epoch change
    #[error("Validator temporarily stopped processing transactions due to epoch change")]
    ValidatorHaltedAtEpochEnd,

    /// Error when attempting operations for an epoch that has already ended
    #[error("Operations for epoch {0} have ended")]
    EpochEnded(EpochId),

    /// Error when advancing to a new epoch
    #[error("Error when advancing epoch: {:?}", error)]
    AdvanceEpochError { error: String },

    /// Error when an operation times out
    #[error("Operation timed out")]
    TimeoutError,

    /// Error when local transaction execution fails in the Orchestrator
    #[error("Failed to execute transaction locally by Orchestrator: {error:?}")]
    TransactionOrchestratorLocalExecutionError { error: String },

    /// Error when submitting a transaction to consensus fails
    #[error("Failed to submit transaction to consensus: {0}")]
    FailedToSubmitToConsensus(String),

    /// Generic authority-related error
    #[error("Authority Error: {error:?}")]
    GenericAuthorityError { error: String },

    /// Error when committee information is missing for an epoch
    #[error("Missing committee information for epoch {0}")]
    MissingCommitteeAtEpoch(EpochId),

    /// Error when communication with the Quorum Driver fails
    #[error("Unable to communicate with the Quorum Driver channel: {:?}", error)]
    QuorumDriverCommunicationError { error: String },

    /// Error when transaction certificate verification fails
    #[error(
        "Failed to verify Tx certificate with executed effects, error: {error:?}, validator: \
         {validator_name:?}"
    )]
    FailedToVerifyTxCertWithExecutedEffects { validator_name: AuthorityName, error: String },
    /// Error when too many authorities report errors for an operation
    #[error("Too many authority errors were detected for {}: {:?}", action, errors)]
    TooManyIncorrectAuthorities { errors: Vec<(AuthorityName, SomaError)>, action: String },

    /// Error when a signature or certificate is from the wrong epoch
    #[error(
        "Signature or certificate from wrong epoch, expected {expected_epoch}, got {actual_epoch}"
    )]
    WrongEpoch { expected_epoch: EpochId, actual_epoch: EpochId },

    /// Error when the number of signatures doesn't match expected signers
    #[error("Expect {expected} signer signatures but got {actual}")]
    SignerSignatureNumberMismatch { expected: usize, actual: usize },

    /// Error when a required signature is missing
    #[error("Required Signature from {expected} is absent {:?}", actual)]
    SignerSignatureAbsent { expected: String, actual: Vec<String> },

    /// Error when a signature is from an unknown authority
    #[error(
        "Value was not signed by a known authority. signer: {:?}, index: {:?}, committee: \
         {committee}",
        signer,
        index
    )]
    UnknownSigner { signer: Option<String>, index: Option<u32>, committee: Box<Committee> },

    /// Error when an authenticator is invalid
    #[error("Invalid authenticator")]
    InvalidAuthenticator,

    /// Error when signatures in a certificate don't form a quorum
    #[error("Signatures in a certificate must form a quorum")]
    CertificateRequiresQuorum,

    /// Error when a validator provides multiple signatures for the same message
    #[error(
        "Validator {:?} responded multiple signatures for the same message, conflicting: {:?}",
        signer,
        conflicting_sig
    )]
    StakeAggregatorRepeatedSigner { signer: AuthorityName, conflicting_sig: bool },

    /// Error when a validator is suspected of Byzantine behavior
    #[error("Validator {authority:?} is faulty in a Byzantine manner: {reason:?}")]
    ByzantineAuthoritySuspicion { authority: AuthorityName, reason: String },

    /// Error when a digest has an invalid length
    #[error("Invalid digest length. Expected {expected}, got {actual}")]
    InvalidDigestLength { expected: usize, actual: usize },

    /// Error when failing to get a quorum of signed effects for a transaction
    #[error(
        "Failed to get a quorum of signed effects when processing transaction: {effects_map:?}"
    )]
    QuorumFailedToGetEffectsQuorumWhenProcessingTransaction {
        effects_map: BTreeMap<TransactionEffectsDigest, (Vec<AuthorityName>, VotingPower)>,
    },

    /// Error when processing a transaction certificate fails
    #[error("Transaction certificate processing failed: {err}")]
    ErrorWhileProcessingCertificate { err: String },

    /// Error when a fullnode attempts to handle a certificate (unsupported operation)
    #[error("Fullnode does not support handle_certificate")]
    FullNodeCantHandleCertificate,

    /// Error when a transaction digest is invalid
    #[error("Invalid transaction digest.")]
    InvalidTransactionDigest,

    /// Error when a referenced object cannot be found
    #[error("Could not find the referenced object {:?} at version {:?}", object_id, version)]
    ObjectNotFound { object_id: ObjectID, version: Option<Version> },
    /// Error when an object is already locked by a different transaction
    #[error(
        "Object {obj_ref:?} already locked by a different transaction: {pending_transaction:?}"
    )]
    ObjectLockConflict { obj_ref: ObjectRef, pending_transaction: TransactionDigest },

    /// Error when an object is not available for consumption due to version mismatch
    #[error(
        "Object {provided_obj_ref:?} is not available for consumption, its current version: \
         {current_version:?}"
    )]
    ObjectVersionUnavailableForConsumption { provided_obj_ref: ObjectRef, current_version: Version },

    /// Error when an object's digest doesn't match the expected value
    #[error(
        "Invalid Object digest for object {object_id:?}. Expected digest : {expected_digest:?}"
    )]
    InvalidObjectDigest { object_id: ObjectID, expected_digest: ObjectDigest },

    /// Error when a storage operation fails
    #[error("Storage error: {0}")]
    Storage(String),

    /// Error when attempting to re-initialize a transaction lock
    #[error("Attempt to re-initialize a transaction lock for objects {:?}.", refs)]
    ObjectLockAlreadyInitialized { refs: Vec<ObjectRef> },

    /// Error when failing to read or deserialize system state
    #[error("Failed to read or deserialize system state related data structures on-chain: {0}")]
    SystemStateReadError(String),

    /// Error in network configuration
    #[error("Network config error: {0:?}")]
    NetworkConfig(String),

    /// Error when failing to connect as a client
    #[error("Failed to connect as client: {0:?}")]
    NetworkClientConnection(String),

    /// Error when failing to connect as a server
    #[error("Failed to connect as server: {0:?}")]
    NetworkServerConnection(String),

    /// Error when a peer is not found
    #[error("Peer {0} not found")]
    PeerNotFound(PeerId),

    /// Error when a consensus operation fails
    #[error("Consensus error: {0}")]
    Consensus(String),

    /// Error when no committee exists for an epoch
    #[error("No committee for epoch: {0}")]
    NoCommitteeForEpoch(Epoch),

    /// Error when expecting a single owner but finding shared ownership
    #[error("Expecting a single owner, shared ownership found")]
    UnexpectedOwnerType,

    /// Error when transaction inputs contain duplicate object references
    #[error("The transaction inputs contain duplicated ObjectRef's")]
    DuplicateObjectRefInput,

    /// Error when serialization fails
    #[error("Error serializing: {0}")]
    SerializationFailure(String),

    #[error("VDF failed: {0}")]
    FailedVDF(String),

    #[error("Shard sampling failed: {0}")]
    ShardSamplingError(String),

    #[error("Invalid finality proof")]
    InvalidFinalityProof(String),

    #[error("Verified checkpoint not found for sequence number: {0}")]
    VerifiedCheckpointNotFound(CheckpointSequenceNumber),

    #[error("Verified checkpoint not found for digest: {0}")]
    VerifiedCheckpointDigestNotFound(String),

    #[error("Latest checkpoint sequence number not found")]
    LatestCheckpointSequenceNumberNotFound,

    #[error("Checkpoint contents not found for digest: {0}")]
    CheckpointContentsNotFound(CheckpointContentsDigest),

    // Errors returned by authority and client read API's
    #[error("Failure serializing transaction in the requested format: {error}")]
    TransactionSerializationError { error: String },
    #[error("Failure deserializing transaction from the provided format: {error}")]
    TransactionDeserializationError { error: String },
    #[error("Failure serializing transaction effects from the provided format: {error}")]
    TransactionEffectsSerializationError { error: String },
    #[error("Failure deserializing transaction effects from the provided format: {error}")]
    TransactionEffectsDeserializationError { error: String },
    #[error("Failure serializing object in the requested format: {error}")]
    ObjectSerializationError { error: String },
    #[error("Failure deserializing object in the requested format: {error}")]
    ObjectDeserializationError { error: String },

    #[error("Failed to perform file operation: {0}")]
    FileIOError(String),

    #[error("Failed to serialize {type_info}, error: {error}")]
    GrpcMessageSerializeError { type_info: String, error: String },

    #[error("Failed to deserialize {type_info}, error: {error}")]
    GrpcMessageDeserializeError { type_info: String, error: String },

    #[error("Invalid request: {0}")]
    InvalidRequest(String),

    #[error(
        "Total transactions size ({size}) bytes exceeds the maximum allowed ({limit}) bytes in a Soft Bundle"
    )]
    TotalTransactionSizeTooLargeInBatch { size: usize, limit: u64 },

    #[error("Use of disabled feature: {error}")]
    UnsupportedFeatureError { error: String },

    #[error("Too many requests")]
    TooManyRequests,

    #[error("Transaction was not signed by the correct sender: {}", error)]
    IncorrectUserSignature { error: String },

    #[error("Genesis transaction not found")]
    GenesisTransactionNotFound,

    #[error("unknown error: {0}")]
    Unknown(String),

    #[error("{TRANSACTION_NOT_FOUND_MSG_PREFIX} [{:?}].", digest)]
    TransactionNotFound { digest: TransactionDigest },
    #[error("{TRANSACTIONS_NOT_FOUND_MSG_PREFIX} [{:?}].", digests)]
    TransactionsNotFound { digests: Vec<TransactionDigest> },

    #[error("There are too many transactions pending in consensus")]
    TooManyTransactionsPendingConsensus,

    #[error(
        "Validator consensus rounds are lagging behind. last committed leader round: {last_committed_round}, requested round: {round}"
    )]
    ValidatorConsensusLagging { round: u32, last_committed_round: u32 },

    #[error("Error executing {0}")]
    ExecutionError(String),

    #[error("Transaction is already finalized but with different user signatures")]
    TxAlreadyFinalizedWithDifferentUserSigs,

    #[error("Transaction is denied: {error}")]
    TransactionDenied { error: String },

    #[error("Wrong initial version given for shared object")]
    SharedObjectStartingVersionMismatch,

    #[error("Object used as shared is not shared")]
    NotSharedObjectError,

    #[error("Versions above the maximal value are not usable for transfers")]
    InvalidSequenceNumber,

    #[error("Object used as owned is not owned")]
    NotOwnedObjectError,

    #[error("Mutable object {object_id} cannot appear more than one in one transaction")]
    MutableObjectUsedMoreThanOnce { object_id: ObjectID },
    #[error("Wrong number of parameters for the transaction")]
    ObjectInputArityViolation,

    #[error("Immutable parameter provided, mutable parameter expected")]
    MutableParameterExpected { object_id: ObjectID },

    #[error("Gas payment error {0}")]
    GasPaymentError(String),

    #[error("Encoder service unavailable")]
    EncoderServiceUnavailable,

    #[error("Transaction not finalized in a checkpoint")]
    TransactionNotFinalized,

    #[error("Not an EmbedData transaction")]
    NotEmbedDataTransaction,

    #[error("Transaction execution failed: {0}")]
    TransactionFailed(String),

    #[error("VDF computation failed: {0}")]
    VdfComputationFailed(String),
}

impl SomaError {
    pub fn individual_error_indicates_epoch_change(&self) -> bool {
        matches!(self, SomaError::ValidatorHaltedAtEpochEnd | SomaError::MissingCommitteeAtEpoch(_))
    }

    /// Returns if the error is retryable and if the error's retryability is
    /// explicitly categorized.
    /// There should be only a handful of retryable errors. For now we list common
    /// non-retryable error below to help us find more retryable errors in logs.
    pub fn is_retryable(&self) -> (bool, bool) {
        let retryable = match self {
            // Network error
            SomaError::RpcError { .. } => true,

            // Reconfig error
            SomaError::ValidatorHaltedAtEpochEnd => true,
            SomaError::MissingCommitteeAtEpoch(..) => true,
            SomaError::WrongEpoch { .. } => true,
            SomaError::EpochEnded(..) => true,

            SomaError::ObjectNotFound { .. } => true,

            // SomaError::PotentiallyTemporarilyInvalidSignature { .. } => true,

            // Overload errors
            // SomaError::TooManyTransactionsPendingExecution { .. } => true,
            // SomaError::TooManyTransactionsPendingOnObject { .. } => true,
            // SomaError::TooOldTransactionPendingOnObject { .. } => true,
            // SomaError::TooManyTransactionsPendingConsensus => true,
            // SomaError::ValidatorOverloadedRetryAfter { .. } => true,

            // Non retryable error
            SomaError::ExecutionError(..) => false,
            SomaError::ByzantineAuthoritySuspicion { .. } => false,
            SomaError::QuorumFailedToGetEffectsQuorumWhenProcessingTransaction { .. } => false,
            SomaError::TxAlreadyFinalizedWithDifferentUserSigs => false,
            SomaError::FailedToVerifyTxCertWithExecutedEffects { .. } => false,
            SomaError::ObjectLockConflict { .. } => false,

            // NB: This is not an internal overload, but instead an imposed rate
            // limit / blocking of a client. It must be non-retryable otherwise
            // we will make the threat worse through automatic retries.
            SomaError::TooManyRequests => false,

            // For all un-categorized errors, return here with categorized = false.
            _ => return (false, false),
        };

        (retryable, true)
    }

    pub fn is_object_not_found(&self) -> bool {
        matches!(self, SomaError::ObjectNotFound { .. })
    }

    // pub fn is_overload(&self) -> bool {
    //     matches!(
    //         self,
    //         SomaError::TooManyTransactionsPendingExecution { .. }
    //             | SomaError::TooManyTransactionsPendingOnObject { .. }
    //             | SomaError::TooOldTransactionPendingOnObject { .. }
    //             | SomaError::TooManyTransactionsPendingConsensus
    //     )
    // }

    // pub fn is_retryable_overload(&self) -> bool {
    //     matches!(self, SomaError::ValidatorOverloadedRetryAfter { .. })
    // }

    // pub fn retry_after_secs(&self) -> u64 {
    //     match self {
    //         SomaError::ValidatorOverloadedRetryAfter { retry_after_secs } => *retry_after_secs,
    //         _ => 0,
    //     }
    // }

    /// Categorizes SuiError into ErrorCategory.
    pub fn categorize(&self) -> ErrorCategory {
        match self {
            SomaError::ObjectNotFound { .. } => ErrorCategory::Aborted,

            SomaError::InvalidSignature { .. }
            | SomaError::SignerSignatureAbsent { .. }
            | SomaError::SignerSignatureNumberMismatch { .. }
            | SomaError::IncorrectSigner { .. }
            | SomaError::UnknownSigner { .. } => ErrorCategory::InvalidTransaction,

            SomaError::ObjectLockConflict { .. } => ErrorCategory::LockConflict,

            // Using a stale object version is a permanent error — retrying won't help.
            SomaError::ObjectVersionUnavailableForConsumption { .. } => {
                ErrorCategory::InvalidTransaction
            }

            SomaError::Unknown { .. }
            | SomaError::GrpcMessageSerializeError { .. }
            | SomaError::GrpcMessageDeserializeError { .. }
            | SomaError::ByzantineAuthoritySuspicion { .. }
            | SomaError::UnsupportedFeatureError { .. }
            | SomaError::InvalidRequest { .. } => ErrorCategory::Internal,

            // SomaError::TooManyTransactionsPendingExecution { .. }
            // | SomaError::TooManyTransactionsPendingOnObject { .. }
            // | SomaError::TooOldTransactionPendingOnObject { .. }
            // | SomaError::TooManyTransactionsPendingConsensus
            // | SomaError::ValidatorOverloadedRetryAfter { .. } => ErrorCategory::ValidatorOverloaded,
            SomaError::TimeoutError => ErrorCategory::Unavailable,

            // Other variants are assumed to be retriable with new transaction submissions.
            _ => ErrorCategory::Aborted,
        }
    }
}

impl From<Status> for SomaError {
    fn from(status: Status) -> Self {
        // if status.message() == "Too many requests" {
        //     return Self::TooManyRequests;
        // }

        let result = bcs::from_bytes::<SomaError>(status.details());
        if let Ok(error) = result {
            error
        } else {
            Self::RpcError(status.message().to_owned(), status.code().description().to_owned())
        }
    }
}

impl From<SomaError> for Status {
    fn from(error: SomaError) -> Self {
        let bytes = bcs::to_bytes(&error).unwrap();
        Status::with_details(tonic::Code::Internal, error.to_string(), bytes.into())
    }
}

impl From<SomaError> for ExecutionFailureStatus {
    fn from(error: SomaError) -> Self {
        ExecutionFailureStatus::SomaError(error)
    }
}

impl From<String> for SomaError {
    fn from(error: String) -> Self {
        SomaError::GenericAuthorityError { error }
    }
}

impl From<&str> for SomaError {
    fn from(error: &str) -> Self {
        SomaError::GenericAuthorityError { error: error.to_string() }
    }
}

impl From<ConsensusError> for SomaError {
    fn from(e: ConsensusError) -> Self {
        Self::Consensus(e.to_string())
    }
}

#[cfg(feature = "storage")]
impl From<TypedStoreError> for SomaError {
    fn from(e: TypedStoreError) -> Self {
        Self::Storage(e.to_string())
    }
}

#[cfg(feature = "storage")]
impl From<TypedStoreError> for crate::storage::storage_error::Error {
    fn from(error: TypedStoreError) -> Self {
        crate::storage::storage_error::Error::custom(error)
    }
}

type BoxError = Box<dyn std::error::Error + Send + Sync + 'static>;

pub type ExecutionResult<T = ()> = Result<T, ExecutionFailureStatus>;

pub type ExecutionErrorKind = ExecutionFailureStatus;

#[derive(Debug)]
pub struct ExecutionError {
    inner: Box<ExecutionErrorInner>,
}

#[derive(Debug)]
struct ExecutionErrorInner {
    /// The specific kind of execution error
    kind: ExecutionErrorKind,

    /// An optional source error providing additional context
    source: Option<BoxError>,
}

impl ExecutionError {
    pub fn new(kind: ExecutionErrorKind, source: Option<BoxError>) -> Self {
        Self { inner: Box::new(ExecutionErrorInner { kind, source }) }
    }

    pub fn new_with_source<E: Into<BoxError>>(kind: ExecutionErrorKind, source: E) -> Self {
        Self::new(kind, Some(source.into()))
    }

    pub fn from_kind(kind: ExecutionErrorKind) -> Self {
        Self::new(kind, None)
    }

    pub fn kind(&self) -> &ExecutionErrorKind {
        &self.inner.kind
    }

    pub fn source(&self) -> &Option<BoxError> {
        &self.inner.source
    }

    pub fn to_execution_status(&self) -> ExecutionFailureStatus {
        self.kind().clone()
    }
}

impl std::fmt::Display for ExecutionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ExecutionError: {:?}", self)
    }
}

impl std::error::Error for ExecutionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.inner.source.as_ref().map(|e| &**e as _)
    }
}

impl From<ExecutionErrorKind> for ExecutionError {
    fn from(kind: ExecutionErrorKind) -> Self {
        Self::from_kind(kind)
    }
}

impl From<crate::storage::storage_error::Error> for SomaError {
    fn from(e: crate::storage::storage_error::Error) -> Self {
        Self::Storage(e.to_string())
    }
}

/// Errors that can occur when processing blocks, reading from storage, or encountering shutdown.
#[derive(Clone, Debug, Error, IntoStaticStr)]
pub enum ConsensusError {
    #[error("Error deserializing block: {0}")]
    MalformedBlock(bcs::Error),

    #[error("Error deserializing commit: {0}")]
    MalformedCommit(bcs::Error),

    #[error("Error serializing: {0}")]
    SerializationFailure(bcs::Error),

    #[error("Block contains a transaction that is too large: {size} > {limit}")]
    TransactionTooLarge { size: usize, limit: usize },

    #[error("Block contains too many transactions: {count} > {limit}")]
    TooManyTransactions { count: usize, limit: usize },

    #[error("Block contains too many transaction bytes: {size} > {limit}")]
    TooManyTransactionBytes { size: usize, limit: usize },

    #[error("Unexpected block authority {0} from peer {1}")]
    UnexpectedAuthority(AuthorityIndex, AuthorityIndex),

    #[error("Block has wrong epoch: expected {expected}, actual {actual}")]
    WrongEpoch { expected: Epoch, actual: Epoch },

    #[error("Genesis blocks should only be generated from Committee!")]
    UnexpectedGenesisBlock,

    #[error("Genesis blocks should not be queried!")]
    UnexpectedGenesisBlockRequested,

    #[error(
        "Expected {requested} but received {received} blocks returned from authority {authority}"
    )]
    UnexpectedNumberOfBlocksFetched { authority: AuthorityIndex, requested: usize, received: usize },

    #[error("Unexpected block returned while fetching missing blocks")]
    UnexpectedFetchedBlock { index: AuthorityIndex, block_ref: BlockRef },

    #[error(
        "Unexpected block {block_ref} returned while fetching last own block from peer {index}"
    )]
    UnexpectedLastOwnBlock { index: AuthorityIndex, block_ref: BlockRef },

    #[error(
        "Too many blocks have been returned from authority {0} when requesting to fetch missing blocks"
    )]
    TooManyFetchedBlocksReturned(AuthorityIndex),

    #[error("Too many authorities have been provided from authority {0}")]
    TooManyAuthoritiesProvided(AuthorityIndex),

    #[error(
        "Provided size of highest accepted rounds parameter, {0}, is different than committee size, {1}"
    )]
    InvalidSizeOfHighestAcceptedRounds(usize, usize),

    #[error("Invalid authority index: {index} > {max}")]
    InvalidAuthorityIndex { index: AuthorityIndex, max: usize },

    #[error("Missing authority: {0}")]
    MissingAuthority(AuthorityIndex),

    #[error("Failed to deserialize signature: {0}")]
    MalformedSignature(FastCryptoError),

    #[error("Failed to verify the block's signature: {0}")]
    SignatureVerificationFailure(FastCryptoError),

    #[error("Synchronizer for fetching blocks directly from {0} is saturated")]
    SynchronizerSaturated(AuthorityIndex),

    #[error("Block {block_ref:?} rejected: {reason}")]
    BlockRejected { block_ref: BlockRef, reason: String },

    #[error(
        "Ancestor is in wrong position: block {block_authority}, ancestor {ancestor_authority}, position {position}"
    )]
    InvalidAncestorPosition {
        block_authority: AuthorityIndex,
        ancestor_authority: AuthorityIndex,
        position: usize,
    },

    #[error("Ancestor's round ({ancestor}) should be lower than the block's round ({block})")]
    InvalidAncestorRound { ancestor: Round, block: Round },

    #[error("Ancestor {0} not found among genesis blocks!")]
    InvalidGenesisAncestor(BlockRef),

    #[error("Too many ancestors in the block: {0} > {1}")]
    TooManyAncestors(usize, usize),

    #[error("Ancestors from the same authority {0}")]
    DuplicatedAncestorsAuthority(AuthorityIndex),

    #[error("Insufficient stake from parents: {parent_stakes} < {quorum}")]
    InsufficientParentStakes { parent_stakes: Stake, quorum: Stake },

    #[error("Invalid transaction: {0}")]
    InvalidTransaction(String),

    #[error("Received no commit from peer {peer}")]
    NoCommitReceived { peer: AuthorityIndex },

    #[error(
        "Received unexpected start commit from peer {peer}: requested {start}, received {commit:?}"
    )]
    UnexpectedStartCommit { peer: AuthorityIndex, start: CommitIndex, commit: Box<Commit> },

    #[error(
        "Received unexpected commit sequence from peer {peer}: {prev_commit:?}, {curr_commit:?}"
    )]
    UnexpectedCommitSequence {
        peer: AuthorityIndex,
        prev_commit: Box<Commit>,
        curr_commit: Box<Commit>,
    },

    #[error("Not enough votes ({stake}) on end commit from peer {peer}: {commit:?}")]
    NotEnoughCommitVotes { stake: Stake, peer: AuthorityIndex, commit: Box<Commit> },

    #[error("Received unexpected block from peer {peer}: {requested:?} vs {received:?}")]
    UnexpectedBlockForCommit { peer: AuthorityIndex, requested: BlockRef, received: BlockRef },

    #[error(
        "Unexpected certified commit index and last committed index. Expected next commit index to be {expected_commit_index}, but found {commit_index}"
    )]
    UnexpectedCertifiedCommitIndex { expected_commit_index: CommitIndex, commit_index: CommitIndex },

    #[cfg(feature = "storage")]
    #[error("RocksDB failure: {0}")]
    RocksDBFailure(#[from] TypedStoreError),

    #[error("Unknown network peer: {0}")]
    UnknownNetworkPeer(String),

    #[error("Peer {0} is disconnected.")]
    PeerDisconnected(String),

    #[error("Network config error: {0:?}")]
    NetworkConfig(String),

    #[error("Failed to connect as client: {0:?}")]
    NetworkClientConnection(String),

    #[error("Failed to send request: {0:?}")]
    NetworkRequest(String),

    #[error("Request timeout: {0:?}")]
    NetworkRequestTimeout(String),

    #[error("Consensus has shut down!")]
    Shutdown,
}

impl ConsensusError {
    /// Returns the error name - only the enun name without any parameters - as a static string.
    pub fn name(&self) -> &'static str {
        self.into()
    }
}

pub type ConsensusResult<T> = Result<T, ConsensusError>;

#[macro_export]
macro_rules! bail {
    ($e:expr) => {
        return Err($e);
    };
}

#[macro_export(local_inner_macros)]
macro_rules! ensure {
    ($cond:expr, $e:expr) => {
        if !($cond) {
            bail!($e);
        }
    };
}

#[derive(Debug, Error, IntoStaticStr)]
pub enum BlobError {
    #[error("fast crypto error: {0}")]
    FastCrypto(String),
    #[error("reqwest error: {0}")]
    ReqwestError(String),
    #[error("Network config error: {0:?}")]
    NetworkConfig(String),
    #[error("Parse error: {0}")]
    ParseError(String),
    #[error("Failed to send request: {0:?}")]
    NetworkRequest(String),
    #[error("write error: {0}")]
    WriteError(String),
    #[error("read error: {0}")]
    ReadError(String),
    #[error("ObjectStorage: {0}")]
    ObjectStorage(String),
    #[error("Not found: {0}")]
    NotFound(String),
    #[error("checksum mismatch: expected {expected}, computed {actual}")]
    ChecksumMismatch { expected: String, actual: String },
    #[error("size mismatch: expected {expected} bytes, got {actual} bytes")]
    SizeMismatch { expected: u64, actual: u64 },
    #[error("invalid chunk size: {size} bytes (must be between {min} and {max})")]
    InvalidChunkSize { size: u64, min: u64, max: u64 },
    #[error("url error: {0}")]
    UrlError(String),
    #[error("Io error: {0}, {1}")]
    Io(String, String),
    #[error("missing header: {0}")]
    MissingHeader(String),
    #[error("file too large")]
    FileTooLarge,
    #[error("HTTP {status} from {url}")]
    HttpStatus { status: u16, url: String },
    #[cfg(feature = "cloud-storage")]
    #[error("object store error: {0}")]
    ObjectStoreError(object_store::Error),
    #[error("Storage failed: {0}")]
    StorageFailure(String),
    #[error("Timeout hit")]
    Timeout,
}
pub type BlobResult<T> = Result<T, BlobError>;

pub type RuntimeResult<T> = Result<T, RuntimeError>;

#[derive(Debug, Error, IntoStaticStr)]
pub enum RuntimeError {
    #[error("Network config error: {0:?}")]
    NetworkConfig(String),
    #[error("Failed to connect as client: {0:?}")]
    NetworkClientConnection(String),
    #[error("Failed to send request: {0:?}")]
    NetworkRequest(String),
    #[error("Failed to parse URL: {0}")]
    UrlParseError(String),
    #[error("Network request error: {0:?}")]
    NetworkRequestError(String),
    #[error("Deserialize error: {0:?}")]
    DeserializeError(String),
    #[error("Validation error: {0:?}")]
    ValidationError(String),
    #[error("Error deserializing type: {0}")]
    MalformedType(bcs::Error),
    #[error("Error serializing: {0}")]
    SerializationFailure(bcs::Error),
    #[error("core processor error: {0:?}")]
    CoreProcessorError(String),
    #[error("storage failure: {0:?}")]
    StorageFailure(String),
    #[cfg(feature = "cloud-storage")]
    #[error("Object store error: {0}")]
    ObjectStoreError(object_store::Error),
    #[cfg(feature = "tls")]
    #[error("Reqwest error: {0}")]
    ReqwestError(reqwest::Error),
    #[error("Blob error: {0}")]
    BlobError(BlobError),
    #[error("Model error: {0:?}")]
    ModelError(String),
    #[error("Data not available: {0}")]
    DataNotAvailable(String),
    #[error("Data hash mismatch")]
    DataHashMismatch,
    #[error("Model not available: {0:?}")]
    ModelNotAvailable(String),
}

#[derive(Debug, Error, IntoStaticStr)]
pub enum ModelError {
    #[error("Network config error: {0:?}")]
    NetworkConfig(String),
    #[error("Failed to connect as client: {0:?}")]
    NetworkClientConnection(String),
    #[error("Failed to send request: {0:?}")]
    NetworkRequest(String),
    #[error("Error deserializing type: {0}")]
    MalformedType(bcs::Error),
    #[error("Error serializing: {0}")]
    SerializationFailure(bcs::Error),
    #[error("Storage failed: {0}")]
    StorageFailure(String),
    #[error("Safetensors failed: {0}")]
    SafeTensorsFailure(String),
    #[error("Failed type verification: {0}")]
    FailedTypeVerification(String),
    #[cfg(feature = "ml")]
    #[error("SafeTensor store error: {0}")]
    SafeTensorStoreError(SafetensorsStoreError),
    #[error("Apply error")]
    ApplyError,
    #[error("Empty data: {0}")]
    EmptyData(String),
}

pub type ModelResult<T> = Result<T, ModelError>;

/// Types of SomaError.
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq, IntoStaticStr)]
pub enum ErrorCategory {
    // A generic error that is retriable with new transaction resubmissions.
    Aborted,
    // Any validator or full node can check if a transaction is valid.
    InvalidTransaction,
    // Lock conflict on the transaction input.
    LockConflict,
    // Unexpected client error, for example generating invalid request or entering into invalid state.
    // And unexpected error from the remote peer. The validator may be malicious or there is a software bug.
    Internal,
    // Validator is overloaded.
    // ValidatorOverloaded,
    // Target validator is down or there are network issues.
    Unavailable,
}

impl ErrorCategory {
    // Whether the failure is retriable with new transaction submission.
    pub fn is_submission_retriable(&self) -> bool {
        matches!(
            self,
            ErrorCategory::Aborted
                // | ErrorCategory::ValidatorOverloaded
                | ErrorCategory::Unavailable
        )
    }
}
