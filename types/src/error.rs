//! # Error Types and Handling
//!
//! ## Overview
//! This module defines the error types, handling mechanisms, and result types
//! for the Soma blockchain. It provides a comprehensive error system that enables
//! detailed error reporting and proper error propagation throughout the codebase.
//!
//! ## Responsibilities
//! - Define the primary error types used throughout the system
//! - Provide conversion functions between different error representations
//! - Support error reporting for both internal and network-facing errors
//! - Enable proper error propagation and handling patterns
//!
//! ## Component Relationships
//! - Used by all modules to standardize error reporting and handling
//! - Interacts with RPC system to expose errors to clients
//! - Provides error types specific to various subsystems (consensus, authority, etc.)
//! - Defines execution errors for transaction processing
//!
//! ## Key Workflows
//! 1. Error creation and propagation through the Result type system
//! 2. Conversion between internal errors and network-facing error status codes
//! 3. Specialized error handling for critical operations
//! 4. Error categorization for different subsystems
//!
//! ## Design Patterns
//! - Error type hierarchies for domain-specific errors
//! - Comprehensive error variants with detailed context information
//! - Conversion traits between different error representations
//! - Macros for concise error handling patterns

use std::collections::BTreeMap;

use fastcrypto::error;
use fastcrypto::{error::FastCryptoError, hash::Digest};
use serde::{Deserialize, Serialize};
use store::TypedStoreError;
use strum::IntoStaticStr;
use thiserror::Error;
use tonic::Status;

use crate::committee::{AuthorityIndex, Epoch, Stake};
use crate::consensus::{
    block::{BlockRef, Round},
    commit::{Commit, CommitIndex},
};

use crate::crypto::NetworkPublicKey;
use crate::transaction::TransactionKind;
use crate::{
    base::AuthorityName,
    committee::{Committee, EpochId, VotingPower},
    digests::{ObjectDigest, TransactionDigest, TransactionEffectsDigest},
    effects::ExecutionFailureStatus,
    object::{ObjectID, ObjectRef, Version},
    peer_id::PeerId,
};

/// Standard Result type for Soma operations.
///
/// This type alias provides a consistent Result type used throughout the codebase,
/// with SomaError as the error type. The generic parameter T allows specifying
/// the success type, defaulting to () for operations that don't return a value.
///
/// # Examples
///
/// ```
/// # use crate::types::error::{SomaResult, SomaError};
/// fn operation_that_may_fail() -> SomaResult<u64> {
///     // If successful, return a value
///     Ok(42)
///     
///     // If failure, return an error
///     // Err(SomaError::Storage("Failed to read value".into()))
/// }
///
/// fn operation_with_no_return_value() -> SomaResult {
///     // For operations that just need to indicate success/failure
///     Ok(())
/// }
/// ```
pub type SomaResult<T = ()> = Result<T, SomaError>;

/// Primary error type for the Soma blockchain.
///
/// This enum represents all possible errors that can occur within the system.
/// It provides detailed contextual information about each error to facilitate
/// debugging and proper error handling.
///
/// The errors are grouped into related categories:
/// - Committee and validator related errors
/// - Cryptographic and signature errors
/// - Epoch management errors
/// - Execution and transaction processing errors
/// - Consensus and authority errors
/// - Object and storage errors
/// - Network and RPC errors
///
/// ## Thread Safety
/// This type is immutable and can be safely shared across threads.
///
/// ## Examples
///
/// ```
/// # use crate::types::error::SomaError;
/// # use crate::types::base::AuthorityName;
/// # fn example() {
/// // Creating a simple error
/// let error = SomaError::InvalidAddress;
///
/// // Creating an error with context
/// let error_with_context = SomaError::InvalidCommittee(
///     "Committee members do not have sufficient voting power".to_string()
/// );
///
/// // Creating an error with structured data
/// let validator_name = AuthorityName::default();
/// let authority_error = SomaError::FailedToVerifyTxCertWithExecutedEffects {
///     validator_name,
///     error: "Signature verification failed".to_string(),
/// };
/// # }
/// ```
#[derive(Eq, PartialEq, Clone, Debug, Serialize, Deserialize, Error, Hash)]
pub enum SomaError {
    /// Error when the committee configuration is invalid
    #[error("Invalid committee composition")]
    InvalidCommittee(String),

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
    FailedToVerifyTxCertWithExecutedEffects {
        validator_name: AuthorityName,
        error: String,
    },
    /// Error when too many authorities report errors for an operation
    #[error("Too many authority errors were detected for {}: {:?}", action, errors)]
    TooManyIncorrectAuthorities {
        errors: Vec<(AuthorityName, SomaError)>,
        action: String,
    },

    /// Error when a signature or certificate is from the wrong epoch
    #[error(
        "Signature or certificate from wrong epoch, expected {expected_epoch}, got {actual_epoch}"
    )]
    WrongEpoch {
        expected_epoch: EpochId,
        actual_epoch: EpochId,
    },

    /// Error when the number of signatures doesn't match expected signers
    #[error("Expect {expected} signer signatures but got {actual}")]
    SignerSignatureNumberMismatch { expected: usize, actual: usize },

    /// Error when a required signature is missing
    #[error("Required Signature from {expected} is absent {:?}", actual)]
    SignerSignatureAbsent {
        expected: String,
        actual: Vec<String>,
    },

    /// Error when a signature is from an unknown authority
    #[error(
        "Value was not signed by a known authority. signer: {:?}, index: {:?}, committee: \
         {committee}",
        signer,
        index
    )]
    UnknownSigner {
        signer: Option<String>,
        index: Option<u32>,
        committee: Box<Committee>,
    },

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
    StakeAggregatorRepeatedSigner {
        signer: AuthorityName,
        conflicting_sig: bool,
    },

    /// Error when a validator is suspected of Byzantine behavior
    #[error("Validator {authority:?} is faulty in a Byzantine manner: {reason:?}")]
    ByzantineAuthoritySuspicion {
        authority: AuthorityName,
        reason: String,
    },

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
    #[error(
        "Could not find the referenced object {:?} at version {:?}",
        object_id,
        version
    )]
    ObjectNotFound {
        object_id: ObjectID,
        version: Option<Version>,
    },
    /// Error when an object is already locked by a different transaction
    #[error(
        "Object {obj_ref:?} already locked by a different transaction: {pending_transaction:?}"
    )]
    ObjectLockConflict {
        obj_ref: ObjectRef,
        pending_transaction: TransactionDigest,
    },

    /// Error when an object is not available for consumption due to version mismatch
    #[error(
        "Object {provided_obj_ref:?} is not available for consumption, its current version: \
         {current_version:?}"
    )]
    ObjectVersionUnavailableForConsumption {
        provided_obj_ref: ObjectRef,
        current_version: Version,
    },

    /// Error when an object's digest doesn't match the expected value
    #[error(
        "Invalid Object digest for object {object_id:?}. Expected digest : {expected_digest:?}"
    )]
    InvalidObjectDigest {
        object_id: ObjectID,
        expected_digest: ObjectDigest,
    },

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
}

impl From<Status> for SomaError {
    /// Converts a gRPC Status to a SomaError.
    ///
    /// This allows error information to be properly preserved when errors
    /// cross RPC boundaries.
    ///
    /// # Arguments
    /// * `status` - The gRPC status to convert
    ///
    /// # Returns
    /// A SomaError representing the same error condition
    fn from(status: Status) -> Self {
        // if status.message() == "Too many requests" {
        //     return Self::TooManyRequests;
        // }

        let result = bcs::from_bytes::<SomaError>(status.details());
        if let Ok(error) = result {
            error
        } else {
            Self::RpcError(
                status.message().to_owned(),
                status.code().description().to_owned(),
            )
        }
    }
}

impl From<SomaError> for Status {
    /// Converts a SomaError to a gRPC Status.
    ///
    /// This allows error information to be properly preserved when errors
    /// are sent across RPC boundaries.
    ///
    /// # Arguments
    /// * `error` - The SomaError to convert
    ///
    /// # Returns
    /// A gRPC Status representing the same error condition
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
    /// Converts a String to a GenericAuthorityError SomaError.
    ///
    /// This convenience method allows arbitrary string errors to be
    /// quickly converted to SomaErrors.
    ///
    /// # Arguments
    /// * `error` - The error message string
    ///
    /// # Returns
    /// A SomaError::GenericAuthorityError containing the error message
    fn from(error: String) -> Self {
        SomaError::GenericAuthorityError { error }
    }
}

impl From<&str> for SomaError {
    /// Converts a string slice to a GenericAuthorityError SomaError.
    ///
    /// This convenience method allows arbitrary string literal errors to be
    /// quickly converted to SomaErrors.
    ///
    /// # Arguments
    /// * `error` - The error message string slice
    ///
    /// # Returns
    /// A SomaError::GenericAuthorityError containing the error message
    fn from(error: &str) -> Self {
        SomaError::GenericAuthorityError {
            error: error.to_string(),
        }
    }
}

impl From<ConsensusError> for SomaError {
    /// Converts a ConsensusError to a SomaError.
    ///
    /// This enables error propagation from the consensus subsystem
    /// to the broader error handling system.
    ///
    /// # Arguments
    /// * `e` - The ConsensusError to convert
    ///
    /// # Returns
    /// A SomaError::Consensus containing the error message
    fn from(e: ConsensusError) -> Self {
        Self::Consensus(e.to_string())
    }
}

impl From<TypedStoreError> for SomaError {
    fn from(e: TypedStoreError) -> Self {
        Self::Storage(e.to_string())
    }
}

impl From<TypedStoreError> for ShardError {
    fn from(e: TypedStoreError) -> Self {
        Self::DatastoreError(e.to_string())
    }
}

impl From<TypedStoreError> for crate::storage::storage_error::Error {
    fn from(error: TypedStoreError) -> Self {
        crate::storage::storage_error::Error::custom(error)
    }
}

/// Type alias for a boxed Error trait object with Send + Sync requirements.
///
/// This provides a standard error type that can be used for dynamic errors
/// that need to be transported across thread boundaries.
type BoxError = Box<dyn std::error::Error + Send + Sync + 'static>;

pub type ExecutionResult<T = ()> = Result<T, ExecutionFailureStatus>;

/// Type alias for the specific error kind used in transaction execution.
///
/// This delegates to the ExecutionFailureStatus enum for detailed error information
/// about transaction execution failures.
pub type ExecutionErrorKind = ExecutionFailureStatus;

/// Error type for transaction execution failures.
///
/// This structure provides detailed information about errors that occur during
/// transaction execution, including both the error kind and an optional source
/// error for additional context.
///
/// ## Thread Safety
/// This type is safe to send across thread boundaries when wrapped in an Arc.
#[derive(Debug)]
pub struct ExecutionError {
    inner: Box<ExecutionErrorInner>,
}

/// Inner structure containing execution error details.
///
/// This is boxed in ExecutionError to minimize the size of the ExecutionError
/// when passed around.
#[derive(Debug)]
struct ExecutionErrorInner {
    /// The specific kind of execution error
    kind: ExecutionErrorKind,

    /// An optional source error providing additional context
    source: Option<BoxError>,
}

impl ExecutionError {
    /// Creates a new ExecutionError with the given kind and source.
    ///
    /// # Arguments
    /// * `kind` - The specific kind of execution error
    /// * `source` - An optional source error providing additional context
    ///
    /// # Returns
    /// A new ExecutionError instance
    pub fn new(kind: ExecutionErrorKind, source: Option<BoxError>) -> Self {
        Self {
            inner: Box::new(ExecutionErrorInner { kind, source }),
        }
    }

    /// Creates a new ExecutionError with the given kind and source error.
    ///
    /// This is a convenience method for creating an ExecutionError with a source
    /// error that can be converted into a BoxError.
    ///
    /// # Arguments
    /// * `kind` - The specific kind of execution error
    /// * `source` - A source error that can be converted into a BoxError
    ///
    /// # Returns
    /// A new ExecutionError instance
    pub fn new_with_source<E: Into<BoxError>>(kind: ExecutionErrorKind, source: E) -> Self {
        Self::new(kind, Some(source.into()))
    }

    /// Creates a new ExecutionError with just an error kind.
    ///
    /// This is a convenience method for creating an ExecutionError without a
    /// source error.
    ///
    /// # Arguments
    /// * `kind` - The specific kind of execution error
    ///
    /// # Returns
    /// A new ExecutionError instance with no source error
    pub fn from_kind(kind: ExecutionErrorKind) -> Self {
        Self::new(kind, None)
    }

    /// Returns a reference to the error kind.
    ///
    /// # Returns
    /// A reference to the ExecutionErrorKind contained in this error
    pub fn kind(&self) -> &ExecutionErrorKind {
        &self.inner.kind
    }

    /// Returns a reference to the optional source error.
    ///
    /// # Returns
    /// A reference to the optional BoxError source error
    pub fn source(&self) -> &Option<BoxError> {
        &self.inner.source
    }

    /// Converts this error to an ExecutionFailureStatus.
    ///
    /// This is useful when only the error kind is needed without
    /// the source information.
    ///
    /// # Returns
    /// An ExecutionFailureStatus representing the error kind
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
    /// Creates an ExecutionError from an ExecutionErrorKind.
    ///
    /// This allows for convenient conversion from error kinds to full errors.
    ///
    /// # Arguments
    /// * `kind` - The error kind to convert
    ///
    /// # Returns
    /// A new ExecutionError with the given kind and no source error
    fn from(kind: ExecutionErrorKind) -> Self {
        Self::from_kind(kind)
    }
}

impl From<crate::storage::storage_error::Error> for SomaError {
    /// Converts a storage error to a SomaError.
    ///
    /// This enables error propagation from the storage subsystem
    /// to the broader error handling system.
    ///
    /// # Arguments
    /// * `e` - The storage error to convert
    ///
    /// # Returns
    /// A SomaError::Storage containing the error message
    fn from(e: crate::storage::storage_error::Error) -> Self {
        Self::Storage(e.to_string())
    }
}

/// Error type for consensus-related operations.
///
/// This enum represents all possible errors that can occur within the consensus
/// subsystem. It provides detailed context for debugging and diagnosing issues
/// related to the Byzantine Fault Tolerant consensus mechanism.
///
/// ## Thread Safety
/// This type is immutable and can be safely shared across threads.
#[derive(Error, Debug)]
pub enum ConsensusError {
    /// Error when attempting to query genesis blocks directly
    #[error("Genesis blocks should not be queried!")]
    UnexpectedGenesisBlockRequested,

    /// Error when too many authorities are provided from a single authority
    #[error("Too many authorities have been provided from authority {0}")]
    TooManyAuthoritiesProvided(AuthorityIndex),

    /// Error when requesting too many blocks from an authority
    #[error("Too many blocks have been requested from authority {0}")]
    TooManyFetchBlocksRequested(AuthorityIndex),

    /// Error when receiving a block from an unexpected authority
    #[error("Unexpected block authority {0} from peer {1}")]
    UnexpectedAuthority(AuthorityIndex, AuthorityIndex),

    /// Error when highest accepted rounds parameter size doesn't match committee size
    #[error(
        "Provided size of highest accepted rounds parameter, {0}, is different than committee \
         size, {1}"
    )]
    InvalidSizeOfHighestAcceptedRounds(usize, usize),

    /// Error when a block is rejected
    #[error("Block {block_ref:?} rejected: {reason}")]
    BlockRejected { block_ref: BlockRef, reason: String },

    /// Error when a block cannot be deserialized
    #[error("Error deserializing block: {0}")]
    MalformedBlock(bcs::Error),

    /// Error when a commit cannot be deserialized
    #[error("Error deserializing commit: {0}")]
    MalformedCommit(bcs::Error),

    /// Error when no commit is received from a peer
    #[error("Received no commit from peer {peer}")]
    NoCommitReceived { peer: String },

    /// Error when unexpected start commit is received
    #[error(
        "Received unexpected start commit from peer {peer}: requested {start}, received {commit:?}"
    )]
    UnexpectedStartCommit {
        peer: String,
        start: CommitIndex,
        commit: Box<Commit>,
    },

    /// Error when commit sequence is unexpected
    #[error(
        "Received unexpected commit sequence from peer {peer}: {prev_commit:?}, {curr_commit:?}"
    )]
    UnexpectedCommitSequence {
        peer: String,
        prev_commit: Box<Commit>,
        curr_commit: Box<Commit>,
    },

    /// Error when unexpected number of blocks is returned
    #[error("Expected {requested} but received {received} blocks returned from peer {peer}")]
    UnexpectedNumberOfBlocksFetched {
        peer: String,
        requested: usize,
        received: usize,
    },

    /// Error when unexpected block is received for a commit
    #[error("Received unexpected block from peer {peer}: {requested:?} vs {received:?}")]
    UnexpectedBlockForCommit {
        peer: String,
        requested: BlockRef,
        received: BlockRef,
    },

    /// Error when no blocks are received for a commit
    #[error("Received no blocks from peer's commit {peer}: {commit:?}")]
    NoBlocksForCommit { peer: String, commit: Box<Commit> },

    /// Error when no authority is available to fetch commits
    #[error("No available authority to fetch commits")]
    NoAvailableAuthorityToFetchCommits,

    /// Error when not enough votes exist for a commit
    #[error("Not enough votes ({stake}) on end commit from peer {peer}: {commit:?}")]
    NotEnoughCommitVotes {
        stake: Stake,
        peer: String,
        commit: Box<Commit>,
    },

    /// Error when an ancestor is in the wrong position
    #[error(
        "Ancestor is in wrong position: block {block_authority}, ancestor {ancestor_authority}, \
         position {position}"
    )]
    InvalidAncestorPosition {
        block_authority: AuthorityIndex,
        ancestor_authority: AuthorityIndex,
        position: usize,
    },

    /// Error when an ancestor's round is not lower than block's round
    #[error("Ancestor's round ({ancestor}) should be lower than the block's round ({block})")]
    InvalidAncestorRound { ancestor: Round, block: Round },

    /// Error when an ancestor is not found in genesis blocks
    #[error("Ancestor {0} not found among genesis blocks!")]
    InvalidGenesisAncestor(BlockRef),

    /// Error when a block has too many ancestors
    #[error("Too many ancestors in the block: {0} > {1}")]
    TooManyAncestors(usize, usize),

    /// Error when ancestors are from the same authority
    #[error("Ancestors from the same authority {0}")]
    DuplicatedAncestorsAuthority(AuthorityIndex),

    /// Error when a block has the wrong epoch
    #[error("Block has wrong epoch: expected {expected}, actual {actual}")]
    WrongEpoch { expected: Epoch, actual: Epoch },

    /// Error when parent stakes are insufficient
    #[error("Insufficient stake from parents: {parent_stakes} < {quorum}")]
    InsufficientParentStakes { parent_stakes: Stake, quorum: Stake },

    /// Error when genesis blocks are generated incorrectly
    #[error("Genesis blocks should only be generated from Committee!")]
    UnexpectedGenesisBlock,

    /// Error when a transaction is invalid
    #[error("Invalid transaction: {0}")]
    InvalidTransaction(String),

    /// Error when block timestamp is invalid
    #[error("Ancestors max timestamp {max_timestamp_ms} > block timestamp {block_timestamp_ms}")]
    InvalidBlockTimestamp {
        max_timestamp_ms: u64,
        block_timestamp_ms: u64,
    },

    /// Error when serialization fails
    #[error("Error serializing: {0}")]
    SerializationFailure(bcs::Error),

    /// Error when authority index is invalid
    #[error("Invalid authority index: {index} > {max}")]
    InvalidAuthorityIndex { index: AuthorityIndex, max: usize },

    /// Error when signature deserialization fails
    #[error("Failed to deserialize signature: {0}")]
    MalformedSignature(FastCryptoError),

    /// Error when block signature verification fails
    #[error("Failed to verify the block's signature: {0}")]
    SignatureVerificationFailure(FastCryptoError),

    /// Error when synchronizer is saturated
    #[error("Synchronizer for fetching blocks directly from {0} is saturated")]
    SynchronizerSaturated(AuthorityIndex),

    /// Error when too many blocks are returned when fetching missing blocks
    #[error(
        "Too many blocks have been returned from authority {0} when requesting to fetch missing \
         blocks"
    )]
    TooManyFetchedBlocksReturned(AuthorityIndex),

    /// Error when unexpected block is returned when fetching missing blocks
    #[error("Unexpected block returned while fetching missing blocks")]
    UnexpectedFetchedBlock {
        index: AuthorityIndex,
        block_ref: BlockRef,
    },

    /// Error when unexpected block is returned when fetching own block
    #[error(
        "Unexpected block {block_ref} returned while fetching last own block from peer {index}"
    )]
    UnexpectedLastOwnBlock {
        index: AuthorityIndex,
        block_ref: BlockRef,
    },

    /// Error in network configuration
    #[error("Network config error: {0:?}")]
    NetworkConfig(String),

    /// Error when connecting as client
    #[error("Failed to connect as client: {0:?}")]
    NetworkClientConnection(String),

    /// Error when connecting as server
    #[error("Failed to connect as server: {0:?}")]
    NetworkServerConnection(String),

    /// Error when sending request
    #[error("Failed to send request: {0:?}")]
    NetworkRequest(String),

    /// Error when request times out
    #[error("Request timeout: {0:?}")]
    NetworkRequestTimeout(String),

    /// Error when consensus has shut down
    #[error("Consensus has shut down!")]
    Shutdown,

    /// Error when state hash is invalid
    #[error("Invalid state hash: {expected:?} != {actual:?}")]
    InvalidStateHash {
        expected: Digest<32>,
        actual: Digest<32>,
    },

    /// Error in storage operations
    #[error("Storage error: {0}")]
    Storage(String),

    /// Error when no committee exists for an epoch
    #[error("No committee for epoch: {0}")]
    NoCommitteeForEpoch(Epoch),

    /// Error when end of epoch data in block is invalid
    #[error("Invalid end of epoch data in block: {0}")]
    InvalidEndOfEpoch(String),

    #[error("RocksDB failure: {0}")]
    RocksDBFailure(#[from] TypedStoreError),
}

/// Standard Result type for consensus operations.
///
/// This type alias provides a consistent Result type for consensus operations,
/// with ConsensusError as the error type.
pub type ConsensusResult<T> = Result<T, ConsensusError>;

impl From<crate::storage::storage_error::Error> for ConsensusError {
    /// Converts a storage error to a ConsensusError.
    ///
    /// This enables error propagation from the storage subsystem
    /// to the consensus error handling system.
    ///
    /// # Arguments
    /// * `e` - The storage error to convert
    ///
    /// # Returns
    /// A ConsensusError::Storage containing the error message
    fn from(e: crate::storage::storage_error::Error) -> Self {
        Self::Storage(e.to_string())
    }
}

/// Macro to return an error early.
///
/// This is a convenience macro similar to the `bail!` macro in anyhow,
/// allowing for early returns with errors.
///
/// # Examples
///
/// ```
/// # use crate::types::error::SomaError;
/// # fn example() -> Result<(), SomaError> {
/// #     let condition = false;
/// if condition {
///     bail!(SomaError::TimeoutError);
/// }
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! bail {
    ($e:expr) => {
        return Err($e);
    };
}

/// Macro to ensure a condition is true, returning an error if not.
///
/// This is a convenience macro similar to the `ensure!` macro in anyhow,
/// allowing for condition checking with early returns on failure.
///
/// # Examples
///
/// ```
/// # use crate::types::error::SomaError;
/// # fn example(value: u64) -> Result<(), SomaError> {
/// ensure!(value > 0, SomaError::InvalidTransactionDigest);
/// # Ok(())
/// # }
/// ```
#[macro_export(local_inner_macros)]
macro_rules! ensure {
    ($cond:expr, $e:expr) => {
        if !($cond) {
            bail!($e);
        }
    };
}

/// Errors that can occur when processing blocks, reading from storage, or encountering shutdown.
#[derive(Clone, Debug, Error, IntoStaticStr)]
pub enum SharedError {
    #[error("fast crypto error: {0}")]
    FastCrypto(String),
    #[error("Quorum failed")]
    QuorumFailed,

    #[error("wrong epoch")]
    WrongEpoch,
    #[error("Validation error: {0}")]
    ValidationError(String),
    #[error("Shard Not Found: {0}")]
    ShardNotFound(String),

    #[error("Actor error: {0}")]
    ActorError(String),

    #[error("Thread error: {0}")]
    ThreadError(String),

    #[error("Compression failed: {0}")]
    CompressionFailed(String),

    #[error("ObjectStorage: {0}")]
    ObjectStorage(String),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Error deserializing block: {0}")]
    MalformedBlock(bcs::Error),

    #[error("Error deserializing commit: {0}")]
    MalformedCommit(bcs::Error),

    #[error("Error serializing: {0}")]
    SerializationFailure(bcs::Error),

    #[error("Error deserializing type: {0}")]
    MalformedType(bcs::Error),

    #[error("Block contains a transaction that is too large: {size} > {limit}")]
    TransactionTooLarge { size: usize, limit: usize },

    #[error("Block contains too many transactions: {count} > {limit}")]
    TooManyTransactions { count: usize, limit: usize },

    #[error("Block contains too many transaction bytes: {size} > {limit}")]
    TooManyTransactionBytes { size: usize, limit: usize },

    // #[error("Unexpected block authority {0} from peer {1}")]
    // UnexpectedAuthority(AuthorityIndex, AuthorityIndex),

    // #[error("Block has wrong epoch: expected {expected}, actual {actual}")]
    // WrongEpoch { expected: Epoch, actual: Epoch },
    #[error("Genesis blocks should only be generated from Committee!")]
    UnexpectedGenesisBlock,

    #[error("Genesis blocks should not be queried!")]
    UnexpectedGenesisBlockRequested,

    #[error("Contacting peer is unauthorized")]
    UnauthorizedPeer,

    #[error("Failed loading python code: {0}")]
    FailedLoadingPythonModule(String),
    #[error("Provided path failed: {0}")]
    PathError(String),
    #[error("Calling module failed: {0}")]
    FailedCallingPythonModule(String),
    #[error("Input batch size does not match output batch size: {0}")]
    BatchSizeMismatch(String),
    #[error("Could not get the shape of the array")]
    ArrayShapeError,
    #[error("Could not spawn blocking threadi: {0}")]
    SpawnBlockingError(String),
    // #[error(
    //     "Expected {requested} but received {received} blocks returned from authority {authority}"
    // )]
    // UnexpectedNumberOfBlocksFetched {
    //     authority: AuthorityIndex,
    //     requested: usize,
    //     received: usize,
    // },

    // #[error("Unexpected block returned while fetching missing blocks")]
    // UnexpectedFetchedBlock {
    //     index: AuthorityIndex,
    //     block_ref: BlockRef,
    // },

    // #[error(
    //     "Unexpected block {block_ref} returned while fetching last own block from peer {index}"
    // )]
    // UnexpectedLastOwnBlock {
    //     index: AuthorityIndex,
    //     block_ref: BlockRef,
    // },

    // #[error("Too many blocks have been returned from authority {0} when requesting to fetch missing blocks")]
    // TooManyFetchedBlocksReturned(AuthorityIndex),

    // #[error("Too many blocks have been requested from authority {0}")]
    // TooManyFetchBlocksRequested(AuthorityIndex),

    // #[error("Too many authorities have been provided from authority {0}")]
    // TooManyAuthoritiesProvided(AuthorityIndex),
    #[error(
        "Provided size of highest accepted rounds parameter, {0}, is different than committee \
         size, {1}"
    )]
    InvalidSizeOfHighestAcceptedRounds(usize, usize),

    #[error("Invalid authority index: {index} > {max}")]
    InvalidAuthorityIndex { index: AuthorityIndex, max: usize },

    #[error("Failed to deserialize signature: {0}")]
    MalformedSignature(FastCryptoError),
    #[error("VDF failed: {0}")]
    FailedVDF(String),
    #[error("failed type verification: {0}")]
    FailedTypeVerification(String),
    #[error("Failed to verify the block's signature: {0}")]
    SignatureVerificationFailure(FastCryptoError),

    #[error("Failed building reqwest client")]
    FailedBuildingHttpClient,

    #[error("URL failed: {0}")]
    UrlError(String),

    // #[error("Synchronizer for fetching blocks directly from {0} is saturated")]
    // SynchronizerSaturated(AuthorityIndex),

    // #[error("Block {block_ref:?} rejected: {reason}")]
    // BlockRejected { block_ref: BlockRef, reason: String },

    // #[error("Ancestor is in wrong position: block {block_authority}, ancestor {ancestor_authority}, position {position}")]
    // InvalidAncestorPosition {
    //     block_authority: AuthorityIndex,
    //     ancestor_authority: AuthorityIndex,
    //     position: usize,
    // },

    // #[error("Ancestor's round ({ancestor}) should be lower than the block's round ({block})")]
    // InvalidAncestorRound { ancestor: Round, block: Round },

    // #[error("Ancestor {0} not found among genesis blocks!")]
    // InvalidGenesisAncestor(BlockRef),
    #[error("Too many ancestors in the block: {0} > {1}")]
    TooManyAncestors(usize, usize),

    // #[error("Ancestors from the same authority {0}")]
    // DuplicatedAncestorsAuthority(AuthorityIndex),

    // #[error("Insufficient stake from parents: {parent_stakes} < {quorum}")]
    // InsufficientParentStakes { parent_stakes: Stake, quorum: Stake },
    #[error("Invalid transaction: {0}")]
    InvalidTransaction(String),

    #[error("Ancestors max timestamp {max_timestamp_ms} > block timestamp {block_timestamp_ms}")]
    InvalidBlockTimestamp {
        max_timestamp_ms: u64,
        block_timestamp_ms: u64,
    },

    #[error("No available authority to fetch commits")]
    NoAvailableAuthorityToFetchCommits,

    #[error("This request conflicts with a previous request")]
    ConflictingRequest,

    #[error("Sending to core thread failed: {0}")]
    FailedToSendToCoreThread(String),
    // #[error("Received no commit from peer {peer}")]
    // NoCommitReceived { peer: AuthorityIndex },

    // #[error(
    //     "Received unexpected start commit from peer {peer}: requested {start}, received {commit:?}"
    // )]
    // UnexpectedStartCommit {
    //     peer: AuthorityIndex,
    //     start: CommitIndex,
    //     commit: Box<Commit>,
    // },

    // #[error(
    //     "Received unexpected commit sequence from peer {peer}: {prev_commit:?}, {curr_commit:?}"
    // )]
    // UnexpectedCommitSequence {
    //     peer: AuthorityIndex,
    //     prev_commit: Box<Commit>,
    //     curr_commit: Box<Commit>,
    // },

    // #[error("Not enough votes ({stake}) on end commit from peer {peer}: {commit:?}")]
    // NotEnoughCommitVotes {
    //     stake: Stake,
    //     peer: AuthorityIndex,
    //     commit: Box<Commit>,
    // },

    // #[error("Received unexpected block from peer {peer}: {requested:?} vs {received:?}")]
    // UnexpectedBlockForCommit {
    //     peer: AuthorityIndex,
    //     requested: BlockRef,
    //     received: BlockRef,
    // },
    #[error("Error with IO: {0}")]
    IOError(String),

    // #[error("RocksDB failure: {0}")]
    // RocksDBFailure(#[from] TypedStoreError),
    #[error("Datastore error: {0}")]
    DatastoreError(String),

    #[error("Unknown network peer: {0}")]
    UnknownNetworkPeer(String),

    #[error("Peer {0} is disconnected.")]
    PeerDisconnected(String),

    #[error("Network config error: {0:?}")]
    NetworkConfig(String),

    #[error("Failed to connect as client: {0:?}")]
    NetworkClientConnection(String),

    #[error("Failed to connect as server: {0:?}")]
    NetworkServerConnection(String),

    #[error("Failed to send request: {0:?}")]
    NetworkRequest(String),

    #[error("Request timeout: {0:?}")]
    NetworkRequestTimeout(String),

    #[error("Consensus has shut down!")]
    Shutdown,

    #[error("Invalid digest length")]
    InvalidDigestLength,

    #[error("Shard error: {0}")]
    Shard(String),
}

pub type SharedResult<T> = Result<T, SharedError>;

/// Errors that can occur when processing blocks, reading from storage, or encountering shutdown.
#[derive(Debug, Error, IntoStaticStr)]
pub enum ShardError {
    #[error("Recv duplicate error")]
    RecvDuplicate,
    #[error("Send duplicate error")]
    SendDuplicate,
    #[error("Evaluation error: {0}")]
    EvaluationError(EvaluationError),
    #[error("Model error: {0}")]
    InferenceError(InferenceError),
    #[error("Intelligence error: {0}")]
    IntelligenceError(IntelligenceError),
    #[error("Not a member of the shard")]
    InvalidShardMember,
    #[error("Cache error")]
    CacheError,
    #[error("Wrong epoch")]
    WrongEpoch,
    #[error("Quorum failed")]
    QuorumFailed,
    #[error("Missing compression metadata")]
    MissingCompressionMetadata,
    #[error("Missing data")]
    MissingData,
    #[error("Encryption failed")]
    EncryptionFailed,
    #[error("weighted sample error: {0}")]
    WeightedSampleError(String),

    #[error("Shard Not Found: {0}")]
    ShardNotFound(String),

    #[error("Object validation error: {0}")]
    ObjectValidation(String),
    #[error("Actor error: {0}")]
    ActorError(String),

    #[error("Thread error: {0}")]
    ThreadError(String),

    #[error("Conflict: {0}")]
    Conflict(String),
    #[error("invalid reveal: {0}")]
    InvalidReveal(String),
    #[error("Compression failed: {0}")]
    CompressionFailed(String),

    #[error("Shard irrecoverably failed: {0}")]
    ShardFailure(String),

    #[error("Invalid shard token: {0}")]
    InvalidShardToken(String),

    #[error("ObjectStorage: {0}")]
    ObjectStorage(String),

    #[error("Digest failure: {0}")]
    DigestFailure(SharedError),

    #[error("Object error: {0}")]
    ObjectError(ObjectError),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Error deserializing block: {0}")]
    MalformedBlock(bcs::Error),

    #[error("Error deserializing commit: {0}")]
    MalformedCommit(bcs::Error),

    #[error("Error serializing: {0}")]
    SerializationFailure(String),

    #[error("Error deserializing type: {0}")]
    MalformedType(bcs::Error),

    #[error("Block contains a transaction that is too large: {size} > {limit}")]
    TransactionTooLarge { size: usize, limit: usize },

    #[error("Block contains too many transactions: {count} > {limit}")]
    TooManyTransactions { count: usize, limit: usize },

    #[error("Block contains too many transaction bytes: {size} > {limit}")]
    TooManyTransactionBytes { size: usize, limit: usize },

    #[error("Genesis blocks should only be generated from Committee!")]
    UnexpectedGenesisBlock,

    #[error("Genesis blocks should not be queried!")]
    UnexpectedGenesisBlockRequested,

    #[error("Contacting peer is unauthorized")]
    UnauthorizedPeer,

    #[error("Failed loading python code: {0}")]
    FailedLoadingPythonModule(String),
    #[error("Provided path failed: {0}")]
    PathError(String),
    #[error("Calling module failed: {0}")]
    FailedCallingPythonModule(String),
    #[error("Input batch size does not match output batch size: {0}")]
    BatchSizeMismatch(String),
    #[error("Could not get the shape of the array")]
    ArrayShapeError,

    #[error(
        "Provided size of highest accepted rounds parameter, {0}, is different than committee \
         size, {1}"
    )]
    InvalidSizeOfHighestAcceptedRounds(usize, usize),

    #[error("Failed to deserialize signature: {0}")]
    MalformedSignature(FastCryptoError),

    #[error("Signature aggregation failure: {0}")]
    SignatureAggregationFailure(FastCryptoError),

    #[error("Failed to verify the block's signature: {0}")]
    SignatureVerificationFailure(FastCryptoError),

    #[error("Failed building reqwest client")]
    FailedBuildingHttpClient,

    #[error("Failed to parse URL: {0}")]
    UrlParseError(String),

    #[error("Too many ancestors in the block: {0} > {1}")]
    TooManyAncestors(usize, usize),

    #[error("Invalid transaction: {0}")]
    InvalidTransaction(String),

    #[error("Ancestors max timestamp {max_timestamp_ms} > block timestamp {block_timestamp_ms}")]
    InvalidBlockTimestamp {
        max_timestamp_ms: u64,
        block_timestamp_ms: u64,
    },

    #[error("No available authority to fetch commits")]
    NoAvailableAuthorityToFetchCommits,

    #[error("This request conflicts with a previous request")]
    ConflictingRequest,

    #[error("Sending to core thread failed: {0}")]
    FailedToSendToCoreThread(String),

    #[error("failed type verification: {0}")]
    FailedTypeVerification(String),

    #[error("Error with IO: {0}")]
    IOError(String),

    #[error("Datastore error: {0}")]
    DatastoreError(String),

    #[error("Unknown network peer: {0}")]
    UnknownNetworkPeer(String),

    #[error("Peer {0} is disconnected.")]
    PeerDisconnected(String),

    #[error("Network config error: {0:?}")]
    NetworkConfig(String),

    #[error("Failed to connect as client: {0:?}")]
    NetworkClientConnection(String),

    #[error("Failed to connect as server: {0:?}")]
    NetworkServerConnection(String),

    #[error("Failed to send request: {0:?}")]
    NetworkRequest(String),

    #[error("Request timeout: {0:?}")]
    NetworkRequestTimeout(String),

    #[error("Concurrency error: {0:?}")]
    ConcurrencyError(String),

    #[error("Consensus has shut down!")]
    Shutdown,

    #[error("Serialization error: {0:?}")]
    SerializationError(String),

    #[error("Wallet error: {0:?}")]
    WalletError(String),

    #[error("Transaction failed: {0:?}")]
    TransactionFailed(String),

    #[error("Other: {0:?}")]
    Other(String),
}

pub type ShardResult<T> = Result<T, ShardError>;

#[derive(Clone, Debug, Error, IntoStaticStr)]
pub enum ObjectError {
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
    #[error("Verification error: {0}")]
    VerificationError(String),
    #[error("tus error: {0}")]
    TusError(String),
    #[error("Io error: {0}, {1}")]
    Io(String, String),
    #[error("missing header: {0}")]
    MissingHeader(String),
    #[error("file too large")]
    FileTooLarge,
    #[error("unexpected status code: {0}")]
    UnexpectedStatusCode(u16),
}
pub type ObjectResult<T> = Result<T, ObjectError>;

pub type InferenceResult<T> = Result<T, InferenceError>;

#[derive(Clone, Debug, Error, IntoStaticStr)]
pub enum InferenceError {
    #[error("Network config error: {0:?}")]
    NetworkConfig(String),
    #[error("Failed to parse URL: {0}")]
    UrlParseError(String),
    #[error("Network request error: {0:?}")]
    NetworkRequestError(String),
    #[error("Deserialize error: {0:?}")]
    DeserializeError(String),
    #[error("Validation error: {0:?}")]
    ValidationError(String),
}

pub type EvaluationResult<T> = Result<T, EvaluationError>;

#[derive(Clone, Debug, Error, IntoStaticStr)]
pub enum EvaluationError {
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
}

impl From<ShardError> for SharedError {
    fn from(e: ShardError) -> Self {
        Self::Shard(e.to_string())
    }
}

pub type IntelligenceResult<T> = Result<T, IntelligenceError>;

#[derive(Debug, Error, IntoStaticStr)]
pub enum IntelligenceError {
    #[error("Error deserializing type: {0}")]
    MalformedType(bcs::Error),
    #[error("Error serializing: {0}")]
    SerializationFailure(bcs::Error),
    #[error("Reqwest error: {0}")]
    ReqwestError(reqwest::Error),
    #[error("Parse error: {0}")]
    ParseError(String),
}
