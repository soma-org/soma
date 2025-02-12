use std::collections::BTreeMap;

use fastcrypto::error;
use fastcrypto::{error::FastCryptoError, hash::Digest};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tonic::Status;

use crate::committee::{AuthorityIndex, Epoch, Stake};
use crate::consensus::{
    block::{BlockRef, Round},
    commit::{Commit, CommitIndex},
};

use crate::crypto::NetworkPublicKey;
use crate::{
    base::AuthorityName,
    committee::{Committee, EpochId, VotingPower},
    digests::{ObjectDigest, TransactionDigest, TransactionEffectsDigest},
    effects::ExecutionFailureStatus,
    object::{ObjectID, ObjectRef, Version},
    peer_id::PeerId,
};
pub type SomaResult<T = ()> = Result<T, SomaError>;

/// Custom error type for Sui.
#[derive(Eq, PartialEq, Clone, Debug, Serialize, Deserialize, Error, Hash)]
pub enum SomaError {
    #[error("Invalid committee composition")]
    InvalidCommittee(String),

    #[error("Key Conversion Error: {0}")]
    KeyConversionError(String),

    #[error("Signature is not valid: {}", error)]
    InvalidSignature { error: String },

    #[error("Invalid address")]
    InvalidAddress,

    #[error("Value was not signed by the correct sender: {}", error)]
    IncorrectSigner { error: String },

    #[error("Cannot add validator that is already active or pending")]
    DuplicateValidator,

    #[error("Cannot remove validator that is not active")]
    NotAValidator,

    #[error("Cannot remove validator that is already removed")]
    ValidatorAlreadyRemoved,

    #[error("Advanced to wrong epoch")]
    AdvancedToWrongEpoch,

    // These are errors that occur when an RPC fails and is simply the utf8 message sent in a
    // Tonic::Status
    #[error("{1} - {0}")]
    RpcError(String, String),

    // Epoch related errors.
    #[error("Validator temporarily stopped processing transactions due to epoch change")]
    ValidatorHaltedAtEpochEnd,
    #[error("Operations for epoch {0} have ended")]
    EpochEnded(EpochId),
    #[error("Error when advancing epoch: {:?}", error)]
    AdvanceEpochError { error: String },

    #[error("Operation timed out")]
    TimeoutError,

    #[error("Failed to execute transaction locally by Orchestrator: {error:?}")]
    TransactionOrchestratorLocalExecutionError { error: String },

    // Errors related to the authority-consensus interface.
    #[error("Failed to submit transaction to consensus: {0}")]
    FailedToSubmitToConsensus(String),

    #[error("Authority Error: {error:?}")]
    GenericAuthorityError { error: String },

    #[error("Missing committee information for epoch {0}")]
    MissingCommitteeAtEpoch(EpochId),

    #[error("Unable to communicate with the Quorum Driver channel: {:?}", error)]
    QuorumDriverCommunicationError { error: String },

    #[error(
        "Failed to verify Tx certificate with executed effects, error: {error:?}, validator: {validator_name:?}"
    )]
    FailedToVerifyTxCertWithExecutedEffects {
        validator_name: AuthorityName,
        error: String,
    },
    #[error("Too many authority errors were detected for {}: {:?}", action, errors)]
    TooManyIncorrectAuthorities {
        errors: Vec<(AuthorityName, SomaError)>,
        action: String,
    },

    // Certificate verification and execution
    #[error(
        "Signature or certificate from wrong epoch, expected {expected_epoch}, got {actual_epoch}"
    )]
    WrongEpoch {
        expected_epoch: EpochId,
        actual_epoch: EpochId,
    },

    #[error("Expect {expected} signer signatures but got {actual}")]
    SignerSignatureNumberMismatch { expected: usize, actual: usize },
    #[error("Required Signature from {expected} is absent {:?}", actual)]
    SignerSignatureAbsent {
        expected: String,
        actual: Vec<String>,
    },
    #[error("Value was not signed by a known authority. signer: {:?}, index: {:?}, committee: {committee}", signer, index)]
    UnknownSigner {
        signer: Option<String>,
        index: Option<u32>,
        committee: Box<Committee>,
    },

    #[error("Invalid authenticator")]
    InvalidAuthenticator,

    #[error("Signatures in a certificate must form a quorum")]
    CertificateRequiresQuorum,

    #[error(
        "Validator {:?} responded multiple signatures for the same message, conflicting: {:?}",
        signer,
        conflicting_sig
    )]
    StakeAggregatorRepeatedSigner {
        signer: AuthorityName,
        conflicting_sig: bool,
    },

    #[error("Validator {authority:?} is faulty in a Byzantine manner: {reason:?}")]
    ByzantineAuthoritySuspicion {
        authority: AuthorityName,
        reason: String,
    },

    #[error("Invalid digest length. Expected {expected}, got {actual}")]
    InvalidDigestLength { expected: usize, actual: usize },

    #[error(
        "Failed to get a quorum of signed effects when processing transaction: {effects_map:?}"
    )]
    QuorumFailedToGetEffectsQuorumWhenProcessingTransaction {
        effects_map: BTreeMap<TransactionEffectsDigest, (Vec<AuthorityName>, VotingPower)>,
    },

    #[error("Transaction certificate processing failed: {err}")]
    ErrorWhileProcessingCertificate { err: String },

    // Unsupported Operations on Fullnode
    #[error("Fullnode does not support handle_certificate")]
    FullNodeCantHandleCertificate,

    #[error("Invalid transaction digest.")]
    InvalidTransactionDigest,

    #[error(
        "Could not find the referenced object {:?} at version {:?}",
        object_id,
        version
    )]
    ObjectNotFound {
        object_id: ObjectID,
        version: Option<Version>,
    },
    #[error(
        "Object {obj_ref:?} already locked by a different transaction: {pending_transaction:?}"
    )]
    ObjectLockConflict {
        obj_ref: ObjectRef,
        pending_transaction: TransactionDigest,
    },
    #[error("Object {provided_obj_ref:?} is not available for consumption, its current version: {current_version:?}")]
    ObjectVersionUnavailableForConsumption {
        provided_obj_ref: ObjectRef,
        current_version: Version,
    },
    #[error(
        "Invalid Object digest for object {object_id:?}. Expected digest : {expected_digest:?}"
    )]
    InvalidObjectDigest {
        object_id: ObjectID,
        expected_digest: ObjectDigest,
    },

    #[error("Storage error: {0}")]
    Storage(String),

    #[error("Attempt to re-initialize a transaction lock for objects {:?}.", refs)]
    ObjectLockAlreadyInitialized { refs: Vec<ObjectRef> },

    #[error("Failed to read or deserialize system state related data structures on-chain: {0}")]
    SystemStateReadError(String),

    #[error("Network config error: {0:?}")]
    NetworkConfig(String),

    #[error("Failed to connect as client: {0:?}")]
    NetworkClientConnection(String),

    #[error("Failed to connect as server: {0:?}")]
    NetworkServerConnection(String),

    #[error("Peer {0} not found")]
    PeerNotFound(PeerId),

    #[error("Consensus error: {0}")]
    Consensus(String),

    #[error("No committee for epoch: {0}")]
    NoCommitteeForEpoch(Epoch),
}

impl From<Status> for SomaError {
    fn from(status: Status) -> Self {
        // if status.message() == "Too many requests" {
        //     return Self::TooManyRequests;
        // }

        let result = bcs::from_bytes::<SomaError>(status.details());
        if let Ok(sui_error) = result {
            sui_error
        } else {
            Self::RpcError(
                status.message().to_owned(),
                status.code().description().to_owned(),
            )
        }
    }
}

impl From<SomaError> for Status {
    fn from(error: SomaError) -> Self {
        let bytes = bcs::to_bytes(&error).unwrap();
        Status::with_details(tonic::Code::Internal, error.to_string(), bytes.into())
    }
}

impl From<String> for SomaError {
    fn from(error: String) -> Self {
        SomaError::GenericAuthorityError { error }
    }
}

impl From<&str> for SomaError {
    fn from(error: &str) -> Self {
        SomaError::GenericAuthorityError {
            error: error.to_string(),
        }
    }
}

impl From<ConsensusError> for SomaError {
    fn from(e: ConsensusError) -> Self {
        Self::Consensus(e.to_string())
    }
}

type BoxError = Box<dyn std::error::Error + Send + Sync + 'static>;

pub type ExecutionErrorKind = ExecutionFailureStatus;

#[derive(Debug)]
pub struct ExecutionError {
    inner: Box<ExecutionErrorInner>,
}

#[derive(Debug)]
struct ExecutionErrorInner {
    kind: ExecutionErrorKind,
    source: Option<BoxError>,
}

impl ExecutionError {
    pub fn new(kind: ExecutionErrorKind, source: Option<BoxError>) -> Self {
        Self {
            inner: Box::new(ExecutionErrorInner { kind, source }),
        }
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

#[derive(Error, Debug)]
pub enum ConsensusError {
    #[error("Genesis blocks should not be queried!")]
    UnexpectedGenesisBlockRequested,

    #[error("Too many authorities have been provided from authority {0}")]
    TooManyAuthoritiesProvided(AuthorityIndex),

    #[error("Too many blocks have been requested from authority {0}")]
    TooManyFetchBlocksRequested(AuthorityIndex),

    #[error("Unexpected block authority {0} from peer {1}")]
    UnexpectedAuthority(AuthorityIndex, AuthorityIndex),

    #[error("Provided size of highest accepted rounds parameter, {0}, is different than committee size, {1}")]
    InvalidSizeOfHighestAcceptedRounds(usize, usize),

    #[error("Block {block_ref:?} rejected: {reason}")]
    BlockRejected { block_ref: BlockRef, reason: String },

    #[error("Error deserializing block: {0}")]
    MalformedBlock(bcs::Error),

    #[error("Error deserializing commit: {0}")]
    MalformedCommit(bcs::Error),

    #[error("Received no commit from peer {peer}")]
    NoCommitReceived { peer: String },

    #[error(
        "Received unexpected start commit from peer {peer}: requested {start}, received {commit:?}"
    )]
    UnexpectedStartCommit {
        peer: String,
        start: CommitIndex,
        commit: Box<Commit>,
    },

    #[error(
        "Received unexpected commit sequence from peer {peer}: {prev_commit:?}, {curr_commit:?}"
    )]
    UnexpectedCommitSequence {
        peer: String,
        prev_commit: Box<Commit>,
        curr_commit: Box<Commit>,
    },

    #[error("Expected {requested} but received {received} blocks returned from peer {peer}")]
    UnexpectedNumberOfBlocksFetched {
        peer: String,
        requested: usize,
        received: usize,
    },

    #[error("Received unexpected block from peer {peer}: {requested:?} vs {received:?}")]
    UnexpectedBlockForCommit {
        peer: String,
        requested: BlockRef,
        received: BlockRef,
    },

    #[error("Received no blocks from peer's commit {peer}: {commit:?}")]
    NoBlocksForCommit { peer: String, commit: Box<Commit> },

    #[error("No available authority to fetch commits")]
    NoAvailableAuthorityToFetchCommits,

    #[error("Not enough votes ({stake}) on end commit from peer {peer}: {commit:?}")]
    NotEnoughCommitVotes {
        stake: Stake,
        peer: String,
        commit: Box<Commit>,
    },

    #[error("Ancestor is in wrong position: block {block_authority}, ancestor {ancestor_authority}, position {position}")]
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

    #[error("Block has wrong epoch: expected {expected}, actual {actual}")]
    WrongEpoch { expected: Epoch, actual: Epoch },

    #[error("Insufficient stake from parents: {parent_stakes} < {quorum}")]
    InsufficientParentStakes { parent_stakes: Stake, quorum: Stake },

    #[error("Genesis blocks should only be generated from Committee!")]
    UnexpectedGenesisBlock,

    #[error("Invalid transaction: {0}")]
    InvalidTransaction(String),

    #[error("Ancestors max timestamp {max_timestamp_ms} > block timestamp {block_timestamp_ms}")]
    InvalidBlockTimestamp {
        max_timestamp_ms: u64,
        block_timestamp_ms: u64,
    },

    #[error("Error serializing: {0}")]
    SerializationFailure(bcs::Error),

    #[error("Invalid authority index: {index} > {max}")]
    InvalidAuthorityIndex { index: AuthorityIndex, max: usize },

    #[error("Failed to deserialize signature: {0}")]
    MalformedSignature(FastCryptoError),

    #[error("Failed to verify the block's signature: {0}")]
    SignatureVerificationFailure(FastCryptoError),

    #[error("Synchronizer for fetching blocks directly from {0} is saturated")]
    SynchronizerSaturated(AuthorityIndex),

    #[error("Too many blocks have been returned from authority {0} when requesting to fetch missing blocks")]
    TooManyFetchedBlocksReturned(AuthorityIndex),

    #[error("Unexpected block returned while fetching missing blocks")]
    UnexpectedFetchedBlock {
        index: AuthorityIndex,
        block_ref: BlockRef,
    },

    #[error(
        "Unexpected block {block_ref} returned while fetching last own block from peer {index}"
    )]
    UnexpectedLastOwnBlock {
        index: AuthorityIndex,
        block_ref: BlockRef,
    },

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

    #[error("Invalid state hash: {expected:?} != {actual:?}")]
    InvalidStateHash {
        expected: Digest<32>,
        actual: Digest<32>,
    },

    #[error("Storage error: {0}")]
    Storage(String),

    #[error("No committee for epoch: {0}")]
    NoCommitteeForEpoch(Epoch),

    #[error("Invalid end of epoch data in block: {0}")]
    InvalidEndOfEpoch(String),
}

pub type ConsensusResult<T> = Result<T, ConsensusError>;

impl From<crate::storage::storage_error::Error> for ConsensusError {
    fn from(e: crate::storage::storage_error::Error) -> Self {
        Self::Storage(e.to_string())
    }
}

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
