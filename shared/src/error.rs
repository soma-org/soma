// use consensus_config::{AuthorityIndex, Epoch, Stake};
use fastcrypto::error::FastCryptoError;
use strum_macros::IntoStaticStr;
use thiserror::Error;
// use typed_store::TypedStoreError;

// use crate::{
//     block::{BlockRef, Round},
//     commit::{Commit, CommitIndex},
// };

// use consensus_config::{AuthorityIndex, Epoch, Stake};

// use typed_store::TypedStoreError;

// use crate::{
//     block::{BlockRef, Round},
//     commit::{Commit, CommitIndex},
// };

use crate::{authority_committee::AuthorityIndex, block::TransactionIndex};

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
    #[error("Invalid transaction index: {index}")]
    InvalidTransactionIndex { index: TransactionIndex },
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

    #[error("Failed to parse URL: {0}")]
    UrlParseError(String),

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

    #[error("Shard error: {0}")]
    Shard(String),
}

pub type SharedResult<T> = Result<T, SharedError>;

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

/// Errors that can occur when processing blocks, reading from storage, or encountering shutdown.
#[derive(Clone, Debug, Error, IntoStaticStr)]
pub enum ShardError {
    #[error("Recv duplicate error")]
    RecvDuplicate,
    #[error("Send duplicate error")]
    SendDuplicate,
    #[error("Evaluation error: {0}")]
    EvaluationError(EvaluationError),
    #[error("Model error: {0}")]
    InferenceError(InferenceError),
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
    #[error("failed type verification: {0}")]
    FailedTypeVerification(String),
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

    #[error("Concurrency error: {0:?}")]
    ConcurrencyError(String),

    #[error("Consensus has shut down!")]
    Shutdown,

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
    #[error("Failed to parse URL: {0}")]
    UrlParseError(String),
    #[error("Failed to send request: {0:?}")]
    NetworkRequest(String),
    #[error("write error: {0}")]
    WriteError(String),
    #[error("ObjectStorage: {0}")]
    ObjectStorage(String),
    #[error("Not found: {0}")]
    NotFound(String),
    #[error("Verification error: {0}")]
    VerificationError(String),
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
}

impl From<ShardError> for SharedError {
    fn from(e: ShardError) -> Self {
        Self::Shard(e.to_string())
    }
}
