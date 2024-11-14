use fastcrypto::{error::FastCryptoError, hash::Digest};
use thiserror::Error;

use crate::{block::BlockRef, commit::Commit, CommitIndex, Round};
use types::committee::{AuthorityIndex, Epoch, Stake};
#[derive(Error, Debug)]
pub(crate) enum ConsensusError {
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
    NoCommitReceived { peer: AuthorityIndex },

    #[error(
        "Received unexpected start commit from peer {peer}: requested {start}, received {commit:?}"
    )]
    UnexpectedStartCommit {
        peer: AuthorityIndex,
        start: CommitIndex,
        commit: Box<Commit>,
    },

    #[error(
        "Received unexpected commit sequence from peer {peer}: {prev_commit:?}, {curr_commit:?}"
    )]
    UnexpectedCommitSequence {
        peer: AuthorityIndex,
        prev_commit: Box<Commit>,
        curr_commit: Box<Commit>,
    },

    #[error(
        "Expected {requested} but received {received} blocks returned from authority {authority}"
    )]
    UnexpectedNumberOfBlocksFetched {
        authority: AuthorityIndex,
        requested: usize,
        received: usize,
    },

    #[error("Received unexpected block from peer {peer}: {requested:?} vs {received:?}")]
    UnexpectedBlockForCommit {
        peer: AuthorityIndex,
        requested: BlockRef,
        received: BlockRef,
    },

    #[error("No available authority to fetch commits")]
    NoAvailableAuthorityToFetchCommits,

    #[error("Not enough votes ({stake}) on end commit from peer {peer}: {commit:?}")]
    NotEnoughCommitVotes {
        stake: Stake,
        peer: AuthorityIndex,
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
