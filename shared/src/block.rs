use std::{
    fmt,
    hash::{Hash, Hasher},
};

use crate::{commit::CommitVote, digest::Digest, signed::Signed};
use enum_dispatch::enum_dispatch;
use fastcrypto::ed25519::Ed25519Signature;
use serde::{Deserialize, Serialize};

use crate::{authority_committee::AuthorityIndex, transaction::SignedTransaction};

pub(crate) type Epoch = u64;

/// Round number of a block.
type Round = u32;

/// Round zero for the genesis round
const GENESIS_ROUND: Round = 0;

/// Block proposal timestamp in milliseconds.
type BlockTimestampMs = u64;

pub type TransactionIndex = u16;

/// Votes on transactions in a specific block.
/// Reject votes are explicit. The rest of transactions in the block receive implicit accept votes.
// TODO: look into making fields `pub`.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub(crate) struct BlockTransactionVotes {
    pub(crate) block_ref: BlockRef,
    pub(crate) rejects: Vec<TransactionIndex>,
}

#[enum_dispatch]
pub trait BlockAPI {
    fn epoch(&self) -> Epoch;
    fn round(&self) -> Round;
    fn author(&self) -> AuthorityIndex;
    fn slot(&self) -> Slot;
    fn timestamp_ms(&self) -> BlockTimestampMs;
    fn ancestors(&self) -> &[BlockRef];
    fn transactions(&self) -> &[SignedTransaction];
    fn commit_votes(&self) -> &[CommitVote];
    fn transaction_votes(&self) -> &[BlockTransactionVotes];
    fn misbehavior_reports(&self) -> &[MisbehaviorReport];
}

#[derive(Clone, Deserialize, Serialize, PartialEq, Eq)]
#[enum_dispatch(BlockAPI)]
pub enum Block {
    V1(BlockV1),
}

/// BlockV1 is the first implementation of block
#[derive(Clone, Deserialize, Serialize, PartialEq, Eq)]
struct BlockV1 {
    epoch: Epoch,
    round: Round,
    author: AuthorityIndex,
    timestamp_ms: BlockTimestampMs,
    ancestors: Vec<BlockRef>,
    transactions: Vec<SignedTransaction>,
    transaction_votes: Vec<BlockTransactionVotes>,
    commit_votes: Vec<CommitVote>,
    misbehavior_reports: Vec<MisbehaviorReport>,
}

impl BlockV1 {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        epoch: Epoch,
        round: Round,
        author: AuthorityIndex,
        timestamp_ms: BlockTimestampMs,
        ancestors: Vec<BlockRef>,
        transactions: Vec<SignedTransaction>,
        commit_votes: Vec<CommitVote>,
        transaction_votes: Vec<BlockTransactionVotes>,
        misbehavior_reports: Vec<MisbehaviorReport>,
    ) -> Self {
        Self {
            epoch,
            round,
            author,
            timestamp_ms,
            ancestors,
            transactions,
            commit_votes,
            transaction_votes,
            misbehavior_reports,
        }
    }

    fn genesis_block(epoch: Epoch, author: AuthorityIndex) -> Self {
        Self {
            epoch,
            round: GENESIS_ROUND,
            author,
            timestamp_ms: 0,
            ancestors: vec![],
            transactions: vec![],
            commit_votes: vec![],
            transaction_votes: vec![],
            misbehavior_reports: vec![],
        }
    }
}

impl BlockAPI for BlockV1 {
    fn epoch(&self) -> Epoch {
        self.epoch
    }

    fn round(&self) -> Round {
        self.round
    }

    fn author(&self) -> AuthorityIndex {
        self.author
    }

    fn slot(&self) -> Slot {
        Slot::new(self.round, self.author)
    }

    fn timestamp_ms(&self) -> BlockTimestampMs {
        self.timestamp_ms
    }

    fn ancestors(&self) -> &[BlockRef] {
        &self.ancestors
    }

    fn transactions(&self) -> &[SignedTransaction] {
        &self.transactions
    }

    fn transaction_votes(&self) -> &[BlockTransactionVotes] {
        &self.transaction_votes
    }

    fn commit_votes(&self) -> &[CommitVote] {
        &self.commit_votes
    }

    fn misbehavior_reports(&self) -> &[MisbehaviorReport] {
        &self.misbehavior_reports
    }
}

/// `BlockRef` uniquely identifies a `BlockHeader` and indirectly a Block via `digest`. It also contains the slot
/// info (round and author) so it can be used in logic such as aggregating stakes for a round.
#[derive(Clone, Copy, Serialize, Default, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct BlockRef {
    /// returns the round for the blockheader
    round: Round,
    /// returns the author
    author: AuthorityIndex,
    /// the digest for a blockheader (this is the unique reference)
    digest: Digest<Signed<Block, Ed25519Signature>>,
}

impl BlockRef {
    /// MIN for lex
    const MIN: Self = Self {
        round: 0,
        author: AuthorityIndex::MIN,
        digest: Digest::MIN,
    };

    /// MIN for lex
    const MAX: Self = Self {
        round: u32::MAX,
        author: AuthorityIndex::MAX,
        digest: Digest::MAX,
    };

    /// creates a new block header ref
    pub fn new(
        round: Round,
        author: AuthorityIndex,
        digest: Digest<Signed<Block, Ed25519Signature>>,
    ) -> Self {
        Self {
            round,
            author,
            digest,
        }
    }
}

// TODO: re-evaluate formats for production debugging.
impl fmt::Display for BlockRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(f, "B{}({},{})", self.round, self.author, self.digest)
    }
}

impl fmt::Debug for BlockRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(f, "B{}({},{:?})", self.round, self.author, self.digest)
    }
}

impl Hash for BlockRef {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.digest.hash(state);
    }
}
/// A block can attach reports of misbehavior by other authorities.
#[derive(Clone, Serialize, Deserialize, Debug, PartialEq, Eq)]
struct MisbehaviorReport {
    /// the author being reported
    target: AuthorityIndex,
    /// An enum to switch on different proofs
    proof: MisbehaviorProof,
}

/// Proof of misbehavior are usually signed block(s) from the misbehaving authority.
#[derive(Clone, Serialize, Deserialize, Debug, PartialEq, Eq)]
enum MisbehaviorProof {
    /// Proof of an invalid block
    InvalidBlock(BlockRef),
}

/// `MisbehaviorReportRef` uniquely identifies a `VerifiedMisbehaviorReport` via `digest`. It also contains the slot
/// info (round and author) so it can be used in logic such as aggregating stakes for a round.
#[derive(Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
struct MisbehaviorReportRef {
    /// digest is a hash of the misbehavior report to create a constantly sized report
    digest: Digest<MisbehaviorReport>,
}

impl MisbehaviorReportRef {
    /// lexigraphical min
    const MIN: Self = Self {
        digest: Digest::MIN,
    };

    /// lexigraphical max
    const MAX: Self = Self {
        digest: Digest::MAX,
    };

    /// creates a new misbehavior report ref
    const fn new(digest: Digest<MisbehaviorReport>) -> Self {
        Self { digest }
    }
}

// TODO: re-evaluate formats for production debugging.
impl fmt::Display for MisbehaviorReportRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(f, "T{}", self.digest)
    }
}

impl fmt::Debug for MisbehaviorReportRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(f, "T{}", self.digest)
    }
}

impl Hash for MisbehaviorReportRef {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.digest.hash(state)
    }
}

/// Slot is the position of blocks in the DAG. It can contain 0, 1 or multiple blocks
/// from the same authority at the same round.
#[derive(Clone, Copy, PartialEq, PartialOrd, Default, Hash)]
pub struct Slot {
    pub round: Round,
    pub authority: AuthorityIndex,
}

impl Slot {
    pub fn new(round: Round, authority: AuthorityIndex) -> Self {
        Self { round, authority }
    }
}
