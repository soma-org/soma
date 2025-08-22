use bytes::Bytes;
use fastcrypto::hash::{Digest, HashFunction};
use serde::{Deserialize, Serialize};
use std::{
    fmt,
    hash::{Hash, Hasher},
    ops::Deref,
    sync::Arc,
};

use super::context::Context;
use super::{commit::CommitVote, validator_set::ValidatorSet};
use crate::{
    accumulator::{Accumulator, CommitIndex},
    committee::{AuthorityIndex, EncoderCommittee, Epoch, NetworkingCommittee},
};
use crate::{
    committee::Committee,
    intent::{Intent, IntentMessage, IntentScope},
};
use crate::{
    crypto::{
        AggregateAuthoritySignature, AuthorityPublicKeyBytes, AuthoritySignature,
        DefaultHash as DefaultHashFunction, ProtocolKeyPair, ProtocolKeySignature,
        ProtocolPublicKey, DIGEST_LENGTH,
    },
    digests::ECMHLiveObjectSetDigest,
};
use crate::{
    ensure,
    error::{ConsensusError, ConsensusResult},
};

pub type Round = u32;

pub const GENESIS_ROUND: Round = 0;

/// Block proposal timestamp in milliseconds.
pub type BlockTimestampMs = u64;

#[derive(Clone, Eq, PartialEq, Serialize, Deserialize, Default, Debug)]
pub struct Transaction {
    data: Bytes,
}

impl Transaction {
    pub fn new(data: Vec<u8>) -> Self {
        Self { data: data.into() }
    }

    pub fn data(&self) -> &[u8] {
        &self.data
    }

    pub fn into_data(self) -> Bytes {
        self.data
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EndOfEpochData {
    pub next_epoch_start_timestamp_ms: u64,
    /// The proposed validator set for next epoch, with each validator's public key and voting power
    pub next_validator_set: Option<ValidatorSet>,
    /// The proposed encoder committee for next epoch
    pub next_encoder_committee: Option<EncoderCommittee>,
    /// The proposed networking committee for next epoch
    pub next_networking_committee: Option<NetworkingCommittee>,

    /// Accumulated state hash digest of the last commit of the epoch
    pub state_hash: Option<ECMHLiveObjectSetDigest>,

    /// BLS signature from this block's author on next_validator_set and next_epoch_committee from blocks in ancestry
    /// Only included if a valid validator set was found in ancestry
    pub validator_set_signature: Option<AuthoritySignature>,
    pub encoder_committee_signature: Option<AuthoritySignature>,

    /// Aggregate BLS signature from ancestor blocks' signatures on next_validator_set and next_encoder_committee
    /// Only included if quorum of ancestor signatures found
    pub validator_aggregate_signature: Option<AggregateAuthoritySignature>,
    pub encoder_aggregate_signature: Option<AggregateAuthoritySignature>,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct Block {
    epoch: Epoch,
    round: Round,
    author: AuthorityIndex,
    timestamp_ms: BlockTimestampMs,
    ancestors: Vec<BlockRef>,
    transactions: Vec<Transaction>,
    commit_votes: Vec<CommitVote>,
    end_of_epoch_data: Option<EndOfEpochData>,
}

impl Block {
    pub fn new(
        epoch: Epoch,
        round: Round,
        author: AuthorityIndex,
        timestamp_ms: BlockTimestampMs,
        ancestors: Vec<BlockRef>,
        transactions: Vec<Transaction>,
        commit_votes: Vec<CommitVote>,
        end_of_epoch_data: Option<EndOfEpochData>,
    ) -> Block {
        Self {
            epoch,
            round,
            author,
            timestamp_ms,
            ancestors,
            transactions,
            commit_votes,
            end_of_epoch_data,
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
            end_of_epoch_data: None,
        }
    }
}

pub trait BlockAPI {
    fn epoch(&self) -> Epoch;
    fn round(&self) -> Round;
    fn author(&self) -> AuthorityIndex;
    fn slot(&self) -> Slot;
    fn timestamp_ms(&self) -> BlockTimestampMs;
    fn ancestors(&self) -> &[BlockRef];
    fn transactions(&self) -> &[Transaction];
    fn commit_votes(&self) -> &[CommitVote];
    fn end_of_epoch_data(&self) -> &Option<EndOfEpochData>;
}

impl BlockAPI for Block {
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

    fn transactions(&self) -> &[Transaction] {
        &self.transactions
    }

    fn commit_votes(&self) -> &[CommitVote] {
        &self.commit_votes
    }

    fn end_of_epoch_data(&self) -> &Option<EndOfEpochData> {
        &self.end_of_epoch_data
    }
}

/// `BlockRef` uniquely identifies a `VerifiedBlock` via `digest`. It also contains the slot
/// info (round and author) so it can be used in logic such as aggregating stakes for a round.
#[derive(Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct BlockRef {
    pub round: Round,
    pub author: AuthorityIndex,
    pub digest: BlockDigest,
    pub epoch: Epoch,
}

impl BlockRef {
    pub const MIN: Self = Self {
        round: 0,
        author: AuthorityIndex::MIN,
        digest: BlockDigest::MIN,
        epoch: 0,
    };

    pub const MAX: Self = Self {
        round: u32::MAX,
        author: AuthorityIndex::MAX,
        digest: BlockDigest::MAX,
        epoch: Epoch::MAX,
    };

    pub fn new(round: Round, author: AuthorityIndex, digest: BlockDigest, epoch: Epoch) -> Self {
        Self {
            round,
            author,
            digest,
            epoch,
        }
    }
}

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
        state.write(&self.digest.0[..8]);
    }
}

#[derive(Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct BlockDigest([u8; DIGEST_LENGTH]);

impl BlockDigest {
    /// Lexicographic min & max digest.
    pub const MIN: Self = Self([u8::MIN; DIGEST_LENGTH]);
    pub const MAX: Self = Self([u8::MAX; DIGEST_LENGTH]);
}

impl Hash for BlockDigest {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write(&self.0[..8]);
    }
}

impl From<BlockDigest> for Digest<{ DIGEST_LENGTH }> {
    fn from(hd: BlockDigest) -> Self {
        Digest::new(hd.0)
    }
}

impl fmt::Display for BlockDigest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(
            f,
            "{}",
            base64::Engine::encode(&base64::engine::general_purpose::STANDARD, self.0)
                .get(0..4)
                .ok_or(fmt::Error)?
        )
    }
}

impl fmt::Debug for BlockDigest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(
            f,
            "{}",
            base64::Engine::encode(&base64::engine::general_purpose::STANDARD, self.0)
        )
    }
}

impl AsRef<[u8]> for BlockDigest {
    fn as_ref(&self) -> &[u8] {
        &self.0
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

    pub fn new_for_test(round: Round, authority: u32) -> Self {
        Self {
            round,
            authority: AuthorityIndex::new_for_test(authority),
        }
    }
}

impl From<BlockRef> for Slot {
    fn from(value: BlockRef) -> Self {
        Slot::new(value.round, value.author)
    }
}

impl fmt::Display for Slot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.authority, self.round,)
    }
}

impl fmt::Debug for Slot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self)
    }
}

/// A Block with its signature, before they are verified.
///
/// Note: `BlockDigest` is computed over this struct, so any added field (without `#[serde(skip)]`)
/// will affect the values of `BlockDigest` and `BlockRef`.
#[derive(Deserialize, Serialize, Clone)]
pub struct SignedBlock {
    inner: Block,
    signature: Bytes,
}

impl SignedBlock {
    /// Should only be used when constructing the genesis blocks
    pub(crate) fn new_genesis(block: Block) -> Self {
        Self {
            inner: block,
            signature: Bytes::default(),
        }
    }

    pub fn new(block: Block, protocol_keypair: &ProtocolKeyPair) -> ConsensusResult<Self> {
        let signature = compute_block_signature(&block, protocol_keypair)?;
        Ok(Self {
            inner: block,
            signature: Bytes::copy_from_slice(signature.to_bytes()),
        })
    }

    pub(crate) fn signature(&self) -> &Bytes {
        &self.signature
    }

    /// This method only verifies this block's signature. Verification of the full block
    /// should be done via BlockVerifier.
    pub(crate) fn verify_signature(&self, committee: &Committee) -> ConsensusResult<()> {
        let block = &self.inner;
        ensure!(
            committee.is_valid_index(block.author()),
            ConsensusError::InvalidAuthorityIndex {
                index: block.author(),
                max: committee.size() - 1
            }
        );
        let authority = committee
            .authority_by_authority_index(block.author())
            .unwrap();
        verify_block_signature(block, self.signature(), &authority.protocol_key)
    }

    /// Serialises the block using the bcs serializer
    pub fn serialize(&self) -> Result<Bytes, bcs::Error> {
        let bytes = bcs::to_bytes(self)?;
        Ok(bytes.into())
    }

    /// Clears signature for testing.
    #[cfg(test)]
    pub(crate) fn clear_signature(&mut self) {
        self.signature = Bytes::default();
    }
}

/// Allow quick access on the underlying Block without having to always refer to the inner block ref.
impl Deref for SignedBlock {
    type Target = Block;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

/// Digest of a block, covering all `Block` fields without its signature.
/// This is used during Block signing and signature verification.
/// This should never be used outside of this file, to avoid confusion with `BlockDigest`.
#[derive(Serialize, Deserialize)]
struct InnerBlockDigest([u8; DIGEST_LENGTH]);

/// Computes the digest of a Block, only for signing and verifications.
fn compute_inner_block_digest(block: &Block) -> ConsensusResult<InnerBlockDigest> {
    let mut hasher = DefaultHashFunction::new();
    hasher.update(bcs::to_bytes(block).map_err(ConsensusError::SerializationFailure)?);
    Ok(InnerBlockDigest(hasher.finalize().into()))
}

/// Wrap a InnerBlockDigest in the intent message.
fn to_consensus_block_intent(digest: InnerBlockDigest) -> IntentMessage<InnerBlockDigest> {
    IntentMessage::new(Intent::consensus_app(IntentScope::ConsensusBlock), digest)
}

/// Process for signing & verying a block signature:
/// 1. Compute the digest of `Block`.
/// 2. Wrap the digest in `IntentMessage`.
/// 3. Sign the serialized `IntentMessage`, or verify signature against it.
fn compute_block_signature(
    block: &Block,
    protocol_keypair: &ProtocolKeyPair,
) -> ConsensusResult<ProtocolKeySignature> {
    let digest = compute_inner_block_digest(block)?;
    let message = bcs::to_bytes(&to_consensus_block_intent(digest))
        .map_err(ConsensusError::SerializationFailure)?;
    Ok(protocol_keypair.sign(&message))
}

fn verify_block_signature(
    block: &Block,
    signature: &[u8],
    protocol_pubkey: &ProtocolPublicKey,
) -> ConsensusResult<()> {
    let digest = compute_inner_block_digest(block)?;
    let message = bcs::to_bytes(&to_consensus_block_intent(digest))
        .map_err(ConsensusError::SerializationFailure)?;
    let sig =
        ProtocolKeySignature::from_bytes(signature).map_err(ConsensusError::MalformedSignature)?;
    protocol_pubkey
        .verify(&message, &sig)
        .map_err(ConsensusError::SignatureVerificationFailure)
}

#[derive(Clone)]
pub struct VerifiedBlock {
    block: Arc<SignedBlock>,

    // Cached Block digest and serialized SignedBlock, to avoid re-computing these values.
    digest: BlockDigest,
    serialized: Bytes,
}

impl VerifiedBlock {
    /// Creates VerifiedBlock from a verified SignedBlock and its serialized bytes.
    pub fn new_verified(signed_block: SignedBlock, serialized: Bytes) -> Self {
        let digest = Self::compute_digest(&serialized);
        VerifiedBlock {
            block: Arc::new(signed_block),
            digest,
            serialized,
        }
    }

    pub fn new_for_test(block: Block) -> Self {
        // Use empty signature in test.
        let signed_block = SignedBlock {
            inner: block,
            signature: Default::default(),
        };
        let serialized: Bytes = bcs::to_bytes(&signed_block)
            .expect("Serialization should not fail")
            .into();
        let digest = Self::compute_digest(&serialized);
        VerifiedBlock {
            block: Arc::new(signed_block),
            digest,
            serialized,
        }
    }

    /// Returns reference to the block.
    pub fn reference(&self) -> BlockRef {
        BlockRef {
            round: self.round,
            author: self.author,
            digest: self.digest,
            epoch: self.epoch,
        }
    }

    pub(crate) fn digest(&self) -> BlockDigest {
        self.digest
    }

    /// Returns the serialized block with signature.
    pub fn serialized(&self) -> &Bytes {
        &self.serialized
    }

    /// Computes digest from the serialized block with signature.
    pub fn compute_digest(serialized: &[u8]) -> BlockDigest {
        let mut hasher = DefaultHashFunction::new();
        hasher.update(serialized);
        BlockDigest(hasher.finalize().into())
    }
}

impl Deref for VerifiedBlock {
    type Target = Block;

    fn deref(&self) -> &Self::Target {
        &self.block.inner
    }
}

impl PartialEq for VerifiedBlock {
    fn eq(&self, other: &Self) -> bool {
        self.digest == other.digest
    }
}

impl fmt::Display for VerifiedBlock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(f, "{}", self.reference())
    }
}

// TODO: re-evaluate formats for production debugging.
impl fmt::Debug for VerifiedBlock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(
            f,
            "{:?}({}ms;)",
            self.reference(),
            // "{:?}({}ms;{:?};{}t;{}c)",
            self.timestamp_ms(),
            // self.ancestors(),
            // self.transactions().len(),
            // self.commit_votes().len(),
        )
    }
}

/// Generates the genesis blocks for the current Committee.
/// The blocks are returned in authority index order.
pub fn genesis_blocks(context: Arc<Context>) -> Vec<VerifiedBlock> {
    context
        .committee
        .authorities()
        .map(|(authority_index, _)| {
            let signed_block = SignedBlock::new_genesis(Block::genesis_block(
                context.committee.epoch(),
                authority_index,
            ));
            let serialized = signed_block
                .serialize()
                .expect("Genesis block serialization failed.");
            // Unnecessary to verify genesis blocks.
            VerifiedBlock::new_verified(signed_block, serialized)
        })
        .collect::<Vec<VerifiedBlock>>()
}

pub fn genesis_blocks_from_committee(committee: Arc<Committee>) -> Vec<VerifiedBlock> {
    committee
        .authorities()
        .map(|(authority_index, _)| {
            let signed_block =
                SignedBlock::new_genesis(Block::genesis_block(committee.epoch(), authority_index));
            let serialized = signed_block
                .serialize()
                .expect("Genesis block serialization failed.");
            // Unnecessary to verify genesis blocks.
            VerifiedBlock::new_verified(signed_block, serialized)
        })
        .collect::<Vec<VerifiedBlock>>()
}

/// Creates fake blocks for testing.
// #[cfg(test)]
#[derive(Clone)]
pub struct TestBlock {
    block: Block,
}

// #[cfg(test)]
impl TestBlock {
    pub fn new(round: Round, author: u32) -> Self {
        Self {
            block: Block {
                round,
                author: AuthorityIndex::new_for_test(author),
                ..Default::default()
            },
        }
    }

    pub(crate) fn set_epoch(mut self, epoch: Epoch) -> Self {
        self.block.epoch = epoch;
        self
    }

    pub(crate) fn set_round(mut self, round: Round) -> Self {
        self.block.round = round;
        self
    }

    pub(crate) fn set_author(mut self, author: AuthorityIndex) -> Self {
        self.block.author = author;
        self
    }

    pub fn set_timestamp_ms(mut self, timestamp_ms: BlockTimestampMs) -> Self {
        self.block.timestamp_ms = timestamp_ms;
        self
    }

    pub fn set_ancestors(mut self, ancestors: Vec<BlockRef>) -> Self {
        self.block.ancestors = ancestors;
        self
    }

    pub fn set_transactions(mut self, transactions: Vec<Transaction>) -> Self {
        self.block.transactions = transactions;
        self
    }

    pub fn set_commit_votes(mut self, commit_votes: Vec<CommitVote>) -> Self {
        self.block.commit_votes = commit_votes;
        self
    }

    pub fn build(self) -> Block {
        self.block
    }
}

#[cfg(test)]
mod tests {

    use crate::consensus::{
        block::{SignedBlock, TestBlock},
        context::Context,
    };
    use crate::error::ConsensusError;
    use fastcrypto::error::FastCryptoError;
    use std::sync::Arc;

    #[tokio::test]
    async fn test_sign_and_verify() {
        let (context, key_pairs, _) = Context::new_for_test(4);
        let context = Arc::new(context);

        // Create a block that authority 2 has created
        let block = TestBlock::new(10, 2).build();

        // Create a signed block with authority's 2 private key
        let author_two_key = &key_pairs[2].1;
        let signed_block = SignedBlock::new(block, author_two_key).expect("Shouldn't fail signing");

        // Now verify the block's signature
        let result = signed_block.verify_signature(&context.committee);
        assert!(result.is_ok());

        // Try to sign authority's 2 block with authority's 1 key
        let block = TestBlock::new(10, 2).build();
        let author_one_key = &key_pairs[1].1;
        let signed_block = SignedBlock::new(block, author_one_key).expect("Shouldn't fail signing");

        // Now verify the block, it should fail
        let result = signed_block.verify_signature(&context.committee);
        match result.err().unwrap() {
            ConsensusError::SignatureVerificationFailure(err) => {
                assert_eq!(err, FastCryptoError::InvalidSignature);
            }
            err => panic!("Unexpected error: {err:?}"),
        }
    }
}
