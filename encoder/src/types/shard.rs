use std::hash::{Hash, Hasher};

use enum_dispatch::enum_dispatch;
use serde::{Deserialize, Serialize};
use shared::{
    crypto::keys::EncoderAggregateSignature, digest::Digest, metadata::Metadata,
transaction::SignedTransaction,
};
use strum_macros::Display;

use super::encoder_committee::EncoderIndex;

type Epoch = u64;
type QuorumUnit = u32;


pub(crate) struct Shard {
    epoch: Epoch,
    quorum_threshold: QuorumUnit,
    encoders: Vec<EncoderIndex>,
}

impl Shard {
    pub(crate) fn new(epoch: Epoch, quorum_threshold: QuorumUnit, encoders: Vec<EncoderIndex>) -> Self {
        Self {
            epoch,
            quorum_threshold,
            encoders
        }
    }
    pub(crate) fn epoch(&self) -> Epoch {
        self.epoch
    }
    pub(crate) fn encoders(&self) -> Vec<EncoderIndex> {
        self.encoders.clone()
    }

    pub(crate) fn size(&self) -> usize {
        self.encoders.len()
    }

    pub(crate) fn quorum_threshold(&self) -> QuorumUnit {
        self.quorum_threshold
    }

    #[cfg(test)]
    pub(crate) fn new_for_test(epoch: Epoch, quorum_threshold: QuorumUnit, encoders: Vec<EncoderIndex>) -> Self {
        Self::new(epoch, quorum_threshold, encoders)
    }

}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct ShardEntropy {
    // Note: intentionally breaking the ordering of metadata and nonce due to serialization with BCS being
    // sequential. 

    /// digest of the metadata, for valid inputs the data cannot be encrypted and the uncompressed
    /// size should match. Digests without encryption keys should be consistent for the same bytes data every time.  
    metadata_digest: Digest<Metadata>,
    /// manipulation free randomness that cannot be biased or known beforehand
    // TODO: need to change this to the proper signature type but will do later
    threshold_block_signature: EncoderAggregateSignature,
    /// ensures that two identical metadata/nonce combinations cannot overlap in the same block
    /// unique for the individual + object version e.g. account balance
    transaction_digest: Digest<SignedTransaction>,
    /// especially useful for batch processing when a single transaction can contain multiple metadata
    /// commitments but they are not allowed to repeat commitment digests. The nonce is factored in
    /// such that identical data inside a batch still gets a unique shard.
    nonce: [u8;32],
}


/// Shard commit is the wrapper that contains the versioned shard commit. It
/// represents the encoders response to a batch of data
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord, Display)]
#[enum_dispatch(ShardRefAPI)]
pub enum ShardRef {
    V1(ShardRefV1),
}

/// `ShardRefAPI` is the trait that every shard commit version must implement
#[enum_dispatch]
trait ShardRefAPI {
    fn epoch(&self) -> &Epoch;
    fn seed(&self) -> &Digest<ShardEntropy>;
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
struct ShardRefV1 {
    /// the epoch that this shard was sampled from, important since committees change each epoch
    epoch: Epoch,
    /// the digest from the tbls threshold signature and data hash that when combined forms a unique source of randomness
    seed: Digest<ShardEntropy>,
}

impl ShardRefV1 {
    /// create a shard commit v1
    pub(crate) const fn new(epoch: Epoch, seed: Digest<ShardEntropy>) -> Self {
        Self { epoch, seed }
    }
}

impl ShardRefAPI for ShardRefV1 {
    fn epoch(&self) -> &Epoch {
        &self.epoch
    }

    fn seed(&self) -> &Digest<ShardEntropy> {
        &self.seed
    }
}

// impl ShardRef {
//     /// lex min.
//     const MIN: Self = Self {
//         epoch: 0,
//         leader: NetworkingIndex::MIN,
//         entropy_digest: Digest::MIN,
//         modality: Modality::text(),
//     };

//     /// lex max
//     const MAX: Self = Self {
//         epoch: u64::MAX,
//         leader: NetworkingIndex::MAX,
//         entropy_digest: Digest::MAX,
//         modality: Modality::video(),
//     };

// }

impl Hash for ShardRef {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write(&self.seed().as_ref()[..8]);
    }
}