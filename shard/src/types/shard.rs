use crate::ProtocolKeySignature;
use enum_dispatch::enum_dispatch;
use serde::{Deserialize, Serialize};
use std::hash::{Hash, Hasher};

use super::authority_committee::Epoch;
use super::data::Data;
use super::digest::Digest;
use super::manifest::Manifest;
use super::modality::Modality;
use super::network_committee::NetworkingIndex;
use super::transaction::SignedTransaction;

pub(crate) struct Shard {
    members: Vec<NetworkingIndex>,
}

impl Shard {
    pub(crate) fn new(members: Vec<NetworkingIndex>) -> Self {
        Self { members }
    }

    pub(crate) fn members(&self) -> Vec<NetworkingIndex> {
        self.members.clone()
    }
}

/// Contains the manifest digest and leader. By keeping these details
/// secret from the broader network and only sharing with selected shard members
/// we can reduce censorship related attacks that target specific users
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct ShardSecret {
    data_digest: Digest<Data>, //TODO: switch to a manifest ref to be more in-line?
}

impl ShardSecret {
    /// creates a new shard secret given a manifest digest and leader
    const fn new(data_digest: Digest<Data>) -> Self {
        Self { data_digest }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShardEntropy {
    signature: ProtocolKeySignature,
    transaction_digest: Digest<SignedTransaction>,
}

/// Shard commit is the wrapper that contains the versioned shard commit. It
/// represents the encoders response to a batch of data
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[enum_dispatch(ShardRefAPI)]
pub enum ShardRef {
    V1(ShardRefV1),
}

/// `ShardRefAPI` is the trait that every shard commit version must implement
#[enum_dispatch]
trait ShardRefAPI {
    fn epoch(&self) -> &Epoch;
    fn modality(&self) -> &Modality;
    fn seed(&self) -> &Digest<ShardEntropy>;
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
struct ShardRefV1 {
    /// the epoch that this shard was sampled from, important since committees change each epoch
    epoch: Epoch,
    /// modality
    modality: Modality,
    /// the digest from the tbls threshold signature and data hash that when combined forms a unique source of randomness
    seed: Digest<ShardEntropy>,
}

impl ShardRefV1 {
    /// create a shard commit v1
    pub(crate) const fn new(epoch: Epoch, modality: Modality, seed: Digest<ShardEntropy>) -> Self {
        Self {
            epoch,
            modality,
            seed,
        }
    }
}

impl ShardRefAPI for ShardRefV1 {
    fn epoch(&self) -> &Epoch {
        &self.epoch
    }
    fn modality(&self) -> &Modality {
        &self.modality
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
