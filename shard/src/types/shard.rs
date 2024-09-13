use crate::error::ShardError;
use crate::types::manifest::ManifestDigest;
use crate::ProtocolKeySignature;
use crate::{
    crypto::{DefaultHashFunction, DIGEST_LENGTH},
    error::ShardResult,
};
use bytes::Bytes;
use fastcrypto::hash::{Digest, HashFunction};
use serde::{Deserialize, Serialize};
use std::{
    fmt,
    hash::{Hash, Hasher},
};

use super::authority_committee::Epoch;
use super::modality::Modality;
use super::network_committee::NetworkIdentityIndex;
use super::transaction::SignedTransactionDigest;

/// Contains the manifest digest and leader. By keeping these details
/// secret from the broader network and only sharing with selected shard members
/// we can reduce censorship related attacks that target specific users
#[derive(Default, Clone, Debug, Serialize, Deserialize)]
pub(crate) struct ShardSecret {
    /// the digest that uniquely identifies a manifest
    manifest_digest: ManifestDigest, //TODO: switch to a manifest ref to be more in-line?
    /// the node that will be coordinating communication and selecting the commits
    leader: NetworkIdentityIndex,
}

impl ShardSecret {
    /// creates a new shard secret given a manifest digest and leader
    const fn new(manifest_digest: ManifestDigest, leader: NetworkIdentityIndex) -> Self {
        Self {
            manifest_digest,
            leader,
        }
    }
}

macros::generate_digest_type!(ShardSecret);

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShardEntropy {
    signature: ProtocolKeySignature,
    transaction_digest: SignedTransactionDigest,
}

macros::generate_digest_type!(ShardEntropy);

/// Uniquely identifies a shard by the epoch, leader, entropy, and modality
#[derive(Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct ShardRef {
    /// the epoch that this shard was sampled from, important since committees change each epoch
    epoch: Epoch,
    /// the leader of the shard
    leader: NetworkIdentityIndex,
    /// the tbls threshold signature that acts as a safe source of randomness
    entropy_digest: ShardEntropyDigest,
    /// modality
    modality: Modality,
}

impl ShardRef {
    /// lex min.
    const MIN: Self = Self {
        epoch: 0,
        leader: NetworkIdentityIndex::MIN,
        entropy_digest: ShardEntropyDigest::MIN,
        modality: Modality::text(),
    };

    /// lex max
    const MAX: Self = Self {
        epoch: u64::MAX,
        leader: NetworkIdentityIndex::MAX,
        entropy_digest: ShardEntropyDigest::MAX,
        modality: Modality::video(),
    };

    /// creates a new shard ref
    const fn new(
        epoch: Epoch,
        leader: NetworkIdentityIndex,
        entropy_digest: ShardEntropyDigest,
        modality: Modality,
    ) -> Self {
        Self {
            epoch,
            leader,
            entropy_digest,
            modality,
        }
    }
}

// TODO: re-evaluate formats for production debugging.
impl fmt::Display for ShardRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(
            f,
            "Shard({},{},{})",
            self.epoch, self.leader, self.entropy_digest
        )
    }
}

impl fmt::Debug for ShardRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(
            f,
            "Shard({},{},{})",
            self.epoch, self.leader, self.entropy_digest
        )
    }
}

impl Hash for ShardRef {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write(&self.entropy_digest.0[..8]);
    }
}
