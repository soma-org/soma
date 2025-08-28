//! In a previous version, the shard was composed of a single set of encoders.
//! What was realized is that the security (probability of a dishonest majority)
//! should be scaled seperately from the number of computers performing computation.
//! This allows for the computation set that is generating an embedding to be tuned
//! independently of security considerations. The seperation of concerns is also slightly
//! more secure compared to encoders that are directly impacted by the outcome.
use crate::{
    authority_committee::{AuthorityBitSet, AuthorityCommittee, AuthorityIndex},
    block::BlockRef,
    crypto::{
        address::Address,
        keys::{
            AuthorityAggregateSignature, AuthorityKeyPair, AuthoritySignature,
            EncoderAggregateSignature, EncoderPublicKey, ProtocolKeySignature,
        },
    },
    digest::Digest,
    entropy::{BlockEntropy, BlockEntropyProof, EntropyVDF},
    finality_proof::{BlockClaim, FinalityProof},
    metadata::{Metadata, MetadataCommitment},
    scope::{Scope, ScopedMessage},
    transaction::{
        ShardTransaction, SignedTransaction, TransactionData, TransactionExpiration,
        TransactionKind,
    },
};

use serde::{Deserialize, Serialize};

use crate::error::{ShardError, ShardResult};

use super::encoder_committee::{CountUnit, Epoch};

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct Shard {
    quorum_threshold: CountUnit,
    encoders: Vec<EncoderPublicKey>,
    seed: Digest<ShardEntropy>,
    epoch: Epoch,
}

impl Shard {
    pub fn new(
        quorum_threshold: CountUnit,
        encoders: Vec<EncoderPublicKey>,
        seed: Digest<ShardEntropy>,
        epoch: Epoch,
    ) -> Self {
        Self {
            quorum_threshold,
            encoders,
            seed,
            epoch,
        }
    }
    pub fn encoders(&self) -> Vec<EncoderPublicKey> {
        self.encoders.clone()
    }
    pub fn size(&self) -> usize {
        self.encoders.len()
    }

    pub fn contains(&self, encoder: &EncoderPublicKey) -> bool {
        self.encoders.contains(encoder)
    }

    pub fn quorum_threshold(&self) -> CountUnit {
        self.quorum_threshold
    }

    pub fn rejection_threshold(&self) -> CountUnit {
        self.size() as u32 - self.quorum_threshold + 1
    }

    pub fn digest(&self) -> ShardResult<Digest<Self>> {
        Digest::new(self).map_err(ShardError::DigestFailure)
    }
    pub fn epoch(&self) -> Epoch {
        self.epoch
    }
}

/// The Digest<ShardEntropy> acts as a seed for random sampling from the encoder committee.
/// Digest<MetadataCommitment> is included inside of a tx which is a one way fn whereas
/// this entropy uses the actual values of the serialized type of MetadataCommitment to create the Digest.
///
/// BlockEntropy is derived from VDF(Epoch, BlockRef, iterations)
#[derive(Debug, Serialize, Deserialize)]
pub struct ShardEntropy {
    metadata_commitment: MetadataCommitment,
    entropy: BlockEntropy,
}

impl ShardEntropy {
    pub fn new(metadata_commitment: MetadataCommitment, entropy: BlockEntropy) -> Self {
        Self {
            metadata_commitment,
            entropy,
        }
    }
}
