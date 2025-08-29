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
            EncoderAggregateSignature, EncoderPublicKey, PeerPublicKey, ProtocolKeySignature,
        },
    },
    digest::Digest,
    entropy::{BlockEntropy, BlockEntropyProof, EntropyVDF},
    finality_proof::{BlockClaim, FinalityProof},
    metadata::{
        DownloadableMetadata, DownloadableMetadataV1, Metadata, MetadataCommitment, MetadataV1,
    },
    scope::{Scope, ScopedMessage},
    transaction::{
        ShardTransaction, SignedTransaction, TransactionData, TransactionExpiration,
        TransactionKind,
    },
};

use serde::{Deserialize, Serialize};
use soma_network::multiaddr::Multiaddr;

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

// #[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
// pub struct ShardAuthToken {
//     pub proof: FinalityProof,
//     pub metadata_commitment: MetadataCommitment,
//     pub block_entropy: BlockEntropy,
//     pub entropy_proof: BlockEntropyProof,
// }

// impl ShardAuthToken {
//     pub fn metadata_commitment(&self) -> MetadataCommitment {
//         self.metadata_commitment.clone()
//     }
//     pub fn epoch(&self) -> Epoch {
//         self.proof.epoch()
//     }

//     pub fn new_for_test(peer: PeerPublicKey, address: Multiaddr) -> Self {
//         fn mock_tx() -> SignedTransaction {
//             let sig = ProtocolKeySignature::from_bytes(&[1u8; 64]).unwrap();
//             let tx = ShardTransaction::new(Digest::new_from_bytes(b"test"), 100);
//             let tx_kind = TransactionKind::ShardTransaction(tx);
//             SignedTransaction {
//                 scoped_message: ScopedMessage::new(
//                     Scope::TransactionData,
//                     TransactionData::new_v1(
//                         tx_kind,
//                         Address::default(),
//                         TransactionExpiration::None,
//                     ),
//                 ),
//                 tx_signatures: vec![sig],
//             }
//         }
//         let epoch = 0_u64;
//         let stakes = vec![1u64; 4];
//         let (authority_committee, authority_keypairs) =
//             AuthorityCommittee::local_test_committee(0, stakes);
//         let metadata = MetadataV1::new(
//             Default::default(), // default checksum
//             1024,               // size in bytes
//         );

//         let downloadable_metadata =
//             DownloadableMetadata::V1(DownloadableMetadataV1::new(peer, address, metadata));
//         let metadata_commitment = MetadataCommitment::new(downloadable_metadata, [0u8; 32]);

//         // Create a test claim
//         let claim = BlockClaim::new(epoch, BlockRef::default(), mock_tx());

//         // Sign with 3 out of 4 authorities (meeting 2f+1 threshold)
//         let message = bcs::to_bytes(&claim).unwrap();
//         let signatures: Vec<AuthoritySignature> = authority_keypairs[..3]
//             .iter()
//             .map(|(_, authority_kp)| authority_kp.sign(&message))
//             .collect();

//         // Create authority bitset for the signing authorities
//         let authorities =
//             AuthorityBitSet::new(&(0..3).map(AuthorityIndex::new_for_test).collect::<Vec<_>>());

//         // Create and verify the proof
//         let proof = FinalityProof::new(
//             claim,
//             authorities,
//             AuthorityAggregateSignature::new(&signatures).unwrap(),
//         );

//         Self {
//             proof,
//             metadata_commitment,
//             block_entropy: BlockEntropy::default(),
//             entropy_proof: BlockEntropyProof::default(),
//         }
//     }
// }
