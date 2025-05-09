use quick_cache::sync::Cache;
use serde::{Deserialize, Serialize};
use shared::{
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
use tokio_util::sync::CancellationToken;

use crate::{
    actors::{workers::vdf::VDFProcessor, ActorHandle},
    error::{ShardError, ShardResult},
};

use super::{
    context::Context,
    encoder_committee::Epoch,
    shard::{Shard, ShardEntropy},
};

/// Tracks the cached verification status for a given auth token
#[derive(Clone)]
enum VerificationStatus {
    /// valid, containing shard
    Valid(Shard),
    /// invalid
    Invalid,
}
/// Verifies shard auth tokens and returns a shard
pub(crate) struct ShardVerifier {
    /// caches verification status for a given auth token digest
    cache: Cache<Digest<ShardAuthToken>, VerificationStatus>,
    cancellation_cache: Cache<Digest<Shard>, CancellationToken>,
    /// holds the actor wrapped VDF
    vdf: ActorHandle<VDFProcessor<EntropyVDF>>,
    own_key: EncoderPublicKey,
}

impl ShardVerifier {
    pub(crate) fn new(
        capacity: usize,
        vdf: ActorHandle<VDFProcessor<EntropyVDF>>,
        own_key: EncoderPublicKey,
    ) -> Self {
        let cache: Cache<Digest<ShardAuthToken>, VerificationStatus> = Cache::new(capacity);
        Self {
            cache,
            cancellation_cache: Cache::new(capacity),
            vdf,
            own_key,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct ShardAuthToken {
    pub proof: FinalityProof,
    pub metadata_commitment: MetadataCommitment,
    pub block_entropy: BlockEntropy,
    pub entropy_proof: BlockEntropyProof,
}

impl ShardAuthToken {
    pub fn metadata_commitment(&self) -> MetadataCommitment {
        self.metadata_commitment.clone()
    }
    pub fn epoch(&self) -> Epoch {
        self.proof.epoch()
    }

    pub fn new_for_test() -> Self {
        fn mock_tx() -> SignedTransaction {
            let sig = ProtocolKeySignature::from_bytes(&[1u8; 64]).unwrap();
            let tx = ShardTransaction::new(Digest::new_from_bytes(b"test"), 100);
            let tx_kind = TransactionKind::ShardTransaction(tx);
            SignedTransaction {
                scoped_message: ScopedMessage::new(
                    Scope::TransactionData,
                    TransactionData::new_v1(
                        tx_kind,
                        Address::default(),
                        TransactionExpiration::None,
                    ),
                ),
                tx_signatures: vec![sig],
            }
        }
        let epoch = 0_u64;
        let stakes = vec![1u64; 4];
        let (authority_committee, authority_keypairs) =
            AuthorityCommittee::local_test_committee(0, stakes);
        let metadata = Metadata::new_v1(
            None,               // no compression
            None,               // no encryption
            Default::default(), // default checksum
            1024,               // size in bytes
        );
        let metadata_commitment = MetadataCommitment::new(metadata, [0u8; 32]);

        // Create a test claim
        let claim = BlockClaim::new(epoch, BlockRef::default(), mock_tx());

        // Sign with 3 out of 4 authorities (meeting 2f+1 threshold)
        let message = bcs::to_bytes(&claim).unwrap();
        let signatures: Vec<AuthoritySignature> = authority_keypairs[..3]
            .iter()
            .map(|(_, authority_kp)| authority_kp.sign(&message))
            .collect();

        // Create authority bitset for the signing authorities
        let authorities =
            AuthorityBitSet::new(&(0..3).map(AuthorityIndex::new_for_test).collect::<Vec<_>>());

        // Create and verify the proof
        let proof = FinalityProof::new(
            claim,
            authorities,
            AuthorityAggregateSignature::new(&signatures).unwrap(),
        );

        Self {
            proof,
            metadata_commitment,
            block_entropy: BlockEntropy::default(),
            entropy_proof: BlockEntropyProof::default(),
        }
    }
}

impl ShardVerifier {
    pub(crate) async fn verify(
        &self,
        context: &Context,
        token: &ShardAuthToken,
    ) -> ShardResult<(Shard, CancellationToken)> {
        tracing::info!(
            "Starting ShardVerifier::verify with token epoch: {}",
            token.epoch()
        );

        let digest = match Digest::new(token) {
            Ok(d) => {
                tracing::debug!("Successfully created digest for token");
                d
            }
            Err(e) => {
                tracing::error!("Failed to create digest for token: {:?}", e);
                return Err(ShardError::InvalidShardToken(e.to_string()));
            }
        };

        // Check cache
        if let Some(status) = self.cache.get(&digest) {
            tracing::debug!("Found cached verification status");
            return match status {
                VerificationStatus::Valid(shard) => {
                    tracing::debug!("Cache hit: valid shard");
                    let shard_digest = shard.digest()?;
                    let cancellation = self
                        .cancellation_cache
                        .get_or_insert_with(&shard_digest, || Ok(CancellationToken::new()))
                        .map_err(|e: ShardError| {
                            // unreachable, cancellation token new does not fail
                            ShardError::InvalidShardToken(
                                "issue with cancellation token".to_string(),
                            )
                        })?;

                    Ok((shard, cancellation))
                }
                VerificationStatus::Invalid => {
                    tracing::debug!("Cache hit: invalid shard token");
                    Err(ShardError::InvalidShardToken(
                        "invalid shard auth token".to_string(),
                    ))
                }
            };
        }

        tracing::debug!("Accessing inner context from Context");
        let inner_context = context.inner();

        tracing::debug!("Getting committees for epoch: {}", token.epoch());
        let committees = match inner_context.committees(token.epoch()) {
            Ok(c) => {
                tracing::debug!("Successfully retrieved committees");
                c
            }
            Err(e) => {
                tracing::error!("Failed to get committees: {:?}", e);
                return Err(e);
            }
        };

        // Debug finality proof
        tracing::debug!("Verifying finality proof against authority committee");
        if let Err(e) = token.proof.verify(&committees.authority_committee) {
            tracing::error!("Finality proof verification failed: {:?}", e);
            self.cache.insert(digest, VerificationStatus::Invalid);
            return Err(ShardError::InvalidShardToken(e.to_string()));
        }

        // Debug VDF
        tracing::debug!(
            "Starting VDF verification with epoch: {}, iterations: {}",
            token.proof.epoch(),
            committees.vdf_iterations
        );
        let vdf_params = (
            token.proof.epoch(),
            token.proof.block_ref(),
            token.block_entropy.clone(),
            token.entropy_proof.clone(),
            committees.vdf_iterations,
        );
        tracing::debug!("VDF params created");

        let vdf_result = self.vdf.process(vdf_params, CancellationToken::new()).await;

        if let Err(e) = vdf_result {
            tracing::error!("VDF verification failed: {:?}", e);
            self.cache.insert(digest, VerificationStatus::Invalid);
            return Err(e);
        }

        tracing::debug!("VDF verification succeeded");

        // Debug shard entropy
        tracing::debug!("Creating ShardEntropy with metadata commitment and block entropy");
        let shard_entropy_input = ShardEntropy::new(
            token.metadata_commitment.clone(),
            token.block_entropy.clone(),
        );

        tracing::debug!("Creating digest from ShardEntropy");
        let shard_entropy = match Digest::new(&shard_entropy_input) {
            Ok(entropy) => {
                tracing::debug!("Successfully created shard entropy digest");
                entropy
            }
            Err(e) => {
                tracing::error!("Failed to create shard entropy digest: {:?}", e);
                self.cache.insert(digest, VerificationStatus::Invalid);
                return Err(ShardError::InvalidShardToken(e.to_string()));
            }
        };

        tracing::debug!("Sampling shard from encoder committee");
        let shard = match committees.encoder_committee.sample_shard(shard_entropy) {
            Ok(s) => {
                tracing::debug!(
                    "Successfully sampled shard with {} encoders: {:?}",
                    s.encoders().len(),
                    s.encoders()
                );
                s
            }
            Err(e) => {
                tracing::error!("Failed to sample shard: {:?}", e);
                self.cache.insert(digest, VerificationStatus::Invalid);
                return Err(e);
            }
        };

        if !shard.contains(&self.own_key) {
            return Err(ShardError::InvalidShardMember);
        }

        let shard_digest = shard.digest()?;

        tracing::debug!("Caching successful verification result");
        self.cache
            .insert(digest, VerificationStatus::Valid(shard.clone()));
        tracing::info!("ShardVerifier::verify completed successfully");
        let cancellation = self
            .cancellation_cache
            .get_or_insert_with(&shard_digest, || Ok(CancellationToken::new()))
            .map_err(|_: ShardError| {
                // unreachable, cancellation token new does not fail
                ShardError::InvalidShardToken("issue with cancellation token".to_string())
            })?;

        Ok((shard, cancellation))
    }
}

// #[cfg(test)]
// mod tests {
//     use super::{Context, ShardAuthToken, ShardVerifier, VerificationStatus};
//     use crate::{
//         actors::{workers::vdf::VDFProcessor, ActorHandle, ActorManager},
//         error::ShardResult,
//         types::{
//             context::InnerContext,
//             encoder_committee::{EncoderCommittee, Epoch},
//         },
//     };
//     use quick_cache::sync::Cache;
//     use shared::{
//         authority_committee::{AuthorityBitSet, AuthorityCommittee, AuthorityIndex},
//         block::BlockRef,
//         crypto::{
//             address::Address,
//             keys::{AuthorityAggregateSignature, AuthoritySignature, ProtocolKeySignature},
//         },
//         digest::Digest,
//         entropy::{EntropyAPI, EntropyVDF},
//         finality_proof::{BlockClaim, FinalityProof},
//         metadata::{Metadata, MetadataCommitment},
//         scope::{Scope, ScopedMessage},
//         transaction::{
//             ShardTransaction, SignedTransaction, TransactionData, TransactionExpiration,
//             TransactionKind,
//         },
//     };

//     const TEST_ENTROPY_ITERATIONS: u64 = 1;
//     const TEST_CACHE_CAPACITY: usize = 100;

//     fn mock_tx() -> SignedTransaction {
//         let sig = ProtocolKeySignature::from_bytes(&[1u8; 64]).unwrap();
//         let tx = ShardTransaction::new(Digest::new_from_bytes(b"test"), 100);
//         let tx_kind = TransactionKind::ShardTransaction(tx);
//         SignedTransaction {
//             scoped_message: ScopedMessage::new(
//                 Scope::TransactionData,
//                 TransactionData::new_v1(tx_kind, Address::default(), TransactionExpiration::None),
//             ),
//             tx_signatures: vec![sig],
//         }
//     }
//     const STARTING_PORT: u16 = 8000;
//     const EPOCH: Epoch = 0;

//     async fn setup_test_environment() -> (
//         ShardVerifier,
//         ShardAuthToken,
//         ActorHandle<VDFProcessor<EntropyVDF>>,
//         Context,
//     ) {
//         // Set up VDF processor
//         let vdf = EntropyVDF::new(TEST_ENTROPY_ITERATIONS);
//         let vdf_processor = VDFProcessor::new(vdf, 1);
//         let vdf_handle = ActorManager::new(1, vdf_processor).handle();

//         // Create encoder committee and context
//         let encoder_indices: Vec<EncoderIndex> = (0..4).map(EncoderIndex::new_for_test).collect();
//         let encoder_details = vec![(1u16, ProbeMetadata::new_for_test(&[0u8; 8])); 4];
//         let (encoder_committee, encoder_keypairs) = EncoderCommittee::local_test_committee(
//             EPOCH,
//             encoder_details,
//             4,
//             3,
//             4,
//             3,
//             STARTING_PORT,
//         );
//         // Create committee with 4 authorities, each with stake 1
//         let stakes = vec![1u64; 4];
//         let (authority_committee, authority_keypairs) =
//             AuthorityCommittee::local_test_committee(EPOCH, stakes, STARTING_PORT);

//         // Create encoder context (using index 0 as our own encoder)
//         let encoder_context = InnerContext::new(
//             authority_committee,
//             NetworkCommittee::default(),
//             NetworkingIndex::new_for_test(0),
//             encoder_committee,
//             EncoderIndex::new_for_test(0),
//         );

//         // Create cache
//         let cache: Cache<Digest<ShardAuthToken>, VerificationStatus> =
//             Cache::new(TEST_CACHE_CAPACITY);

//         // Create ShardVerifier
//         let verifier = ShardVerifier::new(cache);

//         // Create a test ShardAuthToken
//         let (block_entropy, entropy_proof) = EntropyVDF::new(TEST_ENTROPY_ITERATIONS)
//             .get_entropy(EPOCH, Default::default())
//             .unwrap();

//         // Create a test metadata commitment
//         let metadata = Metadata::new_v1(
//             None,               // no compression
//             None,               // no encryption
//             Default::default(), // default checksum
//             1024,               // size in bytes
//         );
//         let metadata_commitment = MetadataCommitment::new(metadata, [0u8; 32]);

//         // Create a test claim
//         let claim = BlockClaim::new(EPOCH, BlockRef::default(), mock_tx());

//         // Sign with 3 out of 4 authorities (meeting 2f+1 threshold)
//         let message = bcs::to_bytes(&claim).unwrap();
//         let signatures: Vec<AuthoritySignature> = authority_keypairs[..3]
//             .iter()
//             .map(|(_, _, authority_kp)| authority_kp.sign(&message))
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

//         let auth_token = ShardAuthToken {
//             proof,
//             metadata_commitment,
//             block_entropy,
//             entropy_proof,
//         };

//         (verifier, auth_token, vdf_handle, encoder_context)
//     }

//     #[tokio::test]
//     async fn test_shard_verifier_e2e() -> ShardResult<()> {
//         let (verifier, auth_token, vdf_handle, encoder_context) = setup_test_environment().await;

//         // Verify the token
//         let (digest, shard) = verifier
//             .verify(&encoder_context, &vdf_handle, &auth_token)
//             .await?;

//         // Verify the same token again (should hit cache)
//         let (cached_digest, cached_shard) = verifier
//             .verify(&encoder_context, &vdf_handle, &auth_token)
//             .await?;

//         // Check that cached results match
//         assert_eq!(digest, cached_digest);
//         assert_eq!(shard, cached_shard);

//         // Verify that our encoder is in the shard
//         assert!(shard.contains(&encoder_context.own_encoder_index));

//         Ok(())
//     }
// }
