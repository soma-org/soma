use crate::committee::Committee;
use crate::encoder_committee::EncoderCommittee;
use crate::metadata::MetadataAPI;
use crate::transaction::{verify_sender_signed_data_message_signatures, TransactionKind};
use crate::{entropy::SimpleVDF, shard::ShardAuthToken};
use crate::{
    error::{ShardError, ShardResult},
    shard::{Shard, ShardEntropy},
    shard_crypto::digest::Digest,
};
use quick_cache::sync::Cache;
use tokio_util::sync::CancellationToken;

/// Tracks the cached verification status for a given auth token
#[derive(Clone)]
enum VerificationStatus {
    /// valid, containing shard
    Valid(Shard),
    /// invalid
    Invalid,
}
/// Verifies shard auth tokens and returns a shard
pub struct ShardVerifier {
    cache: Cache<Digest<ShardAuthToken>, VerificationStatus>,
    cancellation_cache: Cache<Digest<Shard>, CancellationToken>,
}

impl ShardVerifier {
    pub fn new(capacity: usize) -> Self {
        let cache: Cache<Digest<ShardAuthToken>, VerificationStatus> = Cache::new(capacity);
        Self {
            cache,
            cancellation_cache: Cache::new(capacity),
        }
    }
}

impl ShardVerifier {
    pub fn verify(
        &self,
        authority_committee: Committee,
        encoder_committee: EncoderCommittee,
        vdf_iterations: u64,
        token: &ShardAuthToken,
    ) -> ShardResult<(Shard, CancellationToken)> {
        let auth_token_digest = Digest::new(token).map_err(ShardError::DigestFailure)?;
        if let Some(status) = self.cache.get(&auth_token_digest) {
            return match status {
                VerificationStatus::Valid(shard) => {
                    let shard_digest = shard.digest()?;
                    let cancellation = self
                        .cancellation_cache
                        .get_or_insert_with(&shard_digest, || Ok(CancellationToken::new()))
                        .map_err(|e: ShardError| {
                            ShardError::InvalidShardToken(
                                "issue with getting existing cancellation token".to_string(),
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

        let result = self.verify_inner(
            &authority_committee,
            &encoder_committee,
            vdf_iterations,
            token,
        );

        match result {
            Ok(shard) => {
                let shard_digest = shard.digest()?;
                self.cache
                    .insert(auth_token_digest, VerificationStatus::Valid(shard.clone()));
                let cancellation = self
                    .cancellation_cache
                    .get_or_insert_with(&shard_digest, || Ok(CancellationToken::new()))
                    .map_err(|_: ShardError| {
                        // unreachable, cancellation token new does not fail
                        ShardError::InvalidShardToken("issue with cancellation token".to_string())
                    })?;

                Ok((shard, cancellation))
            }
            Err(e) => {
                self.cache
                    .insert(auth_token_digest, VerificationStatus::Invalid);
                Err(e)
            }
        }
    }

    fn verify_inner(
        &self,
        authority_committee: &Committee,
        encoder_committee: &EncoderCommittee,
        vdf_iterations: u64,
        token: &ShardAuthToken,
    ) -> ShardResult<Shard> {
        // 1. Verify the finality proof (checkpoint certification, inclusion, effects)
        token
            .finality_proof
            .verify(authority_committee)
            .map_err(|e| ShardError::InvalidShardToken(format!("finality proof invalid: {}", e)))?;

        // 2. Verify the transaction is an EmbedData transaction with matching metadata
        let tx_kind = token.finality_proof.transaction.transaction_data().kind();

        let TransactionKind::EmbedData {
            download_metadata, ..
        } = tx_kind
        else {
            return Err(ShardError::InvalidShardToken(
                "transaction is not EmbedData type".to_string(),
            ));
        };

        // 3. Verify VDF entropy was correctly computed from checkpoint digest
        let vdf = SimpleVDF::new(vdf_iterations);
        let checkpoint_digest = token.finality_proof.checkpoint_digest();

        vdf.verify_entropy(
            checkpoint_digest,
            &token.checkpoint_entropy,
            &token.checkpoint_entropy_proof,
        )
        .map_err(|e| ShardError::InvalidShardToken(format!("VDF verification failed: {}", e)))?;

        // 4. Compute shard selection using the verified entropy
        let shard_entropy = ShardEntropy::new(
            download_metadata.metadata().clone(),
            token.checkpoint_entropy.clone(),
        );
        let shard_seed = Digest::new(&shard_entropy).map_err(|e| {
            ShardError::InvalidShardToken(format!("failed to compute shard seed: {}", e))
        })?;

        let shard = encoder_committee.sample_shard(shard_seed)?;

        Ok(shard)
    }
}

// TODO: write shard verifier tests
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
