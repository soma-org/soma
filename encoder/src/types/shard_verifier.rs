use quick_cache::sync::Cache;
use serde::{Deserialize, Serialize};
use shared::{
    digest::Digest,
    entropy::{BlockEntropyOutput, BlockEntropyProof, EntropyVDF},
    finality_proof::FinalityProof,
    metadata::MetadataCommitment,
};
use tokio_util::sync::CancellationToken;

use crate::{
    actors::{workers::vdf::VDFProcessor, ActorHandle},
    error::{ShardError, ShardResult},
};

use super::{
    encoder_committee::{EncoderIndex, Epoch},
    encoder_context::EncoderContext,
    shard::{Shard, ShardEntropy},
};

#[derive(Clone)]
pub(crate) enum VerificationStatus {
    Valid(Shard),
    Invalid,
}
pub(crate) struct ShardVerifier {
    cache: Cache<Digest<ShardAuthToken>, VerificationStatus>,
}

impl ShardVerifier {
    pub(crate) fn new(cache: Cache<Digest<ShardAuthToken>, VerificationStatus>) -> Self {
        Self { cache }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct ShardAuthToken {
    proof: FinalityProof,
    metadata_commitment: MetadataCommitment,
    block_entropy: BlockEntropyOutput,
    entropy_proof: BlockEntropyProof,
}

impl ShardAuthToken {
    pub fn epoch(&self) -> Epoch {
        self.proof.epoch()
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct Route {
    // the selected encoder to commit on their behalf
    destination: EncoderIndex,
    // digest is used to stop replay attacks
    auth_token_digest: Digest<ShardAuthToken>,
}

impl Route {
    pub fn destination(&self) -> EncoderIndex {
        self.destination
    }
    pub fn auth_token_digest(&self) -> Digest<ShardAuthToken> {
        self.auth_token_digest
    }
}

impl ShardVerifier {
    pub(crate) async fn verify(
        &self,
        context: &EncoderContext,
        vdf: &ActorHandle<VDFProcessor<EntropyVDF>>,
        token: &ShardAuthToken,
    ) -> ShardResult<(Digest<ShardAuthToken>, Shard)> {
        let digest =
            Digest::new(token).map_err(|e| ShardError::InvalidShardToken(e.to_string()))?;
        // check cache
        if let Some(status) = self.cache.get(&digest) {
            return match status {
                VerificationStatus::Valid(shard) => Ok((digest, shard)),
                VerificationStatus::Invalid => Err(ShardError::InvalidShardToken(
                    "invalid shard token".to_string(),
                )),
            };
        }
        let valid_shard_result: ShardResult<Shard> = {
            // check that the finality proof is valid against the epoch's authorities
            token
                .proof
                .verify(&context.authority_committee)
                .map_err(|e| ShardError::InvalidShardToken(e.to_string()))?;

            // check that the vdf entropy (block entropy) passes verification with the provided proof
            vdf.process(
                (
                    token.proof.epoch(),
                    token.proof.block_ref(),
                    token.block_entropy.clone(),
                    token.entropy_proof.clone(),
                ),
                CancellationToken::new(),
            )
            .await?;

            // TODO: need to actually check the transaction itself
            // specifically the metadata_commitment digest matches what is provided
            let _tx = token.proof.transaction();
            // TODO: check metadata_commitment digest matches
            // TODO: check that tx value matches metadata provided size

            // create the shard entropy seed using the VDF (good source of randomness) + the metadata/nonce
            let shard_entropy = Digest::new(&ShardEntropy::new(
                token.metadata_commitment.clone(),
                token.block_entropy.clone(),
            ))
            .map_err(|e| ShardError::InvalidShardToken(e.to_string()))?;

            let shard = context.encoder_committee.sample_shard(shard_entropy)?;

            if !shard.contains(&context.own_encoder_index) {
                return Err(ShardError::InvalidShardToken(
                    "encoder is not contained in shard".to_string(),
                ));
            }
            Ok(shard)
        };
        match &valid_shard_result {
            Ok(shard) => {
                self.cache
                    .insert(digest, VerificationStatus::Valid(shard.clone()));
                return Ok((digest, shard.to_owned()));
            }
            // unwrapping manually here so we can cache the invalid verification
            Err(_) => self.cache.insert(digest, VerificationStatus::Invalid),
        }
        Err(ShardError::InvalidShardToken(
            "invalid shard token".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::{EncoderContext, EncoderIndex, ShardAuthToken, ShardVerifier, VerificationStatus};
    use crate::{
        actors::{workers::vdf::VDFProcessor, ActorHandle, ActorManager},
        error::ShardResult,
        types::encoder_committee::{EncoderCommittee, Epoch},
    };
    use quick_cache::sync::Cache;
    use shared::{
        authority_committee::{AuthorityBitSet, AuthorityCommittee, AuthorityIndex},
        block::BlockRef,
        crypto::{
            address::Address,
            keys::{AuthorityAggregateSignature, AuthoritySignature, ProtocolKeySignature},
        },
        digest::Digest,
        entropy::{BlockEntropyOutput, BlockEntropyProof, EntropyAPI, EntropyVDF},
        finality_proof::{BlockClaim, FinalityProof},
        metadata::{Metadata, MetadataCommitment},
        network_committee::{NetworkCommittee, NetworkIdentity, NetworkingIndex},
        probe::ProbeMetadata,
        scope::{Scope, ScopedMessage},
        transaction::{
            ShardTransaction, SignedTransaction, TransactionData, TransactionExpiration,
            TransactionKind,
        },
    };
    use tokio_util::sync::CancellationToken;

    const TEST_ENTROPY_ITERATIONS: u64 = 1;
    const TEST_CACHE_CAPACITY: usize = 100;

    fn mock_tx() -> SignedTransaction {
        let sig = ProtocolKeySignature::from_bytes(&[1u8; 64]).unwrap();
        let tx = ShardTransaction::new(Digest::new_from_bytes(b"test"), 100);
        let tx_kind = TransactionKind::ShardTransaction(tx);
        SignedTransaction {
            scoped_message: ScopedMessage::new(
                Scope::TransactionData,
                TransactionData::new_v1(tx_kind, Address::default(), TransactionExpiration::None),
            ),
            tx_signatures: vec![sig],
        }
    }
    const STARTING_PORT: u16 = 8000;
    const EPOCH: Epoch = 0;

    async fn setup_test_environment() -> (
        ShardVerifier,
        ShardAuthToken,
        ActorHandle<VDFProcessor<EntropyVDF>>,
        EncoderContext,
    ) {
        // Set up VDF processor
        let vdf = EntropyVDF::new(TEST_ENTROPY_ITERATIONS);
        let vdf_processor = VDFProcessor::new(vdf, 1);
        let vdf_handle = ActorManager::new(1, vdf_processor).handle();

        // Create encoder committee and context
        let encoder_indices: Vec<EncoderIndex> = (0..4).map(EncoderIndex::new_for_test).collect();
        let encoder_details = vec![(1u16, ProbeMetadata::new_for_test(&[0u8; 8])); 4];
        let (encoder_committee, encoder_keypairs) = EncoderCommittee::local_test_committee(
            EPOCH,
            encoder_details,
            4,
            3,
            4,
            3,
            STARTING_PORT,
        );
        // Create committee with 4 authorities, each with stake 1
        let stakes = vec![1u64; 4];
        let (authority_committee, authority_keypairs) =
            AuthorityCommittee::local_test_committee(EPOCH, stakes, STARTING_PORT);

        // Create encoder context (using index 0 as our own encoder)
        let encoder_context = EncoderContext::new(
            authority_committee,
            NetworkCommittee::default(),
            NetworkingIndex::new_for_test(0),
            encoder_committee,
            EncoderIndex::new_for_test(0),
        );

        // Create cache
        let cache: Cache<Digest<ShardAuthToken>, VerificationStatus> =
            Cache::new(TEST_CACHE_CAPACITY);

        // Create ShardVerifier
        let verifier = ShardVerifier::new(cache);

        // Create a test ShardAuthToken
        let (block_entropy, entropy_proof) = EntropyVDF::new(TEST_ENTROPY_ITERATIONS)
            .get_entropy(EPOCH, Default::default())
            .unwrap();

        // Create a test metadata commitment
        let metadata = Metadata::new_v1(
            None,               // no compression
            None,               // no encryption
            Default::default(), // default checksum
            vec![1, 1],         // simple shape
            1024,               // size in bytes
        );
        let metadata_commitment = MetadataCommitment::new(metadata, [0u8; 32]);

        // Create a test claim
        let claim = BlockClaim::new(EPOCH, BlockRef::default(), mock_tx());

        // Sign with 3 out of 4 authorities (meeting 2f+1 threshold)
        let message = bcs::to_bytes(&claim).unwrap();
        let signatures: Vec<AuthoritySignature> = authority_keypairs[..3]
            .iter()
            .map(|(_, _, authority_kp)| authority_kp.sign(&message))
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

        let auth_token = ShardAuthToken {
            proof,
            metadata_commitment,
            block_entropy,
            entropy_proof,
        };

        (verifier, auth_token, vdf_handle, encoder_context)
    }

    #[tokio::test]
    async fn test_shard_verifier_e2e() -> ShardResult<()> {
        let (verifier, auth_token, vdf_handle, encoder_context) = setup_test_environment().await;

        // Verify the token
        let (digest, shard) = verifier
            .verify(&encoder_context, &vdf_handle, &auth_token)
            .await?;

        // Verify the same token again (should hit cache)
        let (cached_digest, cached_shard) = verifier
            .verify(&encoder_context, &vdf_handle, &auth_token)
            .await?;

        // Check that cached results match
        assert_eq!(digest, cached_digest);
        assert_eq!(shard, cached_shard);

        // Verify that our encoder is in the shard
        assert!(shard.contains(&encoder_context.own_encoder_index));

        Ok(())
    }
}
