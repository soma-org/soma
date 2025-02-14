use quick_cache::sync::Cache;
use serde::{Deserialize, Serialize};
use shared::{
    crypto::AesKey,
    digest::Digest,
    entropy::{BlockEntropyOutput, BlockEntropyProof},
    finality_proof::FinalityProof,
    metadata::MetadataCommitment,
};
use tokio_util::sync::CancellationToken;

use crate::{
    actors::{workers::vdf::VDFProcessor, ActorHandle},
    error::{ShardError, ShardResult},
};

use super::{
    encoder_committee::EncoderIndex, encoder_context::EncoderContext, shard::ShardEntropy,
};

#[derive(Clone)]
enum VerificationStatus {
    Valid,
    Invalid,
}
pub(crate) struct ShardVerifier {
    encoder_context: EncoderContext,
    vdf: ActorHandle<VDFProcessor>,
    cache: Cache<Digest<ShardAuthToken>, VerificationStatus>,
}

impl ShardVerifier {
    pub(crate) fn new(
        encoder_context: EncoderContext,
        vdf: ActorHandle<VDFProcessor>,
        cache: Cache<Digest<ShardAuthToken>, VerificationStatus>,
    ) -> Self {
        Self {
            encoder_context,
            vdf,
            cache,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ShardAuthToken {
    proof: FinalityProof,
    metadata_commitment: MetadataCommitment,
    block_entropy: BlockEntropyOutput,
    entropy_proof: BlockEntropyProof,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Route {
    // the inference-eligible encoder
    source: EncoderIndex,
    // the selected encoder to commit on their behalf
    destination: EncoderIndex,
    // digest is used to stop replay attacks
    auth_token_digest: Digest<ShardAuthToken>,
}

impl ShardVerifier {
    pub(crate) async fn verify(&self, token: ShardAuthToken) -> ShardResult<()> {
        let digest =
            Digest::new(&token).map_err(|e| ShardError::InvalidShardToken(e.to_string()))?;
        // check cache
        if let Some(status) = self.cache.get(&digest) {
            return match status {
                VerificationStatus::Valid => Ok(()),
                VerificationStatus::Invalid => Err(ShardError::InvalidShardToken(
                    "invalid shard token".to_string(),
                )),
            };
        }
        let verification_result: ShardResult<()> = {
            // check that the finality proof is valid against the epoch's authorities
            let _ = token
                .proof
                .verify(&self.encoder_context.authority_committee)
                .map_err(|e| ShardError::InvalidShardToken(e.to_string()))?;

            // check that the vdf entropy (block entropy) passes verification with the provided proof
            let _ = self
                .vdf
                .process(
                    (
                        token.proof.epoch(),
                        token.proof.block_ref(),
                        token.block_entropy.clone(),
                        token.entropy_proof,
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
                token.metadata_commitment,
                token.block_entropy,
            ))
            .map_err(|e| ShardError::InvalidShardToken(e.to_string()))?;

            let shard = self
                .encoder_context
                .encoder_committee
                .sample_shard(shard_entropy)?;

            if !shard.contains(&self.encoder_context.own_encoder_index) {
                return Err(ShardError::InvalidShardToken(
                    "encoder is not contained in shard".to_string(),
                ));
            }
            Ok(())
        };
        match &verification_result {
            Ok(_) => self.cache.insert(digest, VerificationStatus::Valid),
            Err(_) => self.cache.insert(digest, VerificationStatus::Invalid),
        }

        Ok(())
    }
}
