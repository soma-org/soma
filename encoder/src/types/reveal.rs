use enum_dispatch::enum_dispatch;
use evaluation::{EvaluationScore, EvaluationScoreAPI, ProbeSet, SummaryEmbedding};
use fastcrypto::bls12381::min_sig;
use serde::{Deserialize, Serialize};
use shared::{
    crypto::keys::EncoderPublicKey,
    error::SharedResult,
    metadata::{verify_metadata, DownloadableMetadata, DownloadableMetadataAPI},
    scope::Scope,
    signed::Signed,
};

use shared::shard::Shard;
use types::shard::ShardAuthToken;

/// Shard commit is the wrapper that contains the versioned shard commit. It
/// represents the encoders response to a batch of data
#[derive(Clone, Debug, Deserialize, Serialize)]
#[enum_dispatch(RevealAPI)]
pub enum Reveal {
    V1(RevealV1),
}

/// `RevealAPI` is the trait that every shard commit version must implement
#[enum_dispatch]
pub(crate) trait RevealAPI {
    fn auth_token(&self) -> &ShardAuthToken;
    fn author(&self) -> &EncoderPublicKey;
    fn score(&self) -> &EvaluationScore;
    fn probe_set(&self) -> &ProbeSet;
    fn tensors(&self) -> &DownloadableMetadata;
    fn summary_embedding(&self) -> SummaryEmbedding;
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct RevealV1 {
    auth_token: ShardAuthToken,
    author: EncoderPublicKey,
    score: EvaluationScore,
    probe_set: ProbeSet,
    tensors: DownloadableMetadata,
    summary_embedding: SummaryEmbedding,
}

impl RevealV1 {
    pub(crate) const fn new(
        auth_token: ShardAuthToken,
        author: EncoderPublicKey,
        score: EvaluationScore,
        probe_set: ProbeSet,
        tensors: DownloadableMetadata,
        summary_embedding: SummaryEmbedding,
    ) -> Self {
        Self {
            auth_token,
            author,
            score,
            probe_set,
            tensors,
            summary_embedding,
        }
    }
}

impl RevealAPI for RevealV1 {
    fn auth_token(&self) -> &ShardAuthToken {
        &self.auth_token
    }
    fn author(&self) -> &EncoderPublicKey {
        &self.author
    }
    fn score(&self) -> &EvaluationScore {
        &self.score
    }
    fn probe_set(&self) -> &ProbeSet {
        &self.probe_set
    }
    fn tensors(&self) -> &DownloadableMetadata {
        &self.tensors
    }
    fn summary_embedding(&self) -> SummaryEmbedding {
        self.summary_embedding.clone()
    }
}

pub(crate) fn verify_signed_reveal(
    signed_reveal: &Signed<Reveal, min_sig::BLS12381Signature>,
    shard: &Shard,
) -> SharedResult<()> {
    if !shard.contains(&signed_reveal.author()) {
        return Err(shared::error::SharedError::ValidationError(
            "encoder is not in the shard".to_string(),
        ));
    }
    if signed_reveal.score().value() <= 0.0 {
        return Err(shared::error::SharedError::ValidationError(
            "score value must be greater than zero".to_string(),
        ));
    }

    // TODO: verify the probe_set's validity
    // TODO: verify the summary embedding's length

    let max_size = None;
    verify_metadata(&signed_reveal.tensors().metadata(), max_size)?;

    let _ = signed_reveal.verify_signature(Scope::Reveal, signed_reveal.author().inner())?;
    Ok(())
}
pub(crate) fn verify_reveal_score_matches(
    score: EvaluationScore,
    summary_embedding: SummaryEmbedding,
    signed_reveal: &Signed<Reveal, min_sig::BLS12381Signature>,
) -> SharedResult<()> {
    Ok(())
}
