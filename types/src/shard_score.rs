use enum_dispatch::enum_dispatch;
use evaluation::{EvaluationScore, ProbeSet, SummaryEmbedding};
use fastcrypto::bls12381::min_sig;
use serde::{Deserialize, Serialize};
use shared::{
    crypto::keys::EncoderPublicKey,
    digest::Digest,
    error::SharedResult,
    scope::Scope,
    shard::{Shard, ShardAuthToken},
    signed::Signed,
};

/// Shard commit is the wrapper that contains the versioned shard commit. It
/// represents the encoders response to a batch of data
#[derive(Clone, Debug, Deserialize, Serialize)]
#[enum_dispatch(ShardScoreAPI)]
pub enum ShardScore {
    V1(ShardScoreV1),
}

/// `ShardScoreAPI` is the trait that every shard commit version must implement
#[enum_dispatch]
pub trait ShardScoreAPI {
    fn auth_token(&self) -> &ShardAuthToken;
    fn author(&self) -> &EncoderPublicKey;
    fn signed_score_set(&self) -> Signed<ScoreSet, min_sig::BLS12381Signature>;
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ShardScoreV1 {
    auth_token: ShardAuthToken,
    author: EncoderPublicKey,
    signed_score_set: Signed<ScoreSet, min_sig::BLS12381Signature>,
}

impl ShardScoreV1 {
    pub const fn new(
        auth_token: ShardAuthToken,
        author: EncoderPublicKey,
        signed_score_set: Signed<ScoreSet, min_sig::BLS12381Signature>,
    ) -> Self {
        Self {
            auth_token,
            author,
            signed_score_set,
        }
    }
}

impl ShardScoreAPI for ShardScoreV1 {
    fn auth_token(&self) -> &ShardAuthToken {
        &self.auth_token
    }
    fn author(&self) -> &EncoderPublicKey {
        &self.author
    }
    fn signed_score_set(&self) -> Signed<ScoreSet, min_sig::BLS12381Signature> {
        self.signed_score_set.clone()
    }
}

#[enum_dispatch]
pub trait ScoreSetAPI {
    fn winner(&self) -> &EncoderPublicKey;
    fn score(&self) -> &EvaluationScore;
    fn summary_embedding(&self) -> &SummaryEmbedding;
    fn probe_set(&self) -> &ProbeSet;
    fn shard_digest(&self) -> Digest<Shard>;
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[enum_dispatch(ScoreSetAPI)]
pub enum ScoreSet {
    V1(ScoreSetV1),
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ScoreSetV1 {
    winner: EncoderPublicKey,
    score: EvaluationScore,
    summary_embedding: SummaryEmbedding,
    probe_set: ProbeSet,
    shard_digest: Digest<Shard>,
}
impl ScoreSetV1 {
    pub fn new(
        winner: EncoderPublicKey,
        score: EvaluationScore,
        summary_embedding: SummaryEmbedding,
        probe_set: ProbeSet,
        shard_digest: Digest<Shard>,
    ) -> Self {
        Self {
            winner,
            score,
            summary_embedding,
            probe_set,
            shard_digest,
        }
    }
}

impl ScoreSetAPI for ScoreSetV1 {
    fn winner(&self) -> &EncoderPublicKey {
        &self.winner
    }
    fn score(&self) -> &EvaluationScore {
        &self.score
    }
    fn summary_embedding(&self) -> &SummaryEmbedding {
        &self.summary_embedding
    }
    fn probe_set(&self) -> &ProbeSet {
        &self.probe_set
    }
    fn shard_digest(&self) -> Digest<Shard> {
        self.shard_digest.clone()
    }
}

pub fn verify_signed_score(
    signed_scores: &Signed<ShardScore, min_sig::BLS12381Signature>,
    shard: &Shard,
) -> SharedResult<()> {
    // if !shard.contains(&signed_scores.evaluator()) {
    //     return Err(SharedError::ValidationError(
    //         "evaluator is not in the shard".to_string(),
    //     ));
    // }

    // if signed_scores.unique_scores() != shard.size() {
    //     return Err(SharedError::ValidationError(
    //         "unique scores does not match shard size".to_string(),
    //     ));
    // }
    // for encoder in signed_scores.encoders() {
    //     if !shard.contains(&encoder) {
    //         return Err(SharedError::ValidationError(
    //             "scored encoder is not in shard".to_string(),
    //         ));
    //     }
    // }

    let _ = signed_scores.verify_signature(Scope::Score, signed_scores.author().inner())?;

    Ok(())
}
