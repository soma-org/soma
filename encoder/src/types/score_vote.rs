use enum_dispatch::enum_dispatch;
use fastcrypto::bls12381::min_sig;
use serde::{Deserialize, Serialize};
use shared::{
    crypto::keys::EncoderPublicKey, error::SharedResult, scope::Scope, shard::Shard, signed::Signed,
};
use types::{score_set::ScoreSet, shard::ShardAuthToken};

/// Shard commit is the wrapper that contains the versioned shard commit. It
/// represents the encoders response to a batch of data
#[derive(Clone, Debug, Deserialize, Serialize)]
#[enum_dispatch(ScoreVoteAPI)]
pub enum ScoreVote {
    V1(ScoreVoteV1),
}

/// `ScoreVoteAPI` is the trait that every shard commit version must implement
#[enum_dispatch]
pub trait ScoreVoteAPI {
    fn auth_token(&self) -> &ShardAuthToken;
    fn author(&self) -> &EncoderPublicKey;
    fn signed_score_set(&self) -> Signed<ScoreSet, min_sig::BLS12381Signature>;
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ScoreVoteV1 {
    auth_token: ShardAuthToken,
    author: EncoderPublicKey,
    signed_score_set: Signed<ScoreSet, min_sig::BLS12381Signature>,
}

impl ScoreVoteV1 {
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

impl ScoreVoteAPI for ScoreVoteV1 {
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

pub fn verify_signed_score_vote(
    signed_scores: &Signed<ScoreVote, min_sig::BLS12381Signature>,
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
