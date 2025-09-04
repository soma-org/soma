use enum_dispatch::enum_dispatch;
use fastcrypto::bls12381::min_sig;
use serde::{Deserialize, Serialize};
use types::shard_crypto::{keys::EncoderPublicKey, scope::Scope, signed::Signed};
use types::{
    error::{SharedError, SharedResult},
    score_set::ScoreSet,
    shard::Shard,
    shard::ShardAuthToken,
};

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

pub fn verify_score_vote(
    score_vote: &ScoreVote,
    peer: &EncoderPublicKey,
    shard: &Shard,
) -> SharedResult<()> {
    if peer != score_vote.author() {
        return Err(SharedError::FailedTypeVerification(
            "sending peer must be author".to_string(),
        ));
    }

    // score set signature is valid
    //

    Ok(())
}
