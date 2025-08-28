use enum_dispatch::enum_dispatch;
use fastcrypto::bls12381::min_sig;
use serde::{Deserialize, Serialize};
use shared::{
    crypto::keys::EncoderPublicKey, digest::Digest, error::SharedResult, scope::Scope,
    signed::Signed,
};

use super::shard_commit::ShardCommit;
use shared::shard::Shard;
use types::shard::ShardAuthToken;
/// Reject votes are explicit. The rest of encoders in a shard receive implicit accept votes.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[enum_dispatch(ShardCommitVotesAPI)]
pub enum ShardCommitVotes {
    V1(ShardCommitVotesV1),
}

#[enum_dispatch]
pub trait ShardCommitVotesAPI {
    fn voter(&self) -> &EncoderPublicKey;
    fn auth_token(&self) -> &ShardAuthToken;
    fn accepts(
        &self,
    ) -> &[(
        EncoderPublicKey,
        Digest<Signed<ShardCommit, min_sig::BLS12381Signature>>,
    )];
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct ShardCommitVotesV1 {
    /// stateless auth + stops replay attacks
    auth_token: ShardAuthToken,
    voter: EncoderPublicKey,
    accepts: Vec<(
        EncoderPublicKey,
        Digest<Signed<ShardCommit, min_sig::BLS12381Signature>>,
    )>,
}

impl ShardCommitVotesV1 {
    pub(crate) const fn new(
        auth_token: ShardAuthToken,
        voter: EncoderPublicKey,
        accepts: Vec<(
            EncoderPublicKey,
            Digest<Signed<ShardCommit, min_sig::BLS12381Signature>>,
        )>,
    ) -> Self {
        Self {
            auth_token,
            voter,
            accepts,
        }
    }
}

impl ShardCommitVotesAPI for ShardCommitVotesV1 {
    fn voter(&self) -> &EncoderPublicKey {
        &self.voter
    }
    fn auth_token(&self) -> &ShardAuthToken {
        &self.auth_token
    }
    fn accepts(
        &self,
    ) -> &[(
        EncoderPublicKey,
        Digest<Signed<ShardCommit, min_sig::BLS12381Signature>>,
    )] {
        &self.accepts
    }
}

pub(crate) fn verify_shard_commit_votes(
    votes: &Signed<ShardCommitVotes, min_sig::BLS12381Signature>,
    shard: &Shard,
) -> SharedResult<()> {
    // the voter must be a member of the evaluation set
    // evaluation sets do not change for a given shard
    if !shard.contains(&votes.voter()) {
        return Err(shared::error::SharedError::ValidationError(
            "voter is not in evaluation set".to_string(),
        ));
    }
    for (encoder, signed_commit_digest) in votes.accepts() {
        if !shard.contains(encoder) {
            return Err(shared::error::SharedError::ValidationError(
                "rejected inference encoder not in inference set".to_string(),
            ));
        }
    }
    // the signature of the vote message must match the voter. The inclusion of the voter in the
    // evaluation set is checked above
    votes.verify(Scope::ShardCommitVotes, votes.voter().inner())?;

    Ok(())
}
