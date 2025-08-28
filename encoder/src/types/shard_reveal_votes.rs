use enum_dispatch::enum_dispatch;
use fastcrypto::bls12381::min_sig;
use serde::{Deserialize, Serialize};
use shared::shard::Shard;
use shared::{crypto::keys::EncoderPublicKey, error::SharedResult, scope::Scope, signed::Signed};
use types::shard::ShardAuthToken;

/// Reject votes are explicit. The rest of encoders in a shard receive implicit accept votes.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[enum_dispatch(ShardRevealVotesAPI)]
pub enum ShardRevealVotes {
    V1(ShardRevealVotesV1),
}

#[enum_dispatch]
pub trait ShardRevealVotesAPI {
    fn voter(&self) -> &EncoderPublicKey;
    fn auth_token(&self) -> &ShardAuthToken;
    fn accepts(&self) -> &[EncoderPublicKey];
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct ShardRevealVotesV1 {
    /// stateless auth + stops replay attacks
    auth_token: ShardAuthToken,
    voter: EncoderPublicKey,
    /// Reject votes are explicit. The rest of encoders in a shard receive implicit accept votes.
    accepts: Vec<EncoderPublicKey>,
}

impl ShardRevealVotesV1 {
    pub(crate) const fn new(
        auth_token: ShardAuthToken,
        voter: EncoderPublicKey,
        accepts: Vec<EncoderPublicKey>,
    ) -> Self {
        Self {
            auth_token,
            voter,
            accepts,
        }
    }
}

impl ShardRevealVotesAPI for ShardRevealVotesV1 {
    fn voter(&self) -> &EncoderPublicKey {
        &self.voter
    }
    fn auth_token(&self) -> &ShardAuthToken {
        &self.auth_token
    }
    fn accepts(&self) -> &[EncoderPublicKey] {
        &self.accepts
    }
}

pub(crate) fn verify_shard_reveal_votes(
    votes: &Signed<ShardRevealVotes, min_sig::BLS12381Signature>,
    shard: &Shard,
) -> SharedResult<()> {
    // the voter must be a member of the evaluation set
    // evaluation sets do not change for a given shard
    if !shard.contains(votes.voter()) {
        return Err(shared::error::SharedError::ValidationError(
            "voter is not in evaluation set".to_string(),
        ));
    }
    // verify that the reject encoder indices are valid slots
    // may want to check for uniqueness but since acceptance votes are implicit
    // and rejection votes are implicit multiple redundant votes is fine unless they are counted twice
    for encoder in votes.accepts() {
        if !shard.contains(encoder) {
            return Err(shared::error::SharedError::ValidationError(
                "rejected inference encoder not in inference set".to_string(),
            ));
        }
    }
    // the signature of the vote message must match the voter. The inclusion of the voter in the
    // evaluation set is checked above
    votes.verify(Scope::ShardRevealVotes, votes.voter().inner())?;

    Ok(())
}
