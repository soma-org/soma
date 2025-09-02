use enum_dispatch::enum_dispatch;
use fastcrypto::bls12381::min_sig;
use serde::{Deserialize, Serialize};
use shared::{
    crypto::keys::EncoderPublicKey,
    digest::Digest,
    error::{SharedError, SharedResult},
    scope::Scope,
    signed::Signed,
};

use shared::shard::Shard;
use types::shard::ShardAuthToken;

use super::reveal::Reveal;
// reject votes are implicit
#[derive(Clone, Debug, Deserialize, Serialize)]
#[enum_dispatch(CommitVotesAPI)]
pub(crate) enum CommitVotes {
    V1(CommitVotesV1),
}

#[enum_dispatch]
pub(crate) trait CommitVotesAPI {
    fn auth_token(&self) -> &ShardAuthToken;
    fn author(&self) -> &EncoderPublicKey;
    fn accepts(
        &self,
    ) -> &[(
        EncoderPublicKey,
        Digest<Signed<Reveal, min_sig::BLS12381Signature>>,
    )];
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct CommitVotesV1 {
    /// stateless auth + stops replay attacks
    auth_token: ShardAuthToken,
    author: EncoderPublicKey,
    accepts: Vec<(
        EncoderPublicKey,
        Digest<Signed<Reveal, min_sig::BLS12381Signature>>,
    )>,
}

impl CommitVotesV1 {
    pub(crate) const fn new(
        auth_token: ShardAuthToken,
        author: EncoderPublicKey,
        accepts: Vec<(
            EncoderPublicKey,
            Digest<Signed<Reveal, min_sig::BLS12381Signature>>,
        )>,
    ) -> Self {
        Self {
            auth_token,
            author,
            accepts,
        }
    }
}

impl CommitVotesAPI for CommitVotesV1 {
    fn auth_token(&self) -> &ShardAuthToken {
        &self.auth_token
    }
    fn author(&self) -> &EncoderPublicKey {
        &self.author
    }
    fn accepts(
        &self,
    ) -> &[(
        EncoderPublicKey,
        Digest<Signed<Reveal, min_sig::BLS12381Signature>>,
    )] {
        &self.accepts
    }
}

pub(crate) fn verify_commit_votes(
    commit_votes: &Signed<CommitVotes, min_sig::BLS12381Signature>,
    peer: &EncoderPublicKey,
    shard: &Shard,
) -> SharedResult<()> {
    if peer != commit_votes.author() {
        return Err(SharedError::FailedTypeVerification(
            "sending peer must be author".to_string(),
        ));
    }
    for (encoder, _commit_digest) in commit_votes.accepts() {
        if !shard.contains(encoder) {
            return Err(shared::error::SharedError::ValidationError(
                "encoder not in shard".to_string(),
            ));
        }
    }
    // the signature of the vote message must match the voter. The inclusion of the voter in the
    // evaluation set is checked above
    commit_votes.verify_signature(Scope::CommitVotes, commit_votes.author().inner())?;

    Ok(())
}
