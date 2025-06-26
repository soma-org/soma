use enum_dispatch::enum_dispatch;
use fastcrypto::bls12381::min_sig;
use serde::{Deserialize, Serialize};
use shared::shard::{Shard, ShardAuthToken};
use shared::{
    crypto::keys::EncoderPublicKey, digest::Digest, error::SharedResult, scope::Scope,
    signed::Signed,
};

use super::reveal::Reveal;

#[enum_dispatch]
pub(crate) trait CommitAPI {
    fn auth_token(&self) -> &ShardAuthToken;
    fn author(&self) -> &EncoderPublicKey;
    fn reveal_digest(&self) -> &Digest<Signed<Reveal, min_sig::BLS12381Signature>>;
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub(crate) struct CommitV1 {
    auth_token: ShardAuthToken,
    author: EncoderPublicKey,
    reveal_digest: Digest<Signed<Reveal, min_sig::BLS12381Signature>>,
}

impl CommitV1 {
    pub(crate) fn new(
        auth_token: ShardAuthToken,
        author: EncoderPublicKey,
        reveal_digest: Digest<Signed<Reveal, min_sig::BLS12381Signature>>,
    ) -> Self {
        Self {
            auth_token,
            author,
            reveal_digest,
        }
    }
}

impl CommitAPI for CommitV1 {
    fn auth_token(&self) -> &ShardAuthToken {
        &self.auth_token
    }
    fn author(&self) -> &EncoderPublicKey {
        &self.author
    }
    fn reveal_digest(&self) -> &Digest<Signed<Reveal, min_sig::BLS12381Signature>> {
        &self.reveal_digest
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[enum_dispatch(CommitAPI)]
pub enum Commit {
    V1(CommitV1),
}

pub(crate) fn verify_signed_commit(
    signed_commit: &Signed<Commit, min_sig::BLS12381Signature>,
    shard: &Shard,
) -> SharedResult<()> {
    if !shard.contains(signed_commit.author()) {
        return Err(shared::error::SharedError::ValidationError(
            "author is not in shard".to_string(),
        ));
    }

    signed_commit.verify_signature(Scope::Commit, signed_commit.author().inner())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use fastcrypto::traits::KeyPair;
    use shared::shard::{Shard, ShardAuthToken, ShardEntropy};
    use shared::{
        crypto::keys::EncoderKeyPair, digest::Digest, entropy::BlockEntropy,
        metadata::MetadataCommitment, scope::Scope, signed::Signed,
    };

    use super::{verify_signed_commit, Commit};

    // fn test_verify_signed_commit() {
    //     let mut rng = rand::thread_rng();
    //     let encoder_key = EncoderKeyPair::generate(&mut rng);
    //     let inner_keypair = encoder_key.inner();

    //     let epoch: u64 = 1;
    //     let quorum_threshold: u32 = 1;
    //     let encoders = vec![encoder_key.public()];
    //     let seed = Digest::new(&ShardEntropy::new(
    //         MetadataCommitment::default(),
    //         BlockEntropy::default(),
    //     ))
    //     .unwrap();

    //     let shard = Shard::new(quorum_threshold, encoders, seed, epoch);
    //     let commit = Commit::new_v1(ShardAuthToken::new_for_test(), encoder_key.public());

    //     let signed_commit =
    //         Signed::new(commit, Scope::Commit, &inner_keypair.copy().private()).unwrap();

    //     verify_signed_commit(&signed_commit, &shard).unwrap();
    // }
}

// mismatched auth token digests in route
// shard that doesn't contain
// route that contains a shard member
