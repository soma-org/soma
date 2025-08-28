use enum_dispatch::enum_dispatch;
use fastcrypto::bls12381::min_sig;
use serde::{Deserialize, Serialize};
use shared::shard::Shard;
use shared::{
    crypto::{keys::EncoderPublicKey, EncryptionKey},
    error::SharedResult,
    scope::Scope,
    signed::Signed,
};
use types::shard::ShardAuthToken;

/// Shard commit is the wrapper that contains the versioned shard commit. It
/// represents the encoders response to a batch of data
#[derive(Clone, Debug, Deserialize, Serialize)]
#[enum_dispatch(ShardRevealAPI)]
pub enum ShardReveal {
    V1(ShardRevealV1),
}

/// `ShardRevealAPI` is the trait that every shard commit version must implement
#[enum_dispatch]
pub(crate) trait ShardRevealAPI {
    fn auth_token(&self) -> &ShardAuthToken;
    fn encoder(&self) -> &EncoderPublicKey;
    /// returns the encryption key
    fn key(&self) -> &EncryptionKey;
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct ShardRevealV1 {
    auth_token: ShardAuthToken,
    encoder: EncoderPublicKey,
    key: EncryptionKey,
}

impl ShardRevealV1 {
    pub(crate) const fn new(
        auth_token: ShardAuthToken,
        encoder: EncoderPublicKey,
        key: EncryptionKey,
    ) -> Self {
        Self {
            auth_token,
            encoder,
            key,
        }
    }
}

impl ShardRevealAPI for ShardRevealV1 {
    fn auth_token(&self) -> &ShardAuthToken {
        &self.auth_token
    }
    fn encoder(&self) -> &EncoderPublicKey {
        &self.encoder
    }
    fn key(&self) -> &EncryptionKey {
        &self.key
    }
}

pub(crate) fn verify_signed_shard_reveal(
    signed_shard_reveal: &Signed<ShardReveal, min_sig::BLS12381Signature>,
    shard: &Shard,
) -> SharedResult<()> {
    // the reveal slot must be a member of the shard inference slot
    // in the case of routing, the original slot is still expected to handle the reveal since this allows
    // routing to take place without needing to reorganize all the communication of the shard
    if !shard.contains(&signed_shard_reveal.encoder()) {
        return Err(shared::error::SharedError::ValidationError(
            "inference encoder is not in inference set".to_string(),
        ));
    }

    // the reveal message must be signed by the slot
    let _ =
        signed_shard_reveal.verify(Scope::ShardReveal, signed_shard_reveal.encoder().inner())?;
    Ok(())
}
