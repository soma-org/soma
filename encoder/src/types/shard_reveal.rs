use enum_dispatch::enum_dispatch;
use serde::{Deserialize, Serialize};
use shared::crypto::EncryptionKey;

use super::{
    encoder_committee::{EncoderIndex, InferenceEncoder},
    shard_verifier::ShardAuthToken,
};

/// Shard commit is the wrapper that contains the versioned shard commit. It
/// represents the encoders response to a batch of data
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[enum_dispatch(ShardRevealAPI)]
pub enum ShardReveal {
    V1(ShardRevealV1),
}

/// `ShardRevealAPI` is the trait that every shard commit version must implement
#[enum_dispatch]
pub(crate) trait ShardRevealAPI {
    fn auth_token(&self) -> &ShardAuthToken;
    fn inference_encoder(&self) -> &InferenceEncoder;
    /// returns the encryption key
    fn key(&self) -> &EncryptionKey;
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub(crate) struct ShardRevealV1 {
    auth_token: ShardAuthToken,
    inference_encoder: InferenceEncoder,
    key: EncryptionKey,
}

impl ShardRevealV1 {
    pub(crate) const fn new(
        auth_token: ShardAuthToken,
        inference_encoder: InferenceEncoder,
        key: EncryptionKey,
    ) -> Self {
        Self {
            auth_token,
            inference_encoder,
            key,
        }
    }
}

impl ShardRevealAPI for ShardRevealV1 {
    fn auth_token(&self) -> &ShardAuthToken {
        &self.auth_token
    }
    fn inference_encoder(&self) -> &InferenceEncoder {
        &self.inference_encoder
    }
    fn key(&self) -> &EncryptionKey {
        &self.key
    }
}
