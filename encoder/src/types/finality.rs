use enum_dispatch::enum_dispatch;
use fastcrypto::bls12381::min_sig;
use serde::{Deserialize, Serialize};
use shared::{crypto::keys::EncoderPublicKey, error::SharedResult, scope::Scope, signed::Signed};

use shared::shard::Shard;
use types::shard::ShardAuthToken;

#[derive(Debug, Clone, Deserialize, Serialize)]
#[enum_dispatch(FinalityAPI)]
pub enum Finality {
    V1(FinalityV1),
}

#[enum_dispatch]
pub trait FinalityAPI {
    fn auth_token(&self) -> &ShardAuthToken;
    fn encoder(&self) -> &EncoderPublicKey;
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FinalityV1 {
    auth_token: ShardAuthToken,
    encoder: EncoderPublicKey,
}

impl FinalityV1 {
    pub fn new(auth_token: ShardAuthToken, encoder: EncoderPublicKey) -> Self {
        Self {
            auth_token,
            encoder,
        }
    }
}

impl FinalityAPI for FinalityV1 {
    fn auth_token(&self) -> &ShardAuthToken {
        &self.auth_token
    }
    fn encoder(&self) -> &EncoderPublicKey {
        &self.encoder
    }
}

pub(crate) fn verify_signed_finality(
    signed_finality: &Signed<Finality, min_sig::BLS12381Signature>,
    shard: &Shard,
) -> SharedResult<()> {
    if !shard.contains(&signed_finality.encoder()) {
        return Err(shared::error::SharedError::ValidationError(
            "encoder is not in the shard".to_string(),
        ));
    }

    let _ = signed_finality.verify_signature(Scope::Finality, signed_finality.encoder().inner())?;

    Ok(())
}
