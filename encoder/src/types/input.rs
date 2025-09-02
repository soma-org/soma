use enum_dispatch::enum_dispatch;
use fastcrypto::bls12381::min_sig;
use serde::{Deserialize, Serialize};
use shared::{crypto::keys::EncoderPublicKey, error::SharedResult, shard::Shard, signed::Signed};
use types::shard::ShardAuthToken;

#[derive(Debug, Clone, Deserialize, Serialize)]
#[enum_dispatch(InputAPI)]
pub enum Input {
    V1(InputV1),
}

#[enum_dispatch]
pub trait InputAPI {
    fn auth_token(&self) -> &ShardAuthToken;
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct InputV1 {
    auth_token: ShardAuthToken,
}

impl InputV1 {
    pub fn new(auth_token: ShardAuthToken) -> Self {
        Self { auth_token }
    }
}

impl InputAPI for InputV1 {
    fn auth_token(&self) -> &ShardAuthToken {
        &self.auth_token
    }
}

pub(crate) fn verify_input(
    input: &Signed<Input, min_sig::BLS12381Signature>,
    shard: &Shard,
) -> SharedResult<()> {
    // TODO: need to fix this to work with the correct signature
    // input.verify_signature(Scope::Input, .author().inner())?;
    Ok(())
}
