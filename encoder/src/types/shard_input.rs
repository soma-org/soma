use enum_dispatch::enum_dispatch;
use serde::{Deserialize, Serialize};

use super::shard_verifier::ShardAuthToken;

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[enum_dispatch(ShardInputAPI)]
pub enum ShardInput {
    V1(ShardInputV1),
}

#[enum_dispatch]
pub trait ShardInputAPI {
    fn auth_token(&self) -> &ShardAuthToken;
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct ShardInputV1 {
    auth_token: ShardAuthToken,
}

impl ShardInputV1 {
    pub(crate) fn new(auth_token: ShardAuthToken) -> Self {
        Self { auth_token }
    }
}

impl ShardInputAPI for ShardInputV1 {
    fn auth_token(&self) -> &ShardAuthToken {
        &self.auth_token
    }
}
