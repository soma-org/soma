use enum_dispatch::enum_dispatch;
use serde::{Deserialize, Serialize};

use shared::shard::ShardAuthToken;

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[enum_dispatch(InputAPI)]
pub enum Input {
    V1(InputV1),
}

#[enum_dispatch]
pub trait InputAPI {
    fn auth_token(&self) -> &ShardAuthToken;
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
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
