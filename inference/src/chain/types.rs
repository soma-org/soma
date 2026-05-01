// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};
use types::base::SomaAddress;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProviderRecord {
    pub address: SomaAddress,
    pub pubkey_hex: String,
    pub endpoint: String,
    pub last_heartbeat_ms: u64,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct ChannelHandle(pub String);

impl std::fmt::Display for ChannelHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OpenChannelParams {
    pub client: SomaAddress,
    pub client_pubkey_hex: String,
    pub provider: SomaAddress,
    pub deposit_micros: u64,
    pub expires_ms: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChannelState {
    pub handle: ChannelHandle,
    pub client: SomaAddress,
    pub client_pubkey_hex: String,
    pub provider: SomaAddress,
    pub deposit_micros: u64,
    pub status: ChannelStatus,
    pub opened_ms: u64,
    pub expires_ms: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChannelStatus {
    Open,
    Closed,
    Expired,
}

#[derive(Debug, thiserror::Error)]
pub enum ChainError {
    #[error("not found")]
    NotFound,
    #[error("invalid: {0}")]
    Invalid(String),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("serde: {0}")]
    Serde(#[from] serde_json::Error),
}
