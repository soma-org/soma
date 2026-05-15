// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Types shared by the two chain seams: [`super::ProviderRegistry`]
//! (provider lookup, file-backed for now) and [`super::ChannelSurface`]
//! (chain-backed payment channels).
//!
//! Channels themselves are now `types::channel::Channel` (on-chain
//! shared object) — there is no separate inference-side channel
//! handle/status. Provider records remain inference-local for one
//! more PR, until the on-chain `Provider` object lands.

use serde::{Deserialize, Serialize};
use types::base::SomaAddress;

/// In-memory projection of an on-chain `Provider` record, used by
/// [`super::ProviderRegistry`] consumers (the proxy router, in
/// particular). The on-chain `Provider` object is the source of
/// truth; `ProviderRecord` is what the indexer-backed registry
/// returns to the proxy.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProviderRecord {
    pub address: SomaAddress,
    pub pubkey_hex: String,
    pub endpoint: String,
}

/// Computed channel status for client/provider display. Derived from
/// the on-chain `Channel.close_requested_at_ms` plus the protocol's
/// grace period — there is no on-chain `status` field.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChannelStatus {
    /// Channel is live and accepting `Settle` calls.
    Open,
    /// `RequestClose` has been called; payer can `WithdrawAfterTimeout`
    /// once `current_ts_ms >= earliest_withdrawable_ms`.
    Closing { earliest_withdrawable_ms: u64 },
}

impl ChannelStatus {
    /// Compute the status of an on-chain `Channel` given the current
    /// chain time and the protocol's `channel_grace_period_ms`.
    pub fn from_channel(ch: &types::channel::Channel, grace_period_ms: u64) -> Self {
        match ch.close_requested_at_ms() {
            None => Self::Open,
            Some(at_ms) => {
                Self::Closing { earliest_withdrawable_ms: at_ms.saturating_add(grace_period_ms) }
            }
        }
    }
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
    #[error("rpc: {0}")]
    Rpc(String),
    #[error("tx: {0}")]
    Tx(String),
}
