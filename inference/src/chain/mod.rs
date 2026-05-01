// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Provider discovery + (off-chain) channel registry.
//!
//! The MVP uses a filesystem-backed [`LocalDiscovery`]; production wires this
//! to the chain's `Offering` GraphQL query and `RegisterProvider` /
//! `Heartbeat` transactions. Channels are local-only state until the chain
//! ships an on-chain channel object — the MVP settles via batched
//! `PayProvider` instead.

pub mod local;
pub mod memory;
pub mod types;

use async_trait::async_trait;

pub use types::*;

#[async_trait]
pub trait Discovery: Send + Sync + 'static {
    async fn list_providers(&self) -> Result<Vec<ProviderRecord>, ChainError>;

    async fn register_provider(&self, record: ProviderRecord) -> Result<(), ChainError>;

    async fn open_channel(&self, params: OpenChannelParams) -> Result<ChannelHandle, ChainError>;

    async fn channel(&self, handle: &ChannelHandle) -> Result<ChannelState, ChainError>;
}
