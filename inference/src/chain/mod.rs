// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Two chain seams used by the inference crate:
//!
//! - [`ProviderRegistry`] — provider lookup. Indexer-backed
//!   ([`indexer::IndexerProviderRegistry`]) queries the soma-graphql
//!   `providers()` endpoint, which mirrors the on-chain `Provider`
//!   shared object. [`memory::MemoryDiscovery`] exists for in-process
//!   tests.
//! - [`ChannelSurface`] — payment-channel ops. Chain-backed
//!   ([`chain::ChainChannelSurface`]) — opens / reads / settles the
//!   on-chain `types::channel::Channel`.

pub mod chain;
pub mod indexer;
pub mod memory;
pub mod types;

use ::types::base::SomaAddress;
use ::types::channel::{Channel, Voucher};
use ::types::crypto::GenericSignature;
use ::types::object::{CoinType, ObjectID};
use async_trait::async_trait;

pub use self::types::*;

/// Off-chain provider lookup. Returns enough metadata for the proxy
/// to choose a provider for a model and reach its HTTP endpoint.
#[async_trait]
pub trait ProviderRegistry: Send + Sync + 'static {
    async fn list_providers(&self) -> Result<Vec<ProviderRecord>, ChainError>;
    async fn register_provider(&self, record: ProviderRecord) -> Result<(), ChainError>;
}

/// On-chain payment-channel surface. Wraps the SDK's `sdk::channel`
/// helpers so the inference crate doesn't have to know about
/// `WalletContext`/transaction building.
#[async_trait]
pub trait ChannelSurface: Send + Sync + 'static {
    /// Open a new channel with `payee` as the provider's address,
    /// bound to the given `model_id`. The chain executor snapshots
    /// the provider's offering for that model into the channel.
    async fn open(
        &self,
        payee: SomaAddress,
        coin_type: CoinType,
        deposit_amount: u64,
        model_id: String,
    ) -> Result<ObjectID, ChainError>;

    /// Look up the on-chain `Channel` object by id.
    async fn get(&self, id: ObjectID) -> Result<Channel, ChainError>;

    /// Settle on-chain (provider-side caller). Submits the latest
    /// voucher signature.
    async fn settle(&self, voucher: Voucher, sig: GenericSignature) -> Result<(), ChainError>;

    /// Top up the deposit (payer-only on-chain).
    async fn top_up(
        &self,
        id: ObjectID,
        coin_type: CoinType,
        amount: u64,
    ) -> Result<(), ChainError>;

    /// Begin the close timer (payer-only on-chain).
    async fn request_close(&self, id: ObjectID) -> Result<(), ChainError>;

    /// Withdraw remainder after grace (payer-only on-chain).
    async fn withdraw_after_timeout(&self, id: ObjectID) -> Result<(), ChainError>;

    /// Address whose key signs vouchers (typically the payer's).
    /// Used by the proxy to sign with the right key.
    fn signer_address(&self) -> SomaAddress;
}
