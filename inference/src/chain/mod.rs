// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Two chain seams used by the inference crate:
//!
//! - [`ProviderRegistry`] — provider lookup. File-backed for now
//!   ([`local::LocalDiscovery`]); will be replaced by the on-chain
//!   `Provider` shared object in the next PR.
//! - [`ChannelSurface`] — payment-channel ops. Chain-backed
//!   ([`chain::ChainChannelSurface`]) — opens / reads / settles the
//!   on-chain `types::channel::Channel`.

pub mod chain;
pub mod local;
pub mod memory;
pub mod types;

use async_trait::async_trait;
use ::types::base::SomaAddress;
use ::types::channel::{Channel, Voucher};
use ::types::crypto::GenericSignature;
use ::types::object::{CoinType, ObjectID};

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
    /// Open a new channel with `payee` as the provider's address.
    /// Returns the on-chain `Channel`'s `ObjectID`.
    async fn open(
        &self,
        payee: SomaAddress,
        coin_type: CoinType,
        deposit_amount: u64,
    ) -> Result<ObjectID, ChainError>;

    /// Look up the on-chain `Channel` object by id.
    async fn get(&self, id: ObjectID) -> Result<Channel, ChainError>;

    /// Settle on-chain (provider-side caller). Submits the latest
    /// voucher signature.
    async fn settle(
        &self,
        voucher: Voucher,
        sig: GenericSignature,
    ) -> Result<(), ChainError>;

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
