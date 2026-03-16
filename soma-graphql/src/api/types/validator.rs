// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use async_graphql::*;

use crate::api::scalars::{BigInt, SomaAddress};

/// A validator on the Soma network at a specific epoch.
pub struct Validator {
    pub address: Vec<u8>,
    pub epoch: i64,
    pub voting_power: i64,
    pub commission_rate: i64,
    pub next_epoch_commission_rate: i64,
    pub staking_pool_id: Vec<u8>,
    pub stake: i64,
    pub pending_stake: i64,
    pub name: Option<String>,
    pub network_address: Option<String>,
    pub proxy_address: Option<String>,
}

#[Object]
impl Validator {
    /// The validator's address (hex with 0x prefix).
    async fn address(&self) -> SomaAddress {
        SomaAddress(self.address.clone())
    }

    /// The epoch this snapshot is from.
    async fn epoch(&self) -> BigInt {
        BigInt(self.epoch)
    }

    /// The validator's voting power in consensus.
    async fn voting_power(&self) -> BigInt {
        BigInt(self.voting_power)
    }

    /// Current commission rate (basis points).
    async fn commission_rate(&self) -> BigInt {
        BigInt(self.commission_rate)
    }

    /// Commission rate for the next epoch (basis points).
    async fn next_epoch_commission_rate(&self) -> BigInt {
        BigInt(self.next_epoch_commission_rate)
    }

    /// The staking pool object ID.
    async fn staking_pool_id(&self) -> SomaAddress {
        SomaAddress(self.staking_pool_id.clone())
    }

    /// Total staked amount (shannons).
    async fn stake(&self) -> BigInt {
        BigInt(self.stake)
    }

    /// Pending stake for next epoch (shannons).
    async fn pending_stake(&self) -> BigInt {
        BigInt(self.pending_stake)
    }

    /// Validator name (if set in metadata).
    async fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Network address (multiaddr).
    async fn network_address(&self) -> Option<&str> {
        self.network_address.as_deref()
    }

    /// Proxy address for client downloads.
    async fn proxy_address(&self) -> Option<&str> {
        self.proxy_address.as_deref()
    }
}
