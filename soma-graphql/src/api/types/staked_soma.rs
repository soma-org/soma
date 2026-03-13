// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use async_graphql::*;

use crate::api::scalars::{BigInt, SomaAddress};

/// A staking position — SOMA tokens delegated to a model's staking pool.
pub struct StakedSoma {
    pub staked_soma_id: Vec<u8>,
    pub owner: Vec<u8>,
    pub pool_id: Vec<u8>,
    pub stake_activation_epoch: i64,
    pub principal: i64,
}

#[Object]
impl StakedSoma {
    /// The StakedSoma object's ID.
    async fn staked_soma_id(&self) -> SomaAddress {
        SomaAddress(self.staked_soma_id.clone())
    }

    /// The owner of this staking position.
    async fn owner(&self) -> SomaAddress {
        SomaAddress(self.owner.clone())
    }

    /// The staking pool this position belongs to.
    async fn pool_id(&self) -> SomaAddress {
        SomaAddress(self.pool_id.clone())
    }

    /// The epoch when this stake becomes active.
    async fn stake_activation_epoch(&self) -> BigInt {
        BigInt(self.stake_activation_epoch)
    }

    /// The principal amount staked (in shannons).
    async fn principal(&self) -> BigInt {
        BigInt(self.principal)
    }
}
