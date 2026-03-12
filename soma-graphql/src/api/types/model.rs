// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use async_graphql::*;

use crate::api::scalars::{Base64, BigInt, SomaAddress};

/// A model registered in the Soma network's model registry.
pub struct Model {
    pub model_id: Vec<u8>,
    pub epoch: i64,
    pub status: String,
    pub owner: Vec<u8>,
    pub architecture_version: i64,
    pub commit_epoch: i64,
    pub stake: i64,
    pub commission_rate: i64,
    pub has_embedding: bool,
    pub state_bcs: Vec<u8>,
}

#[Object]
impl Model {
    /// The model's ID.
    async fn model_id(&self) -> SomaAddress {
        SomaAddress(self.model_id.clone())
    }

    /// The epoch this snapshot was taken at.
    async fn epoch(&self) -> BigInt {
        BigInt(self.epoch)
    }

    /// Status: active, pending, or inactive.
    async fn status(&self) -> &str {
        &self.status
    }

    /// The model owner's address.
    async fn owner(&self) -> SomaAddress {
        SomaAddress(self.owner.clone())
    }

    /// Protocol-versioned model architecture version.
    async fn architecture_version(&self) -> BigInt {
        BigInt(self.architecture_version)
    }

    /// The epoch when this model was committed.
    async fn commit_epoch(&self) -> BigInt {
        BigInt(self.commit_epoch)
    }

    /// The model's staked SOMA balance.
    async fn stake(&self) -> BigInt {
        BigInt(self.stake)
    }

    /// Commission rate in basis points.
    async fn commission_rate(&self) -> BigInt {
        BigInt(self.commission_rate)
    }

    /// Whether the model has revealed its embedding.
    async fn has_embedding(&self) -> bool {
        self.has_embedding
    }

    /// Full BCS-serialized ModelV1 state.
    async fn state_bcs(&self) -> Base64 {
        Base64(self.state_bcs.clone())
    }
}
