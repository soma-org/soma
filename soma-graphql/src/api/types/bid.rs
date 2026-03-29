// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use async_graphql::*;

use crate::api::scalars::{BigInt, Digest, SomaAddress};

/// A seller's offer to fulfill an ask.
#[derive(Clone)]
pub struct Bid {
    pub bid_id: Vec<u8>,
    pub ask_id: Vec<u8>,
    pub seller: Vec<u8>,
    pub price: i64,
    pub response_digest: Vec<u8>,
    pub created_at_ms: i64,
    pub status: String,
}

#[Object]
impl Bid {
    async fn bid_id(&self) -> SomaAddress {
        SomaAddress(self.bid_id.clone())
    }

    async fn ask_id(&self) -> SomaAddress {
        SomaAddress(self.ask_id.clone())
    }

    async fn seller(&self) -> SomaAddress {
        SomaAddress(self.seller.clone())
    }

    /// Bid price in USDC microdollars.
    async fn price(&self) -> BigInt {
        BigInt(self.price)
    }

    /// Blake2b hash of the seller's off-chain response content.
    async fn response_digest(&self) -> Digest {
        Digest(self.response_digest.clone())
    }

    async fn created_at_ms(&self) -> BigInt {
        BigInt(self.created_at_ms)
    }

    /// Current status: pending, accepted, rejected, or expired.
    async fn status(&self) -> &str {
        &self.status
    }
}
