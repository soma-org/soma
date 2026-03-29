// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use async_graphql::*;

use crate::api::scalars::{BigInt, Digest, SomaAddress};

/// A settlement — created when a buyer accepts a bid. The core edge in the reputation graph.
#[derive(Clone)]
pub struct Settlement {
    pub settlement_id: Vec<u8>,
    pub ask_id: Vec<u8>,
    pub bid_id: Vec<u8>,
    pub buyer: Vec<u8>,
    pub seller: Vec<u8>,
    pub amount: i64,
    pub task_digest: Vec<u8>,
    pub response_digest: Vec<u8>,
    pub settled_at_ms: i64,
    pub seller_rating: String,
    pub rating_deadline_ms: i64,
}

#[Object]
impl Settlement {
    async fn settlement_id(&self) -> SomaAddress {
        SomaAddress(self.settlement_id.clone())
    }

    async fn ask_id(&self) -> SomaAddress {
        SomaAddress(self.ask_id.clone())
    }

    async fn bid_id(&self) -> SomaAddress {
        SomaAddress(self.bid_id.clone())
    }

    async fn buyer(&self) -> SomaAddress {
        SomaAddress(self.buyer.clone())
    }

    async fn seller(&self) -> SomaAddress {
        SomaAddress(self.seller.clone())
    }

    /// Payment amount in USDC microdollars (bid.price - value_fee).
    async fn amount(&self) -> BigInt {
        BigInt(self.amount)
    }

    async fn task_digest(&self) -> Digest {
        Digest(self.task_digest.clone())
    }

    async fn response_digest(&self) -> Digest {
        Digest(self.response_digest.clone())
    }

    /// Consensus timestamp of settlement.
    async fn settled_at_ms(&self) -> BigInt {
        BigInt(self.settled_at_ms)
    }

    /// Seller rating: "positive" (default) or "negative".
    async fn seller_rating(&self) -> &str {
        &self.seller_rating
    }

    /// Deadline for the buyer to submit a negative rating.
    async fn rating_deadline_ms(&self) -> BigInt {
        BigInt(self.rating_deadline_ms)
    }
}
