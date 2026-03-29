// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use async_graphql::*;

use crate::api::scalars::{BigInt, Digest, SomaAddress};

/// A buyer's request for work on the marketplace.
#[derive(Clone)]
pub struct Ask {
    pub ask_id: Vec<u8>,
    pub buyer: Vec<u8>,
    pub task_digest: Vec<u8>,
    pub max_price_per_bid: i64,
    pub num_bids_wanted: i32,
    pub timeout_ms: i64,
    pub created_at_ms: i64,
    pub status: String,
    pub accepted_bid_count: i32,
}

#[Object]
impl Ask {
    async fn ask_id(&self) -> SomaAddress {
        SomaAddress(self.ask_id.clone())
    }

    async fn buyer(&self) -> SomaAddress {
        SomaAddress(self.buyer.clone())
    }

    async fn task_digest(&self) -> Digest {
        Digest(self.task_digest.clone())
    }

    /// Maximum USDC price per bid (microdollars).
    async fn max_price_per_bid(&self) -> BigInt {
        BigInt(self.max_price_per_bid)
    }

    /// How many bids the buyer intends to accept.
    async fn num_bids_wanted(&self) -> i32 {
        self.num_bids_wanted
    }

    /// Deadline for bids (milliseconds from creation).
    async fn timeout_ms(&self) -> BigInt {
        BigInt(self.timeout_ms)
    }

    /// Consensus timestamp at creation.
    async fn created_at_ms(&self) -> BigInt {
        BigInt(self.created_at_ms)
    }

    /// Current status: open, filled, cancelled, or expired.
    async fn status(&self) -> &str {
        &self.status
    }

    /// How many bids have been accepted so far.
    async fn accepted_bid_count(&self) -> i32 {
        self.accepted_bid_count
    }
}
