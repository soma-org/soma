// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use async_graphql::*;

use crate::api::scalars::{BigInt, SomaAddress};

/// Reputation data for an address, derived from settlement history.
pub struct Reputation {
    pub address: Vec<u8>,
    /// As buyer
    pub total_asks_created: i64,
    pub total_bids_accepted: i64,
    pub total_volume_spent: i64,
    pub unique_sellers: i64,
    /// As seller
    pub total_bids_submitted: i64,
    pub total_bids_won: i64,
    pub total_volume_earned: i64,
    pub negative_ratings_received: i64,
    pub total_settlements_as_seller: i64,
}

#[Object]
impl Reputation {
    async fn address(&self) -> SomaAddress {
        SomaAddress(self.address.clone())
    }

    // --- Buyer metrics ---

    async fn total_asks_created(&self) -> i32 {
        self.total_asks_created as i32
    }

    async fn total_bids_accepted(&self) -> i32 {
        self.total_bids_accepted as i32
    }

    async fn total_volume_spent(&self) -> BigInt {
        BigInt(self.total_volume_spent)
    }

    /// Number of unique sellers this buyer has transacted with (anti-sybil signal).
    async fn unique_sellers(&self) -> i32 {
        self.unique_sellers as i32
    }

    // --- Seller metrics ---

    async fn total_bids_submitted(&self) -> i32 {
        self.total_bids_submitted as i32
    }

    async fn total_bids_won(&self) -> i32 {
        self.total_bids_won as i32
    }

    async fn total_volume_earned(&self) -> BigInt {
        BigInt(self.total_volume_earned)
    }

    /// Percentage of settlements without a negative rating (0.0 to 1.0).
    async fn seller_approval_rate(&self) -> Option<f64> {
        if self.total_settlements_as_seller == 0 {
            None
        } else {
            let positive =
                self.total_settlements_as_seller - self.negative_ratings_received;
            Some(positive as f64 / self.total_settlements_as_seller as f64)
        }
    }

    /// Ratio of bids won to bids submitted (0.0 to 1.0).
    async fn bid_to_win_ratio(&self) -> Option<f64> {
        if self.total_bids_submitted == 0 {
            None
        } else {
            Some(self.total_bids_won as f64 / self.total_bids_submitted as f64)
        }
    }
}

/// An edge in the counterparty graph between two addresses.
pub struct CounterpartyEdge {
    pub address: Vec<u8>,
    pub transaction_count: i64,
    pub total_volume: i64,
    pub negative_ratings: i64,
}

#[Object]
impl CounterpartyEdge {
    async fn address(&self) -> SomaAddress {
        SomaAddress(self.address.clone())
    }

    async fn transaction_count(&self) -> i32 {
        self.transaction_count as i32
    }

    async fn total_volume(&self) -> BigInt {
        BigInt(self.total_volume)
    }

    async fn negative_ratings(&self) -> i32 {
        self.negative_ratings as i32
    }
}
