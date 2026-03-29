// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};

use crate::base::SomaAddress;
use crate::digests::{AskId, BidId, ResponseDigest, SettlementId, TaskDigest};

/// Created when a buyer accepts a bid. AcceptBid = settlement = payment.
/// This is the core edge in the reputation graph.
#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct Settlement {
    pub id: SettlementId,
    pub ask_id: AskId,
    pub bid_id: BidId,
    pub buyer: SomaAddress,
    pub seller: SomaAddress,
    pub amount: u64,
    pub task_digest: TaskDigest,
    pub response_digest: ResponseDigest,
    pub settled_at_ms: u64,
    pub seller_rating: SellerRating,
    pub rating_deadline_ms: u64,
}

/// Only negative ratings go on-chain. The default is Positive (no tx needed).
/// Pending + past deadline = Positive.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum SellerRating {
    Positive,
    Negative,
}

impl std::fmt::Display for SellerRating {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SellerRating::Positive => write!(f, "positive"),
            SellerRating::Negative => write!(f, "negative"),
        }
    }
}
