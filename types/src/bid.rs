// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};

use crate::base::SomaAddress;
use crate::digests::{AskId, BidId, ResponseDigest};

/// A seller's offer to fulfill an ask.
#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct Bid {
    pub id: BidId,
    pub ask_id: AskId,
    pub seller: SomaAddress,
    pub price: u64,
    pub response_digest: ResponseDigest,
    pub created_at_ms: u64,
    pub status: BidStatus,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum BidStatus {
    Pending,
    Accepted,
    Rejected,
    Expired,
}

impl std::fmt::Display for BidStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BidStatus::Pending => write!(f, "pending"),
            BidStatus::Accepted => write!(f, "accepted"),
            BidStatus::Rejected => write!(f, "rejected"),
            BidStatus::Expired => write!(f, "expired"),
        }
    }
}
