// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};

use crate::base::SomaAddress;
use crate::digests::{AskId, TaskDigest};

/// A buyer's request for work on the marketplace.
#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct Ask {
    pub id: AskId,
    pub buyer: SomaAddress,
    pub task_digest: TaskDigest,
    pub max_price_per_bid: u64,
    pub num_bids_wanted: u32,
    pub timeout_ms: u64,
    pub created_at_ms: u64,
    pub status: AskStatus,
    pub accepted_bid_count: u32,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum AskStatus {
    Open,
    Filled,
    Cancelled,
    Expired,
}

impl std::fmt::Display for AskStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AskStatus::Open => write!(f, "open"),
            AskStatus::Filled => write!(f, "filled"),
            AskStatus::Cancelled => write!(f, "cancelled"),
            AskStatus::Expired => write!(f, "expired"),
        }
    }
}
