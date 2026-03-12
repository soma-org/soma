// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use async_graphql::*;

use crate::api::scalars::{Base64, BigInt, SomaAddress};

/// A target on the Soma network — represents an inference task.
pub struct Target {
    pub target_id: Vec<u8>,
    pub cp_sequence_number: i64,
    pub epoch: i64,
    pub status: String,
    pub submitter: Option<Vec<u8>>,
    pub winning_model_id: Option<Vec<u8>>,
    pub reward_pool: i64,
    pub bond_amount: i64,
    pub report_count: i32,
    pub state_bcs: Vec<u8>,
}

#[Object]
impl Target {
    /// The target's object ID.
    async fn target_id(&self) -> SomaAddress {
        SomaAddress(self.target_id.clone())
    }

    /// The checkpoint where this version of the target was written.
    async fn checkpoint_sequence_number(&self) -> BigInt {
        BigInt(self.cp_sequence_number)
    }

    /// The epoch this target was created in.
    async fn epoch(&self) -> BigInt {
        BigInt(self.epoch)
    }

    /// The target's status: Open, Filled, or Claimed.
    async fn status(&self) -> &str {
        &self.status
    }

    /// The address that filled this target (if any).
    async fn submitter(&self) -> Option<SomaAddress> {
        self.submitter.as_ref().map(|s| SomaAddress(s.clone()))
    }

    /// The model used in the winning submission (if any).
    async fn winning_model_id(&self) -> Option<SomaAddress> {
        self.winning_model_id
            .as_ref()
            .map(|id| SomaAddress(id.clone()))
    }

    /// The pre-allocated reward amount.
    async fn reward_pool(&self) -> BigInt {
        BigInt(self.reward_pool)
    }

    /// The submission bond held.
    async fn bond_amount(&self) -> BigInt {
        BigInt(self.bond_amount)
    }

    /// Number of fraud reports filed against this target.
    async fn report_count(&self) -> i32 {
        self.report_count
    }

    /// Full BCS-serialized TargetV1 state.
    async fn state_bcs(&self) -> Base64 {
        Base64(self.state_bcs.clone())
    }
}

/// Filter for querying targets.
#[derive(InputObject, Default)]
pub struct TargetFilter {
    /// Filter by status (Open, Filled, Claimed).
    pub status: Option<String>,
    /// Filter by epoch.
    pub epoch: Option<i64>,
}
