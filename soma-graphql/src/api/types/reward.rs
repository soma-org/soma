// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use async_graphql::*;

use crate::api::scalars::{BigInt, Digest, SomaAddress};

/// A single balance change from a reward claim.
#[derive(Clone)]
pub struct RewardBalance {
    pub recipient: Vec<u8>,
    pub amount: i64,
}

#[Object]
impl RewardBalance {
    /// The address that received (or sent) funds.
    async fn recipient(&self) -> SomaAddress {
        SomaAddress(self.recipient.clone())
    }

    /// The balance change amount (positive = received, negative = sent).
    async fn amount(&self) -> BigInt {
        BigInt(self.amount)
    }
}

/// A reward claim event — derived from ClaimRewards transactions.
#[derive(Clone)]
pub struct Reward {
    pub target_id: Vec<u8>,
    pub cp_sequence_number: i64,
    pub epoch: i64,
    pub tx_digest: Vec<u8>,
    pub balances: Vec<RewardBalance>,
}

#[Object]
impl Reward {
    /// The target whose rewards were claimed.
    async fn target_id(&self) -> SomaAddress {
        SomaAddress(self.target_id.clone())
    }

    /// The checkpoint of the claim.
    async fn checkpoint_sequence_number(&self) -> BigInt {
        BigInt(self.cp_sequence_number)
    }

    /// The epoch of the claim.
    async fn epoch(&self) -> BigInt {
        BigInt(self.epoch)
    }

    /// The transaction that performed the claim.
    async fn tx_digest(&self) -> Digest {
        Digest(self.tx_digest.clone())
    }

    /// Per-recipient balance changes from this reward claim.
    async fn balances(&self) -> &[RewardBalance] {
        &self.balances
    }
}
