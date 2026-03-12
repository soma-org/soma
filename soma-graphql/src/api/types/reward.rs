// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use async_graphql::*;

use crate::api::scalars::{Base64, BigInt, Digest, SomaAddress};

/// A reward claim event — derived from ClaimRewards transactions.
pub struct Reward {
    pub target_id: Vec<u8>,
    pub cp_sequence_number: i64,
    pub epoch: i64,
    pub tx_digest: Vec<u8>,
    pub balance_changes_bcs: Vec<u8>,
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

    /// BCS-serialized balance changes showing who received what.
    async fn balance_changes_bcs(&self) -> Base64 {
        Base64(self.balance_changes_bcs.clone())
    }
}
