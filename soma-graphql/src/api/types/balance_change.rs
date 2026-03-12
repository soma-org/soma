// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use async_graphql::*;

use crate::api::scalars::{BigInt, SomaAddress};

/// A balance change for an address in a transaction.
pub struct BalanceChange {
    pub owner: Vec<u8>,
    pub coin_type: String,
    pub amount: i128,
}

#[Object]
impl BalanceChange {
    /// The address whose balance changed.
    async fn owner(&self) -> SomaAddress {
        SomaAddress(self.owner.clone())
    }

    /// The coin type (e.g. "SOMA").
    async fn coin_type(&self) -> &str {
        &self.coin_type
    }

    /// The signed amount of the change. Positive means inflow to the owner.
    async fn amount(&self) -> String {
        self.amount.to_string()
    }
}
