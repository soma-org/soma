// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use async_graphql::*;

use crate::api::scalars::BigInt;

/// The range of checkpoints for which index-backed queries are guaranteed to
/// have complete data. Outside this range, data may have been pruned.
pub struct AvailableRange {
    /// Lowest checkpoint guaranteed available.
    pub first: i64,
    /// Highest indexed checkpoint.
    pub last: i64,
}

#[Object]
impl AvailableRange {
    /// Lowest checkpoint guaranteed available for index queries.
    async fn first(&self) -> BigInt {
        BigInt(self.first)
    }

    /// Highest indexed checkpoint.
    async fn last(&self) -> BigInt {
        BigInt(self.last)
    }
}
