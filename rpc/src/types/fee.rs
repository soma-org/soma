// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransactionFee {
    /// Total fee deducted (in shannons).
    pub total_fee: u64,
}

impl TransactionFee {
    pub fn new(total_fee: u64) -> TransactionFee {
        TransactionFee { total_fee }
    }
}

impl std::fmt::Display for TransactionFee {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "total_fee: {}", self.total_fee)
    }
}
