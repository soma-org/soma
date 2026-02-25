// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransactionFee {
    // Base transaction fee
    pub base_fee: u64,
    // Fee for each object operation (reads and writes)
    pub operation_fee: u64,
    // Fee based on transaction value
    pub value_fee: u64,
    // Total fee deducted
    pub total_fee: u64,
}

impl TransactionFee {
    pub fn new(
        base_fee: u64,
        operation_fee: u64,
        value_fee: u64,
        total_fee: u64,
    ) -> TransactionFee {
        TransactionFee { base_fee, operation_fee, value_fee, total_fee }
    }
}

impl std::fmt::Display for TransactionFee {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "base_fee: {}, ", self.base_fee)?;
        write!(f, "operation_fee: {}, ", self.operation_fee)?;
        write!(f, "value_fee: {}, ", self.value_fee)?;
        write!(f, "total_fee: {}", self.total_fee)
    }
}
