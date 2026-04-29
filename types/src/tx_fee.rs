// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};

use crate::effects::{TransactionEffects, TransactionEffectsAPI};

/// The total fee deducted from a transaction's gas coin.
///
/// Tx fee = `unit_fee * executor.fee_units(...)`. Each executor decides how
/// many units its op costs based on op shape; the protocol-level `unit_fee`
/// (in shannons) sets the price per unit.
#[derive(Eq, PartialEq, Clone, Debug, Serialize, Deserialize, Default)]
pub struct TransactionFee {
    /// Total fee deducted (in shannons).
    pub total_fee: u64,
}

impl TransactionFee {
    pub fn new(total_fee: u64) -> Self {
        Self { total_fee }
    }

    /// Sum the fees across an iterator of transaction effects.
    pub fn new_from_txn_effects<'a>(
        transactions: impl Iterator<Item = &'a TransactionEffects>,
    ) -> TransactionFee {
        let total_fee = transactions.map(|e| e.transaction_fee().total_fee).sum();
        TransactionFee { total_fee }
    }
}
