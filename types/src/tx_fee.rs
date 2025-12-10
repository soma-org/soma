use itertools::Itertools;
use serde::{Deserialize, Serialize};

use crate::effects::{TransactionEffects, TransactionEffectsAPI};

#[derive(Eq, PartialEq, Clone, Debug, Serialize, Deserialize, Default)]
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
    pub fn new(base_fee: u64, operation_fee: u64, value_fee: u64) -> Self {
        let total_fee = base_fee + operation_fee + value_fee;

        Self {
            base_fee,
            operation_fee,
            value_fee,
            total_fee,
        }
    }

    pub fn new_from_txn_effects<'a>(
        transactions: impl Iterator<Item = &'a TransactionEffects>,
    ) -> TransactionFee {
        let (base_fees, operation_fees, value_fees, total_fees): (
            Vec<u64>,
            Vec<u64>,
            Vec<u64>,
            Vec<u64>,
        ) = transactions
            .map(|e| {
                (
                    e.transaction_fee().base_fee,
                    e.transaction_fee().operation_fee,
                    e.transaction_fee().value_fee,
                    e.transaction_fee().total_fee,
                )
            })
            .multiunzip();

        TransactionFee {
            base_fee: base_fees.iter().sum(),
            operation_fee: operation_fees.iter().sum(),
            value_fee: value_fees.iter().sum(),
            total_fee: total_fees.iter().sum(),
        }
    }
}
