use serde::{Deserialize, Serialize};

use crate::object::ObjectRef;

#[derive(Eq, PartialEq, Clone, Debug, Serialize, Deserialize)]
pub struct TransactionFee {
    // Base transaction fee
    pub base_fee: u64,
    // Fee for each object operation (reads and writes)
    pub operation_fee: u64,
    // Fee based on transaction value
    pub value_fee: u64,
    // Total fee deducted
    pub total_fee: u64,
    // Reference to the gas object after fee deduction
    pub gas_object_ref: ObjectRef,
}

impl TransactionFee {
    pub fn new(
        base_fee: u64,
        operation_fee: u64,
        value_fee: u64,
        gas_object_ref: ObjectRef,
    ) -> Self {
        let total_fee = base_fee + operation_fee + value_fee;

        Self {
            base_fee,
            operation_fee,
            value_fee,
            total_fee,
            gas_object_ref,
        }
    }
}
