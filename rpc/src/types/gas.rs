use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GasCostSummary {
    /// Cost of computation/execution
    pub computation_cost: u64,

    /// Storage cost, it's the sum of all storage cost for all objects created or mutated.
    pub storage_cost: u64,

    /// The amount of storage cost refunded to the user for all objects deleted or mutated in the
    /// transaction.
    pub storage_rebate: u64,

    /// The fee for the rebate. The portion of the storage rebate kept by the system.
    pub non_refundable_storage_fee: u64,
}

impl GasCostSummary {
    /// Create a new gas cost summary.
    ///
    /// # Arguments
    /// * `computation_cost` - Cost of computation cost/execution.
    /// * `storage_cost` - Storage cost, it's the sum of all storage cost for all objects created or mutated.
    /// * `storage_rebate` - The amount of storage cost refunded to the user for all objects deleted or mutated in the transaction.
    /// * `non_refundable_storage_fee` - The fee for the rebate. The portion of the storage rebate kept by the system.
    pub fn new(
        computation_cost: u64,
        storage_cost: u64,
        storage_rebate: u64,
        non_refundable_storage_fee: u64,
    ) -> GasCostSummary {
        GasCostSummary {
            computation_cost,
            storage_cost,
            storage_rebate,
            non_refundable_storage_fee,
        }
    }

    /// The total gas used, which is the sum of computation and storage costs.
    pub fn gas_used(&self) -> u64 {
        self.computation_cost + self.storage_cost
    }

    /// The net gas usage, which is the total gas used minus the storage rebate.
    /// A positive number means used gas; negative number means refund.
    pub fn net_gas_usage(&self) -> i64 {
        self.gas_used() as i64 - self.storage_rebate as i64
    }
}

impl std::fmt::Display for GasCostSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "computation_cost: {}, ", self.computation_cost)?;
        write!(f, "storage_cost: {}, ", self.storage_cost)?;
        write!(f, "storage_rebate: {}, ", self.storage_rebate)?;
        write!(
            f,
            "non_refundable_storage_fee: {}",
            self.non_refundable_storage_fee
        )
    }
}
