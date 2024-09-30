use std::sync::Arc;

use crate::{effects::TransactionEffects, transaction::VerifiedTransaction};

/// TransactionOutputs
pub struct TransactionOutputs {
    pub transaction: Arc<VerifiedTransaction>,
    pub effects: TransactionEffects,
}

impl TransactionOutputs {
    // Convert Effects into the exact set of updates to the store
    pub fn build_transaction_outputs(
        transaction: VerifiedTransaction,
        effects: TransactionEffects,
    ) -> TransactionOutputs {
        TransactionOutputs {
            transaction: Arc::new(transaction),
            effects,
        }
    }
}
