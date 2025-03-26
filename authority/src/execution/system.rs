use types::{
    base::SomaAddress,
    digests::TransactionDigest,
    error::{SomaError, SomaResult},
    object::ObjectID,
    temporary_store::TemporaryStore,
    transaction::TransactionKind,
};

use super::TransactionExecutor;

/// Executor for Genesis transactions
pub struct GenesisExecutor;

impl GenesisExecutor {
    pub(crate) fn new() -> Self {
        Self {}
    }
}

impl TransactionExecutor for GenesisExecutor {
    fn execute(
        &self,
        store: &mut TemporaryStore,
        _signer: SomaAddress,
        kind: TransactionKind,
        tx_digest: TransactionDigest,
        // _gas_object_id: Option<ObjectID>,
    ) -> SomaResult<()> {
        if let TransactionKind::Genesis(genesis) = kind {
            // Create all genesis objects
            for object in genesis.objects {
                store.create_object(object.clone());
            }
            Ok(())
        } else {
            Err(SomaError::from(format!(
                "Invalid transaction type for validator executor"
            )))
        }
    }
}

/// Executor for consensus commit transactions
pub struct ConsensusCommitExecutor;

impl ConsensusCommitExecutor {
    pub(crate) fn new() -> Self {
        Self {}
    }
}

impl TransactionExecutor for ConsensusCommitExecutor {
    fn execute(
        &self,
        _store: &mut TemporaryStore,
        _signer: SomaAddress,
        _kind: TransactionKind,
        _tx_digest: TransactionDigest,
        // _gas_object_id: Option<ObjectID>,
    ) -> SomaResult<()> {
        // For consensus commit, we don't process any state changes, just return success
        Ok(())
    }
}
