use types::{
    base::SomaAddress,
    digests::TransactionDigest,
    effects::ExecutionFailureStatus,
    error::{ExecutionResult, SomaError, SomaResult},
    object::{Object, ObjectRef, Owner},
    temporary_store::TemporaryStore,
    transaction::TransactionKind,
};

use super::{FeeCalculator, TransactionExecutor};

/// Executor for object transfer transactions
pub struct ObjectExecutor;

impl ObjectExecutor {
    pub fn new() -> Self {
        Self {}
    }

    /// Execute a TransferObjects transaction
    fn execute_transfer_objects(
        &self,
        store: &mut TemporaryStore,
        object_refs: Vec<ObjectRef>,
        recipient: SomaAddress,
        signer: SomaAddress,
        tx_digest: TransactionDigest,
    ) -> ExecutionResult<()> {
        if object_refs.is_empty() {
            return Err(ExecutionFailureStatus::InvalidArguments {
                reason: "Must provide at least one object to transfer".to_string(),
            });
        }

        for object_ref in &object_refs {
            let object_id = object_ref.0;
            let object = store.read_object(&object_id).unwrap();
            check_ownership(&object, signer)?;

            // Update object ownership
            let mut updated_object = object.clone();
            updated_object.owner = Owner::AddressOwner(recipient);
            store.mutate_input_object(updated_object);
        }

        Ok(())
    }
}

impl TransactionExecutor for ObjectExecutor {
    fn execute(
        &self,
        store: &mut TemporaryStore,
        signer: SomaAddress,
        kind: TransactionKind,
        tx_digest: TransactionDigest,
    ) -> ExecutionResult<()> {
        match kind {
            TransactionKind::TransferObjects { objects, recipient } => {
                self.execute_transfer_objects(store, objects, recipient, signer, tx_digest)
            }
            _ => Err(ExecutionFailureStatus::InvalidTransactionType),
        }
    }
}

impl FeeCalculator for ObjectExecutor {}

/// Checks ownership of an object against the expected owner
pub(crate) fn check_ownership(
    object: &Object,
    expected_owner: SomaAddress,
) -> Result<(), ExecutionFailureStatus> {
    match object.owner().get_owner_address() {
        Ok(actual_owner) if actual_owner == expected_owner => Ok(()),
        Ok(actual_owner) => Err(ExecutionFailureStatus::InvalidOwnership {
            object_id: object.id(),
            expected_owner,
            actual_owner: Some(actual_owner),
        }),
        Err(_) => Err(ExecutionFailureStatus::InvalidOwnership {
            object_id: object.id(),
            expected_owner,
            actual_owner: None,
        }),
    }
}
