use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use change_epoch::ChangeEpochExecutor;
use coin::CoinExecutor;
use object::ObjectExecutor;
use system::{ConsensusCommitExecutor, GenesisExecutor};
use types::{
    base::SomaAddress,
    committee::EpochId,
    digests::TransactionDigest,
    effects::{
        object_change::{EffectsObjectChange, IDOperation, ObjectIn, ObjectOut},
        ExecutionFailureStatus, ExecutionStatus, TransactionEffects,
    },
    error::{ExecutionError, ExecutionResult, SomaError, SomaResult},
    object::{ObjectID, Version},
    storage::object_store::ObjectStore,
    temporary_store::{InnerTemporaryStore, SharedInput, TemporaryStore},
    transaction::{InputObjectKind, InputObjects, ObjectReadResultKind, TransactionKind},
};
use validator::ValidatorExecutor;

mod change_epoch;
mod coin;
mod object;
mod system;
mod validator;

/// Core trait for all transaction executors
trait TransactionExecutor {
    fn execute(
        &self,
        store: &mut TemporaryStore,
        signer: SomaAddress,
        kind: TransactionKind,
        tx_digest: TransactionDigest,
        // gas_object_id: Option<ObjectID>,
    ) -> ExecutionResult<()>;
}

pub fn execute_transaction(
    epoch_id: EpochId,
    store: &dyn ObjectStore,
    tx_digest: TransactionDigest,
    kind: TransactionKind,
    signer: SomaAddress,
    input_objects: InputObjects,
) -> (
    InnerTemporaryStore,
    TransactionEffects,
    Option<ExecutionError>,
) {
    // Extract common information
    let shared_object_refs = input_objects.filter_shared_objects();
    let transaction_dependencies = input_objects.transaction_dependencies();

    // Check for shared objects with assigned versions
    let (has_assigned_shared_versions, shared_objects_to_load, shared_object_versions) =
        has_assigned_shared_versions(store, &input_objects);

    // Special handling for transactions with assigned shared versions
    if has_assigned_shared_versions {
        return handle_shared_object_transaction(
            store,
            tx_digest,
            kind.clone(),
            signer,
            input_objects,
            shared_objects_to_load,
            shared_object_versions,
            shared_object_refs,
            transaction_dependencies,
            epoch_id,
        );
    }

    // Standard execution path for all transactions (including ConsensusCommitPrologue)
    let mut temporary_store =
        TemporaryStore::new(input_objects, kind.receiving_objects(), tx_digest, epoch_id);

    // Create appropriate executor - this will create the ConsensusCommitExecutor for commit prologues
    let executor = create_executor(&kind);
    let is_epoch_change = kind.is_epoch_change();

    // Execute the transaction
    let result = executor.execute(&mut temporary_store, signer, kind, tx_digest);

    // Convert result to execution status and error
    let (execution_status, execution_error) = match result {
        Ok(()) => (ExecutionStatus::Success, None),
        Err(err) => (
            ExecutionStatus::Failure { error: err.clone() },
            Some(ExecutionError::new(err, None)),
        ),
    };

    // Get mutable input IDs for invariant checks
    let mutable_inputs = temporary_store.mutable_input_refs.keys().cloned().collect();

    // Check any required invariants
    if let Err(err) =
        temporary_store.check_ownership_invariants(&signer, &mutable_inputs, is_epoch_change)
    {
        let error_msg = format!("Ownership invariant violated: {}", err);
        return error_result(
            tx_digest,
            shared_object_refs,
            transaction_dependencies,
            error_msg,
            epoch_id,
        );
    }

    // Generate final effects
    let (inner, effects) = temporary_store.into_effects(
        shared_object_refs,
        &tx_digest,
        transaction_dependencies,
        execution_status,
        epoch_id,
    );

    (inner, effects, execution_error)
}

fn create_executor(kind: &TransactionKind) -> Box<dyn TransactionExecutor> {
    match kind {
        TransactionKind::AddValidator(_) | TransactionKind::RemoveValidator(_) => {
            Box::new(ValidatorExecutor::new())
        }

        TransactionKind::ChangeEpoch(_) => Box::new(ChangeEpochExecutor::new()),

        TransactionKind::Genesis(_) => Box::new(GenesisExecutor::new()),

        TransactionKind::ConsensusCommitPrologue(_) => Box::new(ConsensusCommitExecutor::new()),

        TransactionKind::TransferCoin { .. } | TransactionKind::PayCoins { .. } => {
            Box::new(CoinExecutor::new())
        }

        TransactionKind::TransferObjects { .. } => Box::new(ObjectExecutor::new()),
    }
}

/// Detects if a transaction has shared objects with assigned versions
fn has_assigned_shared_versions(
    store: &dyn ObjectStore,
    input_objects: &InputObjects,
) -> (bool, HashSet<ObjectID>, HashMap<ObjectID, Version>) {
    let mut shared_objects_to_load = HashSet::new();
    let mut has_assigned_shared_version_placeholders = false;
    let mut shared_object_versions = HashMap::new();

    // Scan input objects to find any with assigned versions
    for obj in input_objects.iter() {
        if let ObjectReadResultKind::CancelledTransactionSharedObject(version) = &obj.object {
            if !version.is_cancelled() {
                // This is a placeholder for an assigned shared version
                if let InputObjectKind::SharedObject {
                    id,
                    initial_shared_version,
                    ..
                } = obj.input_object_kind
                {
                    shared_objects_to_load.insert(id);
                    shared_object_versions.insert(id, *version);
                    has_assigned_shared_version_placeholders = true;
                }
            }
        }
    }

    (
        has_assigned_shared_version_placeholders,
        shared_objects_to_load,
        shared_object_versions,
    )
}

/// Loads shared objects with their assigned versions
fn load_shared_objects(
    store: &dyn ObjectStore,
    temporary_store: &mut TemporaryStore,
    shared_objects_to_load: &HashSet<ObjectID>,
) -> SomaResult<()> {
    for object_id in shared_objects_to_load {
        match store.get_object(object_id) {
            Ok(Some(object)) => {
                // Add to temporary store - directly overwriting the placeholder
                temporary_store.input_objects.insert(*object_id, object);
            }
            Ok(None) => {
                // Any required shared object not found is an error
                return Err(SomaError::from(format!(
                    "Required shared object {} not found in store",
                    object_id
                )));
            }
            Err(err) => {
                return Err(SomaError::from(format!(
                    "Failed to fetch shared object {}: {}",
                    object_id, err
                )));
            }
        }
    }

    Ok(())
}

/// Process a transaction with assigned shared object versions
fn handle_shared_object_transaction(
    store: &dyn ObjectStore,
    tx_digest: TransactionDigest,
    kind: TransactionKind,
    signer: SomaAddress,
    input_objects: InputObjects,
    shared_objects_to_load: HashSet<ObjectID>,
    shared_object_versions: HashMap<ObjectID, Version>,
    shared_object_refs: Vec<SharedInput>,
    transaction_dependencies: BTreeSet<TransactionDigest>,
    epoch_id: EpochId,
) -> (
    InnerTemporaryStore,
    TransactionEffects,
    Option<ExecutionError>,
) {
    // Create temporary store with the input objects
    let mut temporary_store =
        TemporaryStore::new(input_objects, kind.receiving_objects(), tx_digest, epoch_id);

    // Load all required shared objects
    if let Err(err) = load_shared_objects(store, &mut temporary_store, &shared_objects_to_load) {
        return error_result(
            tx_digest,
            shared_object_refs,
            transaction_dependencies,
            err.to_string(),
            epoch_id,
        );
    }

    // Create appropriate executor
    let executor = create_executor(&kind);

    // Execute the transaction
    let result = executor.execute(&mut temporary_store, signer, kind, tx_digest);

    // Convert result to execution status and error
    let (execution_status, execution_error) = match result {
        Ok(()) => (ExecutionStatus::Success, None),
        Err(err) => (
            ExecutionStatus::Failure { error: err.clone() },
            Some(ExecutionError::new(err, None)),
        ),
    };

    // Prepare to collect objects for the output
    let mut object_changes = BTreeMap::new();
    let mut written_objects = BTreeMap::new();
    let mut input_objects_map = BTreeMap::new();
    let mut mutable_inputs = BTreeMap::new();

    // Process all modified shared objects generically
    for object_id in &shared_objects_to_load {
        if let Some(obj) = temporary_store.read_object(object_id) {
            // Get target version from shared_object_versions
            let target_version = shared_object_versions
                .get(object_id)
                .cloned()
                .unwrap_or_else(|| obj.version().next());

            // Create a copy with the target version
            let mut final_obj = obj.clone();

            // Set version directly - bypassing increment
            final_obj.data.set_version_to(target_version);

            // Update the previous_transaction field
            final_obj.previous_transaction = tx_digest;

            // Add to results
            let id = final_obj.id();
            let input_version = obj.version();
            let input_digest = obj.digest();

            // Add to written objects
            written_objects.insert(id, final_obj.clone());

            // Add to input objects
            input_objects_map.insert(id, obj.clone());

            // Record mutable input if not immutable
            if !obj.owner().is_immutable() {
                mutable_inputs.insert(id, ((input_version, input_digest), obj.owner().clone()));
            }

            // Create effect object change
            let input_state = ObjectIn::Exist(((input_version, input_digest), obj.owner().clone()));
            let output_state =
                ObjectOut::ObjectWrite((final_obj.digest(), final_obj.owner().clone()));

            object_changes.insert(
                id,
                EffectsObjectChange {
                    input_state,
                    output_state,
                    id_operation: IDOperation::None,
                },
            );
        }
    }

    // Filter out shared objects that we've loaded
    let filtered_shared_refs = shared_object_refs
        .into_iter()
        .filter(|shared_input| match shared_input {
            SharedInput::Existing((id, _, _))
            | SharedInput::Deleted((id, _, _, _))
            | SharedInput::Cancelled((id, _)) => !shared_objects_to_load.contains(id),
        })
        .collect();

    // Create effects
    let effects = TransactionEffects::new(
        execution_status.clone(),
        epoch_id,
        filtered_shared_refs,
        tx_digest,
        Version::MAX, // Use MAX to avoid version conflicts
        object_changes,
        transaction_dependencies.into_iter().collect(),
    );

    // Create InnerTemporaryStore
    let inner_store = InnerTemporaryStore::new(
        input_objects_map,
        written_objects,
        mutable_inputs,
        Version::MAX,    // Use MAX as lamport_timestamp
        BTreeMap::new(), // No deleted shared objects
    );

    (inner_store, effects, execution_error)
}
// Helper function to generate error results consistently
fn error_result(
    tx_digest: TransactionDigest,
    shared_object_refs: Vec<SharedInput>,
    transaction_dependencies: BTreeSet<TransactionDigest>,
    error_msg: String,
    epoch_id: EpochId,
) -> (
    InnerTemporaryStore,
    TransactionEffects,
    Option<ExecutionError>,
) {
    let error = ExecutionFailureStatus::SomaError(SomaError::from(error_msg.clone()));
    let execution_error = Some(ExecutionError::new(error.clone(), None));

    let effects = TransactionEffects::new(
        ExecutionStatus::Failure { error },
        epoch_id,
        shared_object_refs,
        tx_digest,
        Version::MAX,
        BTreeMap::new(),
        transaction_dependencies.into_iter().collect(),
    );

    let inner_store = InnerTemporaryStore::new(
        BTreeMap::new(),
        BTreeMap::new(),
        BTreeMap::new(),
        Version::MAX,
        BTreeMap::new(),
    );

    (inner_store, effects, execution_error)
}
