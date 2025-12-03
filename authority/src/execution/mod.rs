use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use change_epoch::ChangeEpochExecutor;
use coin::CoinExecutor;
use encoder::EncoderExecutor;
use object::ObjectExecutor;
use prepare_gas::{calculate_and_deduct_remaining_fees, prepare_gas, GasPreparationResult};
use shard::ShardExecutor;
use staking::StakingExecutor;
use system::{ConsensusCommitExecutor, GenesisExecutor};
use tracing::info;
use types::{
    base::SomaAddress,
    committee::EpochId,
    digests::TransactionDigest,
    effects::{
        object_change::{EffectsObjectChange, IDOperation, ObjectIn, ObjectOut},
        ExecutionFailureStatus, ExecutionStatus, TransactionEffects,
    },
    error::{ExecutionError, ExecutionResult, SomaError, SomaResult},
    execution::ExecutionOrEarlyError,
    object::{Object, ObjectID, ObjectRef, Version},
    storage::object_store::ObjectStore,
    temporary_store::{self, InnerTemporaryStore, SharedInput, TemporaryStore},
    transaction::{
        CheckedInputObjects, InputObjectKind, InputObjects, ObjectReadResultKind, TransactionKind,
    },
    tx_fee::TransactionFee,
};
use validator::ValidatorExecutor;

mod change_epoch;
mod coin;
mod encoder;
mod object;
mod prepare_gas;
mod shard;
mod staking;
mod system;
mod validator;

/// Core trait for all transaction executors
trait TransactionExecutor: FeeCalculator {
    fn execute(
        &mut self,
        store: &mut TemporaryStore,
        signer: SomaAddress,
        kind: TransactionKind,
        tx_digest: TransactionDigest,
        value_fee: u64,
    ) -> ExecutionResult<()>;
}

/// Trait for calculating transaction fees based on transaction type
pub trait FeeCalculator {
    /// Calculate the value-based fee for a transaction
    fn calculate_value_fee(&self, store: &TemporaryStore, kind: &TransactionKind) -> u64 {
        // Default value fee
        0
    }

    /// Get the base fee for transaction type
    fn base_fee(&self) -> u64 {
        // Default base fee
        1000
    }

    /// Calculate fee per object write
    fn write_fee_per_object(&self) -> u64 {
        // Default per-write fee
        300
    }

    /// Calculate operation fee for a given number of objects
    fn calculate_operation_fee(&self, num_objects: u64) -> u64 {
        num_objects * self.write_fee_per_object()
    }
}

pub fn execute_transaction(
    epoch_id: EpochId,
    store: &dyn ObjectStore,
    tx_digest: TransactionDigest,
    kind: TransactionKind,
    signer: SomaAddress,
    gas_payment: Vec<ObjectRef>,
    input_objects: CheckedInputObjects,
    execution_params: ExecutionOrEarlyError,
) -> (
    InnerTemporaryStore,
    TransactionEffects,
    Option<ExecutionError>,
) {
    let input_objects = input_objects.into_inner();
    // Extract common information
    let shared_object_refs = input_objects.filter_shared_objects();
    let transaction_dependencies = input_objects.transaction_dependencies();

    // Create temporary store for gas validation
    let mut temporary_store = TemporaryStore::new(
        input_objects.clone(),
        kind.receiving_objects(),
        tx_digest,
        epoch_id,
    );

    let mut executor = create_executor(&kind);

    // Phase 1: Gas preparation (validation, smashing, base fee deduction)
    let gas_result = match prepare_gas(
        &mut temporary_store,
        &kind,
        &signer,
        gas_payment,
        &*executor,
    ) {
        Ok(result) => result,
        Err((error_status, transaction_fee)) => {
            // Gas preparation failed, return with error
            return error_result(
                tx_digest,
                shared_object_refs,
                transaction_dependencies,
                epoch_id,
                transaction_fee,
                error_status,
            );
        }
    };

    // Phase 2: Check for early validation errors AFTER gas is prepared
    if let Err(early_error) = execution_params {
        // Simply use the temporary store's built-in conversion
        // The gas object changes from prepare_gas are already in the store
        let (inner, effects) = temporary_store.into_effects(
            shared_object_refs,
            &tx_digest,
            transaction_dependencies,
            ExecutionStatus::Failure {
                error: early_error.clone(),
            },
            epoch_id,
            gas_result.transaction_fee.clone(),
        );

        return (inner, effects, Some(ExecutionError::new(early_error, None)));
    }

    // Phase 3: Check for assigned shared versions and handle them specially
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
            temporary_store,
            gas_result,
        );
    }

    // Save a snapshot of current object state for potential reversion
    let pre_execution_objects = temporary_store.execution_results.written_objects.clone();
    let pre_execution_deleted = temporary_store.execution_results.deleted_object_ids.clone();

    // Execute the transaction
    let result = executor.execute(
        &mut temporary_store,
        signer,
        kind.clone(),
        tx_digest,
        gas_result.value_fee,
    );

    // Check execution status
    let (mut execution_status, mut execution_error) = match result {
        Ok(()) => (ExecutionStatus::Success, None),
        Err(err) => {
            // Execution failed, revert non-gas changes
            revert_non_gas_changes(
                &mut temporary_store,
                pre_execution_objects.clone(),
                pre_execution_deleted.clone(),
            );

            (
                ExecutionStatus::Failure { error: err.clone() },
                Some(ExecutionError::new(err, None)),
            )
        }
    };

    // Initialize the final transaction fee to what was deducted during gas preparation
    let mut final_transaction_fee = gas_result.transaction_fee.clone();

    // Phase 4: Calculate and deduct remaining transaction fee if execution succeeded
    if execution_status.is_ok() {
        match calculate_and_deduct_remaining_fees(
            &mut temporary_store,
            &kind,
            &*executor,
            &gas_result,
        ) {
            Ok(updated_fee) => {
                final_transaction_fee = updated_fee;

                // Phase 5: Check ownership invariants
                let is_epoch_change = kind.is_epoch_change();
                let mutable_inputs = temporary_store.get_mutable_input_ids();

                if let Err(err) = temporary_store.check_ownership_invariants(
                    &signer,
                    &mutable_inputs,
                    is_epoch_change,
                ) {
                    // Ownership invariant check failed, revert to post-gas changes state
                    revert_non_gas_changes(
                        &mut temporary_store,
                        pre_execution_objects,
                        pre_execution_deleted,
                    );

                    // Update execution status to failure
                    let error_msg = format!("Ownership invariant violated: {}", err);
                    let error_status =
                        ExecutionFailureStatus::SomaError(SomaError::from(error_msg));
                    execution_status = ExecutionStatus::Failure {
                        error: error_status.clone(),
                    };
                    execution_error = Some(ExecutionError::new(error_status, None));
                }
            }
            Err(err) => {
                // Execution succeeded but fee deduction failed, revert to post-gas changes state
                revert_non_gas_changes(
                    &mut temporary_store,
                    pre_execution_objects,
                    pre_execution_deleted,
                );

                // Update execution status to failure
                execution_status = ExecutionStatus::Failure { error: err.clone() };
                execution_error = Some(ExecutionError::new(err, None));
            }
        }
    }

    // Generate effects
    let (inner, effects) = temporary_store.into_effects(
        shared_object_refs,
        &tx_digest,
        transaction_dependencies,
        execution_status,
        epoch_id,
        final_transaction_fee,
    );

    (inner, effects, execution_error)
}

fn create_executor(kind: &TransactionKind) -> Box<dyn TransactionExecutor> {
    match kind {
        // Validator management transactions
        TransactionKind::AddValidator(_)
        | TransactionKind::RemoveValidator(_)
        | TransactionKind::ReportValidator { .. }
        | TransactionKind::UndoReportValidator { .. }
        | TransactionKind::SetCommissionRate { .. }
        | TransactionKind::UpdateValidatorMetadata(_) => Box::new(ValidatorExecutor::new()),

        // Encoder management transactions
        TransactionKind::AddEncoder(_)
        | TransactionKind::RemoveEncoder { .. }
        | TransactionKind::ReportEncoder { .. }
        | TransactionKind::UndoReportEncoder { .. }
        | TransactionKind::SetEncoderCommissionRate { .. }
        | TransactionKind::SetEncoderBytePrice { .. }
        | TransactionKind::UpdateEncoderMetadata(_) => Box::new(EncoderExecutor::new()),

        // System transactions
        TransactionKind::ChangeEpoch(_) => Box::new(ChangeEpochExecutor::new()),
        TransactionKind::Genesis(_) => Box::new(GenesisExecutor::new()),
        TransactionKind::ConsensusCommitPrologue(_) => Box::new(ConsensusCommitExecutor::new()),

        // Coin and object transactions
        TransactionKind::TransferCoin { .. } | TransactionKind::PayCoins { .. } => {
            Box::new(CoinExecutor::new())
        }
        TransactionKind::TransferObjects { .. } => Box::new(ObjectExecutor::new()),

        // Staking transactions - both validator and encoder staking
        TransactionKind::AddStake { .. }
        | TransactionKind::WithdrawStake { .. }
        | TransactionKind::AddStakeToEncoder { .. } => Box::new(StakingExecutor::new()),

        // Shard transactions
        TransactionKind::EmbedData { .. }
        | TransactionKind::ClaimEscrow { .. }
        | TransactionKind::ReportWinner { .. } => Box::new(ShardExecutor::new()),
    }
}

// Helper function to revert non-gas changes after failed execution
fn revert_non_gas_changes(
    store: &mut TemporaryStore,
    pre_execution_objects: BTreeMap<ObjectID, Object>,
    pre_execution_deleted: BTreeSet<ObjectID>,
) {
    // Replace written objects with our reverted state
    store.execution_results.written_objects = pre_execution_objects;
    store.execution_results.deleted_object_ids = pre_execution_deleted;
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
            Some(object) => {
                // Add to temporary store - directly overwriting the placeholder
                temporary_store.input_objects.insert(*object_id, object);
            }
            None => {
                // Any required shared object not found is an error
                return Err(SomaError::from(format!(
                    "Required shared object {} not found in store",
                    object_id
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
    mut temporary_store: TemporaryStore,
    gas_result: GasPreparationResult,
) -> (
    InnerTemporaryStore,
    TransactionEffects,
    Option<ExecutionError>,
) {
    // Save pre-execution state for potential reversion
    let pre_execution_objects = temporary_store.execution_results.written_objects.clone();
    let pre_execution_deleted = temporary_store.execution_results.deleted_object_ids.clone();

    // Load all required shared objects
    if let Err(err) = load_shared_objects(store, &mut temporary_store, &shared_objects_to_load) {
        // Shared object loading failed - just return the current state with error
        let execution_status = ExecutionStatus::Failure {
            error: ExecutionFailureStatus::SomaError(err.clone()),
        };

        let (inner, effects) = temporary_store.into_effects(
            shared_object_refs,
            &tx_digest,
            transaction_dependencies,
            execution_status,
            epoch_id,
            gas_result.transaction_fee,
        );

        return (
            inner,
            effects,
            Some(ExecutionError::new(
                ExecutionFailureStatus::SomaError(err),
                None,
            )),
        );
    }

    // Create appropriate executor
    let mut executor = create_executor(&kind);

    // Execute the transaction
    let result = executor.execute(
        &mut temporary_store,
        signer,
        kind.clone(),
        tx_digest,
        gas_result.value_fee,
    );

    // Convert result to execution status and error
    let (execution_status, execution_error) = match result {
        Ok(()) => (ExecutionStatus::Success, None),
        Err(err) => {
            // Execution failed, revert to post-gas state
            revert_non_gas_changes(
                &mut temporary_store,
                pre_execution_objects.clone(),
                pre_execution_deleted.clone(),
            );

            // Return with failure status but keep gas changes
            let execution_status = ExecutionStatus::Failure { error: err.clone() };

            let (inner, effects) = temporary_store.into_effects(
                shared_object_refs,
                &tx_digest,
                transaction_dependencies,
                execution_status,
                epoch_id,
                gas_result.transaction_fee,
            );

            return (inner, effects, Some(ExecutionError::new(err, None)));
        }
    };

    // Execution status should be success here

    // Initialize the final transaction fee to what was deducted during gas preparation
    let mut final_transaction_fee = gas_result.transaction_fee.clone();

    match calculate_and_deduct_remaining_fees(&mut temporary_store, &kind, &*executor, &gas_result)
    {
        Ok(updated_fee) => {
            final_transaction_fee = updated_fee;
        }
        Err(err) => {
            revert_non_gas_changes(
                &mut temporary_store,
                pre_execution_objects.clone(),
                pre_execution_deleted.clone(),
            );

            // Return with failure status but keep gas changes
            let execution_status = ExecutionStatus::Failure { error: err.clone() };

            let (inner, effects) = temporary_store.into_effects(
                shared_object_refs,
                &tx_digest,
                transaction_dependencies,
                execution_status,
                epoch_id,
                gas_result.transaction_fee, // Use original fee that was successfully deducted
            );

            return (inner, effects, Some(ExecutionError::new(err, None)));
        }
    }
    // Prepare to collect objects for the output
    let mut object_changes = BTreeMap::new();
    let mut written_objects = BTreeMap::new();
    let mut input_objects_map: BTreeMap<ObjectID, Object> = BTreeMap::new();
    let mut mutable_inputs = BTreeMap::new();

    // Process all objects in temporary_store's execution results

    // 1. First handle all shared objects specially
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

            let Some(original) = temporary_store.input_objects.get(&id) else {
                panic!("Shared object is not in inputs")
            };

            // Add to input objects map
            input_objects_map.insert(id.into(), original.clone());

            // Get original version and digest
            let input_version = original.version();
            let input_digest = original.digest();

            // Add to written objects
            written_objects.insert(id, final_obj.clone());

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

    // 2. Now add all non-shared objects (including gas objects)
    for (id, obj) in &temporary_store.execution_results.written_objects {
        // Skip objects we've already processed (shared objects)
        if shared_objects_to_load.contains(id) {
            continue;
        }

        // Create a copy with updated version and previous transaction
        let mut final_obj = obj.clone();

        // Set version to temporary_store's lamport_timestamp
        final_obj
            .data
            .increment_version_to(temporary_store.lamport_timestamp);

        // Update the previous_transaction field
        final_obj.previous_transaction = tx_digest;

        // Add to written objects
        written_objects.insert(*id, final_obj.clone());

        // Get original object if it exists
        if let Some(original) = temporary_store.input_objects.get(id) {
            // This was an existing object that was modified
            let input_version = original.version();
            let input_digest = original.digest();

            // Add to input objects map
            input_objects_map.insert(*id, original.clone());

            // Add to mutable inputs if not immutable
            if !original.owner().is_immutable() {
                mutable_inputs.insert(
                    *id,
                    ((input_version, input_digest), original.owner().clone()),
                );
            }

            // Create effect object change
            object_changes.insert(
                *id,
                EffectsObjectChange {
                    input_state: ObjectIn::Exist((
                        (input_version, input_digest),
                        original.owner().clone(),
                    )),
                    output_state: ObjectOut::ObjectWrite((
                        final_obj.digest(),
                        final_obj.owner().clone(),
                    )),
                    id_operation: IDOperation::None,
                },
            );
        } else {
            // This is a newly created object
            object_changes.insert(
                *id,
                EffectsObjectChange {
                    input_state: ObjectIn::NotExist,
                    output_state: ObjectOut::ObjectWrite((
                        final_obj.digest(),
                        final_obj.owner().clone(),
                    )),
                    id_operation: IDOperation::Created,
                },
            );
        }
    }

    // 3. Add deleted objects that aren't shared
    for id in &temporary_store.execution_results.deleted_object_ids {
        // Skip shared objects (should be handled above)
        if shared_objects_to_load.contains(id) {
            continue;
        }

        // Add deletion to object changes
        if let Some(original) = temporary_store.input_objects.get(id) {
            // Add to input objects map
            input_objects_map.insert(*id, original.clone());

            // Get original version and digest
            let input_version = original.version();
            let input_digest = original.digest();

            // Create effect object change for deletion
            object_changes.insert(
                *id,
                EffectsObjectChange {
                    input_state: ObjectIn::Exist((
                        (input_version, input_digest),
                        original.owner().clone(),
                    )),
                    output_state: ObjectOut::NotExist,
                    id_operation: IDOperation::Deleted,
                },
            );
        }
    }

    // Filter out shared objects that we've loaded from shared object refs
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
        final_transaction_fee, // Include transaction fee
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

// Helper function to generate error results consistently - only gets called in a case when gas object can't get read
fn error_result(
    tx_digest: TransactionDigest,
    shared_object_refs: Vec<SharedInput>,
    transaction_dependencies: BTreeSet<TransactionDigest>,
    epoch_id: EpochId,
    transaction_fee: Option<TransactionFee>,
    error_status: ExecutionFailureStatus,
) -> (
    InnerTemporaryStore,
    TransactionEffects,
    Option<ExecutionError>,
) {
    let execution_error = Some(ExecutionError::new(error_status.clone(), None));

    let effects = TransactionEffects::new(
        ExecutionStatus::Failure {
            error: error_status,
        },
        epoch_id,
        shared_object_refs,
        tx_digest,
        Version::MAX,
        BTreeMap::new(),
        transaction_dependencies.into_iter().collect(),
        transaction_fee,
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
