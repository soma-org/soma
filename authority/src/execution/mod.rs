// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use balance_transfer::BalanceTransferExecutor;
use bridge::BridgeExecutor;
use change_epoch::ChangeEpochExecutor;
use channel::ChannelExecutor;
// Stage 13b: CoinExecutor deleted along with the Transfer /
// MergeCoins tx kinds.
use object::ObjectExecutor;
use prepare_gas::{GasPreparationResult, prepare_gas};
use settlement::SettlementExecutor;
use staking::StakingExecutor;
use system::{ConsensusCommitExecutor, GenesisExecutor};
use tracing::info;
use types::base::SomaAddress;
use types::committee::EpochId;
use types::digests::TransactionDigest;
use types::effects::object_change::{EffectsObjectChange, IDOperation, ObjectIn, ObjectOut};
use types::effects::{ExecutionFailureStatus, ExecutionStatus, TransactionEffects};
use types::error::{ExecutionError, ExecutionResult, SomaError, SomaResult};
use types::execution::ExecutionOrEarlyError;
use types::object::{Object, ObjectID, ObjectRef, Version};
use types::storage::object_store::ObjectStore;
use types::system_state::FeeParameters;
use types::temporary_store::{self, InnerTemporaryStore, SharedInput, TemporaryStore};
use types::transaction::{
    CheckedInputObjects, InputObjectKind, InputObjects, ObjectReadResultKind, TransactionKind,
};
use types::tx_fee::TransactionFee;
use validator::ValidatorExecutor;

mod balance_transfer;
mod bridge;
mod change_epoch;
mod channel;
// Stage 13b: mod coin removed.
mod object;
mod prepare_gas;
mod settlement;
mod staking;
mod system;
mod validator;

/// Basis points for percentage calculations (10000 = 100%)
pub(crate) const BPS_DENOMINATOR: u64 = 10000;

// ---------------------------------------------------------------------------
// Checked arithmetic helpers – every arithmetic operation in execution flows
// through these to guarantee no silent overflow/underflow.
// ---------------------------------------------------------------------------

/// Safe BPS calculation using u128 intermediates to prevent overflow.
/// Returns `(amount * bps) / BPS_DENOMINATOR`.
pub(crate) fn bps_mul(amount: u64, bps: u64) -> u64 {
    ((amount as u128) * (bps as u128) / (BPS_DENOMINATOR as u128)) as u64
}

/// Checked addition returning `ArithmeticOverflow` on overflow.
pub(crate) fn checked_add(a: u64, b: u64) -> Result<u64, ExecutionFailureStatus> {
    a.checked_add(b).ok_or(ExecutionFailureStatus::ArithmeticOverflow)
}

/// Checked subtraction returning `ArithmeticOverflow` on underflow.
pub(crate) fn checked_sub(a: u64, b: u64) -> Result<u64, ExecutionFailureStatus> {
    a.checked_sub(b).ok_or(ExecutionFailureStatus::ArithmeticOverflow)
}

/// Checked multiplication returning `ArithmeticOverflow` on overflow.
pub(crate) fn checked_mul(a: u64, b: u64) -> Result<u64, ExecutionFailureStatus> {
    a.checked_mul(b).ok_or(ExecutionFailureStatus::ArithmeticOverflow)
}

/// Checked sum of an iterator of u64 values.
pub(crate) fn checked_sum<I: Iterator<Item = u64>>(
    mut iter: I,
) -> Result<u64, ExecutionFailureStatus> {
    iter.try_fold(0u64, |acc, x| checked_add(acc, x))
}

/// Core trait for all transaction executors.
///
/// Each executor reports its **fee units** for an op based on the op's actual
/// shape (e.g. `MergeCoins` returns `coins.len()`). The protocol-level
/// `unit_fee` is multiplied by these units to produce the total tx fee, which
/// is deducted up front in `prepare_gas`.
trait TransactionExecutor {
    /// Number of fee units this op should be charged for.
    /// Returning 0 means the op is gasless (system tx). The default is 1.
    fn fee_units(&self, _store: &TemporaryStore, _kind: &TransactionKind) -> u32 {
        1
    }

    fn execute(
        &mut self,
        store: &mut TemporaryStore,
        signer: SomaAddress,
        kind: TransactionKind,
        tx_digest: TransactionDigest,
    ) -> ExecutionResult<()>;
}

pub fn execute_transaction(
    epoch_id: EpochId,
    execution_version: u64,
    store: &dyn ObjectStore,
    tx_digest: TransactionDigest,
    kind: TransactionKind,
    signer: SomaAddress,
    gas_payment: Vec<ObjectRef>,
    input_objects: CheckedInputObjects,
    execution_params: ExecutionOrEarlyError,
    fee_parameters: FeeParameters,
    chain: protocol_config::Chain,
    // Stage 6c: pre-read sender USDC balance for balance-mode gas.
    // `Some(balance)` when `gas_payment.is_empty()` and the tx is not
    // a system tx; `None` otherwise. The caller is responsible for
    // reading the accumulator before calling.
    sender_usdc_balance: Option<u64>,
    // Stage 9d-C2: pre-read signer's delegation rows so the staking
    // executor can fold F1 rewards without reaching into the
    // perpetual store. Caller reads `iter_delegations_for_staker(signer)`
    // for AddStake/WithdrawStake and passes the (pool_id → Delegation)
    // map; empty for other tx kinds.
    prefetched_delegations: std::collections::BTreeMap<
        types::object::ObjectID,
        types::system_state::staking::Delegation,
    >,
    // Stage 14c.7: pre-loaded `Owner::Accumulator` objects for the
    // settlement system tx. The SettlementScheduler reads each
    // touched accumulator from the cache at dispatch time and passes
    // them through here so the settlement executor can mutate them
    // via `mutate_input_object` without going through the standard
    // InputObjectKind pipeline. Empty for every other tx kind.
    pre_loaded_accumulators: Vec<types::object::Object>,
) -> (InnerTemporaryStore, TransactionEffects, Option<ExecutionError>) {
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
        fee_parameters,
        execution_version,
        chain,
    );
    temporary_store.prefetched_delegations = prefetched_delegations;

    // SIP-58 single-path replay-safety: for the per-cp Settlement
    // system tx, load every BalanceAccumulator / DelegationAccumulator
    // object referenced in `settlement.changes` / `delegation_changes`
    // **directly from the canonical object store** (the `store: &dyn
    // ObjectStore` argument), not from `ExecutionEnv`. This makes
    // settlement effects deterministic across:
    //   - the original-execution path (driven by SettlementScheduler
    //     after the cp builder publishes the TX), AND
    //   - state-sync / checkpoint replay (which doesn't go through
    //     SettlementScheduler at all — `ExecutionEnv::pre_loaded_accumulators`
    //     is empty in that path).
    // For non-Settlement transactions the `pre_loaded_accumulators`
    // arg passes through unchanged.
    //
    // Settlement tx-data has no traditional inputs, so its
    // `lamport_timestamp` defaults to `Version::MIN + 1 = 1`. Fold
    // each accumulator's prior version into `lamport_timestamp` so
    // the settlement always lands at one above the highest input.
    let resolved_accumulators: Vec<types::object::Object> = match &kind {
        TransactionKind::Settlement(settlement) => {
            use types::accumulator::{BalanceAccumulator, DelegationAccumulator};
            let mut out: Vec<types::object::Object> = Vec::new();
            let mut seen: std::collections::HashSet<types::object::ObjectID> =
                std::collections::HashSet::new();
            for ev in &settlement.changes {
                let id = BalanceAccumulator::derive_id(ev.owner(), ev.coin_type());
                if seen.insert(id) {
                    if let Some(obj) = store.get_object(&id) {
                        out.push(obj);
                    }
                }
            }
            for de in &settlement.delegation_changes {
                let id = DelegationAccumulator::derive_id(de.pool_id, de.staker);
                if seen.insert(id) {
                    if let Some(obj) = store.get_object(&id) {
                        out.push(obj);
                    }
                }
            }
            out
        }
        TransactionKind::ChangeEpoch(_) => {
            // F1/F9 audit fix: ChangeEpoch credits validator-commission
            // rewards directly to each validator's `(pool, validator)`
            // delegation row. Pre-Stage-14d the credit landed only in
            // the `delegations` CF, leaving the `DelegationAccumulator`
            // object stale and the row's `last_collected_period`
            // unchanged. The executor now mutates the object directly
            // via `mutate_input_object` (so the object world stays in
            // sync with the CF), and advances `last_collected_period`
            // (so subsequent F1 reads don't over-pay rewards on the
            // commission for periods predating it). To enable that, we
            // pre-load every active+pending+inactive validator's
            // `DelegationAccumulator` from the canonical store here —
            // mirroring the Settlement pre-load above.
            use types::accumulator::DelegationAccumulator;
            use types::SYSTEM_STATE_OBJECT_ID;
            let mut out: Vec<types::object::Object> = Vec::new();
            let mut seen: std::collections::HashSet<types::object::ObjectID> =
                std::collections::HashSet::new();
            // SystemState lives in input_objects (declared as mutable
            // shared input by every ChangeEpoch tx). Find + deserialize
            // it to enumerate validators.
            if let Some(state_obj) = input_objects
                .iter_objects()
                .find(|o| o.id() == SYSTEM_STATE_OBJECT_ID)
            {
                if let Ok(state) = bcs::from_bytes::<types::system_state::SystemState>(
                    state_obj.as_inner().data.contents(),
                ) {
                    let mut visit = |pool_id: types::object::ObjectID,
                                     validator: types::base::SomaAddress| {
                        let id = DelegationAccumulator::derive_id(pool_id, validator);
                        if seen.insert(id) {
                            if let Some(obj) = store.get_object(&id) {
                                out.push(obj);
                            }
                        }
                    };
                    for v in &state.validators().validators {
                        visit(v.staking_pool.id, v.metadata.soma_address);
                    }
                    for v in &state.validators().pending_validators {
                        visit(v.staking_pool.id, v.metadata.soma_address);
                    }
                    for v in state.validators().inactive_validators.values() {
                        visit(v.staking_pool.id, v.metadata.soma_address);
                    }
                }
            }
            out
        }
        _ => pre_loaded_accumulators,
    };

    for acc in resolved_accumulators {
        let acc_version = acc.version();
        temporary_store.add_object_from_store(acc);
        if acc_version.value() >= temporary_store.lamport_timestamp.value() {
            temporary_store.lamport_timestamp =
                types::object::Version::lamport_increment([acc_version]);
        }
    }

    let mut executor = create_executor(&kind);

    // Phase 1: Gas preparation (validation, smashing, base fee deduction)
    let gas_result =
        match prepare_gas(
            &mut temporary_store,
            &kind,
            &signer,
            gas_payment,
            &*executor,
            sender_usdc_balance,
        ) {
            Ok(result) => result,
            Err((error_status, transaction_fee)) => {
                // Gas preparation failed.
                // If gas was actually deducted (fee > 0), the gas coin was modified
                // in the temporary store. Use into_effects() to preserve those mutations
                // so the object version advances and locks are released.
                // If fee is 0 (e.g., coin had 0 balance), no mutations occurred.
                let gas_object_id = temporary_store.gas_object_id;
                if gas_object_id.is_some() && transaction_fee.total_fee > 0 {
                    if temporary_store.execution_version >= 1 {
                        temporary_store.ensure_active_inputs_mutated();
                    }
                    let (inner, effects) = temporary_store.into_effects(
                        shared_object_refs,
                        &tx_digest,
                        transaction_dependencies,
                        ExecutionStatus::Failure { error: error_status.clone() },
                        epoch_id,
                        transaction_fee,
                        gas_object_id,
                    );
                    return (inner, effects, Some(ExecutionError::new(error_status, None)));
                }
                // smash_gas_coins itself failed — no objects modified, use error_result
                return error_result(
                    tx_digest,
                    shared_object_refs,
                    transaction_dependencies,
                    epoch_id,
                    transaction_fee,
                    None,
                    error_status,
                );
            }
        };

    // Phase 2: Check for early validation errors AFTER gas is prepared
    if let Err(early_error) = execution_params {
        // Simply use the temporary store's built-in conversion
        // The gas object changes from prepare_gas are already in the store
        if temporary_store.execution_version >= 1 {
            temporary_store.ensure_active_inputs_mutated();
        }
        let (inner, effects) = temporary_store.into_effects(
            shared_object_refs,
            &tx_digest,
            transaction_dependencies,
            ExecutionStatus::Failure { error: early_error.clone() },
            epoch_id,
            gas_result.transaction_fee.clone(),
            gas_result.primary_gas_id,
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
    let result = executor.execute(&mut temporary_store, signer, kind.clone(), tx_digest);

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

            (ExecutionStatus::Failure { error: err.clone() }, Some(ExecutionError::new(err, None)))
        }
    };

    // Fee was fully deducted up front in prepare_gas; this is what gets reported.
    let final_transaction_fee = gas_result.transaction_fee.clone();

    // Phase 4: Check ownership invariants if execution succeeded
    if execution_status.is_ok() {
        let is_epoch_change = kind.is_epoch_change();
        let mutable_inputs = temporary_store.get_mutable_input_ids();

        if let Err(err) =
            temporary_store.check_ownership_invariants(&signer, &mutable_inputs, is_epoch_change)
        {
            // Ownership invariant check failed, revert to post-gas changes state
            revert_non_gas_changes(
                &mut temporary_store,
                pre_execution_objects,
                pre_execution_deleted,
            );

            let error_msg = format!("Ownership invariant violated: {}", err);
            let error_status = ExecutionFailureStatus::SomaError(SomaError::from(error_msg));
            execution_status = ExecutionStatus::Failure { error: error_status.clone() };
            execution_error = Some(ExecutionError::new(error_status, None));
        }
    }

    // Before generating effects, ensure all mutable inputs are written
    // so their versions advance and epoch-store locks become stale.
    if temporary_store.execution_version >= 1 {
        temporary_store.ensure_active_inputs_mutated();
    }

    // Generate effects
    let (inner, effects) = temporary_store.into_effects(
        shared_object_refs,
        &tx_digest,
        transaction_dependencies,
        execution_status,
        epoch_id,
        final_transaction_fee,
        gas_result.primary_gas_id,
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

        // System transactions
        TransactionKind::ChangeEpoch(_) => Box::new(ChangeEpochExecutor::new()),
        TransactionKind::Genesis(_) => Box::new(GenesisExecutor::new()),
        TransactionKind::ConsensusCommitPrologueV1(_) => Box::new(ConsensusCommitExecutor::new()),

        // Object transactions (Stage 13b: Transfer / MergeCoins deleted)
        TransactionKind::TransferObjects { .. } => Box::new(ObjectExecutor::new()),

        // Staking transactions (validator only)
        TransactionKind::AddStake { .. } | TransactionKind::WithdrawStake { .. } => {
            Box::new(StakingExecutor::new())
        }

        // Bridge transactions
        TransactionKind::BridgeDeposit(_)
        | TransactionKind::BridgeWithdraw(_)
        | TransactionKind::BridgeEmergencyPause(_)
        | TransactionKind::BridgeEmergencyUnpause(_) => Box::new(BridgeExecutor::new()),

        // Payment-channel transactions
        TransactionKind::OpenChannel(_)
        | TransactionKind::Settle(_)
        | TransactionKind::RequestClose(_)
        | TransactionKind::WithdrawAfterTimeout(_)
        | TransactionKind::TopUp(_) => Box::new(ChannelExecutor::new()),

        // Per-commit balance settlement (Stage 3)
        TransactionKind::Settlement(_) => Box::new(SettlementExecutor::new()),

        // Balance-mode value transfer (Stage 7)
        TransactionKind::BalanceTransfer(_) => Box::new(BalanceTransferExecutor::new()),
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
                if let InputObjectKind::SharedObject { id, initial_shared_version, .. } =
                    obj.input_object_kind
                {
                    shared_objects_to_load.insert(id);
                    shared_object_versions.insert(id, *version);
                    has_assigned_shared_version_placeholders = true;
                }
            }
        }
    }

    (has_assigned_shared_version_placeholders, shared_objects_to_load, shared_object_versions)
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
) -> (InnerTemporaryStore, TransactionEffects, Option<ExecutionError>) {
    // Save pre-execution state for potential reversion
    let pre_execution_objects = temporary_store.execution_results.written_objects.clone();
    let pre_execution_deleted = temporary_store.execution_results.deleted_object_ids.clone();

    // Load all required shared objects
    if let Err(err) = load_shared_objects(store, &mut temporary_store, &shared_objects_to_load) {
        // Shared object loading failed - just return the current state with error
        let execution_status =
            ExecutionStatus::Failure { error: ExecutionFailureStatus::SomaError(err.clone()) };

        if temporary_store.execution_version >= 1 {
            temporary_store.ensure_active_inputs_mutated();
        }
        let (inner, effects) = temporary_store.into_effects(
            shared_object_refs,
            &tx_digest,
            transaction_dependencies,
            execution_status,
            epoch_id,
            gas_result.transaction_fee,
            gas_result.primary_gas_id,
        );

        return (
            inner,
            effects,
            Some(ExecutionError::new(ExecutionFailureStatus::SomaError(err), None)),
        );
    }

    // Create appropriate executor
    let mut executor = create_executor(&kind);

    // Execute the transaction
    let result = executor.execute(&mut temporary_store, signer, kind.clone(), tx_digest);

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

            if temporary_store.execution_version >= 1 {
                temporary_store.ensure_active_inputs_mutated();
            }
            let (inner, effects) = temporary_store.into_effects(
                shared_object_refs,
                &tx_digest,
                transaction_dependencies,
                execution_status,
                epoch_id,
                gas_result.transaction_fee,
                gas_result.primary_gas_id,
            );

            return (inner, effects, Some(ExecutionError::new(err, None)));
        }
    };

    // Fee was fully deducted up front in prepare_gas.
    let final_transaction_fee = gas_result.transaction_fee.clone();

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
                EffectsObjectChange { input_state, output_state, id_operation: IDOperation::None },
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
        final_obj.data.increment_version_to(temporary_store.lamport_timestamp);

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
                mutable_inputs
                    .insert(*id, ((input_version, input_digest), original.owner().clone()));
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

    // Create effects. This path is the legacy/object-only construction
    // (used by some test/synthetic flows that don't run through
    // TemporaryStore::into_effects). It emits no balance/delegation
    // events because no executor mutates those families on this path.
    let effects = TransactionEffects::new(
        execution_status.clone(),
        epoch_id,
        filtered_shared_refs,
        tx_digest,
        Version::MAX, // Use MAX to avoid version conflicts
        object_changes,
        transaction_dependencies.into_iter().collect(),
        final_transaction_fee, // Include transaction fee
        gas_result.primary_gas_id,
        Vec::new(),
        Vec::new(),
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
    transaction_fee: TransactionFee,
    gas_object: Option<ObjectID>,
    error_status: ExecutionFailureStatus,
) -> (InnerTemporaryStore, TransactionEffects, Option<ExecutionError>) {
    let execution_error = Some(ExecutionError::new(error_status.clone(), None));

    let effects = TransactionEffects::new(
        ExecutionStatus::Failure { error: error_status },
        epoch_id,
        shared_object_refs,
        tx_digest,
        Version::MAX,
        BTreeMap::new(),
        transaction_dependencies.into_iter().collect(),
        transaction_fee,
        gas_object,
        Vec::new(),
        Vec::new(),
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
