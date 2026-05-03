// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use serde::{Deserialize, Serialize};
use tracing::info;

use crate::balance::BalanceEvent;
use crate::base::{FullObjectID, SomaAddress};
use crate::committee::EpochId;
use crate::digests::{ObjectDigest, TransactionDigest};
use crate::effects::object_change::EffectsObjectChange;
use crate::effects::{ExecutionStatus, TransactionEffects, TransactionEffectsAPI};
use crate::error::SomaResult;
use crate::object::{Object, ObjectID, ObjectRef, Owner, Version, VersionDigest};
use crate::storage::InputKey;
use crate::system_state::FeeParameters;
use crate::transaction::InputObjects;
use crate::transaction_outputs::WrittenObjects;
use crate::tx_fee::TransactionFee;

/// # DeletedSharedObjectInfo
///
/// A type containing all of the information needed to work with a deleted shared object in
/// execution and when committing the execution effects of the transaction.
///
/// ## Components
/// 0. The object ID of the deleted shared object
/// 1. The version of the shared object
/// 2. Whether the object appeared as mutable (or owned) in the transaction, or as a read-only shared object
/// 3. The transaction digest of the previous transaction that used this shared object mutably or
///    took it by value
pub type DeletedSharedObjectInfo = (ObjectID, Version, bool, TransactionDigest);

/// # DeletedSharedObjects
///
/// A sequence of information about deleted shared objects in the transaction's inputs.
///
/// ## Purpose
/// Tracks all shared objects that were deleted during transaction execution,
/// which is important for maintaining object history and preventing double-spending.
pub type DeletedSharedObjects = Vec<DeletedSharedObjectInfo>;

/// # SharedInput
///
/// Represents different types of shared object inputs to a transaction.
///
/// ## Purpose
/// Tracks the state and access pattern of shared objects used in a transaction,
/// which is essential for proper sequencing and conflict detection.
///
/// ## Variants
/// - Existing: A shared object that exists and is being accessed
/// - Deleted: A shared object that has been deleted
/// - Cancelled: A shared object in a cancelled transaction
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SharedInput {
    /// A shared object that exists in storage
    Existing(ObjectRef),

    /// A shared object that has been deleted
    Deleted(DeletedSharedObjectInfo),

    /// A shared object in a cancelled transaction
    Cancelled((ObjectID, Version)),
}

/// # ExecutionResults
///
/// The primitive information that can be used to construct transaction effects.
///
/// ## Purpose
/// Collects all object changes that occurred during transaction execution,
/// providing the foundation for generating transaction effects.
///
/// ## Thread Safety
/// This structure is not thread-safe and should only be accessed from a single thread
/// during transaction execution.
#[derive(Debug, Default)]
pub struct ExecutionResults {
    /// All objects written regardless of whether they were mutated, created, or unwrapped
    pub written_objects: BTreeMap<ObjectID, Object>,

    /// All objects that existed prior to this transaction, and are modified in this transaction
    /// This includes any type of modification, including mutated, wrapped and deleted objects
    pub modified_objects: BTreeSet<ObjectID>,

    /// All object IDs created in this transaction
    pub created_object_ids: BTreeSet<ObjectID>,

    /// All object IDs deleted in this transaction
    /// No object ID should be in both created_object_ids and deleted_object_ids
    pub deleted_object_ids: BTreeSet<ObjectID>,
}

/// # DynamicallyLoadedObjectMetadata
///
/// Metadata for objects that are dynamically loaded during transaction execution.
///
/// ## Purpose
/// Tracks essential information about objects that are loaded during execution
/// but were not part of the initial transaction inputs.
///
/// ## Usage
/// Used to maintain consistency and track dependencies for objects that are
/// accessed dynamically during transaction execution.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct DynamicallyLoadedObjectMetadata {
    /// The version of the dynamically loaded object
    pub version: Version,

    /// The digest of the dynamically loaded object
    pub digest: ObjectDigest,

    /// The owner of the dynamically loaded object
    pub owner: Owner,

    /// The transaction that last modified this object
    pub previous_transaction: TransactionDigest,
}

impl ExecutionResults {
    /// Update version and previous transaction for all written objects.
    ///
    /// IMPORTANT: This method must be called for ALL execution paths
    /// to ensure consistent object state across validators and fullnodes.
    /// The previous_transaction field is critical for maintaining the state hash.
    pub fn update_version_and_previous_tx(
        &mut self,
        lamport_version: Version,
        prev_tx: TransactionDigest,
        input_objects: &BTreeMap<ObjectID, Object>,
    ) {
        info!(
            lamport_version = ?lamport_version,
            prev_tx = ?prev_tx,
            "Updating versions and previous_tx for {} objects",
            self.written_objects.len()
        );

        for (id, obj) in self.written_objects.iter_mut() {
            let old_previous_tx = obj.previous_transaction;
            // Update the version for the written object.
            obj.data.increment_version_to(lamport_version);

            // Record the version that the shared object was created at in its owner field.  Note,
            // this only works because shared objects must be created as shared (not created as
            // owned in one transaction and later converted to shared in another).
            if let Owner::Shared { initial_shared_version } = &mut obj.owner {
                if self.created_object_ids.contains(id) {
                    // Objects created with Version::new() (0) get their initial_shared_version
                    // set to the lamport timestamp. Objects created with OBJECT_START_VERSION (1)
                    // keep that value - this allows dynamically created shared objects (like
                    // Challenges) to have a predictable initial_shared_version for easy reference.
                    if *initial_shared_version == Version::new() {
                        *initial_shared_version = lamport_version;
                    } else {
                        debug_assert_eq!(
                            *initial_shared_version,
                            crate::object::OBJECT_START_VERSION,
                            "Newly created shared objects should have initial_shared_version \
                             of either Version::new() (to be set by lamport) or OBJECT_START_VERSION \
                             (to keep fixed), not {:?} for {id:?}",
                            initial_shared_version,
                        );
                    }
                }

                // Update initial_shared_version for reshared objects
                if let Some(Owner::Shared {
                    initial_shared_version: previous_initial_shared_version,
                }) = input_objects.get(id).map(|obj| &obj.owner)
                {
                    debug_assert!(!self.created_object_ids.contains(id));
                    debug_assert!(!self.deleted_object_ids.contains(id));
                    debug_assert!(
                        *initial_shared_version == Version::new()
                            || *initial_shared_version == *previous_initial_shared_version
                    );

                    *initial_shared_version = *previous_initial_shared_version;
                }
            }

            info!(
                object_id = ?id,
                old_previous_tx = ?old_previous_tx,
                new_previous_tx = ?prev_tx,
                "Setting previous_transaction on object"
            );

            obj.previous_transaction = prev_tx;
        }
    }
}

/// # TemporaryStore
///
/// A temporary storage mechanism for transaction execution that tracks object state changes.
///
/// ## Purpose
/// Provides a temporary view of object state during transaction execution,
/// tracking all modifications, creations, and deletions before they are
/// committed to permanent storage.
///
/// ## Lifecycle
/// 1. Created at the start of transaction execution with input objects
/// 2. Modified during execution as objects are created, modified, or deleted
/// 3. Used to generate transaction effects after execution completes
/// 4. Converted to InnerTemporaryStore for effects processing
///
/// ## Thread Safety
/// This structure is not thread-safe and should only be accessed from a single thread
/// during transaction execution.
pub struct TemporaryStore {
    /// The digest of the transaction being executed
    tx_digest: TransactionDigest,

    /// Objects that were inputs to the transaction
    pub input_objects: BTreeMap<ObjectID, Object>,

    /// The version to assign to all objects written by the transaction using this store
    pub lamport_timestamp: Version,

    /// Results of execution, including all object changes
    pub execution_results: ExecutionResults,

    /// Objects that were loaded during execution but were not inputs
    loaded_runtime_objects: BTreeMap<ObjectID, DynamicallyLoadedObjectMetadata>,

    /// Inputs that are mutable, with their original version, digest, and owner
    pub mutable_input_refs: BTreeMap<ObjectID, (VersionDigest, Owner)>,

    /// The set of objects that we may receive during execution
    /// Not guaranteed to receive all, or any of the objects referenced in this set
    receiving_objects: Vec<ObjectRef>,

    /// Consensus objects that were deleted in previous transactions
    deleted_consensus_objects: BTreeMap<ObjectID, Version /* start_version */>,

    creation_counter: u64,

    /// The current epoch
    cur_epoch: EpochId,

    pub gas_object_id: Option<ObjectID>,

    /// Fee parameters for this epoch
    pub fee_parameters: FeeParameters,

    /// Execution version for this epoch (from ProtocolConfig::execution_version).
    /// Used by executors to gate behavior changes across protocol upgrades.
    pub execution_version: u64,

    /// Which chain we are running on. Used by executors that need chain-specific
    /// protocol config (e.g. ChangeEpochExecutor).
    pub chain: protocol_config::Chain,

    /// Balance accumulator events emitted by executors during this
    /// transaction. Each `Deposit`/`Withdraw` is an *intent* — it does
    /// not touch the `accumulator_balances` column family directly.
    /// The settlement system transaction (Stage 3) drains these from
    /// every tx in a consensus commit, aggregates per `(owner, coin_type)`,
    /// and applies the net delta atomically.
    ///
    /// Order is preserved (insertion order). Stage 1a's `aggregate_events`
    /// is order-independent for the *summed delta*, so settlement can
    /// reorder freely; we keep insertion order here purely for debugging
    /// and for the audit trail in transaction effects.
    balance_events: Vec<BalanceEvent>,

    /// Stage 9d-C1: F1-shaped delegation events emitted by executors
    /// during this transaction. Applied to the `delegations` column
    /// family by `write_one_transaction_outputs` atomically with the
    /// rest of the commit's outputs.
    delegation_events: Vec<DelegationEvent>,

    /// Stage 9d-C2: signer's delegation rows pre-fetched before
    /// execution so the staking executor can compute F1
    /// fold-to-balance without reaching into the perpetual store. The
    /// caller (authority.rs) reads `iter_delegations_for_staker(signer)`
    /// for AddStake/WithdrawStake and passes the result here. Empty
    /// for txs that don't need delegation reads.
    pub prefetched_delegations: BTreeMap<ObjectID, crate::system_state::staking::Delegation>,

    /// Stage 14c.2: per-tx accumulator delta records. Aggregated by
    /// canonical accumulator ID — multiple emissions to the same
    /// accumulator within a single tx (e.g. gas + transfer Withdraw
    /// against the same sender's USDC) are netted into a single
    /// `(operation, magnitude)` record at emission time, so each
    /// accumulator ID appears at most once per tx in the resulting
    /// `effects.changed_objects`.
    ///
    /// At `into_effects` time, these are folded into `changed_objects`
    /// alongside regular object writes — they ride
    /// `EffectsObjectChange { input: NotExist, output:
    /// AccumulatorWriteV1(...), id_op: None }`.
    accumulator_writes:
        BTreeMap<ObjectID, crate::effects::object_change::AccumulatorWriteV1>,
}

/// An event recorded during execution that describes a change to the
/// (pool, staker) row in the `delegations` column family. Stage 9d-C1
/// reshapes the event around the F1 row schema:
///
/// - `delta`: signed principal change (positive for AddStake / reward
///   credit, negative for WithdrawStake).
/// - `set_period`: when present, sets `last_collected_period` to the
///   given F1 period. AddStake / WithdrawStake supply this so the
///   delegator's collected-up-to mark advances atomically with the
///   principal change. Reward dual-write callers that don't fold (eg.
///   epoch validator-commission credit) leave it `None` to avoid
///   silently advancing the period for unfolded delegations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct DelegationEvent {
    pub pool_id: ObjectID,
    pub staker: SomaAddress,
    pub delta: i128,
    pub set_period: Option<u64>,
}

impl TemporaryStore {
    pub fn new(
        input_objects: InputObjects,
        receiving_objects: Vec<ObjectRef>,
        tx_digest: TransactionDigest,
        cur_epoch: EpochId,
        fee_parameters: FeeParameters,
        execution_version: u64,
        chain: protocol_config::Chain,
    ) -> Self {
        let mutable_input_refs = input_objects.mutable_inputs();
        let lamport_timestamp = input_objects.lamport_timestamp(&receiving_objects);
        let deleted_consensus_objects = input_objects.deleted_consensus_objects();
        let objects = input_objects.into_object_map();
        #[cfg(debug_assertions)]
        {
            // Ensure that input objects and receiving objects must not overlap.
            assert!(
                objects
                    .keys()
                    .collect::<HashSet<_>>()
                    .intersection(
                        &receiving_objects.iter().map(|oref| &oref.0).collect::<HashSet<_>>()
                    )
                    .next()
                    .is_none()
            );
        }

        Self {
            tx_digest,
            input_objects: objects,
            lamport_timestamp,
            execution_results: ExecutionResults::default(),
            receiving_objects,
            cur_epoch,
            loaded_runtime_objects: BTreeMap::new(),
            mutable_input_refs,
            deleted_consensus_objects,
            gas_object_id: None,
            creation_counter: 0,
            fee_parameters,
            execution_version,
            chain,
            balance_events: Vec::new(),
            delegation_events: Vec::new(),
            prefetched_delegations: BTreeMap::new(),
            accumulator_writes: BTreeMap::new(),
        }
    }

    /// Record a balance accumulator event from this transaction.
    /// Called by executors that move fungibles via the address-balance
    /// model (Stage 6+: gas, transfers, channels, staking). Idempotency
    /// and ordering guarantees: events are deduplicated downstream by
    /// `aggregate_events`, so callers may emit independent events freely.
    pub fn emit_balance_event(&mut self, event: BalanceEvent) {
        self.balance_events.push(event);
    }

    /// Read-only view of balance events emitted so far. Useful for tests
    /// and for executors that need to introspect their own emissions.
    pub fn balance_events(&self) -> &[BalanceEvent] {
        &self.balance_events
    }

    /// Stage 9d-C1: record a F1-shaped delegation event. Positive
    /// `delta` for AddStake / reward credit, negative for
    /// WithdrawStake. `set_period` advances the row's
    /// `last_collected_period` mark atomically with the principal
    /// change — set it on AddStake / WithdrawStake so the F1 fold
    /// is recorded on disk; leave it `None` for dual-write callers
    /// (eg. epoch reward credit) that don't perform a fold.
    pub fn emit_delegation_event(
        &mut self,
        pool_id: ObjectID,
        staker: SomaAddress,
        delta: i128,
        set_period: Option<u64>,
    ) {
        self.delegation_events.push(DelegationEvent {
            pool_id,
            staker,
            delta,
            set_period,
        });
    }

    /// Read-only view of delegation events emitted so far.
    pub fn delegation_events(&self) -> &[DelegationEvent] {
        &self.delegation_events
    }

    /// Stage 14c.2: emit a per-tx accumulator delta record (Sui
    /// SIP-58 style). The executor calls this when the tx semantically
    /// deposits to or withdraws from an accumulator (typically via
    /// `BalanceAccumulator::derive_id(owner, coin_type)` for balance
    /// accumulators). Multiple emissions to the same accumulator ID
    /// within a single tx are netted into a single record at emission
    /// time — the net delta is what ends up on
    /// `effects.changed_objects`.
    ///
    /// Stage 14c.7 architecture: user-tx executors emit ONLY via
    /// `emit_accumulator_event` (no more `emit_balance_event` calls
    /// from balance-touching executors). The two stores are driven
    /// by:
    ///   * **CF** — per-tx drain in
    ///     `authority_store::write_one_transaction_outputs` converts
    ///     each `effects.accumulator_events()` entry to a
    ///     `BalanceEvent` and applies it via `apply_settlement_events`.
    ///   * **Object world** — `SettlementScheduler` aggregates the
    ///     same `accumulator_events` per-cp and dispatches a
    ///     settlement system tx that mutates the corresponding
    ///     `BalanceAccumulator` objects (state-hash coverage).
    /// The two paths process equivalent inputs at different
    /// aggregation cardinalities and produce numerically identical
    /// state.
    pub fn emit_accumulator_event(
        &mut self,
        address: crate::effects::object_change::AccumulatorAddress,
        operation: crate::effects::object_change::AccumulatorOperation,
        magnitude: u64,
    ) {
        use crate::effects::object_change::{
            AccumulatorOperation, AccumulatorValue, AccumulatorWriteV1,
        };

        let acc_id = address.object_id();

        // Aggregate against any existing record for this accumulator
        // in this tx. Net the deltas via i128 (the same domain
        // `aggregate_events` uses), then collapse back to (operation,
        // magnitude). A net of zero leaves the entry behind as a
        // no-op rather than removing it — downstream consumers
        // tolerate zero-magnitude records.
        let new_delta = match operation {
            AccumulatorOperation::Merge => magnitude as i128,
            AccumulatorOperation::Split => -(magnitude as i128),
        };
        let net = match self.accumulator_writes.get(&acc_id) {
            Some(prior) => prior.signed_delta() + new_delta,
            None => new_delta,
        };

        let (op, mag) = if net >= 0 {
            (AccumulatorOperation::Merge, net as u64)
        } else {
            // unsigned_abs() avoids overflow at i128::MIN.
            (AccumulatorOperation::Split, net.unsigned_abs() as u64)
        };

        self.accumulator_writes.insert(
            acc_id,
            AccumulatorWriteV1 { address, operation: op, value: AccumulatorValue::U64(mag) },
        );
    }

    /// Read-only view of accumulator writes emitted so far. Test
    /// helpers and executor invariant checks consult this directly;
    /// production consumers read `effects.accumulator_events()`.
    pub fn accumulator_writes(
        &self,
    ) -> &BTreeMap<ObjectID, crate::effects::object_change::AccumulatorWriteV1> {
        &self.accumulator_writes
    }

    pub fn update_object_version_and_prev_tx(&mut self) {
        self.execution_results.update_version_and_previous_tx(
            self.lamport_timestamp,
            self.tx_digest,
            &self.input_objects,
        );
    }

    /// Get the next creation number and increment the counter
    pub fn next_creation_num(&mut self) -> u64 {
        let num = self.creation_counter;
        self.creation_counter += 1;
        num
    }

    /// Given an object ID, if it's not modified, returns None.
    fn get_object_modified_at(
        &self,
        object_id: &ObjectID,
    ) -> Option<DynamicallyLoadedObjectMetadata> {
        if self.execution_results.modified_objects.contains(object_id) {
            self.mutable_input_refs
                .get(object_id)
                .map(|((version, digest), owner)| DynamicallyLoadedObjectMetadata {
                    version: *version,
                    digest: *digest,
                    owner: owner.clone(),
                    previous_transaction: self.input_objects[object_id].previous_transaction,
                })
                .or_else(|| self.loaded_runtime_objects.get(object_id).cloned())
            // if let Some(obj) = self.input_objects.get(object_id) {
            //     return Some(obj.clone());
            // }
            // None
        } else {
            None
        }
    }

    pub fn get_object_changes(&self) -> BTreeMap<ObjectID, EffectsObjectChange> {
        use crate::effects::object_change::{IDOperation, ObjectIn, ObjectOut};

        let results = &self.execution_results;
        let all_ids = results
            .created_object_ids
            .iter()
            .chain(&results.deleted_object_ids)
            .chain(&results.modified_objects)
            .chain(results.written_objects.keys())
            .collect::<BTreeSet<_>>();
        let mut changes: BTreeMap<ObjectID, EffectsObjectChange> = all_ids
            .into_iter()
            .map(|id| {
                (
                    *id,
                    EffectsObjectChange::new(
                        self.get_object_modified_at(id)
                            .map(|metadata| ((metadata.version, metadata.digest), metadata.owner)),
                        results.written_objects.get(id),
                        results.created_object_ids.contains(id),
                        results.deleted_object_ids.contains(id),
                    ),
                )
            })
            .collect();

        // Stage 14c.2: fold accumulator writes into the same
        // `changed_objects` map so they ride the standard effects
        // pipeline. The accumulator's canonical ObjectID never
        // collides with a regular object ID (deterministic
        // derivation under a domain-separation tag), so insertion
        // here cannot overwrite a pre-existing entry — an executor
        // that mutates the accumulator object directly AND emits an
        // AccumulatorWriteV1 for it would be a programming error.
        for (acc_id, write) in &self.accumulator_writes {
            debug_assert!(
                !changes.contains_key(acc_id),
                "Stage 14c.2: accumulator write at {acc_id:?} collides with a regular \
                 object change in the same tx — executors must emit AccumulatorWriteV1 \
                 OR mutate the accumulator object, not both."
            );
            changes.insert(
                *acc_id,
                EffectsObjectChange {
                    input_state: ObjectIn::NotExist,
                    output_state: ObjectOut::AccumulatorWriteV1(*write),
                    id_operation: IDOperation::None,
                },
            );
        }

        changes
    }

    #[allow(clippy::too_many_arguments)]
    pub fn into_effects(
        mut self,
        shared_object_refs: Vec<SharedInput>,
        transaction_digest: &TransactionDigest,
        mut transaction_dependencies: BTreeSet<TransactionDigest>,
        status: ExecutionStatus,
        epoch: EpochId,
        fee: TransactionFee,
        gas_object_id: Option<ObjectID>,
    ) -> (InnerTemporaryStore, TransactionEffects) {
        self.ensure_mutable_shared_objects_written();

        self.update_object_version_and_prev_tx();

        // Regardless of execution status (including aborts), we insert the previous transaction
        // for any successfully received objects during the transaction.
        for (id, expected_version, expected_digest) in &self.receiving_objects {
            // If the receiving object is in the loaded runtime objects, then that means that it
            // was actually successfully loaded (so existed, and there was authenticated mutable
            // access to it). So we insert the previous transaction as a dependency.
            if let Some(obj_meta) = self.loaded_runtime_objects.get(id) {
                // Check that the expected version, digest, and owner match the loaded version,
                // digest, and owner. If they don't then don't register a dependency.
                // This is because this could be "spoofed" by loading a dynamic object field.
                let loaded_via_receive = obj_meta.version == *expected_version
                    && obj_meta.digest == *expected_digest
                    && obj_meta.owner.is_address_owned();
                if loaded_via_receive {
                    transaction_dependencies.insert(obj_meta.previous_transaction);
                }
            }
        }

        let object_changes = self.get_object_changes();
        let lamport_version = self.lamport_timestamp;

        // Drain accumulator events out of `self` BEFORE handing the
        // remaining state to `into_inner`. Effects becomes the single
        // source of truth — the perpetual store reads them off the
        // effects struct, indexers attribute by tx digest off the same
        // struct, and InnerTemporaryStore stops carrying its own copy.
        let balance_events = std::mem::take(&mut self.balance_events);
        let delegation_events = std::mem::take(&mut self.delegation_events);

        // Create the inner temporary store to return
        let inner = self.into_inner();

        let effects = TransactionEffects::new(
            status,
            epoch,
            shared_object_refs,
            *transaction_digest,
            lamport_version,
            object_changes,
            transaction_dependencies.into_iter().collect(),
            fee,
            gas_object_id,
            balance_events,
            delegation_events,
        );

        (inner, effects)
    }

    /// Crate a new objcet. This is used to create objects outside of PT execution.
    pub fn create_object(&mut self, object: Object) {
        // Created mutable objects' versions are set to the store's lamport timestamp when it is
        // committed to effects. Creating an object at a non-zero version risks violating the
        // lamport timestamp invariant (that a transaction's lamport timestamp is strictly greater
        // than all versions witnessed by the transaction).
        debug_assert!(
            object.version() == Version::MIN,
            "Created mutable objects should not have a version set",
        );
        let id = object.id();
        self.execution_results.created_object_ids.insert(id);
        self.execution_results.written_objects.insert(id, object);
    }

    /// Delete a mutable input object. This is used to delete input objects outside of PT execution.
    pub fn delete_input_object(&mut self, id: &ObjectID) {
        // there should be no deletion after write
        // debug_assert!(!self.execution_results.written_objects.contains_key(id));
        debug_assert!(self.input_objects.contains_key(id));
        self.execution_results.modified_objects.insert(*id);
        self.execution_results.deleted_object_ids.insert(*id);
        self.execution_results.written_objects.remove(id);
    }

    pub fn is_deleted(&self, id: &ObjectID) -> bool {
        self.execution_results.deleted_object_ids.contains(id)
    }

    pub fn read_object(&self, id: &ObjectID) -> Option<Object> {
        // there should be no read after delete
        debug_assert!(!self.execution_results.deleted_object_ids.contains(id));
        self.execution_results
            .written_objects
            .get(id)
            .cloned()
            .or_else(|| self.input_objects.get(id).cloned())
    }

    /// Read the current consensus-agreed wall-clock timestamp from the
    /// Clock object. The transaction must have declared
    /// [`crate::transaction::SharedInputObject::CLOCK_OBJ_READ`] (or
    /// `CLOCK_OBJ_MUT` for the prologue) in `shared_input_objects` so the
    /// scheduler loaded Clock into the input set. Returns `None` if Clock
    /// was not declared as an input.
    pub fn read_clock_timestamp_ms(&self) -> Option<crate::base::TimestampMs> {
        self.read_object(&crate::CLOCK_OBJECT_ID)
            .as_ref()
            .and_then(Object::as_clock)
            .map(|c| c.timestamp_ms)
    }

    pub fn mutate_input_object(&mut self, object: Object) {
        let id = object.id();
        debug_assert!(self.input_objects.contains_key(&id));
        debug_assert!(!self.is_deleted(&id));

        self.execution_results.modified_objects.insert(id);
        self.execution_results.written_objects.insert(id, object);
    }

    /// Ensure all mutable input objects appear in `written_objects`, even if they
    /// were not modified by execution. This guarantees that their versions are
    /// bumped by `update_object_version_and_prev_tx`, which makes the old
    /// epoch-store locks stale and prevents permanent `ObjectLockConflict` errors
    /// after failed transactions.
    ///
    /// Ported from Sui's `TemporaryStore::ensure_active_inputs_mutated`.
    pub fn ensure_active_inputs_mutated(&mut self) {
        let mut to_be_updated = vec![];
        for id in self.mutable_input_refs.keys() {
            if !self.execution_results.modified_objects.contains(id) {
                to_be_updated.push(self.input_objects[id].clone());
            }
        }
        for object in to_be_updated {
            self.mutate_input_object(object);
        }
    }

    // check that every object read is owned directly or indirectly by sender, sponsor,
    // or a shared object input
    pub fn check_ownership_invariants(
        &self,
        sender: &SomaAddress,
        mutable_inputs: &HashSet<ObjectID>,
        is_epoch_change: bool,
    ) -> SomaResult<()> {
        // mark input objects as authenticated
        let mut authenticated_for_mutation: HashSet<_> = self
            .input_objects
            .iter()
            .filter_map(|(id, obj)| {
                match &obj.owner {
                    Owner::AddressOwner(a) => {
                        assert!(sender == a, "Input object must be owned by sender");
                        Some(id)
                    }
                    Owner::Shared { .. } => Some(id),
                    Owner::Immutable => {
                        // object is authenticated, but it cannot own other objects,
                        // so we should not add it to `authenticated_objs`
                        // However, we would definitely want to add immutable objects
                        // to the set of authenticated roots if we were doing runtime
                        // checks inside the VM instead of after-the-fact in the temporary
                        // store. Here, we choose not to add them because this will catch a
                        // bug where we mutate or delete an object that belongs to an immutable
                        // object (though it will show up somewhat opaquely as an authentication
                        // failure), whereas adding the immutable object to the roots will prevent
                        // us from catching this.
                        None
                    }
                    Owner::Accumulator { .. } => {
                        // Stage 14a: accumulator objects are authenticated by the
                        // privileged executor that mutates them (Settlement,
                        // staking, ChangeEpoch). They appear as inputs only to
                        // those system transactions, never to user transactions,
                        // so the sender-ownership check above does not apply.
                        // Treat them like Shared inputs for the purpose of
                        // authenticated mutation in the temporary store.
                        Some(id)
                    }
                }
            })
            .filter(|id| {
                // remove any non-mutable inputs. This will remove deleted or readonly shared
                // objects
                mutable_inputs.contains(id)
            })
            .copied()
            .collect();

        // check all modified objects are authenticated (excluding gas objects)
        let mut objects_to_authenticate =
            self.execution_results.modified_objects.iter().copied().collect::<Vec<_>>();
        // Map from an ObjectID to the ObjectID that covers it.
        while let Some(to_authenticate) = objects_to_authenticate.pop() {
            if authenticated_for_mutation.contains(&to_authenticate) {
                // object has been authenticated
                continue;
            }

            // we now assume the object is authenticated
            authenticated_for_mutation.insert(to_authenticate);
        }
        Ok(())
    }

    pub fn save_loaded_runtime_objects(
        &mut self,
        loaded_runtime_objects: BTreeMap<ObjectID, DynamicallyLoadedObjectMetadata>,
    ) {
        #[cfg(debug_assertions)]
        {
            for (id, v1) in &loaded_runtime_objects {
                if let Some(v2) = self.loaded_runtime_objects.get(id) {
                    assert_eq!(v1, v2);
                }
            }
            for (id, v1) in &self.loaded_runtime_objects {
                if let Some(v2) = loaded_runtime_objects.get(id) {
                    assert_eq!(v1, v2);
                }
            }
        }
        // Merge the two maps because we may be calling the execution engine more than once
        // (e.g. in advance epoch transaction, where we may be publishing a new system package).
        self.loaded_runtime_objects.extend(loaded_runtime_objects);
    }

    /// Break up the structure and return its internal stores for the transaction effects
    pub fn into_inner(self) -> InnerTemporaryStore {
        InnerTemporaryStore::new(
            self.input_objects,
            self.execution_results.written_objects,
            self.mutable_input_refs,
            self.lamport_timestamp,
            self.deleted_consensus_objects,
        )
    }

    // Add an object from the object store to the temporary store
    /// This is used to add actual shared objects that correspond to placeholder versions
    pub fn add_object_from_store(&mut self, object: Object) {
        let id = object.id();

        // Create a modified object with version set to MINIMUM_VERSION
        // to ensure that later incrementing to the lamport timestamp will work
        let mut modified_object = object.clone();

        // Reset the object's version to Version::MIN so it can be properly
        // incremented later in update_object_version_and_prev_tx
        if modified_object.version() > Version::MIN {
            // Save the current version and digest for the mutable_input_refs
            let original_version = modified_object.version();
            let original_digest = modified_object.digest();

            // Reset the version to MIN so it can be incremented without panicking
            modified_object.data.set_version_to(Version::MIN);

            // Add to input objects with the reset version
            self.input_objects.insert(id, modified_object);

            // If this object is at all mutable, we should track it in mutable inputs
            // using the ORIGINAL version and digest
            if !matches!(object.owner, Owner::Immutable) {
                let version_digest = (original_version, original_digest);
                self.mutable_input_refs.insert(id, (version_digest, object.owner().clone()));
            }
        } else {
            // For objects already at Version::MIN, we can just add them as is
            self.input_objects.insert(id, modified_object.clone());

            // If this object is at all mutable, we should track it in mutable inputs
            if !matches!(object.owner, Owner::Immutable) {
                let version_digest = (modified_object.version(), modified_object.digest());
                self.mutable_input_refs.insert(id, (version_digest, object.owner().clone()));
            }
        }
    }

    /// Helper method to get the set of mutable input object IDs
    pub fn get_mutable_input_ids(&self) -> HashSet<ObjectID> {
        self.mutable_input_refs.keys().cloned().collect()
    }

    /// Ensures all mutable shared objects are included in written_objects
    /// so they get the new Lamport version, even if not explicitly modified.
    /// This is required because the shared object version manager always advances
    /// the next version for shared objects based on Lamport timestamps.
    fn ensure_mutable_shared_objects_written(&mut self) {
        for id in self.mutable_input_refs.keys() {
            if let Some(obj) = self.input_objects.get(id) {
                // Only handle shared objects - owned objects don't have this issue
                if matches!(obj.owner, Owner::Shared { .. }) {
                    // If not already written and not deleted, add to written objects
                    if !self.execution_results.written_objects.contains_key(id)
                        && !self.execution_results.deleted_object_ids.contains(id)
                    {
                        self.execution_results.written_objects.insert(*id, obj.clone());
                        self.execution_results.modified_objects.insert(*id);
                    }
                }
            }
        }
    }
}

/// # InnerTemporaryStore
///
/// A structure to hold the data extracted from TemporaryStore for effects processing.
///
/// ## Purpose
/// Contains all the information needed for effects processing after transaction execution,
/// in a more compact form than the full TemporaryStore.
///
/// ## Usage
/// This is returned from execute_transaction and used during effects processing
/// to generate and commit transaction effects.
///
/// ## Thread Safety
/// This structure is immutable after creation and can be safely shared across threads.
pub struct InnerTemporaryStore {
    /// Objects that were in the input to the transaction
    pub input_objects: BTreeMap<ObjectID, Object>,

    /// Objects that were created or modified during execution
    pub written: WrittenObjects,

    /// The mutable input references used in the transaction
    pub mutable_inputs: BTreeMap<ObjectID, (VersionDigest, Owner)>,

    /// The lamport version assigned to the transaction
    pub lamport_version: Version,

    /// Shared objects that were deleted during the transaction
    pub deleted_shared_objects: BTreeMap<ObjectID, Version /* start_version */>,
}

impl InnerTemporaryStore {
    pub fn new(
        input_objects: BTreeMap<ObjectID, Object>,
        written: WrittenObjects,
        mutable_inputs: BTreeMap<ObjectID, (VersionDigest, Owner)>,
        lamport_version: Version,
        deleted_shared_objects: BTreeMap<ObjectID, Version>,
    ) -> Self {
        Self {
            input_objects,
            written,
            mutable_inputs,
            lamport_version,
            deleted_shared_objects,
        }
    }

    pub fn get_output_keys(&self, effects: &TransactionEffects) -> Vec<InputKey> {
        let mut output_keys: Vec<_> = self
            .written
            .iter()
            .map(|(id, obj)| InputKey::VersionedObject {
                id: obj.full_id(),
                version: obj.version(),
            })
            .collect();

        let deleted: HashMap<_, _> =
            effects.deleted().iter().map(|oref| (oref.0, oref.1)).collect();

        // add deleted shared objects to the outputkeys that then get sent to notify_commit
        let deleted_output_keys = deleted
            .iter()
            .filter_map(|(id, seq)| {
                self.input_objects
                    .get(id)
                    .and_then(|obj| obj.is_shared().then_some((obj.full_id(), *seq)))
            })
            .map(|(full_id, seq)| InputKey::VersionedObject { id: full_id, version: seq });
        output_keys.extend(deleted_output_keys);

        // For any previously deleted shared objects that appeared mutably in the transaction,
        // synthesize a notification for the next version of the object.
        let smeared_version = self.lamport_version;
        let deleted_accessed_objects = effects.deleted_mutably_accessed_shared_objects();
        for object_id in deleted_accessed_objects.into_iter() {
            let id =
                self.input_objects.get(&object_id).map(|obj| obj.full_id()).unwrap_or_else(|| {
                    let start_version = self.deleted_shared_objects.get(&object_id).expect(
                        "deleted object must be in either input_objects or \
                         deleted_consensus_objects",
                    );
                    FullObjectID::new(object_id, Some(*start_version))
                });
            let key = InputKey::VersionedObject { id, version: smeared_version };
            output_keys.push(key);
        }

        output_keys
    }
}

#[cfg(test)]
mod tests {
    //! Stage 2 plumbing tests: balance events flow through
    //! TemporaryStore → InnerTemporaryStore correctly. End-to-end
    //! verification (events → settlement tx → balance table delta) is
    //! Stage 3.

    use super::*;
    use crate::object::CoinType;
    use crate::system_state::FeeParameters;
    use crate::transaction::InputObjects;

    fn empty_store() -> TemporaryStore {
        TemporaryStore::new(
            InputObjects::new(Vec::new()),
            Vec::new(),
            TransactionDigest::default(),
            0,
            FeeParameters { unit_fee: 0 },
            0,
            protocol_config::Chain::Unknown,
        )
    }

    #[test]
    fn balance_events_default_empty() {
        let store = empty_store();
        assert!(store.balance_events().is_empty());
    }

    #[test]
    fn emit_balance_event_records_in_order() {
        let mut store = empty_store();
        let alice = SomaAddress::random();
        let bob = SomaAddress::random();
        let e1 = BalanceEvent::withdraw(alice, CoinType::Usdc, 100);
        let e2 = BalanceEvent::deposit(bob, CoinType::Usdc, 100);
        let e3 = BalanceEvent::withdraw(alice, CoinType::Soma, 7);

        store.emit_balance_event(e1);
        store.emit_balance_event(e2);
        store.emit_balance_event(e3);

        // Insertion order is preserved — settlement may reorder freely
        // (sums are commutative) but we keep insertion order for the
        // audit trail.
        assert_eq!(store.balance_events(), &[e1, e2, e3]);
    }

    #[test]
    fn balance_events_propagate_to_effects() {
        let mut store = empty_store();
        let alice = SomaAddress::random();
        let event = BalanceEvent::deposit(alice, CoinType::Usdc, 42);
        store.emit_balance_event(event);

        let (_inner, effects) = store.into_effects(
            Vec::new(),
            &TransactionDigest::default(),
            BTreeSet::new(),
            crate::effects::ExecutionStatus::Success,
            0,
            crate::tx_fee::TransactionFee::default(),
            None,
        );
        assert_eq!(effects.balance_events(), &[event]);
    }

    #[test]
    fn effects_balance_events_default_empty() {
        // Sanity: a store that emitted no events produces a TransactionEffects
        // with no events. Settlement must treat this as a no-op.
        let store = empty_store();
        let (_inner, effects) = store.into_effects(
            Vec::new(),
            &TransactionDigest::default(),
            BTreeSet::new(),
            crate::effects::ExecutionStatus::Success,
            0,
            crate::tx_fee::TransactionFee::default(),
            None,
        );
        assert!(effects.balance_events().is_empty());
    }
}
