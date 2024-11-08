use std::collections::{BTreeMap, BTreeSet};

use crate::{
    committee::EpochId,
    digests::TransactionDigest,
    effects::{object_change::EffectsObjectChange, ExecutionStatus, TransactionEffects},
    object::{Object, ObjectID, Version},
    tx_outputs::WrittenObjects,
};

/// The results represent the primitive information that can then be used to construct
/// transaction effects.
#[derive(Debug, Default)]
pub struct ExecutionResults {
    /// All objects written regardless of whether they were mutated, created, or unwrapped.
    pub written_objects: BTreeMap<ObjectID, Object>,
    /// All objects that existed prior to this transaction, and are modified in this transaction.
    /// This includes any type of modification, including mutated, wrapped and deleted objects.
    pub modified_objects: BTreeSet<ObjectID>,
    /// All object IDs created in this transaction.
    pub created_object_ids: BTreeSet<ObjectID>,
    /// All object IDs deleted in this transaction.
    /// No object ID should be in both created_object_ids and deleted_object_ids.
    pub deleted_object_ids: BTreeSet<ObjectID>,
}

impl ExecutionResults {
    pub fn update_version_and_previous_tx(
        &mut self,
        lamport_timestamp: Version,
        prev_tx: TransactionDigest,
    ) {
        for (_id, obj) in self.written_objects.iter_mut() {
            // Update the version for the written object.
            obj.data.increment_version_to(lamport_timestamp);
            obj.previous_transaction = prev_tx;
        }
    }
}

pub struct TemporaryStore {
    tx_digest: TransactionDigest,
    input_objects: BTreeMap<ObjectID, Object>,
    /// The version to assign to all objects written by the transaction using this store.
    lamport_timestamp: Version,
    execution_results: ExecutionResults,
}

impl TemporaryStore {
    pub fn new(
        input_objects: BTreeMap<ObjectID, Object>,
        tx_digest: TransactionDigest,
        lamport_timestamp: Version,
    ) -> Self {
        Self {
            tx_digest,
            input_objects,
            lamport_timestamp,
            execution_results: ExecutionResults::default(),
        }
    }

    pub fn update_object_version_and_prev_tx(&mut self) {
        self.execution_results
            .update_version_and_previous_tx(self.lamport_timestamp, self.tx_digest);
    }

    fn get_object_changes(&self) -> BTreeMap<ObjectID, EffectsObjectChange> {
        let results = &self.execution_results;
        let all_ids = results
            .created_object_ids
            .iter()
            .chain(&results.deleted_object_ids)
            .chain(&results.modified_objects)
            .chain(results.written_objects.keys())
            .collect::<BTreeSet<_>>();
        all_ids
            .into_iter()
            .map(|id| {
                (
                    *id,
                    EffectsObjectChange::new(
                        self.read_object(id)
                            .map(|obj| ((obj.data.version(), obj.digest()))),
                        results.written_objects.get(id),
                        results.created_object_ids.contains(id),
                        results.deleted_object_ids.contains(id),
                    ),
                )
            })
            .collect()
    }

    pub fn into_effects(
        mut self,
        transaction_digest: &TransactionDigest,
        status: ExecutionStatus,
        epoch: EpochId,
    ) -> (WrittenObjects, TransactionEffects) {
        self.update_object_version_and_prev_tx();

        let object_changes = self.get_object_changes();
        let written_objects = self.execution_results.written_objects;
        let lamport_version = self.lamport_timestamp;

        let effects = TransactionEffects::new(
            status,
            epoch,
            *transaction_digest,
            lamport_version,
            object_changes,
        );

        (written_objects, effects)
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
        debug_assert!(!self.execution_results.written_objects.contains_key(id));
        debug_assert!(self.input_objects.contains_key(id));
        self.execution_results.modified_objects.insert(*id);
        self.execution_results.deleted_object_ids.insert(*id);
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

    pub fn mutate_input_object(&mut self, object: Object) {
        let id = object.id();
        debug_assert!(self.input_objects.contains_key(&id));

        self.execution_results.modified_objects.insert(id);
        self.execution_results.written_objects.insert(id, object);
    }
}
