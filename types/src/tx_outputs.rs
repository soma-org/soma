use std::{
    collections::{BTreeMap, HashMap, HashSet},
    sync::Arc,
};

use crate::{
    base::FullObjectID,
    effects::{TransactionEffects, TransactionEffectsAPI},
    object::{Object, ObjectID, ObjectRef, Owner, Version, VersionDigest},
    storage::{FullObjectKey, MarkerValue, ObjectKey},
    temporary_store::InnerTemporaryStore,
    transaction::VerifiedTransaction,
};

pub type ObjectMap = BTreeMap<ObjectID, Object>;
pub type WrittenObjects = BTreeMap<ObjectID, Object>;
/// TransactionOutputs
pub struct TransactionOutputs {
    pub transaction: Arc<VerifiedTransaction>,
    pub effects: TransactionEffects,

    pub markers: Vec<(FullObjectKey, MarkerValue)>,

    pub deleted: Vec<ObjectKey>,
    pub written: WrittenObjects,

    pub locks_to_delete: Vec<ObjectRef>,
    pub new_locks_to_init: Vec<ObjectRef>,
}

impl TransactionOutputs {
    // Convert Effects into the exact set of updates to the store
    pub fn build_transaction_outputs(
        transaction: VerifiedTransaction,
        effects: TransactionEffects,
        inner_temporary_store: InnerTemporaryStore,
    ) -> TransactionOutputs {
        let InnerTemporaryStore {
            input_objects,
            deleted_shared_objects,
            mutable_inputs,
            written,
            lamport_version,
        } = inner_temporary_store;

        let tx_digest = *transaction.digest();

        let deleted: HashMap<_, _> = effects.all_tombstones().into_iter().collect();

        // Get the actual set of objects that have been received -- any received
        // object will show up in the modified-at set.
        let modified_at: HashSet<_> = effects.modified_at_versions().into_iter().collect();
        let possible_to_receive = transaction.transaction_data().receiving_objects();
        let received_objects = possible_to_receive
            .into_iter()
            .filter(|obj_ref| modified_at.contains(&(obj_ref.0, obj_ref.1)));

        // We record any received or deleted objects since they could be pruned, and smear shared
        // object deletions in the marker table. For deleted entries in the marker table we need to
        // make sure we don't accidentally overwrite entries.
        let markers: Vec<_> = {
            let received = received_objects.clone().map(|objref| {
                (
                    // TODO: Add support for receiving ConsensusV2 objects. For now this assumes fastpath.
                    FullObjectKey::new(FullObjectID::new(objref.0, None), objref.1),
                    MarkerValue::Received,
                )
            });

            let deleted = deleted.into_iter().map(|(object_id, version)| {
                let shared_key = input_objects
                    .get(&object_id)
                    .filter(|o| o.is_shared())
                    .map(|o| FullObjectKey::new(o.full_id(), version));
                if let Some(shared_key) = shared_key {
                    (shared_key, MarkerValue::SharedDeleted(tx_digest))
                } else {
                    (
                        FullObjectKey::new(FullObjectID::new(object_id, None), version),
                        MarkerValue::OwnedDeleted,
                    )
                }
            });

            // We "smear" shared deleted objects in the marker table to allow for proper sequencing
            // of transactions that are submitted after the deletion of the shared object.
            // NB: that we do _not_ smear shared objects that were taken immutably in the
            // transaction.
            let smeared_objects = effects.deleted_mutably_accessed_shared_objects();
            let shared_smears = smeared_objects.into_iter().map(|object_id| {
                let id = input_objects
                    .get(&object_id)
                    .map(|obj| obj.full_id())
                    .unwrap_or_else(|| {
                        let start_version = deleted_shared_objects.get(&object_id).expect(
                            "deleted object must be in either input_objects or \
                             deleted_consensus_objects",
                        );
                        FullObjectID::new(object_id, Some(*start_version))
                    });
                (
                    FullObjectKey::new(id, lamport_version),
                    MarkerValue::SharedDeleted(tx_digest),
                )
            });

            received.chain(deleted).chain(shared_smears).collect()
        };

        let locks_to_delete: Vec<_> = mutable_inputs
            .into_iter()
            .filter_map(|(id, ((version, digest), owner))| {
                owner.is_address_owned().then_some((id, version, digest))
            })
            .chain(received_objects)
            .collect();

        let new_locks_to_init: Vec<_> = written
            .values()
            .filter_map(|new_object| {
                if new_object.is_address_owned() {
                    Some(new_object.compute_object_reference())
                } else {
                    None
                }
            })
            .collect();

        let deleted = effects.deleted().into_iter().map(ObjectKey::from).collect();

        TransactionOutputs {
            transaction: Arc::new(transaction),
            effects,
            markers,
            deleted,
            written,
            locks_to_delete,
            new_locks_to_init,
        }
    }
}
