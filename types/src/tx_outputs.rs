use std::{collections::BTreeMap, sync::Arc};

use crate::{
    effects::{TransactionEffects, TransactionEffectsAPI},
    object::{Object, ObjectID, ObjectRef},
    storage::ObjectKey,
    transaction::VerifiedTransaction,
};

pub type WrittenObjects = BTreeMap<ObjectID, Object>;
/// TransactionOutputs
pub struct TransactionOutputs {
    pub transaction: Arc<VerifiedTransaction>,
    pub effects: TransactionEffects,

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
        written: WrittenObjects,
    ) -> TransactionOutputs {
        let deleted = effects.deleted().into_iter().map(ObjectKey::from).collect();

        let locks_to_delete: Vec<_> = Vec::new();

        // TODO: mutable_inputs
        //     .into_iter()
        //     .filter_map(|(id, ((version, digest), owner))| {
        //         owner.is_address_owned().then_some((id, version, digest))
        //     })
        //     .chain(received_objects)
        //     .collect();

        let new_locks_to_init: Vec<_> = written
            .values()
            .filter_map(|new_object| {
                // if new_object.is_address_owned() {
                Some(new_object.compute_object_reference())
                // } else {
                //     None
                // }
            })
            .collect();

        TransactionOutputs {
            transaction: Arc::new(transaction),
            effects,
            deleted,
            written,
            locks_to_delete,
            new_locks_to_init,
        }
    }
}
