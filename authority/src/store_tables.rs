use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use types::{
    committee::EpochId,
    digests::{ObjectDigest, TransactionDigest, TransactionEffectsDigest},
    effects::TransactionEffects,
    error::{SomaError, SomaResult},
    object::{Object, ObjectID, ObjectInner, ObjectRef, Version},
    storage::{object_store::ObjectStore, ObjectKey},
    system_state::EpochStartSystemStateTrait,
    transaction::TrustedTransaction,
};

use crate::{
    start_epoch::{EpochStartConfigTrait, EpochStartConfiguration},
    store::LockDetails,
};

#[derive(Eq, PartialEq, Debug, Clone, Deserialize, Serialize, Hash)]
pub enum StoreObject {
    Value(ObjectInner),
    Deleted,
}

/// AuthorityPerpetualTables contains data that must be preserved from one epoch to the next.
pub struct AuthorityPerpetualTables {
    /// This is a map between the object (ID, version) and the latest state of the object, namely the
    /// state that is needed to process new transactions.
    /// State is represented by `StoreObject`.
    ///
    /// Note that while this map can store all versions of an object, we will eventually
    /// prune old object versions from the db.
    ///
    /// IMPORTANT: object versions must *only* be pruned if they appear as inputs in some
    /// TransactionEffects. Simply pruning all objects but the most recent is an error!
    /// This is because there can be partially executed transactions whose effects have not yet
    /// been written out, and which must be retried. But, they cannot be retried unless their input
    /// objects are still accessible!
    pub(crate) objects: RwLock<BTreeMap<ObjectKey, StoreObject>>,

    /// This is a map between object references of currently active objects that can be mutated.
    ///
    /// For old epochs, it may also contain the transaction that they are lock on for use by this
    /// specific validator. The transaction locks themselves are now in AuthorityPerEpochStore.
    pub(crate) object_transaction_locks: RwLock<BTreeMap<ObjectRef, Option<LockDetails>>>,

    /// This is a map between the transaction digest and the corresponding transaction that's known to be
    /// executable. This means that it may have been executed locally, or it may have been synced through
    /// state-sync but hasn't been executed yet.
    pub(crate) transactions: RwLock<BTreeMap<TransactionDigest, TrustedTransaction>>,

    /// A map between the transaction digest of a certificate to the effects of its execution.
    /// We store effects into this table in two different cases:
    /// 1. When a transaction is synced through state_sync, we store the effects here. These effects
    ///     are known to be final in the network, but may not have been executed locally yet.
    /// 2. When the transaction is executed locally on this node, we store the effects here. This means that
    ///     it's possible to store the same effects twice (once for the synced transaction, and once for the executed).
    ///
    /// It's also possible for the effects to be reverted if the transaction didn't make it into the epoch.
    pub(crate) effects: RwLock<BTreeMap<TransactionEffectsDigest, TransactionEffects>>,

    /// Transactions that have been executed locally on this node. We need this table since the `effects` table
    /// doesn't say anything about the execution status of the transaction on this node. When we wait for transactions
    /// to be executed, we wait for them to appear in this table. When we revert transactions, we remove them from both
    /// tables.
    pub(crate) executed_effects: RwLock<BTreeMap<TransactionDigest, TransactionEffectsDigest>>,

    /// Parameters of the system fixed at the epoch start
    pub(crate) epoch_start_configuration: RwLock<BTreeMap<(), EpochStartConfiguration>>,
}

impl AuthorityPerpetualTables {
    pub fn path(parent_path: &Path) -> PathBuf {
        parent_path.join("perpetual")
    }

    pub fn open(parent_path: &Path) -> Self {
        // Self::open_tables_read_write(Self::path(parent_path))
        Self {
            transactions: RwLock::new(BTreeMap::new()),
            effects: RwLock::new(BTreeMap::new()),
            executed_effects: RwLock::new(BTreeMap::new()),
            epoch_start_configuration: RwLock::new(BTreeMap::new()),
            objects: RwLock::new(BTreeMap::new()),
            object_transaction_locks: RwLock::new(BTreeMap::new()),
        }
    }

    pub fn find_object_lt_or_eq_version(
        &self,
        object_id: ObjectID,
        version: Version,
    ) -> SomaResult<Option<Object>> {
        // Create boundary keys
        let start_key = ObjectKey::min_for_id(&object_id);
        let end_key = ObjectKey(object_id, version);

        // Use bounded range and get the highest version that's <= our target
        match self.objects.read()
            .range(start_key..=end_key)  // Get range from min to target version
            .rev()  // Get highest version first
            .next()  // Take the first one (highest version in our range)
        {
            Some((key, obj)) => self.object(key, obj.clone()),
            None => Ok(None),
        }
    }

    fn construct_object(&self, store_object: StoreObject) -> Result<Object, SomaError> {
        try_construct_object(store_object)
    }

    // Constructs `sui_types::object::Object` from `StoreObjectWrapper`.
    // Returns `None` if object was deleted/wrapped
    pub fn object(
        &self,
        object_key: &ObjectKey,
        store_object: StoreObject,
    ) -> Result<Option<Object>, SomaError> {
        Ok(Some(self.construct_object(store_object)?))
    }

    pub fn object_reference(
        &self,
        object_key: &ObjectKey,
        store_object: StoreObject,
    ) -> Result<ObjectRef, SomaError> {
        let obj_ref = match store_object {
            StoreObject::Value(_) => self
                .construct_object(store_object.clone())?
                .compute_object_reference(),
            StoreObject::Deleted => (
                object_key.0,
                object_key.1,
                ObjectDigest::OBJECT_DIGEST_DELETED,
            ),
        };
        Ok(obj_ref)
    }

    pub fn get_latest_object_ref_or_tombstone(
        &self,
        object_id: ObjectID,
    ) -> Result<Option<ObjectRef>, SomaError> {
        // Create boundary keys for this object_id
        let start_key = ObjectKey::min_for_id(&object_id);
        let end_key = ObjectKey::max_for_id(&object_id);

        // Get the latest version within this range
        let latest = self
            .objects
            .read()
            .range(start_key..=end_key) // Get exact range for this object_id
            .next_back() // Get the last entry (highest version)
            .map(|(key, value)| (key.clone(), value.clone()));

        match latest {
            Some((object_key, value)) => Ok(Some(self.object_reference(&object_key, value)?)),
            None => Ok(None),
        }
    }

    pub fn tombstone_reference(
        &self,
        object_key: &ObjectKey,
        store_object: &StoreObject,
    ) -> Result<Option<ObjectRef>, SomaError> {
        let obj_ref = match store_object {
            StoreObject::Deleted => Some((
                object_key.0,
                object_key.1,
                ObjectDigest::OBJECT_DIGEST_DELETED,
            )),
            _ => None,
        };
        Ok(obj_ref)
    }

    pub fn get_latest_object_or_tombstone(
        &self,
        object_id: ObjectID,
    ) -> Result<Option<(ObjectKey, StoreObject)>, SomaError> {
        // Create boundary keys for this object_id
        let start_key = ObjectKey::min_for_id(&object_id);
        let end_key = ObjectKey::max_for_id(&object_id);

        // Get the latest version within this range
        Ok(self
            .objects
            .read()
            .range(start_key..=end_key) // Get exact range for this object_id
            .next_back() // Get the last entry (highest version)
            .map(|(key, value)| (key.clone(), value.clone())))
    }

    pub fn get_recovery_epoch_at_restart(&self) -> SomaResult<EpochId> {
        Ok(self
            .epoch_start_configuration
            .read()
            .get(&())
            .expect("Must have current epoch.")
            .epoch_start_state()
            .epoch())
    }

    pub fn set_epoch_start_configuration(
        &self,
        epoch_start_configuration: &EpochStartConfiguration,
    ) -> SomaResult {
        self.epoch_start_configuration
            .write()
            .insert((), epoch_start_configuration.clone());
        Ok(())
    }

    pub fn get_transaction(
        &self,
        digest: &TransactionDigest,
    ) -> SomaResult<Option<TrustedTransaction>> {
        let transaction_read = self.transactions.read();
        let Some(transaction) = transaction_read.get(digest) else {
            return Ok(None);
        };
        Ok(Some(transaction.clone()))
    }

    pub fn get_effects(
        &self,
        digest: &TransactionDigest,
    ) -> SomaResult<Option<TransactionEffects>> {
        let executed_effects_read = self.executed_effects.read();
        let Some(effect_digest) = executed_effects_read.get(digest) else {
            return Ok(None);
        };
        Ok(self.effects.read().get(&effect_digest).cloned())
    }

    pub fn database_is_empty(&self) -> SomaResult<bool> {
        let guard = self.objects.read();
        Ok(guard.is_empty() || {
            // If not empty, check if there are any entries >= ZERO
            guard.range(ObjectKey::ZERO..).next().is_none()
        })
    }

    pub fn get_newer_object_keys(
        &self,
        object: &(ObjectID, Version),
    ) -> SomaResult<Vec<ObjectKey>> {
        let start_key = ObjectKey(object.0, object.1.next());
        let end_key = ObjectKey(object.0, Version::MAX);

        // Collect all keys in range, maintaining object ID match
        Ok(self
            .objects
            .read()
            .range(start_key..=end_key)
            .take_while(|(key, _)| key.0 == object.0)
            .map(|(key, _)| key.clone())
            .collect())
    }

    pub fn range_iter_live_object_set(
        &self,
        lower_bound: Option<ObjectID>,
        upper_bound: Option<ObjectID>,
        include_wrapped_object: bool,
    ) -> LiveSetIter<'_> {
        let lower_bound_key = lower_bound.as_ref().map(ObjectKey::min_for_id);
        let upper_bound_key = upper_bound.as_ref().map(ObjectKey::max_for_id);

        let guard = self.objects.read();

        // Create a new BTreeMap containing only the entries within the specified range
        let filtered_map: BTreeMap<ObjectKey, StoreObject> =
            match (lower_bound_key, upper_bound_key) {
                (Some(start), Some(end)) => guard.range(start..=end),
                (Some(start), None) => guard.range(start..),
                (None, Some(end)) => guard.range(..=end),
                (None, None) => guard.range(..),
            }
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        LiveSetIter {
            iter: filtered_map,
            tables: self,
            prev: None,
            include_wrapped_object,
        }
    }
}

impl ObjectStore for AuthorityPerpetualTables {
    /// Read an object and return it, or Ok(None) if the object was not found.
    fn get_object(
        &self,
        object_id: &ObjectID,
    ) -> Result<Option<Object>, types::storage::storage_error::Error> {
        // Get read lock on objects
        let objects = self.objects.read();

        // Find the latest version of the object using range
        let obj_entry = objects
            .range(..=ObjectKey::max_for_id(object_id)) // Get all entries up to max version
            .rev() // Reverse to get highest version first
            .find(|(key, _)| key.0 == *object_id) // Find first entry matching our object_id
            .map(|(key, value)| (key.clone(), value.clone()));

        match obj_entry {
            Some((key, obj)) => Ok(self
                .object(&key, obj)
                .map_err(types::storage::storage_error::Error::custom)?),
            None => Ok(None),
        }
    }

    fn get_object_by_key(
        &self,
        object_id: &ObjectID,
        version: Version,
    ) -> Result<Option<Object>, types::storage::storage_error::Error> {
        Ok(self
            .objects
            .read()
            .get(&ObjectKey(*object_id, version))
            // .map_err(types::storage::storage_error::Error::custom)?
            .map(|object| self.object(&ObjectKey(*object_id, version), object.clone()))
            .transpose()
            .map_err(types::storage::storage_error::Error::custom)?
            .flatten())
    }
}

type Map<K, V> = BTreeMap<K, V>;

pub struct LiveSetIter<'a> {
    iter: BTreeMap<ObjectKey, StoreObject>,
    tables: &'a AuthorityPerpetualTables,
    prev: Option<(ObjectKey, StoreObject)>,
    /// Whether a wrapped object is considered as a live object.
    include_wrapped_object: bool,
}

#[derive(Eq, PartialEq, Debug, Clone, Deserialize, Serialize, Hash)]
pub enum LiveObject {
    Normal(Object),
}

impl LiveObject {
    pub fn object_id(&self) -> ObjectID {
        match self {
            LiveObject::Normal(obj) => obj.id(),
        }
    }

    pub fn version(&self) -> Version {
        match self {
            LiveObject::Normal(obj) => obj.version(),
        }
    }

    pub fn object_reference(&self) -> ObjectRef {
        match self {
            LiveObject::Normal(obj) => obj.compute_object_reference(),
        }
    }

    pub fn to_normal(self) -> Option<Object> {
        match self {
            LiveObject::Normal(object) => Some(object),
        }
    }
}

impl LiveSetIter<'_> {
    fn store_object_wrapper_to_live_object(
        &self,
        object_key: ObjectKey,
        store_object: StoreObject,
    ) -> Option<LiveObject> {
        match store_object {
            StoreObject::Value(_) => {
                let object = self
                    .tables
                    .construct_object(store_object.clone())
                    .expect("Constructing object from store cannot fail");
                Some(LiveObject::Normal(object))
            }
            StoreObject::Deleted => None,
        }
    }
}

impl Iterator for LiveSetIter<'_> {
    type Item = LiveObject;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some((next_key, next_value)) = self.iter.iter().next() {
                let prev = self.prev.take();
                self.prev = Some((next_key.clone(), next_value.clone()));

                if let Some((prev_key, prev_value)) = prev {
                    if prev_key.0 != next_key.0 {
                        let live_object =
                            self.store_object_wrapper_to_live_object(prev_key, prev_value);
                        if live_object.is_some() {
                            return live_object;
                        }
                    }
                }
                continue;
            }
            if let Some((key, value)) = self.prev.take() {
                let live_object = self.store_object_wrapper_to_live_object(key, value);
                if live_object.is_some() {
                    return live_object;
                }
            }
            return None;
        }
    }
}

pub(crate) fn try_construct_object(store_object: StoreObject) -> Result<Object, SomaError> {
    let data = match store_object {
        StoreObject::Value(object) => object,
        _ => {
            return Err(SomaError::Storage(
                "corrupted field: inconsistent object representation".to_string(),
            ))
        }
    };

    Ok(data.into())
}
