// Copyright (c) Mysten Labs, Inc.
// Portions of this file are derived from Sui (https://github.com/MystenLabs/sui).
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use store::{
    DBMapUtils, TypedStoreError,
    rocks::{
        DBBatch, DBMap, DBMapTableConfigMap, DBOptions, default_db_options, read_size_from_env,
    },
    rocksdb::compaction_filter::Decision,
};
use store::{DbIterator, Map as _};
use tracing::{error, info};
use types::{
    base::SomaAddress,
    checkpoints::{CheckpointSequenceNumber, GlobalStateHash},
    committee::EpochId,
    digests::{ObjectDigest, TransactionDigest, TransactionEffectsDigest},
    effects::TransactionEffects,
    error::{SomaError, SomaResult},
    object::{LiveObject, Object, ObjectID, ObjectInner, ObjectRef, ObjectType, Owner, Version},
    storage::{FullObjectKey, MarkerValue, ObjectKey, object_store::ObjectStore},
    system_state::epoch_start::EpochStartSystemStateTrait,
    transaction::TrustedTransaction,
};

use crate::{
    authority_store::LockDetails,
    authority_store_pruner::ObjectsCompactionFilter,
    start_epoch::{EpochStartConfigTrait, EpochStartConfiguration},
};

const ENV_VAR_OBJECTS_BLOCK_CACHE_SIZE: &str = "OBJECTS_BLOCK_CACHE_MB";
pub(crate) const ENV_VAR_LOCKS_BLOCK_CACHE_SIZE: &str = "LOCKS_BLOCK_CACHE_MB";
const ENV_VAR_TRANSACTIONS_BLOCK_CACHE_SIZE: &str = "TRANSACTIONS_BLOCK_CACHE_MB";
const ENV_VAR_EFFECTS_BLOCK_CACHE_SIZE: &str = "EFFECTS_BLOCK_CACHE_MB";

/// Options to apply to every column family of the `perpetual` DB.
#[derive(Default)]
pub struct AuthorityPerpetualTablesOptions {
    /// Whether to enable write stalling on all column families.
    pub enable_write_stall: bool,
    pub compaction_filter: Option<ObjectsCompactionFilter>,
}

impl AuthorityPerpetualTablesOptions {
    fn apply_to(&self, mut db_options: DBOptions) -> DBOptions {
        if !self.enable_write_stall {
            db_options = db_options.disable_write_throttling();
        }
        db_options
    }
}

#[derive(Eq, PartialEq, Debug, Clone, Deserialize, Serialize, Hash)]
pub enum StoreObject {
    Value(ObjectInner),
    Deleted,
}

pub fn get_store_object(object: Object) -> StoreObject {
    StoreObject::Value(object.into_inner())
}

/// AuthorityPerpetualTables contains data that must be preserved from one epoch to the next.
#[derive(DBMapUtils)]
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
    pub(crate) objects: DBMap<ObjectKey, StoreObject>,

    /// This is a map between object references of currently active objects that can be mutated.
    ///
    /// For old epochs, it may also contain the transaction that they are lock on for use by this
    /// specific validator. The transaction locks themselves are now in AuthorityPerEpochStore.
    pub(crate) object_transaction_locks: DBMap<ObjectRef, Option<LockDetails>>,

    /// This is a map between the transaction digest and the corresponding transaction that's known to be
    /// executable. This means that it may have been executed locally, or it may have been synced through
    /// state-sync but hasn't been executed yet.
    pub(crate) transactions: DBMap<TransactionDigest, TrustedTransaction>,

    /// A map between the transaction digest of a certificate to the effects of its execution.
    /// We store effects into this table in two different cases:
    /// 1. When a transaction is synced through state_sync, we store the effects here. These effects
    ///    are known to be final in the network, but may not have been executed locally yet.
    /// 2. When the transaction is executed locally on this node, we store the effects here. This means that
    ///    it's possible to store the same effects twice (once for the synced transaction, and once for the executed).
    ///
    /// It's also possible for the effects to be reverted if the transaction didn't make it into the epoch.
    pub(crate) effects: DBMap<TransactionEffectsDigest, TransactionEffects>,

    /// Transactions that have been executed locally on this node. We need this table since the `effects` table
    /// doesn't say anything about the execution status of the transaction on this node. When we wait for transactions
    /// to be executed, we wait for them to appear in this table. When we revert transactions, we remove them from both
    /// tables.
    pub(crate) executed_effects: DBMap<TransactionDigest, TransactionEffectsDigest>,

    // Finalized root state hash for epoch, to be included in CheckpointSummary
    // of last checkpoint of epoch. These values should only ever be written once
    // and never changed
    pub(crate) root_state_hash_by_epoch:
        DBMap<EpochId, (CheckpointSequenceNumber, GlobalStateHash)>,

    /// Parameters of the system fixed at the epoch start
    pub(crate) epoch_start_configuration: DBMap<(), EpochStartConfiguration>,

    /// A singleton table that stores latest pruned checkpoint. Used to keep objects pruner progress
    pub(crate) pruned_checkpoint: DBMap<(), CheckpointSequenceNumber>,

    /// Table that stores the set of received objects and deleted objects and the version at
    /// which they were received. This is used to prevent possible race conditions around receiving
    /// objects (since they are not locked by the transaction manager) and for tracking shared
    /// objects that have been deleted. This table is meant to be pruned per-epoch, and all
    /// previous epochs other than the current epoch may be pruned safely.
    pub(crate) object_per_epoch_marker_table: DBMap<(EpochId, FullObjectKey), MarkerValue>,
}

impl AuthorityPerpetualTables {
    pub fn path(parent_path: &Path) -> PathBuf {
        parent_path.join("perpetual")
    }

    pub fn open(
        parent_path: &Path,
        db_options_override: Option<AuthorityPerpetualTablesOptions>,
    ) -> Self {
        let db_options_override = db_options_override.unwrap_or_default();
        let db_options =
            db_options_override.apply_to(default_db_options().optimize_db_for_write_throughput(4));
        let table_options = DBMapTableConfigMap::new(BTreeMap::from([
            (
                "objects".to_string(),
                objects_table_config(db_options.clone(), db_options_override.compaction_filter),
            ),
            (
                "owned_object_transaction_locks".to_string(),
                owned_object_transaction_locks_table_config(db_options.clone()),
            ),
            ("transactions".to_string(), transactions_table_config(db_options.clone())),
            ("effects".to_string(), effects_table_config(db_options.clone())),
        ]));

        Self::open_tables_read_write(
            Self::path(parent_path),
            Some(db_options.options),
            Some(table_options),
        )
    }

    pub fn open_readonly(parent_path: &Path) -> AuthorityPerpetualTablesReadOnly {
        Self::get_read_only_handle(Self::path(parent_path), None, None)
    }

    pub fn find_object_lt_or_eq_version(
        &self,
        object_id: ObjectID,
        version: Version,
    ) -> SomaResult<Option<Object>> {
        let mut iter = self.objects.reversed_safe_iter_with_bounds(
            Some(ObjectKey::min_for_id(&object_id)),
            Some(ObjectKey(object_id, version)),
        )?;
        match iter.next() {
            Some(Ok((key, o))) => self.object(&key, o),
            Some(Err(e)) => Err(e.into()),
            None => Ok(None),
        }
    }

    fn construct_object(&self, store_object: StoreObject) -> Result<Object, SomaError> {
        try_construct_object(store_object)
    }

    // Constructs `types::object::Object` from `StoreObjectWrapper`.
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
            StoreObject::Value(_) => {
                self.construct_object(store_object.clone())?.compute_object_reference()
            }
            StoreObject::Deleted => {
                (object_key.0, object_key.1, ObjectDigest::OBJECT_DIGEST_DELETED)
            }
        };
        Ok(obj_ref)
    }

    pub fn get_latest_object_ref_or_tombstone(
        &self,
        object_id: ObjectID,
    ) -> Result<Option<ObjectRef>, SomaError> {
        let mut iterator = self.objects.reversed_safe_iter_with_bounds(
            Some(ObjectKey::min_for_id(&object_id)),
            Some(ObjectKey::max_for_id(&object_id)),
        )?;

        if let Some(Ok((object_key, value))) = iterator.next() {
            if object_key.0 == object_id {
                return Ok(Some(self.object_reference(&object_key, value)?));
            }
        }
        Ok(None)
    }

    pub fn tombstone_reference(
        &self,
        object_key: &ObjectKey,
        store_object: &StoreObject,
    ) -> Result<Option<ObjectRef>, SomaError> {
        let obj_ref = match store_object {
            StoreObject::Deleted => {
                Some((object_key.0, object_key.1, ObjectDigest::OBJECT_DIGEST_DELETED))
            }
            _ => None,
        };
        Ok(obj_ref)
    }

    pub fn get_latest_object_or_tombstone(
        &self,
        object_id: ObjectID,
    ) -> Result<Option<(ObjectKey, StoreObject)>, SomaError> {
        let mut iterator = self.objects.reversed_safe_iter_with_bounds(
            Some(ObjectKey::min_for_id(&object_id)),
            Some(ObjectKey::max_for_id(&object_id)),
        )?;

        if let Some(Ok((object_key, value))) = iterator.next() {
            if object_key.0 == object_id {
                return Ok(Some((object_key, value)));
            }
        }
        Ok(None)
    }

    pub fn get_recovery_epoch_at_restart(&self) -> SomaResult<EpochId> {
        Ok(self
            .epoch_start_configuration
            .get(&())?
            .expect("Must have current epoch.")
            .epoch_start_state()
            .epoch())
    }

    pub fn set_epoch_start_configuration(
        &self,
        epoch_start_configuration: &EpochStartConfiguration,
    ) -> SomaResult {
        let mut wb = self.epoch_start_configuration.batch();
        wb.insert_batch(
            &self.epoch_start_configuration,
            std::iter::once(((), epoch_start_configuration)),
        )?;
        wb.write()?;
        Ok(())
    }

    pub fn get_highest_pruned_checkpoint(
        &self,
    ) -> Result<Option<CheckpointSequenceNumber>, TypedStoreError> {
        self.pruned_checkpoint.get(&())
    }

    pub fn set_highest_pruned_checkpoint(
        &self,
        wb: &mut DBBatch,
        checkpoint_number: CheckpointSequenceNumber,
    ) -> SomaResult {
        wb.insert_batch(&self.pruned_checkpoint, [((), checkpoint_number)])?;
        Ok(())
    }

    pub fn set_highest_pruned_checkpoint_without_wb(
        &self,
        checkpoint_number: CheckpointSequenceNumber,
    ) -> SomaResult {
        let mut wb = self.pruned_checkpoint.batch();
        self.set_highest_pruned_checkpoint(&mut wb, checkpoint_number)?;
        wb.write()?;
        Ok(())
    }

    pub fn get_root_state_hash(
        &self,
        epoch: EpochId,
    ) -> SomaResult<Option<(CheckpointSequenceNumber, GlobalStateHash)>> {
        Ok(self.root_state_hash_by_epoch.get(&epoch)?)
    }

    pub fn insert_root_state_hash(
        &self,
        epoch: EpochId,
        last_checkpoint_of_epoch: CheckpointSequenceNumber,
        hash: GlobalStateHash,
    ) -> SomaResult {
        self.root_state_hash_by_epoch.insert(&epoch, &(last_checkpoint_of_epoch, hash))?;
        Ok(())
    }

    pub fn get_transaction(
        &self,
        digest: &TransactionDigest,
    ) -> SomaResult<Option<TrustedTransaction>> {
        let Some(transaction) = self.transactions.get(digest)? else {
            return Ok(None);
        };
        Ok(Some(transaction))
    }

    pub fn get_effects(
        &self,
        digest: &TransactionDigest,
    ) -> SomaResult<Option<TransactionEffects>> {
        let Some(effect_digest) = self.executed_effects.get(digest)? else {
            return Ok(None);
        };
        Ok(self.effects.get(&effect_digest)?)
    }

    pub fn database_is_empty(&self) -> SomaResult<bool> {
        Ok(self.objects.safe_iter().next().is_none())
    }

    pub fn get_newer_object_keys(
        &self,
        object: &(ObjectID, Version),
    ) -> SomaResult<Vec<ObjectKey>> {
        let mut objects = vec![];
        for result in self.objects.safe_iter_with_bounds(
            Some(ObjectKey(object.0, object.1.next())),
            Some(ObjectKey(object.0, Version::MAX)),
        ) {
            let (key, _) = result?;
            objects.push(key);
        }
        Ok(objects)
    }

    pub fn iter_live_object_set(&self) -> LiveSetIter<'_> {
        LiveSetIter { iter: Box::new(self.objects.safe_iter()), tables: self, prev: None }
    }

    pub fn range_iter_live_object_set(
        &self,
        lower_bound: Option<ObjectID>,
        upper_bound: Option<ObjectID>,
        include_wrapped_object: bool,
    ) -> LiveSetIter<'_> {
        let lower_bound = lower_bound.as_ref().map(ObjectKey::min_for_id);
        let upper_bound = upper_bound.as_ref().map(ObjectKey::max_for_id);

        LiveSetIter {
            iter: Box::new(self.objects.safe_iter_with_bounds(lower_bound, upper_bound)),
            tables: self,
            prev: None,
        }
    }

    pub fn get_object_fallible(&self, object_id: &ObjectID) -> SomaResult<Option<Object>> {
        let obj_entry = self
            .objects
            .reversed_safe_iter_with_bounds(None, Some(ObjectKey::max_for_id(object_id)))?
            .next();

        match obj_entry.transpose()? {
            Some((ObjectKey(obj_id, version), obj)) if obj_id == *object_id => {
                Ok(self.object(&ObjectKey(obj_id, version), obj)?)
            }
            _ => Ok(None),
        }
    }

    pub fn get_object_by_key_fallible(
        &self,
        object_id: &ObjectID,
        version: Version,
    ) -> SomaResult<Option<Object>> {
        Ok(self.objects.get(&ObjectKey(*object_id, version))?.and_then(|object| {
            self.object(&ObjectKey(*object_id, version), object).expect("object construction error")
        }))
    }
}

impl ObjectStore for AuthorityPerpetualTables {
    /// Read an object and return it, or Ok(None) if the object was not found.
    fn get_object(&self, object_id: &ObjectID) -> Option<Object> {
        self.get_object_fallible(object_id).expect("db error")
    }

    fn get_object_by_key(&self, object_id: &ObjectID, version: Version) -> Option<Object> {
        self.get_object_by_key_fallible(object_id, version).expect("db error")
    }
}

type Map<K, V> = BTreeMap<K, V>;

pub struct LiveSetIter<'a> {
    iter: DbIterator<'a, (ObjectKey, StoreObject)>,
    tables: &'a AuthorityPerpetualTables,
    prev: Option<(ObjectKey, StoreObject)>,
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
            if let Some(Ok((next_key, next_value))) = self.iter.next() {
                let prev = self.prev.take();
                self.prev = Some((next_key, next_value));

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
            ));
        }
    };

    Ok(data.into())
}

#[derive(DBMapUtils)]
pub struct AuthorityPrunerTables {
    pub(crate) object_tombstones: DBMap<ObjectID, Version>,
}

impl AuthorityPrunerTables {
    pub fn path(parent_path: &Path) -> PathBuf {
        parent_path.join("pruner")
    }

    pub fn open(parent_path: &Path) -> Self {
        Self::open_tables_read_write(Self::path(parent_path), None, None)
    }
}

// These functions are used to initialize the DB tables
fn owned_object_transaction_locks_table_config(db_options: DBOptions) -> DBOptions {
    DBOptions {
        options: db_options
            .clone()
            .optimize_for_write_throughput()
            .optimize_for_read(read_size_from_env(ENV_VAR_LOCKS_BLOCK_CACHE_SIZE).unwrap_or(1024))
            .options,
        rw_options: db_options.rw_options.set_ignore_range_deletions(false),
    }
}

fn objects_table_config(
    mut db_options: DBOptions,
    compaction_filter: Option<ObjectsCompactionFilter>,
) -> DBOptions {
    if let Some(mut compaction_filter) = compaction_filter {
        db_options.options.set_compaction_filter("objects", move |_, key, value| {
            match compaction_filter.filter(key, value) {
                Ok(decision) => decision,
                Err(err) => {
                    error!("Compaction error: {:?}", err);
                    Decision::Keep
                }
            }
        });
    }
    db_options
        .optimize_for_write_throughput()
        .optimize_for_read(read_size_from_env(ENV_VAR_OBJECTS_BLOCK_CACHE_SIZE).unwrap_or(5 * 1024))
}

fn transactions_table_config(db_options: DBOptions) -> DBOptions {
    db_options.optimize_for_write_throughput().optimize_for_point_lookup(
        read_size_from_env(ENV_VAR_TRANSACTIONS_BLOCK_CACHE_SIZE).unwrap_or(512),
    )
}

fn effects_table_config(db_options: DBOptions) -> DBOptions {
    db_options.optimize_for_write_throughput().optimize_for_point_lookup(
        read_size_from_env(ENV_VAR_EFFECTS_BLOCK_CACHE_SIZE).unwrap_or(1024),
    )
}
