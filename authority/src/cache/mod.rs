// Portions of this file are derived from Sui (MystenLabs/sui).
// Original source: https://github.com/MystenLabs/sui/tree/main/crates/sui-core/src/
// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

use std::{collections::HashSet, path::Path, sync::Arc};

use crate::{
    authority::ExecutionLockWriteGuard,
    authority_per_epoch_store::AuthorityPerEpochStore,
    authority_store::{AuthorityStore, LockResult},
    backpressure_manager::BackpressureManager,
    global_state_hasher::GlobalStateHashStore,
    start_epoch::EpochStartConfiguration,
};
use futures::{
    FutureExt,
    future::{BoxFuture, Either},
};
use protocol_config::ProtocolVersion;
use store::rocks::DBBatch;
use tracing::{debug, instrument};
use types::{
    base::{FullObjectID, SomaAddress, VerifiedExecutionData},
    checkpoints::CheckpointSequenceNumber,
    committee::EpochId,
    config::node_config::ExecutionCacheConfig,
    digests::{TransactionDigest, TransactionEffectsDigest},
    effects::TransactionEffects,
    error::{SomaError, SomaResult},
    object::{Object, ObjectID, ObjectRef, Version},
    storage::{
        FullObjectKey, InputKey, MarkerValue, ObjectKey, ObjectOrTombstone,
        object_store::ObjectStore,
    },
    system_state::SystemState,
    transaction::{VerifiedExecutableTransaction, VerifiedSignedTransaction, VerifiedTransaction},
    transaction_outputs::TransactionOutputs,
};
use writeback_cache::WritebackCache;

pub(crate) mod cache_types;
pub(crate) mod object_locks;
pub(crate) mod writeback_cache;

#[cfg(test)]
mod tests;

#[derive(Clone)]
pub struct ExecutionCacheTraitPointers {
    pub object_cache_reader: Arc<dyn ObjectCacheRead>,
    pub transaction_cache_reader: Arc<dyn TransactionCacheRead>,
    pub cache_writer: Arc<dyn ExecutionCacheWrite>,
    pub object_store: Arc<dyn ObjectStore + Send + Sync>,
    pub reconfig_api: Arc<dyn ExecutionCacheReconfigAPI>,
    pub global_state_hash_store: Arc<dyn GlobalStateHashStore>,
    pub state_sync_store: Arc<dyn StateSyncAPI>,
    pub cache_commit: Arc<dyn ExecutionCacheCommit>,
    pub testing_api: Arc<dyn TestingAPI>,
}

impl ExecutionCacheTraitPointers {
    pub fn new<T>(cache: Arc<T>) -> Self
    where
        T: TransactionCacheRead
            + ExecutionCacheWrite
            + ObjectCacheRead
            + ObjectStore
            + ExecutionCacheReconfigAPI
            + GlobalStateHashStore
            + StateSyncAPI
            + ExecutionCacheCommit
            + TestingAPI
            + 'static,
    {
        Self {
            transaction_cache_reader: cache.clone(),
            cache_writer: cache.clone(),
            object_cache_reader: cache.clone(),
            object_store: cache.clone(),
            // backing_store: cache.clone(),
            reconfig_api: cache.clone(),
            global_state_hash_store: cache.clone(),
            state_sync_store: cache.clone(),
            cache_commit: cache.clone(),
            testing_api: cache.clone(),
        }
    }
}

pub fn build_execution_cache(
    cache_config: &ExecutionCacheConfig,
    store: &Arc<AuthorityStore>,
    backpressure_manager: Arc<BackpressureManager>,
) -> ExecutionCacheTraitPointers {
    ExecutionCacheTraitPointers::new(
        WritebackCache::new(cache_config, store.clone(), backpressure_manager).into(),
    )
}

pub type Batch = (Vec<Arc<TransactionOutputs>>, DBBatch);

pub trait ExecutionCacheCommit: Send + Sync {
    /// Build a DBBatch containing the given transaction outputs.
    fn build_db_batch(&self, epoch: EpochId, digests: &[TransactionDigest]) -> Batch;

    /// Durably commit the outputs of the given transactions to the database.
    /// Will be called by CheckpointExecutor to ensure that transaction outputs are
    /// written durably before marking a checkpoint as finalized.
    fn commit_transaction_outputs(
        &self,
        epoch: EpochId,
        batch: Batch,
        digests: &[TransactionDigest],
    );

    /// Durably commit a transaction to the database. Used to store any transactions
    /// that cannot be reconstructed at start-up by consensus replay. Currently the only
    /// case of this is RandomnessStateUpdate.
    fn persist_transaction(&self, transaction: &VerifiedExecutableTransaction);

    // Number of pending uncommitted transactions
    fn approximate_pending_transaction_count(&self) -> u64;
}

pub trait ObjectCacheRead: Send + Sync {
    fn get_object(&self, id: &ObjectID) -> Option<Object>;

    fn get_objects(&self, objects: &[ObjectID]) -> Vec<Option<Object>> {
        let mut ret = Vec::with_capacity(objects.len());
        for object_id in objects {
            ret.push(self.get_object(object_id));
        }
        ret
    }

    fn get_latest_object_ref_or_tombstone(&self, object_id: ObjectID) -> Option<ObjectRef>;

    fn get_latest_object_or_tombstone(
        &self,
        object_id: ObjectID,
    ) -> Option<(ObjectKey, ObjectOrTombstone)>;

    fn get_object_by_key(&self, object_id: &ObjectID, version: Version) -> Option<Object>;

    fn multi_get_objects_by_key(&self, object_keys: &[ObjectKey]) -> Vec<Option<Object>>;

    fn object_exists_by_key(&self, object_id: &ObjectID, version: Version) -> bool;

    fn multi_object_exists_by_key(&self, object_keys: &[ObjectKey]) -> Vec<bool>;

    /// Load a list of objects from the store by object reference.
    /// If they exist in the store, they are returned directly.
    /// If any object missing, we try to figure out the best error to return.
    /// If the object we are asking is currently locked at a future version, we know this
    /// transaction is out-of-date and we return a ObjectVersionUnavailableForConsumption,
    /// which indicates this is not retriable.
    /// Otherwise, we return a ObjectNotFound error, which indicates this is retriable.
    fn multi_get_objects_with_more_accurate_error_return(
        &self,
        object_refs: &[ObjectRef],
    ) -> Result<Vec<Object>, SomaError> {
        let objects = self
            .multi_get_objects_by_key(&object_refs.iter().map(ObjectKey::from).collect::<Vec<_>>());
        let mut result = Vec::new();
        for (object_opt, object_ref) in objects.into_iter().zip(object_refs) {
            match object_opt {
                None => {
                    let live_objref = self._get_live_objref(object_ref.0)?;
                    let error = if live_objref.1 >= object_ref.1 {
                        SomaError::ObjectVersionUnavailableForConsumption {
                            provided_obj_ref: *object_ref,
                            current_version: live_objref.1,
                        }
                    } else {
                        SomaError::ObjectNotFound {
                            object_id: object_ref.0,
                            version: Some(object_ref.1),
                        }
                    };
                    return Err(error);
                }
                Some(object) => {
                    result.push(object);
                }
            }
        }
        assert_eq!(result.len(), object_refs.len());
        Ok(result)
    }

    /// Used by execution scheduler to determine if input objects are ready. Distinct from multi_get_object_by_key
    /// because it also consults markers to handle the case where an object will never become available (e.g.
    /// because it has been received by some other transaction already).
    fn multi_input_objects_available(
        &self,
        keys: &[InputKey],
        receiving_objects: &HashSet<InputKey>,
        epoch: EpochId,
    ) -> Vec<bool> {
        let mut results = vec![false; keys.len()];
        let non_canceled_keys = keys.iter().enumerate().filter(|(idx, key)| {
            if key.is_cancelled() {
                // Shared objects in canceled transactions are always available.
                results[*idx] = true;
                false
            } else {
                true
            }
        });
        let object_keys: Vec<_> = non_canceled_keys
            .map(|(idx, key)| match key {
                InputKey::VersionedObject { id, version } => (idx, (id, version)),
            })
            .collect();

        for ((idx, (id, version)), has_key) in object_keys.iter().zip(
            self.multi_object_exists_by_key(
                &object_keys.iter().map(|(_, k)| ObjectKey(k.0.id(), *k.1)).collect::<Vec<_>>(),
            )
            .into_iter(),
        ) {
            // If the key exists at the specified version, then the object is available.
            if has_key {
                results[*idx] = true;
            } else if receiving_objects
                .contains(&InputKey::VersionedObject { id: **id, version: **version })
            {
                // There could be a more recent version of this object, and the object at the
                // specified version could have already been pruned. In such a case `has_key` will
                // be false, but since this is a receiving object we should mark it as available if
                // we can determine that an object with a version greater than or equal to the
                // specified version exists or was deleted. We will then let mark it as available
                // to let the transaction through so it can fail at execution.
                let is_available = self
                    .get_object(&id.id())
                    .map(|obj| obj.version() >= **version)
                    .unwrap_or(false)
                    || self.fastpath_stream_ended_at_version_or_after(id.id(), **version, epoch);
                results[*idx] = is_available;
            } else {
                // If the object is an already-removed consensus object, mark it as available if the
                // version for that object is in the marker table.
                let is_consensus_stream_ended = self
                    .get_consensus_stream_end_tx_digest(FullObjectKey::new(**id, **version), epoch)
                    .is_some();
                results[*idx] = is_consensus_stream_ended;
            }
        }

        results
    }

    fn multi_input_objects_available_cache_only(&self, keys: &[InputKey]) -> Vec<bool>;

    /// Return the object with version less then or eq to the provided seq number.
    /// This is used by indexer to find the correct version of dynamic field child object.
    /// We do not store the version of the child object, but because of lamport timestamp,
    /// we know the child must have version number less then or eq to the parent.
    fn find_object_lt_or_eq_version(&self, object_id: ObjectID, version: Version)
    -> Option<Object>;

    fn get_lock(&self, obj_ref: ObjectRef, epoch_store: &AuthorityPerEpochStore) -> LockResult;

    // This method is considered "private" - only used by multi_get_objects_with_more_accurate_error_return
    fn _get_live_objref(&self, object_id: ObjectID) -> SomaResult<ObjectRef>;

    // Check that the given set of objects are live at the given version. This is used as a
    // safety check before execution, and could potentially be deleted or changed to a debug_assert
    fn check_owned_objects_are_live(&self, owned_object_refs: &[ObjectRef]) -> SomaResult;

    fn get_system_state_object(&self) -> SomaResult<SystemState>;

    // Marker methods

    /// Get the marker at a specific version
    fn get_marker_value(&self, object_key: FullObjectKey, epoch_id: EpochId)
    -> Option<MarkerValue>;

    /// Get the latest marker for a given object.
    fn get_latest_marker(
        &self,
        object_id: FullObjectID,
        epoch_id: EpochId,
    ) -> Option<(Version, MarkerValue)>;

    /// If the given consensus object stream was ended, return related
    /// version and transaction digest.
    fn get_last_consensus_stream_end_info(
        &self,
        object_id: FullObjectID,
        epoch_id: EpochId,
    ) -> Option<(Version, TransactionDigest)> {
        match self.get_latest_marker(object_id, epoch_id) {
            Some((version, MarkerValue::SharedDeleted(digest))) => Some((version, digest)),
            _ => None,
        }
    }

    /// If the given consensus object stream was ended at the specified version,
    /// return related transaction digest.
    fn get_consensus_stream_end_tx_digest(
        &self,
        object_key: FullObjectKey,
        epoch_id: EpochId,
    ) -> Option<TransactionDigest> {
        match self.get_marker_value(object_key, epoch_id) {
            Some(MarkerValue::SharedDeleted(digest)) => Some(digest),
            _ => None,
        }
    }

    fn have_received_object_at_version(
        &self,
        object_key: FullObjectKey,
        epoch_id: EpochId,
    ) -> bool {
        matches!(self.get_marker_value(object_key, epoch_id), Some(MarkerValue::Received))
    }

    fn fastpath_stream_ended_at_version_or_after(
        &self,
        object_id: ObjectID,
        version: Version,
        epoch_id: EpochId,
    ) -> bool {
        let full_id = FullObjectID::Fastpath(object_id); // function explicitly assumes "fastpath"
        matches!(
            self.get_latest_marker(full_id, epoch_id),
            Some((marker_version, MarkerValue::OwnedDeleted)) if marker_version >= version
        )
    }

    /// Return the watermark for the highest checkpoint for which we've pruned objects.
    fn get_highest_pruned_checkpoint(&self) -> Option<CheckpointSequenceNumber>;

    /// Given a list of input and receiving objects for a transaction,
    /// wait until all of them become available, so that the transaction
    /// can start execution.
    /// `input_and_receiving_keys` contains both input objects and receiving
    /// input objects, including canceled objects.
    /// TODO: Eventually this can return the objects read results,
    /// so that execution does not need to load them again.
    fn notify_read_input_objects<'a>(
        &'a self,
        input_and_receiving_keys: &'a [InputKey],
        receiving_keys: &'a HashSet<InputKey>,
        epoch: EpochId,
    ) -> BoxFuture<'a, ()>;
}
pub trait TransactionCacheRead: Send + Sync {
    fn multi_get_transaction_blocks(
        &self,
        digests: &[TransactionDigest],
    ) -> Vec<Option<Arc<VerifiedTransaction>>>;

    fn get_transaction_block(
        &self,
        digest: &TransactionDigest,
    ) -> Option<Arc<VerifiedTransaction>> {
        self.multi_get_transaction_blocks(&[*digest])
            .pop()
            .expect("multi-get must return correct number of items")
    }

    #[instrument(level = "trace", skip_all)]
    fn get_transactions_and_serialized_sizes(
        &self,
        digests: &[TransactionDigest],
    ) -> SomaResult<Vec<Option<(VerifiedTransaction, usize)>>> {
        let txns = self.multi_get_transaction_blocks(digests);
        txns.into_iter()
            .map(|txn| {
                txn.map(|txn| {
                    // Note: if the transaction is read from the db, we are wasting some
                    // effort relative to reading the raw bytes from the db instead of
                    // calling serialized_size. However, transactions should usually be
                    // fetched from cache.
                    match txn.serialized_size() {
                        Ok(size) => Ok(((*txn).clone(), size)),
                        Err(e) => Err(e),
                    }
                })
                .transpose()
            })
            .collect::<Result<Vec<_>, _>>()
    }

    fn multi_get_executed_effects_digests(
        &self,
        digests: &[TransactionDigest],
    ) -> Vec<Option<TransactionEffectsDigest>>;

    fn is_tx_already_executed(&self, digest: &TransactionDigest) -> bool {
        self.multi_get_executed_effects_digests(&[*digest])
            .pop()
            .expect("multi-get must return correct number of items")
            .is_some()
    }

    fn multi_get_executed_effects(
        &self,
        digests: &[TransactionDigest],
    ) -> Vec<Option<TransactionEffects>> {
        let effects_digests = self.multi_get_executed_effects_digests(digests);
        assert_eq!(effects_digests.len(), digests.len());

        let mut results = vec![None; digests.len()];
        let mut fetch_digests = Vec::with_capacity(digests.len());
        let mut fetch_indices = Vec::with_capacity(digests.len());

        for (i, digest) in effects_digests.into_iter().enumerate() {
            if let Some(digest) = digest {
                fetch_digests.push(digest);
                fetch_indices.push(i);
            }
        }

        let effects = self.multi_get_effects(&fetch_digests);
        for (i, effects) in fetch_indices.into_iter().zip(effects.into_iter()) {
            results[i] = effects;
        }

        results
    }

    fn get_executed_effects(&self, digest: &TransactionDigest) -> Option<TransactionEffects> {
        self.multi_get_executed_effects(&[*digest])
            .pop()
            .expect("multi-get must return correct number of items")
    }

    fn multi_get_effects(
        &self,
        digests: &[TransactionEffectsDigest],
    ) -> Vec<Option<TransactionEffects>>;

    fn get_effects(&self, digest: &TransactionEffectsDigest) -> Option<TransactionEffects> {
        self.multi_get_effects(&[*digest])
            .pop()
            .expect("multi-get must return correct number of items")
    }

    fn notify_read_executed_effects_digests<'a>(
        &'a self,
        digests: &'a [TransactionDigest],
    ) -> BoxFuture<'a, Vec<TransactionEffectsDigest>>;

    /// Wait until the effects of the given transactions are available and return them.
    /// WARNING: If calling this on a transaction that could be reverted, you must be
    /// sure that this function cannot be called during reconfiguration. The best way to
    /// do this is to wrap your future in EpochStore::within_alive_epoch. Holding an
    /// ExecutionLockReadGuard would also prevent reconfig from happening while waiting,
    /// but this is very dangerous, as it could prevent reconfiguration from ever
    /// occurring!
    fn notify_read_executed_effects<'a>(
        &'a self,
        digests: &'a [TransactionDigest],
    ) -> BoxFuture<'a, Vec<TransactionEffects>> {
        async move {
            let digests = self.notify_read_executed_effects_digests(digests).await;
            // once digests are available, effects must be present as well
            self.multi_get_effects(&digests)
                .into_iter()
                .map(|e| e.unwrap_or_else(|| panic!("digests must exist")))
                .collect()
        }
        .boxed()
    }

    /// Get the execution outputs of a mysticeti fastpath certified transaction, if it exists.
    fn get_mysticeti_fastpath_outputs(
        &self,
        tx_digest: &TransactionDigest,
    ) -> Option<Arc<TransactionOutputs>>;

    /// Wait until the outputs of the given transactions are available
    /// in the temporary buffer holding mysticeti fastpath outputs.
    fn notify_read_fastpath_transaction_outputs<'a>(
        &'a self,
        tx_digests: &'a [TransactionDigest],
    ) -> BoxFuture<'a, Vec<Arc<TransactionOutputs>>>;
}
pub trait ExecutionCacheWrite: Send + Sync {
    /// Write the output of a transaction.
    ///
    /// Because of the child object consistency rule (readers that observe parents must observe all
    /// children of that parent, up to the parent's version bound), implementations of this method
    /// must not write any top-level (address-owned or shared) objects before they have written all
    /// of the object-owned objects (i.e. child objects) in the `objects` list.
    ///
    /// In the future, we may modify this method to expose finer-grained information about
    /// parent/child relationships. (This may be especially necessary for distributed object
    /// storage, but is unlikely to be an issue before we tackle that problem).
    ///
    /// This function may evict the mutable input objects (and successfully received objects) of
    /// transaction from the cache, since they cannot be read by any other transaction.
    ///
    /// Any write performed by this method immediately notifies any waiter that has previously
    /// called notify_read_objects_for_execution or notify_read_objects_for_signing for the object
    /// in question.
    fn write_transaction_outputs(&self, epoch_id: EpochId, tx_outputs: Arc<TransactionOutputs>);

    /// Write the output of a Mysticeti fastpath certified transaction.
    /// Such output cannot be written to the dirty cache right away because
    /// the transaction may end up rejected by consensus later. We need to make sure
    /// that it is not visible to any subsequent transaction until we observe it
    /// from consensus or checkpoints.
    fn write_fastpath_transaction_outputs(&self, tx_outputs: Arc<TransactionOutputs>);

    /// Attempt to acquire object locks for all of the owned input locks.
    fn acquire_transaction_locks(
        &self,
        epoch_store: &AuthorityPerEpochStore,
        owned_input_objects: &[ObjectRef],
        tx_digest: TransactionDigest,
        signed_transaction: Option<VerifiedSignedTransaction>,
    ) -> SomaResult;

    /// Write an object entry directly to the cache for testing.
    /// This allows us to write an object without constructing the entire
    /// transaction outputs.
    #[cfg(test)]
    fn write_object_entry_for_test(&self, object: Object);
}
pub trait ExecutionCacheReconfigAPI: Send + Sync {
    fn insert_genesis_object(&self, object: Object);
    fn bulk_insert_genesis_objects(&self, objects: &[Object]);

    fn set_epoch_start_configuration(&self, epoch_start_config: &EpochStartConfiguration);

    fn clear_state_end_of_epoch(&self, execution_guard: &ExecutionLockWriteGuard<'_>);

    // fn checkpoint_db(&self, path: &Path) -> SomaResult;

    /// Reconfigure the cache itself.
    /// TODO: this is only needed for ProxyCache to switch between cache impls. It can be removed
    /// once WritebackCache is the sole cache impl.
    fn reconfigure_cache<'a>(
        &'a self,
        epoch_start_config: &'a EpochStartConfiguration,
    ) -> BoxFuture<'a, ()>;
}

// StateSyncAPI is for writing any data that was not the result of transaction execution,
// but that arrived via state sync. The fact that it came via state sync implies that it
// is certified output, and can be immediately persisted to the store.
pub trait StateSyncAPI: Send + Sync {
    fn insert_transaction_and_effects(
        &self,
        transaction: &VerifiedTransaction,
        transaction_effects: &TransactionEffects,
    );

    fn multi_insert_transaction_and_effects(
        &self,
        transactions_and_effects: &[VerifiedExecutionData],
    );
}

pub trait TestingAPI: Send + Sync {
    fn database_for_testing(&self) -> Arc<AuthorityStore>;
}

macro_rules! implement_storage_traits {
    ($implementor: ident) => {
        impl ObjectStore for $implementor {
            fn get_object(&self, object_id: &ObjectID) -> Option<Object> {
                ObjectCacheRead::get_object(self, object_id)
            }

            fn get_object_by_key(
                &self,
                object_id: &ObjectID,
                version: types::object::Version,
            ) -> Option<Object> {
                ObjectCacheRead::get_object_by_key(self, object_id, version)
            }
        }
    };
}

// Implement traits for a cache implementation that always go directly to the store.
macro_rules! implement_passthrough_traits {
    ($implementor: ident) => {
        impl ExecutionCacheReconfigAPI for $implementor {
            fn insert_genesis_object(&self, object: Object) {
                self.insert_genesis_object_impl(object)
            }

            fn bulk_insert_genesis_objects(&self, objects: &[Object]) {
                self.bulk_insert_genesis_objects_impl(objects)
            }

            fn set_epoch_start_configuration(&self, epoch_start_config: &EpochStartConfiguration) {
                self.store.set_epoch_start_configuration(epoch_start_config).expect("db error");
            }

            fn clear_state_end_of_epoch(&self, execution_guard: &ExecutionLockWriteGuard<'_>) {
                self.clear_state_end_of_epoch_impl(execution_guard)
            }

            // fn checkpoint_db(&self, path: &std::path::Path) -> SomaResult {
            //     self.store.perpetual_tables.checkpoint_db(path)
            // }

            fn reconfigure_cache<'a>(
                &'a self,
                _: &'a EpochStartConfiguration,
            ) -> BoxFuture<'a, ()> {
                // Since we now use WritebackCache directly at startup (if the epoch flag is set),
                // this can be called at reconfiguration time. It is a no-op.
                // TODO: remove this once we completely remove ProxyCache.
                std::future::ready(()).boxed()
            }
        }

        impl TestingAPI for $implementor {
            fn database_for_testing(&self) -> Arc<AuthorityStore> {
                self.store.clone()
            }
        }
    };
}

pub(crate) use implement_passthrough_traits;

implement_storage_traits!(WritebackCache);

pub trait ExecutionCacheAPI:
    ObjectCacheRead
    + ExecutionCacheWrite
    + ExecutionCacheCommit
    + ExecutionCacheReconfigAPI
    + StateSyncAPI
{
}
