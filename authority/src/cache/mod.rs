use std::{collections::HashSet, sync::Arc};

use futures::{future::BoxFuture, FutureExt};
use tracing::debug;
use types::{
    accumulator::AccumulatorStore,
    base::{FullObjectID, SomaAddress},
    committee::EpochId,
    digests::{TransactionDigest, TransactionEffectsDigest},
    effects::TransactionEffects,
    error::{SomaError, SomaResult},
    object::{Object, ObjectID, ObjectRef, Version},
    storage::{
        object_store::ObjectStore, FullObjectKey, InputKey, MarkerValue, ObjectKey,
        ObjectOrTombstone,
    },
    system_state::SystemState,
    transaction::{VerifiedSignedTransaction, VerifiedTransaction},
    tx_outputs::TransactionOutputs,
};
use writeback_cache::WritebackCache;

use crate::{
    epoch_store::AuthorityPerEpochStore,
    state::ExecutionLockWriteGuard,
    store::{AuthorityStore, LockResult},
};

pub(crate) mod cache_types;
pub(crate) mod object_locks;
pub(crate) mod writeback_cache;

#[derive(Clone)]
pub struct ExecutionCacheTraitPointers {
    pub transaction_cache_reader: Arc<dyn TransactionCacheRead>,
    pub cache_writer: Arc<dyn ExecutionCacheWrite>,
    pub object_cache_reader: Arc<dyn ObjectCacheRead>,
    pub object_store: Arc<dyn ObjectStore + Send + Sync>,
    pub reconfig_api: Arc<dyn ExecutionCacheReconfigAPI>,
    pub accumulator_store: Arc<dyn AccumulatorStore>,
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
            + AccumulatorStore
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
            accumulator_store: cache.clone(),
            state_sync_store: cache.clone(),
            cache_commit: cache.clone(),
            testing_api: cache.clone(),
        }
    }
}

pub fn build_execution_cache(store: &Arc<AuthorityStore>) -> ExecutionCacheTraitPointers {
    ExecutionCacheTraitPointers::new(WritebackCache::new(store.clone()).into())
}

pub trait TransactionCacheRead: Send + Sync {
    fn multi_get_transaction_blocks(
        &self,
        digests: &[TransactionDigest],
    ) -> SomaResult<Vec<Option<Arc<VerifiedTransaction>>>>;

    fn get_transaction_block(
        &self,
        digest: &TransactionDigest,
    ) -> SomaResult<Option<Arc<VerifiedTransaction>>> {
        self.multi_get_transaction_blocks(&[*digest])
            .map(|mut blocks| {
                blocks
                    .pop()
                    .expect("multi-get must return correct number of items")
            })
    }

    fn multi_get_executed_effects_digests(
        &self,
        digests: &[TransactionDigest],
    ) -> SomaResult<Vec<Option<TransactionEffectsDigest>>>;

    fn is_tx_already_executed(&self, digest: &TransactionDigest) -> SomaResult<bool> {
        self.multi_get_executed_effects_digests(&[*digest])
            .map(|mut digests| {
                digests
                    .pop()
                    .expect("multi-get must return correct number of items")
                    .is_some()
            })
    }

    fn multi_get_executed_effects(
        &self,
        digests: &[TransactionDigest],
    ) -> SomaResult<Vec<Option<TransactionEffects>>> {
        let effects_digests = self.multi_get_executed_effects_digests(digests)?;
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

        let effects = self.multi_get_effects(&fetch_digests)?;
        for (i, effects) in fetch_indices.into_iter().zip(effects.into_iter()) {
            results[i] = effects;
        }

        Ok(results)
    }

    fn get_executed_effects(
        &self,
        digest: &TransactionDigest,
    ) -> SomaResult<Option<TransactionEffects>> {
        self.multi_get_executed_effects(&[*digest])
            .map(|mut effects| {
                effects
                    .pop()
                    .expect("multi-get must return correct number of items")
            })
    }

    fn multi_get_effects(
        &self,
        digests: &[TransactionEffectsDigest],
    ) -> SomaResult<Vec<Option<TransactionEffects>>>;

    fn get_effects(
        &self,
        digest: &TransactionEffectsDigest,
    ) -> SomaResult<Option<TransactionEffects>> {
        self.multi_get_effects(&[*digest]).map(|mut effects| {
            effects
                .pop()
                .expect("multi-get must return correct number of items")
        })
    }

    fn notify_read_executed_effects_digests<'a>(
        &'a self,
        digests: &'a [TransactionDigest],
    ) -> BoxFuture<'a, SomaResult<Vec<TransactionEffectsDigest>>>;

    fn notify_read_executed_effects<'a>(
        &'a self,
        digests: &'a [TransactionDigest],
    ) -> BoxFuture<'a, SomaResult<Vec<TransactionEffects>>> {
        async move {
            let digests = self.notify_read_executed_effects_digests(digests).await?;
            debug!("notify_read_executed_effects: {:?}", digests);
            // once digests are available, effects must be present as well
            self.multi_get_effects(&digests).map(|effects| {
                effects
                    .into_iter()
                    .map(|e| e.expect("digests must exist"))
                    .collect()
            })
        }
        .boxed()
    }
}

pub trait ExecutionCacheWrite: Send + Sync {
    /// Write the output of a transaction.
    fn write_transaction_outputs(
        &self,
        epoch_id: EpochId,
        tx_outputs: Arc<TransactionOutputs>,
    ) -> BoxFuture<'_, SomaResult>;

    /// Attempt to acquire object locks for all of the owned input locks.
    fn acquire_transaction_locks(
        &self,
        epoch_store: &AuthorityPerEpochStore,
        owned_input_objects: &[ObjectRef],
        tx_digest: TransactionDigest,
        signed_transaction: VerifiedSignedTransaction,
    ) -> SomaResult;
}

pub trait ExecutionCacheCommit: Send + Sync {
    /// Durably commit the outputs of the given transactions to the database.
    /// Will be called by CheckpointExecutor to ensure that transaction outputs are
    /// written durably before marking a checkpoint as finalized.
    fn commit_transaction_outputs<'a>(
        &'a self,
        epoch: EpochId,
        digests: &'a [TransactionDigest],
    ) -> BoxFuture<'a, SomaResult>;

    /// Durably commit transactions (but not their outputs) to the database.
    /// Called before writing a locally built checkpoint to the CheckpointStore, so that
    /// the inputs of the checkpoint cannot be lost.
    /// These transactions are guaranteed to be final unless this validator
    /// forks (i.e. constructs a checkpoint which will never be certified). In this case
    /// some non-final transactions could be left in the database.
    ///
    /// This is an intermediate solution until we delay commits to the epoch db. After
    /// we have done that, crash recovery will be done by re-processing consensus commits
    /// and pending_consensus_transactions, and this method can be removed.
    fn persist_transactions<'a>(
        &'a self,
        digests: &'a [TransactionDigest],
    ) -> BoxFuture<'a, SomaResult>;
}

pub trait ObjectCacheRead: Send + Sync {
    fn get_object(&self, id: &ObjectID) -> SomaResult<Option<Object>>;

    fn get_objects(&self, objects: &[ObjectID]) -> SomaResult<Vec<Option<Object>>> {
        let mut ret = Vec::with_capacity(objects.len());
        for object_id in objects {
            ret.push(self.get_object(object_id)?);
        }
        Ok(ret)
    }

    fn get_latest_object_ref_or_tombstone(
        &self,
        object_id: ObjectID,
    ) -> SomaResult<Option<ObjectRef>>;

    fn get_latest_object_or_tombstone(
        &self,
        object_id: ObjectID,
    ) -> SomaResult<Option<(ObjectKey, ObjectOrTombstone)>>;

    fn get_object_by_key(
        &self,
        object_id: &ObjectID,
        version: Version,
    ) -> SomaResult<Option<Object>>;

    fn multi_get_objects_by_key(
        &self,
        object_keys: &[ObjectKey],
    ) -> SomaResult<Vec<Option<Object>>>;

    fn object_exists_by_key(&self, object_id: &ObjectID, version: Version) -> SomaResult<bool>;

    fn multi_object_exists_by_key(&self, object_keys: &[ObjectKey]) -> SomaResult<Vec<bool>>;

    /// Return the object with version less then or eq to the provided seq number.
    /// This is used by indexer to find the correct version of dynamic field child object.
    /// We do not store the version of the child object, but because of lamport timestamp,
    /// we know the child must have version number less then or eq to the parent.
    fn find_object_lt_or_eq_version(
        &self,
        object_id: ObjectID,
        version: Version,
    ) -> SomaResult<Option<Object>>;

    fn get_lock(&self, obj_ref: ObjectRef, epoch_store: &AuthorityPerEpochStore) -> LockResult;

    fn get_system_state_object(&self) -> SomaResult<SystemState>;

    // Check that the given set of objects are live at the given version. This is used as a
    // safety check before execution, and could potentially be deleted or changed to a debug_assert
    fn check_owned_objects_are_live(&self, owned_object_refs: &[ObjectRef]) -> SomaResult;

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
        let objects = self.multi_get_objects_by_key(
            &object_refs.iter().map(ObjectKey::from).collect::<Vec<_>>(),
        )?;
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

    /// Used by transaction manager to determine if input objects are ready. Distinct from multi_get_object_by_key
    /// because it also consults markers to handle the case where an object will never become available (e.g.
    /// because it has been received by some other transaction already).
    fn multi_input_objects_available(
        &self,
        keys: &[InputKey],
        receiving_objects: HashSet<InputKey>,
        epoch: EpochId,
    ) -> Vec<bool> {
        let (keys_with_version, keys_without_version): (Vec<_>, Vec<_>) = keys
            .iter()
            .enumerate()
            .partition(|(_, key)| key.version().is_some());

        let mut versioned_results = vec![];

        // Get existence results and handle the Result
        let existence_results = self
            .multi_object_exists_by_key(
                &keys_with_version
                    .iter()
                    .map(|(_, k)| ObjectKey(k.id().id(), k.version().unwrap()))
                    .collect::<Vec<_>>(),
            )
            .expect("Failed to check if objects exist"); // Unwrap the Result

        // Iterate through both collections together
        for (i, (idx, input_key)) in keys_with_version.iter().enumerate() {
            // Get the corresponding has_key value
            let has_key = existence_results.get(i).cloned().unwrap_or(false);

            assert!(
                input_key.version().is_none() || input_key.version().unwrap().is_valid(),
                "Shared objects in cancelled transaction should always be available immediately, 
                 but it appears that transaction manager is waiting for {:?} to become available",
                input_key
            );

            // Rest of the function remains the same...
            if has_key {
                versioned_results.push((*idx, true))
            } else if receiving_objects.contains(input_key) {
                let is_available = self
                    .get_object(&input_key.id().id())
                    .map(|obj| obj.unwrap().version() >= input_key.version().unwrap())
                    .unwrap_or(false)
                    || self.have_deleted_fastpath_object_at_version_or_after(
                        input_key.id().id(),
                        input_key.version().unwrap(),
                        epoch,
                    );
                versioned_results.push((*idx, is_available));
            } else if self
                .get_deleted_shared_object_previous_tx_digest(
                    FullObjectKey::new(input_key.id(), input_key.version().unwrap()),
                    epoch,
                )
                .is_some()
            {
                versioned_results.push((*idx, true));
            } else {
                versioned_results.push((*idx, false));
            }
        }

        let unversioned_results = keys_without_version.into_iter().map(|(idx, key)| {
            let is_available = self
                .get_latest_object_ref_or_tombstone(key.id().id())
                .map_or(false, |opt| opt.map_or(false, |entry| entry.2.is_alive()));
            (idx, is_available)
        });

        let mut results = versioned_results
            .into_iter()
            .chain(unversioned_results)
            .collect::<Vec<_>>();
        results.sort_by_key(|(idx, _)| *idx);
        results.into_iter().map(|(_, result)| result).collect()
    }

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

    /// If the shared object was deleted, return deletion info for the current live version
    fn get_last_shared_object_deletion_info(
        &self,
        object_id: FullObjectID,
        epoch_id: EpochId,
    ) -> Option<(Version, TransactionDigest)> {
        match self.get_latest_marker(object_id, epoch_id) {
            Some((version, MarkerValue::SharedDeleted(digest))) => Some((version, digest)),
            _ => None,
        }
    }

    /// If the shared object was deleted, return deletion info for the specified version.
    fn get_deleted_shared_object_previous_tx_digest(
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
        matches!(
            self.get_marker_value(object_key, epoch_id),
            Some(MarkerValue::Received)
        )
    }

    fn have_deleted_fastpath_object_at_version_or_after(
        &self,
        object_id: ObjectID,
        version: Version,
        epoch_id: EpochId,
    ) -> bool {
        let full_id = FullObjectID::Fastpath(object_id); // function explicilty assumes "fastpath"
        matches!(
            self.get_latest_marker(full_id, epoch_id),
            Some((marker_version, MarkerValue::OwnedDeleted)) if marker_version >= version
        )
    }

    // This method is considered "private" - only used by multi_get_objects_with_more_accurate_error_return
    fn _get_live_objref(&self, object_id: ObjectID) -> SomaResult<ObjectRef>;

    /// Get gas objects (coins) owned by an address, up to the specified limit
    fn get_gas_objects_owned_by_address(
        &self,
        address: SomaAddress,
        limit: Option<usize>,
    ) -> SomaResult<Vec<ObjectRef>>;
}

// StateSyncAPI is for writing any data that was not the result of transaction execution,
// but that arrived via state sync. The fact that it came via state sync implies that it
// is certified output, and can be immediately persisted to the store.
pub trait StateSyncAPI: Send + Sync {
    fn multi_insert_transactions(&self, transactions: &[VerifiedTransaction]) -> SomaResult;
}

pub trait ExecutionCacheReconfigAPI: Send + Sync {
    fn insert_genesis_object(&self, object: Object);
    fn bulk_insert_genesis_objects(&self, objects: &[Object]);
    fn clear_state_end_of_epoch(&self, execution_guard: &ExecutionLockWriteGuard<'_>);

    // TODO: Implement other ReconfigAPI methods
    // fn set_epoch_start_configuration(&self, epoch_start_config: &EpochStartConfiguration);
    // fn revert_state_update(&self, digest: &TransactionDigest);
    // fn checkpoint_db(&self, path: &Path) -> SomaResult;
}

pub trait TestingAPI: Send + Sync {
    fn database_for_testing(&self) -> Arc<AuthorityStore>;
}
