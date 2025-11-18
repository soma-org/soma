use std::{
    collections::{BTreeMap, HashMap},
    iter,
    sync::Arc,
};

use fastcrypto::hash::{HashFunction, Sha3_256};
use futures::future::Either;
use itertools::izip;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use store::{
    rocks::{DBBatch, DBMap},
    Map as _, TypedStoreError,
};
use tokio::sync::{RwLockReadGuard, RwLockWriteGuard};
use tracing::{debug, error, info, instrument, trace};
use types::{
    base::{FullObjectID, VerifiedExecutionData},
    checkpoints::{CheckpointSequenceNumber, GlobalStateHash},
    committee::{Committee, EpochId},
    config::node_config::{AuthorityStorePruningConfig, NodeConfig},
    digests::{ObjectDigest, TransactionDigest, TransactionEffectsDigest},
    effects::{TransactionEffects, TransactionEffectsAPI},
    envelope::Message,
    error::{SomaError, SomaResult},
    genesis::Genesis,
    mutex_table::{Lock, MutexGuard, MutexTable, RwLockGuard, RwLockTable},
    object::{self, LiveObject, Object, ObjectID, ObjectRef, Version},
    protocol::ProtocolVersion,
    storage::{
        object_store::ObjectStore, FullObjectKey, MarkerValue, ObjectKey, ObjectOrTombstone,
    },
    system_state::{get_system_state, SystemState, SystemStateTrait},
    transaction::{VerifiedExecutableTransaction, VerifiedSignedTransaction, VerifiedTransaction},
    transaction_outputs::TransactionOutputs,
};

use crate::{
    authority_per_epoch_store::AuthorityPerEpochStore,
    authority_store_pruner::{AuthorityStorePruner, EPOCH_DURATION_MS_FOR_TESTING},
    authority_store_tables::{get_store_object, AuthorityPerpetualTables, StoreObject},
    checkpoints::CheckpointStore,
    global_state_hasher::GlobalStateHashStore,
    rpc_index::RpcIndexStore,
    start_epoch::EpochStartConfiguration,
};

const NUM_SHARDS: usize = 4096;

pub struct AuthorityStore {
    /// Internal vector of locks to manage concurrent writes to the database
    mutex_table: MutexTable<ObjectDigest>,

    pub(crate) perpetual_tables: Arc<AuthorityPerpetualTables>,
}

pub type ExecutionLockReadGuard<'a> = RwLockReadGuard<'a, EpochId>;
pub type ExecutionLockWriteGuard<'a> = RwLockWriteGuard<'a, EpochId>;

pub type LockResult = SomaResult<ObjectLockStatus>;

#[derive(Debug, PartialEq, Eq)]
pub enum ObjectLockStatus {
    Initialized,
    LockedToTx { locked_by_tx: TransactionDigest },
    LockedAtDifferentVersion { locked_ref: ObjectRef },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LockDetails {
    pub epoch: EpochId,
    pub tx_digest: TransactionDigest,
}

impl AuthorityStore {
    /// Open an authority store by directory path.
    /// If the store is empty, initialize it using genesis.
    pub async fn open(
        perpetual_tables: Arc<AuthorityPerpetualTables>,
        genesis: &Genesis,
        config: &NodeConfig,
    ) -> SomaResult<Arc<Self>> {
        let epoch_start_configuration = if perpetual_tables.database_is_empty()? {
            info!("Creating new epoch start config from genesis");

            let epoch_start_configuration = EpochStartConfiguration::new(
                genesis.sui_system_object().into_epoch_start_state(),
                *genesis.checkpoint().digest(),
            )?;
            perpetual_tables.set_epoch_start_configuration(&epoch_start_configuration)?;
            epoch_start_configuration
        } else {
            info!("Loading epoch start config from DB");
            perpetual_tables
                .epoch_start_configuration
                .get(&())?
                .expect("Epoch start configuration must be set in non-empty DB")
        };
        let cur_epoch = perpetual_tables.get_recovery_epoch_at_restart()?;
        info!("Epoch start config: {:?}", epoch_start_configuration);
        info!("Cur epoch: {:?}", cur_epoch);
        let this = Self::open_inner(genesis, perpetual_tables).await?;
        this.update_epoch_flags_metrics(&[], epoch_start_configuration.flags());
        Ok(this)
    }

    // NB: This must only be called at time of reconfiguration. We take the execution lock write
    // guard as an argument to ensure that this is the case.
    pub fn clear_object_per_epoch_marker_table(
        &self,
        _execution_guard: &ExecutionLockWriteGuard<'_>,
    ) -> SomaResult<()> {
        // We can safely delete all entries in the per epoch marker table since this is only called
        // at epoch boundaries (during reconfiguration). Therefore any entries that currently
        // exist can be removed. Because of this we can use the `schedule_delete_all` method.
        self.perpetual_tables
            .object_per_epoch_marker_table
            .schedule_delete_all()?;
        Ok(self
            .perpetual_tables
            .object_per_epoch_marker_table_v2
            .schedule_delete_all()?)
    }

    pub async fn open_with_committee_for_testing(
        perpetual_tables: Arc<AuthorityPerpetualTables>,
        committee: &Committee,
        genesis: &Genesis,
    ) -> SomaResult<Arc<Self>> {
        // TODO: Since we always start at genesis, the committee should be technically the same
        // as the genesis committee.
        assert_eq!(committee.epoch, 0);
        Self::open_inner(genesis, perpetual_tables).await
    }

    async fn open_inner(
        genesis: &Genesis,
        perpetual_tables: Arc<AuthorityPerpetualTables>,
    ) -> SomaResult<Arc<Self>> {
        let store = Arc::new(Self {
            mutex_table: MutexTable::new(NUM_SHARDS),
            perpetual_tables,
        });
        // Only initialize an empty database.
        if store
            .database_is_empty()
            .expect("Database read should not fail at init.")
        {
            store
                .bulk_insert_genesis_objects(genesis.objects())
                .expect("Cannot bulk insert genesis objects");

            // insert txn and effects of genesis
            let transaction = VerifiedTransaction::new_unchecked(genesis.transaction().clone());

            store
                .perpetual_tables
                .transactions
                .insert(transaction.digest(), transaction.serializable_ref())
                .unwrap();

            store
                .perpetual_tables
                .effects
                .insert(&genesis.effects().digest(), genesis.effects())
                .unwrap();
            // We don't insert the effects to executed_effects yet because the genesis tx hasn't but will be executed.
            // This is important for fullnodes to be able to generate indexing data right now.
        }

        Ok(store)
    }

    /// Open authority store without any operations that require
    /// genesis, such as constructing EpochStartConfiguration
    /// or inserting genesis objects.
    pub fn open_no_genesis(
        perpetual_tables: Arc<AuthorityPerpetualTables>,
        enable_epoch_sui_conservation_check: bool,
    ) -> SomaResult<Arc<Self>> {
        let store = Arc::new(Self {
            mutex_table: MutexTable::new(NUM_SHARDS),
            perpetual_tables,
        });
        Ok(store)
    }

    pub fn get_recovery_epoch_at_restart(&self) -> SomaResult<EpochId> {
        self.perpetual_tables.get_recovery_epoch_at_restart()
    }

    pub fn get_effects(
        &self,
        effects_digest: &TransactionEffectsDigest,
    ) -> SomaResult<Option<TransactionEffects>> {
        Ok(self.perpetual_tables.effects.get(effects_digest)?)
    }

    /// Returns true if we have an effects structure for this transaction digest
    pub fn effects_exists(&self, effects_digest: &TransactionEffectsDigest) -> SomaResult<bool> {
        self.perpetual_tables
            .effects
            .contains_key(effects_digest)
            .map_err(|e| e.into())
    }

    pub fn get_unchanged_loaded_runtime_objects(
        &self,
        digest: &TransactionDigest,
    ) -> Result<Option<Vec<ObjectKey>>, TypedStoreError> {
        self.perpetual_tables
            .unchanged_loaded_runtime_objects
            .get(digest)
    }

    pub fn multi_get_effects<'a>(
        &self,
        effects_digests: impl Iterator<Item = &'a TransactionEffectsDigest>,
    ) -> Result<Vec<Option<TransactionEffects>>, TypedStoreError> {
        self.perpetual_tables.effects.multi_get(effects_digests)
    }

    pub fn get_executed_effects(
        &self,
        tx_digest: &TransactionDigest,
    ) -> Result<Option<TransactionEffects>, TypedStoreError> {
        let effects_digest = self.perpetual_tables.executed_effects.get(tx_digest)?;
        match effects_digest {
            Some(digest) => Ok(self.perpetual_tables.effects.get(&digest)?),
            None => Ok(None),
        }
    }

    /// Given a list of transaction digests, returns a list of the corresponding effects only if they have been
    /// executed. For transactions that have not been executed, None is returned.
    pub fn multi_get_executed_effects_digests(
        &self,
        digests: &[TransactionDigest],
    ) -> Result<Vec<Option<TransactionEffectsDigest>>, TypedStoreError> {
        self.perpetual_tables.executed_effects.multi_get(digests)
    }

    /// Given a list of transaction digests, returns a list of the corresponding effects only if they have been
    /// executed. For transactions that have not been executed, None is returned.
    pub fn multi_get_executed_effects(
        &self,
        digests: &[TransactionDigest],
    ) -> Result<Vec<Option<TransactionEffects>>, TypedStoreError> {
        let executed_effects_digests = self.perpetual_tables.executed_effects.multi_get(digests)?;
        let effects = self.multi_get_effects(executed_effects_digests.iter().flatten())?;
        let mut tx_to_effects_map = effects
            .into_iter()
            .flatten()
            .map(|effects| (*effects.transaction_digest(), effects))
            .collect::<HashMap<_, _>>();
        Ok(digests
            .iter()
            .map(|digest| tx_to_effects_map.remove(digest))
            .collect())
    }

    pub fn is_tx_already_executed(&self, digest: &TransactionDigest) -> SomaResult<bool> {
        Ok(self
            .perpetual_tables
            .executed_effects
            .contains_key(digest)?)
    }

    pub fn get_marker_value(
        &self,
        object_key: FullObjectKey,
        epoch_id: EpochId,
    ) -> SomaResult<Option<MarkerValue>> {
        Ok(self
            .perpetual_tables
            .object_per_epoch_marker_table_v2
            .get(&(epoch_id, object_key))?)
    }

    pub fn get_latest_marker(
        &self,
        object_id: FullObjectID,
        epoch_id: EpochId,
    ) -> SomaResult<Option<(Version, MarkerValue)>> {
        let min_key = (epoch_id, FullObjectKey::min_for_id(&object_id));
        let max_key = (epoch_id, FullObjectKey::max_for_id(&object_id));

        let marker_entry = self
            .perpetual_tables
            .object_per_epoch_marker_table_v2
            .reversed_safe_iter_with_bounds(Some(min_key), Some(max_key))?
            .next();
        match marker_entry {
            Some(Ok(((epoch, key), marker))) => {
                // because of the iterator bounds these cannot fail
                assert_eq!(epoch, epoch_id);
                assert_eq!(key.id(), object_id);
                Ok(Some((key.version(), marker)))
            }
            Some(Err(e)) => Err(e.into()),
            None => Ok(None),
        }
    }

    /// Returns future containing the state hash for the given epoch
    /// once available
    pub async fn notify_read_root_state_hash(
        &self,
        epoch: EpochId,
    ) -> SomaResult<(CheckpointSequenceNumber, GlobalStateHash)> {
        // We need to register waiters _before_ reading from the database to avoid race conditions
        let registration = self.root_state_notify_read.register_one(&epoch);
        let hash = self.perpetual_tables.root_state_hash_by_epoch.get(&epoch)?;

        let result = match hash {
            // Note that Some() clause also drops registration that is already fulfilled
            Some(ready) => Either::Left(futures::future::ready(ready)),
            None => Either::Right(registration),
        }
        .await;

        Ok(result)
    }

    /// Returns true if there are no objects in the database
    pub fn database_is_empty(&self) -> SomaResult<bool> {
        self.perpetual_tables.database_is_empty()
    }

    /// A function that acquires all locks associated with the objects (in order to avoid deadlocks).
    fn acquire_locks(&self, input_objects: &[ObjectRef]) -> Vec<MutexGuard> {
        self.mutex_table
            .acquire_locks(input_objects.iter().map(|(_, _, digest)| *digest))
    }

    pub fn object_exists_by_key(&self, object_id: &ObjectID, version: Version) -> SomaResult<bool> {
        Ok(self
            .perpetual_tables
            .objects
            .contains_key(&ObjectKey(*object_id, version))?)
    }

    pub fn multi_object_exists_by_key(&self, object_keys: &[ObjectKey]) -> SomaResult<Vec<bool>> {
        Ok(self
            .perpetual_tables
            .objects
            .multi_contains_keys(object_keys.to_vec())?
            .into_iter()
            .collect())
    }

    fn get_object_ref_prior_to_key(
        &self,
        object_id: &ObjectID,
        version: Version,
    ) -> Result<Option<ObjectRef>, SomaError> {
        let Some(prior_version) = version.one_before() else {
            return Ok(None);
        };
        let mut iterator = self
            .perpetual_tables
            .objects
            .reversed_safe_iter_with_bounds(
                Some(ObjectKey::min_for_id(object_id)),
                Some(ObjectKey(*object_id, prior_version)),
            )?;

        if let Some((object_key, value)) = iterator.next().transpose()? {
            if object_key.0 == *object_id {
                return Ok(Some(
                    self.perpetual_tables.object_reference(&object_key, value)?,
                ));
            }
        }
        Ok(None)
    }

    pub fn multi_get_objects_by_key(
        &self,
        object_keys: &[ObjectKey],
    ) -> Result<Vec<Option<Object>>, SomaError> {
        let wrappers = self
            .perpetual_tables
            .objects
            .multi_get(object_keys.to_vec())?;
        let mut ret = vec![];

        for (idx, w) in wrappers.into_iter().enumerate() {
            ret.push(
                w.map(|object| self.perpetual_tables.object(&object_keys[idx], object))
                    .transpose()?
                    .flatten(),
            );
        }
        Ok(ret)
    }

    /// Get many objects
    pub fn get_objects(&self, objects: &[ObjectID]) -> Result<Vec<Option<Object>>, SomaError> {
        let mut result = Vec::new();
        for id in objects {
            result.push(self.get_object(id));
        }
        Ok(result)
    }

    // Methods to mutate the store

    /// Insert a genesis object.
    /// TODO: delete this method entirely (still used by authority_tests.rs)
    pub(crate) fn insert_genesis_object(&self, object: Object) -> SomaResult {
        // We only side load objects with a genesis parent transaction.
        debug_assert!(object.previous_transaction == TransactionDigest::genesis_marker());
        let object_ref = object.compute_object_reference();
        self.insert_object_direct(object_ref, &object)
    }

    /// Insert an object directly into the store, and also update relevant tables
    /// NOTE: does not handle transaction lock.
    /// This is used to insert genesis objects
    fn insert_object_direct(&self, object_ref: ObjectRef, object: &Object) -> SomaResult {
        let mut write_batch = self.perpetual_tables.objects.batch();

        // Insert object
        let store_object = get_store_object(object.clone());
        write_batch.insert_batch(
            &self.perpetual_tables.objects,
            std::iter::once((ObjectKey::from(object_ref), store_object)),
        )?;

        // Update the index
        if object.get_single_owner().is_some() {
            // Only initialize lock for address owned objects.
            if !object.is_child_object() {
                self.initialize_live_object_markers_impl(&mut write_batch, &[object_ref], false)?;
            }
        }

        write_batch.write()?;

        Ok(())
    }

    /// This function should only be used for initializing genesis and should remain private.
    #[instrument(level = "debug", skip_all)]
    pub(crate) fn bulk_insert_genesis_objects(&self, objects: &[Object]) -> SomaResult<()> {
        let mut batch = self.perpetual_tables.objects.batch();
        let ref_and_objects: Vec<_> = objects
            .iter()
            .map(|o| (o.compute_object_reference(), o))
            .collect();

        batch.insert_batch(
            &self.perpetual_tables.objects,
            ref_and_objects
                .iter()
                .map(|(oref, o)| (ObjectKey::from(oref), get_store_object((*o).clone()))),
        )?;

        let non_child_object_refs: Vec<_> = ref_and_objects
            .iter()
            .filter(|(_, object)| !object.is_child_object())
            .map(|(oref, _)| *oref)
            .collect();

        self.initialize_live_object_markers_impl(
            &mut batch,
            &non_child_object_refs,
            false, // is_force_reset
        )?;

        batch.write()?;

        Ok(())
    }

    pub fn bulk_insert_live_objects(
        perpetual_db: &AuthorityPerpetualTables,
        live_objects: impl Iterator<Item = LiveObject>,
        expected_sha3_digest: &[u8; 32],
    ) -> SomaResult<()> {
        let mut hasher = Sha3_256::default();
        let mut batch = perpetual_db.objects.batch();
        let mut written = 0usize;
        const MAX_BATCH_SIZE: usize = 100_000;
        for object in live_objects {
            hasher.update(object.object_reference().2.inner());
            match object {
                LiveObject::Normal(object) => {
                    let store_object_wrapper = get_store_object(object.clone());
                    batch.insert_batch(
                        &perpetual_db.objects,
                        std::iter::once((
                            ObjectKey::from(object.compute_object_reference()),
                            store_object_wrapper,
                        )),
                    )?;

                    Self::initialize_live_object_markers(
                        &perpetual_db.live_owned_object_markers,
                        &mut batch,
                        &[object.compute_object_reference()],
                        false, // is_force_reset
                    )?;
                }
            }
            written += 1;
            if written > MAX_BATCH_SIZE {
                batch.write()?;
                batch = perpetual_db.objects.batch();
                written = 0;
            }
        }
        let sha3_digest = hasher.finalize().digest;
        if *expected_sha3_digest != sha3_digest {
            error!(
                "Sha does not match! expected: {:?}, actual: {:?}",
                expected_sha3_digest, sha3_digest
            );
            return Err(SomaError::from("Sha does not match"));
        }
        batch.write()?;
        Ok(())
    }

    pub fn set_epoch_start_configuration(
        &self,
        epoch_start_configuration: &EpochStartConfiguration,
    ) -> SomaResult {
        self.perpetual_tables
            .set_epoch_start_configuration(epoch_start_configuration)?;
        Ok(())
    }

    pub fn get_epoch_start_configuration(&self) -> SomaResult<Option<EpochStartConfiguration>> {
        Ok(self.perpetual_tables.epoch_start_configuration.get(&())?)
    }

    /// Updates the state resulting from the execution of a certificate.
    ///
    /// Internally it checks that all locks for active inputs are at the correct
    /// version, and then writes objects, certificates, parents and clean up locks atomically.
    #[instrument(level = "debug", skip_all)]
    pub fn build_db_batch(
        &self,
        epoch_id: EpochId,
        tx_outputs: &[Arc<TransactionOutputs>],
    ) -> SomaResult<DBBatch> {
        let mut written = Vec::with_capacity(tx_outputs.len());
        for outputs in tx_outputs {
            written.extend(outputs.written.values().cloned());
        }

        let mut write_batch = self.perpetual_tables.transactions.batch();
        for outputs in tx_outputs {
            self.write_one_transaction_outputs(&mut write_batch, epoch_id, outputs)?;
        }

        trace!(
            "built batch for committed transactions: {:?}",
            tx_outputs
                .iter()
                .map(|tx| tx.transaction.digest())
                .collect::<Vec<_>>()
        );

        Ok(write_batch)
    }

    fn write_one_transaction_outputs(
        &self,
        write_batch: &mut DBBatch,
        epoch_id: EpochId,
        tx_outputs: &TransactionOutputs,
    ) -> SomaResult {
        let TransactionOutputs {
            transaction,
            effects,
            markers,

            deleted,
            written,

            unchanged_loaded_runtime_objects,
            locks_to_delete,
            new_locks_to_init,
            ..
        } = tx_outputs;

        let effects_digest = effects.digest();
        let transaction_digest = transaction.digest();
        // effects must be inserted before the corresponding dependent entries
        // because they carry epoch information necessary for correct pruning via relocation filters
        write_batch
            .insert_batch(
                &self.perpetual_tables.effects,
                [(effects_digest, effects.clone())],
            )?
            .insert_batch(
                &self.perpetual_tables.executed_effects,
                [(transaction_digest, effects_digest)],
            )?;

        // Store the certificate indexed by transaction digest
        write_batch.insert_batch(
            &self.perpetual_tables.transactions,
            iter::once((transaction_digest, transaction.serializable_ref())),
        )?;

        // Add batched writes for objects and locks.
        write_batch.insert_batch(
            &self.perpetual_tables.object_per_epoch_marker_table_v2,
            markers
                .iter()
                .map(|(key, marker_value)| ((epoch_id, *key), *marker_value)),
        )?;
        write_batch.insert_batch(
            &self.perpetual_tables.objects,
            deleted.iter().map(|key| (key, StoreObject::Deleted)),
        )?;

        // Insert each output object into the stores
        let new_objects = written.iter().map(|(id, new_object)| {
            let version = new_object.version();
            trace!(?id, ?version, "writing object");
            let store_object = get_store_object(new_object.clone());
            (ObjectKey(*id, version), store_object)
        });

        write_batch.insert_batch(&self.perpetual_tables.objects, new_objects)?;

        // Write unchanged_loaded_runtime_objects
        if !unchanged_loaded_runtime_objects.is_empty() {
            write_batch.insert_batch(
                &self.perpetual_tables.unchanged_loaded_runtime_objects,
                [(transaction_digest, unchanged_loaded_runtime_objects)],
            )?;
        }

        self.initialize_live_object_markers_impl(write_batch, new_locks_to_init, false)?;

        // Note: deletes locks for received objects as well (but not for objects that were in
        // `Receiving` arguments which were not received)
        self.delete_live_object_markers(write_batch, locks_to_delete)?;

        debug!(effects_digest = ?effects.digest(), "commit_certificate finished");

        Ok(())
    }

    /// Commits transactions only (not effects or other transaction outputs) to the db.
    /// See ExecutionCache::persist_transaction for more info
    pub(crate) fn persist_transaction(&self, tx: &VerifiedExecutableTransaction) -> SomaResult {
        let mut batch = self.perpetual_tables.transactions.batch();
        batch.insert_batch(
            &self.perpetual_tables.transactions,
            [(tx.digest(), tx.clone().into_unsigned().serializable_ref())],
        )?;
        batch.write()?;
        Ok(())
    }

    pub fn acquire_transaction_locks(
        &self,
        epoch_store: &AuthorityPerEpochStore,
        owned_input_objects: &[ObjectRef],
        tx_digest: TransactionDigest,
        signed_transaction: Option<VerifiedSignedTransaction>,
    ) -> SomaResult {
        let epoch = epoch_store.epoch();
        // Other writers may be attempting to acquire locks on the same objects, so a mutex is
        // required.
        // TODO: replace with optimistic db_transactions (i.e. set lock to tx if none)
        let _mutexes = self.acquire_locks(owned_input_objects);

        trace!(?owned_input_objects, "acquire_locks");
        let mut locks_to_write = Vec::new();

        let live_object_markers = self
            .perpetual_tables
            .live_owned_object_markers
            .multi_get(owned_input_objects)?;

        let epoch_tables = epoch_store.tables()?;

        let locks = epoch_tables.multi_get_locked_transactions(owned_input_objects)?;

        assert_eq!(locks.len(), live_object_markers.len());

        for (live_marker, lock, obj_ref) in izip!(
            live_object_markers.into_iter(),
            locks.into_iter(),
            owned_input_objects
        ) {
            let Some(live_marker) = live_marker else {
                let latest_lock = self.get_latest_live_version_for_object_id(obj_ref.0)?;
                SomaError::ObjectVersionUnavailableForConsumption {
                    provided_obj_ref: *obj_ref,
                    current_version: latest_lock.1,
                }
            };

            let live_marker = live_marker.map(|l| l.migrate().into_inner());

            if let Some(LockDetails {
                epoch: previous_epoch,
                ..
            }) = &live_marker
            {
                // this must be from a prior epoch, because we no longer write LockDetails to
                // owned_object_transaction_locks
                assert!(
                    previous_epoch < &epoch,
                    "lock for {:?} should be from a prior epoch",
                    obj_ref
                );
            }

            if let Some(previous_tx_digest) = &lock {
                if previous_tx_digest == &tx_digest {
                    // no need to re-write lock
                    continue;
                } else {
                    // TODO: add metrics here
                    info!(prev_tx_digest = ?previous_tx_digest,
                          cur_tx_digest = ?tx_digest,
                          "Cannot acquire lock: conflicting transaction!");
                    return Err(SomaError::ObjectLockConflict {
                        obj_ref: *obj_ref,
                        pending_transaction: *previous_tx_digest,
                    }
                    .into());
                }
            }

            locks_to_write.push((*obj_ref, tx_digest));
        }

        if !locks_to_write.is_empty() {
            trace!(?locks_to_write, "Writing locks");
            epoch_tables.write_transaction_locks(signed_transaction, locks_to_write.into_iter())?;
        }

        Ok(())
    }

    /// Gets ObjectLockInfo that represents state of lock on an object.
    /// Returns UserInputError::ObjectNotFound if cannot find lock record for this object
    pub(crate) fn get_lock(
        &self,
        obj_ref: ObjectRef,
        epoch_store: &AuthorityPerEpochStore,
    ) -> LockResult {
        if self
            .perpetual_tables
            .live_owned_object_markers
            .get(&obj_ref)?
            .is_none()
        {
            return Ok(ObjectLockStatus::LockedAtDifferentVersion {
                locked_ref: self.get_latest_live_version_for_object_id(obj_ref.0)?,
            });
        }

        let tables = epoch_store.tables()?;
        let epoch_id = epoch_store.epoch();

        if let Some(tx_digest) = tables.get_locked_transaction(&obj_ref)? {
            Ok(ObjectLockStatus::LockedToTx {
                locked_by_tx: LockDetails {
                    epoch: epoch_id,
                    tx_digest,
                },
            })
        } else {
            Ok(ObjectLockStatus::Initialized)
        }
    }

    /// Returns UserInputError::ObjectNotFound if no lock records found for this object.
    pub(crate) fn get_latest_live_version_for_object_id(
        &self,
        object_id: ObjectID,
    ) -> SomaResult<ObjectRef> {
        let mut iterator = self
            .perpetual_tables
            .live_owned_object_markers
            .reversed_safe_iter_with_bounds(
                None,
                Some((object_id, Version::MAX, ObjectDigest::MAX)),
            )?;
        Ok(iterator
            .next()
            .transpose()?
            .and_then(|value| {
                if value.0 .0 == object_id {
                    Some(value)
                } else {
                    None
                }
            })
            .ok_or_else(|| SomaError::ObjectNotFound {
                object_id,
                version: None,
            })?
            .0)
    }

    /// Checks multiple object locks exist.
    /// Returns UserInputError::ObjectNotFound if cannot find lock record for at least one of the objects.
    /// Returns UserInputError::ObjectVersionUnavailableForConsumption if at least one object lock is not initialized
    ///     at the given version.
    pub fn check_owned_objects_are_live(&self, objects: &[ObjectRef]) -> SomaResult {
        let locks = self
            .perpetual_tables
            .live_owned_object_markers
            .multi_get(objects)?;
        for (lock, obj_ref) in locks.into_iter().zip(objects) {
            if lock.is_none() {
                let latest_lock = self.get_latest_live_version_for_object_id(obj_ref.0)?;
                SomaError::ObjectVersionUnavailableForConsumption {
                    provided_obj_ref: *obj_ref,
                    current_version: latest_lock.1,
                }
            }
        }
        Ok(())
    }

    /// Initialize a lock to None (but exists) for a given list of ObjectRefs.
    /// Returns SomaErrorKind::ObjectLockAlreadyInitialized if the lock already exists and is locked to a transaction
    fn initialize_live_object_markers_impl(
        &self,
        write_batch: &mut DBBatch,
        objects: &[ObjectRef],
        is_force_reset: bool,
    ) -> SomaResult {
        AuthorityStore::initialize_live_object_markers(
            &self.perpetual_tables.live_owned_object_markers,
            write_batch,
            objects,
            is_force_reset,
        )
    }

    pub fn initialize_live_object_markers(
        live_object_marker_table: &DBMap<ObjectRef, Option<LockDetails>>,
        write_batch: &mut DBBatch,
        objects: &[ObjectRef],
        is_force_reset: bool,
    ) -> SomaResult {
        trace!(?objects, "initialize_locks");

        let live_object_markers = live_object_marker_table.multi_get(objects)?;

        if !is_force_reset {
            // If any live_object_markers exist and are not None, return errors for them
            // Note we don't check if there is a pre-existing lock. this is because initializing the live
            // object marker will not overwrite the lock and cause the validator to equivocate.
            let existing_live_object_markers: Vec<ObjectRef> = live_object_markers
                .iter()
                .zip(objects)
                .filter_map(|(lock_opt, objref)| {
                    lock_opt.clone().flatten().map(|_tx_digest| *objref)
                })
                .collect();
            if !existing_live_object_markers.is_empty() {
                info!(
                    ?existing_live_object_markers,
                    "Cannot initialize live_object_markers because some exist already"
                );
                return Err(SomaError::ObjectLockAlreadyInitialized {
                    refs: existing_live_object_markers,
                }
                .into());
            }
        }

        write_batch.insert_batch(
            live_object_marker_table,
            objects.iter().map(|obj_ref| (obj_ref, None)),
        )?;
        Ok(())
    }

    /// Removes locks for a given list of ObjectRefs.
    fn delete_live_object_markers(
        &self,
        write_batch: &mut DBBatch,
        objects: &[ObjectRef],
    ) -> SomaResult {
        trace!(?objects, "delete_locks");
        write_batch.delete_batch(
            &self.perpetual_tables.live_owned_object_markers,
            objects.iter(),
        )?;
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn reset_locks_for_test(
        &self,
        transactions: &[TransactionDigest],
        objects: &[ObjectRef],
        epoch_store: &AuthorityPerEpochStore,
    ) {
        for tx in transactions {
            epoch_store.delete_signed_transaction_for_test(tx);
            epoch_store.delete_object_locks_for_test(objects);
        }

        let mut batch = self.perpetual_tables.live_owned_object_markers.batch();
        batch
            .delete_batch(
                &self.perpetual_tables.live_owned_object_markers,
                objects.iter(),
            )
            .unwrap();
        batch.write().unwrap();

        let mut batch = self.perpetual_tables.live_owned_object_markers.batch();
        self.initialize_live_object_markers_impl(&mut batch, objects, false)
            .unwrap();
        batch.write().unwrap();
    }

    /// Return the object with version less then or eq to the provided seq number.
    /// This is used by indexer to find the correct version of dynamic field child object.
    /// We do not store the version of the child object, but because of lamport timestamp,
    /// we know the child must have version number less then or eq to the parent.
    pub fn find_object_lt_or_eq_version(
        &self,
        object_id: ObjectID,
        version: Version,
    ) -> SomaResult<Option<Object>> {
        self.perpetual_tables
            .find_object_lt_or_eq_version(object_id, version)
    }

    /// Returns the latest object reference we have for this object_id in the objects table.
    ///
    /// The method may also return the reference to a deleted object with a digest of
    /// ObjectDigest::deleted() or ObjectDigest::wrapped() and lamport version
    /// of a transaction that deleted the object.
    /// Note that a deleted object may re-appear if the deletion was the result of the object
    /// being wrapped in another object.
    ///
    /// If no entry for the object_id is found, return None.
    pub fn get_latest_object_ref_or_tombstone(
        &self,
        object_id: ObjectID,
    ) -> Result<Option<ObjectRef>, SomaError> {
        self.perpetual_tables
            .get_latest_object_ref_or_tombstone(object_id)
    }

    /// Returns the latest object reference if and only if the object is still live (i.e. it does
    /// not return tombstones)
    pub fn get_latest_object_ref_if_alive(
        &self,
        object_id: ObjectID,
    ) -> Result<Option<ObjectRef>, SomaError> {
        match self.get_latest_object_ref_or_tombstone(object_id)? {
            Some(objref) if objref.2.is_alive() => Ok(Some(objref)),
            _ => Ok(None),
        }
    }

    /// Returns the latest object we have for this object_id in the objects table.
    ///
    /// If no entry for the object_id is found, return None.
    pub fn get_latest_object_or_tombstone(
        &self,
        object_id: ObjectID,
    ) -> Result<Option<(ObjectKey, ObjectOrTombstone)>, SomaError> {
        let Some((object_key, store_object)) = self
            .perpetual_tables
            .get_latest_object_or_tombstone(object_id)?
        else {
            return Ok(None);
        };

        if let Some(object_ref) = self
            .perpetual_tables
            .tombstone_reference(&object_key, &store_object)?
        {
            return Ok(Some((object_key, ObjectOrTombstone::Tombstone(object_ref))));
        }

        let object = self
            .perpetual_tables
            .object(&object_key, store_object)?
            .expect("Non tombstone store object could not be converted to object");

        Ok(Some((object_key, ObjectOrTombstone::Object(object))))
    }

    pub fn insert_transaction_and_effects(
        &self,
        transaction: &VerifiedTransaction,
        transaction_effects: &TransactionEffects,
    ) -> Result<(), TypedStoreError> {
        let mut write_batch = self.perpetual_tables.transactions.batch();
        // effects must be inserted before the corresponding transaction entry
        // because they carry epoch information necessary for correct pruning via relocation filters
        write_batch
            .insert_batch(
                &self.perpetual_tables.effects,
                [(transaction_effects.digest(), transaction_effects)],
            )?
            .insert_batch(
                &self.perpetual_tables.transactions,
                [(transaction.digest(), transaction.serializable_ref())],
            )?;

        write_batch.write()?;
        Ok(())
    }

    pub fn multi_insert_transaction_and_effects<'a>(
        &self,
        transactions: impl Iterator<Item = &'a VerifiedExecutionData>,
    ) -> Result<(), TypedStoreError> {
        let mut write_batch = self.perpetual_tables.transactions.batch();
        for tx in transactions {
            write_batch
                .insert_batch(
                    &self.perpetual_tables.effects,
                    [(tx.effects.digest(), &tx.effects)],
                )?
                .insert_batch(
                    &self.perpetual_tables.transactions,
                    [(tx.transaction.digest(), tx.transaction.serializable_ref())],
                )?;
        }

        write_batch.write()?;
        Ok(())
    }

    pub fn multi_get_transaction_blocks(
        &self,
        tx_digests: &[TransactionDigest],
    ) -> Result<Vec<Option<VerifiedTransaction>>, TypedStoreError> {
        self.perpetual_tables
            .transactions
            .multi_get(tx_digests)
            .map(|v| v.into_iter().map(|v| v.map(|v| v.into())).collect())
    }

    pub fn get_transaction_block(
        &self,
        tx_digest: &TransactionDigest,
    ) -> Result<Option<VerifiedTransaction>, TypedStoreError> {
        self.perpetual_tables
            .transactions
            .get(tx_digest)
            .map(|v| v.map(|v| v.into()))
    }

    /// This function reads the DB directly to get the system state object.
    /// If reconfiguration is happening at the same time, there is no guarantee whether we would be getting
    /// the old or the new system state object.
    /// Hence this function should only be called during RPC reads where data race is not a major concern.
    /// In general we should avoid this as much as possible.
    /// If the intent is for testing, you can use AuthorityState:: get_sui_system_state_object_for_testing.
    pub fn get_system_state_object_unsafe(&self) -> SomaResult<SystemState> {
        get_system_state(self.perpetual_tables.as_ref())
    }

    pub async fn prune_objects_and_compact_for_testing(
        &self,
        checkpoint_store: &Arc<CheckpointStore>,
        rpc_index: Option<&RpcIndexStore>,
    ) {
        let pruning_config = AuthorityStorePruningConfig {
            num_epochs_to_retain: 0,
            ..Default::default()
        };
        let _ = AuthorityStorePruner::prune_objects_for_eligible_epochs(
            &self.perpetual_tables,
            checkpoint_store,
            rpc_index,
            None,
            pruning_config,
            EPOCH_DURATION_MS_FOR_TESTING,
        )
        .await;
        let _ = AuthorityStorePruner::compact(&self.perpetual_tables);
    }

    #[cfg(test)]
    pub async fn prune_objects_immediately_for_testing(
        &self,
        transaction_effects: Vec<TransactionEffects>,
    ) -> anyhow::Result<()> {
        let mut wb = self.perpetual_tables.objects.batch();

        let mut object_keys_to_prune = vec![];
        for effects in &transaction_effects {
            for (object_id, seq_number) in effects.modified_at_versions() {
                info!("Pruning object {:?} version {:?}", object_id, seq_number);
                object_keys_to_prune.push(ObjectKey(object_id, seq_number));
            }
        }

        wb.delete_batch(
            &self.perpetual_tables.objects,
            object_keys_to_prune.into_iter(),
        )?;
        wb.write()?;
        Ok(())
    }

    // Counts the number of versions exist in object store for `object_id`. This includes tombstone.
    #[cfg(msim)]
    pub fn count_object_versions(&self, object_id: ObjectID) -> usize {
        self.perpetual_tables
            .objects
            .safe_iter_with_bounds(
                Some(ObjectKey(object_id, Version::MIN)),
                Some(ObjectKey(object_id, Version::MAX)),
            )
            .collect::<Result<Vec<_>, _>>()
            .unwrap()
            .len()
    }
}

impl GlobalStateHashStore for AuthorityStore {
    fn get_root_state_hash_for_epoch(
        &self,
        epoch: EpochId,
    ) -> SomaResult<Option<(CheckpointSequenceNumber, GlobalStateHash)>> {
        self.perpetual_tables
            .root_state_hash_by_epoch
            .get(&epoch)
            .map_err(Into::into)
    }

    fn get_root_state_hash_for_highest_epoch(
        &self,
    ) -> SomaResult<Option<(EpochId, (CheckpointSequenceNumber, GlobalStateHash))>> {
        Ok(self
            .perpetual_tables
            .root_state_hash_by_epoch
            .reversed_safe_iter_with_bounds(None, None)?
            .next()
            .transpose()?)
    }

    fn insert_state_hash_for_epoch(
        &self,
        epoch: EpochId,
        last_checkpoint_of_epoch: &CheckpointSequenceNumber,
        acc: &GlobalStateHash,
    ) -> SomaResult {
        self.perpetual_tables
            .root_state_hash_by_epoch
            .insert(&epoch, &(*last_checkpoint_of_epoch, acc.clone()))?;
        self.root_state_notify_read
            .notify(&epoch, &(*last_checkpoint_of_epoch, acc.clone()));

        Ok(())
    }

    fn iter_live_object_set(&self) -> Box<dyn Iterator<Item = LiveObject> + '_> {
        Box::new(self.perpetual_tables.iter_live_object_set())
    }
}

impl ObjectStore for AuthorityStore {
    /// Read an object and return it, or Ok(None) if the object was not found.
    fn get_object(&self, object_id: &ObjectID) -> Option<Object> {
        self.perpetual_tables.as_ref().get_object(object_id)
    }

    fn get_object_by_key(&self, object_id: &ObjectID, version: Version) -> Option<Object> {
        self.perpetual_tables.get_object_by_key(object_id, version)
    }
}
