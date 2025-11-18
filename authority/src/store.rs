use std::{
    collections::{BTreeMap, HashMap},
    iter,
    sync::Arc,
};

use fastcrypto::hash::{HashFunction, Sha3_256};
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
    base::FullObjectID,
    committee::{Committee, EpochId},
    config::node_config::NodeConfig,
    digests::{ObjectDigest, TransactionDigest, TransactionEffectsDigest},
    effects::{TransactionEffects, TransactionEffectsAPI},
    envelope::Message,
    error::{SomaError, SomaResult},
    genesis::Genesis,
    mutex_table::{MutexGuard, MutexTable, RwLockGuard, RwLockTable},
    object::{self, LiveObject, Object, ObjectID, ObjectRef, Version},
    storage::{
        object_store::ObjectStore, FullObjectKey, MarkerValue, ObjectKey, ObjectOrTombstone,
    },
    system_state::{get_system_state, SystemState, SystemStateTrait},
    transaction::{VerifiedSignedTransaction, VerifiedTransaction},
    transaction_outputs::TransactionOutputs,
};

use crate::{
    epoch_store::AuthorityPerEpochStore,
    start_epoch::EpochStartConfiguration,
    store_tables::{get_store_object, AuthorityPerpetualTables, StoreObject},
};

pub struct AuthorityStore {
    /// Internal vector of locks to manage concurrent writes to the database
    mutex_table: MutexTable<ObjectDigest>,

    pub(crate) perpetual_tables: Arc<AuthorityPerpetualTables>,
    // pub(crate) root_state_notify_read: NotifyRead<EpochId, (CheckpointSequenceNumber, Accumulator)>,
    /// Guards reference count updates to `objects` table
    pub(crate) objects_lock_table: Arc<RwLockTable<ObjectDigest>>,
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

            let epoch_start_configuration =
                EpochStartConfiguration::new(genesis.system_object().into_epoch_start_state());
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
        Ok(this)
    }

    /// Returns true if there are no objects in the database
    pub fn database_is_empty(&self) -> SomaResult<bool> {
        self.perpetual_tables.database_is_empty()
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
            perpetual_tables,
            mutex_table: MutexTable::new(4096),
            objects_lock_table: Arc::new(RwLockTable::new(4096)),
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

    // NB: This must only be called at time of reconfiguration. We take the execution lock write
    // guard as an argument to ensure that this is the case.
    pub fn clear_object_per_epoch_marker_table(
        &self,
        _execution_guard: &ExecutionLockWriteGuard<'_>,
    ) -> SomaResult<()> {
        // We can safely delete all entries in the per epoch marker table since this is only called
        // at epoch boundaries (during reconfiguration). Therefore any entries that currently
        // exist can be removed. Because of this we can use the `schedule_delete_all` method.x
        Ok(self
            .perpetual_tables
            .object_per_epoch_marker_table
            .schedule_delete_all()?)
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

    pub fn multi_get_effects<'a>(
        &self,
        effects_digests: impl Iterator<Item = &'a TransactionEffectsDigest>,
    ) -> SomaResult<Vec<Option<TransactionEffects>>> {
        self.perpetual_tables
            .effects
            .multi_get(effects_digests)
            .map_err(|e| e.into())
    }

    pub fn get_executed_effects(
        &self,
        tx_digest: &TransactionDigest,
    ) -> SomaResult<Option<TransactionEffects>> {
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
    ) -> SomaResult<Vec<Option<TransactionEffectsDigest>>> {
        self.perpetual_tables
            .executed_effects
            .multi_get(digests)
            .map_err(|e| e.into())
    }

    /// Given a list of transaction digests, returns a list of the corresponding effects only if they have been
    /// executed. For transactions that have not been executed, None is returned.
    pub fn multi_get_executed_effects(
        &self,
        digests: &[TransactionDigest],
    ) -> SomaResult<Vec<Option<TransactionEffects>>> {
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

    pub fn multi_insert_transactions<'a>(
        &self,
        transactions: impl Iterator<Item = &'a VerifiedTransaction>,
    ) -> Result<(), TypedStoreError> {
        let mut write_batch = self.perpetual_tables.transactions.batch();
        for tx in transactions {
            write_batch.insert_batch(
                &self.perpetual_tables.transactions,
                [(tx.digest(), tx.serializable_ref())],
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

    /// Updates the state resulting from the execution of a certificate.
    #[instrument(level = "debug", skip_all)]
    pub async fn write_transaction_outputs(
        &self,
        epoch_id: EpochId,
        tx_outputs: &[Arc<TransactionOutputs>],
    ) -> SomaResult {
        let mut written = Vec::with_capacity(tx_outputs.len());
        for outputs in tx_outputs {
            written.extend(outputs.written.values().cloned());
        }

        let _locks = self.acquire_read_locks_for_objects(&written).await;

        // Create a single batch for all operations
        let mut write_batch = self.perpetual_tables.transactions.batch();

        for outputs in tx_outputs {
            self.write_one_transaction_outputs(&mut write_batch, epoch_id, outputs)?;
        }

        // Write all changes atomically
        write_batch.write()?;

        trace!(
            "committed transactions: {:?}",
            tx_outputs
                .iter()
                .map(|tx| tx.transaction.digest())
                .collect::<Vec<_>>()
        );

        Ok(())
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
        written,
        deleted,
        locks_to_delete,
        new_locks_to_init,
        ..  // Add this if there are other fields
    } = tx_outputs;

        let effects_digest = effects.digest();
        let transaction_digest = transaction.digest();

        // Effects must be inserted before the corresponding dependent entries
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

        // Add batched writes for markers
        write_batch.insert_batch(
            &self.perpetual_tables.object_per_epoch_marker_table,
            markers
                .iter()
                .map(|(key, marker_value)| ((epoch_id, *key), *marker_value)),
        )?;

        // Add batched writes for deleted objects
        write_batch.insert_batch(
            &self.perpetual_tables.objects,
            deleted.iter().map(|key| (key, StoreObject::Deleted)),
        )?;

        // Insert each output object into the stores
        let new_objects = written.iter().map(|(id, new_object)| {
            let version = new_object.version();
            debug!(?id, ?version, "writing object");
            let store_object = get_store_object(new_object.clone());
            (ObjectKey(*id, version), store_object)
        });

        write_batch.insert_batch(&self.perpetual_tables.objects, new_objects)?;

        // Initialize and delete locks
        self.initialize_object_transaction_locks_impl(write_batch, new_locks_to_init, false)?;
        self.delete_object_transaction_locks(write_batch, locks_to_delete)?;

        debug!(effects_digest = ?effects.digest(), "commit_certificate finished");

        Ok(())
    }

    /// Commits transactions only to the db. Called by checkpoint builder.
    pub(crate) fn commit_transactions(
        &self,
        transactions: &[(TransactionDigest, VerifiedTransaction)],
    ) -> SomaResult {
        info!(?transactions, "commit_transactions");

        let mut batch = self.perpetual_tables.transactions.batch();

        batch.insert_batch(
            &self.perpetual_tables.transactions,
            transactions
                .iter()
                .map(|(digest, transaction)| (*digest, transaction.serializable_ref())),
        )?;

        batch.write()?;
        Ok(())
    }

    /// A function that acquires all locks associated with the objects (in order to avoid deadlocks).
    async fn acquire_locks(&self, input_objects: &[ObjectRef]) -> Vec<MutexGuard> {
        self.mutex_table
            .acquire_locks(input_objects.iter().map(|(_, _, digest)| *digest))
            .await
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
            result.push(self.get_object(id)?);
        }
        Ok(result)
    }

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

            self.initialize_object_transaction_locks_impl(&mut write_batch, &[object_ref], false)?;
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

        let refs: Vec<_> = ref_and_objects.iter().map(|(oref, _)| *oref).collect();

        self.initialize_object_transaction_locks_impl(
            &mut batch, &refs, false, // is_force_reset
        )?;

        batch.write()?;

        Ok(())
    }

    pub fn bulk_insert_live_objects(
        perpetual_db: &AuthorityPerpetualTables,
        live_objects: impl Iterator<Item = LiveObject>,
        indirect_objects_threshold: usize,
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

                    Self::initialize_object_transaction_locks(
                        &perpetual_db.object_transaction_locks,
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

    /// Acquires read locks for affected indirect objects
    #[instrument(level = "trace", skip_all)]
    async fn acquire_read_locks_for_objects(&self, written: &[Object]) -> Vec<RwLockGuard> {
        // locking is required to avoid potential race conditions with the pruner
        // potential race:
        //   - transaction execution branches to reference count increment
        //   - pruner decrements ref count to 0
        //   - compaction job compresses existing merge values to an empty vector
        //   - tx executor commits ref count increment instead of the full value making object inaccessible
        // read locks are sufficient because ref count increments are safe,
        // concurrent transaction executions produce independent ref count increments and don't corrupt the state
        let digests = written
            .iter()
            .filter_map(|object| Some(object.digest()))
            .collect();
        self.objects_lock_table.acquire_read_locks(digests).await
    }

    pub async fn acquire_transaction_locks(
        &self,
        epoch_store: &AuthorityPerEpochStore,
        owned_input_objects: &[ObjectRef],
        transaction: VerifiedSignedTransaction,
    ) -> SomaResult {
        let tx_digest = *transaction.digest();
        let epoch = epoch_store.epoch();
        // Other writers may be attempting to acquire locks on the same objects, so a mutex is
        // required.
        // TODO: replace with optimistic db_transactions (i.e. set lock to tx if none)
        let _mutexes = self.acquire_locks(owned_input_objects);

        trace!(?owned_input_objects, "acquire_locks");
        let mut locks_to_write = Vec::new();

        let live_object_markers = self
            .perpetual_tables
            .object_transaction_locks
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

                return Err(SomaError::ObjectVersionUnavailableForConsumption {
                    provided_obj_ref: *obj_ref,
                    current_version: latest_lock.1,
                });
            };

            if let Some(LockDetails {
                epoch: previous_epoch,
                ..
            }) = &live_marker
            {
                // this must be from a prior epoch, because we no longer write LockDetails to
                // owned_object_transaction_locks
                assert!(
                    previous_epoch.clone() < epoch,
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
                    });
                }
            }

            locks_to_write.push((*obj_ref, tx_digest));
        }

        if !locks_to_write.is_empty() {
            trace!(?locks_to_write, "Writing locks");
            epoch_tables.write_transaction_locks(transaction, locks_to_write.into_iter())?;
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
            .object_transaction_locks
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
                locked_by_tx: tx_digest,
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
            .object_transaction_locks
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

    fn initialize_object_transaction_locks_impl(
        &self,
        write_batch: &mut DBBatch,
        objects: &[ObjectRef],
        is_force_reset: bool,
    ) -> SomaResult {
        AuthorityStore::initialize_object_transaction_locks(
            &self.perpetual_tables.object_transaction_locks,
            write_batch,
            objects,
            is_force_reset,
        )
    }

    pub fn initialize_object_transaction_locks(
        object_transaction_locks_table: &DBMap<ObjectRef, Option<LockDetails>>,
        write_batch: &mut DBBatch,
        objects: &[ObjectRef],
        is_force_reset: bool,
    ) -> SomaResult {
        trace!(?objects, "initialize_locks");

        let object_transaction_locks = object_transaction_locks_table.multi_get(objects)?;

        if !is_force_reset {
            // If any live_object_markers exist and are not None, return errors for them
            // Note we don't check if there is a pre-existing lock. this is because initializing the live
            // object marker will not overwrite the lock and cause the validator to equivocate.
            let existing_live_object_markers: Vec<ObjectRef> = object_transaction_locks
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
                });
            }
        }

        write_batch.insert_batch(
            object_transaction_locks_table,
            objects.iter().map(|obj_ref| (obj_ref, None)),
        )?;
        Ok(())
    }

    /// Removes locks for a given list of ObjectRefs.
    fn delete_object_transaction_locks(
        &self,
        write_batch: &mut DBBatch,
        objects: &[ObjectRef],
    ) -> SomaResult {
        trace!(?objects, "delete_locks");
        write_batch.delete_batch(
            &self.perpetual_tables.object_transaction_locks,
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

        let mut batch = self.perpetual_tables.object_transaction_locks.batch();
        batch
            .delete_batch(
                &self.perpetual_tables.object_transaction_locks,
                objects.iter(),
            )
            .unwrap();
        batch.write().unwrap();

        let mut batch = self.perpetual_tables.object_transaction_locks.batch();
        self.initialize_object_transaction_locks_impl(&mut batch, objects, false)
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

    /// This function reads the DB directly to get the system state object.
    /// If reconfiguration is happening at the same time, there is no guarantee whether we would be getting
    /// the old or the new system state object.
    /// Hence this function should only be called during RPC reads where data race is not a major concern.
    /// In general we should avoid this as much as possible.
    pub fn get_system_state_object(&self) -> SomaResult<SystemState> {
        get_system_state(self.perpetual_tables.as_ref())
    }

    pub fn get_marker_value(
        &self,
        object_key: FullObjectKey,
        epoch_id: EpochId,
    ) -> SomaResult<Option<MarkerValue>> {
        Ok(self
            .perpetual_tables
            .object_per_epoch_marker_table
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
            .object_per_epoch_marker_table
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
}

impl ObjectStore for AuthorityStore {
    /// Read an object and return it, or Ok(None) if the object was not found.
    fn get_object(
        &self,
        object_id: &ObjectID,
    ) -> Result<Option<Object>, types::storage::storage_error::Error> {
        self.perpetual_tables.as_ref().get_object(object_id)
    }

    fn get_object_by_key(
        &self,
        object_id: &ObjectID,
        version: Version,
    ) -> Result<Option<Object>, types::storage::storage_error::Error> {
        self.perpetual_tables.get_object_by_key(object_id, version)
    }
}
