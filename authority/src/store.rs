use std::{
    collections::{BTreeMap, HashMap},
    iter,
    sync::Arc,
};

use fastcrypto::hash::{HashFunction, Sha3_256};
use itertools::izip;
use parking_lot::RwLock;
use tokio::sync::{RwLockReadGuard, RwLockWriteGuard};
use tracing::{debug, error, info, instrument, trace};
use types::{
    accumulator::{Accumulator, AccumulatorStore, CommitIndex},
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
    tx_outputs::TransactionOutputs,
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
                .read()
                .get(&())
                .expect("Epoch start configuration must be set in non-empty DB")
                .clone()
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

            store.perpetual_tables.transactions.write().insert(
                *transaction.digest(),
                transaction.serializable_ref().clone(),
            );

            store
                .perpetual_tables
                .effects
                .write()
                .insert(genesis.effects().digest(), genesis.effects().clone());
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
        // exist can be removed. Because of this we can use the `schedule_delete_all` method.
        Ok(self
            .perpetual_tables
            .object_per_epoch_marker_table
            .write()
            .clear())
    }

    pub fn get_recovery_epoch_at_restart(&self) -> SomaResult<EpochId> {
        self.perpetual_tables.get_recovery_epoch_at_restart()
    }

    pub fn get_effects(
        &self,
        effects_digest: &TransactionEffectsDigest,
    ) -> SomaResult<Option<TransactionEffects>> {
        Ok(self
            .perpetual_tables
            .effects
            .read()
            .get(effects_digest)
            .cloned())
    }

    /// Returns true if we have an effects structure for this transaction digest
    pub fn effects_exists(&self, effects_digest: &TransactionEffectsDigest) -> SomaResult<bool> {
        Ok(self
            .perpetual_tables
            .effects
            .read()
            .contains_key(effects_digest))
    }

    pub fn multi_get_effects<'a>(
        &self,
        effects_digests: impl Iterator<Item = &'a TransactionEffectsDigest>,
    ) -> SomaResult<Vec<Option<TransactionEffects>>> {
        let read_guard = self.perpetual_tables.effects.read();
        Ok(effects_digests
            .map(|key| read_guard.get(key).cloned())
            .collect())
    }

    pub fn get_executed_effects(
        &self,
        tx_digest: &TransactionDigest,
    ) -> SomaResult<Option<TransactionEffects>> {
        let executed_effects_read = self.perpetual_tables.executed_effects.read();
        let effects_digest = executed_effects_read.get(tx_digest);
        match effects_digest {
            Some(digest) => Ok(self.perpetual_tables.effects.read().get(&digest).cloned()),
            None => Ok(None),
        }
    }

    /// Given a list of transaction digests, returns a list of the corresponding effects only if they have been
    /// executed. For transactions that have not been executed, None is returned.
    pub fn multi_get_executed_effects_digests(
        &self,
        digests: &[TransactionDigest],
    ) -> SomaResult<Vec<Option<TransactionEffectsDigest>>> {
        let read_guard = self.perpetual_tables.executed_effects.read();

        Ok(digests
            .iter()
            .map(|key| read_guard.get(key).cloned())
            .collect())
    }

    /// Given a list of transaction digests, returns a list of the corresponding effects only if they have been
    /// executed. For transactions that have not been executed, None is returned.
    pub fn multi_get_executed_effects(
        &self,
        digests: &[TransactionDigest],
    ) -> SomaResult<Vec<Option<TransactionEffects>>> {
        let read_guard = self.perpetual_tables.executed_effects.read();
        let executed_effects_digests: Vec<Option<TransactionEffectsDigest>> = digests
            .iter()
            .map(|key| read_guard.get(key).cloned())
            .collect();
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
            .read()
            .contains_key(digest))
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
        Ok(self
            .perpetual_tables
            .epoch_start_configuration
            .read()
            .get(&())
            .cloned())
    }

    pub fn insert_transaction_and_effects(
        &self,
        transaction: &VerifiedTransaction,
        transaction_effects: &TransactionEffects,
    ) -> Result<(), TypedStoreError> {
        self.perpetual_tables.transactions.write().insert(
            *transaction.digest(),
            transaction.serializable_ref().clone(),
        );

        self.perpetual_tables
            .effects
            .write()
            .insert(transaction_effects.digest(), transaction_effects.clone());
        Ok(())
    }

    pub fn multi_insert_transactions<'a>(
        &self,
        transactions: impl Iterator<Item = &'a VerifiedTransaction>,
    ) -> Result<(), TypedStoreError> {
        for tx in transactions {
            self.perpetual_tables
                .transactions
                .write()
                .insert(*tx.digest(), tx.serializable_ref().clone());
        }

        Ok(())
    }

    pub fn multi_get_transaction_blocks(
        &self,
        tx_digests: &[TransactionDigest],
    ) -> SomaResult<Vec<Option<VerifiedTransaction>>> {
        let read_guard = self.perpetual_tables.transactions.read();
        Ok(tx_digests
            .iter()
            .map(|key| read_guard.get(key).cloned().map(|v| v.into()))
            .collect())
    }

    pub fn get_transaction_block(
        &self,
        tx_digest: &TransactionDigest,
    ) -> Option<VerifiedTransaction> {
        self.perpetual_tables
            .transactions
            .read()
            .get(tx_digest)
            .cloned()
            .map(|v| v.into())
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

        for outputs in tx_outputs {
            let TransactionOutputs {
                transaction,
                effects,
                markers,
                written,
                deleted,
                locks_to_delete,
                new_locks_to_init,
            } = outputs as &TransactionOutputs;

            // Store the certificate indexed by transaction digest
            let transaction_digest = transaction.digest();
            self.perpetual_tables
                .transactions
                .write()
                .insert(*transaction_digest, transaction.serializable_ref().clone());

            let effects_digest = effects.digest();

            markers
                .iter()
                .map(|(key, marker_value)| ((epoch_id, *key), *marker_value))
                .for_each(|(key, value)| {
                    self.perpetual_tables
                        .object_per_epoch_marker_table
                        .write()
                        .insert(key, value);
                });

            deleted
                .iter()
                .map(|key| (key, StoreObject::Deleted))
                .for_each(|(key, store_object)| {
                    self.perpetual_tables
                        .objects
                        .write()
                        .insert(*key, store_object);
                });

            // Insert each output object into the stores
            let new_objects: Vec<(ObjectKey, &Object)> = written
                .iter()
                .map(|(id, new_object)| {
                    let version = new_object.version();
                    debug!(?id, ?version, "writing object");

                    (ObjectKey(*id, version), new_object)
                })
                .collect();

            for (key, object) in new_objects.iter() {
                self.perpetual_tables
                    .objects
                    .write()
                    .insert(*key, StoreObject::Value((*object).clone().into_inner()));
            }

            self.initialize_object_transaction_locks_impl(new_locks_to_init, false)?;

            // Note: deletes locks for received objects as well (but not for objects that were in
            // `Receiving` arguments which were not received)
            self.delete_object_transaction_locks(locks_to_delete)?;

            self.perpetual_tables
                .effects
                .write()
                .insert(effects_digest, effects.clone());

            self.perpetual_tables
                .executed_effects
                .write()
                .insert(*transaction_digest, effects_digest);

            debug!(effects_digest = ?effects.digest(), "commit_certificate finished");
        }

        trace!(
            "committed transactions: {:?}",
            tx_outputs
                .iter()
                .map(|tx| tx.transaction.digest())
                .collect::<Vec<_>>()
        );

        Ok(())
    }

    /// Commits transactions only to the db. Called by checkpoint builder. See
    /// ExecutionCache::commit_transactions for more info
    pub(crate) fn commit_transactions(
        &self,
        transactions: &[(TransactionDigest, VerifiedTransaction)],
    ) -> SomaResult {
        info!(?transactions, "commit_transactions");
        for (digest, transaction) in transactions {
            self.perpetual_tables
                .transactions
                .write()
                .insert(*digest, transaction.serializable_ref().clone());
        }
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
            .read()
            .contains_key(&ObjectKey(*object_id, version)))
    }

    pub fn multi_object_exists_by_key(&self, object_keys: &[ObjectKey]) -> SomaResult<Vec<bool>> {
        let objects_guard = self.perpetual_tables.objects.read();

        Ok(object_keys
            .iter()
            .map(|key| objects_guard.contains_key(key))
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
        // Get read lock on objects
        let objects = self.perpetual_tables.objects.read();

        // Find the entry with version less than or equal to prior_version
        let target_key = ObjectKey(*object_id, prior_version);

        // Use range to get all entries for this object_id up to and including prior_version
        let matching_entry = objects
            .range(..=target_key) // Get all entries up to and including our target
            .rev() // Reverse to get the highest version first
            .find(|(key, _)| key.0 == *object_id) // Find first entry matching our object_id
            .map(|(key, value)| (key.clone(), value.clone()));

        // If we found a matching entry, convert it to ObjectRef
        if let Some((object_key, value)) = matching_entry {
            Ok(Some(
                self.perpetual_tables.object_reference(&object_key, value)?,
            ))
        } else {
            Ok(None)
        }
    }

    pub fn multi_get_objects_by_key(
        &self,
        object_keys: &[ObjectKey],
    ) -> Result<Vec<Option<Object>>, SomaError> {
        // Get single read lock to avoid multiple lock acquisitions
        let objects_guard = self.perpetual_tables.objects.read();

        // Pre-allocate the result vector to avoid reallocations
        let mut ret = Vec::with_capacity(object_keys.len());

        // Process each key and transform the result
        for key in object_keys {
            let wrapper = objects_guard.get(key).cloned();
            let obj = wrapper
                .map(|object| self.perpetual_tables.object(key, object))
                .transpose()?
                .flatten();
            ret.push(obj);
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
        // Insert object
        let store_object = get_store_object(object.clone());
        self.perpetual_tables
            .objects
            .write()
            .insert(ObjectKey::from(object_ref), store_object);

        // Update the index
        if object.get_single_owner().is_some() {
            // Only initialize lock for address owned objects.
            self.initialize_object_transaction_locks_impl(&[object_ref], false)?;
        }

        Ok(())
    }

    /// This function should only be used for initializing genesis and should remain private.
    #[instrument(level = "debug", skip_all)]
    pub(crate) fn bulk_insert_genesis_objects(&self, objects: &[Object]) -> SomaResult<()> {
        let ref_and_objects: Vec<_> = objects
            .iter()
            .map(|o| (o.compute_object_reference(), o))
            .collect();

        for (oref, o) in &ref_and_objects {
            self.perpetual_tables.objects.write().insert(
                ObjectKey::from(oref),
                StoreObject::Value((**o).clone().into_inner()),
            );
        }

        let refs: Vec<_> = ref_and_objects.iter().map(|(oref, _)| *oref).collect();

        self.initialize_object_transaction_locks_impl(
            &(refs),
            false, // is_force_reset
        )?;

        Ok(())
    }

    pub fn bulk_insert_live_objects(
        perpetual_db: &AuthorityPerpetualTables,
        live_objects: impl Iterator<Item = LiveObject>,
        indirect_objects_threshold: usize,
        expected_sha3_digest: &[u8; 32],
    ) -> SomaResult<()> {
        let mut hasher = Sha3_256::default();

        for object in live_objects {
            hasher.update(object.object_reference().2.inner());
            match object {
                LiveObject::Normal(object) => {
                    perpetual_db.objects.write().insert(
                        ObjectKey::from(object.compute_object_reference()),
                        StoreObject::Value(object.clone().into_inner()),
                    );

                    Self::initialize_object_transaction_locks(
                        &perpetual_db.object_transaction_locks,
                        &[object.compute_object_reference()],
                        false, // is_force_reset
                    )?;
                }
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
        let _mutexes = self.acquire_locks(owned_input_objects).await;

        trace!(?owned_input_objects, "acquire_locks");
        let mut locks_to_write = Vec::new();

        let object_transaction_locks = {
            let locks = self.perpetual_tables.object_transaction_locks.read();
            owned_input_objects
                .iter()
                .map(|key| locks.get(key).cloned())
                .collect::<Vec<_>>()
        };

        let epoch_tables = epoch_store.tables()?;

        let locks = epoch_tables.multi_get_locked_transactions(owned_input_objects)?;

        assert_eq!(locks.len(), object_transaction_locks.len());

        for (live_marker, lock, obj_ref) in izip!(
            object_transaction_locks.into_iter(),
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
            .read()
            .get(&obj_ref)
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
        let locks_guard = self.perpetual_tables.object_transaction_locks.read();

        let max_key = (object_id, Version::MAX, ObjectDigest::MAX);
        let mut iterator = locks_guard
            .range(..=max_key) // Get all entries up to and including max key
            .rev() // Reverse to get highest version first
            .filter(|(key, _)| key.0 == object_id); // Only get entries for this object_id
        Ok(*iterator
            .next()
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

    /// This function is called at the end of epoch for each transaction that's
    /// executed locally on the validator but didn't make to the last commit.
    /// The effects of the execution is reverted here.
    /// The following things are reverted:
    /// 1. All new object states are deleted.
    /// 2. owner_index table change is reverted.
    ///
    /// NOTE: transaction and effects are intentionally not deleted. It's
    /// possible that if this node is behind, the network will execute the
    /// transaction in a later epoch. In that case, we need to keep it saved
    /// so that when we receive the commit that includes it from state
    /// sync, we are able to execute the commit.
    /// TODO: implement GC for transactions that are no longer needed.
    pub fn revert_state_update(&self, tx_digest: &TransactionDigest) -> SomaResult {
        let Some(effects) = self.get_executed_effects(tx_digest)? else {
            info!("Not reverting {:?} as it was not executed", tx_digest);
            return Ok(());
        };

        info!(?tx_digest, ?effects, "reverting transaction");

        self.perpetual_tables
            .executed_effects
            .write()
            .remove(tx_digest);

        // Remove tombstones
        for (id, version) in effects.all_tombstones() {
            let key = ObjectKey(id, version);
            self.perpetual_tables.objects.write().remove(&key);
        }

        // Remove changed objects
        let all_new_object_keys: Vec<ObjectKey> = effects
            .all_changed_objects()
            .into_iter()
            .map(|((id, version, _), _, _)| ObjectKey(id, version))
            .collect();

        // Get write lock once and perform all removals
        let mut objects = self.perpetual_tables.objects.write();
        for key in &all_new_object_keys {
            objects.remove(key);
        }

        let modified_object_keys = effects
            .modified_at_versions()
            .into_iter()
            .map(|(id, version)| ObjectKey(id, version));

        let old_locks: Vec<_> = modified_object_keys
            .map(|key| (self.perpetual_tables.objects.read().get(&key).cloned(), key))
            .filter_map(|(obj_opt, key)| {
                let obj = self
                    .perpetual_tables
                    .object(
                        &key,
                        obj_opt
                            .unwrap_or_else(|| panic!("Older object version not found: {:?}", key)),
                    )
                    .expect("Matching indirect object not found")?;

                let obj_ref = obj.compute_object_reference();
                Some(obj_ref)
            })
            .collect();

        let new_locks = all_new_object_keys
            .iter()
            .map(|key| (self.perpetual_tables.objects.read().get(&key).cloned(), key))
            .filter_map(|(obj_opt, key)| {
                let obj = self
                    .perpetual_tables
                    .object(
                        &key,
                        obj_opt
                            .unwrap_or_else(|| panic!("Older object version not found: {:?}", key)),
                    )
                    .expect("Matching indirect object not found")?;

                let obj_ref = obj.compute_object_reference();
                Some(obj_ref)
            });

        // Re-create old locks.
        self.initialize_object_transaction_locks_impl(&old_locks, true)?;

        // Delete new locks
        for lock in new_locks {
            self.perpetual_tables
                .object_transaction_locks
                .write()
                .remove(&lock);
        }

        Ok(())
    }

    /// Initialize a lock to None (but exists) for a given list of ObjectRefs.
    /// Returns SuiError::ObjectLockAlreadyInitialized if the lock already exists and is locked to a transaction
    fn initialize_object_transaction_locks_impl(
        &self,
        objects: &[ObjectRef],
        is_force_reset: bool,
    ) -> SomaResult {
        AuthorityStore::initialize_object_transaction_locks(
            &self.perpetual_tables.object_transaction_locks,
            objects,
            is_force_reset,
        )
    }

    pub fn initialize_object_transaction_locks(
        object_transaction_locks_table: &RwLock<BTreeMap<ObjectRef, Option<LockDetails>>>,
        objects: &[ObjectRef],
        is_force_reset: bool,
    ) -> SomaResult {
        trace!(?objects, "initialize_locks");

        // First check for existing locks if not force resetting
        if !is_force_reset {
            let locks_guard = object_transaction_locks_table.read();

            // Check for existing locks
            let existing_object_transaction_locks: Vec<ObjectRef> = objects
                .iter()
                .filter(|objref| {
                    locks_guard
                        .get(objref)
                        .and_then(|lock_opt| lock_opt.clone())
                        .is_some()
                })
                .copied()
                .collect();

            if !existing_object_transaction_locks.is_empty() {
                info!(
                    ?existing_object_transaction_locks,
                    "Cannot initialize object_transaction_locks because some exist already"
                );
                return Err(SomaError::ObjectLockAlreadyInitialized {
                    refs: existing_object_transaction_locks,
                });
            }
        }

        // Initialize all locks
        let mut locks_guard = object_transaction_locks_table.write();
        for obj_ref in objects {
            locks_guard.insert(*obj_ref, None);
        }

        Ok(())
    }

    /// Removes locks for a given list of ObjectRefs.
    fn delete_object_transaction_locks(&self, objects: &[ObjectRef]) -> SomaResult {
        trace!(?objects, "delete_locks");
        for object in objects {
            self.perpetual_tables
                .object_transaction_locks
                .write()
                .remove(object);
        }

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

        // let mut batch = self.perpetual_tables.live_owned_object_markers.batch();
        // batch
        //     .delete_batch(
        //         &self.perpetual_tables.live_owned_object_markers,
        //         objects.iter(),
        //     )
        //     .unwrap();
        // batch.write().unwrap();

        for object in objects {
            self.perpetual_tables
                .object_transaction_locks
                .write()
                .remove(object);
        }

        self.initialize_object_transaction_locks_impl(objects, false)
            .unwrap();
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
            .read()
            .get(&(epoch_id, object_key))
            .cloned())
    }

    pub fn get_latest_marker(
        &self,
        object_id: FullObjectID,
        epoch_id: EpochId,
    ) -> SomaResult<Option<(Version, MarkerValue)>> {
        let min_key = (epoch_id, FullObjectKey::min_for_id(&object_id));
        let max_key = (epoch_id, FullObjectKey::max_for_id(&object_id));

        // Acquire read lock on the BTreeMap
        let marker_map = self.perpetual_tables.object_per_epoch_marker_table.read();

        // Find the entry with the highest key that's less than or equal to max_key
        // This is equivalent to the skip_prior_to(&max_key) behavior
        let marker_entry = marker_map
            .range((
                std::ops::Bound::Included(min_key),
                std::ops::Bound::Included(max_key),
            ))
            .next_back();

        match marker_entry {
            Some(((epoch, key), marker)) => {
                // Verify the bounds
                assert_eq!(*epoch, epoch_id);
                assert_eq!(key.id(), object_id);
                Ok(Some((key.version(), marker.clone())))
            }
            None => Ok(None),
        }
    }

    pub fn have_deleted_owned_object_at_version_or_after(
        &self,
        object_id: &ObjectID,
        version: Version,
        epoch_id: EpochId,
    ) -> Result<bool, SomaError> {
        let object_key = ObjectKey::max_for_id(object_id);
        let marker_key = (epoch_id, FullObjectKey::Fastpath(object_key));

        // Acquire read lock on the BTreeMap
        let marker_map = self.perpetual_tables.object_per_epoch_marker_table.read();

        // Get the first entry equal to or greater than marker_key
        let marker_entry = marker_map
            .range((
                std::ops::Bound::Included(marker_key),
                std::ops::Bound::Unbounded,
            ))
            .next();

        match marker_entry {
            Some(((epoch, key), marker)) => {
                // For FullObjectKey::Fastpath, we need to extract the inner ObjectKey
                let object_id_matches = match key {
                    FullObjectKey::Fastpath(obj_key) => obj_key.0 == *object_id,
                    FullObjectKey::Consensus(cons_key) => false, // Not handling consensus case
                };

                let version_matches = match key {
                    FullObjectKey::Fastpath(obj_key) => obj_key.1 >= version,
                    FullObjectKey::Consensus(_) => false,
                };

                // Check all conditions
                let object_data_ok = object_id_matches && version_matches;
                let epoch_data_ok = *epoch == epoch_id;
                let mark_data_ok = *marker == MarkerValue::OwnedDeleted;

                Ok(object_data_ok && epoch_data_ok && mark_data_ok)
            }
            None => Ok(false),
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

impl AccumulatorStore for AuthorityStore {
    fn get_root_state_accumulator_for_epoch(
        &self,
        epoch: EpochId,
    ) -> SomaResult<Option<(CommitIndex, Accumulator)>> {
        Ok(self
            .perpetual_tables
            .root_state_hash_by_epoch
            .read()
            .get(&epoch)
            .cloned())
    }

    fn get_root_state_accumulator_for_highest_epoch(
        &self,
    ) -> SomaResult<Option<(EpochId, (CommitIndex, Accumulator))>> {
        Ok(self
            .perpetual_tables
            .root_state_hash_by_epoch
            .read()
            .iter()
            .next_back()
            .map(|(seq, acc)| (*seq, acc.clone())))
    }

    fn insert_state_accumulator_for_epoch(
        &self,
        epoch: EpochId,
        commit: &CommitIndex,
        acc: &Accumulator,
    ) -> SomaResult {
        self.perpetual_tables
            .root_state_hash_by_epoch
            .write()
            .insert(epoch, (*commit, acc.clone()));

        Ok(())
    }

    fn iter_live_object_set(&self) -> Box<dyn Iterator<Item = LiveObject> + '_> {
        Box::new(self.perpetual_tables.iter_live_object_set())
    }

    fn iter_cached_live_object_set_for_testing(&self) -> Box<dyn Iterator<Item = LiveObject> + '_> {
        self.iter_live_object_set()
    }
}

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[non_exhaustive]
#[derive(Error, Debug, Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Ord, PartialOrd)]
pub enum TypedStoreError {
    #[error("rocksdb error: {0}")]
    RocksDBError(String),
    #[error("(de)serialization error: {0}")]
    SerializationError(String),
    #[error("the column family {0} was not registered with the database")]
    UnregisteredColumn(String),
    #[error("a batch operation can't operate across databases")]
    CrossDBBatch,
    #[error("Transaction should be retried")]
    RetryableTransactionError,
}

impl From<TypedStoreError> for SomaError {
    fn from(e: TypedStoreError) -> Self {
        Self::Storage(e.to_string())
    }
}

impl From<TypedStoreError> for types::storage::storage_error::Error {
    fn from(error: TypedStoreError) -> Self {
        types::storage::storage_error::Error::custom(error)
    }
}
