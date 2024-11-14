//! This is a cache for the transaction execution which delays writes to the database until
//! transaction results are certified (i.e. they appear in a certified checkpoint, or an effects cert
//! is observed by a fullnode). The cache also stores committed data in memory in order to serve
//! future reads without hitting the database.
//!
//! For storing uncommitted transaction outputs, we cannot evict the data at all until it is written
//! to disk. Committed data not only can be evicted, but it is also unbounded (imagine a stream of
//! transactions that keep splitting a coin into smaller coins).
//!
//! To achieve both of these goals, we split the cache data into two pieces, a dirty set and a cached
//! set. The dirty set has no automatic evictions, data is only removed after being committed. The
//! cached set is in a bounded-sized cache with automatic evictions. In order to support negative
//! cache hits, we treat the two halves of the cache as FIFO queue. Newly written (dirty) versions are
//! inserted to one end of the dirty queue. As versions are committed to disk, they are
//! removed from the other end of the dirty queue and inserted into the cache queue. The cache queue
//! is truncated if it exceeds its maximum size, by removing all but the N newest entries.

use crate::{
    epoch_store::AuthorityPerEpochStore,
    store::{AuthorityStore, ExecutionLockWriteGuard, LockDetails, LockResult, ObjectLockStatus},
};
use core::hash::Hash;
use dashmap::{mapref::entry::Entry as DashMapEntry, DashMap};
use futures::{future::BoxFuture, FutureExt};
use moka::sync::Cache as MokaCache;
use parking_lot::Mutex;
use std::sync::Arc;
use tap::TapOptional;
use tracing::{debug, info, instrument, trace, warn};
use types::{
    accumulator::{Accumulator, AccumulatorStore, CommitIndex},
    committee::EpochId,
    digests::{ObjectDigest, TransactionDigest, TransactionEffectsDigest},
    effects::TransactionEffects,
    envelope::Message,
    error::{SomaError, SomaResult},
    object::{LiveObject, Object, ObjectID, ObjectRef, Version},
    storage::{object_store::ObjectStore, ObjectKey, ObjectOrTombstone},
    system_state::{get_system_state, SystemState},
    transaction::VerifiedTransaction,
    tx_outputs::TransactionOutputs,
};
use utils::notify_read::NotifyRead;

use super::{
    cache_types::CachedVersionMap, object_locks::ObjectLocks, ExecutionCacheCommit,
    ExecutionCacheWrite, ObjectCacheRead, TransactionCacheRead,
};

enum CacheResult<T> {
    /// Entry is in the cache
    Hit(T),
    /// Entry is not in the cache and is known to not exist
    NegativeHit,
    /// Entry is not in the cache and may or may not exist in the store
    Miss,
}

#[derive(Clone, PartialEq, Eq)]
enum ObjectEntry {
    Object(Object),
    Deleted,
}

impl std::fmt::Debug for ObjectEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ObjectEntry::Object(o) => {
                write!(f, "ObjectEntry::Object({:?})", o.compute_object_reference())
            }
            ObjectEntry::Deleted => write!(f, "ObjectEntry::Deleted"),
        }
    }
}

impl From<Object> for ObjectEntry {
    fn from(object: Object) -> Self {
        ObjectEntry::Object(object)
    }
}

impl From<ObjectOrTombstone> for ObjectEntry {
    fn from(object: ObjectOrTombstone) -> Self {
        match object {
            ObjectOrTombstone::Object(o) => o.into(),
            ObjectOrTombstone::Tombstone(obj_ref) => {
                if obj_ref.2.is_deleted() {
                    ObjectEntry::Deleted
                } else {
                    panic!("tombstone digest must either be deleted or wrapped");
                }
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum LatestObjectCacheEntry {
    Object(Version, ObjectEntry),
    NonExistent,
}

impl LatestObjectCacheEntry {
    fn is_newer_than(&self, other: &LatestObjectCacheEntry) -> bool {
        match (self, other) {
            (LatestObjectCacheEntry::Object(v1, _), LatestObjectCacheEntry::Object(v2, _)) => {
                v1 > v2
            }
            (LatestObjectCacheEntry::Object(_, _), LatestObjectCacheEntry::NonExistent) => true,
            _ => false,
        }
    }
}

/// UncommitedData stores execution outputs that are not yet written to the db. Entries in this
/// struct can only be purged after they are committed.
struct UncommittedData {
    /// The object dirty set. All writes go into this table first. After we flush the data to the
    /// db, the data is removed from this table and inserted into the object_cache.
    ///
    /// Further, we only remove objects in FIFO order, which ensures that the cached
    /// sequence of objects has no gaps. In other words, if we have versions 4, 8, 13 of
    /// an object, we can deduce that version 9 does not exist. This also makes child object
    /// reads efficient. `object_cache` cannot contain a more recent version of an object than
    /// `objects`, and neither can have any gaps. Therefore if there is any object <= the version
    /// bound for a child read in objects, it is the correct object to return.
    objects: DashMap<ObjectID, CachedVersionMap<ObjectEntry>>,

    transaction_effects: DashMap<TransactionEffectsDigest, TransactionEffects>,

    executed_effects_digests: DashMap<TransactionDigest, TransactionEffectsDigest>,
    // Transaction outputs that have not yet been written to the DB. Items are removed from this
    // table as they are flushed to the db.
    pending_transaction_writes: DashMap<TransactionDigest, Arc<TransactionOutputs>>,
}

impl UncommittedData {
    fn new() -> Self {
        Self {
            objects: DashMap::new(),
            transaction_effects: DashMap::new(),
            executed_effects_digests: DashMap::new(),
            pending_transaction_writes: DashMap::new(),
        }
    }

    fn clear(&self) {
        self.objects.clear();
        self.transaction_effects.clear();
        self.executed_effects_digests.clear();
        self.pending_transaction_writes.clear();
    }

    fn is_empty(&self) -> bool {
        self.objects.is_empty()
            && self.transaction_effects.is_empty()
            && self.executed_effects_digests.is_empty()
            && self.pending_transaction_writes.is_empty()
    }
}

// TODO: set this via the config
static MAX_CACHE_SIZE: u64 = 10000;

/// CachedData stores data that has been committed to the db, but is likely to be read soon.
struct CachedCommittedData {
    object_cache: MokaCache<ObjectID, Arc<Mutex<CachedVersionMap<ObjectEntry>>>>,

    // We separately cache the latest version of each object. Although this seems
    // redundant, it is the only way to support populating the cache after a read.
    // We cannot simply insert objects that we read off the disk into `object_cache`,
    // since that may violate the no-missing-versions property.
    // `object_by_id_cache` is also written to on writes so that it is always coherent.
    object_by_id_cache: MokaCache<ObjectID, Arc<Mutex<LatestObjectCacheEntry>>>,

    transactions: MokaCache<TransactionDigest, Arc<VerifiedTransaction>>,

    transaction_effects: MokaCache<TransactionEffectsDigest, Arc<TransactionEffects>>,

    executed_effects_digests: MokaCache<TransactionDigest, TransactionEffectsDigest>,
}

impl CachedCommittedData {
    fn new() -> Self {
        let object_cache = MokaCache::builder()
            .max_capacity(MAX_CACHE_SIZE)
            .max_capacity(MAX_CACHE_SIZE)
            .build();
        let object_by_id_cache = MokaCache::builder()
            .max_capacity(MAX_CACHE_SIZE)
            .max_capacity(MAX_CACHE_SIZE)
            .build();
        let transactions = MokaCache::builder()
            .max_capacity(MAX_CACHE_SIZE)
            .max_capacity(MAX_CACHE_SIZE)
            .build();
        let transaction_effects = MokaCache::builder()
            .max_capacity(MAX_CACHE_SIZE)
            .max_capacity(MAX_CACHE_SIZE)
            .build();

        let executed_effects_digests = MokaCache::builder()
            .max_capacity(MAX_CACHE_SIZE)
            .max_capacity(MAX_CACHE_SIZE)
            .build();

        Self {
            object_cache,
            object_by_id_cache,
            transactions,
            transaction_effects,
            executed_effects_digests,
        }
    }

    fn clear_and_assert_empty(&self) {
        self.object_cache.invalidate_all();
        self.object_by_id_cache.invalidate_all();
        self.transactions.invalidate_all();
        self.transaction_effects.invalidate_all();
        self.executed_effects_digests.invalidate_all();
        assert_empty(&self.object_cache);
        assert_empty(&self.object_by_id_cache);
        assert_empty(&self.transactions);
        assert_empty(&self.transaction_effects);
        assert_empty(&self.executed_effects_digests);
    }
}

fn assert_empty<K, V>(cache: &MokaCache<K, V>)
where
    K: std::hash::Hash + std::cmp::Eq + std::cmp::PartialEq + Send + Sync + 'static,
    V: std::clone::Clone + std::marker::Send + std::marker::Sync + 'static,
{
    if cache.iter().next().is_some() {
        panic!("cache should be empty");
    }
}

macro_rules! check_cache_entry_by_version {
    ($self: ident, $cache: expr, $version: expr) => {
        if let Some(cache) = $cache {
            if let Some(entry) = cache.get(&$version) {
                return CacheResult::Hit(entry.clone());
            }

            if let Some(least_version) = cache.get_least() {
                if least_version.0 < $version {
                    // If the version is greater than the least version in the cache, then we know
                    // that the object does not exist anywhere
                    return CacheResult::NegativeHit;
                }
            }
        }
    };
}

macro_rules! check_cache_entry_by_latest {
    ($self: ident,  $cache: expr) => {
        if let Some(cache) = $cache {
            if let Some((version, entry)) = cache.get_highest() {
                return CacheResult::Hit((*version, entry.clone()));
            } else {
                panic!("empty CachedVersionMap should have been removed");
            }
        }
    };
}

pub struct WritebackCache {
    dirty: UncommittedData,
    cached: CachedCommittedData,
    object_locks: ObjectLocks,
    executed_effects_digests_notify_read: NotifyRead<TransactionDigest, TransactionEffectsDigest>,
    store: Arc<AuthorityStore>,
}

impl WritebackCache {
    pub fn new(store: Arc<AuthorityStore>) -> Self {
        Self {
            dirty: UncommittedData::new(),
            cached: CachedCommittedData::new(),
            object_locks: ObjectLocks::new(),
            executed_effects_digests_notify_read: NotifyRead::new(),
            store,
        }
    }

    async fn write_object_entry(
        &self,
        object_id: &ObjectID,
        version: Version,
        object: ObjectEntry,
    ) {
        debug!(?object_id, ?version, ?object, "inserting object entry");
        self.dirty
            .objects
            .entry(*object_id)
            .or_default()
            .insert(version, object.clone());
        self.cached.object_by_id_cache.insert(
            *object_id,
            Arc::new(Mutex::new(LatestObjectCacheEntry::Object(version, object))),
        );
    }

    // lock both the dirty and committed sides of the cache, and then pass the entries to
    // the callback. Written with the `with` pattern because any other way of doing this
    // creates lifetime hell.
    fn with_locked_cache_entries<K, V, R>(
        dirty_map: &DashMap<K, CachedVersionMap<V>>,
        cached_map: &MokaCache<K, Arc<Mutex<CachedVersionMap<V>>>>,
        key: &K,
        cb: impl FnOnce(Option<&CachedVersionMap<V>>, Option<&CachedVersionMap<V>>) -> R,
    ) -> R
    where
        K: Copy + Eq + Hash + Send + Sync + 'static,
        V: Send + Sync + 'static,
    {
        let dirty_entry = dirty_map.entry(*key);
        let dirty_entry = match &dirty_entry {
            DashMapEntry::Occupied(occupied) => Some(occupied.get()),
            DashMapEntry::Vacant(_) => None,
        };

        let cached_entry = cached_map.get(key);
        let cached_lock = cached_entry.as_ref().map(|entry| entry.lock());
        let cached_entry = cached_lock.as_deref();

        cb(dirty_entry, cached_entry)
    }

    // Attempt to get an object from the cache. The DB is not consulted.
    // Can return Hit, Miss, or NegativeHit (if the object is known to not exist).
    fn get_object_entry_by_key_cache_only(
        &self,
        object_id: &ObjectID,
        version: Version,
    ) -> CacheResult<ObjectEntry> {
        Self::with_locked_cache_entries(
            &self.dirty.objects,
            &self.cached.object_cache,
            object_id,
            |dirty_entry, cached_entry| {
                check_cache_entry_by_version!(self, dirty_entry, version);
                check_cache_entry_by_version!(self, cached_entry, version);
                CacheResult::Miss
            },
        )
    }

    fn get_object_by_key_cache_only(
        &self,
        object_id: &ObjectID,
        version: Version,
    ) -> CacheResult<Object> {
        match self.get_object_entry_by_key_cache_only(object_id, version) {
            CacheResult::Hit(entry) => match entry {
                ObjectEntry::Object(object) => CacheResult::Hit(object),
                ObjectEntry::Deleted => CacheResult::NegativeHit,
            },
            CacheResult::Miss => CacheResult::Miss,
            CacheResult::NegativeHit => CacheResult::NegativeHit,
        }
    }

    fn get_object_entry_by_id_cache_only(
        &self,
        object_id: &ObjectID,
    ) -> CacheResult<(Version, ObjectEntry)> {
        let entry = self.cached.object_by_id_cache.get(object_id);

        if cfg!(debug_assertions) {
            if let Some(entry) = &entry {
                // check that cache is coherent
                let highest: Option<ObjectEntry> = self
                    .dirty
                    .objects
                    .get(object_id)
                    .and_then(|entry| entry.get_highest().map(|(_, o)| o.clone()))
                    .or_else(|| {
                        let obj: Option<ObjectEntry> = self
                            .store
                            .get_latest_object_or_tombstone(*object_id)
                            .unwrap()
                            .map(|(_, o)| o.into());
                        obj
                    });

                let cache_entry = match &*entry.lock() {
                    LatestObjectCacheEntry::Object(_, entry) => Some(entry.clone()),
                    LatestObjectCacheEntry::NonExistent => None,
                };

                if highest != cache_entry {
                    tracing::error!(
                        ?highest,
                        ?cache_entry,
                        "object_by_id cache is incoherent for {:?}",
                        object_id
                    );
                    panic!("object_by_id cache is incoherent for {:?}", object_id);
                }
            }
        }

        if let Some(entry) = entry {
            let entry = entry.lock();
            match &*entry {
                LatestObjectCacheEntry::Object(latest_version, latest_object) => {
                    return CacheResult::Hit((*latest_version, latest_object.clone()));
                }
                LatestObjectCacheEntry::NonExistent => {
                    return CacheResult::NegativeHit;
                }
            }
        }

        Self::with_locked_cache_entries(
            &self.dirty.objects,
            &self.cached.object_cache,
            object_id,
            |dirty_entry, cached_entry| {
                check_cache_entry_by_latest!(self, dirty_entry);
                check_cache_entry_by_latest!(self, cached_entry);
                CacheResult::Miss
            },
        )
    }

    fn get_object_by_id_cache_only(&self, object_id: &ObjectID) -> CacheResult<(Version, Object)> {
        match self.get_object_entry_by_id_cache_only(object_id) {
            CacheResult::Hit((version, entry)) => match entry {
                ObjectEntry::Object(object) => CacheResult::Hit((version, object)),
                ObjectEntry::Deleted => CacheResult::NegativeHit,
            },
            CacheResult::NegativeHit => CacheResult::NegativeHit,
            CacheResult::Miss => CacheResult::Miss,
        }
    }

    fn get_object_impl(&self, id: &ObjectID) -> SomaResult<Option<Object>> {
        match self.get_object_by_id_cache_only(id) {
            CacheResult::Hit((_, object)) => Ok(Some(object)),
            CacheResult::NegativeHit => Ok(None),
            CacheResult::Miss => {
                let obj = self.store.get_object(id)?;
                if let Some(obj) = &obj {
                    self.cache_latest_object_by_id(
                        id,
                        LatestObjectCacheEntry::Object(obj.version(), obj.clone().into()),
                    );
                } else {
                    self.cache_object_not_found(id);
                }
                Ok(obj)
            }
        }
    }

    #[instrument(level = "debug", skip_all)]
    async fn write_transaction_outputs(
        &self,
        epoch_id: EpochId,
        tx_outputs: Arc<TransactionOutputs>,
    ) -> SomaResult {
        trace!(digest = ?tx_outputs.transaction.digest(), "writing transaction outputs to cache");

        let TransactionOutputs {
            transaction,
            effects,
            written,
            deleted,
            ..
        } = &*tx_outputs;

        // Deletions must be written first. The reason is that one of the deletes
        // may be a child object, and if we write the parent object first, a reader may or may
        // not see the previous version of the child object which would cause an execution fork
        for ObjectKey(id, version) in deleted.iter() {
            self.write_object_entry(id, *version, ObjectEntry::Deleted)
                .await;
        }

        for (object_id, object) in written.iter() {
            self.write_object_entry(object_id, object.version(), object.clone().into())
                .await;
        }

        let tx_digest = *transaction.digest();
        let effects_digest = effects.digest();

        self.dirty
            .pending_transaction_writes
            .insert(tx_digest, tx_outputs.clone());

        // insert transaction effects before executed_effects_digests so that there
        // are never dangling entries in executed_effects_digests
        self.dirty
            .transaction_effects
            .insert(effects_digest, effects.clone());

        self.dirty
            .executed_effects_digests
            .insert(tx_digest, effects_digest);

        self.executed_effects_digests_notify_read
            .notify(&tx_digest, &effects_digest);

        Ok(())
    }

    // Commits dirty data for the given TransactionDigest to the db.
    #[instrument(level = "debug", skip_all)]
    async fn commit_transaction_outputs(
        &self,
        epoch: EpochId,
        digests: &[TransactionDigest],
    ) -> SomaResult {
        trace!(?digests);

        let mut all_outputs = Vec::with_capacity(digests.len());
        for tx in digests {
            let Some(outputs) = self
                .dirty
                .pending_transaction_writes
                .get(tx)
                .map(|o| o.clone())
            else {
                // This can happen in the following rare case:
                // All transactions in the checkpoint are committed to the db (by commit_transaction_outputs,
                // called in CheckpointExecutor::process_executed_transactions), but the process crashes before
                // the checkpoint water mark is bumped. We will then re-commit thhe checkpoint at startup,
                // despite that all transactions are already executed.
                warn!("Attempt to commit unknown transaction {:?}", tx);
                continue;
            };
            all_outputs.push(outputs);
        }

        // Flush writes to disk before removing anything from dirty set. otherwise,
        // a cache eviction could cause a value to disappear briefly, even if we insert to the
        // cache before removing from the dirty set.
        self.store
            .write_transaction_outputs(epoch, &all_outputs)
            .await?;

        for outputs in all_outputs.iter() {
            let tx_digest = outputs.transaction.digest();
            assert!(self
                .dirty
                .pending_transaction_writes
                .remove(tx_digest)
                .is_some());
            self.flush_transactions_from_dirty_to_cached(epoch, *tx_digest, outputs);
        }

        Ok(())
    }

    fn flush_transactions_from_dirty_to_cached(
        &self,
        epoch: EpochId,
        tx_digest: TransactionDigest,
        outputs: &TransactionOutputs,
    ) {
        // Now, remove each piece of committed data from the dirty state and insert it into the cache.
        // TODO: outputs should have a strong count of 1 so we should be able to move out of it
        let TransactionOutputs {
            transaction,
            effects,
            written,
            deleted,
            ..
        } = outputs;

        let effects_digest = effects.digest();

        // Update cache before removing from self.dirty to avoid
        // unnecessary cache misses
        self.cached
            .transactions
            .insert(tx_digest, transaction.clone());
        self.cached
            .transaction_effects
            .insert(effects_digest, effects.clone().into());
        self.cached
            .executed_effects_digests
            .insert(tx_digest, effects_digest);

        self.dirty
            .transaction_effects
            .remove(&effects_digest)
            .expect("effects must exist");

        self.dirty
            .executed_effects_digests
            .remove(&tx_digest)
            .expect("executed effects must exist");

        for (object_id, object) in written.iter() {
            Self::move_version_from_dirty_to_cache(
                &self.dirty.objects,
                &self.cached.object_cache,
                *object_id,
                object.version(),
                &ObjectEntry::Object(object.clone()),
            );
        }

        for ObjectKey(object_id, version) in deleted.iter() {
            Self::move_version_from_dirty_to_cache(
                &self.dirty.objects,
                &self.cached.object_cache,
                *object_id,
                *version,
                &ObjectEntry::Deleted,
            );
        }
    }

    async fn persist_transactions(&self, digests: &[TransactionDigest]) -> SomaResult {
        let mut txns = Vec::with_capacity(digests.len());
        for tx_digest in digests {
            let Some(tx) = self
                .dirty
                .pending_transaction_writes
                .get(tx_digest)
                .map(|o| o.transaction.clone())
            else {
                // tx should exist in the db if it is not in dirty set.
                debug_assert!(self.store.get_transaction_block(tx_digest).is_some());
                // If the transaction is not in dirty, it does not need to be committed.
                // This situation can happen if we build a checkpoint locally which was just executed
                // via state sync.
                continue;
            };

            txns.push((*tx_digest, (*tx).clone()));
        }

        self.store.commit_transactions(&txns)
    }

    // Move the oldest/least entry from the dirty queue to the cache queue.
    // This is called after the entry is committed to the db.
    fn move_version_from_dirty_to_cache<K, V>(
        dirty: &DashMap<K, CachedVersionMap<V>>,
        cache: &MokaCache<K, Arc<Mutex<CachedVersionMap<V>>>>,
        key: K,
        version: Version,
        value: &V,
    ) where
        K: Eq + std::hash::Hash + Clone + Send + Sync + Copy + 'static,
        V: Send + Sync + Clone + Eq + std::fmt::Debug + 'static,
    {
        static MAX_VERSIONS: usize = 3;

        // IMPORTANT: lock both the dirty set entry and the cache entry before modifying either.
        // this ensures that readers cannot see a value temporarily disappear.
        let dirty_entry = dirty.entry(key);
        let cache_entry = cache.entry(key).or_default();
        let mut cache_map = cache_entry.value().lock();

        // insert into cache and drop old versions.
        cache_map.insert(version, value.clone());
        // TODO: make this automatic by giving CachedVersionMap an optional max capacity
        cache_map.truncate_to(MAX_VERSIONS);

        let DashMapEntry::Occupied(mut occupied_dirty_entry) = dirty_entry else {
            panic!("dirty map must exist");
        };

        let removed = occupied_dirty_entry.get_mut().pop_oldest(&version);

        assert_eq!(removed.as_ref(), Some(value), "dirty version must exist");

        // if there are no versions remaining, remove the map entry
        if occupied_dirty_entry.get().is_empty() {
            occupied_dirty_entry.remove();
        }
    }

    // Updates the latest object id cache with an entry that was read from the db.
    // Writes bypass this function, because an object write is guaranteed to be the
    // most recent version (and cannot race with any other writes to that object id)
    //
    // If there are racing calls to this function, it is guaranteed that after a call
    // has returned, reads from that thread will not observe a lower version than the
    // one they inserted
    fn cache_latest_object_by_id(&self, object_id: &ObjectID, object: LatestObjectCacheEntry) {
        trace!("caching object by id: {:?} {:?}", object_id, object);
        // Warning: tricky code!
        let entry = self
            .cached
            .object_by_id_cache
            .entry(*object_id)
            // only one racing insert will call the closure
            .or_insert_with(|| Arc::new(Mutex::new(object.clone())));

        // We may be racing with another thread that observed an older version of the object
        if !entry.is_fresh() {
            // !is_fresh means we lost the race, and entry holds the value that was
            // inserted by the other thread. We need to check if we have a more recent version
            // than the other reader.
            //
            // This could also mean that the entry was inserted by a transaction write. This
            // could occur in the following case:
            //
            // THREAD 1            | THREAD 2
            // reads object at v1  |
            //                     | tx writes object at v2
            // tries to cache v1
            //
            // Thread 1 will see that v2 is already in the cache when it tries to cache it,
            // and will try to update the cache with v1. But the is_newer_than check will fail,
            // so v2 will remain in the cache

            // Ensure only the latest version is inserted.
            let mut entry = entry.value().lock();
            if object.is_newer_than(&entry) {
                *entry = object;
            }
        }
    }

    fn cache_object_not_found(&self, object_id: &ObjectID) {
        self.cache_latest_object_by_id(object_id, LatestObjectCacheEntry::NonExistent);
    }

    fn clear_state_end_of_epoch_impl(&self, _execution_guard: &ExecutionLockWriteGuard<'_>) {
        info!("clearing state at end of epoch");
        assert!(
            self.dirty.pending_transaction_writes.is_empty(),
            "should be empty due to revert_state_update"
        );
        self.dirty.clear();
        info!("clearing old transaction locks");
        self.object_locks.clear();
    }

    fn revert_state_update_impl(&self, tx: &TransactionDigest) -> SomaResult {
        // TODO: remove revert_state_update_impl entirely, and simply drop all dirty
        // state when clear_state_end_of_epoch_impl is called.
        // Futher, once we do this, we can delay the insertion of the transaction into
        // pending_consensus_transactions until after the transaction has executed.
        let Some((_, outputs)) = self.dirty.pending_transaction_writes.remove(tx) else {
            assert!(
                !self.is_tx_already_executed(tx).expect("read cannot fail"),
                "attempt to revert committed transaction"
            );

            // A transaction can be inserted into pending_consensus_transactions, but then reconfiguration
            // can happen before the transaction executes.
            info!("Not reverting {:?} as it was not executed", tx);
            return Ok(());
        };

        for (object_id, object) in outputs.written.iter() {
            self.cached.object_by_id_cache.invalidate(object_id);
        }

        for ObjectKey(object_id, _) in outputs.deleted.iter() {
            self.cached.object_by_id_cache.invalidate(&object_id);
        }

        // Note: individual object entries are removed when clear_state_end_of_epoch_impl is called
        Ok(())
    }

    pub fn clear_caches_and_assert_empty(&self) {
        info!("clearing caches");
        self.cached.clear_and_assert_empty();
    }
}

impl TransactionCacheRead for WritebackCache {
    fn multi_get_transaction_blocks(
        &self,
        digests: &[TransactionDigest],
    ) -> SomaResult<Vec<Option<Arc<VerifiedTransaction>>>> {
        do_fallback_lookup(
            digests,
            |digest| {
                // if let Some(tx) = self.dirty.pending_transaction_writes.get(digest) {

                //     return Ok(CacheResult::Hit(Some(tx.transaction.clone())));
                // }

                if let Some(tx) = self.cached.transactions.get(digest) {
                    return Ok(CacheResult::Hit(Some(tx.clone())));
                }

                Ok(CacheResult::Miss)
            },
            |remaining| {
                self.store
                    .multi_get_transaction_blocks(remaining)
                    .map(|v| v.into_iter().map(|o| o.map(Arc::new)).collect())
            },
        )
    }

    fn multi_get_executed_effects_digests(
        &self,
        digests: &[TransactionDigest],
    ) -> SomaResult<Vec<Option<TransactionEffectsDigest>>> {
        do_fallback_lookup(
            digests,
            |digest| {
                if let Some(digest) = self.dirty.executed_effects_digests.get(digest) {
                    return Ok(CacheResult::Hit(Some(*digest)));
                }

                if let Some(digest) = self.cached.executed_effects_digests.get(digest) {
                    return Ok(CacheResult::Hit(Some(digest)));
                }

                Ok(CacheResult::Miss)
            },
            |remaining| self.store.multi_get_executed_effects_digests(remaining),
        )
    }

    fn multi_get_effects(
        &self,
        digests: &[TransactionEffectsDigest],
    ) -> SomaResult<Vec<Option<TransactionEffects>>> {
        do_fallback_lookup(
            digests,
            |digest| {
                if let Some(effects) = self.dirty.transaction_effects.get(digest) {
                    return Ok(CacheResult::Hit(Some(effects.clone())));
                }

                if let Some(effects) = self.cached.transaction_effects.get(digest) {
                    return Ok(CacheResult::Hit(Some((*effects).clone())));
                }

                Ok(CacheResult::Miss)
            },
            |remaining| self.store.multi_get_effects(remaining.iter()),
        )
    }

    fn notify_read_executed_effects_digests<'a>(
        &'a self,
        digests: &'a [TransactionDigest],
    ) -> BoxFuture<'a, SomaResult<Vec<TransactionEffectsDigest>>> {
        self.executed_effects_digests_notify_read
            .read(digests, |digests| {
                self.multi_get_executed_effects_digests(digests)
            })
            .boxed()
    }
}

/// do_fallback_lookup is a helper function for multi-get operations.
/// It takes a list of keys and first attempts to look up each key in the cache.
/// The cache can return a hit, a miss, or a negative hit (if the object is known to not exist).
/// Any keys that result in a miss are then looked up in the store.
///
/// The "get from cache" and "get from store" behavior are implemented by the caller and provided
/// via the get_cached_key and multiget_fallback functions.
fn do_fallback_lookup<K: Copy, V: Default + Clone>(
    keys: &[K],
    get_cached_key: impl Fn(&K) -> SomaResult<CacheResult<V>>,
    multiget_fallback: impl Fn(&[K]) -> SomaResult<Vec<V>>,
) -> SomaResult<Vec<V>> {
    let mut results = vec![V::default(); keys.len()];
    let mut fallback_keys = Vec::with_capacity(keys.len());
    let mut fallback_indices = Vec::with_capacity(keys.len());

    for (i, key) in keys.iter().enumerate() {
        match get_cached_key(key)? {
            CacheResult::Miss => {
                fallback_keys.push(*key);
                fallback_indices.push(i);
            }
            CacheResult::NegativeHit => (),
            CacheResult::Hit(value) => {
                results[i] = value;
            }
        }
    }

    let fallback_results = multiget_fallback(&fallback_keys)?;
    assert_eq!(fallback_results.len(), fallback_indices.len());
    assert_eq!(fallback_results.len(), fallback_keys.len());

    for (i, result) in fallback_indices
        .into_iter()
        .zip(fallback_results.into_iter())
    {
        results[i] = result;
    }
    Ok(results)
}

impl ExecutionCacheWrite for WritebackCache {
    fn write_transaction_outputs(
        &self,
        epoch_id: EpochId,
        tx_outputs: Arc<TransactionOutputs>,
    ) -> BoxFuture<'_, SomaResult> {
        WritebackCache::write_transaction_outputs(self, epoch_id, tx_outputs).boxed()
    }
}

impl ExecutionCacheCommit for WritebackCache {
    fn commit_transaction_outputs<'a>(
        &'a self,
        epoch: EpochId,
        digests: &'a [TransactionDigest],
    ) -> BoxFuture<'a, SomaResult> {
        WritebackCache::commit_transaction_outputs(self, epoch, digests).boxed()
    }

    fn persist_transactions<'a>(
        &'a self,
        digests: &'a [TransactionDigest],
    ) -> BoxFuture<'a, SomaResult> {
        WritebackCache::persist_transactions(self, digests).boxed()
    }
}

impl ObjectStore for WritebackCache {
    fn get_object(
        &self,
        object_id: &ObjectID,
    ) -> types::storage::storage_error::Result<Option<Object>> {
        ObjectCacheRead::get_object(self, object_id)
            .map_err(types::storage::storage_error::Error::custom)
    }

    fn get_object_by_key(
        &self,
        object_id: &ObjectID,
        version: Version,
    ) -> types::storage::storage_error::Result<Option<Object>> {
        ObjectCacheRead::get_object_by_key(self, object_id, version)
            .map_err(types::storage::storage_error::Error::custom)
    }
}

impl ObjectCacheRead for WritebackCache {
    // get_object and variants.

    fn get_object(&self, id: &ObjectID) -> SomaResult<Option<Object>> {
        self.get_object_impl(id)
    }

    fn get_object_by_key(
        &self,
        object_id: &ObjectID,
        version: Version,
    ) -> SomaResult<Option<Object>> {
        match self.get_object_by_key_cache_only(object_id, version) {
            CacheResult::Hit(object) => Ok(Some(object)),
            CacheResult::NegativeHit => Ok(None),
            CacheResult::Miss => Ok(self.store.get_object_by_key(object_id, version)?),
        }
    }

    fn multi_get_objects_by_key(
        &self,
        object_keys: &[ObjectKey],
    ) -> Result<Vec<Option<Object>>, SomaError> {
        do_fallback_lookup(
            object_keys,
            |key| {
                Ok(match self.get_object_by_key_cache_only(&key.0, key.1) {
                    CacheResult::Hit(maybe_object) => CacheResult::Hit(Some(maybe_object)),
                    CacheResult::NegativeHit => CacheResult::NegativeHit,
                    CacheResult::Miss => CacheResult::Miss,
                })
            },
            |remaining| {
                self.store
                    .multi_get_objects_by_key(remaining)
                    .map_err(Into::into)
            },
        )
    }

    fn object_exists_by_key(&self, object_id: &ObjectID, version: Version) -> SomaResult<bool> {
        match self.get_object_by_key_cache_only(object_id, version) {
            CacheResult::Hit(_) => Ok(true),
            CacheResult::NegativeHit => Ok(false),
            CacheResult::Miss => self.store.object_exists_by_key(object_id, version),
        }
    }

    fn multi_object_exists_by_key(&self, object_keys: &[ObjectKey]) -> SomaResult<Vec<bool>> {
        do_fallback_lookup(
            object_keys,
            |key| {
                Ok(match self.get_object_by_key_cache_only(&key.0, key.1) {
                    CacheResult::Hit(_) => CacheResult::Hit(true),
                    CacheResult::NegativeHit => CacheResult::Hit(false),
                    CacheResult::Miss => CacheResult::Miss,
                })
            },
            |remaining| self.store.multi_object_exists_by_key(remaining),
        )
    }

    fn get_latest_object_ref_or_tombstone(
        &self,
        object_id: ObjectID,
    ) -> SomaResult<Option<ObjectRef>> {
        match self.get_object_entry_by_id_cache_only(&object_id) {
            CacheResult::Hit((version, entry)) => Ok(Some(match entry {
                ObjectEntry::Object(object) => object.compute_object_reference(),
                ObjectEntry::Deleted => (object_id, version, ObjectDigest::OBJECT_DIGEST_DELETED),
            })),
            CacheResult::NegativeHit => Ok(None),
            CacheResult::Miss => self.store.get_latest_object_ref_or_tombstone(object_id),
        }
    }

    fn get_latest_object_or_tombstone(
        &self,
        object_id: ObjectID,
    ) -> Result<Option<(ObjectKey, ObjectOrTombstone)>, SomaError> {
        match self.get_object_entry_by_id_cache_only(&object_id) {
            CacheResult::Hit((version, entry)) => {
                let key = ObjectKey(object_id, version);
                Ok(Some(match entry {
                    ObjectEntry::Object(object) => (key, object.into()),
                    ObjectEntry::Deleted => (
                        key,
                        ObjectOrTombstone::Tombstone((
                            object_id,
                            version,
                            ObjectDigest::OBJECT_DIGEST_DELETED,
                        )),
                    ),
                }))
            }
            CacheResult::NegativeHit => Ok(None),
            CacheResult::Miss => self.store.get_latest_object_or_tombstone(object_id),
        }
    }

    #[instrument(level = "trace", skip_all, fields(object_id, version_bound))]
    fn find_object_lt_or_eq_version(
        &self,
        object_id: ObjectID,
        version_bound: Version,
    ) -> SomaResult<Option<Object>> {
        macro_rules! check_cache_entry {
            ($objects: expr) => {
                if let Some(objects) = $objects {
                    if let Some((_, object)) = objects
                        .all_versions_lt_or_eq_descending(&version_bound)
                        .next()
                    {
                        if let ObjectEntry::Object(object) = object {
                            return Ok(Some(object.clone()));
                        } else {
                            // if we find a tombstone, the object does not exist
                            return Ok(None);
                        }
                    }
                }
            };
        }

        // if we have the latest version cached, and it is within the bound, we are done
        if let Some(latest) = self.cached.object_by_id_cache.get(&object_id) {
            let latest = latest.lock();
            match &*latest {
                LatestObjectCacheEntry::Object(latest_version, object) => {
                    if *latest_version <= version_bound {
                        if let ObjectEntry::Object(object) = object {
                            return Ok(Some(object.clone()));
                        } else {
                            // object is a tombstone, but is still within the version bound

                            return Ok(None);
                        }
                    }
                    // latest object is not within the version bound. fall through.
                }
                // No object by this ID exists at all
                LatestObjectCacheEntry::NonExistent => {
                    return Ok(None);
                }
            }
        }

        Self::with_locked_cache_entries(
            &self.dirty.objects,
            &self.cached.object_cache,
            &object_id,
            |dirty_entry, cached_entry| {
                check_cache_entry!(dirty_entry);
                check_cache_entry!(cached_entry);

                // Much of the time, the query will be for the very latest object version, so
                // try that first. But we have to be careful:
                // 1. We must load the tombstone if it is present, because its version may exceed
                //    the version_bound, in which case we must do a scan.
                // 2. You might think we could just call `self.store.get_latest_object_or_tombstone` here.
                //    But we cannot, because there may be a more recent version in the dirty set, which
                //    we skipped over in check_cache_entry! because of the version bound. However, if we
                //    skipped it above, we will skip it here as well, again due to the version bound.
                // 3. Despite that, we really want to warm the cache here. Why? Because if the object is
                //    cold (not being written to), then we will very soon be able to start serving reads
                //    of it from the object_by_id cache, IF we can warm the cache. If we don't warm the
                //    the cache here, and no writes to the object occur, then we will always have to go
                //    to the db for the object.
                //
                // Lastly, it is important to understand the rationale for all this: If the object is
                // write-hot, we will serve almost all reads to it from the dirty set (or possibly the
                // cached set if it is only written to once every few checkpoints). If the object is
                // write-cold (or non-existent) and read-hot, then we will serve almost all reads to it
                // from the object_by_id cache check above.  Most of the apparently wasteful code here
                // exists only to ensure correctness in all the edge cases.
                let latest: Option<(Version, ObjectEntry)> = if let Some(dirty_set) = dirty_entry {
                    dirty_set
                        .get_highest()
                        .cloned()
                        .tap_none(|| panic!("dirty set cannot be empty"))
                } else {
                    self.store.get_latest_object_or_tombstone(object_id)?.map(
                        |(ObjectKey(_, version), obj_or_tombstone)| {
                            (version, ObjectEntry::from(obj_or_tombstone))
                        },
                    )
                };

                if let Some((obj_version, obj_entry)) = latest {
                    // we can always cache the latest object (or tombstone), even if it is not within the
                    // version_bound. This is done in order to warm the cache in the case where a sequence
                    // of transactions all read the same child object without writing to it.
                    self.cache_latest_object_by_id(
                        &object_id,
                        LatestObjectCacheEntry::Object(obj_version, obj_entry.clone()),
                    );

                    if obj_version <= version_bound {
                        match obj_entry {
                            ObjectEntry::Object(object) => Ok(Some(object)),
                            ObjectEntry::Deleted => Ok(None),
                        }
                    } else {
                        // The latest object exceeded the bound, so now we have to do a scan
                        // But we already know there is no dirty entry within the bound,
                        // so we go to the db.
                        self.store
                            .find_object_lt_or_eq_version(object_id, version_bound)
                    }
                } else {
                    // no object found in dirty set or db, object does not exist
                    // When this is called from a read api (i.e. not the execution path) it is
                    // possible that the object has been deleted and pruned. In this case,
                    // there would be no entry at all on disk, but we may have a tombstone in the
                    // cache
                    let highest = cached_entry.and_then(|c| c.get_highest());
                    assert!(highest.is_none());
                    self.cache_object_not_found(&object_id);
                    Ok(None)
                }
            },
        )
    }

    fn get_lock(&self, obj_ref: ObjectRef, epoch_store: &AuthorityPerEpochStore) -> LockResult {
        let cur_epoch = epoch_store.epoch();
        match self.get_object_by_id_cache_only(&obj_ref.0) {
            CacheResult::Hit((_, obj)) => {
                let actual_objref = obj.compute_object_reference();
                if obj_ref != actual_objref {
                    Ok(ObjectLockStatus::LockedAtDifferentVersion {
                        locked_ref: actual_objref,
                    })
                } else {
                    // requested object ref is live, check if there is a lock
                    Ok(
                        match self
                            .object_locks
                            .get_transaction_lock(&obj_ref, epoch_store)?
                        {
                            Some(tx_digest) => ObjectLockStatus::LockedToTx {
                                locked_by_tx: tx_digest,
                            },
                            None => ObjectLockStatus::Initialized,
                        },
                    )
                }
            }
            CacheResult::NegativeHit => {
                Err(SomaError::ObjectNotFound {
                    object_id: obj_ref.0,
                    // even though we know the requested version, we leave it as None to indicate
                    // that the object does not exist at any version
                    version: None,
                })
            }
            CacheResult::Miss => self.store.get_lock(obj_ref, epoch_store),
        }
    }

    fn get_system_state_object(&self) -> SomaResult<SystemState> {
        get_system_state(self)
    }
}

impl AccumulatorStore for WritebackCache {
    fn get_root_state_accumulator_for_commit(
        &self,
        commit: CommitIndex,
    ) -> SomaResult<Option<Accumulator>> {
        self.store.get_root_state_accumulator_for_commit(commit)
    }

    fn get_root_state_accumulator_for_highest_commit(
        &self,
    ) -> SomaResult<Option<(CommitIndex, Accumulator)>> {
        self.store.get_root_state_accumulator_for_highest_commit()
    }

    fn insert_state_accumulator_for_commit(
        &self,
        commit: &CommitIndex,
        acc: &Accumulator,
    ) -> SomaResult {
        self.store.insert_state_accumulator_for_commit(commit, acc)
    }

    fn iter_live_object_set(&self) -> Box<dyn Iterator<Item = LiveObject> + '_> {
        // The only time it is safe to iterate the live object set is at an epoch boundary,
        // at which point the db is consistent and the dirty cache is empty. So this does
        // read the cache
        assert!(
            self.dirty.is_empty(),
            "cannot iterate live object set with dirty data"
        );
        self.store.iter_live_object_set()
    }
}
