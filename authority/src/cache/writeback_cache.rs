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

use dashmap::DashMap;
use futures::{future::BoxFuture, FutureExt};
use moka::sync::Cache as MokaCache;
use std::sync::Arc;
use tracing::{instrument, trace, warn};
use types::{
    committee::EpochId,
    digests::{TransactionDigest, TransactionEffectsDigest},
    effects::TransactionEffects,
    envelope::Message,
    error::SomaResult,
    transaction::VerifiedTransaction,
    tx_outputs::TransactionOutputs,
};
use utils::notify_read::NotifyRead;

use crate::store::AuthorityStore;

use super::{ExecutionCacheCommit, ExecutionCacheWrite, TransactionCacheRead};

enum CacheResult<T> {
    /// Entry is in the cache
    Hit(T),
    /// Entry is not in the cache and is known to not exist
    NegativeHit,
    /// Entry is not in the cache and may or may not exist in the store
    Miss,
}

/// UncommitedData stores execution outputs that are not yet written to the db. Entries in this
/// struct can only be purged after they are committed.
struct UncommittedData {
    transaction_effects: DashMap<TransactionEffectsDigest, TransactionEffects>,

    executed_effects_digests: DashMap<TransactionDigest, TransactionEffectsDigest>,
    // Transaction outputs that have not yet been written to the DB. Items are removed from this
    // table as they are flushed to the db.
    pending_transaction_writes: DashMap<TransactionDigest, Arc<TransactionOutputs>>,
}

impl UncommittedData {
    fn new() -> Self {
        Self {
            transaction_effects: DashMap::new(),
            executed_effects_digests: DashMap::new(),
            pending_transaction_writes: DashMap::new(),
        }
    }

    fn clear(&self) {
        self.transaction_effects.clear();
        self.executed_effects_digests.clear();
        self.pending_transaction_writes.clear();
    }

    fn is_empty(&self) -> bool {
        self.transaction_effects.is_empty()
            && self.executed_effects_digests.is_empty()
            && self.pending_transaction_writes.is_empty()
    }
}

// TODO: set this via the config
static MAX_CACHE_SIZE: u64 = 10000;

/// CachedData stores data that has been committed to the db, but is likely to be read soon.
struct CachedCommittedData {
    transactions: MokaCache<TransactionDigest, Arc<VerifiedTransaction>>,

    transaction_effects: MokaCache<TransactionEffectsDigest, Arc<TransactionEffects>>,

    executed_effects_digests: MokaCache<TransactionDigest, TransactionEffectsDigest>,
}

impl CachedCommittedData {
    fn new() -> Self {
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
            transactions,
            transaction_effects,

            executed_effects_digests,
        }
    }

    fn clear_and_assert_empty(&self) {
        self.transactions.invalidate_all();
        self.executed_effects_digests.invalidate_all();
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

pub struct WritebackCache {
    dirty: UncommittedData,
    cached: CachedCommittedData,
    executed_effects_digests_notify_read: NotifyRead<TransactionDigest, TransactionEffectsDigest>,
    store: Arc<AuthorityStore>,
}

impl WritebackCache {
    pub fn new(store: Arc<AuthorityStore>) -> Self {
        Self {
            dirty: UncommittedData::new(),
            cached: CachedCommittedData::new(),
            executed_effects_digests_notify_read: NotifyRead::new(),
            store,
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
            ..
        } = &*tx_outputs;

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
