use std::{
    cmp::{max, Reverse},
    collections::{hash_map, BTreeSet, BinaryHeap, HashMap, HashSet},
    sync::Arc,
};

use lru::LruCache;
use parking_lot::RwLock;
use tokio::{sync::mpsc::UnboundedSender, time::Instant};
use tracing::{debug, error, info, instrument, trace, warn};
use types::{
    accumulator::CommitIndex,
    base::FullObjectID,
    committee::EpochId,
    digests::{TransactionDigest, TransactionEffectsDigest},
    object::Version,
    storage::InputKey,
    transaction::{VerifiedCertificate, VerifiedExecutableTransaction},
};

use crate::{
    cache::{ObjectCacheRead, TransactionCacheRead},
    epoch_store::AuthorityPerEpochStore,
};

/// Minimum capacity of HashMaps used in TransactionManager.
const MIN_HASHMAP_CAPACITY: usize = 1000;

/// TransactionManager is responsible for publishing a stream of certified transactions (certificates) ready to execute.
/// It receives certificates from conseensus, validator RPC handlers, and checkpoint executor.
/// Execution driver subscribes to the stream of ready certificates from TransactionManager, and
/// executes them in parallel.
/// The actual execution logic is inside AuthorityState. After a transaction commits and updates
/// storage, committed certificates are notified back to TransactionManager.
pub struct TransactionManager {
    // transaction_cache_read: Arc<dyn TransactionCacheRead>,
    tx_ready_certificates: UnboundedSender<PendingCertificate>,
    // inner is a doubly nested lock so that we can enforce that an outer lock (for read) is held
    // before the inner lock (for read or write) can be acquired. During reconfiguration, we acquire
    // the outer lock for write, to ensure that no other threads can be running while we reconfigure.
    inner: RwLock<RwLock<Inner>>,

    transaction_cache_read: Arc<dyn TransactionCacheRead>,

    object_cache_read: Arc<dyn ObjectCacheRead>,
}

#[derive(Clone, Debug)]
pub struct PendingCertificate {
    // Certified transaction to be executed.
    pub certificate: VerifiedExecutableTransaction,
    // When executing from checkpoint, the certified effects digest is provided, so that forks can
    // be detected prior to committing the transaction.
    pub expected_effects_digest: Option<TransactionEffectsDigest>,
    // The input object this certificate is waiting for to become available in order to be executed.
    pub waiting_input_objects: BTreeSet<InputKey>,
    // Commit index of the certificate.
    pub commit: Option<CommitIndex>,
}

struct Inner {
    // Current epoch of TransactionManager.
    epoch: EpochId,

    // Maps missing input objects to transactions in pending_certificates.
    missing_inputs: HashMap<InputKey, BTreeSet<TransactionDigest>>,

    // Stores age info for all transactions depending on each object.
    // Used for throttling signing and submitting transactions depending on hot objects.
    // An `IndexMap` is used to ensure that the insertion order is preserved.
    input_objects: HashMap<FullObjectID, TransactionQueue>,

    // Maps object IDs to the highest observed sequence number of the object. When the value is
    // None, indicates that the object is immutable, corresponding to an InputKey with no sequence
    // number.
    available_objects_cache: AvailableObjectsCache,

    // A transaction enqueued to TransactionManager must be in either pending_certificates or
    // executing_certificates.

    // Maps transaction digests to their content and missing input objects.
    pending_certificates: HashMap<TransactionDigest, PendingCertificate>,

    // Transactions that  have not finished execution.
    executing_certificates: HashSet<TransactionDigest>,
}

impl Inner {
    fn new(epoch: EpochId) -> Inner {
        Inner {
            epoch,
            missing_inputs: HashMap::with_capacity(MIN_HASHMAP_CAPACITY),
            input_objects: HashMap::with_capacity(MIN_HASHMAP_CAPACITY),
            available_objects_cache: AvailableObjectsCache::new(),
            pending_certificates: HashMap::with_capacity(MIN_HASHMAP_CAPACITY),
            executing_certificates: HashSet::with_capacity(MIN_HASHMAP_CAPACITY),
        }
    }

    // Checks if there is any transaction waiting on `input_key`. Returns all the pending
    // transactions that are ready to be executed.
    // Must ensure input_key is available in storage before calling this function.
    fn find_ready_transactions(
        &mut self,
        input_key: InputKey,
        update_cache: bool,
    ) -> Vec<PendingCertificate> {
        if update_cache {
            self.available_objects_cache.insert(&input_key);
        }

        let mut ready_certificates = Vec::new();

        let Some(digests) = self.missing_inputs.remove(&input_key) else {
            // No transaction is waiting on the object yet.
            return ready_certificates;
        };

        let input_txns = self
            .input_objects
            .get_mut(&input_key.id())
            .unwrap_or_else(|| {
                panic!(
                    "# of transactions waiting on object {:?} cannot be 0",
                    input_key.id()
                )
            });
        for digest in digests.iter() {
            let age_opt = input_txns.remove(digest).expect("digest must be in map");
        }

        if input_txns.is_empty() {
            self.input_objects.remove(&input_key.id());
        }

        for digest in digests {
            // Pending certificate must exist.
            let pending_cert = self.pending_certificates.get_mut(&digest).unwrap();
            assert!(pending_cert.waiting_input_objects.remove(&input_key));
            // When a certificate has all its input objects, it is ready to execute.
            if pending_cert.waiting_input_objects.is_empty() {
                let pending_cert = self.pending_certificates.remove(&digest).unwrap();
                ready_certificates.push(pending_cert);
            } else {
                // TODO: we should start logging this at a higher level after some period of
                // time has elapsed.
                trace!(tx_digest = ?digest,missing = ?pending_cert.waiting_input_objects, "Certificate waiting on missing inputs");
            }
        }

        ready_certificates
    }

    fn maybe_reserve_capacity(&mut self) {
        self.missing_inputs.maybe_reserve_capacity();
        self.input_objects.maybe_reserve_capacity();
        self.pending_certificates.maybe_reserve_capacity();
        self.executing_certificates.maybe_reserve_capacity();
    }

    /// After reaching 1/4 load in hashmaps, decrease capacity to increase load to 1/2.
    fn maybe_shrink_capacity(&mut self) {
        self.missing_inputs.maybe_shrink_capacity();
        self.input_objects.maybe_shrink_capacity();
        self.pending_certificates.maybe_shrink_capacity();
        self.executing_certificates.maybe_shrink_capacity();
    }
}

impl TransactionManager {
    /// If a node restarts, transaction manager recovers in-memory data from pending_certificates,
    /// which contains certified transactions from consensus output and RPC that are not executed.
    /// Transactions from other sources, e.g. checkpoint executor, have own persistent storage to
    /// retry transactions.
    pub(crate) fn new(
        object_cache_read: Arc<dyn ObjectCacheRead>,
        // transaction_cache_read: Arc<dyn TransactionCacheRead>,
        epoch_store: &AuthorityPerEpochStore,
        tx_ready_certificates: UnboundedSender<PendingCertificate>,
        transaction_cache_read: Arc<dyn TransactionCacheRead>,
    ) -> TransactionManager {
        let transaction_manager = TransactionManager {
            object_cache_read,
            inner: RwLock::new(RwLock::new(Inner::new(epoch_store.epoch()))),
            tx_ready_certificates,
            transaction_cache_read,
        };
        // TODO: Recover pending certificates and fetch commit index for each when node restarts
        transaction_manager.enqueue(
            epoch_store.all_pending_execution().unwrap(),
            epoch_store,
            None,
        );
        transaction_manager
    }

    /// Enqueues certificates / verified transactions into TransactionManager. Once all of the input objects are available
    /// locally for a certificate, the certified transaction will be sent to execution driver.
    ///
    /// REQUIRED: Shared object locks must be taken before calling enqueueing transactions
    /// with shared objects!
    #[instrument(level = "trace", skip_all)]
    pub(crate) fn enqueue_certificates(
        &self,
        certs: Vec<VerifiedCertificate>,
        epoch_store: &AuthorityPerEpochStore,
        commit: Option<CommitIndex>,
    ) {
        let executable_txns = certs
            .into_iter()
            .map(VerifiedExecutableTransaction::new_from_certificate)
            .collect();
        self.enqueue(executable_txns, epoch_store, commit)
    }

    #[instrument(level = "trace", skip_all)]
    pub(crate) fn enqueue(
        &self,
        certs: Vec<VerifiedExecutableTransaction>,
        epoch_store: &AuthorityPerEpochStore,
        commit: Option<CommitIndex>,
    ) {
        let certs = certs.into_iter().map(|cert| (cert, None, commit)).collect();
        self.enqueue_impl(certs, epoch_store)
    }

    #[instrument(level = "trace", skip_all)]
    pub(crate) fn enqueue_with_expected_effects_digest(
        &self,
        certs: Vec<(VerifiedExecutableTransaction, TransactionEffectsDigest)>,
        epoch_store: &AuthorityPerEpochStore,
    ) {
        let certs = certs
            .into_iter()
            .map(|(cert, fx)| (cert, Some(fx), None))
            .collect();
        self.enqueue_impl(certs, epoch_store)
    }

    fn enqueue_impl(
        &self,
        certs: Vec<(
            VerifiedExecutableTransaction,
            Option<TransactionEffectsDigest>,
            Option<CommitIndex>,
        )>,
        epoch_store: &AuthorityPerEpochStore,
    ) {
        let reconfig_lock = self.inner.read();

        // filter out already executed certs
        let certs: Vec<_> = certs
            .into_iter()
            .filter(|(cert, _, _)| {
                let digest = *cert.digest();
                // skip already executed txes
                if self
                    .transaction_cache_read
                    .is_tx_already_executed(&digest)
                    .unwrap_or_else(|err| {
                        panic!("Failed to check if tx is already executed: {:?}", err)
                    })
                {
                    false
                } else {
                    true
                }
            })
            .collect();

        let mut object_availability: HashMap<InputKey, Option<bool>> = HashMap::new();
        let mut receiving_objects: HashSet<InputKey> = HashSet::new();
        let certs: Vec<_> = certs
            .into_iter()
            .filter_map(|(cert, fx_digest, commit)| {
                let input_object_kinds = cert
                    .data()
                    .intent_message()
                    .value
                    .input_objects()
                    .expect("input_objects() cannot fail");
                let mut input_object_keys =
                    match epoch_store.get_input_object_keys(&cert.key(), &input_object_kinds) {
                        Ok(keys) => keys,
                        Err(e) => {
                            // Because we do not hold the transaction lock during enqueue, it is possible
                            // that the transaction was executed and the shared version assignments deleted
                            // since the earlier check. This is a rare race condition, and it is better to
                            // handle it ad-hoc here than to hold tx locks for every cert for the duration
                            // of this function in order to remove the race.
                            if self
                                .transaction_cache_read
                                .is_tx_already_executed(cert.digest())
                                .is_ok_and(|v| v)
                            {
                                return None;
                            }
                            panic!("Failed to get input object keys: {:?}", e);
                        }
                    };

                if input_object_kinds.len() != input_object_keys.len() {
                    error!("Duplicated input objects: {:?}", input_object_kinds);
                }

                let receiving_object_entries =
                    cert.data().intent_message().value.receiving_objects();
                for entry in receiving_object_entries {
                    let key = InputKey::VersionedObject {
                        // TODO: Add support for receiving ConsensusV2 objects. For now this assumes fastpath.
                        id: FullObjectID::new(entry.0, None),
                        version: entry.1,
                    };
                    receiving_objects.insert(key);
                    input_object_keys.insert(key);
                }

                for key in input_object_keys.iter() {
                    if key.is_cancelled() {
                        // Cancelled txn objects should always be available immediately.
                        // Don't need to wait on these objects for execution.
                        object_availability.insert(*key, Some(true));
                    } else {
                        object_availability.insert(*key, None);
                    }
                }


                // ADDED: Check for transactions with shared objects that have assigned versions
                let has_shared_objects = cert.contains_shared_object();
                
                if has_shared_objects {
                    if let Some(assigned_versions) = epoch_store.get_assigned_shared_object_versions(&cert.key()) {
                        debug!(
                            tx_digest = ?cert.digest(),
                            "Found assigned shared versions for tx: {:?}",
                            assigned_versions
                        );
                        
                        // Mark assigned versions as available to break circular dependency
                        for ((id, _), assigned_version) in &assigned_versions {
                            for key in input_object_keys.iter() {
                                if key.id().id() == *id && key.version() == Some(*assigned_version) {
                                    debug!(
                                        tx_digest = ?cert.digest(),
                                        object_id = ?id,
                                        version = ?assigned_version,
                                        "Marking shared object as immediately available for transaction with assigned version"
                                    );
                                    object_availability.insert(*key, Some(true));
                                }
                            }
                        }
                    }
                }

                Some((cert, fx_digest, input_object_keys, commit))
            })
            .collect();

        {
            let mut inner = reconfig_lock.write();
            for (key, value) in object_availability.iter_mut() {
                if value.is_some_and(|available| available) {
                    continue;
                }
                if let Some(available) = inner.available_objects_cache.is_object_available(key) {
                    *value = Some(available);
                }
            }
            // make sure we don't miss any cache entries while the lock is not held.
            inner.available_objects_cache.enable_unbounded_cache();
        }

        let input_object_cache_misses = object_availability
            .iter()
            .filter_map(|(key, value)| if value.is_none() { Some(*key) } else { None })
            .collect::<Vec<_>>();

        // Checking object availability without holding TM lock to reduce contention.
        // But input objects can become available before TM lock is acquired.
        // So missing objects' availability are checked again after acquiring TM lock.
        let cache_miss_availability = self
            .object_cache_read
            .multi_input_objects_available(
                &input_object_cache_misses,
                receiving_objects,
                epoch_store.epoch(),
            )
            .into_iter()
            .zip(input_object_cache_misses);

        // After this point, the function cannot return early and must run to the end. Otherwise,
        // it can lead to data inconsistencies and potentially some transactions will never get
        // executed.

        // Internal lock is held only for updating the internal state.
        let mut inner = reconfig_lock.write();

        for (available, key) in cache_miss_availability {
            if available && key.version().is_none() {
                // Mutable objects obtained from cache_miss_availability usually will not be read
                // again, so we do not want to evict other objects in order to insert them into the
                // cache. However, packages will likely be read often, so we do want to insert them
                // even if they cause evictions.
                inner.available_objects_cache.insert(&key);
            }
            object_availability
                .insert(key, Some(available))
                .expect("entry must already exist");
        }

        // Now recheck the cache for anything that became available (via notify_commit) since we
        // read cache_miss_availability - because the cache is unbounded mode it is guaranteed to
        // contain all notifications that arrived since we released the lock on self.inner.
        for (key, value) in object_availability.iter_mut() {
            if !value.expect("all objects must have been checked by now") {
                if let Some(true) = inner.available_objects_cache.is_object_available(key) {
                    *value = Some(true);
                }
            }
        }

        inner.available_objects_cache.disable_unbounded_cache();

        let mut pending = Vec::new();
        let pending_cert_enqueue_time = Instant::now();

        for (cert, expected_effects_digest, input_object_keys, commit) in certs {
            pending.push(PendingCertificate {
                certificate: cert,
                expected_effects_digest,
                commit,
                waiting_input_objects: input_object_keys,
            });
        }

        for mut pending_cert in pending {
            // Tx lock is not held here, which makes it possible to send duplicated transactions to
            // the execution driver after crash-recovery, when the same transaction is recovered
            // from recovery log and pending certificates table. The transaction will still only
            // execute once, because tx lock is acquired in execution driver and executed effects
            // table is consulted. So this behavior is benigh.
            let digest = *pending_cert.certificate.digest();

            if inner.epoch != pending_cert.certificate.epoch() {
                warn!(
                    "Ignoring enqueued certificate from wrong epoch. Expected={} Certificate={:?}",
                    inner.epoch, pending_cert.certificate
                );
                continue;
            }

            // skip already pending txes
            if inner.pending_certificates.contains_key(&digest) {
                continue;
            }
            // skip already executing txes
            if inner.executing_certificates.contains(&digest) {
                continue;
            }
            // skip already executed txes
            let is_tx_already_executed = self
                .transaction_cache_read
                .is_tx_already_executed(&digest)
                .expect("Check if tx is already executed should not fail");
            if is_tx_already_executed {
                continue;
            }

            // ADDED: Special handling for transactions with shared objects that have assigned versions
            let has_shared_objects = pending_cert.certificate.contains_shared_object();
                    
            if has_shared_objects {
                // For transactions with shared objects that have assigned versions,
                // check which objects we can skip waiting for
                if let Some(assigned_versions) = epoch_store.get_assigned_shared_object_versions(&pending_cert.certificate.key()) {
                    debug!(
                        tx_digest = ?digest,
                        "Transaction has assigned versions: {:?}",
                        assigned_versions
                    );
                    
                    // Create a map for quick lookups
                    let assigned_version_map: HashMap<_, _> = assigned_versions
                        .iter()
                        .map(|((id, _), version)| (*id, *version))
                        .collect();
                    
                    // Filter out objects that should be considered available
                    pending_cert.waiting_input_objects.retain(|key| {
                        if let Some(version) = key.version() {
                            if let Some(&assigned_version) = assigned_version_map.get(&key.id().id()) {
                                if version == assigned_version {
                                    // This is an assigned shared object - don't wait for it
                                    debug!(
                                        tx_digest = ?digest,
                                        object_id = ?key.id().id(),
                                        version = ?version,
                                        "Skipping wait for shared object with assigned version"
                                    );
                                    return false;
                                }
                            }
                        }
                        true
                    });
                }
            } else {
                // Original handling for transactions without shared objects
                let mut waiting_input_objects = BTreeSet::new();
                std::mem::swap(
                    &mut waiting_input_objects,
                    &mut pending_cert.waiting_input_objects,
                );
                
                for key in waiting_input_objects {
                    if !object_availability[&key].unwrap() {
                        // The input object is not yet available.
                        info!(
                            "input object is not yet available {} {:?}",
                            key.id().id(),
                            key.version()
                        );
                        pending_cert.waiting_input_objects.insert(key);

                        assert!(
                            inner.missing_inputs.entry(key).or_default().insert(digest),
                            "Duplicated certificate {:?} for missing object {:?}",
                            digest,
                            key
                        );
                        let input_txns = inner.input_objects.entry(key.id()).or_default();
                        input_txns.insert(digest, pending_cert_enqueue_time);
                    }
                }
            }

            // Ready transactions can start to execute.
            if pending_cert.waiting_input_objects.is_empty() {
                // Send to execution driver for execution.
                debug!(
                    tx_digest = ?digest, 
                    has_shared_objects = ?has_shared_objects,
                    "Certificate ready for immediate execution"
                );
                self.certificate_ready(&mut inner, pending_cert);
                continue;
            }

            debug!(
                tx_digest = ?digest,
                remaining_objects = ?pending_cert.waiting_input_objects.len(),
                "Certificate not yet ready, waiting for input objects"
            );

            assert!(
                inner
                    .pending_certificates
                    .insert(digest, pending_cert)
                    .is_none(),
                "Duplicated pending certificate {:?}",
                digest
            );
        }

        inner.maybe_reserve_capacity();
    }

    /// Notifies TransactionManager about a transaction that has been committed.
    #[instrument(level = "trace", skip_all)]
    pub(crate) fn notify_commit(
        &self,
        digest: &TransactionDigest,
        output_object_keys: Vec<InputKey>,
        epoch_store: &AuthorityPerEpochStore,
    ) {
        let reconfig_lock = self.inner.read();
        {
            let commit_time = Instant::now();
            let mut inner = reconfig_lock.write();

            if inner.epoch != epoch_store.epoch() {
                warn!("Ignoring committed certificate from wrong epoch. Expected={} Actual={} CertificateDigest={:?}", inner.epoch, epoch_store.epoch(), digest);
                return;
            }

            self.objects_available_locked(
                &mut inner,
                epoch_store,
                output_object_keys,
                true,
                commit_time,
            );

            if !inner.executing_certificates.remove(digest) {
                trace!("{:?} not found in executing certificates, likely because it is a system transaction", digest);
                return;
            }

            inner.maybe_shrink_capacity();
        }
    }

    #[instrument(level = "trace", skip_all)]
    fn objects_available_locked(
        &self,
        inner: &mut Inner,
        epoch_store: &AuthorityPerEpochStore,
        input_keys: Vec<InputKey>,
        update_cache: bool,
        available_time: Instant,
    ) {
        if inner.epoch != epoch_store.epoch() {
            warn!(
                "Ignoring objects committed from wrong epoch. Expected={} Actual={} \
                 Objects={:?}",
                inner.epoch,
                epoch_store.epoch(),
                input_keys,
            );
            return;
        }

        for input_key in input_keys {
            trace!(?input_key, "object available");
            for mut ready_cert in inner.find_ready_transactions(input_key, update_cache) {
                self.certificate_ready(inner, ready_cert);
            }
        }
    }

    /// Sends the ready certificate for execution.
    fn certificate_ready(&self, inner: &mut Inner, pending_certificate: PendingCertificate) {
        debug!(tx_digest = ?pending_certificate.certificate.digest(), "certificate ready");
        assert_eq!(pending_certificate.waiting_input_objects.len(), 0);
        // Record as an executing certificate.
        assert!(inner
            .executing_certificates
            .insert(*pending_certificate.certificate.digest()));

        let _ = self.tx_ready_certificates.send(pending_certificate);
    }

    // Returns the number of transactions pending or being executed right now.
    pub(crate) fn inflight_queue_len(&self) -> usize {
        let reconfig_lock = self.inner.read();
        let inner = reconfig_lock.read();
        inner.pending_certificates.len() + inner.executing_certificates.len()
    }

    // Reconfigures the TransactionManager for a new epoch. Existing transactions will be dropped
    // because they are no longer relevant and may be incorrect in the new epoch.
    pub(crate) fn reconfigure(&self, new_epoch: EpochId) {
        let reconfig_lock = self.inner.write();
        let mut inner = reconfig_lock.write();
        *inner = Inner::new(new_epoch);
    }
}

trait ResizableHashMap<K, V> {
    fn maybe_reserve_capacity(&mut self);
    fn maybe_shrink_capacity(&mut self);
}

impl<K, V> ResizableHashMap<K, V> for HashMap<K, V>
where
    K: std::cmp::Eq + std::hash::Hash,
{
    /// After reaching 3/4 load in hashmaps, increase capacity to decrease load to 1/2.
    fn maybe_reserve_capacity(&mut self) {
        if self.len() > self.capacity() * 3 / 4 {
            self.reserve(self.capacity() / 2);
        }
    }

    /// After reaching 1/4 load in hashmaps, decrease capacity to increase load to 1/2.
    fn maybe_shrink_capacity(&mut self) {
        if self.len() > MIN_HASHMAP_CAPACITY && self.len() < self.capacity() / 4 {
            self.shrink_to(max(self.capacity() / 2, MIN_HASHMAP_CAPACITY))
        }
    }
}

trait ResizableHashSet<K> {
    fn maybe_reserve_capacity(&mut self);
    fn maybe_shrink_capacity(&mut self);
}

impl<K> ResizableHashSet<K> for HashSet<K>
where
    K: std::cmp::Eq + std::hash::Hash,
{
    /// After reaching 3/4 load in hashset, increase capacity to decrease load to 1/2.
    fn maybe_reserve_capacity(&mut self) {
        if self.len() > self.capacity() * 3 / 4 {
            self.reserve(self.capacity() / 2);
        }
    }

    /// After reaching 1/4 load in hashset, decrease capacity to increase load to 1/2.
    fn maybe_shrink_capacity(&mut self) {
        if self.len() > MIN_HASHMAP_CAPACITY && self.len() < self.capacity() / 4 {
            self.shrink_to(max(self.capacity() / 2, MIN_HASHMAP_CAPACITY))
        }
    }
}

#[derive(Default, Debug)]
struct TransactionQueue {
    digests: HashMap<TransactionDigest, Instant>,
    ages: BinaryHeap<(Reverse<Instant>, TransactionDigest)>,
}

impl TransactionQueue {
    fn len(&self) -> usize {
        self.digests.len()
    }

    fn is_empty(&self) -> bool {
        self.digests.is_empty()
    }

    /// Insert the digest into the queue with the given time. If the digest is
    /// already in the queue, this is a no-op.
    fn insert(&mut self, digest: TransactionDigest, time: Instant) {
        if let hash_map::Entry::Vacant(entry) = self.digests.entry(digest) {
            entry.insert(time);
            self.ages.push((Reverse(time), digest));
        }
    }

    /// Remove the digest from the queue. Returns the time the digest was
    /// inserted into the queue, if it was present.
    ///
    /// After removing the digest, first() will return the new oldest entry
    /// in the queue (which may be unchanged).
    fn remove(&mut self, digest: &TransactionDigest) -> Option<Instant> {
        let when = self.digests.remove(digest)?;

        // This loop removes all previously inserted entries that no longer
        // correspond to live entries in self.digests. When the loop terminates,
        // the top of the heap will be the oldest live entry.
        // Amortized complexity of `remove` is O(lg(n)).
        while !self.ages.is_empty() {
            let first = self.ages.peek().expect("heap cannot be empty");

            // We compare the exact time of the entry, because there may be an
            // entry in the heap that was previously inserted and removed from
            // digests, and we want to ignore it. (see test_transaction_queue_remove_in_order)
            if self.digests.get(&first.1) == Some(&first.0 .0) {
                break;
            }

            self.ages.pop();
        }

        Some(when)
    }

    /// Return the oldest entry in the queue.
    fn first(&self) -> Option<(Instant, TransactionDigest)> {
        self.ages.peek().map(|(time, digest)| (time.0, *digest))
    }
}

struct AvailableObjectsCache {
    cache: CacheInner,
    unbounded_cache_enabled: usize,
}

impl AvailableObjectsCache {
    fn new() -> Self {
        Self::new_with_size(100000)
    }

    fn new_with_size(size: usize) -> Self {
        Self {
            cache: CacheInner::new(size),
            unbounded_cache_enabled: 0,
        }
    }

    fn enable_unbounded_cache(&mut self) {
        self.unbounded_cache_enabled += 1;
    }

    fn disable_unbounded_cache(&mut self) {
        assert!(self.unbounded_cache_enabled > 0);
        self.unbounded_cache_enabled -= 1;
    }

    fn insert(&mut self, object: &InputKey) {
        self.cache.insert(object);
        if self.unbounded_cache_enabled == 0 {
            self.cache.shrink();
        }
    }

    fn is_object_available(&mut self, object: &InputKey) -> Option<bool> {
        self.cache.is_object_available(object)
    }
}

struct CacheInner {
    versioned_cache: LruCache<FullObjectID, Version>,
    // we cache packages separately, because they are more expensive to look up in the db, so we
    // don't want to evict packages in favor of mutable objects.
    unversioned_cache: LruCache<FullObjectID, ()>,

    max_size: usize,
}

impl CacheInner {
    fn new(max_size: usize) -> Self {
        Self {
            versioned_cache: LruCache::unbounded(),
            unversioned_cache: LruCache::unbounded(),
            max_size,
        }
    }
}

impl CacheInner {
    fn shrink(&mut self) {
        while self.versioned_cache.len() > self.max_size {
            self.versioned_cache.pop_lru();
        }
        while self.unversioned_cache.len() > self.max_size {
            self.unversioned_cache.pop_lru();
        }
    }

    fn insert(&mut self, object: &InputKey) {
        if let Some(version) = object.version() {
            if let Some((previous_id, previous_version)) =
                self.versioned_cache.push(object.id(), version)
            {
                if previous_id == object.id() && previous_version > version {
                    // do not allow highest known version to decrease
                    // This should not be possible unless bugs are introduced elsewhere in this
                    // module.
                    self.versioned_cache.put(object.id(), previous_version);
                }
            }
        } else if let Some((previous_id, _)) = self.unversioned_cache.push(object.id(), ()) {
            // lru_cache will does not check if the value being evicted is the same as the value
            // being inserted, so we do need to check if the id is different before counting this
            // as an eviction.
        }
    }

    // Returns Some(true/false) for a definitive result. Returns None if the caller must defer to
    // the db.
    fn is_object_available(&mut self, object: &InputKey) -> Option<bool> {
        if let Some(version) = object.version() {
            if let Some(current) = self.versioned_cache.get(&object.id()) {
                Some(*current >= version)
            } else {
                None
            }
        } else {
            self.unversioned_cache.get(&object.id()).map(|_| true)
        }
    }
}
