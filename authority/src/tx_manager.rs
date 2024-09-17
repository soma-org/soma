use std::{
    cmp::max,
    collections::{HashMap, HashSet},
};

use parking_lot::RwLock;
use tokio::{sync::mpsc::UnboundedSender, time::Instant};
use tracing::{instrument, trace, warn};
use types::{
    committee::EpochId,
    digests::TransactionDigest,
    transaction::{VerifiedCertificate, VerifiedExecutableTransaction},
};

use crate::epoch_store::AuthorityPerEpochStore;

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
}

#[derive(Clone, Debug)]
pub struct PendingCertificate {
    // Certified transaction to be executed.
    pub certificate: VerifiedExecutableTransaction,
}

struct Inner {
    // Current epoch of TransactionManager.
    epoch: EpochId,

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
            pending_certificates: HashMap::with_capacity(MIN_HASHMAP_CAPACITY),
            executing_certificates: HashSet::with_capacity(MIN_HASHMAP_CAPACITY),
        }
    }

    fn maybe_reserve_capacity(&mut self) {
        self.pending_certificates.maybe_reserve_capacity();
        self.executing_certificates.maybe_reserve_capacity();
    }

    /// After reaching 1/4 load in hashmaps, decrease capacity to increase load to 1/2.
    fn maybe_shrink_capacity(&mut self) {
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
        // transaction_cache_read: Arc<dyn TransactionCacheRead>,
        epoch_store: &AuthorityPerEpochStore,
        tx_ready_certificates: UnboundedSender<PendingCertificate>,
    ) -> TransactionManager {
        let transaction_manager = TransactionManager {
            inner: RwLock::new(RwLock::new(Inner::new(epoch_store.epoch()))),
            tx_ready_certificates,
        };
        transaction_manager.enqueue(epoch_store.all_pending_execution().unwrap(), epoch_store);
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
    ) {
        let executable_txns = certs
            .into_iter()
            .map(VerifiedExecutableTransaction::new_from_certificate)
            .collect();
        self.enqueue(executable_txns, epoch_store)
    }

    #[instrument(level = "trace", skip_all)]
    pub(crate) fn enqueue(
        &self,
        certs: Vec<VerifiedExecutableTransaction>,
        epoch_store: &AuthorityPerEpochStore,
    ) {
        self.enqueue_impl(certs, epoch_store)
    }

    // #[instrument(level = "trace", skip_all)]
    // pub(crate) fn enqueue_with_expected_effects_digest(
    //     &self,
    //     certs: Vec<(VerifiedExecutableTransaction, TransactionEffectsDigest)>,
    //     epoch_store: &AuthorityPerEpochStore,
    // ) {
    //     let certs = certs
    //         .into_iter()
    //         .map(|(cert, fx)| (cert, Some(fx)))
    //         .collect();
    //     self.enqueue_impl(certs, epoch_store)
    // }

    fn enqueue_impl(
        &self,
        certs: Vec<
            VerifiedExecutableTransaction, // Option<TransactionEffectsDigest>,
        >,
        epoch_store: &AuthorityPerEpochStore,
    ) {
        let reconfig_lock = self.inner.read();

        // TODO: filter out already executed certs
        // let certs: Vec<_> = certs
        //     .into_iter()
        //     .filter(|(cert, _)| {
        //         let digest = *cert.digest();
        //         // skip already executed txes
        //         if self
        //             .transaction_cache_read
        //             .is_tx_already_executed(&digest)
        //             .unwrap_or_else(|err| {
        //                 panic!("Failed to check if tx is already executed: {:?}", err)
        //             })
        //         {
        //             false
        //         } else {
        //             true
        //         }
        //     })
        //     .collect();

        // After this point, the function cannot return early and must run to the end. Otherwise,
        // it can lead to data inconsistencies and potentially some transactions will never get
        // executed.

        // Internal lock is held only for updating the internal state.
        let mut inner = reconfig_lock.write();

        let mut pending = Vec::new();
        let pending_cert_enqueue_time = Instant::now();

        for cert in certs {
            pending.push(PendingCertificate { certificate: cert });
        }

        for mut pending_cert in pending {
            // Tx lock is not held here, which makes it possible to send duplicated transactions to
            // the execution driver after crash-recovery, when the same transaction is recovered
            // from recovery log and pending certificates table. The transaction will still only
            // execute once, because tx lock is acquired in execution driver and executed effects
            // table is consulted. So this behavior is benigh.
            let digest = *pending_cert.certificate.digest();

            // TODO: verify epoch
            // if inner.epoch != pending_cert.certificate.epoch() {
            //     warn!(
            //         "Ignoring enqueued certificate from wrong epoch. Expected={} Certificate={:?}",
            //         inner.epoch, pending_cert.certificate
            //     );
            //     continue;
            // }

            // skip already pending txes
            if inner.pending_certificates.contains_key(&digest) {
                continue;
            }
            // skip already executing txes
            if inner.executing_certificates.contains(&digest) {
                continue;
            }
            // TODO: skip already executed txes
            // let is_tx_already_executed = self
            //     .transaction_cache_read
            //     .is_tx_already_executed(&digest)
            //     .expect("Check if tx is already executed should not fail");
            // if is_tx_already_executed {
            //     continue;
            // }

            // Ready transactions can start to execute.
            // Send to execution driver for execution.
            self.certificate_ready(&mut inner, pending_cert);
            continue;
        }

        inner.maybe_reserve_capacity();
    }

    /// Notifies TransactionManager about a transaction that has been committed.
    #[instrument(level = "trace", skip_all)]
    pub(crate) fn notify_commit(
        &self,
        digest: &TransactionDigest,
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

            if !inner.executing_certificates.remove(digest) {
                trace!("{:?} not found in executing certificates, likely because it is a system transaction", digest);
                return;
            }

            inner.maybe_shrink_capacity();
        }
    }

    /// Sends the ready certificate for execution.
    fn certificate_ready(&self, inner: &mut Inner, pending_certificate: PendingCertificate) {
        trace!(tx_digest = ?pending_certificate.certificate.digest(), "certificate ready");
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
