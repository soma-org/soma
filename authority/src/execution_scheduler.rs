use crate::authority::ExecutionEnv;
use crate::cache::{ObjectCacheRead, TransactionCacheRead};
use crate::epoch_store::AuthorityPerEpochStore;
use std::collections::{BTreeSet, HashSet};
use std::time::Instant;
use std::{collections::BTreeMap, sync::Arc};
use tokio::sync::mpsc::UnboundedSender;
use tracing::instrument;
use tracing::{debug, info};
use types::base::FullObjectID;
use types::digests::TransactionDigest;
use types::object::ObjectID;
use types::storage::InputKey;
use types::transaction::{SharedInputObject, TransactionData, VerifiedExecutableTransaction};

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum SchedulingSource {
    MysticetiFastPath,
    NonFastPath,
}

#[derive(Debug)]
pub struct PendingCertificate {
    // Certified transaction to be executed.
    pub certificate: VerifiedExecutableTransaction,
    // Environment in which the transaction will be executed.
    pub execution_env: ExecutionEnv,
}

/// Utility struct for collecting barrier dependencies
pub(crate) struct BarrierDependencyBuilder {
    dep_state: BTreeMap<ObjectID, BTreeSet<TransactionDigest>>,
}

impl BarrierDependencyBuilder {
    pub fn new() -> Self {
        Self {
            dep_state: Default::default(),
        }
    }

    /// process_tx must be called for each transaction in scheduling order. If the
    /// transaction has a non-exclusive write to an object, the transaction digest is
    /// stored to become a dependency of the eventual barrier transaction. If a
    /// transaction has an exclusive write to an object, all pending non-exclusive write
    /// transactions for that object are added to the barrier dependencies.
    pub fn process_tx(
        &mut self,
        tx_digest: TransactionDigest,
        tx: &TransactionData,
    ) -> BTreeSet<TransactionDigest> {
        let mut barrier_deps = BTreeSet::new();
        for SharedInputObject { id, .. } in tx.kind().shared_input_objects() {
            // If there were preceding non-exclusive writes to this object id, this
            // transaction is a barrier and must wait for them to finish.
            if let Some(deps) = self.dep_state.remove(&id) {
                barrier_deps.extend(deps);
            }
        }
        barrier_deps
    }
}

#[derive(Clone)]
pub struct ExecutionScheduler {
    object_cache_read: Arc<dyn ObjectCacheRead>,
    transaction_cache_read: Arc<dyn TransactionCacheRead>,

    tx_ready_certificates: UnboundedSender<PendingCertificate>,
}

struct PendingGuard<'a> {
    scheduler: &'a ExecutionScheduler,
    cert: &'a VerifiedExecutableTransaction,
}

impl<'a> PendingGuard<'a> {
    pub fn new(scheduler: &'a ExecutionScheduler, cert: &'a VerifiedExecutableTransaction) -> Self {
        // scheduler
        //     .overload_tracker
        //     .add_pending_certificate(cert.data());
        Self { scheduler, cert }
    }
}

impl Drop for PendingGuard<'_> {
    fn drop(&mut self) {

        // self.scheduler
        //     .overload_tracker
        //     .remove_pending_certificate(self.cert.data());
    }
}

impl ExecutionScheduler {
    pub fn new(
        object_cache_read: Arc<dyn ObjectCacheRead>,
        transaction_cache_read: Arc<dyn TransactionCacheRead>,
        tx_ready_certificates: UnboundedSender<PendingCertificate>,
        epoch_store: &Arc<AuthorityPerEpochStore>,
    ) -> Self {
        tracing::info!("Creating new ExecutionScheduler");

        Self {
            object_cache_read,
            transaction_cache_read,

            tx_ready_certificates,
        }
    }

    #[instrument(level = "debug", skip_all, fields(tx_digest = ?cert.digest()))]
    async fn schedule_transaction(
        self,
        cert: VerifiedExecutableTransaction,
        execution_env: ExecutionEnv,
        epoch_store: &Arc<AuthorityPerEpochStore>,
    ) {
        let enqueue_time = Instant::now();
        let tx_digest = cert.digest();
        let digests = [*tx_digest];

        let tx_data = cert.transaction_data();
        let input_object_kinds = tx_data
            .input_objects()
            .expect("input_objects() cannot fail");
        let input_object_keys: Vec<_> = epoch_store
            .get_input_object_keys(
                &cert.key(),
                &input_object_kinds,
                &execution_env.assigned_versions,
            )
            .into_iter()
            .collect();

        let receiving_object_keys: HashSet<_> = tx_data
            .receiving_objects()
            .into_iter()
            .map(|entry| {
                InputKey::VersionedObject {
                    // TODO: Add support for receiving ConsensusV2 objects. For now this assumes fastpath.
                    id: FullObjectID::new(entry.0, None),
                    version: entry.1,
                }
            })
            .collect();
        let input_and_receiving_keys = [
            input_object_keys,
            receiving_object_keys.iter().cloned().collect(),
        ]
        .concat();

        let epoch = epoch_store.epoch();
        debug!(
            ?tx_digest,
            "Scheduled transaction, waiting for input objects: {:?}", input_and_receiving_keys,
        );

        let availability = self
            .object_cache_read
            .multi_input_objects_available_cache_only(&input_and_receiving_keys);
        // Most of the times, the transaction's input objects are already available.
        // We can check the availability of the input objects first, and only wait for the
        // missing input objects if necessary.
        let missing_input_keys: Vec<_> = input_and_receiving_keys
            .into_iter()
            .zip(availability)
            .filter_map(|(key, available)| if !available { Some(key) } else { None })
            .collect();
        if missing_input_keys.is_empty() {
            debug!(?tx_digest, "Input objects already available");
            self.send_transaction_for_execution(&cert, execution_env, enqueue_time);
            return;
        }

        let _pending_guard = PendingGuard::new(&self, &cert);

        if !execution_env.barrier_dependencies.is_empty() {
            debug!(
                "waiting for barrier dependencies to be executed: {:?}",
                execution_env.barrier_dependencies
            );
            self.transaction_cache_read
                .notify_read_executed_effects_digests(&execution_env.barrier_dependencies)
                .await;
        }

        tokio::select! {
            _ = self.object_cache_read
                .notify_read_input_objects(&missing_input_keys, &receiving_object_keys, epoch)
                => {

                    debug!(?tx_digest, "Input objects available");
                    // TODO: Eventually we could fold execution_driver into the scheduler.
                    self.send_transaction_for_execution(
                        &cert,
                        execution_env,
                        enqueue_time,
                    );
                }
            _ = self.transaction_cache_read.notify_read_executed_effects_digests(
                &digests,
            ) => {
                debug!(?tx_digest, "Transaction already executed");
            }
        };
    }

    fn send_transaction_for_execution(
        &self,
        cert: &VerifiedExecutableTransaction,
        execution_env: ExecutionEnv,
        enqueue_time: Instant,
    ) {
        let pending_cert = PendingCertificate {
            certificate: cert.clone(),
            execution_env,
        };
        let _ = self.tx_ready_certificates.send(pending_cert);
    }

    /// When we schedule a certificate, it should be impossible for it to have been executed in a
    /// previous epoch.
    #[cfg(debug_assertions)]
    fn assert_cert_not_executed_previous_epochs(&self, cert: &VerifiedExecutableTransaction) {
        let epoch = cert.epoch();
        let digest = *cert.digest();
        let digests = [digest];
        let executed = self
            .transaction_cache_read
            .multi_get_executed_effects(&digests)
            .pop()
            .unwrap();
        // Due to pruning, we may not always have an executed effects for the certificate
        // even if it was executed. So this is a best-effort check.
        if let Some(executed) = executed {
            use types::effects::TransactionEffectsAPI;

            assert_eq!(
                executed.executed_epoch(),
                epoch,
                "Transaction {} was executed in epoch {}, but scheduled again in epoch {}",
                digest,
                executed.executed_epoch(),
                epoch
            );
        }
    }
}

impl ExecutionScheduler {
    pub fn enqueue(
        &self,
        certs: Vec<(Schedulable, ExecutionEnv)>,
        epoch_store: &Arc<AuthorityPerEpochStore>,
    ) {
        // schedule all transactions immediately
        let mut ordinary_txns = Vec::with_capacity(certs.len());

        for (schedulable, env) in certs {
            match schedulable {
                Schedulable::Transaction(tx) => {
                    ordinary_txns.push((tx, env));
                }
            }
        }

        self.enqueue_transactions(ordinary_txns, epoch_store);
    }

    pub fn enqueue_transactions(
        &self,
        certs: Vec<(VerifiedExecutableTransaction, ExecutionEnv)>,
        epoch_store: &Arc<AuthorityPerEpochStore>,
    ) {
        // Filter out certificates from wrong epoch.
        let certs: Vec<_> = certs
            .into_iter()
            .filter_map(|cert| {
                if cert.0.epoch() == epoch_store.epoch() {
                    #[cfg(debug_assertions)]
                    self.assert_cert_not_executed_previous_epochs(&cert.0);

                    Some(cert)
                } else {
                    debug!(
                        "We should never enqueue certificate from wrong epoch. Expected={} Certificate={:?}",
                        epoch_store.epoch(),
                        cert.0.epoch()
                    );
                    None
                }
            })
            .collect();
        let digests: Vec<_> = certs.iter().map(|(cert, _)| *cert.digest()).collect();
        let executed = self
            .transaction_cache_read
            .multi_get_executed_effects_digests(&digests);
        let mut already_executed_certs_num = 0;
        let pending_certs =
            certs
                .into_iter()
                .zip(executed)
                .filter_map(|((cert, execution_env), executed)| {
                    if executed.is_none() {
                        Some((cert, execution_env))
                    } else {
                        already_executed_certs_num += 1;
                        None
                    }
                });

        for (cert, execution_env) in pending_certs {
            let scheduler = self.clone();
            let epoch_store = epoch_store.clone();
            tokio::spawn(
                epoch_store.within_alive_epoch(scheduler.schedule_transaction(
                    cert,
                    execution_env,
                    &epoch_store,
                )),
            );
        }
    }
}
