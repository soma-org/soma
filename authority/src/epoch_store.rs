use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque},
    future::Future,
    ops::Deref,
    path::PathBuf,
    sync::Arc,
    time::Instant,
};

use arc_swap::ArcSwapOption;
use futures::{
    future::select,
    future::{join_all, Either},
    FutureExt,
};
use parking_lot::{Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard};
use tracing::{debug, info, instrument, warn};
use types::{
    accumulator::{Accumulator, CommitIndex},
    base::{AuthorityName, ConciseableName, Round, SomaAddress},
    committee::{Committee, EpochId},
    consensus::{
        ConsensusCommitPrologue, ConsensusTransaction, ConsensusTransactionKey,
        ConsensusTransactionKind,
    },
    crypto::{AuthoritySignInfo, AuthorityStrongQuorumSignInfo, Signer},
    digests::{TransactionDigest, TransactionEffectsDigest},
    effects::{self, ExecutionFailureStatus, ExecutionStatus, TransactionEffects},
    envelope::TrustedEnvelope,
    error::{ExecutionError, SomaError, SomaResult},
    execution_indices::ExecutionIndices,
    mutex_table::{MutexGuard, MutexTable},
    object::{Object, ObjectData, ObjectID, ObjectRef, ObjectType, Version},
    protocol::{Chain, ProtocolConfig},
    storage::object_store::ObjectStore,
    system_state::{
        self, get_system_state, EpochStartSystemState, EpochStartSystemStateTrait, SystemState,
    },
    temporary_store::TemporaryStore,
    transaction::{
        self, CertifiedTransaction, EndOfEpochTransactionKind, SenderSignedData,
        StateTransactionKind, Transaction, TransactionKey, TransactionKind,
        TrustedExecutableTransaction, VerifiedCertificate, VerifiedExecutableTransaction,
        VerifiedSignedTransaction, VerifiedTransaction,
    },
    tx_outputs::WrittenObjects,
    SYSTEM_STATE_OBJECT_ID,
};
use utils::{notify_once::NotifyOnce, notify_read::NotifyRead};

use crate::{
    handler::{
        ConsensusCommitInfo, SequencedConsensusTransaction, SequencedConsensusTransactionKey,
        SequencedConsensusTransactionKind, VerifiedSequencedConsensusTransaction,
    },
    reconfiguration::ReconfigState,
    signature_verifier::SignatureVerifier,
    stake_aggregator::StakeAggregator,
    start_epoch::{EpochStartConfigTrait, EpochStartConfiguration},
    store::LockDetails,
};

/// The key where the latest consensus index is stored in the database.
// TODO: Make a single table (e.g., called `variables`) storing all our lonely variables in one place.
const LAST_CONSENSUS_STATS_ADDR: u64 = 0;
const RECONFIG_STATE_INDEX: u64 = 0;

// CertLockGuard and CertTxGuard are functionally identical right now, but we retain a distinction
// anyway. If we need to support distributed object storage, having this distinction will be
// useful, as we will most likely have to re-implement a retry / write-ahead-log at that point.
pub struct CertLockGuard(#[allow(unused)] MutexGuard);
pub struct CertTxGuard(#[allow(unused)] CertLockGuard);

impl CertTxGuard {
    pub fn release(self) {}
    pub fn commit_tx(self) {}
}

pub enum CancelConsensusCertificateReason {}

pub enum ConsensusCertificateResult {
    /// The consensus message was ignored (e.g. because it has already been processed).
    Ignored,
    /// An executable transaction (can be a user tx or a system tx)
    SomaTransaction(VerifiedExecutableTransaction),
    /// The transaction should be re-processed at a future commit, specified by the DeferralKey
    // Deferred(DeferralKey),
    /// A message was processed which updates randomness state.
    // RandomnessConsensusMessage,
    /// Everything else, e.g. AuthorityCapabilities, CheckpointSignatures, etc.
    ConsensusMessage,
    /// A system message in consensus was ignored (e.g. because of end of epoch).
    IgnoredSystem,
    /// A will-be-cancelled transaction. It'll still go through execution engine (but not be executed),
    /// unlock any owned objects, and return corresponding cancellation error according to
    /// `CancelConsensusCertificateReason`.
    Cancelled(
        (
            VerifiedExecutableTransaction,
            CancelConsensusCertificateReason,
        ),
    ),
}

pub struct AuthorityPerEpochStore {
    /// The name of this authority.
    pub(crate) name: AuthorityName,

    /// Committee of validators for the current epoch.
    committee: Arc<Committee>,

    /// Holds the underlying per-epoch typed store tables.
    /// This is an ArcSwapOption because it needs to be used concurrently,
    /// and it needs to be cleared at the end of the epoch.
    tables: ArcSwapOption<AuthorityEpochTables>,

    protocol_config: ProtocolConfig,

    // db_options: Option<Options>,
    //
    /// In-memory cache of the content from the reconfig_state db table.
    reconfig_state_mem: RwLock<ReconfigState>,
    consensus_notify_read: NotifyRead<SequencedConsensusTransactionKey, ()>,

    /// Batch verifier for certificates - also caches certificates and tx sigs that are known to have
    /// valid signatures. Lives in per-epoch store because the caching/batching is only valid
    /// within for certs within the current epoch.
    pub(crate) signature_verifier: SignatureVerifier,

    // pub(crate) checkpoint_state_notify_read: NotifyRead<CheckpointSequenceNumber, Accumulator>,
    running_root_notify_read: NotifyRead<CommitIndex, Accumulator>,
    executed_digests_notify_read: NotifyRead<TransactionKey, TransactionDigest>,

    /// This is used to notify all epoch specific tasks that epoch has ended.
    epoch_alive_notify: NotifyOnce,

    /// Used to notify all epoch specific tasks that user certs are closed.
    user_certs_closed_notify: NotifyOnce,

    /// This lock acts as a barrier for tasks that should not be executed in parallel with reconfiguration
    /// See comments in AuthorityPerEpochStore::epoch_terminated() on how this is used
    /// Crash recovery note: we write next epoch in the database first, and then use this lock to
    /// wait for in-memory tasks for the epoch to finish. If node crashes at this stage validator
    /// will start with the new epoch(and will open instance of per-epoch store for a new epoch).
    epoch_alive: tokio::sync::RwLock<bool>,
    end_of_publish: Mutex<StakeAggregator<(), true>>,
    /// Pending certificates that are waiting to be sequenced by the consensus.
    /// This is an in-memory 'index' of a AuthorityPerEpochTables::pending_consensus_transactions.
    /// We need to keep track of those in order to know when to send EndOfPublish message.
    /// Lock ordering: this is a 'leaf' lock, no other locks should be acquired in the scope of this lock
    /// In particular, this lock is always acquired after taking read or write lock on reconfig state
    pending_consensus_certificates: Mutex<HashSet<TransactionDigest>>,

    /// MutexTable for transaction locks (prevent concurrent execution of same transaction)
    mutex_table: MutexTable<TransactionDigest>,

    /// The moment when the current epoch started locally on this validator. Note that this
    /// value could be skewed if the node crashed and restarted in the middle of the epoch. That's
    /// ok because this is used for metric purposes and we could tolerate some skews occasionally.
    pub(crate) epoch_open_time: Instant,

    /// The moment when epoch is closed. We don't care much about crash recovery because it's
    /// a metric that doesn't have to be available for each epoch, and it's only used during
    /// the last few seconds of an epoch.
    epoch_close_time: RwLock<Option<Instant>>,
    epoch_start_configuration: Arc<EpochStartConfiguration>,
    // executed_in_epoch_table_enabled: once_cell::sync::OnceCell<bool>,

    // chain_identifier: ChainIdentifier,
}

/// AuthorityEpochTables contains tables that contain data that is only valid within an epoch.
#[derive(Default)]
pub struct AuthorityEpochTables {
    /// Certificates that have been received from clients or received from consensus, but not yet
    /// executed. Entries are cleared after execution.
    /// This table is critical for crash recovery, because usually the consensus output progress
    /// is updated after a certificate is committed into this table.
    ///
    /// In theory, this table may be superseded by storing consensus and checkpoint execution
    /// progress. But it is more complex, because it would be necessary to track inflight
    /// executions not ordered by indices. For now, tracking inflight certificates as a map
    /// seems easier.
    pub(crate) pending_execution: RwLock<BTreeMap<TransactionDigest, TrustedExecutableTransaction>>,

    /// Track which transactions have been processed in handle_consensus_transaction. We must be
    /// sure to advance next_shared_object_versions exactly once for each transaction we receive from
    /// consensus. But, we may also be processing transactions from checkpoints, so we need to
    /// track this state separately.
    ///
    /// Entries in this table can be garbage collected whenever we can prove that we won't receive
    /// another handle_consensus_transaction call for the given digest. This probably means at
    /// epoch change.
    consensus_message_processed: RwLock<BTreeMap<SequencedConsensusTransactionKey, bool>>,

    /// Map stores pending transactions that this authority submitted to consensus
    pending_consensus_transactions: RwLock<BTreeMap<ConsensusTransactionKey, ConsensusTransaction>>,

    /// The following table is used to store a single value (the corresponding key is a constant). The value
    /// represents the index of the latest consensus message this authority processed, running hash of
    /// transactions, and accumulated stats of consensus output.
    /// This field is written by a single process (consensus handler).
    last_consensus_stats: RwLock<BTreeMap<u64, ExecutionIndices>>,

    /// This table contains current reconfiguration state for validator for current epoch
    reconfig_state: RwLock<BTreeMap<u64, ReconfigState>>,

    /// Validators that have sent EndOfPublish message in this epoch
    end_of_publish: RwLock<BTreeMap<AuthorityName, ()>>,

    /// Signatures over transaction effects that we have signed and returned to users.
    /// We store this to avoid re-signing the same effects twice.
    /// Note that this may contain signatures for effects from previous epochs, in the case
    /// that a user requests a signature for effects from a previous epoch. However, the
    /// signature is still epoch-specific and so is stored in the epoch store.
    effects_signatures: RwLock<BTreeMap<TransactionDigest, AuthoritySignInfo>>,

    /// When we sign a TransactionEffects, we must record the digest of the effects in order
    /// to detect and prevent equivocation when re-executing a transaction that may not have been
    /// committed to disk.
    /// Entries are removed from this table after the transaction in question has been committed
    /// to disk.
    signed_effects_digests: RwLock<BTreeMap<TransactionDigest, TransactionEffectsDigest>>,

    /// Signatures of transaction certificates that are executed locally.
    transaction_cert_signatures: RwLock<BTreeMap<TransactionDigest, AuthorityStrongQuorumSignInfo>>,

    /// This is map between the transaction digest and transactions found in the `transaction_lock`.
    signed_transactions:
        RwLock<BTreeMap<TransactionDigest, TrustedEnvelope<SenderSignedData, AuthoritySignInfo>>>,

    /// Map from ObjectRef to transaction locking that object
    object_locked_transactions: RwLock<BTreeMap<ObjectRef, TransactionDigest>>,

    // Maps commit index to an accumulator with accumulated state
    // only for the checkpoint that the key references. Append-only, i.e.,
    // the accumulator is complete wrt the commit
    pub state_hash_by_commit: RwLock<BTreeMap<CommitIndex, Accumulator>>,

    /// Maps commit index to the running (non-finalized) root state
    /// accumulator up to that commit. Guaranteed to be written to in commit
    /// index order.
    pub running_root_accumulators: RwLock<BTreeMap<CommitIndex, Accumulator>>,
}

impl AuthorityEpochTables {
    fn load_reconfig_state(&self) -> SomaResult<ReconfigState> {
        let state = self
            .reconfig_state
            .read()
            .get(&RECONFIG_STATE_INDEX)
            .cloned()
            .unwrap_or_default();
        Ok(state)
    }

    pub fn get_all_pending_consensus_transactions(&self) -> Vec<ConsensusTransaction> {
        self.pending_consensus_transactions
            .read()
            .values()
            .cloned()
            .collect()
    }

    pub fn get_last_consensus_stats(&self) -> SomaResult<Option<ExecutionIndices>> {
        Ok(self
            .last_consensus_stats
            .read()
            .get(&LAST_CONSENSUS_STATS_ADDR)
            .copied())
    }

    pub fn get_locked_transaction(
        &self,
        obj_ref: &ObjectRef,
    ) -> SomaResult<Option<TransactionDigest>> {
        Ok(self.object_locked_transactions.read().get(obj_ref).cloned())
    }

    pub fn multi_get_locked_transactions(
        &self,
        owned_input_objects: &[ObjectRef],
    ) -> SomaResult<Vec<Option<TransactionDigest>>> {
        let locks = self.object_locked_transactions.read();
        let mut results = Vec::with_capacity(owned_input_objects.len());

        for obj_ref in owned_input_objects {
            results.push(locks.get(obj_ref).cloned());
        }

        Ok(results)
    }

    pub fn write_transaction_locks(
        &self,
        transaction: VerifiedSignedTransaction,
        locks_to_write: impl Iterator<Item = (ObjectRef, TransactionDigest)>,
    ) -> SomaResult {
        // Insert locks
        {
            let mut locked_transactions = self.object_locked_transactions.write();
            for (obj_ref, lock) in locks_to_write {
                locked_transactions.insert(obj_ref, lock);
            }
        }

        // Insert transaction
        {
            let mut transactions = self.signed_transactions.write();
            transactions.insert(
                *transaction.digest(),
                transaction.serializable_ref().clone(),
            );
        }

        Ok(())
    }
}

pub(crate) const MUTEX_TABLE_SIZE: usize = 1024;

impl AuthorityPerEpochStore {
    #[instrument(name = "AuthorityPerEpochStore::new", level = "error", skip_all, fields(epoch = committee.epoch))]
    pub fn new(
        name: AuthorityName,
        committee: Arc<Committee>,
        epoch_start_configuration: EpochStartConfiguration,
    ) -> Arc<Self> {
        let current_time = Instant::now();
        let epoch_id = committee.epoch;

        let tables = AuthorityEpochTables::default();
        let end_of_publish = StakeAggregator::from_iter(
            committee.clone(),
            tables.end_of_publish.read().iter().map(|(k, _)| (*k, ())),
        );
        let reconfig_state = tables
            .load_reconfig_state()
            .expect("Load reconfig state at initialization cannot fail");

        let epoch_alive_notify = NotifyOnce::new();
        let pending_consensus_transactions = tables.get_all_pending_consensus_transactions();
        let pending_consensus_certificates: HashSet<_> = pending_consensus_transactions
            .iter()
            .filter_map(|transaction| {
                if let ConsensusTransactionKind::UserTransaction(certificate) = &transaction.kind {
                    Some(*certificate.digest())
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(
            epoch_start_configuration.epoch_start_state().epoch(),
            epoch_id
        );
        let epoch_start_configuration = Arc::new(epoch_start_configuration);

        let protocol_config = ProtocolConfig::default();

        let signature_verifier = SignatureVerifier::new(committee.clone());

        let s = Arc::new(Self {
            name,
            committee,
            protocol_config,
            tables: ArcSwapOption::new(Some(Arc::new(tables))),

            reconfig_state_mem: RwLock::new(reconfig_state),
            epoch_alive_notify,
            user_certs_closed_notify: NotifyOnce::new(),
            epoch_alive: tokio::sync::RwLock::new(true),
            consensus_notify_read: NotifyRead::new(),
            signature_verifier,
            executed_digests_notify_read: NotifyRead::new(),
            running_root_notify_read: NotifyRead::new(),
            end_of_publish: Mutex::new(end_of_publish),
            pending_consensus_certificates: Mutex::new(pending_consensus_certificates),
            mutex_table: MutexTable::new(MUTEX_TABLE_SIZE),
            epoch_open_time: current_time,
            epoch_close_time: Default::default(),
            epoch_start_configuration,
        });
        s
    }

    pub fn tables(&self) -> SomaResult<Arc<AuthorityEpochTables>> {
        match self.tables.load_full() {
            Some(tables) => Ok(tables),
            None => Err(SomaError::EpochEnded(self.epoch())),
        }
    }

    // Ideally the epoch tables handle should have the same lifetime as the outer AuthorityPerEpochStore,
    // and this function should be unnecessary. But unfortunately, Arc<AuthorityPerEpochStore> outlives the
    // epoch significantly right now, so we need to manually release the tables to release its memory usage.
    pub fn release_db_handles(&self) {
        // When the logic to release DB handles becomes obsolete, it may still be useful
        // to make sure AuthorityEpochTables is not used after the next epoch starts.
        self.tables.store(None);
    }

    /// Returns `&Arc<EpochStartConfiguration>`
    /// User can treat this `Arc` as `&EpochStartConfiguration`, or clone the Arc to pass as owned object
    pub fn epoch_start_config(&self) -> &Arc<EpochStartConfiguration> {
        &self.epoch_start_configuration
    }

    pub fn epoch_start_state(&self) -> &EpochStartSystemState {
        self.epoch_start_configuration.epoch_start_state()
    }

    pub fn new_at_next_epoch(
        &self,
        name: AuthorityName,
        new_committee: Committee,
        epoch_start_configuration: EpochStartConfiguration,
    ) -> Arc<Self> {
        assert_eq!(self.epoch() + 1, new_committee.epoch);
        Self::new(name, Arc::new(new_committee), epoch_start_configuration)
    }

    pub fn new_at_next_epoch_for_testing(&self) -> Arc<Self> {
        let next_epoch = self.epoch() + 1;
        let next_committee = Committee::new(
            next_epoch,
            self.committee.voting_rights.iter().cloned().collect(),
        );

        let epoch_start_configuration = self.epoch_start_configuration.as_ref().clone();
        self.new_at_next_epoch(
            self.name,
            next_committee,
            epoch_start_configuration.new_at_next_epoch_for_testing(),
        )
    }

    pub fn committee(&self) -> &Arc<Committee> {
        &self.committee
    }

    pub fn protocol_config(&self) -> &ProtocolConfig {
        &self.protocol_config
    }

    pub fn epoch(&self) -> EpochId {
        self.committee.epoch
    }

    pub fn get_state_hash_for_commit(
        &self,
        commit: &CommitIndex,
    ) -> SomaResult<Option<Accumulator>> {
        Ok(self
            .tables()?
            .state_hash_by_commit
            .read()
            .get(commit)
            .cloned())
    }

    pub fn insert_state_hash_for_commit(
        &self,
        commit: &CommitIndex,
        accumulator: &Accumulator,
    ) -> SomaResult {
        self.tables()?
            .state_hash_by_commit
            .write()
            .insert(*commit, accumulator.clone());
        Ok(())
    }

    pub fn get_running_root_accumulator(
        &self,
        commit: &CommitIndex,
    ) -> SomaResult<Option<Accumulator>> {
        Ok(self
            .tables()?
            .running_root_accumulators
            .read()
            .get(commit)
            .cloned())
    }

    pub fn get_highest_running_root_accumulator(
        &self,
    ) -> SomaResult<Option<(CommitIndex, Accumulator)>> {
        Ok(self
            .tables()?
            .running_root_accumulators
            .read()
            .iter()
            .next_back()
            .map(|(k, v)| (*k, v.clone())))
    }

    pub fn insert_running_root_accumulator(
        &self,
        checkpoint: &CommitIndex,
        acc: &Accumulator,
    ) -> SomaResult {
        self.tables()?
            .running_root_accumulators
            .write()
            .insert(*checkpoint, acc.clone());
        self.running_root_notify_read.notify(checkpoint, acc);

        Ok(())
    }

    pub async fn notify_read_running_root(&self, commit: CommitIndex) -> SomaResult<Accumulator> {
        let registration = self.running_root_notify_read.register_one(&commit);
        let acc = self
            .tables()?
            .running_root_accumulators
            .read()
            .get(&commit)
            .cloned();

        let result = match acc {
            Some(ready) => Either::Left(futures::future::ready(ready)),
            None => Either::Right(registration),
        }
        .await;

        Ok(result)
    }

    /// When submitting a certificate caller **must** provide a ReconfigState lock guard
    /// and verify that it allows new user certificates
    pub fn insert_pending_consensus_transactions(
        &self,
        transactions: &[ConsensusTransaction],
        lock: Option<&RwLockReadGuard<ReconfigState>>,
    ) -> SomaResult {
        let tables = self.tables()?;

        {
            let mut pending_transactions = tables.pending_consensus_transactions.write();
            pending_transactions.extend(transactions.iter().map(|tx| (tx.key(), tx.clone())));
        }

        // TODO: lock once for all insert() calls.
        for transaction in transactions {
            if let ConsensusTransactionKind::UserTransaction(cert) = &transaction.kind {
                let state = lock.expect("Must pass reconfiguration lock when storing certificate");
                // Caller is responsible for performing graceful check
                assert!(
                    state.should_accept_user_certs(),
                    "Reconfiguration state should allow accepting user transactions"
                );
                self.pending_consensus_certificates
                    .lock()
                    .insert(*cert.digest());
            }
        }
        Ok(())
    }

    pub fn remove_pending_consensus_transactions(
        &self,
        keys: &[ConsensusTransactionKey],
    ) -> SomaResult {
        let tables = self.tables()?;

        {
            let mut pending_transactions = tables.pending_consensus_transactions.write();
            let keys_set: HashSet<_> = keys.iter().collect();
            pending_transactions.retain(|k, _| !keys_set.contains(k));
        }
        // TODO: lock once for all remove() calls.
        for key in keys {
            if let ConsensusTransactionKey::Certificate(cert) = key {
                self.pending_consensus_certificates.lock().remove(cert);
            }
        }
        Ok(())
    }

    pub fn pending_consensus_certificates_count(&self) -> usize {
        self.pending_consensus_certificates.lock().len()
    }

    pub fn pending_consensus_certificates_empty(&self) -> bool {
        self.pending_consensus_certificates.lock().is_empty()
    }

    pub fn pending_consensus_certificates(&self) -> HashSet<TransactionDigest> {
        self.pending_consensus_certificates.lock().clone()
    }

    /// Check whether certificate was processed by consensus.
    /// For shared lock certificates, if this function returns true means shared locks for this certificate are set
    pub fn is_tx_cert_consensus_message_processed(
        &self,
        certificate: &CertifiedTransaction,
    ) -> SomaResult<bool> {
        self.is_consensus_message_processed(&SequencedConsensusTransactionKey::External(
            ConsensusTransactionKey::Certificate(*certificate.digest()),
        ))
    }

    /// Check whether any certificates were processed by consensus.
    /// This handles multiple certificates at once.
    pub fn is_any_tx_certs_consensus_message_processed<'a>(
        &self,
        certificates: impl Iterator<Item = &'a CertifiedTransaction>,
    ) -> SomaResult<bool> {
        let keys = certificates.map(|cert| {
            SequencedConsensusTransactionKey::External(ConsensusTransactionKey::Certificate(
                *cert.digest(),
            ))
        });
        Ok(self
            .check_consensus_messages_processed(keys)?
            .into_iter()
            .any(|processed| processed))
    }

    /// Check whether any certificates were processed by consensus.
    /// This handles multiple certificates at once.
    pub fn is_all_tx_certs_consensus_message_processed<'a>(
        &self,
        certificates: impl Iterator<Item = &'a VerifiedCertificate>,
    ) -> SomaResult<bool> {
        let keys = certificates.map(|cert| {
            SequencedConsensusTransactionKey::External(ConsensusTransactionKey::Certificate(
                *cert.digest(),
            ))
        });
        Ok(self
            .check_consensus_messages_processed(keys)?
            .into_iter()
            .all(|processed| processed))
    }

    pub fn is_consensus_message_processed(
        &self,
        key: &SequencedConsensusTransactionKey,
    ) -> SomaResult<bool> {
        Ok(self
            .tables()?
            .consensus_message_processed
            .read()
            .contains_key(key))
    }

    pub fn check_consensus_messages_processed(
        &self,
        keys: impl Iterator<Item = SequencedConsensusTransactionKey>,
    ) -> SomaResult<Vec<bool>> {
        let tables = self.tables()?;
        let consensus_message_processed = tables.consensus_message_processed.read(); // Assuming you're using RwLock

        Ok(keys
            .map(|key| consensus_message_processed.contains_key(&key))
            .collect())
    }

    pub async fn consensus_messages_processed_notify(
        &self,
        keys: Vec<SequencedConsensusTransactionKey>,
    ) -> Result<(), SomaError> {
        let registrations = self.consensus_notify_read.register_all(&keys);

        let unprocessed_keys_registrations = registrations
            .into_iter()
            .zip(self.check_consensus_messages_processed(keys.into_iter())?)
            .filter(|(_, processed)| !processed)
            .map(|(registration, _)| registration);

        join_all(unprocessed_keys_registrations).await;
        Ok(())
    }

    pub fn has_sent_end_of_publish(&self, authority: &AuthorityName) -> SomaResult<bool> {
        Ok(self
            .end_of_publish
            .try_lock()
            .expect("No contention on end_of_publish lock")
            .contains_key(authority))
    }

    pub fn get_reconfig_state_read_lock_guard(&self) -> RwLockReadGuard<ReconfigState> {
        self.reconfig_state_mem.read()
    }

    pub fn get_reconfig_state_write_lock_guard(&self) -> RwLockWriteGuard<ReconfigState> {
        self.reconfig_state_mem.write()
    }

    pub fn get_all_pending_consensus_transactions(&self) -> Vec<ConsensusTransaction> {
        self.tables()
            .expect("recovery should not cross epoch boundary")
            .get_all_pending_consensus_transactions()
    }

    pub fn close_user_certs(&self, mut lock_guard: RwLockWriteGuard<'_, ReconfigState>) {
        lock_guard.close_user_certs();
        self.store_reconfig_state(&lock_guard)
            .expect("Updating reconfig state cannot fail");

        // Set epoch_close_time for metric purpose.
        let mut epoch_close_time = self.epoch_close_time.write();
        if epoch_close_time.is_none() {
            // Only update it the first time epoch is closed.
            *epoch_close_time = Some(Instant::now());

            self.user_certs_closed_notify
                .notify()
                .expect("user_certs_closed_notify called twice on same epoch store");
        }
    }

    pub async fn user_certs_closed_notify(&self) {
        self.user_certs_closed_notify.wait().await
    }

    /// Notify epoch is terminated, can only be called once on epoch store
    pub async fn epoch_terminated(&self) {
        // Notify interested tasks that epoch has ended
        self.epoch_alive_notify
            .notify()
            .expect("epoch_terminated called twice on same epoch store");
        // This `write` acts as a barrier - it waits for futures executing in
        // `within_alive_epoch` to terminate before we can continue here
        debug!("Epoch terminated - waiting for pending tasks to complete");
        *self.epoch_alive.write().await = false;
        debug!("All pending epoch tasks completed");
    }

    /// Waits for the notification about epoch termination
    pub async fn wait_epoch_terminated(&self) {
        self.epoch_alive_notify.wait().await
    }

    /// This function executes given future until epoch_terminated is called
    /// If future finishes before epoch_terminated is called, future result is returned
    /// If epoch_terminated is called before future is resolved, error is returned
    ///
    /// In addition to the early termination guarantee, this function also prevents epoch_terminated()
    /// if future is being executed.
    #[allow(clippy::result_unit_err)]
    pub async fn within_alive_epoch<F: Future + Send>(&self, f: F) -> Result<F::Output, ()> {
        // This guard is kept in the future until it resolves, preventing `epoch_terminated` to
        // acquire a write lock
        let guard = self.epoch_alive.read().await;
        if !*guard {
            return Err(());
        }
        let terminated = self.wait_epoch_terminated().boxed();
        let f = f.boxed();
        match select(terminated, f).await {
            Either::Left((_, _f)) => Err(()),
            Either::Right((result, _)) => Ok(result),
        }
    }

    #[instrument(level = "debug", skip_all)]
    pub(crate) async fn process_consensus_transactions_and_commit_boundary<'a>(
        &self,
        transactions: Vec<SequencedConsensusTransaction>,
        consensus_stats: &ExecutionIndices,
        // checkpoint_service: &Arc<C>,
        // cache_reader: &dyn ObjectCacheRead,
        consensus_commit_info: &ConsensusCommitInfo,
        // authority_metrics: &Arc<AuthorityMetrics>,
    ) -> SomaResult<Vec<VerifiedExecutableTransaction>> {
        // Split transactions into different types for processing.
        let verified_transactions: Vec<_> = transactions
            .into_iter()
            .filter_map(|transaction| self.verify_consensus_transaction(transaction))
            .collect();
        let mut system_transactions = Vec::with_capacity(verified_transactions.len());
        let mut current_commit_sequenced_consensus_transactions =
            Vec::with_capacity(verified_transactions.len());

        let mut end_of_publish_transactions = Vec::with_capacity(verified_transactions.len());
        for tx in verified_transactions {
            if tx.0.is_end_of_publish() {
                end_of_publish_transactions.push(tx);
            } else if tx.0.is_system() {
                system_transactions.push(tx);
            } else {
                current_commit_sequenced_consensus_transactions.push(tx);
            }
        }

        let mut output = ConsensusCommitOutput::new();

        // Sequenced_transactions and sequenced_randomness_transactions store all transactions that will be sent to
        // process_consensus_transactions. We put deferred transactions at the beginning of the list before
        // PostConsensusTxReorder::reorder, so that for transactions with the same gas price, deferred transactions
        // will be placed earlier in the execution queue.
        let mut sequenced_transactions: Vec<VerifiedSequencedConsensusTransaction> =
            Vec::with_capacity(
                current_commit_sequenced_consensus_transactions.len(), // + previously_deferred_tx_digests.len(),
            );

        sequenced_transactions.extend(current_commit_sequenced_consensus_transactions);

        // TODO: reorder transactions by gas price

        let consensus_transactions: Vec<_> = system_transactions
            .into_iter()
            .chain(sequenced_transactions)
            .collect();

        let (
            transactions_to_schedule,
            notifications,
            lock,
            final_round,
            consensus_commit_prologue_root,
        ) = self
            .process_consensus_transactions(
                &mut output,
                &consensus_transactions,
                &end_of_publish_transactions,
                consensus_commit_info,
            )
            .await?;
        self.finish_consensus_certificate_process_with_batch(
            &mut output,
            &transactions_to_schedule,
        )?;
        output.record_consensus_commit_stats(consensus_stats.clone());

        // Create pending checkpoints if we are still accepting tx.
        let should_accept_tx = if let Some(lock) = &lock {
            lock.should_accept_tx()
        } else {
            // It is ok to just release lock here as functions called by this one are the
            // only place that transition reconfig state, and this function itself is always
            // executed from consensus task. At this point if the lock was not already provided
            // above, we know we won't be transitioning state for this commit.
            self.get_reconfig_state_read_lock_guard().should_accept_tx()
        };
        let make_checkpoint = should_accept_tx || final_round;
        if make_checkpoint {
            // TODO: Generate pending checkpoint for regular user tx.
        }

        output.write_to_batch(self)?;

        // Only after batch is written, notify checkpoint service to start building any new
        // pending checkpoints.
        if make_checkpoint {
            // TODO: Notify checkpoint service
        }

        self.process_notifications(&notifications, &end_of_publish_transactions);

        if final_round {
            info!(
                epoch=?self.epoch(),
                // Accessing lock on purpose so that the compiler ensures
                // the lock is not yet dropped.
                lock=?lock.as_ref(),
                final_round=?final_round,
                "Notified last checkpoint"
            );
        }

        Ok(transactions_to_schedule)
    }

    // Adds the consensus commit prologue transaction to the beginning of input `transactions` to update
    // the system clock used in all transactions in the current consensus commit.
    // Returns the root of the consensus commit prologue transaction if it was added to the input.
    fn add_consensus_commit_prologue_transaction(
        &self,
        output: &mut ConsensusCommitOutput,
        transactions: &mut VecDeque<VerifiedExecutableTransaction>,
        consensus_commit_info: &ConsensusCommitInfo,
        cancelled_txns: &BTreeMap<TransactionDigest, CancelConsensusCertificateReason>,
    ) -> SomaResult<Option<TransactionKey>> {
        #[cfg(any(test, feature = "test-utils"))]
        {
            if consensus_commit_info.skip_consensus_commit_prologue_in_test() {
                return Ok(None);
            }
        }

        let transaction =
            consensus_commit_info.create_consensus_commit_prologue_transaction(self.epoch());
        let consensus_commit_prologue_root = match self.process_consensus_system_transaction(&transaction) {
            ConsensusCertificateResult::SomaTransaction(processed_tx) => {
                transactions.push_front(processed_tx.clone());
                Some(processed_tx.key())
            }
            ConsensusCertificateResult::IgnoredSystem => None,
            _ => unreachable!("process_consensus_system_transaction returned unexpected ConsensusCertificateResult."),
        };

        output.record_consensus_message_processed(SequencedConsensusTransactionKey::System(
            *transaction.digest(),
        ));
        Ok(consensus_commit_prologue_root)
    }

    /// Depending on the type of the VerifiedSequencedConsensusTransaction wrappers,
    /// - Verify and initialize the state to execute the certificates.
    ///   Return VerifiedCertificates for each executable certificate
    /// - Or update the state for checkpoint or epoch change protocol.
    #[instrument(level = "debug", skip_all)]
    #[allow(clippy::type_complexity)]
    pub(crate) async fn process_consensus_transactions(
        &self,
        output: &mut ConsensusCommitOutput,
        transactions: &[VerifiedSequencedConsensusTransaction],
        end_of_publish_transactions: &[VerifiedSequencedConsensusTransaction],
        consensus_commit_info: &ConsensusCommitInfo,
    ) -> SomaResult<(
        Vec<VerifiedExecutableTransaction>,    // transactions to schedule
        Vec<SequencedConsensusTransactionKey>, // keys to notify as complete
        Option<RwLockWriteGuard<ReconfigState>>,
        bool,                   // true if final round
        Option<TransactionKey>, // consensus commit prologue root
    )> {
        let mut verified_certificates = VecDeque::with_capacity(transactions.len() + 1);
        let mut notifications = Vec::with_capacity(transactions.len());

        let mut cancelled_txns: BTreeMap<TransactionDigest, CancelConsensusCertificateReason> =
            BTreeMap::new();

        for tx in transactions {
            let key = tx.0.transaction.key();
            let mut ignored = false;

            match self
                .process_consensus_transaction(output, tx, consensus_commit_info.round)
                .await?
            {
                ConsensusCertificateResult::SomaTransaction(cert) => {
                    notifications.push(key.clone());
                    verified_certificates.push_back(cert);
                }
                ConsensusCertificateResult::Cancelled((cert, reason)) => {
                    notifications.push(key.clone());
                    assert!(cancelled_txns.insert(*cert.digest(), reason).is_none());
                    verified_certificates.push_back(cert);
                }
                ConsensusCertificateResult::ConsensusMessage => notifications.push(key.clone()),
                ConsensusCertificateResult::IgnoredSystem => {}
                // Note: ignored external transactions must not be recorded as processed. Otherwise
                // they may not get reverted after restart during epoch change.
                ConsensusCertificateResult::Ignored => {
                    ignored = true;
                }
            }
            if !ignored {
                output.record_consensus_message_processed(key.clone());
            }
        }

        // Add the consensus commit prologue transaction to the beginning of `verified_certificates`.
        let consensus_commit_prologue_root = self.add_consensus_commit_prologue_transaction(
            output,
            &mut verified_certificates,
            consensus_commit_info,
            &cancelled_txns,
        )?;

        let verified_certificates: Vec<_> = verified_certificates.into();

        let (lock, final_round) = self.process_end_of_publish_transactions_and_reconfig(
            output,
            end_of_publish_transactions,
        )?;

        Ok((
            verified_certificates,
            notifications,
            lock,
            final_round,
            consensus_commit_prologue_root,
        ))
    }

    fn process_end_of_publish_transactions_and_reconfig(
        &self,
        output: &mut ConsensusCommitOutput,
        transactions: &[VerifiedSequencedConsensusTransaction],
    ) -> SomaResult<(
        Option<RwLockWriteGuard<ReconfigState>>,
        bool, // true if final round
    )> {
        let mut lock = None;

        for transaction in transactions {
            let VerifiedSequencedConsensusTransaction(SequencedConsensusTransaction {
                transaction,
                ..
            }) = transaction;

            if let SequencedConsensusTransactionKind::External(ConsensusTransaction {
                kind: ConsensusTransactionKind::EndOfPublish(authority),
                ..
            }) = transaction
            {
                debug!(
                    "Received EndOfPublish for epoch {} from {:?}",
                    self.committee.epoch,
                    authority.concise()
                );

                // It is ok to just release lock here as this function is the only place that transition into RejectAllCerts state
                // And this function itself is always executed from consensus task
                let collected_end_of_publish = if lock.is_none()
                    && self
                        .get_reconfig_state_read_lock_guard()
                        .should_accept_consensus_certs()
                {
                    output.insert_end_of_publish(*authority);
                    self.end_of_publish.try_lock()
                        .expect("No contention on Authority::end_of_publish as it is only accessed from consensus handler")
                        .insert_generic(*authority, ()).is_quorum_reached()
                    // end_of_publish lock is released here.
                } else {
                    // If we past the stage where we are accepting consensus certificates we also don't record end of publish messages
                    debug!("Ignoring end of publish message from validator {:?} as we already collected enough end of publish messages", authority.concise());
                    false
                };

                if collected_end_of_publish {
                    assert!(lock.is_none());
                    debug!(
                        "Collected enough end_of_publish messages for epoch {} with last message from validator {:?}",
                        self.committee.epoch,
                        authority.concise(),
                    );
                    let mut l = self.get_reconfig_state_write_lock_guard();
                    l.close_all_certs();
                    output.store_reconfig_state(l.clone());
                    // Holding this lock until end of process_consensus_transactions_and_commit_boundary() where we write batch to DB
                    lock = Some(l);
                };
                // Important: we actually rely here on fact that ConsensusHandler panics if its
                // operation returns error. If some day we won't panic in ConsensusHandler on error
                // we need to figure out here how to revert in-memory state of .end_of_publish
                // and .reconfig_state when write fails.
                output.record_consensus_message_processed(transaction.key());
            } else {
                panic!(
                    "process_end_of_publish_transactions_and_reconfig called with non-end-of-publish transaction"
                );
            }
        }

        // Determine if we're ready to advance reconfig state to RejectAllTx.
        let is_reject_all_certs = if let Some(lock) = &lock {
            lock.is_reject_all_certs()
        } else {
            // It is ok to just release lock here as this function is the only place that
            // transitions into RejectAllTx state, and this function itself is always
            // executed from consensus task.
            self.get_reconfig_state_read_lock_guard()
                .is_reject_all_certs()
        };

        if !is_reject_all_certs {
            // // Don't end epoch until all deferred transactions are processed.
            // if is_reject_all_certs {
            //     debug!(
            //         "Blocking end of epoch on deferred transactions, from previous commits?={}, from this commit?={commit_has_deferred_txns}",
            //         !self.deferred_transactions_empty(),
            //     );
            // }
            return Ok((lock, false));
        }

        // Acquire lock to advance state if we don't already have it.
        let mut lock = lock.unwrap_or_else(|| self.get_reconfig_state_write_lock_guard());
        lock.close_all_tx();
        output.store_reconfig_state(lock.clone());
        Ok((Some(lock), true))
    }

    fn process_notifications(
        &self,
        notifications: &[SequencedConsensusTransactionKey],
        end_of_publish: &[VerifiedSequencedConsensusTransaction],
    ) {
        for key in notifications
            .iter()
            .cloned()
            .chain(end_of_publish.iter().map(|tx| tx.0.transaction.key()))
        {
            self.consensus_notify_read.notify(&key, &());
        }
    }

    #[instrument(level = "trace", skip_all)]
    async fn process_consensus_transaction(
        &self,
        output: &mut ConsensusCommitOutput,
        transaction: &VerifiedSequencedConsensusTransaction,
        commit_round: Round,
    ) -> SomaResult<ConsensusCertificateResult> {
        let VerifiedSequencedConsensusTransaction(SequencedConsensusTransaction {
            certificate_author_index: _,
            certificate_author,
            consensus_index,
            transaction,
        }) = transaction;

        match &transaction {
            SequencedConsensusTransactionKind::External(ConsensusTransaction {
                kind: ConsensusTransactionKind::UserTransaction(certificate),
                ..
            }) => {
                if certificate.epoch() != self.epoch() {
                    // Epoch has changed after this certificate was sequenced, ignore it.
                    debug!(
                        "Certificate epoch ({:?}) doesn't match the current epoch ({:?})",
                        certificate.epoch(),
                        self.epoch()
                    );
                    return Ok(ConsensusCertificateResult::Ignored);
                }
                if self.has_sent_end_of_publish(certificate_author)? {
                    // This can not happen with valid authority
                    // With some edge cases consensus might sometimes resend previously seen certificate after EndOfPublish
                    // However this certificate will be filtered out before this line by `consensus_message_processed` call in `verify_consensus_transaction`
                    // If we see some new certificate here it means authority is byzantine and sent certificate after EndOfPublish (or we have some bug in ConsensusAdapter)
                    warn!("[Byzantine authority] Authority {:?} sent a new, previously unseen certificate {:?} after it sent EndOfPublish message to consensus", certificate_author.concise(), certificate.digest());
                    return Ok(ConsensusCertificateResult::Ignored);
                }
                // Safe because signatures are verified when consensus called into SuiTxValidator::validate_batch.
                let certificate = VerifiedCertificate::new_unchecked(*certificate.clone());
                let certificate = VerifiedExecutableTransaction::new_from_certificate(certificate);

                debug!(
                    tx_digest = ?certificate.digest(),
                    "handle_consensus_transaction UserTransaction",
                );

                if !self
                    .get_reconfig_state_read_lock_guard()
                    .should_accept_consensus_certs()
                {
                    debug!("Ignoring consensus certificate for transaction {:?} because of end of epoch",
                    certificate.digest());
                    return Ok(ConsensusCertificateResult::Ignored);
                }

                Ok(ConsensusCertificateResult::SomaTransaction(certificate))
            }
            SequencedConsensusTransactionKind::External(ConsensusTransaction {
                kind: ConsensusTransactionKind::EndOfPublish(_),
                ..
            }) => {
                // these are partitioned earlier
                panic!("process_consensus_transaction called with end-of-publish transaction");
            }

            SequencedConsensusTransactionKind::System(system_transaction) => {
                Ok(self.process_consensus_system_transaction(system_transaction))
            }
        }
    }

    fn process_consensus_system_transaction(
        &self,
        system_transaction: &VerifiedExecutableTransaction,
    ) -> ConsensusCertificateResult {
        if !self.get_reconfig_state_read_lock_guard().should_accept_tx() {
            debug!(
                "Ignoring system transaction {:?} because of end of epoch",
                system_transaction.digest()
            );
            return ConsensusCertificateResult::IgnoredSystem;
        }

        // If needed we can support owned object system transactions as well...
        ConsensusCertificateResult::SomaTransaction(system_transaction.clone())
    }

    #[instrument(level = "trace", skip_all)]
    pub fn verify_transaction(&self, tx: Transaction) -> SomaResult<VerifiedTransaction> {
        self.signature_verifier
            .verify_tx(tx.data())
            .map(|_| VerifiedTransaction::new_from_verified(tx))
    }

    pub fn get_last_consensus_stats(&self) -> SomaResult<ExecutionIndices> {
        match self
            .tables()?
            .get_last_consensus_stats()
            .map_err(SomaError::from)?
        {
            Some(stats) => Ok(stats),
            // TODO: stop reading from last_consensus_index after rollout.
            None => Ok(ExecutionIndices::default()),
        }
    }

    pub fn all_pending_execution(&self) -> SomaResult<Vec<VerifiedExecutableTransaction>> {
        let tables = self.tables()?;
        let pending_execution = tables.pending_execution.read();

        Ok(pending_execution
            .values()
            .cloned()
            .map(|cert| cert.into())
            .collect())
    }

    pub fn store_reconfig_state(&self, new_state: &ReconfigState) -> SomaResult {
        self.tables()?
            .reconfig_state
            .write()
            .insert(RECONFIG_STATE_INDEX, new_state.clone());
        Ok(())
    }

    /// Verifies transaction signatures and other data
    /// Important: This function can potentially be called in parallel and you can not rely on order of transactions to perform verification
    /// If this function return an error, transaction is skipped and is not passed to handle_consensus_transaction
    /// This function returns unit error and is responsible for emitting log messages for internal errors
    fn verify_consensus_transaction(
        &self,
        transaction: SequencedConsensusTransaction,
    ) -> Option<VerifiedSequencedConsensusTransaction> {
        if self
            .is_consensus_message_processed(&transaction.transaction.key())
            .expect("Storage error")
        {
            debug!(
                consensus_index=?transaction.consensus_index.transaction_index,
                "handle_consensus_transaction UserTransaction [skip]",
            );
            return None;
        }
        // Signatures are verified as part of the consensus payload verification in SuiTxValidator
        match &transaction.transaction {
            SequencedConsensusTransactionKind::External(ConsensusTransaction {
                kind: ConsensusTransactionKind::UserTransaction(_certificate),
                ..
            }) => {}
            SequencedConsensusTransactionKind::External(ConsensusTransaction {
                kind: ConsensusTransactionKind::EndOfPublish(authority),
                ..
            }) => {
                if &transaction.sender_authority() != authority {
                    warn!(
                        "EndOfPublish authority {} does not match its author from consensus {}",
                        authority, transaction.certificate_author_index
                    );
                    return None;
                }
            }
            SequencedConsensusTransactionKind::System(_) => {}
        }
        Some(VerifiedSequencedConsensusTransaction(transaction))
    }

    fn finish_consensus_certificate_process_with_batch(
        &self,
        output: &mut ConsensusCommitOutput,
        certificates: &[VerifiedExecutableTransaction],
    ) -> SomaResult {
        output.insert_pending_execution(certificates);
        // output.insert_user_signatures_for_checkpoints(certificates);
        Ok(())
    }

    /// Acquire the lock for a tx without writing to the WAL.
    pub async fn acquire_tx_lock(&self, digest: &TransactionDigest) -> CertLockGuard {
        CertLockGuard(self.mutex_table.acquire_lock(*digest).await)
    }

    pub async fn acquire_tx_guard(
        &self,
        cert: &VerifiedExecutableTransaction,
    ) -> SomaResult<CertTxGuard> {
        let digest = cert.digest();
        Ok(CertTxGuard(self.acquire_tx_lock(digest).await))
    }

    pub fn execute_transaction(
        &self,
        store: &dyn ObjectStore,
        tx_digest: TransactionDigest,
        kind: TransactionKind,
        signer: SomaAddress,
    ) -> (WrittenObjects, TransactionEffects, Option<ExecutionError>) {
        if let TransactionKind::ConsensusCommitPrologue(_prologue) = kind {
            let input_objects: BTreeMap<ObjectID, Object> = BTreeMap::new();
            let lamport_timestamp =
                Version::lamport_increment(input_objects.iter().map(|(_, o)| o.version()));

            let temporary_store = TemporaryStore::new(input_objects, tx_digest, lamport_timestamp);
            let (written_objects, effects) =
                temporary_store.into_effects(&tx_digest, ExecutionStatus::Success, self.epoch());

            return (written_objects, effects, None);
        }

        let mut input_objects: BTreeMap<ObjectID, Object> = BTreeMap::new();

        // TODO: add input objects based on tx kind
        input_objects.insert(
            SYSTEM_STATE_OBJECT_ID,
            store
                .get_object(&SYSTEM_STATE_OBJECT_ID)
                .unwrap()
                .expect("Expected system state object to exist"),
        );

        let lamport_timestamp =
            Version::lamport_increment(input_objects.iter().map(|(_, o)| o.version()));

        let mut temporary_store = TemporaryStore::new(input_objects, tx_digest, lamport_timestamp);

        let mut state_object = temporary_store
            .read_object(&SYSTEM_STATE_OBJECT_ID)
            .expect("Could not get system state object");
        let mut state =
            bcs::from_bytes::<SystemState>(state_object.as_inner().data.contents()).unwrap();

        // apply result to state
        let result = match kind {
            TransactionKind::StateTransaction(tx) => match tx.kind {
                StateTransactionKind::AddValidator(args) => {
                    // TODO: check if the signer is the same as the validator

                    state.request_add_validator(
                        signer,
                        args.pubkey_bytes,
                        args.network_pubkey_bytes,
                        args.worker_pubkey_bytes,
                        args.net_address,
                        args.p2p_address,
                        args.primary_address,
                    )
                }
                StateTransactionKind::RemoveValidator(args) => {
                    // TODO: check if the signer is the same as the validator
                    info!("Init state object: {:?}", state);
                    state.request_remove_validator(signer, args.pubkey_bytes)
                }
            },
            TransactionKind::EndOfEpochTransaction(tx) => match tx {
                EndOfEpochTransactionKind::ChangeEpoch(change_epoch) => {
                    state.advance_epoch(change_epoch.epoch, 0) // TODO: figure out how to make epoch_start_timestamp_ms the same across all SystemState objects
                }
            },
            _ => Ok(()),
        };

        let execution_status = match &result {
            Ok(()) => ExecutionStatus::Success,
            Err(err) => {
                let error = ExecutionFailureStatus::SomaError(err.clone());
                ExecutionStatus::Failure { error }
            }
        };

        let execution_error = match &result {
            Ok(()) => None,
            Err(err) => {
                let error = ExecutionFailureStatus::SomaError(err.clone());
                Some(ExecutionError::new(error, None))
            }
        };
        info!("After state object: {:?}", state);
        // turn state back to object
        state_object
            .data
            .update_contents(bcs::to_bytes(&state).unwrap());

        temporary_store.mutate_input_object(state_object);
        let (written_objects, effects) =
            temporary_store.into_effects(&tx_digest, execution_status, self.epoch());

        return (written_objects, effects, execution_error);
    }

    #[instrument(level = "trace", skip_all)]
    pub fn insert_tx_cert_sig(
        &self,
        tx_digest: &TransactionDigest,
        cert_sig: &AuthorityStrongQuorumSignInfo,
    ) -> SomaResult {
        let tables = self.tables()?;
        let _ = tables
            .transaction_cert_signatures
            .write()
            .insert(*tx_digest, cert_sig.clone());
        Ok(())
    }

    pub fn get_transaction_cert_sig(
        &self,
        tx_digest: &TransactionDigest,
    ) -> SomaResult<Option<AuthorityStrongQuorumSignInfo>> {
        Ok(self
            .tables()?
            .transaction_cert_signatures
            .read()
            .get(tx_digest)
            .cloned())
    }

    pub fn get_effects_signature(
        &self,
        tx_digest: &TransactionDigest,
    ) -> SomaResult<Option<AuthoritySignInfo>> {
        let tables = self.tables()?;
        let effects_signatures = tables.effects_signatures.read();
        Ok(effects_signatures.get(tx_digest).cloned())
    }

    pub fn insert_effects_digest_and_signature(
        &self,
        tx_digest: &TransactionDigest,
        effects_digest: &TransactionEffectsDigest,
        effects_signature: &AuthoritySignInfo,
    ) -> SomaResult {
        let tables = self.tables()?;
        let _ = tables
            .effects_signatures
            .write()
            .insert(*tx_digest, effects_signature.clone());
        let _ = tables
            .signed_effects_digests
            .write()
            .insert(*tx_digest, *effects_digest);

        Ok(())
    }

    #[instrument(level = "trace", skip_all)]
    pub fn insert_tx_key_and_effects_signature(
        &self,
        tx_key: &TransactionKey,
        tx_digest: &TransactionDigest,
        effects_digest: &TransactionEffectsDigest,
        effects_signature: Option<&AuthoritySignInfo>,
    ) -> SomaResult {
        let tables = self.tables()?;

        if let Some(effects_signature) = effects_signature {
            let _ = tables
                .effects_signatures
                .write()
                .insert(*tx_digest, effects_signature.clone());
            let _ = tables
                .signed_effects_digests
                .write()
                .insert(*tx_digest, *effects_digest);
        }

        if !matches!(tx_key, TransactionKey::Digest(_)) {
            self.executed_digests_notify_read.notify(tx_key, tx_digest);
        }
        Ok(())
    }

    pub fn get_signed_effects_digest(
        &self,
        tx_digest: &TransactionDigest,
    ) -> SomaResult<Option<TransactionEffectsDigest>> {
        let tables = self.tables()?;
        let signed_effects_digests = tables.signed_effects_digests.read();
        Ok(signed_effects_digests.get(tx_digest).cloned())
    }
}

#[derive(Default)]
pub(crate) struct ConsensusCommitOutput {
    // Consensus and reconfig state
    consensus_messages_processed: BTreeSet<SequencedConsensusTransactionKey>,
    end_of_publish: BTreeSet<AuthorityName>,
    reconfig_state: Option<ReconfigState>,
    pending_execution: Vec<VerifiedExecutableTransaction>,
    consensus_commit_stats: Option<ExecutionIndices>,
}

impl ConsensusCommitOutput {
    pub fn new() -> Self {
        Default::default()
    }

    fn insert_end_of_publish(&mut self, authority: AuthorityName) {
        self.end_of_publish.insert(authority);
    }

    fn insert_pending_execution(&mut self, transactions: &[VerifiedExecutableTransaction]) {
        self.pending_execution.reserve(transactions.len());
        self.pending_execution.extend_from_slice(transactions);
    }

    fn store_reconfig_state(&mut self, state: ReconfigState) {
        self.reconfig_state = Some(state);
    }

    fn record_consensus_message_processed(&mut self, key: SequencedConsensusTransactionKey) {
        self.consensus_messages_processed.insert(key);
    }

    fn record_consensus_commit_stats(&mut self, stats: ExecutionIndices) {
        self.consensus_commit_stats = Some(stats);
    }

    pub fn write_to_batch(
        self,
        epoch_store: &AuthorityPerEpochStore,
        // batch: &mut DBBatch,
    ) -> SomaResult {
        let tables = epoch_store.tables()?;
        tables.consensus_message_processed.write().extend(
            self.consensus_messages_processed
                .iter()
                .map(|key| (key.clone(), true)),
        );

        tables
            .end_of_publish
            .write()
            .extend(self.end_of_publish.iter().map(|authority| (*authority, ())));

        if let Some(reconfig_state) = &self.reconfig_state {
            tables
                .reconfig_state
                .write()
                .insert(RECONFIG_STATE_INDEX, reconfig_state.clone());
        }

        if let Some(consensus_commit_stats) = &self.consensus_commit_stats {
            tables
                .last_consensus_stats
                .write()
                .insert(LAST_CONSENSUS_STATS_ADDR, consensus_commit_stats.clone());
        }

        tables.pending_execution.write().extend(
            self.pending_execution
                .into_iter()
                .map(|tx| (*tx.inner().digest(), tx.serializable())),
        );

        Ok(())
    }
}
