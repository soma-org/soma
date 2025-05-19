//! # Epoch Store
//!
//! ## Overview
//! The epoch store manages all epoch-specific state and storage for a validator authority.
//! It provides a clean separation between data from different epochs, enabling safe epoch
//! transitions and reconfiguration.
//!
//! ## Responsibilities
//! - Manage epoch-specific database tables and in-memory state
//! - Process consensus transactions and assign shared object versions
//! - Handle transaction execution and effects certification
//! - Manage epoch transitions and reconfiguration state
//! - Coordinate validator committee operations
//!
//! ## Component Relationships
//! - Interacts with AuthorityState to provide epoch-specific storage
//! - Provides transaction processing services to consensus
//! - Manages shared object versioning for transaction execution
//! - Coordinates with reconfiguration process for epoch changes
//!
//! ## Key Workflows
//! 1. Consensus transaction processing and shared object version assignment
//! 2. Transaction execution and effects certification
//! 3. Epoch transition and reconfiguration
//! 4. End-of-epoch protocol coordination
//!
//! ## Design Patterns
//! - Epoch isolation: All epoch-specific data is contained within the epoch store
//! - Clean shutdown: Graceful termination of epoch-specific tasks
//! - Consensus quarantine: Protection against forking during consensus processing
//! - Version assignment: Deterministic shared object version management

use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque},
    future::Future,
    ops::{Bound, Deref},
    path::PathBuf,
    sync::{Arc, Weak},
    time::{Duration, Instant},
};

use arc_swap::ArcSwapOption;
use futures::{
    future::select,
    future::{join_all, Either},
    FutureExt,
};
use itertools::izip;
use parking_lot::{Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard};
use tracing::{debug, info, instrument, trace, warn};
use types::{
    accumulator::{Accumulator, CommitIndex},
    base::{
        AuthorityName, ConciseableName, ConsensusObjectSequenceKey, FullObjectID, Round,
        SomaAddress,
    },
    committee::{Authority, Committee, EncoderCommittee, EpochId},
    consensus::{
        validator_set::ValidatorSet, ConsensusCommitPrologue, ConsensusTransaction,
        ConsensusTransactionKey, ConsensusTransactionKind, EndOfEpochAPI,
    },
    crypto::{AuthorityPublicKeyBytes, AuthoritySignInfo, AuthorityStrongQuorumSignInfo, Signer},
    digests::{ECMHLiveObjectSetDigest, TransactionDigest, TransactionEffectsDigest},
    effects::{
        self,
        object_change::{EffectsObjectChange, IDOperation, ObjectIn, ObjectOut},
        ExecutionFailureStatus, ExecutionStatus, TransactionEffects, UnchangedSharedKind,
    },
    envelope::TrustedEnvelope,
    error::{ExecutionError, SomaError, SomaResult},
    execution_indices::ExecutionIndices,
    mutex_table::{MutexGuard, MutexTable},
    object::{Object, ObjectData, ObjectID, ObjectRef, ObjectType, Owner, Version},
    protocol::{Chain, ProtocolConfig},
    signature_verifier::SignatureVerifier,
    storage::{object_store::ObjectStore, InputKey},
    system_state::{
        self,
        epoch_start::{EpochStartSystemState, EpochStartSystemStateTrait},
        get_system_state, SystemState, SystemStateTrait,
    },
    temporary_store::{InnerTemporaryStore, SharedInput, TemporaryStore},
    transaction::{
        self, CertifiedTransaction, InputObjectKind, InputObjects, ObjectReadResult,
        ObjectReadResultKind, SenderSignedData, Transaction, TransactionKey, TransactionKind,
        TrustedExecutableTransaction, VerifiedCertificate, VerifiedExecutableTransaction,
        VerifiedSignedTransaction, VerifiedTransaction,
    },
    tx_outputs::WrittenObjects,
    SYSTEM_STATE_OBJECT_ID,
};
use utils::{notify_once::NotifyOnce, notify_read::NotifyRead};

use crate::{
    cache::ObjectCacheRead,
    consensus_quarantine::{
        ConsensusCommitOutput, ConsensusOutputCache, ConsensusOutputQuarantine,
        LAST_CONSENSUS_STATS_ADDR, RECONFIG_STATE_INDEX,
    },
    handler::{
        ConsensusCommitInfo, SequencedConsensusTransaction, SequencedConsensusTransactionKey,
        SequencedConsensusTransactionKind, VerifiedSequencedConsensusTransaction,
    },
    reconfiguration::ReconfigState,
    shared_obj_version_manager::{
        AssignedTxAndVersions, ConsensusSharedObjVerAssignment, SharedObjVerManager,
    },
    stake_aggregator::StakeAggregator,
    start_epoch::{EpochStartConfigTrait, EpochStartConfiguration},
    state_accumulator::StateAccumulator,
    store::LockDetails,
};

/// # CertLockGuard and CertTxGuard
///
/// Transaction lock guards for preventing concurrent execution of the same transaction.
/// These types provide a clean abstraction for transaction locking that can be extended
/// in the future to support distributed object storage.
///
/// ## Purpose
/// - Prevent concurrent execution of the same transaction
/// - Provide a clean abstraction for transaction locking
/// - Enable future extensions for distributed storage
///
/// ## Usage
/// - Acquire a lock before executing a transaction
/// - Release the lock after execution is complete
/// - Use commit_tx to signal successful execution
///
/// ## Thread Safety
/// These guards work with the mutex_table to provide fine-grained locking
/// at the transaction level, allowing concurrent execution of different transactions.
pub struct CertLockGuard(#[allow(unused)] MutexGuard);

/// Transaction guard that wraps a CertLockGuard and provides additional
/// transaction-specific operations.
pub struct CertTxGuard(#[allow(unused)] CertLockGuard);

impl CertTxGuard {
    /// Release the transaction lock without any additional actions.
    pub fn release(self) {}

    /// Commit the transaction and release the lock.
    pub fn commit_tx(self) {}

    /// Get a reference to the underlying lock guard.
    pub fn as_lock_guard(&self) -> &CertLockGuard {
        &self.0
    }
}

/// # CancelConsensusCertificateReason
///
/// Reasons why a transaction certificate might be cancelled during consensus processing.
///
/// ## Purpose
/// Provides detailed information about why a transaction was cancelled,
/// which can be used for error reporting and debugging.
///
/// ## Variants
/// Currently only supports cancellation due to congestion on specific objects,
/// but can be extended with additional cancellation reasons in the future.
pub enum CancelConsensusCertificateReason {
    /// Transaction was cancelled due to congestion on specific objects.
    /// Contains the list of object IDs that experienced congestion.
    CongestionOnObjects(Vec<ObjectID>),
}

/// # ConsensusCertificateResult
///
/// Represents the result of processing a consensus transaction or certificate.
/// This enum is used to determine how a transaction should be handled after
/// it has been processed by the consensus handler.
///
/// ## Variants
/// Different outcomes require different handling in the consensus flow:
/// - Some transactions are executed immediately
/// - Some are ignored (already processed or invalid for current epoch)
/// - Some are cancelled but still need to go through execution engine
/// - Some are just consensus protocol messages
pub enum ConsensusCertificateResult {
    /// The consensus message was ignored (e.g. because it has already been processed).
    Ignored,

    /// An executable transaction (can be a user tx or a system tx)
    /// that should be scheduled for execution.
    SomaTransaction(VerifiedExecutableTransaction),

    /// The transaction should be re-processed at a future commit, specified by the DeferralKey
    // Deferred(DeferralKey),

    /// A message was processed which updates randomness state.
    // RandomnessConsensusMessage,

    /// Everything else, e.g. AuthorityCapabilities, CheckpointSignatures, etc.
    /// These messages update consensus state but don't result in transactions.
    ConsensusMessage,

    /// A system message in consensus was ignored (e.g. because of end of epoch).
    /// Different from Ignored in that it specifically applies to system messages.
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

/// # AuthorityPerEpochStore
///
/// The primary store for all epoch-specific state and storage in a validator authority.
///
/// ## Purpose
/// Manages all data that is specific to a single epoch, providing clean isolation between
/// epochs and enabling safe epoch transitions. This includes transaction processing,
/// consensus state, shared object versioning, and reconfiguration state.
///
/// ## Lifecycle
/// - Created at the beginning of an epoch with a specific committee and configuration
/// - Used throughout the epoch for transaction processing and consensus operations
/// - Gracefully shut down during epoch transition, ensuring all in-flight operations complete
/// - Replaced by a new instance for the next epoch
///
/// ## Thread Safety
/// Designed for concurrent access with careful lock management:
/// - Uses RwLocks for shared state that needs concurrent readers
/// - Uses MutexTables for fine-grained locking of specific objects
/// - Implements notification patterns for async coordination
/// - Maintains careful lock ordering to prevent deadlocks
///
/// ## Key Components
/// - Tables: Epoch-specific storage tables for transactions, effects, and state
/// - Consensus quarantine: Protection against forking during consensus processing
/// - Reconfiguration state: Manages the epoch change protocol
/// - Shared object version management: Assigns versions to shared objects
pub struct AuthorityPerEpochStore {
    /// The name of this authority.
    pub(crate) name: AuthorityName,

    /// Committee of validators for the current epoch.
    committee: Arc<Committee>,

    /// Holds the underlying per-epoch typed store tables.
    /// This is an ArcSwapOption because it needs to be used concurrently,
    /// and it needs to be cleared at the end of the epoch.
    tables: ArcSwapOption<AuthorityEpochTables>,

    /// Holds the outputs of consensus handler in memory
    /// until they are proven not to have forked by a commit.
    consensus_quarantine: RwLock<ConsensusOutputQuarantine>,
    /// Holds variouis data from consensus_quarantine in a more easily accessible form.
    consensus_output_cache: ConsensusOutputCache,

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

    /// Mutex table for shared version assignment
    version_assignment_mutex_table: MutexTable<ObjectID>,

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

    // Subscribers will get notified when a transaction is executed via commit execution. (From State Sync)
    executed_transactions_to_commit_notify_read: NotifyRead<TransactionDigest, CommitIndex>,

    /// Caches the highest synced commit index as this has been notified from the CommitExecutor
    highest_synced_commit: RwLock<CommitIndex>,
    /// Get notified when a synced commit has reached CommitExecutor.
    synced_commit_notify_read: NotifyRead<CommitIndex, ()>,

    next_epoch_state: RwLock<Option<(SystemState, ECMHLiveObjectSetDigest)>>,
}

/// # AuthorityEpochTables
///
/// Contains tables that store epoch-specific data that is only valid within a single epoch.
///
/// ## Purpose
/// Provides a clean separation of data between epochs, ensuring that data from previous epochs
/// doesn't interfere with the current epoch's operation. This isolation is critical for
/// maintaining consistency during epoch transitions.
///
/// ## Lifecycle
/// - Created at the beginning of an epoch
/// - Used throughout the epoch for transaction processing and state management
/// - Discarded or archived at the end of the epoch
///
/// ## Thread Safety
/// Most tables are protected by RwLocks to allow concurrent access while maintaining
/// consistency. The lock ordering must be carefully maintained to prevent deadlocks.
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
    pub(crate) consensus_message_processed:
        RwLock<BTreeMap<SequencedConsensusTransactionKey, bool>>,

    /// Map stores pending transactions that this authority submitted to consensus
    pending_consensus_transactions: RwLock<BTreeMap<ConsensusTransactionKey, ConsensusTransaction>>,

    /// The following table is used to store a single value (the corresponding key is a constant). The value
    /// represents the index of the latest consensus message this authority processed, running hash of
    /// transactions, and accumulated stats of consensus output.
    /// This field is written by a single process (consensus handler).
    pub(crate) last_consensus_stats: RwLock<BTreeMap<u64, ExecutionIndices>>,

    /// This table contains current reconfiguration state for validator for current epoch
    pub(crate) reconfig_state: RwLock<BTreeMap<u64, ReconfigState>>,

    /// Validators that have sent EndOfPublish message in this epoch
    pub(crate) end_of_publish: RwLock<BTreeMap<AuthorityName, ()>>,

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

    /// When transaction is executed via commit executor, we store association here
    pub(crate) executed_transactions_to_commit: RwLock<BTreeMap<TransactionDigest, CommitIndex>>,

    /// Next available shared object versions for each shared object.
    pub(crate) next_shared_object_versions: RwLock<BTreeMap<ConsensusObjectSequenceKey, Version>>,
    // TODO:  Transactions that were executed in the current epoch: executed_in_epoch
    // TODO: Transactions that are being deferred until some future time deferred_transactions: DBMap<DeferralKey, Vec<VerifiedSequencedConsensusTransaction>>,
    // TODO: Accumulated per-object debts for congestion control. congestion_control_object_debts: DBMap<ObjectID, CongestionPerObjectDebt>,
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
        signed_transaction: VerifiedSignedTransaction,
        locks_to_write: impl Iterator<Item = (ObjectRef, TransactionDigest)>,
    ) -> SomaResult {
        info!(
            "Writing transaction locks for tx: {}",
            signed_transaction.digest()
        );
        // Insert locks

        let mut locked_transactions = self.object_locked_transactions.write();
        for (obj_ref, lock) in locks_to_write {
            locked_transactions.insert(obj_ref, lock);
        }

        // Insert transaction

        let mut transactions = self.signed_transactions.write();
        transactions.insert(
            *signed_transaction.digest(),
            signed_transaction.serializable_ref().clone(),
        );

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
        highest_executed_commit: CommitIndex,
    ) -> Arc<Self> {
        let current_time = Instant::now();
        let epoch_id = committee.epoch;

        // TODO: change this
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

        let signature_verifier = SignatureVerifier::new(committee.clone(), None);

        let consensus_output_cache = ConsensusOutputCache::new(&epoch_start_configuration, &tables);

        let s = Arc::new(Self {
            name,
            committee,
            protocol_config,
            tables: ArcSwapOption::new(Some(Arc::new(tables))),
            consensus_output_cache,
            consensus_quarantine: RwLock::new(ConsensusOutputQuarantine::new(
                highest_executed_commit,
            )),
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
            version_assignment_mutex_table: MutexTable::new(MUTEX_TABLE_SIZE),
            epoch_open_time: current_time,
            epoch_close_time: Default::default(),
            epoch_start_configuration,
            executed_transactions_to_commit_notify_read: NotifyRead::new(),
            highest_synced_commit: RwLock::new(0),
            synced_commit_notify_read: NotifyRead::new(),
            next_epoch_state: RwLock::new(None),
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
        highest_executed_commit: CommitIndex,
    ) -> Arc<Self> {
        assert_eq!(self.epoch() + 1, new_committee.epoch);

        // Get the last consensus stats from current epoch
        let last_consensus_stats = self
            .tables()
            .expect("Tables should exist when creating next epoch")
            .get_last_consensus_stats()
            .expect("Reading last consensus stats should not fail")
            .unwrap_or_default();

        // Create new tables but initialize with previous consensus stats
        let mut tables = AuthorityEpochTables::default();
        tables
            .last_consensus_stats
            .write()
            .insert(LAST_CONSENSUS_STATS_ADDR, last_consensus_stats);

        let epoch_id = new_committee.epoch;
        let current_time = Instant::now();

        // Rest of initialization...
        let end_of_publish = StakeAggregator::from_iter(
            Arc::new(new_committee.clone()),
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
        let signature_verifier = SignatureVerifier::new(Arc::new(new_committee.clone()), None);
        let consensus_output_cache = ConsensusOutputCache::new(&epoch_start_configuration, &tables);

        Arc::new(Self {
            name,
            committee: Arc::new(new_committee),
            protocol_config,
            tables: ArcSwapOption::new(Some(Arc::new(tables))),
            consensus_output_cache,
            consensus_quarantine: RwLock::new(ConsensusOutputQuarantine::new(
                highest_executed_commit,
            )),
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
            version_assignment_mutex_table: MutexTable::new(MUTEX_TABLE_SIZE),
            epoch_open_time: current_time,
            epoch_close_time: Default::default(),
            epoch_start_configuration,
            executed_transactions_to_commit_notify_read: NotifyRead::new(),
            highest_synced_commit: RwLock::new(0),
            synced_commit_notify_read: NotifyRead::new(),
            next_epoch_state: RwLock::new(None),
        })
    }

    pub fn new_at_next_epoch_for_testing(&self) -> Arc<Self> {
        let next_epoch = self.epoch() + 1;
        let next_committee = Committee::new(
            next_epoch,
            self.committee.voting_rights.iter().cloned().collect(),
            self.committee.authorities.clone().into_iter().collect(),
        );

        let epoch_start_configuration = self.epoch_start_configuration.as_ref().clone();
        self.new_at_next_epoch(
            self.name,
            next_committee,
            epoch_start_configuration.new_at_next_epoch_for_testing(),
            0,
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

    pub fn reference_byte_price(&self) -> u64 {
        self.epoch_start_state().reference_byte_price()
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

    /// Notifies that a synced commit of index `index` is available. The source of the notification
    /// is the CommitExecutor. The consumer here is guaranteed to be notified in index order.
    pub fn notify_synced_commit(&self, index: CommitIndex) {
        let mut highest_synced_commit = self.highest_synced_commit.write();
        *highest_synced_commit = index;
        self.synced_commit_notify_read.notify(&index, &());
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
        cache_reader: &dyn ObjectCacheRead,
        consensus_commit_info: &ConsensusCommitInfo,
    ) -> SomaResult<(Vec<VerifiedExecutableTransaction>, bool)> {
        // Return if end of publish quorum reached to ConsensusHandler
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
            end_of_publish_quorum,
            consensus_commit_prologue_root,
        ) = self
            .process_consensus_transactions(
                &mut output,
                &consensus_transactions,
                &end_of_publish_transactions,
                cache_reader,
                consensus_commit_info,
            )
            .await?;
        self.finish_consensus_certificate_process_with_batch(
            &mut output,
            &transactions_to_schedule,
        )?;
        output.record_consensus_commit_stats(consensus_stats.clone());

        self.consensus_quarantine
            .write()
            .push_consensus_output(output, self)?;

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
        let make_checkpoint = should_accept_tx || end_of_publish_quorum;
        if make_checkpoint {
            // TODO: Generate pending checkpoint for regular user tx.
        }

        // Only after batch is written, notify checkpoint service to start building any new
        // pending checkpoints.
        if make_checkpoint {
            // TODO: Notify checkpoint service
        }

        self.process_notifications(&notifications, &end_of_publish_transactions);

        if end_of_publish_quorum {
            info!(
                epoch=?self.epoch(),
                // Accessing lock on purpose so that the compiler ensures
                // the lock is not yet dropped.
                lock=?lock.as_ref(),
                end_of_publish_quorum=?end_of_publish_quorum,
                "Notified last commit"
            );
        }

        Ok((transactions_to_schedule, end_of_publish_quorum))
    }

    // Caller is not required to set ExecutionIndices with the right semantics in
    // VerifiedSequencedConsensusTransaction.
    // Also, ConsensusStats and hash will not be updated in the db with this function, unlike in
    // process_consensus_transactions_and_commit_boundary().
    pub async fn process_consensus_transactions_for_tests(
        self: &Arc<Self>,
        transactions: Vec<SequencedConsensusTransaction>,
        cache_reader: &dyn ObjectCacheRead,
        // tx_reader: &dyn TransactionCacheRead,
        skip_consensus_commit_prologue_in_test: bool,
    ) -> SomaResult<(Vec<VerifiedExecutableTransaction>, bool)> {
        self.process_consensus_transactions_and_commit_boundary(
            transactions,
            &ExecutionIndices::default(),
            cache_reader,
            // tx_reader,
            &ConsensusCommitInfo::new_for_test(
                0,
                Duration::from_millis(80).as_secs(),
                skip_consensus_commit_prologue_in_test,
            ),
        )
        .await
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

        let mut version_assignment = Vec::new();
        let mut shared_input_next_version = HashMap::new();
        for txn in transactions.iter() {
            match cancelled_txns.get(txn.digest()) {
                Some(CancelConsensusCertificateReason::CongestionOnObjects(_)) => {
                    let assigned_versions = SharedObjVerManager::assign_versions_for_certificate(
                        txn,
                        &mut shared_input_next_version,
                        cancelled_txns,
                    );
                    version_assignment.push((*txn.digest(), assigned_versions));
                }
                None => {}
            }
        }

        let transaction =
            consensus_commit_info.create_consensus_commit_prologue_transaction(self.epoch());
        let consensus_commit_prologue_root =
            match self.process_consensus_system_transaction(&transaction) {
                ConsensusCertificateResult::SomaTransaction(processed_tx) => {
                    transactions.push_front(processed_tx.clone());
                    Some(processed_tx.key())
                }
                ConsensusCertificateResult::IgnoredSystem => None,
                _ => unreachable!(
                    "process_consensus_system_transaction returned unexpected \
                     ConsensusCertificateResult."
                ),
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
        cache_reader: &dyn ObjectCacheRead,
        consensus_commit_info: &ConsensusCommitInfo,
    ) -> SomaResult<(
        Vec<VerifiedExecutableTransaction>,    // transactions to schedule
        Vec<SequencedConsensusTransactionKey>, // keys to notify as complete
        Option<RwLockWriteGuard<ReconfigState>>,
        bool,                   // true if end of publish quorum reached
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

        self.process_consensus_transaction_shared_object_versions(
            cache_reader,
            &verified_certificates,
            &cancelled_txns,
            output,
        )?;

        let (lock, end_of_publish_quorum) = self.process_end_of_publish_transactions_and_reconfig(
            output,
            end_of_publish_transactions,
        )?;

        Ok((
            verified_certificates,
            notifications,
            lock,
            end_of_publish_quorum,
            consensus_commit_prologue_root,
        ))
    }

    fn process_end_of_publish_transactions_and_reconfig(
        &self,
        output: &mut ConsensusCommitOutput,
        transactions: &[VerifiedSequencedConsensusTransaction],
    ) -> SomaResult<(
        Option<RwLockWriteGuard<ReconfigState>>,
        bool, // true if end of publish quorum reached
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
                    self.end_of_publish
                        .try_lock()
                        .expect(
                            "No contention on Authority::end_of_publish as it is only accessed \
                             from consensus handler",
                        )
                        .insert_generic(*authority, ())
                        .is_quorum_reached()
                    // end_of_publish lock is released here.
                } else {
                    // If we past the stage where we are accepting consensus certificates we also don't record end of publish messages
                    debug!(
                        "Ignoring end of publish message from validator {:?} as we already \
                         collected enough end of publish messages",
                        authority.concise()
                    );
                    false
                };

                if collected_end_of_publish {
                    assert!(lock.is_none());
                    debug!(
                        "Collected enough end_of_publish messages for epoch {} with last message \
                         from validator {:?}",
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
                    "process_end_of_publish_transactions_and_reconfig called with \
                     non-end-of-publish transaction"
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
                    warn!(
                        "[Byzantine authority] Authority {:?} sent a new, previously unseen \
                         certificate {:?} after it sent EndOfPublish message to consensus",
                        certificate_author.concise(),
                        certificate.digest()
                    );
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
                    debug!(
                        "Ignoring consensus certificate for transaction {:?} because of end of \
                         epoch",
                        certificate.digest()
                    );
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
            .verify_tx(tx.data(), self.epoch())
            .map(|_| VerifiedTransaction::new_from_verified(tx))
    }

    pub fn get_last_consensus_stats(&self) -> SomaResult<ExecutionIndices> {
        match self
            .tables()?
            .get_last_consensus_stats()
            .map_err(SomaError::from)?
        {
            Some(stats) => {
                info!(
                    "Got last ExecutionIndices for epoch {}: commit index {}",
                    self.epoch(),
                    stats.sub_dag_index
                );
                Ok(stats)
            }
            // TODO: stop reading from last_consensus_index after rollout.
            None => {
                info!(
                    "Did not get last ExecutionIndices for epoch {}",
                    self.epoch()
                );
                Ok(ExecutionIndices::default())
            }
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

    /// Called when transaction outputs are committed to disk
    #[instrument(level = "trace", skip_all)]
    pub fn handle_committed_transactions(
        &self,
        commit: CommitIndex,
        digests: &[TransactionDigest],
    ) -> SomaResult<()> {
        let tables = match self.tables() {
            Ok(tables) => tables,
            // After Epoch ends, it is no longer necessary to remove pending transactions
            // because the table will not be used anymore and be deleted eventually.
            Err(SomaError::EpochEnded(_)) => return Ok(()),
            Err(e) => return Err(e),
        };

        // pending_execution stores transactions received from consensus which may not have
        // been executed yet. At this point, they have been committed to the db durably and
        // can be removed.
        // After end-to-end quarantining, we will not need pending_execution since the consensus
        // log itself will be used for recovery.
        tables
            .pending_execution
            .write()
            .retain(|k, _| !digests.contains(k));

        // Now that the transaction effects are committed, we will never re-execute, so we
        // don't need to worry about equivocating.
        tables
            .signed_effects_digests
            .write()
            .retain(|d, _| !digests.contains(d));

        let mut quarantine = self.consensus_quarantine.write();
        quarantine.update_highest_executed_commit(commit, self)?;

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

    #[cfg(test)]
    pub fn delete_signed_transaction_for_test(&self, transaction: &TransactionDigest) {
        self.tables()
            .expect("test should not cross epoch boundary")
            .signed_transactions
            .write()
            .remove(transaction)
            .unwrap();
    }

    #[cfg(test)]
    pub fn delete_object_locks_for_test(&self, objects: &[ObjectRef]) {
        for object in objects {
            self.tables()
                .expect("test should not cross epoch boundary")
                .object_locked_transactions
                .write()
                .remove(object)
                .unwrap();
        }
    }

    pub fn get_signed_transaction(
        &self,
        tx_digest: &TransactionDigest,
    ) -> SomaResult<Option<VerifiedSignedTransaction>> {
        Ok(self
            .tables()?
            .signed_transactions
            .read()
            .get(tx_digest)
            .cloned()
            .map(|t| t.into()))
    }

    pub async fn acquire_tx_guard(
        &self,
        cert: &VerifiedExecutableTransaction,
    ) -> SomaResult<CertTxGuard> {
        let digest = cert.digest();
        Ok(CertTxGuard(self.acquire_tx_lock(digest).await))
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

    pub(crate) fn remove_shared_version_assignments<'a>(
        &self,
        keys: impl IntoIterator<Item = &'a TransactionKey>,
    ) {
        self.consensus_output_cache
            .remove_shared_object_assignments(keys);
    }

    pub fn num_shared_version_assignments(&self) -> usize {
        self.consensus_output_cache.num_shared_version_assignments()
    }

    pub fn get_signed_effects_digest(
        &self,
        tx_digest: &TransactionDigest,
    ) -> SomaResult<Option<TransactionEffectsDigest>> {
        let tables = self.tables()?;
        let signed_effects_digests = tables.signed_effects_digests.read();
        Ok(signed_effects_digests.get(tx_digest).cloned())
    }

    #[cfg(test)]
    pub fn get_next_object_version(
        &self,
        obj: &ObjectID,
        start_version: Version,
    ) -> Option<Version> {
        self.tables()
            .expect("test should not cross epoch boundary")
            .next_shared_object_versions
            .read()
            .get(&(*obj, start_version))
            .cloned()
    }

    pub fn set_shared_object_versions_for_testing(
        &self,
        tx_digest: &TransactionDigest,
        assigned_versions: &[(ConsensusObjectSequenceKey, Version)],
    ) -> SomaResult {
        self.consensus_output_cache
            .set_shared_object_versions_for_testing(tx_digest, assigned_versions);
        Ok(())
    }

    /// Resolves InputObjectKinds into InputKeys, by consulting the shared object version
    /// assignment table.
    pub(crate) fn get_input_object_keys(
        &self,
        key: &TransactionKey,
        objects: &[InputObjectKind],
    ) -> SomaResult<BTreeSet<InputKey>> {
        let assigned_shared_versions = once_cell::unsync::OnceCell::<
            Option<HashMap<ConsensusObjectSequenceKey, Version>>,
        >::new();
        objects
            .iter()
            .map(|kind| {
                Ok(match kind {
                    InputObjectKind::SharedObject {
                        id,
                        initial_shared_version,
                        ..
                    } => {
                        let assigned_shared_versions = assigned_shared_versions
                            .get_or_init(|| {
                                self.get_assigned_shared_object_versions(key)
                                    .map(|versions| versions.into_iter().collect())
                            })
                            .as_ref()
                            // Shared version assignments could have been deleted if the tx just
                            // finished executing concurrently.
                            .ok_or(SomaError::GenericAuthorityError {
                                error: "no assigned shared versions".to_string(),
                            })?;

                        let modified_initial_shared_version = *initial_shared_version;
                        // If we found assigned versions, but they are missing the assignment for
                        // this object, it indicates a serious inconsistency!
                        let Some(version) =
                            assigned_shared_versions.get(&(*id, modified_initial_shared_version))
                        else {
                            panic!(
                                "Shared object version should have been assigned. key: {key:?}, \
                                 obj id: {id:?}, initial_shared_version: \
                                 {initial_shared_version:?}, assigned_shared_versions: \
                                 {assigned_shared_versions:?}",
                            )
                        };
                        InputKey::VersionedObject {
                            id: FullObjectID::new(*id, Some(*initial_shared_version)),
                            version: *version,
                        }
                    }

                    InputObjectKind::ImmOrOwnedObject(objref) => InputKey::VersionedObject {
                        id: FullObjectID::new(objref.0, None),
                        version: objref.1,
                    },
                })
            })
            .collect()
    }

    pub fn insert_finalized_transactions(
        &self,
        digests: &[TransactionDigest],
        index: CommitIndex,
    ) -> SomaResult {
        self.tables()?
            .executed_transactions_to_commit
            .write()
            .extend(digests.iter().map(|digest| (*digest, index)));
        trace!("Transactions {digests:?} finalized at commit {index}");

        // Notify all readers that the transactions have been finalized as part of a checkpoint execution.
        for digest in digests {
            self.executed_transactions_to_commit_notify_read
                .notify(digest, &index);
        }

        Ok(())
    }

    // Converts transaction keys to digests, waiting for digests to become available for any
    // non-digest keys.
    pub async fn notify_read_executed_digests(
        &self,
        keys: &[TransactionKey],
    ) -> SomaResult<Vec<TransactionDigest>> {
        // TODO: is this function necessary? in resolve_commit_transactions before create_commit

        // TODO: decide if TransactionKey will ever not be a transaction digest, i.e. a randomnous round
        // let non_digest_keys: Vec<_> = keys
        //     .iter()
        //     .filter_map(|key| {
        //         if matches!(key, TransactionKey::Digest(_)) {
        //             None
        //         } else {
        //             Some(*key)
        //         }
        //     })
        //     .collect();

        // let registrations = self
        //     .executed_digests_notify_read
        //     .register_all(&non_digest_keys);
        // let executed_digests = self
        //     .tables()?
        //     .transaction_key_to_digest
        //     .multi_get(&non_digest_keys)?;
        // let futures = executed_digests
        //     .into_iter()
        //     .zip(registrations)
        //     .map(|(d, r)| match d {
        //         // Note that Some() clause also drops registration that is already fulfilled
        //         Some(ready) => Either::Left(futures::future::ready(ready)),
        //         None => Either::Right(r),
        //     });
        // let mut results = VecDeque::from(join_all(futures).await);

        Ok(keys
            .iter()
            .filter_map(|key| {
                if let TransactionKey::Digest(digest) = key {
                    Some(*digest)
                } else {
                    None
                }
            })
            .collect())
    }

    pub fn set_next_epoch_state(
        &self,
        system_state: SystemState,
        epoch_digest: ECMHLiveObjectSetDigest,
    ) {
        if self.next_epoch_state.read().is_none() {
            *self.next_epoch_state.write() = Some((system_state, epoch_digest));
        }
    }

    // For each key in objects_to_init, return the next version for that key as recorded in the
    // next_shared_object_versions table.
    //
    // If any keys are missing, then we need to initialize the table. We first check if a previous
    // version of that object has been written. If so, then the object was written in a previous
    // epoch, and we initialize next_shared_object_versions to that value. If no version of the
    // object has yet been written, we initialize the object to the initial version recorded in the
    // certificate (which is a function of the lamport version computation of the transaction that
    // created the shared object originally - which transaction may not yet have been executed on
    // this node).
    //
    // Because all paths that assign shared versions for a shared object transaction call this
    // function, it is impossible for parent_sync to be updated before this function completes
    // successfully for each affected object id.
    pub(crate) fn get_or_init_next_object_versions(
        &self,
        objects_to_init: &[ConsensusObjectSequenceKey],
        cache_reader: &dyn ObjectCacheRead,
    ) -> SomaResult<HashMap<ConsensusObjectSequenceKey, Version>> {
        // get_or_init_next_object_versions can be called
        // from consensus or checkpoint executor,
        // so we need to protect version assignment with a critical section
        let _locks = self
            .version_assignment_mutex_table
            .acquire_locks(objects_to_init.iter().map(|(id, _)| *id));
        let tables = self.tables()?;

        let next_versions = self
            .consensus_quarantine
            .read()
            .get_next_shared_object_versions(self.epoch_start_config(), &tables, objects_to_init)?;

        let uninitialized_objects: Vec<ConsensusObjectSequenceKey> = next_versions
            .iter()
            .zip(objects_to_init)
            .filter_map(|(next_version, id_and_version)| match next_version {
                None => Some(*id_and_version),
                Some(_) => None,
            })
            .collect();

        // The common case is that there are no uninitialized versions - this early return will
        // happen every time except the first time an object is used in an epoch.
        if uninitialized_objects.is_empty() {
            // unwrap ok - we already verified that next_versions is not missing any keys.
            return Ok(izip!(
                objects_to_init.iter().cloned(),
                next_versions.into_iter().map(|v| v.unwrap())
            )
            .collect());
        }

        let versions_to_write: Vec<_> = uninitialized_objects
            .iter()
            .map(|(id, initial_version)| {
                match cache_reader.get_object(id) {
                    Ok(Some(obj)) => {
                        let obj_version = obj.version();

                        // For any object that exists in the cache, we need to ensure the
                        // next version is strictly greater than the current version,
                        // regardless of what the initial_version is
                        let next_version = if obj_version >= *initial_version {
                            // Object already has a version >= initial_version,
                            // we must use a version higher than the current one
                            obj_version.next()
                        } else {
                            // Current version is less than initial_version,
                            // safe to use the initial version
                            *initial_version
                        };

                        debug!(
                            "Assigning version for object {:?}: current={:?}, initial={:?}, \
                             assigned={:?}",
                            id, obj_version, initial_version, next_version
                        );

                        ((*id, *initial_version), next_version)
                    }
                    _ => {
                        // Object not found in cache, use initial version
                        ((*id, *initial_version), *initial_version)
                    }
                }
            })
            .collect();

        let ret = izip!(objects_to_init.iter().cloned(), next_versions.into_iter(),)
            // take all the previously initialized versions
            .filter_map(|(key, next_version)| next_version.map(|v| (key, v)))
            // add all the versions we're going to write
            .chain(versions_to_write.iter().cloned())
            .collect();

        debug!(
            ?versions_to_write,
            "initializing next_shared_object_versions"
        );

        versions_to_write.iter().for_each(|(k, v)| {
            tables
                .next_shared_object_versions
                .write()
                .insert(k.clone(), v.clone());
        });

        Ok(ret)
    }

    pub fn get_assigned_shared_object_versions(
        &self,
        key: &TransactionKey,
    ) -> Option<Vec<(ConsensusObjectSequenceKey, Version)>> {
        self.consensus_output_cache
            .get_assigned_shared_object_versions(key)
    }

    fn set_assigned_shared_object_versions(&self, versions: AssignedTxAndVersions) {
        self.consensus_output_cache
            .insert_shared_object_assignments(&versions);
    }

    /// Given list of certificates, assign versions for all shared objects used in them.
    /// We start with the current next_shared_object_versions table for each object, and build
    /// up the versions based on the dependencies of each certificate.
    /// However, in the end we do not update the next_shared_object_versions table, which keeps
    /// this function idempotent. We should call this function when we are assigning shared object
    /// versions outside of consensus and do not want to taint the next_shared_object_versions table.
    pub fn assign_shared_object_versions_idempotent(
        &self,
        cache_reader: &dyn ObjectCacheRead,
        certificates: &[VerifiedExecutableTransaction],
    ) -> SomaResult {
        let mut cert_versions_to_assign = Vec::new();

        for cert in certificates {
            let tx_key = cert.key();

            // Check if this is an epoch change transaction
            let is_epoch_change = cert.data().transaction_data().kind.is_epoch_change();

            // Check if this transaction requires shared object versioning
            let requires_versioning = cert.contains_shared_object();

            // Always reassign versions for epoch change transactions to ensure freshness
            // Otherwise, only assign if not already assigned
            if requires_versioning
                && (is_epoch_change || self.get_assigned_shared_object_versions(&tx_key).is_none())
            {
                info!(
                    tx_digest = ?cert.digest(),
                    is_epoch_change = is_epoch_change,
                    "Preparing to assign shared object versions"
                );
                cert_versions_to_assign.push(cert.clone());
            } else if let Some(assigned_versions) =
                self.get_assigned_shared_object_versions(&tx_key)
            {
                debug!(
                    tx_digest = ?cert.digest(),
                    ?assigned_versions,
                    is_epoch_change = is_epoch_change,
                    "Using previously assigned versions for transaction"
                );
            }
        }

        // If all transactions already have assigned versions, we're done
        if cert_versions_to_assign.is_empty() {
            return Ok(());
        }

        // Only compute and assign versions for transactions that need them
        let assigned_versions = SharedObjVerManager::assign_versions_from_consensus(
            self,
            cache_reader,
            &cert_versions_to_assign,
            &BTreeMap::new(),
        )?
        .assigned_versions;

        info!(
            assigned_transactions = cert_versions_to_assign.len(),
            "Assigning shared object versions idempotent"
        );

        // For detailed debugging in critical environments
        for (key, versions) in &assigned_versions {
            debug!(?key, ?versions, "Assigned versions for transaction");
        }

        self.set_assigned_shared_object_versions(assigned_versions);
        Ok(())
    }

    pub fn assign_shared_object_versions_state_sync(
        &self,
        cache_reader: &dyn ObjectCacheRead,
        certificates: &[VerifiedExecutableTransaction],
    ) -> SomaResult {
        let mut cert_versions_to_assign = Vec::new();

        // Step 1: Filter out certificates that already have version assignments
        for cert in certificates {
            let tx_key = cert.key();

            if cert.contains_shared_object() {
                // Check if this transaction already has versions assigned
                let already_assigned = self.get_assigned_shared_object_versions(&tx_key).is_some();

                if !already_assigned {
                    info!(
                        tx_digest = ?cert.digest(),
                        "Adding certificate for shared object version assignment"
                    );
                    cert_versions_to_assign.push(cert.clone());
                } else {
                    // Important: Skip already assigned certificates
                    debug!(
                        tx_digest = ?cert.digest(),
                        "Certificate already has assigned shared object versions"
                    );
                }
            }
        }

        // Step 2: If all transactions already have assigned versions, we're done
        if cert_versions_to_assign.is_empty() {
            return Ok(());
        }

        // Step 3: Only compute and assign versions for transactions that need them
        let ConsensusSharedObjVerAssignment {
            shared_input_next_versions,
            assigned_versions,
        } = SharedObjVerManager::assign_versions_from_consensus(
            self,
            cache_reader,
            &cert_versions_to_assign,
            &BTreeMap::new(),
        )?;

        // Step 4: Update next_shared_object_versions table with new versions
        {
            let tables = self.tables()?;
            let mut next_versions = tables.next_shared_object_versions.write();

            for (key, version) in &shared_input_next_versions {
                // Update only if the new version is higher than what's already there
                let should_update = match next_versions.get(key) {
                    Some(existing) => *version > *existing,
                    None => true,
                };

                if should_update {
                    debug!(?key, ?version, "Updating next_shared_object_versions");
                    next_versions.insert(*key, *version);
                }
            }
        }

        // Step 5: Store the assigned versions in the cache
        self.set_assigned_shared_object_versions(assigned_versions);

        Ok(())
    }
    /// Assign a sequence number for the shared objects of the input transaction based on the
    /// effects of that transaction.
    /// Used by full nodes who don't listen to consensus, and validators who catch up by state sync.
    // TODO: We should be able to pass in a vector of certs/effects and acquire them all at once.
    #[instrument(level = "trace", skip_all)]
    pub fn acquire_shared_version_assignments_from_effects(
        &self,
        certificate: &VerifiedExecutableTransaction,
        effects: &TransactionEffects,
        cache_reader: &dyn ObjectCacheRead,
    ) -> SomaResult {
        let versions = SharedObjVerManager::assign_versions_from_effects(
            &[(certificate, effects)],
            self,
            cache_reader,
        );
        info!(
            "Assigning shared object version from effects: {:?}",
            versions
        );
        self.set_assigned_shared_object_versions(versions);
        Ok(())
    }

    // Assigns shared object versions to transactions and updates the shared object version state.
    // Shared object versions in cancelled transactions are assigned to special versions that will
    // cause the transactions to be cancelled in execution engine.
    fn process_consensus_transaction_shared_object_versions(
        &self,
        cache_reader: &dyn ObjectCacheRead,
        transactions: &[VerifiedExecutableTransaction],
        cancelled_txns: &BTreeMap<TransactionDigest, CancelConsensusCertificateReason>,
        output: &mut ConsensusCommitOutput,
    ) -> SomaResult {
        let ConsensusSharedObjVerAssignment {
            shared_input_next_versions,
            assigned_versions,
        } = SharedObjVerManager::assign_versions_from_consensus(
            self,
            cache_reader,
            transactions,
            cancelled_txns,
        )?;

        self.consensus_output_cache
            .insert_shared_object_assignments(&assigned_versions);

        output.set_next_shared_object_versions(shared_input_next_versions);
        Ok(())
    }

    pub fn assign_shared_object_versions_for_tests(
        self: &Arc<Self>,
        cache_reader: &dyn ObjectCacheRead,
        transactions: &[VerifiedExecutableTransaction],
    ) -> SomaResult {
        let mut output = ConsensusCommitOutput::new();
        self.process_consensus_transaction_shared_object_versions(
            cache_reader,
            transactions,
            &BTreeMap::new(),
            &mut output,
        )?;

        output.write_to_batch(self)?;

        Ok(())
    }
}

impl EndOfEpochAPI for AuthorityPerEpochStore {
    fn get_next_epoch_state(
        &self,
    ) -> Option<(ValidatorSet, EncoderCommittee, ECMHLiveObjectSetDigest, u64)> {
        self.next_epoch_state
            .read()
            .as_ref()
            .map(|(state, digest)| {
                let enc = state.get_current_epoch_encoder_committee();
                info!("NEW ENCODER COMMITTEE: {:?}", enc);
                (
                    ValidatorSet(
                        state
                            .get_current_epoch_committee()
                            .validators()
                            .iter()
                            .map(|(authority_name, (voting_power, network_metadata))| {
                                (
                                    authority_name.clone(),
                                    *voting_power,
                                    network_metadata.clone(),
                                )
                            })
                            .collect(),
                    ),
                    state.get_current_epoch_encoder_committee(),
                    digest.clone(),
                    state.epoch_start_timestamp_ms,
                )
            })
    }
}
