use arc_swap::ArcSwapOption;
use enum_dispatch::enum_dispatch;
use futures::{
    future::select,
    future::{join_all, Either},
    FutureExt,
};
use itertools::izip;
use nonempty::NonEmpty;
use parking_lot::{Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque},
    future::Future,
    ops::{Bound, Deref},
    path::{Path, PathBuf},
    sync::{Arc, Weak},
    time::{Duration, Instant},
};
use store::{rocks::DBBatch, rocksdb::Options};
use store::{
    rocks::{default_db_options, read_size_from_env, DBMap, DBOptions, ReadWriteOptions},
    DBMapUtils, Map as _,
};
use super::consensus_tx_status_cache::{ConsensusTxStatus, ConsensusTxStatusCache};
use super::submitted_transaction_cache::{
    SubmittedTransactionCache, 
};
use super::transaction_reject_reason_cache::TransactionRejectReasonCache;
use crate::{signature_verifier::SignatureVerifier};
use tracing::{debug, info, instrument, trace, warn};
use types::{
    SYSTEM_STATE_OBJECT_ID, base::{
        AuthorityName, ConciseableName, ConsensusObjectSequenceKey, FullObjectID, Round,
        SomaAddress,
    }, checkpoints::{
        CheckpointContents, CheckpointSequenceNumber, CheckpointSignatureMessage,
        CheckpointSummary, ECMHLiveObjectSetDigest, GlobalStateHash,
    }, committee::{Authority, Committee, EpochId, NetworkingCommittee}, consensus::{
        ConsensusCommitPrologue, ConsensusPosition, ConsensusTransaction, ConsensusTransactionKey, ConsensusTransactionKind, EndOfEpochAPI, block::BlockRef, validator_set::ValidatorSet
    }, crypto::{
        AuthorityPublicKeyBytes, AuthoritySignInfo, AuthorityStrongQuorumSignInfo,
        GenericSignature, Signer,
    }, digests::{TransactionDigest, TransactionEffectsDigest}, effects::{
        self, ExecutionFailureStatus, ExecutionStatus, TransactionEffects, UnchangedSharedKind, object_change::{EffectsObjectChange, IDOperation, ObjectIn, ObjectOut}
    }, encoder_committee::EncoderCommittee, envelope::TrustedEnvelope, error::{ExecutionError, SomaError, SomaResult}, finality::{ConsensusFinality, VerifiedSignedConsensusFinality}, mutex_table::{MutexGuard, MutexTable}, object::{Object, ObjectData, ObjectID, ObjectRef, ObjectType, Owner, Version}, storage::{InputKey, object_store::ObjectStore}, system_state::{
        self, SystemState, SystemStateTrait, epoch_start::{EpochStartSystemState, EpochStartSystemStateTrait}, get_system_state
    }, temporary_store::{InnerTemporaryStore, SharedInput, TemporaryStore}, transaction::{
        self, CertifiedTransaction, InputObjectKind, InputObjects, ObjectReadResult,
        ObjectReadResultKind, SenderSignedData, Transaction, TransactionKey, TransactionKind,
        TrustedExecutableTransaction, VerifiedCertificate, VerifiedExecutableTransaction,
        VerifiedSignedTransaction, VerifiedTransaction,
    }, transaction_outputs::WrittenObjects
};
use utils::{notify_once::NotifyOnce, notify_read::NotifyRead};
use protocol_config::{Chain, ProtocolConfig, ProtocolVersion};
use crate::{
    authority_store::LockDetails,
    authority_store_tables::ENV_VAR_LOCKS_BLOCK_CACHE_SIZE,
    cache::{cache_types::CacheResult, ObjectCacheRead},
    checkpoints::{BuilderCheckpointSummary, CheckpointHeight, PendingCheckpoint},
    consensus_handler::{
        ConsensusCommitInfo, SequencedConsensusTransaction, SequencedConsensusTransactionKey,
        SequencedConsensusTransactionKind, VerifiedSequencedConsensusTransaction,
    },
    consensus_quarantine::{
        ConsensusCommitOutput, ConsensusOutputCache, ConsensusOutputQuarantine,
    },
    fallback_fetch::do_fallback_lookup,
    reconfiguration::ReconfigState,
    shared_obj_version_manager::{
        AssignedTxAndVersions, AssignedVersions, ConsensusSharedObjVerAssignment, Schedulable,
        SharedObjVerManager,
    },
    stake_aggregator::StakeAggregator,
    start_epoch::{EpochStartConfigTrait, EpochStartConfiguration},
};

pub(crate) const LAST_CONSENSUS_STATS_ADDR: u64 = 0;
pub(crate) const RECONFIG_STATE_INDEX: u64 = 0;
const OVERRIDE_PROTOCOL_UPGRADE_BUFFER_STAKE_INDEX: u64 = 0;
pub const EPOCH_DB_PREFIX: &str = "epoch_";

// CertLockGuard and CertTxGuard are functionally identical right now, but we retain a distinction
// anyway. If we need to support distributed object storage, having this distinction will be
// useful, as we will most likely have to re-implement a retry / write-ahead-log at that point.
pub struct CertLockGuard(#[allow(unused)] MutexGuard);
pub struct CertTxGuard(#[allow(unused)] CertLockGuard);

impl CertTxGuard {
    pub fn release(self) {}
    pub fn commit_tx(self) {}
    pub fn as_lock_guard(&self) -> &CertLockGuard {
        &self.0
    }
}

impl CertLockGuard {
    pub fn dummy_for_tests() -> Self {
        let lock = Arc::new(parking_lot::Mutex::new(()));
        Self(lock.try_lock_arc().unwrap())
    }
}

pub enum CancelConsensusCertificateReason {
    CongestionOnObjects(Vec<ObjectID>),
}

pub enum ConsensusCertificateResult {
    /// The consensus message was ignored (e.g. because it has already been processed).
    Ignored,
    /// An executable transaction (can be a user tx or a system tx)
    SuiTransaction(VerifiedExecutableTransaction),
    // The transaction should be re-processed at a future commit, specified by the DeferralKey
    // Deferred(DeferralKey),
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

/// ConsensusStats is versioned because we may iterate on the struct, and it is
/// stored on disk.
#[enum_dispatch]
pub trait ConsensusStatsAPI {
    fn is_initialized(&self) -> bool;

    fn get_num_messages(&self, authority: usize) -> u64;
    fn inc_num_messages(&mut self, authority: usize) -> u64;

    fn get_num_user_transactions(&self, authority: usize) -> u64;
    fn inc_num_user_transactions(&mut self, authority: usize) -> u64;
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
#[enum_dispatch(ConsensusStatsAPI)]
pub enum ConsensusStats {
    V1(ConsensusStatsV1),
}

impl ConsensusStats {
    pub fn new(size: usize) -> Self {
        Self::V1(ConsensusStatsV1 {
            num_messages: vec![0; size],
            num_user_transactions: vec![0; size],
        })
    }
}

impl Default for ConsensusStats {
    fn default() -> Self {
        Self::new(0)
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct ConsensusStatsV1 {
    pub num_messages: Vec<u64>,
    pub num_user_transactions: Vec<u64>,
}

impl ConsensusStatsAPI for ConsensusStatsV1 {
    fn is_initialized(&self) -> bool {
        !self.num_messages.is_empty()
    }

    fn get_num_messages(&self, authority: usize) -> u64 {
        self.num_messages[authority]
    }

    fn inc_num_messages(&mut self, authority: usize) -> u64 {
        self.num_messages[authority] += 1;
        self.num_messages[authority]
    }

    fn get_num_user_transactions(&self, authority: usize) -> u64 {
        self.num_user_transactions[authority]
    }

    fn inc_num_user_transactions(&mut self, authority: usize) -> u64 {
        self.num_user_transactions[authority] += 1;
        self.num_user_transactions[authority]
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq, Eq, Copy)]
pub struct ExecutionIndices {
    /// The round number of the last committed leader.
    pub last_committed_round: u64,
    /// The index of the last sub-DAG that was executed (either fully or partially).
    pub sub_dag_index: u64,
    /// The index of the last transaction was executed (used for crash-recovery).
    pub transaction_index: u64,
}

impl Ord for ExecutionIndices {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (
            self.last_committed_round,
            self.sub_dag_index,
            self.transaction_index,
        )
            .cmp(&(
                other.last_committed_round,
                other.sub_dag_index,
                other.transaction_index,
            ))
    }
}

impl PartialOrd for ExecutionIndices {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq, Eq)]
pub struct ExecutionIndicesWithStats {
    pub index: ExecutionIndices,
    // Hash is always 0 and kept for compatibility only.
    pub hash: u64,
    pub stats: ConsensusStats,
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

    /// Holds the outputs of both consensus handler and checkpoint builder in memory
    /// until they are proven not to have forked by a certified checkpoint.
    pub(crate) consensus_quarantine: RwLock<ConsensusOutputQuarantine>,
    /// Holds variouis data from consensus_quarantine in a more easily accessible form.
    pub(crate) consensus_output_cache: ConsensusOutputCache,

    protocol_config: ProtocolConfig,

    // needed for re-opening epoch db.
    parent_path: PathBuf,
    db_options: Option<Options>,

    /// In-memory cache of the content from the reconfig_state db table.
    reconfig_state_mem: RwLock<ReconfigState>,
    consensus_notify_read: NotifyRead<SequencedConsensusTransactionKey, ()>,

      // Subscribers will get notified when a transaction is executed via checkpoint execution.
    executed_transactions_to_checkpoint_notify_read:
        NotifyRead<TransactionDigest, CheckpointSequenceNumber>,

    /// Batch verifier for certificates - also caches certificates and tx sigs that are known to have
    /// valid signatures. Lives in per-epoch store because the caching/batching is only valid
    /// within for certs within the current epoch.
    pub(crate) signature_verifier: SignatureVerifier,

    pub(crate) checkpoint_state_notify_read: NotifyRead<CheckpointSequenceNumber, GlobalStateHash>,

    running_root_notify_read: NotifyRead<CheckpointSequenceNumber, GlobalStateHash>,

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
    pub(crate) end_of_publish: Mutex<StakeAggregator<(), true>>,
    /// Pending certificates that are waiting to be sequenced by the consensus.
    /// This is an in-memory 'index' of a AuthorityPerEpochTables::pending_consensus_transactions.
    /// We need to keep track of those in order to know when to send EndOfPublish message.
    /// Lock ordering: this is a 'leaf' lock, no other locks should be acquired in the scope of this lock
    /// In particular, this lock is always acquired after taking read or write lock on reconfig state
    pending_consensus_certificates: RwLock<HashSet<TransactionDigest>>,

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
    // chain_identifier: ChainIdentifier,
    pub(crate) consensus_tx_status_cache: Option<ConsensusTxStatusCache>,

    /// A cache that maintains the reject vote reason for a transaction.
    pub(crate) tx_reject_reason_cache: Option<TransactionRejectReasonCache>,

    /// A cache that tracks submitted transactions to prevent DoS through excessive resubmissions.
    pub(crate) submitted_transaction_cache: SubmittedTransactionCache,
}


#[derive(DBMapUtils)]
pub struct AuthorityEpochTables {
    /// This is map between the transaction digest and transactions found in the `transaction_lock`.
    #[default_options_override_fn = "signed_transactions_table_default_config"]
    signed_transactions:
        DBMap<TransactionDigest, TrustedEnvelope<SenderSignedData, AuthoritySignInfo>>,

    /// Map from ObjectRef to transaction locking that object
    #[default_options_override_fn = "owned_object_transaction_locks_table_default_config"]
    object_locked_transactions: DBMap<ObjectRef, LockDetails>,

    /// Signatures over transaction effects that we have signed and returned to users.
    /// We store this to avoid re-signing the same effects twice.
    /// Note that this may contain signatures for effects from previous epochs, in the case
    /// that a user requests a signature for effects from a previous epoch. However, the
    /// signature is still epoch-specific and so is stored in the epoch store.
    effects_signatures: DBMap<TransactionDigest, AuthoritySignInfo>,

    /// When we sign a TransactionEffects, we must record the digest of the effects in order
    /// to detect and prevent equivocation when re-executing a transaction that may not have been
    /// committed to disk.
    /// Entries are removed from this table after the transaction in question has been committed
    /// to disk.
    signed_effects_digests: DBMap<TransactionDigest, TransactionEffectsDigest>,

    /// Signatures of transaction certificates that are executed locally.
    transaction_cert_signatures: DBMap<TransactionDigest, AuthorityStrongQuorumSignInfo>,

    /// Next available shared object versions for each shared object.
    pub(crate) next_shared_object_versions: DBMap<ConsensusObjectSequenceKey, Version>,

    /// Track which transactions have been processed in handle_consensus_transaction. We must be
    /// sure to advance next_shared_object_versions exactly once for each transaction we receive from
    /// consensus. But, we may also be processing transactions from checkpoints, so we need to
    /// track this state separately.
    ///
    /// Entries in this table can be garbage collected whenever we can prove that we won't receive
    /// another handle_consensus_transaction call for the given digest. This probably means at
    /// epoch change.
    pub(crate) consensus_message_processed: DBMap<SequencedConsensusTransactionKey, bool>,

    /// Map stores pending transactions that this authority submitted to consensus
    #[default_options_override_fn = "pending_consensus_transactions_table_default_config"]
    pending_consensus_transactions: DBMap<ConsensusTransactionKey, ConsensusTransaction>,

    /// The following table is used to store a single value (the corresponding key is a constant). The value
    /// represents the index of the latest consensus message this authority processed, running hash of
    /// transactions, and accumulated stats of consensus output.
    /// This field is written by a single process (consensus handler).
    pub(crate) last_consensus_stats: DBMap<u64, ExecutionIndicesWithStats>,

    /// This table contains current reconfiguration state for validator for current epoch
    pub(crate) reconfig_state: DBMap<u64, ReconfigState>,

    /// Validators that have sent EndOfPublish message in this epoch
    pub(crate) end_of_publish: DBMap<AuthorityName, ()>,

    /// Checkpoint builder maintains internal list of transactions it included in checkpoints here
    pub(crate) builder_digest_to_checkpoint: DBMap<TransactionDigest, CheckpointSequenceNumber>,

    /// Maps non-digest TransactionKeys to the corresponding digest after execution, for use
    /// by checkpoint builder.
    transaction_key_to_digest: DBMap<TransactionKey, TransactionDigest>,

    /// Stores pending signatures
    /// The key in this table is checkpoint sequence number and an arbitrary integer
    pub(crate) pending_checkpoint_signatures:
        DBMap<(CheckpointSequenceNumber, u64), CheckpointSignatureMessage>,

    /// Maps sequence number to checkpoint summary, used by CheckpointBuilder to build checkpoint within epoch
    pub(crate) builder_checkpoint_summary:
        DBMap<CheckpointSequenceNumber, BuilderCheckpointSummary>,

    // Maps checkpoint sequence number to an accumulator with accumulated state
    // only for the checkpoint that the key references. Append-only, i.e.,
    // the accumulator is complete wrt the checkpoint
    pub state_hash_by_checkpoint: DBMap<CheckpointSequenceNumber, GlobalStateHash>,

    /// Maps checkpoint sequence number to the running (non-finalized) root state
    /// accumulator up th that checkpoint. This should be equivalent to the root
    /// state hash at end of epoch. Guaranteed to be written to in checkpoint
    /// sequence number order.
    #[rename = "running_root_accumulators"]
    pub running_root_state_hash: DBMap<CheckpointSequenceNumber, GlobalStateHash>,

    /// When transaction is executed via checkpoint executor, we store association here
    pub(crate) executed_transactions_to_checkpoint:
        DBMap<TransactionDigest, CheckpointSequenceNumber>,

    // TODO: potentially remove this and use checkpoints instead
    /// Signed consensus finality for EmbedData transactions
    pub(crate) consensus_finalities: DBMap<TransactionDigest, BlockRef>,
}

fn signed_transactions_table_default_config() -> DBOptions {
    default_db_options()
        .optimize_for_write_throughput()
        .optimize_for_large_values_no_scan(1 << 10)
}

fn owned_object_transaction_locks_table_default_config() -> DBOptions {
    DBOptions {
        options: default_db_options()
            .optimize_for_write_throughput()
            .optimize_for_read(read_size_from_env(ENV_VAR_LOCKS_BLOCK_CACHE_SIZE).unwrap_or(1024))
            .options,
        rw_options: ReadWriteOptions::default().set_ignore_range_deletions(false),
    }
}

fn pending_consensus_transactions_table_default_config() -> DBOptions {
    default_db_options()
        .optimize_for_write_throughput()
        .optimize_for_large_values_no_scan(1 << 10)
}

impl AuthorityEpochTables {
    pub fn open(epoch: EpochId, parent_path: &Path, db_options: Option<Options>) -> Self {
        Self::open_tables_read_write(Self::path(epoch, parent_path), db_options, None)
    }

    pub fn open_readonly(epoch: EpochId, parent_path: &Path) -> AuthorityEpochTablesReadOnly {
        Self::get_read_only_handle(Self::path(epoch, parent_path), None, None)
    }

    pub fn path(epoch: EpochId, parent_path: &Path) -> PathBuf {
        parent_path.join(format!("{}{}", EPOCH_DB_PREFIX, epoch))
    }

    fn load_reconfig_state(&self) -> SomaResult<ReconfigState> {
        let state = self
            .reconfig_state
            .get(&RECONFIG_STATE_INDEX)?
            .unwrap_or_default();
        Ok(state)
    }

    pub fn get_all_pending_consensus_transactions(&self) -> SomaResult<Vec<ConsensusTransaction>> {
        Ok(self
            .pending_consensus_transactions
            .safe_iter()
            .map(|item| item.map(|(_k, v)| v))
            .collect::<Result<Vec<_>, _>>()?)
    }

    pub fn get_last_consensus_index(&self) -> SomaResult<Option<ExecutionIndices>> {
        Ok(self
            .last_consensus_stats
            .get(&LAST_CONSENSUS_STATS_ADDR)?
            .map(|s| s.index))
    }

     pub fn get_last_consensus_stats(&self) -> SomaResult<Option<ExecutionIndicesWithStats>> {
        Ok(self.last_consensus_stats.get(&LAST_CONSENSUS_STATS_ADDR)?)
    }

    pub fn get_locked_transaction(&self, obj_ref: &ObjectRef) -> SomaResult<Option<LockDetails>> {
        Ok(self
            .object_locked_transactions
            .get(obj_ref)?
        )
    }

    pub fn multi_get_locked_transactions(
        &self,
        owned_input_objects: &[ObjectRef],
    ) -> SomaResult<Vec<Option<LockDetails>>> {
        Ok(self
            .object_locked_transactions
            .multi_get(owned_input_objects)?
            .into_iter()
            
            .collect())
    }

   pub fn write_transaction_locks(
        &self,
        signed_transaction: Option<VerifiedSignedTransaction>,
        locks_to_write: impl Iterator<Item = (ObjectRef, LockDetails)>,
    ) -> SomaResult {
        let mut batch = self.object_locked_transactions.batch();
        batch.insert_batch(
            &self.object_locked_transactions,
            locks_to_write.map(|(obj_ref, lock)| (obj_ref, lock)),
        )?;
        if let Some(signed_transaction) = signed_transaction {
            batch.insert_batch(
                &self.signed_transactions,
                std::iter::once((
                    *signed_transaction.digest(),
                    signed_transaction.serializable_ref(),
                )),
            )?;
        }
        batch.write()?;
        Ok(())
    }
}

pub(crate) const MUTEX_TABLE_SIZE: usize = 1024;

impl AuthorityPerEpochStore {
    #[instrument(name = "AuthorityPerEpochStore::new", level = "error", skip_all, fields(epoch = committee.epoch))]
    pub fn new(
        name: AuthorityName,
        committee: Arc<Committee>,
        parent_path: &Path,
        db_options: Option<Options>,
        epoch_start_configuration: EpochStartConfiguration,
        //  TODO: chain: (ChainIdentifier, Chain),
        highest_executed_checkpoint: CheckpointSequenceNumber,
    ) -> SomaResult<Arc<Self>> {
        let current_time = Instant::now();
        let epoch_id = committee.epoch;

        let tables = AuthorityEpochTables::open(epoch_id, parent_path, db_options.clone());
        let end_of_publish =
            StakeAggregator::from_iter(committee.clone(), tables.end_of_publish.safe_iter())?;
        let reconfig_state = tables
            .load_reconfig_state()
            .expect("Load reconfig state at initialization cannot fail");

        let epoch_alive_notify = NotifyOnce::new();
        let pending_consensus_transactions = tables.get_all_pending_consensus_transactions()?;
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
  
        let protocol_version = ProtocolVersion::MIN;
        let chain = Chain::Mainnet;
        // TODO: derive protocol version from epoch start config
        // let protocol_version = epoch_start_configuration
        //     .epoch_start_state()
        //     .protocol_version();

        // let chain_from_id = chain.0.chain();
        // if chain_from_id == Chain::Mainnet || chain_from_id == Chain::Testnet {
        //     assert_eq!(
        //         chain_from_id, chain.1,
        //         "cannot override chain on production networks!"
        //     );
        // }
        // info!(
        //     "initializing epoch store from chain id {:?} to chain id {:?}",
        //     chain_from_id, chain.1
        // );

        let protocol_config = ProtocolConfig::get_for_version(protocol_version, chain);

        let signature_verifier = SignatureVerifier::new(committee.clone());

        let consensus_output_cache = ConsensusOutputCache::new(&epoch_start_configuration, &tables);

        let consensus_tx_status_cache =  Some(ConsensusTxStatusCache::new(protocol_config.gc_depth()));

        let tx_reject_reason_cache = Some(TransactionRejectReasonCache::new(None, epoch_id));

        let submitted_transaction_cache =
            SubmittedTransactionCache::new(None);

        let s = Arc::new(Self {
            name,
            committee,
            protocol_config,
            tables: ArcSwapOption::new(Some(Arc::new(tables))),
            consensus_output_cache,
            consensus_quarantine: RwLock::new(ConsensusOutputQuarantine::new(
                highest_executed_checkpoint,
            )),
            parent_path: parent_path.to_path_buf(),
            db_options,
            reconfig_state_mem: RwLock::new(reconfig_state),
            epoch_alive_notify,
            executed_transactions_to_checkpoint_notify_read: NotifyRead::new(),
            user_certs_closed_notify: NotifyOnce::new(),
            epoch_alive: tokio::sync::RwLock::new(true),
            consensus_notify_read: NotifyRead::new(),
            signature_verifier,
            executed_digests_notify_read: NotifyRead::new(),
            running_root_notify_read: NotifyRead::new(),
            end_of_publish: Mutex::new(end_of_publish),
            pending_consensus_certificates: RwLock::new(pending_consensus_certificates),
            mutex_table: MutexTable::new(MUTEX_TABLE_SIZE),
            version_assignment_mutex_table: MutexTable::new(MUTEX_TABLE_SIZE),
            epoch_open_time: current_time,
            epoch_close_time: Default::default(),
            epoch_start_configuration,
            checkpoint_state_notify_read: NotifyRead::new(),
            consensus_tx_status_cache,
            tx_reject_reason_cache,
            submitted_transaction_cache,
        });
        Ok(s)
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

    pub fn get_parent_path(&self) -> PathBuf {
        self.parent_path.clone()
    }

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
        previous_epoch_last_checkpoint: CheckpointSequenceNumber,
    ) -> SomaResult<Arc<Self>> {
        assert_eq!(self.epoch() + 1, new_committee.epoch);
        Self::new(
            name,
            Arc::new(new_committee),
            &self.parent_path,
            self.db_options.clone(),
            epoch_start_configuration,
            previous_epoch_last_checkpoint,
        )
    }

    pub fn new_at_next_epoch_for_testing(
        &self,
        previous_epoch_last_checkpoint: CheckpointSequenceNumber,
    ) -> Arc<Self> {
        let next_epoch = self.epoch() + 1;
        let next_committee = Committee::new(
            next_epoch,
            self.committee.voting_rights.iter().cloned().collect(),
            self.committee.authorities.clone().into_iter().collect(),
        );
        self.new_at_next_epoch(
            self.name,
            next_committee,
            self.epoch_start_configuration
                .new_at_next_epoch_for_testing(),
            previous_epoch_last_checkpoint,
        )
        .expect("failed to create new authority per epoch store")
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

    pub fn get_state_hash_for_checkpoint(
        &self,
        checkpoint: &CheckpointSequenceNumber,
    ) -> SomaResult<Option<GlobalStateHash>> {
        Ok(self
            .tables()?
            .state_hash_by_checkpoint
            .get(checkpoint)
            .expect("db error"))
    }

    pub fn insert_state_hash_for_checkpoint(
        &self,
        checkpoint: &CheckpointSequenceNumber,
        accumulator: &GlobalStateHash,
    ) -> SomaResult {
        self.tables()?
            .state_hash_by_checkpoint
            .insert(checkpoint, accumulator)
            .expect("db error");
        Ok(())
    }

    pub fn get_running_root_state_hash(
        &self,
        checkpoint: CheckpointSequenceNumber,
    ) -> SomaResult<Option<GlobalStateHash>> {
        Ok(self
            .tables()?
            .running_root_state_hash
            .get(&checkpoint)
            .expect("db error"))
    }

    pub fn get_highest_running_root_state_hash(
        &self,
    ) -> SomaResult<Option<(CheckpointSequenceNumber, GlobalStateHash)>> {
        Ok(self
            .tables()?
            .running_root_state_hash
            .reversed_safe_iter_with_bounds(None, None)?
            .next()
            .transpose()?)
    }

    pub fn insert_running_root_state_hash(
        &self,
        checkpoint: &CheckpointSequenceNumber,
        hash: &GlobalStateHash,
    ) -> SomaResult {
        self.tables()?
            .running_root_state_hash
            .insert(checkpoint, hash)?;
        self.running_root_notify_read.notify(checkpoint, hash);

        Ok(())
    }

    pub fn clear_state_hashes_after_checkpoint(
        &self,
        last_committed_checkpoint: CheckpointSequenceNumber,
    ) -> SomaResult {
        let tables = self.tables()?;

        let mut keys_to_remove = Vec::new();
        for kv in tables
            .running_root_state_hash
            .safe_iter_with_bounds(Some(last_committed_checkpoint + 1), None)
        {
            let (checkpoint_seq, _) = kv?;
            if checkpoint_seq > last_committed_checkpoint {
                keys_to_remove.push(checkpoint_seq);
            }
        }

        let mut checkpoint_keys_to_remove = Vec::new();
        for kv in tables
            .state_hash_by_checkpoint
            .safe_iter_with_bounds(Some(last_committed_checkpoint + 1), None)
        {
            let (checkpoint_seq, _) = kv?;
            if checkpoint_seq > last_committed_checkpoint {
                checkpoint_keys_to_remove.push(checkpoint_seq);
            }
        }

        if !keys_to_remove.is_empty() || !checkpoint_keys_to_remove.is_empty() {
            let mut batch = self.db_batch()?;
            if !keys_to_remove.is_empty() {
                batch
                    .delete_batch(&tables.running_root_state_hash, keys_to_remove.clone())
                    .expect("db error");
            }
            if !checkpoint_keys_to_remove.is_empty() {
                batch
                    .delete_batch(
                        &tables.state_hash_by_checkpoint,
                        checkpoint_keys_to_remove.clone(),
                    )
                    .expect("db error");
            }
            batch.write().expect("db error");
            for key in keys_to_remove {
                info!(
                    "Cleared running root state hash for checkpoint {} (after last committed checkpoint {})",
                    key, last_committed_checkpoint
                );
            }
            for key in checkpoint_keys_to_remove {
                info!(
                    "Cleared checkpoint state hash for checkpoint {} (after last committed checkpoint {})",
                    key, last_committed_checkpoint
                );
            }
        }

        Ok(())
    }

    pub fn store_reconfig_state(&self, new_state: &ReconfigState) -> SomaResult {
        self.tables()?
            .reconfig_state
            .insert(&RECONFIG_STATE_INDEX, new_state)?;
        Ok(())
    }

    pub fn insert_signed_transaction(&self, transaction: VerifiedSignedTransaction) -> SomaResult {
        Ok(self
            .tables()?
            .signed_transactions
            .insert(transaction.digest(), transaction.serializable_ref())?)
    }

    #[cfg(test)]
    pub fn delete_signed_transaction_for_test(&self, transaction: &TransactionDigest) {
        self.tables()
            .expect("test should not cross epoch boundary")
            .signed_transactions
            .remove(transaction)
            .unwrap();
    }

    #[cfg(test)]
    pub fn delete_object_locks_for_test(&self, objects: &[ObjectRef]) {
        for object in objects {
            self.tables()
                .expect("test should not cross epoch boundary")
                .object_locked_transactions
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
            .get(tx_digest)?
            .map(|t| t.into()))
    }

    #[instrument(level = "trace", skip_all)]
    pub fn insert_tx_cert_sig(
        &self,
        tx_digest: &TransactionDigest,
        cert_sig: &AuthorityStrongQuorumSignInfo,
    ) -> SomaResult {
        let tables = self.tables()?;
        Ok(tables
            .transaction_cert_signatures
            .insert(tx_digest, cert_sig)?)
    }

    /// Record that a transaction has been executed in the current epoch.
    /// Used by checkpoint builder to cull dependencies from previous epochs.
    #[instrument(level = "trace", skip_all)]
    pub fn insert_executed_in_epoch(&self, tx_digest: &TransactionDigest) {
        self.consensus_output_cache
            .insert_executed_in_epoch(*tx_digest);
    }

    /// Record a mapping from a transaction key (such as TransactionKey::RandomRound) to its digest.
    pub(crate) fn insert_tx_key(
        &self,
        tx_key: TransactionKey,
        tx_digest: TransactionDigest,
    ) -> SomaResult {
        if matches!(tx_key, TransactionKey::Digest(_)) {
            debug!("useless to insert a digest key");
            return Ok(());
        }

        let tables = self.tables()?;
        tables
            .transaction_key_to_digest
            .insert(&tx_key, &tx_digest)?;
        self.executed_digests_notify_read
            .notify(&tx_key, &tx_digest);
        Ok(())
    }

    pub fn tx_key_to_digest(&self, key: &TransactionKey) -> SomaResult<Option<TransactionDigest>> {
        let tables = self.tables()?;
        if let TransactionKey::Digest(digest) = key {
            Ok(Some(*digest))
        } else {
            Ok(tables.transaction_key_to_digest.get(key).expect("db error"))
        }
    }

    pub fn insert_effects_digest_and_signature(
        &self,
        tx_digest: &TransactionDigest,
        effects_digest: &TransactionEffectsDigest,
        effects_signature: &AuthoritySignInfo,
    ) -> SomaResult {
        let tables = self.tables()?;
        let mut batch = tables.effects_signatures.batch();
        batch.insert_batch(&tables.effects_signatures, [(tx_digest, effects_signature)])?;
        batch.insert_batch(
            &tables.signed_effects_digests,
            [(tx_digest, effects_digest)],
        )?;
        batch.write()?;
        Ok(())
    }

    pub fn transactions_executed_in_cur_epoch(
        &self,
        digests: &[TransactionDigest],
    ) -> SomaResult<Vec<bool>> {
        let tables = self.tables()?;
        Ok(do_fallback_lookup(
            digests,
            |digest| {
                if self
                    .consensus_output_cache
                    .executed_in_current_epoch(digest)
                {
                    CacheResult::Hit(true)
                } else {
                    CacheResult::Miss
                }
            },
            |digests| {
                tables
                    .executed_transactions_to_checkpoint
                    .multi_contains_keys(digests)
                    .expect("db error")
            },
        ))
    }

    pub fn get_effects_signature(
        &self,
        tx_digest: &TransactionDigest,
    ) -> SomaResult<Option<AuthoritySignInfo>> {
        let tables = self.tables()?;
        Ok(tables.effects_signatures.get(tx_digest)?)
    }

    pub fn get_signed_effects_digest(
        &self,
        tx_digest: &TransactionDigest,
    ) -> SomaResult<Option<TransactionEffectsDigest>> {
        let tables = self.tables()?;
        Ok(tables.signed_effects_digests.get(tx_digest)?)
    }

    pub fn get_transaction_cert_sig(
        &self,
        tx_digest: &TransactionDigest,
    ) -> SomaResult<Option<AuthorityStrongQuorumSignInfo>> {
        Ok(self.tables()?.transaction_cert_signatures.get(tx_digest)?)
    }

    /// Resolves InputObjectKinds into InputKeys. `assigned_versions` is used to map shared inputs
    /// to specific object versions.
    pub(crate) fn get_input_object_keys(
        &self,
        key: &TransactionKey,
        objects: &[InputObjectKind],
        assigned_versions: &AssignedVersions,
    ) -> BTreeSet<InputKey> {
        let assigned_shared_versions = assigned_versions
            .iter()
            .cloned()
            .collect::<BTreeMap<_, _>>();
        objects
            .iter()
            .map(|kind| {
                match kind {
                    InputObjectKind::SharedObject {
                        id,
                        initial_shared_version,
                        ..
                    } => {
                        // If we found assigned versions, but they are missing the assignment for
                        // this object, it indicates a serious inconsistency!
                        let Some(version) = assigned_shared_versions.get(&(*id, *initial_shared_version)) else {
                            panic!(
                                "Shared object version should have been assigned. key: {key:?}, \
                                obj id: {id:?}, initial_shared_version: {initial_shared_version:?}, \
                                assigned_shared_versions: {assigned_shared_versions:?}",
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
                }
            })
            .collect()
    }

    pub fn get_last_consensus_stats(&self) -> SomaResult<ExecutionIndicesWithStats> {
        assert!(
            self.consensus_quarantine.read().is_empty(),
            "get_last_consensus_stats should only be called at startup"
        );
        match self.tables()?.get_last_consensus_stats()? {
            Some(stats) => Ok(stats),
            None => {
                let indices = self
                    .tables()?
                    .get_last_consensus_index()
                    .map(|x| x.unwrap_or_default())?;
                Ok(ExecutionIndicesWithStats {
                    index: indices,
                    hash: 0, // unused
                    stats: ConsensusStats::default(),
                })
            }
        }
    }

    pub fn get_accumulators_in_checkpoint_range(
        &self,
        from_checkpoint: CheckpointSequenceNumber,
        to_checkpoint: CheckpointSequenceNumber,
    ) -> SomaResult<Vec<(CheckpointSequenceNumber, GlobalStateHash)>> {
        self.tables()?
            .state_hash_by_checkpoint
            .safe_range_iter(from_checkpoint..=to_checkpoint)
            .collect::<Result<Vec<_>, _>>()
            .map_err(Into::into)
    }

    /// Returns future containing the state accumulator for the given epoch
    /// once available.
    pub async fn notify_read_checkpoint_state_hasher(
        &self,
        checkpoints: &[CheckpointSequenceNumber],
    ) -> SomaResult<Vec<GlobalStateHash>> {
        let tables = self.tables()?;
        Ok(self
            .checkpoint_state_notify_read
            .read(
                checkpoints,
                |checkpoints| {
                    tables
                        .state_hash_by_checkpoint
                        .multi_get(checkpoints)
                        .expect("db error")
                },
            )
            .await)
    }

    pub async fn notify_read_running_root(
        &self,
        checkpoint: CheckpointSequenceNumber,
    ) -> SomaResult<GlobalStateHash> {
        let registration = self.running_root_notify_read.register_one(&checkpoint);
        let acc = self.tables()?.running_root_state_hash.get(&checkpoint)?;

        let result = match acc {
            Some(ready) => Either::Left(futures::future::ready(ready)),
            None => Either::Right(registration),
        }
        .await;

        Ok(result)
    }

    /// Called when transaction outputs are committed to disk
    #[instrument(level = "trace", skip_all)]
    pub fn handle_finalized_checkpoint(
        &self,
        checkpoint: &CheckpointSummary,
        digests: &[TransactionDigest],
    ) -> SomaResult<()> {
        let tables = match self.tables() {
            Ok(tables) => tables,
            // After Epoch ends, it is no longer necessary to remove pending transactions
            // because the table will not be used anymore and be deleted eventually.
            Err(e) if matches!(e, SomaError::EpochEnded(_)) => return Ok(()),
            Err(e) => return Err(e),
        };
        let mut batch = tables.signed_effects_digests.batch();

        // Now that the transaction effects are committed, we will never re-execute, so we
        // don't need to worry about equivocating.
        batch.delete_batch(&tables.signed_effects_digests, digests)?;

        let seq = *checkpoint.sequence_number();

        let mut quarantine = self.consensus_quarantine.write();
        quarantine.update_highest_executed_checkpoint(seq, self, &mut batch)?;
        batch.write()?;

        self.consensus_output_cache
            .remove_executed_in_epoch(digests);

        Ok(())
    }

    pub fn get_all_pending_consensus_transactions(&self) -> Vec<ConsensusTransaction> {
        self.tables()
            .expect("recovery should not cross epoch boundary")
            .get_all_pending_consensus_transactions()
            .expect("failed to get pending consensus transactions")
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
            .get(&(*obj, start_version))
            .unwrap()
    }

    pub fn insert_finalized_transactions(
        &self,
        digests: &[TransactionDigest],
        sequence: CheckpointSequenceNumber,
    ) -> SomaResult {
        let mut batch = self.tables()?.executed_transactions_to_checkpoint.batch();
        batch.insert_batch(
            &self.tables()?.executed_transactions_to_checkpoint,
            digests.iter().map(|d| (*d, sequence)),
        )?;
        batch.write()?;
        trace!("Transactions {digests:?} finalized at checkpoint {sequence}");

        // Notify all readers that the transactions have been finalized as part of a checkpoint execution.
        for digest in digests {
            self.executed_transactions_to_checkpoint_notify_read
                .notify(digest, &sequence);
        }

        Ok(())
    }

    pub fn is_transaction_executed_in_checkpoint(
        &self,
        digest: &TransactionDigest,
    ) -> SomaResult<bool> {
        Ok(self
            .tables()?
            .executed_transactions_to_checkpoint
            .contains_key(digest)?)
    }

    pub fn transactions_executed_in_checkpoint(
        &self,
        digests: impl Iterator<Item = TransactionDigest>,
    ) -> SomaResult<Vec<bool>> {
        Ok(self
            .tables()?
            .executed_transactions_to_checkpoint
            .multi_contains_keys(digests)?)
    }

    pub fn get_transaction_checkpoint(
        &self,
        digest: &TransactionDigest,
    ) -> SomaResult<Option<CheckpointSequenceNumber>> {
        Ok(self
            .tables()?
            .executed_transactions_to_checkpoint
            .get(digest)?)
    }

    pub fn multi_get_transaction_checkpoint(
        &self,
        digests: &[TransactionDigest],
    ) -> SomaResult<Vec<Option<CheckpointSequenceNumber>>> {
        Ok(self
            .tables()?
            .executed_transactions_to_checkpoint
            .multi_get(digests)?
            .into_iter()
            .collect())
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
            .get_next_shared_object_versions(&tables, objects_to_init)?;

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
                // Note: we don't actually need to read from the transaction here, as no writer
                // can update object_store until after get_or_init_next_object_versions
                // completes.
                match cache_reader.get_object(id) {
                    Some(obj) => {
                        if obj.owner().start_version() == Some(*initial_version) {
                            ((*id, *initial_version), obj.version())
                        } else {
                            // If we can't find a matching start version, treat the object as
                            // if it's absent.
                            if let Some(obj_start_version) = obj.owner().start_version() {
                                    assert!(*initial_version >= obj_start_version,
                                            "should be impossible to certify a transaction with a start version that must have only existed in a previous epoch; obj = {obj:?} initial_version = {initial_version:?}, obj_start_version = {obj_start_version:?}");
                                }
                            ((*id, *initial_version), *initial_version)
                        }
                    }
                    None => ((*id, *initial_version), *initial_version),
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
        tables
            .next_shared_object_versions
            .multi_insert(versions_to_write)?;

        Ok(ret)
    }

    /// Given list of certificates, assign versions for all shared objects used in them.
    /// We start with the current next_shared_object_versions table for each object, and build
    /// up the versions based on the dependencies of each certificate.
    /// However, in the end we do not update the next_shared_object_versions table, which keeps
    /// this function idempotent. We should call this function when we are assigning shared object
    /// versions outside of consensus and do not want to taint the next_shared_object_versions table.
    pub fn assign_shared_object_versions_idempotent<'a>(
        &self,
        cache_reader: &dyn ObjectCacheRead,
        assignables: impl Iterator<Item = &'a Schedulable<&'a VerifiedExecutableTransaction>> + Clone,
    ) -> SomaResult<AssignedTxAndVersions> {
        Ok(SharedObjVerManager::assign_versions_from_consensus(
            self,
            cache_reader,
            assignables,
        )?
        .assigned_versions)
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
    ) -> SomaResult<AssignedVersions> {
        let assigned_versions = SharedObjVerManager::assign_versions_from_effects(
            &[(certificate, effects)],
            self,
            cache_reader,
        );
        let (_, assigned_versions) = assigned_versions.0.into_iter().next().unwrap();
        Ok(assigned_versions)
    }

    /// When submitting a certificate caller **must** provide a ReconfigState lock guard
    /// and verify that it allows new user certificates
    pub fn insert_pending_consensus_transactions(
        &self,
        transactions: &[ConsensusTransaction],
        lock: Option<&RwLockReadGuard<ReconfigState>>,
    ) -> SomaResult {
        let key_value_pairs = transactions.iter().filter_map(|tx| {
            if tx.is_mfp_transaction() {
                // UserTransaction does not need to be resubmitted on recovery.
                None
            } else {
                debug!("Inserting pending consensus transaction: {:?}", tx.key());
                Some((tx.key(), tx))
            }
        });
        self.tables()?
            .pending_consensus_transactions
            .multi_insert(key_value_pairs)?;

        let digests: Vec<_> = transactions
            .iter()
            .filter_map(|tx| match &tx.kind {
                ConsensusTransactionKind::CertifiedTransaction(cert) => Some(cert.digest()),
                _ => None,
            })
            .collect();
        if !digests.is_empty() {
            let state = lock.expect("Must pass reconfiguration lock when storing certificate");
            // Caller is responsible for performing graceful check
            assert!(
                state.should_accept_user_certs(),
                "Reconfiguration state should allow accepting user transactions"
            );
            let mut pending_consensus_certificates = self.pending_consensus_certificates.write();
            pending_consensus_certificates.extend(digests);
        }

        Ok(())
    }

    pub fn remove_pending_consensus_transactions(
        &self,
        keys: &[ConsensusTransactionKey],
    ) -> SomaResult {
        debug!("Removing pending consensus transactions: {:?}", keys);
        self.tables()?
            .pending_consensus_transactions
            .multi_remove(keys)?;
        let mut pending_consensus_certificates = self.pending_consensus_certificates.write();
        for key in keys {
            if let ConsensusTransactionKey::Certificate(digest) = key {
                pending_consensus_certificates.remove(digest);
            }
        }
        Ok(())
    }

    pub fn pending_consensus_certificates_count(&self) -> usize {
        self.pending_consensus_certificates.read().len()
    }

    pub fn pending_consensus_certificates_empty(&self) -> bool {
        self.pending_consensus_certificates.read().is_empty()
    }

    pub fn pending_consensus_certificates(&self) -> HashSet<TransactionDigest> {
        self.pending_consensus_certificates.read().clone()
    }

    pub fn is_pending_consensus_certificate(&self, tx_digest: &TransactionDigest) -> bool {
        self.pending_consensus_certificates
            .read()
            .contains(tx_digest)
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

    /// Returns true if all messages with the given keys were processed by consensus.
    pub fn all_external_consensus_messages_processed(
        &self,
        keys: impl Iterator<Item = ConsensusTransactionKey>,
    ) -> SomaResult<bool> {
        let keys = keys.map(SequencedConsensusTransactionKey::External);
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
            .consensus_quarantine
            .read()
            .is_consensus_message_processed(key)
            || self
                .tables()?
                .consensus_message_processed
                .contains_key(key)?)
    }

    pub fn check_consensus_messages_processed(
        &self,
        keys: impl Iterator<Item = SequencedConsensusTransactionKey>,
    ) -> SomaResult<Vec<bool>> {
        let keys = keys.collect::<Vec<_>>();

        let consensus_quarantine = self.consensus_quarantine.read();
        let tables = self.tables()?;

        Ok(do_fallback_lookup(
            &keys,
            |key| {
                if consensus_quarantine.is_consensus_message_processed(key) {
                    CacheResult::Hit(true)
                } else {
                    CacheResult::Miss
                }
            },
            |keys| {
                tables
                    .consensus_message_processed
                    .multi_contains_keys(keys)
                    .expect("db error")
            },
        ))
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

    /// Get notified when transactions get executed as part of a checkpoint execution.
    pub async fn transactions_executed_in_checkpoint_notify(
        &self,
        digests: Vec<TransactionDigest>,
    ) -> Result<(), SomaError> {
        let registrations = self
            .executed_transactions_to_checkpoint_notify_read
            .register_all(&digests);

        let unprocessed_keys_registrations = registrations
            .into_iter()
            .zip(self.transactions_executed_in_checkpoint(digests.into_iter())?)
            .filter(|(_, processed)| !*processed)
            .map(|(registration, _)| registration);

        join_all(unprocessed_keys_registrations).await;
        Ok(())
    }

    pub fn has_received_end_of_publish_from(&self, authority: &AuthorityName) -> bool {
        self.end_of_publish
            .try_lock()
            .expect("No contention on end_of_publish lock")
            .contains_key(authority)
    }

    // Converts transaction keys to digests, waiting for digests to become available for any
    // non-digest keys.
    pub async fn notify_read_tx_key_to_digest(
        &self,
        keys: &[TransactionKey],
    ) -> SomaResult<Vec<TransactionDigest>> {
        let non_digest_keys: Vec<_> = keys
            .iter()
            .filter_map(|key| {
                if matches!(key, TransactionKey::Digest(_)) {
                    None
                } else {
                    Some(*key)
                }
            })
            .collect();

        let registrations = self
            .executed_digests_notify_read
            .register_all(&non_digest_keys);
        let executed_digests = self
            .tables()?
            .transaction_key_to_digest
            .multi_get(&non_digest_keys)?;
        let futures = executed_digests
            .into_iter()
            .zip(registrations)
            .map(|(d, r)| match d {
                // Note that Some() clause also drops registration that is already fulfilled
                Some(ready) => Either::Left(futures::future::ready(ready)),
                None => Either::Right(r),
            });
        let mut results = VecDeque::from(join_all(futures).await);

        Ok(keys
            .iter()
            .map(|key| {
                if let TransactionKey::Digest(digest) = key {
                    *digest
                } else {
                    results
                        .pop_front()
                        .expect("number of returned results should match number of non-digest keys")
                }
            })
            .collect())
    }

    /// Caller must call consensus_message_processed_notify before calling this to ensure that all
    /// user signatures are available.
    pub fn user_signatures_for_checkpoint(
        &self,
        transactions: &[VerifiedTransaction],
        digests: &[TransactionDigest],
    ) -> Vec<Vec<GenericSignature>> {
        assert_eq!(transactions.len(), digests.len());

       

        let result: Vec<_> = {
            let mut user_sigs = self
                .consensus_output_cache
                .user_signatures_for_checkpoints
                .lock();
            digests
                .iter()
                .zip(transactions.iter())
                .map(|(d, t)| {
                     // Expect is safe as long as consensus_message_processed_notify is called
                        // before this call, to ensure that all canonical user signatures are
                        // available.
                        user_sigs.remove(d).expect("signature should be available")
                })
                .collect()
        };

        result
    }

    #[cfg(test)]
    pub(crate) fn push_consensus_output_for_tests(&self, output: ConsensusCommitOutput) {
        self.consensus_quarantine
            .write()
            .push_consensus_output(output, self)
            .expect("push_consensus_output should not fail");
    }

    pub(crate) fn process_user_signatures<'a>(
        &self,
        certificates: impl Iterator<Item = &'a Schedulable>,
    ) {
        let sigs: Vec<_> = certificates
            .filter_map(|s| match s {
                Schedulable::Transaction(certificate) => {
                    Some((*certificate.digest(), certificate.tx_signatures().to_vec()))
                }
               
            })
            .collect();

        let mut user_sigs = self
            .consensus_output_cache
            .user_signatures_for_checkpoints
            .lock();

        user_sigs.reserve(sigs.len());
        for (digest, sigs) in sigs {
            // User signatures are written in the same batch as consensus certificate processed flag,
            // which means we won't attempt to insert this twice for the same tx digest
            assert!(
                user_sigs.insert(digest, sigs).is_none(),
                "duplicate user signatures for transaction digest: {:?}",
                digest
            );
        }
    }

    pub fn acquire_tx_guard(&self, cert: &VerifiedExecutableTransaction) -> CertTxGuard {
        let digest = cert.digest();
        CertTxGuard(self.acquire_tx_lock(digest))
    }

    /// Acquire the lock for a tx without writing to the WAL.
    pub fn acquire_tx_lock(&self, digest: &TransactionDigest) -> CertLockGuard {
        CertLockGuard(self.mutex_table.acquire_lock(*digest))
    }


    pub fn get_reconfig_state_read_lock_guard(&self) -> RwLockReadGuard<'_, ReconfigState> {
        self.reconfig_state_mem.read()
    }

    pub(crate) fn get_reconfig_state_write_lock_guard(
        &self,
    ) -> RwLockWriteGuard<'_, ReconfigState> {
        self.reconfig_state_mem.write()
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
        info!("Epoch terminated - waiting for pending tasks to complete");
        *self.epoch_alive.write().await = false;
        info!("All pending epoch tasks completed");
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

    #[instrument(level = "trace", skip_all)]
    pub fn verify_transaction(&self, tx: Transaction) -> SomaResult<VerifiedTransaction> {
        self.signature_verifier
            .verify_tx(tx.data())
            .map(|_| VerifiedTransaction::new_from_verified(tx))
    }

    /// Verifies transaction signatures and other data
    /// Important: This function can potentially be called in parallel and you can not rely on order of transactions to perform verification
    /// If this function return an error, transaction is skipped and is not passed to handle_consensus_transaction
    /// This function returns unit error and is responsible for emitting log messages for internal errors
    pub(crate) fn verify_consensus_transaction(
        &self,
        transaction: SequencedConsensusTransaction,
    ) -> Option<VerifiedSequencedConsensusTransaction> {
        // Signatures are verified as part of the consensus payload verification in SuiTxValidator
        match &transaction.transaction {
            SequencedConsensusTransactionKind::External(ConsensusTransaction {
                kind: ConsensusTransactionKind::CertifiedTransaction(_certificate),
                ..
            }) => {}
            SequencedConsensusTransactionKind::External(ConsensusTransaction {
                kind: ConsensusTransactionKind::UserTransaction(_tx),
                ..
            }) => {}
            SequencedConsensusTransactionKind::External(ConsensusTransaction {
                kind: ConsensusTransactionKind::CheckpointSignature(data),
                ..
            }) => {
                if transaction.sender_authority() != data.summary.auth_sig().authority {
                    warn!(
                        "CheckpointSignature authority {} does not match its author from consensus {}",
                        data.summary.auth_sig().authority,
                        transaction.certificate_author_index
                    );
                    return None;
                }
            }
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

    pub(crate) fn db_batch(&self) -> SomaResult<DBBatch> {
        Ok(self.tables()?.last_consensus_stats.batch())
    }

    #[cfg(test)]
    pub fn db_batch_for_test(&self) -> DBBatch {
        self.db_batch()
            .expect("test should not be write past end of epoch")
    }

    pub(crate) fn calculate_pending_checkpoint_height(&self, consensus_round: u64) -> u64 {
        consensus_round
    }

    // Assigns shared object versions to transactions and updates the next shared object version state.
    // Shared object versions in cancelled transactions are assigned to special versions that will
    // cause the transactions to be cancelled in execution engine.
    pub(crate) fn process_consensus_transaction_shared_object_versions<'a>(
        &'a self,
        cache_reader: &dyn ObjectCacheRead,
        transactions: impl Iterator<Item = &'a Schedulable> + Clone,
        output: &mut ConsensusCommitOutput,
    ) -> SomaResult<AssignedTxAndVersions> {
        let all_certs = transactions;

        let ConsensusSharedObjVerAssignment {
            shared_input_next_versions,
            assigned_versions,
        } = SharedObjVerManager::assign_versions_from_consensus(
            self,
            cache_reader,
            all_certs,
        )?;
        debug!(
            "Assigned versions from consensus processing: {:?}",
            assigned_versions
        );

        output.set_next_shared_object_versions(shared_input_next_versions);
        Ok(assigned_versions)
    }

    pub fn get_highest_pending_checkpoint_height(&self) -> CheckpointHeight {
        self.consensus_quarantine
            .read()
            .get_highest_pending_checkpoint_height()
            .unwrap_or_default()
    }

    pub fn assign_shared_object_versions_for_tests(
        self: &Arc<Self>,
        cache_reader: &dyn ObjectCacheRead,
        transactions: &[VerifiedExecutableTransaction],
    ) -> SomaResult<AssignedTxAndVersions> {
        let mut output = ConsensusCommitOutput::new(0);
        let transactions: Vec<_> = transactions
            .iter()
            .cloned()
            .map(Schedulable::Transaction)
            .collect();

        // Record consensus messages as processed for each transaction
        for tx in transactions.iter() {
            if let Schedulable::Transaction(exec_tx) = tx {
                let key = SequencedConsensusTransactionKey::External(
                    ConsensusTransactionKey::Certificate(*exec_tx.digest()),
                );
                output.record_consensus_message_processed(key);
            }
        }

        let assigned_versions = self.process_consensus_transaction_shared_object_versions(
            cache_reader,
            transactions.iter(),
            &mut output,
        )?;
        let mut batch = self.db_batch()?;
        output.set_default_commit_stats_for_testing();
        output.write_to_batch(self, &mut batch)?;
        batch.write()?;
        Ok(assigned_versions)
    }

    pub(crate) fn process_notifications<'a>(
        &'a self,
        notifications: impl Iterator<Item = &'a SequencedConsensusTransactionKey>,
    ) {
        for key in notifications {
            self.consensus_notify_read.notify(key, &());
        }
    }

    /// If reconfig state is RejectUserCerts, and there is no fastpath transaction left to be
    /// finalized, send EndOfPublish to signal to other authorities that this authority is
    /// not voting for or executing more transactions in this epoch.
    pub(crate) fn should_send_end_of_publish(&self) -> bool {
        let reconfig_state = self.get_reconfig_state_read_lock_guard();
        if !reconfig_state.is_reject_user_certs() {
            // Still accepting user transactions, or already received 2f+1 EOP messages.
            // Either way EOP cannot or does not need to be sent.
            return false;
        }

        // EOP can only be sent after finalizing remaining transactions.
        self.pending_consensus_certificates_empty()
            && self
                .consensus_tx_status_cache
                .as_ref()
                .is_none_or(|c| c.get_num_fastpath_certified() == 0)
    }

    pub(crate) fn write_pending_checkpoint(
        &self,
        output: &mut ConsensusCommitOutput,
        checkpoint: &PendingCheckpoint,
    ) -> SomaResult {
        assert!(
            !self.pending_checkpoint_exists(&checkpoint.height())?,
            "Duplicate pending checkpoint notification at height {:?}",
            checkpoint.height()
        );

        debug!(
            checkpoint_commit_height = checkpoint.height(),
            "Pending checkpoint has {} roots",
            checkpoint.roots().len(),
        );
        trace!(
            checkpoint_commit_height = checkpoint.height(),
            "Transaction roots for pending checkpoint: {:?}",
            checkpoint.roots()
        );

        output.insert_pending_checkpoint(checkpoint.clone());

        Ok(())
    }

    pub fn get_pending_checkpoints(
        &self,
        last: Option<CheckpointHeight>,
    ) -> SomaResult<Vec<(CheckpointHeight, PendingCheckpoint)>> {
        Ok(self
            .consensus_quarantine
            .read()
            .get_pending_checkpoints(last))
    }

    pub fn pending_checkpoint_exists(&self, index: &CheckpointHeight) -> SomaResult<bool> {
        Ok(self
            .consensus_quarantine
            .read()
            .pending_checkpoint_exists(index))
    }

    pub fn process_constructed_checkpoint(
        &self,
        commit_height: CheckpointHeight,
        content_info: NonEmpty<(CheckpointSummary, CheckpointContents)>,
    ) {
        let mut consensus_quarantine = self.consensus_quarantine.write();
        for (position_in_commit, (summary, transactions)) in content_info.into_iter().enumerate() {
            let sequence_number = summary.sequence_number;
            let summary = BuilderCheckpointSummary {
                summary,
                checkpoint_height: Some(commit_height),
                position_in_commit,
            };

            consensus_quarantine.insert_builder_summary(sequence_number, summary, transactions);
        }

        // Because builder can run behind state sync, the data may be immediately ready to be committed.
        consensus_quarantine
            .commit(self)
            .expect("commit cannot fail");
    }

    /// Register genesis checkpoint in builder DB
    pub fn put_genesis_checkpoint_in_builder(
        &self,
        summary: &CheckpointSummary,
        contents: &CheckpointContents,
    ) -> SomaResult<()> {
        let sequence = summary.sequence_number;
        for transaction in contents.iter() {
            let digest = transaction.transaction;
            debug!(
                "Manually inserting genesis transaction in checkpoint DB: {:?}",
                digest
            );
            self.tables()?
                .builder_digest_to_checkpoint
                .insert(&digest, &sequence)?;
        }
        let builder_summary = BuilderCheckpointSummary {
            summary: summary.clone(),
            checkpoint_height: None,
            position_in_commit: 0,
        };
        self.tables()?
            .builder_checkpoint_summary
            .insert(summary.sequence_number(), &builder_summary)?;
        Ok(())
    }

    pub fn last_built_checkpoint_builder_summary(
        &self,
    ) -> SomaResult<Option<BuilderCheckpointSummary>> {
        if let Some(summary) = self.consensus_quarantine.read().last_built_summary() {
            return Ok(Some(summary.clone()));
        }

        Ok(self
            .tables()?
            .builder_checkpoint_summary
            .reversed_safe_iter_with_bounds(None, None)?
            .next()
            .transpose()?
            .map(|(_, s)| s))
    }

    pub fn last_built_checkpoint_summary(
        &self,
    ) -> SomaResult<Option<(CheckpointSequenceNumber, CheckpointSummary)>> {
        if let Some(BuilderCheckpointSummary { summary, .. }) =
            self.consensus_quarantine.read().last_built_summary()
        {
            let seq = *summary.sequence_number();
            debug!(
                "returning last_built_summary from consensus quarantine: {:?}",
                seq
            );
            Ok(Some((seq, summary.clone())))
        } else {
            let seq = self
                .tables()?
                .builder_checkpoint_summary
                .reversed_safe_iter_with_bounds(None, None)?
                .next()
                .transpose()?
                .map(|(seq, s)| (seq, s.summary));
            debug!(
                "returning last_built_summary from builder_checkpoint_summary_v2: {:?}",
                seq
            );
            Ok(seq)
        }
    }

    pub fn get_built_checkpoint_summary(
        &self,
        sequence: CheckpointSequenceNumber,
    ) -> SomaResult<Option<CheckpointSummary>> {
        if let Some(BuilderCheckpointSummary { summary, .. }) =
            self.consensus_quarantine.read().get_built_summary(sequence)
        {
            return Ok(Some(summary.clone()));
        }

        Ok(self
            .tables()?
            .builder_checkpoint_summary
            .get(&sequence)?
            .map(|s| s.summary))
    }

    pub(crate) fn get_lowest_non_genesis_checkpoint_summary(
        &self,
    ) -> SomaResult<Option<CheckpointSummary>> {
        for result in self
            .tables()?
            .builder_checkpoint_summary
            .safe_iter_with_bounds(None, None)
        {
            let (seq, bcs) = result?;
            if seq > 0 {
                return Ok(Some(bcs.summary));
            }
        }
        Ok(None)
    }

    pub fn builder_included_transactions_in_checkpoint<'a>(
        &self,
        digests: impl Iterator<Item = &'a TransactionDigest>,
    ) -> SomaResult<Vec<bool>> {
        let digests: Vec<_> = digests.cloned().collect();
        let tables = self.tables()?;
        Ok(do_fallback_lookup(
            &digests,
            |digest| {
                let consensus_quarantine = self.consensus_quarantine.read();
                if consensus_quarantine.included_transaction_in_checkpoint(digest) {
                    CacheResult::Hit(true)
                } else {
                    CacheResult::Miss
                }
            },
            |remaining| {
                tables
                    .builder_digest_to_checkpoint
                    .multi_contains_keys(remaining)
                    .expect("db error")
            },
        ))
    }

    pub fn get_last_checkpoint_signature_index(&self) -> SomaResult<u64> {
        Ok(self
            .tables()?
            .pending_checkpoint_signatures
            .reversed_safe_iter_with_bounds(None, None)?
            .next()
            .transpose()?
            .map(|((_, index), _)| index)
            .unwrap_or_default())
    }

    pub fn insert_checkpoint_signature(
        &self,
        checkpoint_seq: CheckpointSequenceNumber,
        index: u64,
        info: &CheckpointSignatureMessage,
    ) -> SomaResult<()> {
        Ok(self
            .tables()?
            .pending_checkpoint_signatures
            .insert(&(checkpoint_seq, index), info)?)
    }

    pub fn clear_signature_cache(&self) {
        self.signature_verifier.clear_signature_cache();
    }

     pub(crate) fn set_consensus_tx_status(
        &self,
        position: ConsensusPosition,
        status: ConsensusTxStatus,
    ) {
        if let Some(cache) = self.consensus_tx_status_cache.as_ref() {
            cache.set_transaction_status(position, status);
        }
    }

    pub(crate) fn set_rejection_vote_reason(&self, position: ConsensusPosition, reason: &SomaError) {
        if let Some(tx_reject_reason_cache) = self.tx_reject_reason_cache.as_ref() {
            tx_reject_reason_cache.set_rejection_vote_reason(position, reason);
        }
    }

    pub(crate) fn get_rejection_vote_reason(
        &self,
        position: ConsensusPosition,
    ) -> Option<SomaError> {
        if let Some(tx_reject_reason_cache) = self.tx_reject_reason_cache.as_ref() {
            tx_reject_reason_cache.get_rejection_vote_reason(position)
        } else {
            None
        }
    }

    /// Whether this node is a validator in this epoch.
    pub fn is_validator(&self) -> bool {
        self.committee.authority_exists(&self.name)
    }
}
