// Copyright (c) Mysten Labs, Inc.
// Portions of this file are derived from Sui (https://github.com/MystenLabs/sui).
// SPDX-License-Identifier: Apache-2.0

use crate::{
    authority::{AuthorityState, ExecutionEnv},
    authority_per_epoch_store::{
        AuthorityPerEpochStore, CancelConsensusCertificateReason, ConsensusStats,
        ConsensusStatsAPI, ExecutionIndices, ExecutionIndicesWithStats,
    },
    backpressure_manager::{BackpressureManager, BackpressureSubscriber},
    cache::ObjectCacheRead,
    checkpoints::{
        CheckpointService, CheckpointServiceNotify, PendingCheckpoint, PendingCheckpointInfo,
    },
    consensus_adapter::ConsensusAdapter,
    consensus_output_api::{ConsensusCommitAPI, parse_block_transactions},
    consensus_quarantine::ConsensusCommitOutput,
    consensus_tx_status_cache::ConsensusTxStatus,
    execution_scheduler::{ExecutionScheduler, SchedulingSource},
    reconfiguration::ReconfigState,
    shared_obj_version_manager::{AssignedTxAndVersions, Schedulable, SharedObjVerManager},
    start_epoch::EpochStartConfigTrait,
};
use arc_swap::ArcSwap;
use consensus::CommitConsumerMonitor;
use itertools::Itertools as _;
use lru::LruCache;
use parking_lot::RwLockWriteGuard;
use protocol_config::ProtocolConfig;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, HashMap, HashSet, VecDeque},
    hash::Hash,
    num::NonZeroUsize,
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::{
    sync::{MutexGuard, mpsc},
    task::JoinSet,
};
use tracing::{debug, error, info, instrument, trace, warn};
use types::committee::Committee as ConsensusCommittee;
use types::consensus::block::{CertifiedBlocksOutput, TransactionIndex};
use types::consensus::commit::CommitIndex;
use types::{
    base::{AuthorityName, ConciseableName, ConsensusObjectSequenceKey, SequenceNumber},
    checkpoints::CheckpointSignatureMessage,
    committee::AuthorityIndex,
    consensus::{
        AuthorityCapabilitiesV1, ConsensusPosition, ConsensusTransaction, ConsensusTransactionKey,
        ConsensusTransactionKind,
    },
    digests::{AdditionalConsensusStateDigest, ConsensusCommitDigest, TransactionDigest},
    system_state::epoch_start::EpochStartSystemStateTrait,
    transaction::{SenderSignedData, VerifiedCertificate, VerifiedTransaction},
    transaction::{TrustedExecutableTransaction, VerifiedExecutableTransaction},
};

pub struct ConsensusHandlerInitializer {
    state: Arc<AuthorityState>,
    checkpoint_service: Arc<CheckpointService>,
    epoch_store: Arc<AuthorityPerEpochStore>,
    consensus_adapter: Arc<ConsensusAdapter>,
    low_scoring_authorities: Arc<ArcSwap<HashMap<AuthorityName, u64>>>,
    backpressure_manager: Arc<BackpressureManager>,
}

impl ConsensusHandlerInitializer {
    pub fn new(
        state: Arc<AuthorityState>,
        checkpoint_service: Arc<CheckpointService>,
        epoch_store: Arc<AuthorityPerEpochStore>,
        consensus_adapter: Arc<ConsensusAdapter>,
        low_scoring_authorities: Arc<ArcSwap<HashMap<AuthorityName, u64>>>,
        backpressure_manager: Arc<BackpressureManager>,
    ) -> Self {
        Self {
            state,
            checkpoint_service,
            epoch_store,
            consensus_adapter,
            low_scoring_authorities,
            backpressure_manager,
        }
    }

    pub(crate) fn new_consensus_handler(&self) -> ConsensusHandler<CheckpointService> {
        let new_epoch_start_state = self.epoch_store.epoch_start_state();
        let consensus_committee = new_epoch_start_state.get_committee();

        ConsensusHandler::new(
            self.epoch_store.clone(),
            self.checkpoint_service.clone(),
            self.state.execution_scheduler().clone(),
            self.consensus_adapter.clone(),
            self.state.get_object_cache_reader().clone(),
            self.low_scoring_authorities.clone(),
            consensus_committee,
            self.backpressure_manager.subscribe(),
            // self.state.traffic_controller.clone(),
        )
    }

    pub(crate) fn backpressure_subscriber(&self) -> BackpressureSubscriber {
        self.backpressure_manager.subscribe()
    }
}

mod additional_consensus_state {
    use std::marker::PhantomData;

    use fastcrypto::hash::HashFunction as _;
    use types::crypto::DefaultHash;

    use super::*;
    /// AdditionalConsensusState tracks any in-memory state that is retained by ConsensusHandler
    /// between consensus commits. Because of crash recovery, using such data is inherently risky.
    /// In order to do this safely, we must store data from a fixed number of previous commits.
    /// Then, at start-up, that same fixed number of already processed commits is replayed to
    /// reconstruct the state.
    ///
    /// To make sure that bugs in this process appear immediately, we record the digest of this
    /// state in ConsensusCommitPrologueV1, so that any deviation causes an immediate fork.
    #[derive(Serialize, Deserialize)]
    pub(super) struct AdditionalConsensusState {
        commit_interval_observer: CommitIntervalObserver,
    }

    impl AdditionalConsensusState {
        pub fn new(additional_consensus_state_window_size: u32) -> Self {
            Self {
                commit_interval_observer: CommitIntervalObserver::new(
                    additional_consensus_state_window_size,
                ),
            }
        }

        /// Update all internal state based on the new commit
        pub(crate) fn observe_commit(
            &mut self,
            protocol_config: &ProtocolConfig,
            epoch_start_time: u64,
            consensus_commit: &impl ConsensusCommitAPI,
        ) -> ConsensusCommitInfo {
            self.commit_interval_observer.observe_commit_time(consensus_commit);

            let estimated_commit_period = self
                .commit_interval_observer
                .commit_interval_estimate()
                .unwrap_or(Duration::from_millis(protocol_config.min_checkpoint_interval_ms()));

            info!("estimated commit rate: {:?}", estimated_commit_period);

            self.commit_info_impl(
                epoch_start_time,
                protocol_config,
                consensus_commit,
                Some(estimated_commit_period),
            )
        }

        fn commit_info_impl(
            &self,
            epoch_start_time: u64,
            protocol_config: &ProtocolConfig,
            consensus_commit: &impl ConsensusCommitAPI,
            estimated_commit_period: Option<Duration>,
        ) -> ConsensusCommitInfo {
            let leader_author = consensus_commit.leader_author_index();
            let timestamp = consensus_commit.commit_timestamp_ms();

            let timestamp = if timestamp < epoch_start_time {
                error!(
                    "Unexpected commit timestamp {timestamp} less then epoch start time {epoch_start_time}, author {leader_author:?}"
                );
                epoch_start_time
            } else {
                timestamp
            };

            ConsensusCommitInfo {
                _phantom: PhantomData,
                round: consensus_commit.leader_round(),
                timestamp,
                leader_author,
                sub_dag_index: consensus_commit.commit_sub_dag_index(),
                consensus_commit_digest: consensus_commit.consensus_digest(protocol_config),
                additional_state_digest: Some(self.digest()),
                estimated_commit_period,
                skip_consensus_commit_prologue_in_test: false,
            }
        }

        /// Get the digest of the current state.
        fn digest(&self) -> AdditionalConsensusStateDigest {
            let mut hash = DefaultHash::new();
            bcs::serialize_into(&mut hash, self).unwrap();
            AdditionalConsensusStateDigest::new(hash.finalize().into())
        }
    }

    pub struct ConsensusCommitInfo {
        // prevent public construction
        _phantom: PhantomData<()>,

        pub round: u64,
        pub timestamp: u64,
        pub leader_author: AuthorityIndex,
        pub sub_dag_index: u64,
        pub consensus_commit_digest: ConsensusCommitDigest,

        additional_state_digest: Option<AdditionalConsensusStateDigest>,
        estimated_commit_period: Option<Duration>,

        pub skip_consensus_commit_prologue_in_test: bool,
    }

    impl ConsensusCommitInfo {
        pub fn new_for_test(
            commit_round: u64,
            commit_timestamp: u64,
            estimated_commit_period: Option<Duration>,
            skip_consensus_commit_prologue_in_test: bool,
        ) -> Self {
            Self {
                _phantom: PhantomData,
                round: commit_round,
                timestamp: commit_timestamp,
                leader_author: AuthorityIndex(0),
                sub_dag_index: 0,
                consensus_commit_digest: ConsensusCommitDigest::default(),
                additional_state_digest: Some(AdditionalConsensusStateDigest::ZERO),
                estimated_commit_period,
                skip_consensus_commit_prologue_in_test,
            }
        }

        pub fn new_for_congestion_test(
            commit_round: u64,
            commit_timestamp: u64,
            estimated_commit_period: Duration,
        ) -> Self {
            Self::new_for_test(commit_round, commit_timestamp, Some(estimated_commit_period), true)
        }

        pub fn additional_state_digest(&self) -> AdditionalConsensusStateDigest {
            // this method cannot be called if stateless_commit_info is used
            self.additional_state_digest.expect("additional_state_digest is not available")
        }

        pub fn estimated_commit_period(&self) -> Duration {
            // this method cannot be called if stateless_commit_info is used
            self.estimated_commit_period.expect("estimated commit period is not available")
        }

        fn consensus_commit_prologue_transaction(
            &self,
            epoch: u64,
            additional_state_digest: AdditionalConsensusStateDigest,
        ) -> VerifiedExecutableTransaction {
            let transaction = VerifiedTransaction::new_consensus_commit_prologue(
                epoch,
                self.round,
                self.timestamp,
                self.consensus_commit_digest,
                additional_state_digest,
            );
            VerifiedExecutableTransaction::new_system(transaction, epoch)
        }

        pub fn create_consensus_commit_prologue_transaction(
            &self,
            epoch: u64,
            protocol_config: &ProtocolConfig,
            commit_info: &ConsensusCommitInfo,
            indirect_state_observer: IndirectStateObserver,
        ) -> VerifiedExecutableTransaction {
            let additional_state_digest = {
                let d1 = commit_info.additional_state_digest();
                indirect_state_observer.fold_with(d1)
            };

            self.consensus_commit_prologue_transaction(epoch, additional_state_digest)
        }
    }

    #[derive(Default)]
    pub struct IndirectStateObserver {
        hash: DefaultHash,
    }

    impl IndirectStateObserver {
        pub fn new() -> Self {
            Self::default()
        }

        pub fn observe_indirect_state<T: Serialize>(&mut self, state: &T) {
            bcs::serialize_into(&mut self.hash, state).unwrap();
        }

        pub fn fold_with(
            self,
            d1: AdditionalConsensusStateDigest,
        ) -> AdditionalConsensusStateDigest {
            let hash = self.hash.finalize();
            let d2 = AdditionalConsensusStateDigest::new(hash.into());

            let mut hasher = DefaultHash::new();
            bcs::serialize_into(&mut hasher, &d1).unwrap();
            bcs::serialize_into(&mut hasher, &d2).unwrap();
            AdditionalConsensusStateDigest::new(hasher.finalize().into())
        }
    }
}
use additional_consensus_state::AdditionalConsensusState;
pub(crate) use additional_consensus_state::{ConsensusCommitInfo, IndirectStateObserver};

pub struct ConsensusHandler<C> {
    /// A store created for each epoch. ConsensusHandler is recreated each epoch, with the
    /// corresponding store. This store is also used to get the current epoch ID.
    epoch_store: Arc<AuthorityPerEpochStore>,
    /// Holds the indices, hash and stats after the last consensus commit
    /// It is used for avoiding replaying already processed transactions,
    /// checking chain consistency, and accumulating per-epoch consensus output stats.
    last_consensus_stats: ExecutionIndicesWithStats,
    checkpoint_service: Arc<C>,
    /// cache reader is needed when determining the next version to assign for shared objects.
    cache_reader: Arc<dyn ObjectCacheRead>,
    /// Reputation scores used by consensus adapter that we update, forwarded from consensus
    low_scoring_authorities: Arc<ArcSwap<HashMap<AuthorityName, u64>>>,
    /// The consensus committee used to do stake computations for deciding set of low scoring authorities
    committee: ConsensusCommittee,
    /// Lru cache to quickly discard transactions processed by consensus
    processed_cache: LruCache<SequencedConsensusTransactionKey, ()>,
    /// Enqueues transactions to the execution scheduler via a separate task.
    execution_scheduler_sender: ExecutionSchedulerSender,
    /// Consensus adapter for submitting transactions to consensus
    consensus_adapter: Arc<ConsensusAdapter>,

    additional_consensus_state: AdditionalConsensusState,

    backpressure_subscriber: BackpressureSubscriber,
    // traffic_controller: Option<Arc<TrafficController>>,
}

const PROCESSED_CACHE_CAP: usize = 1024 * 1024;

impl<C> ConsensusHandler<C> {
    pub(crate) fn new(
        epoch_store: Arc<AuthorityPerEpochStore>,
        checkpoint_service: Arc<C>,
        execution_scheduler: Arc<ExecutionScheduler>,
        consensus_adapter: Arc<ConsensusAdapter>,
        cache_reader: Arc<dyn ObjectCacheRead>,
        low_scoring_authorities: Arc<ArcSwap<HashMap<AuthorityName, u64>>>,
        committee: ConsensusCommittee,
        backpressure_subscriber: BackpressureSubscriber,
        // traffic_controller: Option<Arc<TrafficController>>,
    ) -> Self {
        // Recover last_consensus_stats so it is consistent across validators.
        let mut last_consensus_stats = epoch_store
            .get_last_consensus_stats()
            .expect("Should be able to read last consensus index");
        // stats is empty at the beginning of epoch.
        if !last_consensus_stats.stats.is_initialized() {
            last_consensus_stats.stats = ConsensusStats::new(committee.size());
        }
        let execution_scheduler_sender =
            ExecutionSchedulerSender::start(execution_scheduler, epoch_store.clone());
        let commit_rate_estimate_window_size =
            epoch_store.protocol_config().get_consensus_commit_rate_estimation_window_size();
        Self {
            epoch_store,
            last_consensus_stats,
            checkpoint_service,
            cache_reader,
            low_scoring_authorities,
            committee,
            processed_cache: LruCache::new(NonZeroUsize::new(PROCESSED_CACHE_CAP).unwrap()),
            execution_scheduler_sender,
            consensus_adapter,
            additional_consensus_state: AdditionalConsensusState::new(
                commit_rate_estimate_window_size,
            ),
            backpressure_subscriber,
            // traffic_controller,
        }
    }

    /// Returns the last subdag index processed by the handler.
    pub(crate) fn last_processed_subdag_index(&self) -> u64 {
        self.last_consensus_stats.index.sub_dag_index
    }

    pub(crate) fn execution_scheduler_sender(&self) -> &ExecutionSchedulerSender {
        &self.execution_scheduler_sender
    }

    pub(crate) fn new_for_testing(
        epoch_store: Arc<AuthorityPerEpochStore>,
        checkpoint_service: Arc<C>,
        execution_scheduler_sender: ExecutionSchedulerSender,
        consensus_adapter: Arc<ConsensusAdapter>,
        cache_reader: Arc<dyn ObjectCacheRead>,
        low_scoring_authorities: Arc<ArcSwap<HashMap<AuthorityName, u64>>>,
        committee: ConsensusCommittee,
        backpressure_subscriber: BackpressureSubscriber,
        // traffic_controller: Option<Arc<TrafficController>>,
        last_consensus_stats: ExecutionIndicesWithStats,
    ) -> Self {
        let commit_rate_estimate_window_size =
            epoch_store.protocol_config().get_consensus_commit_rate_estimation_window_size();
        Self {
            epoch_store,
            last_consensus_stats,
            checkpoint_service,
            cache_reader,
            low_scoring_authorities,
            committee,
            processed_cache: LruCache::new(NonZeroUsize::new(PROCESSED_CACHE_CAP).unwrap()),
            execution_scheduler_sender,
            consensus_adapter,
            additional_consensus_state: AdditionalConsensusState::new(
                commit_rate_estimate_window_size,
            ),
            backpressure_subscriber,
            // traffic_controller,
        }
    }
}

#[derive(Default)]
struct CommitHandlerInput {
    user_transactions: Vec<VerifiedExecutableTransaction>,
    capability_notifications: Vec<AuthorityCapabilitiesV1>,
    checkpoint_signature_messages: Vec<CheckpointSignatureMessage>,
    end_of_publish_transactions: Vec<AuthorityName>,
}

struct CommitHandlerState {
    output: ConsensusCommitOutput,
    indirect_state_observer: Option<IndirectStateObserver>,
    initial_reconfig_state: ReconfigState,
}

impl CommitHandlerState {
    fn get_notifications(&self) -> Vec<SequencedConsensusTransactionKey> {
        self.output.get_consensus_messages_processed().cloned().collect()
    }
}

impl<C: CheckpointServiceNotify + Send + Sync> ConsensusHandler<C> {
    /// Called during startup to allow us to observe commits we previously processed, for crash recovery.
    /// Any state computed here must be a pure function of the commits observed, it cannot depend on any
    /// state recorded in the epoch db.
    fn handle_prior_consensus_commit(&mut self, consensus_commit: impl ConsensusCommitAPI) {
        let protocol_config = self.epoch_store.protocol_config();
        let epoch_start_time = self.epoch_store.epoch_start_config().epoch_start_timestamp_ms();

        self.additional_consensus_state.observe_commit(
            protocol_config,
            epoch_start_time,
            &consensus_commit,
        );
    }

    #[cfg(test)]
    pub(crate) async fn handle_consensus_commit_for_test(
        &mut self,
        consensus_commit: impl ConsensusCommitAPI,
    ) {
        self.handle_consensus_commit(consensus_commit).await;
    }

    #[instrument(level = "debug", skip_all, fields(epoch = self.epoch_store.epoch(), round = consensus_commit.leader_round()))]
    pub(crate) async fn handle_consensus_commit(
        &mut self,
        consensus_commit: impl ConsensusCommitAPI,
    ) {
        let protocol_config = self.epoch_store.protocol_config();

        // This may block until one of two conditions happens:
        // - Number of uncommitted transactions in the writeback cache goes below the
        //   backpressure threshold.
        // - The highest executed checkpoint catches up to the highest certified checkpoint.
        self.backpressure_subscriber.await_no_backpressure().await;

        let epoch = self.epoch_store.epoch();

        let last_committed_round = self.last_consensus_stats.index.last_committed_round;

        if let Some(consensus_tx_status_cache) = self.epoch_store.consensus_tx_status_cache.as_ref()
        {
            consensus_tx_status_cache
                .update_last_committed_leader_round(last_committed_round as u32)
                .await;
        }
        if let Some(tx_reject_reason_cache) = self.epoch_store.tx_reject_reason_cache.as_ref() {
            tx_reject_reason_cache.set_last_committed_leader_round(last_committed_round as u32);
        }

        let commit_info = self.additional_consensus_state.observe_commit(
            protocol_config,
            self.epoch_store.epoch_start_config().epoch_start_timestamp_ms(),
            &consensus_commit,
        );
        assert!(commit_info.round > last_committed_round);

        let (timestamp, leader_author, commit_sub_dag_index) =
            self.gather_commit_metadata(&consensus_commit);

        info!(
            %consensus_commit,
            "Received consensus output"
        );

        self.last_consensus_stats.index = ExecutionIndices {
            last_committed_round: commit_info.round,
            sub_dag_index: commit_sub_dag_index,
            transaction_index: 0_u64,
        };

        update_low_scoring_authorities(
            self.low_scoring_authorities.clone(),
            &self.committee,
            consensus_commit.reputation_score_sorted_desc(),
            protocol_config.consensus_bad_nodes_stake_threshold(),
        );

        let mut state = CommitHandlerState {
            output: ConsensusCommitOutput::new(commit_info.round),
            indirect_state_observer: Some(IndirectStateObserver::new()),
            initial_reconfig_state: self.epoch_store.get_reconfig_state_read_lock_guard().clone(),
        };

        let transactions = self.filter_consensus_txns(
            state.initial_reconfig_state.clone(),
            &commit_info,
            &consensus_commit,
        );
        let transactions = self.deduplicate_consensus_txns(&mut state, &commit_info, transactions);

        let CommitHandlerInput {
            user_transactions,
            capability_notifications,
            checkpoint_signature_messages,
            end_of_publish_transactions,
        } = self.build_commit_handler_input(transactions);

        self.process_capability_notifications(capability_notifications);
        self.process_checkpoint_signature_messages(checkpoint_signature_messages);

        let (schedulables, assigned_versions) =
            self.process_transactions(&mut state, &commit_info, user_transactions);

        let (should_accept_tx, lock, final_round) =
            self.handle_eop(&mut state, end_of_publish_transactions);

        let make_checkpoint = should_accept_tx || final_round;
        if !make_checkpoint {
            // No need for any further processing
            return;
        }

        self.create_pending_checkpoints(&mut state, &commit_info, &schedulables, final_round);

        let notifications = state.get_notifications();

        state.output.record_consensus_commit_stats(self.last_consensus_stats.clone());

        self.epoch_store
            .consensus_quarantine
            .write()
            .push_consensus_output(state.output, &self.epoch_store)
            .expect("push_consensus_output should not fail");

        utils::fail_point!("crash-after-consensus-commit");

        // Only after batch is written, notify checkpoint service to start building any new
        // pending checkpoints.
        debug!(
            ?commit_info.round,
            "Notifying checkpoint service about new pending checkpoint(s)",
        );
        self.checkpoint_service.notify_checkpoint().expect("failed to notify checkpoint service");

        self.epoch_store.process_notifications(notifications.iter());

        // pass lock by value to ensure that it is held until this point
        self.log_final_round(lock, final_round);

        let mut schedulables = schedulables;

        self.execution_scheduler_sender.send(
            schedulables,
            assigned_versions,
            SchedulingSource::NonFastPath,
        );

        self.send_end_of_publish_if_needed().await;
    }

    fn handle_eop(
        &self,
        state: &mut CommitHandlerState,
        end_of_publish_transactions: Vec<AuthorityName>,
    ) -> (bool, Option<RwLockWriteGuard<'_, ReconfigState>>, bool) {
        let collected_eop =
            self.process_end_of_publish_transactions(state, end_of_publish_transactions);
        if collected_eop {
            let (lock, final_round) = self.advance_eop_state_machine(state);
            (lock.should_accept_tx(), Some(lock), final_round)
        } else {
            (true, None, false)
        }
    }

    fn log_final_round(&self, lock: Option<RwLockWriteGuard<ReconfigState>>, final_round: bool) {
        if final_round {
            let epoch = self.epoch_store.epoch();
            info!(
                ?epoch,
                lock=?lock.as_ref(),
                final_round=?final_round,
                "Notified last checkpoint"
            );
        }
    }

    fn create_pending_checkpoints(
        &self,
        state: &mut CommitHandlerState,
        commit_info: &ConsensusCommitInfo,
        schedulables: &[Schedulable],
        final_round: bool,
    ) {
        let checkpoint_height =
            self.epoch_store.calculate_pending_checkpoint_height(commit_info.round);

        let pending_checkpoint = PendingCheckpoint {
            roots: schedulables.iter().map(|s| s.key()).collect(),
            details: PendingCheckpointInfo {
                timestamp_ms: commit_info.timestamp,
                last_of_epoch: final_round,
                checkpoint_height,
            },
        };
        self.epoch_store
            .write_pending_checkpoint(&mut state.output, &pending_checkpoint)
            .expect("failed to write pending checkpoint");
    }

    fn process_transactions(
        &self,
        state: &mut CommitHandlerState,
        commit_info: &ConsensusCommitInfo,
        user_transactions: Vec<VerifiedExecutableTransaction>,
    ) -> (Vec<Schedulable>, AssignedTxAndVersions) {
        let protocol_config = self.epoch_store.protocol_config();
        let epoch = self.epoch_store.epoch();

        let transactions_to_schedule = user_transactions;

        let consensus_commit_prologue = self.add_consensus_commit_prologue_transaction(
            state,
            commit_info,
            transactions_to_schedule.iter().map(Schedulable::Transaction),
        );

        let schedulables: Vec<_> = itertools::chain!(
            consensus_commit_prologue.into_iter(),
            transactions_to_schedule.into_iter(),
        )
        .map(Schedulable::Transaction)
        .collect();

        let assigned_versions = self
            .epoch_store
            .process_consensus_transaction_shared_object_versions(
                self.cache_reader.as_ref(),
                schedulables.iter(),
                &mut state.output,
            )
            .expect("failed to assign shared object versions");

        self.epoch_store.process_user_signatures(schedulables.iter());

        (schedulables, assigned_versions)
    }

    // Adds the consensus commit prologue transaction to the beginning of input `transactions` to update
    // the system clock used in all transactions in the current consensus commit.
    // Returns the root of the consensus commit prologue transaction if it was added to the input.
    fn add_consensus_commit_prologue_transaction<'a>(
        &'a self,
        state: &'a mut CommitHandlerState,
        commit_info: &'a ConsensusCommitInfo,
        schedulables: impl Iterator<Item = Schedulable<&'a VerifiedExecutableTransaction>>,
    ) -> Option<VerifiedExecutableTransaction> {
        {
            if commit_info.skip_consensus_commit_prologue_in_test {
                return None;
            }
        }

        let transaction = commit_info.create_consensus_commit_prologue_transaction(
            self.epoch_store.epoch(),
            self.epoch_store.protocol_config(),
            commit_info,
            state.indirect_state_observer.take().unwrap(),
        );
        Some(transaction)
    }

    fn process_capability_notifications(
        &self,
        capability_notifications: Vec<AuthorityCapabilitiesV1>,
    ) {
        for capabilities in capability_notifications {
            self.epoch_store.record_capabilities(&capabilities).expect("db error");
        }
    }

    fn process_checkpoint_signature_messages(
        &self,
        checkpoint_signature_messages: Vec<CheckpointSignatureMessage>,
    ) {
        for checkpoint_signature_message in checkpoint_signature_messages {
            self.checkpoint_service
                .notify_checkpoint_signature(&self.epoch_store, &checkpoint_signature_message)
                .expect("db error");
        }
    }

    /// Returns true if we have collected a quorum of end of publish messages (either in this round or a previous round).
    fn process_end_of_publish_transactions(
        &self,
        state: &mut CommitHandlerState,
        end_of_publish_transactions: Vec<AuthorityName>,
    ) -> bool {
        let mut eop_aggregator = self.epoch_store.end_of_publish.try_lock().expect(
            "No contention on end_of_publish as it is only accessed from consensus handler",
        );

        if eop_aggregator.has_quorum() {
            return true;
        }

        if end_of_publish_transactions.is_empty() {
            return false;
        }

        for authority in end_of_publish_transactions {
            info!("Received EndOfPublish from {:?}", authority.concise());

            // It is ok to just release lock here as this function is the only place that transition into RejectAllCerts state
            // And this function itself is always executed from consensus task
            state.output.insert_end_of_publish(authority);
            if eop_aggregator.insert_generic(authority, ()).is_quorum_reached() {
                debug!(
                    "Collected enough end_of_publish messages with last message from validator {:?}",
                    authority.concise(),
                );
                return true;
            }
        }

        false
    }

    /// After we have collected 2f+1 EndOfPublish messages, we call this function every round until the epoch
    /// ends.
    fn advance_eop_state_machine(
        &self,
        state: &mut CommitHandlerState,
    ) -> (
        RwLockWriteGuard<'_, ReconfigState>,
        bool, // true if final round
    ) {
        let mut reconfig_state = self.epoch_store.get_reconfig_state_write_lock_guard();
        let start_state_is_reject_all_tx = reconfig_state.is_reject_all_tx();

        reconfig_state.close_all_certs();

        if !start_state_is_reject_all_tx {
            info!("Transitioning to RejectAllTx");
        }
        reconfig_state.close_all_tx();

        state.output.store_reconfig_state(reconfig_state.clone());

        if !start_state_is_reject_all_tx && reconfig_state.is_reject_all_tx() {
            (reconfig_state, true)
        } else {
            (reconfig_state, false)
        }
    }

    fn gather_commit_metadata(
        &self,
        consensus_commit: &impl ConsensusCommitAPI,
    ) -> (u64, AuthorityIndex, u64) {
        let timestamp = consensus_commit.commit_timestamp_ms();
        let leader_author = consensus_commit.leader_author_index();
        let commit_sub_dag_index = consensus_commit.commit_sub_dag_index();

        let system_time_ms =
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as i64;

        let consensus_timestamp_bias_ms = system_time_ms - (timestamp as i64);
        let consensus_timestamp_bias_seconds = consensus_timestamp_bias_ms as f64 / 1000.0;

        let epoch_start = self.epoch_store.epoch_start_config().epoch_start_timestamp_ms();
        let timestamp = if timestamp < epoch_start {
            error!(
                "Unexpected commit timestamp {timestamp} less then epoch start time {epoch_start}, author {leader_author}"
            );
            epoch_start
        } else {
            timestamp
        };

        (timestamp, leader_author, commit_sub_dag_index)
    }

    // Filters out rejected or deprecated transactions.
    #[instrument(level = "trace", skip_all)]
    fn filter_consensus_txns(
        &mut self,
        initial_reconfig_state: ReconfigState,
        commit_info: &ConsensusCommitInfo,
        consensus_commit: &impl ConsensusCommitAPI,
    ) -> Vec<(SequencedConsensusTransactionKind, u32)> {
        let mut transactions = Vec::new();
        let epoch = self.epoch_store.epoch();
        let mut num_finalized_user_transactions = vec![0; self.committee.size()];
        let mut num_rejected_user_transactions = vec![0; self.committee.size()];
        for (block, parsed_transactions) in consensus_commit.transactions() {
            let author = block.author.value();
            // TODO: consider only messages within 1~3 rounds of the leader?
            self.last_consensus_stats.stats.inc_num_messages(author);

            // Set the "ping" transaction status for this block. This is ncecessary as there might be some ping requests waiting for the ping transaction to be certified.
            self.epoch_store.set_consensus_tx_status(
                ConsensusPosition::ping(epoch, block),
                ConsensusTxStatus::Finalized,
            );

            for (tx_index, parsed) in parsed_transactions.into_iter().enumerate() {
                let position =
                    ConsensusPosition { epoch, block, index: tx_index as TransactionIndex };

                // Transaction has appeared in consensus output, we can increment the submission count
                // for this tx for DoS protection.
                if let ConsensusTransactionKind::UserTransaction(tx) = &parsed.transaction.kind {
                    let digest = tx.digest();
                    if let Some((spam_weight, submitter_client_addrs)) = self
                        .epoch_store
                        .submitted_transaction_cache
                        .increment_submission_count(digest)
                    {
                        // TODO: traffic controller
                        // if let Some(ref traffic_controller) = self.traffic_controller {
                        //     debug!(
                        //         "Transaction {digest} exceeded submission limits, spam_weight: {spam_weight:?} applied to {} client addresses",
                        //         submitter_client_addrs.len()
                        //     );

                        //     // Apply spam weight to all client addresses that submitted this transaction
                        //     for addr in submitter_client_addrs {
                        //         traffic_controller.tally(TrafficTally::new(
                        //             Some(addr),
                        //             None,
                        //             None,
                        //             spam_weight.clone(),
                        //         ));
                        //     }
                        // } else {
                        //     warn!(
                        //         "Transaction {digest} exceeded submission limits, spam_weight: {spam_weight:?} for {} client addresses (traffic controller not configured)",
                        //         submitter_client_addrs.len()
                        //     );
                        // }
                    }
                }

                if parsed.rejected {
                    // TODO(fastpath): Add metrics for rejected transactions.
                    if matches!(
                        parsed.transaction.kind,
                        ConsensusTransactionKind::UserTransaction(_)
                    ) {
                        self.epoch_store
                            .set_consensus_tx_status(position, ConsensusTxStatus::Rejected);
                        num_rejected_user_transactions[author] += 1;
                    }
                    // Skip processing rejected transactions.
                    // TODO(fastpath): Handle unlocking.
                    continue;
                }
                if matches!(parsed.transaction.kind, ConsensusTransactionKind::UserTransaction(_)) {
                    self.epoch_store
                        .set_consensus_tx_status(position, ConsensusTxStatus::Finalized);
                    num_finalized_user_transactions[author] += 1;
                }
                let kind = classify(&parsed.transaction);

                // UserTransaction exists only when mysticeti_fastpath is enabled in protocol config.
                if matches!(
                    &parsed.transaction.kind,
                    ConsensusTransactionKind::CertifiedTransaction(_)
                        | ConsensusTransactionKind::UserTransaction(_)
                ) {
                    self.last_consensus_stats.stats.inc_num_user_transactions(author);
                }

                if !initial_reconfig_state.should_accept_consensus_certs() {
                    // (Note: we no longer need to worry about the previously deferred condition, since we are only
                    // processing newly-received transactions at this time).
                    match &parsed.transaction.kind {
                        ConsensusTransactionKind::UserTransaction(_)
                        | ConsensusTransactionKind::CertifiedTransaction(_)
                        | ConsensusTransactionKind::EndOfPublish(_)
                        | ConsensusTransactionKind::CapabilityNotification(_) => {
                            debug!(
                                "Ignoring consensus transaction {:?} because of end of epoch",
                                parsed.transaction.key()
                            );
                            continue;
                        }

                        // These are the message types that are still processed even if !should_accept_consensus_certs()
                        ConsensusTransactionKind::CheckpointSignature(_) => (),
                    }
                }

                if let ConsensusTransactionKind::CertifiedTransaction(certificate) =
                    &parsed.transaction.kind
                {
                    if certificate.epoch() != epoch {
                        debug!(
                            "Certificate epoch ({:?}) doesn't match the current epoch ({:?})",
                            certificate.epoch(),
                            epoch
                        );
                        continue;
                    }
                }

                if matches!(
                    &parsed.transaction.kind,
                    ConsensusTransactionKind::UserTransaction(_)
                        | ConsensusTransactionKind::CertifiedTransaction(_)
                ) {
                    let author_name =
                        self.epoch_store.committee().authority_by_index(author as u32).unwrap();
                    if self.epoch_store.has_received_end_of_publish_from(author_name) {
                        // In some edge cases, consensus might resend previously seen certificate after EndOfPublish
                        // An honest validator should not send a new transaction after EndOfPublish. Whether the
                        // transaction is duplicate or not, we filter it out here.
                        warn!(
                            "Ignoring consensus transaction {:?} from authority {:?}, which already sent EndOfPublish message to consensus",
                            author_name.concise(),
                            parsed.transaction.key(),
                        );
                        continue;
                    }
                }

                let transaction = SequencedConsensusTransactionKind::External(parsed.transaction);
                transactions.push((transaction, author as u32));
            }
        }

        transactions
    }

    fn deduplicate_consensus_txns(
        &mut self,
        state: &mut CommitHandlerState,
        commit_info: &ConsensusCommitInfo,
        transactions: Vec<(SequencedConsensusTransactionKind, u32)>,
    ) -> Vec<VerifiedSequencedConsensusTransaction> {
        // We need a set here as well, since the processed_cache is a LRU cache and can drop
        // entries while we're iterating over the sequenced transactions.
        let mut processed_set = HashSet::new();

        let mut all_transactions = Vec::new();

        // All of these TODOs are handled here in the new code, whereas in the old code, they were
        // each handled separately. The key thing to see is that all messages are marked as processed
        // here, except for ones that are filtered out earlier (e.g. due to !should_accept_consensus_certs()).

        for (seq, (transaction, cert_origin)) in transactions.into_iter().enumerate() {
            // SequencedConsensusTransaction for commit prologue any more.
            // In process_consensus_transactions_and_commit_boundary(), we will add a system consensus commit
            // prologue transaction, which will be the first transaction in this consensus commit batch.
            // Therefore, the transaction sequence number starts from 1 here.
            let current_tx_index = ExecutionIndices {
                last_committed_round: commit_info.round,
                sub_dag_index: commit_info.sub_dag_index,
                transaction_index: (seq + 1) as u64,
            };

            self.last_consensus_stats.index = current_tx_index;

            let certificate_author =
                *self.epoch_store.committee().authority_by_index(cert_origin).unwrap();

            let sequenced_transaction = SequencedConsensusTransaction {
                certificate_author_index: AuthorityIndex(cert_origin),
                certificate_author,
                consensus_index: current_tx_index,
                transaction,
            };

            let Some(verified_transaction) =
                self.epoch_store.verify_consensus_transaction(sequenced_transaction)
            else {
                continue;
            };

            let key = verified_transaction.0.key();
            let in_set = !processed_set.insert(key.clone());
            let in_cache = self.processed_cache.put(key.clone(), ()).is_some();

            if in_set || in_cache {
                continue;
            }
            if self.epoch_store.is_consensus_message_processed(&key).expect("db error") {
                continue;
            }

            state.output.record_consensus_message_processed(key);

            all_transactions.push(verified_transaction);
        }

        all_transactions
    }

    fn build_commit_handler_input(
        &self,
        transactions: Vec<VerifiedSequencedConsensusTransaction>,
    ) -> CommitHandlerInput {
        let epoch = self.epoch_store.epoch();
        let mut commit_handler_input = CommitHandlerInput::default();

        for VerifiedSequencedConsensusTransaction(transaction) in transactions.into_iter() {
            match transaction.transaction {
                SequencedConsensusTransactionKind::External(consensus_transaction) => {
                    match consensus_transaction.kind {
                        // === User transactions ===
                        ConsensusTransactionKind::CertifiedTransaction(cert) => {
                            // Safe because signatures are verified when consensus called into TxValidator::validate_batch.
                            let cert = VerifiedCertificate::new_unchecked(*cert);
                            let transaction =
                                VerifiedExecutableTransaction::new_from_certificate(cert);
                            commit_handler_input.user_transactions.push(transaction);
                        }
                        ConsensusTransactionKind::UserTransaction(tx) => {
                            // Safe because transactions are certified by consensus.
                            let tx = VerifiedTransaction::new_unchecked(*tx);
                            // TODO(fastpath): accept position in consensus, after plumbing consensus round, authority index, and transaction index here.
                            let transaction =
                                VerifiedExecutableTransaction::new_from_consensus(tx, epoch);
                            commit_handler_input.user_transactions.push(transaction);
                        }

                        // === State machines ===
                        ConsensusTransactionKind::EndOfPublish(authority_public_key_bytes) => {
                            commit_handler_input
                                .end_of_publish_transactions
                                .push(authority_public_key_bytes);
                        }

                        ConsensusTransactionKind::CapabilityNotification(
                            authority_capabilities,
                        ) => {
                            commit_handler_input
                                .capability_notifications
                                .push(authority_capabilities);
                        }

                        ConsensusTransactionKind::CheckpointSignature(
                            checkpoint_signature_message,
                        ) => {
                            commit_handler_input
                                .checkpoint_signature_messages
                                .push(*checkpoint_signature_message);
                        }
                    }
                }
                // TODO: I think we can delete this, it was only used to inject randomness state update into the tx stream.
                SequencedConsensusTransactionKind::System(_verified_envelope) => unreachable!(),
            }
        }

        commit_handler_input
    }

    async fn send_end_of_publish_if_needed(&self) {
        if !self.epoch_store.should_send_end_of_publish() {
            return;
        }

        let end_of_publish = ConsensusTransaction::new_end_of_publish(self.epoch_store.name);
        if let Err(err) =
            self.consensus_adapter.submit(end_of_publish, None, &self.epoch_store, None, None)
        {
            warn!("Error when sending EndOfPublish message from ConsensusHandler: {:?}", err);
        } else {
            info!(epoch=?self.epoch_store.epoch(), "Sending EndOfPublish message to consensus");
        }
    }
}

/// Sends transactions to the execution scheduler in a separate task,
/// to avoid blocking consensus handler.
#[derive(Clone)]
pub(crate) struct ExecutionSchedulerSender {
    // Using unbounded channel to avoid blocking consensus commit and transaction handler.
    // Tuple: (transactions, assigned_versions, scheduling_source)
    sender: mpsc::UnboundedSender<(Vec<Schedulable>, AssignedTxAndVersions, SchedulingSource)>,
}

impl ExecutionSchedulerSender {
    fn start(
        execution_scheduler: Arc<ExecutionScheduler>,
        epoch_store: Arc<AuthorityPerEpochStore>,
    ) -> Self {
        let (sender, recv) = mpsc::unbounded_channel();
        tokio::spawn(Self::run(recv, execution_scheduler, epoch_store));
        Self { sender }
    }

    pub(crate) fn new_for_testing(
        sender: mpsc::UnboundedSender<(Vec<Schedulable>, AssignedTxAndVersions, SchedulingSource)>,
    ) -> Self {
        Self { sender }
    }

    fn send(
        &self,
        transactions: Vec<Schedulable>,
        assigned_versions: AssignedTxAndVersions,
        scheduling_source: SchedulingSource,
    ) {
        let _ = self.sender.send((transactions, assigned_versions, scheduling_source));
    }

    async fn run(
        mut recv: mpsc::UnboundedReceiver<(
            Vec<Schedulable>,
            AssignedTxAndVersions,
            SchedulingSource,
        )>,
        execution_scheduler: Arc<ExecutionScheduler>,
        epoch_store: Arc<AuthorityPerEpochStore>,
    ) {
        while let Some((transactions, assigned_versions, scheduling_source)) = recv.recv().await {
            let assigned_versions = assigned_versions.into_map();
            let txns = transactions
                .into_iter()
                .map(|txn| {
                    let key = txn.key();
                    let env = ExecutionEnv::new()
                        .with_scheduling_source(scheduling_source)
                        .with_assigned_versions(
                            assigned_versions.get(&key).cloned().unwrap_or_default(),
                        );
                    (txn, env)
                })
                .collect();
            execution_scheduler.enqueue(txns, &epoch_store);
        }
    }
}

/// Manages the lifetime of tasks handling the commits and transactions output by consensus.
pub(crate) struct MysticetiConsensusHandler {
    tasks: JoinSet<()>,
}

impl MysticetiConsensusHandler {
    pub(crate) fn new(
        last_processed_commit_at_startup: CommitIndex,
        mut consensus_handler: ConsensusHandler<CheckpointService>,
        consensus_block_handler: ConsensusBlockHandler,
        mut commit_receiver: mpsc::UnboundedReceiver<types::consensus::commit::CommittedSubDag>,
        mut block_receiver: mpsc::UnboundedReceiver<types::consensus::block::CertifiedBlocksOutput>,
        commit_consumer_monitor: Arc<CommitConsumerMonitor>,
    ) -> Self {
        let mut tasks = JoinSet::new();
        tasks.spawn(async move {
            // TODO: pause when execution is overloaded, so consensus can detect the backpressure.
            while let Some(consensus_commit) = commit_receiver.recv().await {
                let commit_index = consensus_commit.commit_ref.index;
                if commit_index <= last_processed_commit_at_startup {
                    consensus_handler.handle_prior_consensus_commit(consensus_commit);
                } else {
                    consensus_handler.handle_consensus_commit(consensus_commit).await;
                }
                commit_consumer_monitor.set_highest_handled_commit(commit_index);
            }
        });

        tasks.spawn(async move {
            while let Some(blocks) = block_receiver.recv().await {
                consensus_block_handler.handle_certified_blocks(blocks).await;
            }
        });

        Self { tasks }
    }

    pub(crate) async fn abort(&mut self) {
        self.tasks.shutdown().await;
    }
}

pub(crate) fn classify(transaction: &ConsensusTransaction) -> &'static str {
    match &transaction.kind {
        ConsensusTransactionKind::CertifiedTransaction(certificate) => {
            if certificate.is_consensus_tx() { "shared_certificate" } else { "owned_certificate" }
        }
        ConsensusTransactionKind::CheckpointSignature(_) => "checkpoint_signature",
        ConsensusTransactionKind::CapabilityNotification(_) => "capability_notification",
        ConsensusTransactionKind::EndOfPublish(_) => "end_of_publish",
        ConsensusTransactionKind::UserTransaction(tx) => {
            if tx.is_consensus_tx() {
                "shared_user_transaction"
            } else {
                "owned_user_transaction"
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SequencedConsensusTransaction {
    pub certificate_author_index: AuthorityIndex,
    pub certificate_author: AuthorityName,
    pub consensus_index: ExecutionIndices,
    pub transaction: SequencedConsensusTransactionKind,
}

#[derive(Debug, Clone)]
#[allow(clippy::large_enum_variant)]
pub enum SequencedConsensusTransactionKind {
    External(ConsensusTransaction),
    System(VerifiedExecutableTransaction),
}

impl Serialize for SequencedConsensusTransactionKind {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let serializable = SerializableSequencedConsensusTransactionKind::from(self);
        serializable.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for SequencedConsensusTransactionKind {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let serializable =
            SerializableSequencedConsensusTransactionKind::deserialize(deserializer)?;
        Ok(serializable.into())
    }
}

// We can't serialize SequencedConsensusTransactionKind directly because it contains a
// VerifiedExecutableTransaction, which is not serializable (by design). This wrapper allows us to
// convert to a serializable format easily.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(clippy::large_enum_variant)]
enum SerializableSequencedConsensusTransactionKind {
    External(ConsensusTransaction),
    System(TrustedExecutableTransaction),
}

impl From<&SequencedConsensusTransactionKind> for SerializableSequencedConsensusTransactionKind {
    fn from(kind: &SequencedConsensusTransactionKind) -> Self {
        match kind {
            SequencedConsensusTransactionKind::External(ext) => {
                SerializableSequencedConsensusTransactionKind::External(ext.clone())
            }
            SequencedConsensusTransactionKind::System(txn) => {
                SerializableSequencedConsensusTransactionKind::System(txn.clone().serializable())
            }
        }
    }
}

impl From<SerializableSequencedConsensusTransactionKind> for SequencedConsensusTransactionKind {
    fn from(kind: SerializableSequencedConsensusTransactionKind) -> Self {
        match kind {
            SerializableSequencedConsensusTransactionKind::External(ext) => {
                SequencedConsensusTransactionKind::External(ext)
            }
            SerializableSequencedConsensusTransactionKind::System(txn) => {
                SequencedConsensusTransactionKind::System(txn.into())
            }
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Hash, PartialEq, Eq, Debug, Ord, PartialOrd)]
pub enum SequencedConsensusTransactionKey {
    External(ConsensusTransactionKey),
    System(TransactionDigest),
}

impl SequencedConsensusTransactionKind {
    pub fn key(&self) -> SequencedConsensusTransactionKey {
        match self {
            SequencedConsensusTransactionKind::External(ext) => {
                SequencedConsensusTransactionKey::External(ext.key())
            }
            SequencedConsensusTransactionKind::System(txn) => {
                SequencedConsensusTransactionKey::System(*txn.digest())
            }
        }
    }

    pub fn get_tracking_id(&self) -> u64 {
        match self {
            SequencedConsensusTransactionKind::External(ext) => ext.get_tracking_id(),
            SequencedConsensusTransactionKind::System(_txn) => 0,
        }
    }

    pub fn is_executable_transaction(&self) -> bool {
        match self {
            SequencedConsensusTransactionKind::External(ext) => ext.is_user_transaction(),
            SequencedConsensusTransactionKind::System(_) => true,
        }
    }

    pub fn executable_transaction_digest(&self) -> Option<TransactionDigest> {
        match self {
            SequencedConsensusTransactionKind::External(ext) => match &ext.kind {
                ConsensusTransactionKind::CertifiedTransaction(txn) => Some(*txn.digest()),
                ConsensusTransactionKind::UserTransaction(txn) => Some(*txn.digest()),
                _ => None,
            },
            SequencedConsensusTransactionKind::System(txn) => Some(*txn.digest()),
        }
    }

    pub fn is_end_of_publish(&self) -> bool {
        match self {
            SequencedConsensusTransactionKind::External(ext) => {
                matches!(ext.kind, ConsensusTransactionKind::EndOfPublish(..))
            }
            SequencedConsensusTransactionKind::System(_) => false,
        }
    }
}

impl SequencedConsensusTransaction {
    pub fn sender_authority(&self) -> AuthorityName {
        self.certificate_author
    }

    pub fn key(&self) -> SequencedConsensusTransactionKey {
        self.transaction.key()
    }

    pub fn is_end_of_publish(&self) -> bool {
        if let SequencedConsensusTransactionKind::External(ref transaction) = self.transaction {
            matches!(transaction.kind, ConsensusTransactionKind::EndOfPublish(..))
        } else {
            false
        }
    }

    pub fn is_system(&self) -> bool {
        matches!(self.transaction, SequencedConsensusTransactionKind::System(_))
    }

    pub fn as_consensus_txn(&self) -> Option<&SenderSignedData> {
        match &self.transaction {
            SequencedConsensusTransactionKind::External(ConsensusTransaction {
                kind: ConsensusTransactionKind::CertifiedTransaction(certificate),
                ..
            }) if certificate.is_consensus_tx() => Some(certificate.data()),
            SequencedConsensusTransactionKind::External(ConsensusTransaction {
                kind: ConsensusTransactionKind::UserTransaction(txn),
                ..
            }) if txn.is_consensus_tx() => Some(txn.data()),
            SequencedConsensusTransactionKind::System(txn) if txn.is_consensus_tx() => {
                Some(txn.data())
            }
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifiedSequencedConsensusTransaction(pub SequencedConsensusTransaction);

#[cfg(test)]
impl VerifiedSequencedConsensusTransaction {
    pub fn new_test(transaction: ConsensusTransaction) -> Self {
        Self(SequencedConsensusTransaction::new_test(transaction))
    }
}

impl SequencedConsensusTransaction {
    pub fn new_test(transaction: ConsensusTransaction) -> Self {
        Self {
            certificate_author_index: AuthorityIndex(0),
            certificate_author: AuthorityName::ZERO,
            consensus_index: Default::default(),
            transaction: SequencedConsensusTransactionKind::External(transaction),
        }
    }
}

/// Handles certified and rejected transactions output by consensus.
pub(crate) struct ConsensusBlockHandler {
    /// Per-epoch store.
    epoch_store: Arc<AuthorityPerEpochStore>,
    /// Enqueues transactions to the execution scheduler via a separate task.
    execution_scheduler_sender: ExecutionSchedulerSender,
    /// Backpressure subscriber to wait for backpressure to be resolved.
    backpressure_subscriber: BackpressureSubscriber,
}

impl ConsensusBlockHandler {
    pub fn new(
        epoch_store: Arc<AuthorityPerEpochStore>,
        execution_scheduler_sender: ExecutionSchedulerSender,
        backpressure_subscriber: BackpressureSubscriber,
    ) -> Self {
        Self { epoch_store, execution_scheduler_sender, backpressure_subscriber }
    }

    #[instrument(level = "debug", skip_all)]
    async fn handle_certified_blocks(&self, blocks_output: CertifiedBlocksOutput) {
        self.backpressure_subscriber.await_no_backpressure().await;

        // Avoid triggering fastpath execution or setting transaction status to fastpath certified, during reconfiguration.
        let reconfiguration_lock = self.epoch_store.get_reconfig_state_read_lock_guard();
        if !reconfiguration_lock.should_accept_user_certs() {
            debug!(
                "Skipping fastpath execution because epoch {} is closing user transactions: {}",
                self.epoch_store.epoch(),
                blocks_output.blocks.iter().map(|b| b.block.reference().to_string()).join(", "),
            );
            return;
        }

        let epoch = self.epoch_store.epoch();
        let parsed_transactions = blocks_output
            .blocks
            .into_iter()
            .map(|certified_block| {
                let block_ref = certified_block.block.reference();
                let transactions =
                    parse_block_transactions(&certified_block.block, &certified_block.rejected);
                (block_ref, transactions)
            })
            .collect::<Vec<_>>();
        let mut executable_transactions = vec![];
        for (block, transactions) in parsed_transactions.into_iter() {
            // Set the "ping" transaction status for this block. This is ncecessary as there might be some ping requests waiting for the ping transaction to be certified.
            self.epoch_store.set_consensus_tx_status(
                ConsensusPosition::ping(epoch, block),
                ConsensusTxStatus::FastpathCertified,
            );

            for (txn_idx, parsed) in transactions.into_iter().enumerate() {
                let position =
                    ConsensusPosition { epoch, block, index: txn_idx as TransactionIndex };

                let status_str = if parsed.rejected { "rejected" } else { "certified" };
                if let ConsensusTransactionKind::UserTransaction(tx) = &parsed.transaction.kind {
                    debug!(
                        "User Transaction in position: {:} with digest {:} is {:}",
                        position,
                        tx.digest(),
                        status_str
                    );
                } else {
                    debug!("System Transaction in position: {:} is {:}", position, status_str);
                }

                if parsed.rejected {
                    // TODO(fastpath): avoid parsing blocks twice between handling commit and fastpath transactions?
                    self.epoch_store.set_consensus_tx_status(position, ConsensusTxStatus::Rejected);
                    continue;
                }

                if let ConsensusTransactionKind::UserTransaction(tx) = parsed.transaction.kind {
                    if tx.is_consensus_tx() {
                        continue;
                    }
                    // Only set fastpath certified status on transactions intended for fastpath execution.
                    self.epoch_store
                        .set_consensus_tx_status(position, ConsensusTxStatus::FastpathCertified);
                    let tx = VerifiedTransaction::new_unchecked(*tx);
                    executable_transactions.push(Schedulable::Transaction(
                        VerifiedExecutableTransaction::new_from_consensus(
                            tx,
                            self.epoch_store.epoch(),
                        ),
                    ));
                }
            }
        }

        if executable_transactions.is_empty() {
            return;
        }

        self.execution_scheduler_sender.send(
            executable_transactions,
            Default::default(),
            SchedulingSource::MysticetiFastPath,
        );
    }
}

#[derive(Serialize, Deserialize)]
pub(crate) struct CommitIntervalObserver {
    ring_buffer: VecDeque<u64>,
}

impl CommitIntervalObserver {
    pub fn new(window_size: u32) -> Self {
        Self { ring_buffer: VecDeque::with_capacity(window_size as usize) }
    }

    pub fn observe_commit_time(&mut self, consensus_commit: &impl ConsensusCommitAPI) {
        let commit_time = consensus_commit.commit_timestamp_ms();
        if self.ring_buffer.len() == self.ring_buffer.capacity() {
            self.ring_buffer.pop_front();
        }
        self.ring_buffer.push_back(commit_time);
    }

    pub fn commit_interval_estimate(&self) -> Option<Duration> {
        if self.ring_buffer.len() <= 1 {
            None
        } else {
            let first = self.ring_buffer.front().unwrap();
            let last = self.ring_buffer.back().unwrap();
            let duration = last.saturating_sub(*first);
            let num_commits = self.ring_buffer.len() as u64;
            Some(Duration::from_millis(duration.div_ceil(num_commits)))
        }
    }
}

/// Updates list of authorities that are deemed to have low reputation scores by consensus
/// these may be lagging behind the network, byzantine, or not reliably participating for any reason.
/// The algorithm is flagging as low scoring authorities all the validators that have the lowest scores
/// up to the defined protocol_config.consensus_bad_nodes_stake_threshold. This is done to align the
/// submission side with the consensus leader election schedule. Practically we don't want to submit
/// transactions for sequencing to validators that have low scores and are not part of the leader
/// schedule since the chances of getting them sequenced are lower.
pub(crate) fn update_low_scoring_authorities(
    low_scoring_authorities: Arc<ArcSwap<HashMap<AuthorityName, u64>>>,
    committee: &ConsensusCommittee,
    reputation_score_sorted_desc: Option<Vec<(AuthorityIndex, u64)>>,
    consensus_bad_nodes_stake_threshold: u64,
) {
    assert!(
        (0..=33).contains(&consensus_bad_nodes_stake_threshold),
        "The bad_nodes_stake_threshold should be in range [0 - 33], out of bounds parameter detected {}",
        consensus_bad_nodes_stake_threshold
    );

    let Some(reputation_scores) = reputation_score_sorted_desc else {
        return;
    };

    // We order the authorities by score ascending order in the exact same way as the reputation
    // scores do - so we keep complete alignment between implementations
    let scores_per_authority_order_asc: Vec<_> = reputation_scores
        .into_iter()
        .rev() // we reverse so we get them in asc order
        .collect();

    let mut final_low_scoring_map = HashMap::new();
    let mut total_stake = 0;
    for (index, score) in scores_per_authority_order_asc {
        let authority_name = committee.authority_by_index(index.0).unwrap();
        let authority_index = committee.to_authority_index(index.value()).unwrap();
        let consensus_authority = committee.authority(authority_name).unwrap();
        let hostname = &consensus_authority.hostname;
        let stake = consensus_authority.stake;
        total_stake += stake;

        let included =
            if total_stake <= consensus_bad_nodes_stake_threshold * committee.total_stake() / 100 {
                final_low_scoring_map.insert(*authority_name, score);
                true
            } else {
                false
            };

        if !hostname.is_empty() {
            debug!("authority {} has score {}, is low scoring: {}", hostname, score, included);
        }
    }
    // Report the actual flagged final low scoring authorities
    low_scoring_authorities.swap(Arc::new(final_low_scoring_map));
}
