// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Test infrastructure for the ConsensusHandler.
//!
//! Provides:
//! - `TestConsensusCommit` — a mock implementation of `ConsensusCommitAPI`
//! - `CapturedTransactions` — thread-safe capture of scheduled transactions
//! - `setup_consensus_handler_for_testing()` — wires up a `ConsensusHandler` with transaction capture

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use arc_swap::ArcSwap;
use parking_lot::Mutex;
use protocol_config::ProtocolConfig;
use tokio::sync::mpsc;
use types::base::AuthorityName;
use types::committee::AuthorityIndex;
use types::consensus::ConsensusTransaction;
use types::consensus::block::BlockRef;
use types::digests::ConsensusCommitDigest;
use types::system_state::epoch_start::EpochStartSystemStateTrait;

use crate::backpressure_manager::BackpressureManager;
use crate::checkpoints::CheckpointServiceNoop;
use crate::consensus_adapter::{ConsensusAdapter, ConsensusClient};
use crate::consensus_handler::{ConsensusHandler, ExecutionSchedulerSender};
use crate::consensus_output_api::{ConsensusCommitAPI, ParsedTransaction};
use crate::execution_scheduler::SchedulingSource;
use crate::shared_obj_version_manager::{AssignedTxAndVersions, Schedulable};

use crate::authority::AuthorityState;
use crate::authority_per_epoch_store::{ConsensusStats, ExecutionIndicesWithStats};
use crate::checkpoints::CheckpointStore;

/// Thread-safe buffer capturing what the ConsensusHandler sends to the execution scheduler.
pub(crate) type CapturedTransactions =
    Arc<Mutex<Vec<(Vec<Schedulable>, AssignedTxAndVersions, SchedulingSource)>>>;

/// A mock consensus commit that implements `ConsensusCommitAPI`.
///
/// Used to feed synthetic consensus output into the ConsensusHandler in tests.
pub(crate) struct TestConsensusCommit {
    pub transactions: Vec<ConsensusTransaction>,
    pub round: u64,
    pub timestamp_ms: u64,
    pub sub_dag_index: u64,
}

impl TestConsensusCommit {
    pub fn new(
        transactions: Vec<ConsensusTransaction>,
        round: u64,
        timestamp_ms: u64,
        sub_dag_index: u64,
    ) -> Self {
        Self { transactions, round, timestamp_ms, sub_dag_index }
    }
}

impl fmt::Display for TestConsensusCommit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TestConsensusCommit(round={}, sub_dag={}, txns={})",
            self.round,
            self.sub_dag_index,
            self.transactions.len()
        )
    }
}

impl ConsensusCommitAPI for TestConsensusCommit {
    fn reputation_score_sorted_desc(&self) -> Option<Vec<(AuthorityIndex, u64)>> {
        None
    }

    fn leader_round(&self) -> u64 {
        self.round
    }

    fn leader_author_index(&self) -> AuthorityIndex {
        AuthorityIndex(0)
    }

    fn commit_timestamp_ms(&self) -> u64 {
        self.timestamp_ms
    }

    fn commit_sub_dag_index(&self) -> u64 {
        self.sub_dag_index
    }

    fn transactions(&self) -> Vec<(BlockRef, Vec<ParsedTransaction>)> {
        let block_ref = BlockRef {
            round: self.round as u32,
            author: AuthorityIndex(0),
            digest: Default::default(),
        };

        let parsed: Vec<ParsedTransaction> = self
            .transactions
            .iter()
            .map(|tx| {
                let serialized = bcs::to_bytes(tx).expect("serialization should not fail");
                let serialized_len = serialized.len();
                ParsedTransaction { transaction: tx.clone(), rejected: false, serialized_len }
            })
            .collect();

        vec![(block_ref, parsed)]
    }

    fn consensus_digest(&self, _protocol_config: &ProtocolConfig) -> ConsensusCommitDigest {
        ConsensusCommitDigest::default()
    }
}

/// Return type from `setup_consensus_handler_for_testing`.
pub(crate) struct TestConsensusHandlerSetup {
    pub consensus_handler: ConsensusHandler<CheckpointServiceNoop>,
    pub captured_transactions: CapturedTransactions,
}

/// Creates a `ConsensusHandler` wired up to capture all scheduled transactions
/// instead of sending them to the real execution scheduler.
///
/// This allows tests to:
/// 1. Feed synthetic `TestConsensusCommit` objects into the handler
/// 2. Inspect `captured_transactions` to verify scheduling behavior
pub(crate) async fn setup_consensus_handler_for_testing(
    authority: &Arc<AuthorityState>,
) -> TestConsensusHandlerSetup {
    let epoch_store_guard = authority.load_epoch_store_one_call_per_task();
    let epoch_store: Arc<crate::authority_per_epoch_store::AuthorityPerEpochStore> =
        Arc::clone(&epoch_store_guard);

    // Create a channel to capture transactions sent to the execution scheduler
    let (tx_sender, mut tx_receiver) =
        mpsc::unbounded_channel::<(Vec<Schedulable>, AssignedTxAndVersions, SchedulingSource)>();

    let captured_transactions: CapturedTransactions = Arc::new(Mutex::new(Vec::new()));
    let captured_clone = captured_transactions.clone();

    // Spawn a task that reads from the channel and stores in captured_transactions
    tokio::spawn(async move {
        while let Some(item) = tx_receiver.recv().await {
            captured_clone.lock().push(item);
        }
    });

    let execution_scheduler_sender = ExecutionSchedulerSender::new_for_testing(tx_sender);

    // Create a noop checkpoint service
    let checkpoint_service = Arc::new(CheckpointServiceNoop {});

    // Create a mock consensus adapter with a noop client
    let consensus_client: Arc<dyn ConsensusClient> = Arc::new(NoopConsensusClient);
    let checkpoint_store = CheckpointStore::new_for_tests();
    let consensus_adapter = Arc::new(ConsensusAdapter::new(
        consensus_client,
        checkpoint_store,
        authority.name,
        1000,
        100,
        None,
        None,
        epoch_store.protocol_config().clone(),
    ));

    let low_scoring_authorities =
        Arc::new(ArcSwap::from_pointee(HashMap::<AuthorityName, u64>::new()));

    let epoch_start_state = epoch_store.epoch_start_state();
    let committee = epoch_start_state.get_committee();

    let backpressure_manager = BackpressureManager::new_for_tests();
    let backpressure_subscriber = backpressure_manager.subscribe();

    let last_consensus_stats = ExecutionIndicesWithStats {
        index: Default::default(),
        hash: Default::default(),
        stats: ConsensusStats::new(committee.size()),
    };

    let consensus_handler = ConsensusHandler::new_for_testing(
        epoch_store,
        checkpoint_service,
        execution_scheduler_sender,
        consensus_adapter,
        authority.get_object_cache_reader().clone(),
        low_scoring_authorities,
        committee.clone(),
        backpressure_subscriber,
        last_consensus_stats,
    );

    TestConsensusHandlerSetup { consensus_handler, captured_transactions }
}

/// A no-op consensus client that accepts submissions but does nothing.
struct NoopConsensusClient;

#[async_trait::async_trait]
impl ConsensusClient for NoopConsensusClient {
    async fn submit(
        &self,
        _transactions: &[ConsensusTransaction],
        _epoch_store: &Arc<crate::authority_per_epoch_store::AuthorityPerEpochStore>,
    ) -> types::error::SomaResult<(
        Vec<types::consensus::ConsensusPosition>,
        crate::consensus_adapter::BlockStatusReceiver,
    )> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        tx.send(consensus::BlockStatus::Sequenced(BlockRef::MIN)).ok();
        Ok((vec![], rx))
    }
}
