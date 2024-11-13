use crate::state_accumulator::CommitIndex;
use crate::{
    epoch_store::AuthorityPerEpochStore, output::ConsensusOutputAPI, state::AuthorityState,
    throughput::ConsensusThroughputCalculator, tx_manager::TransactionManager,
};
use consensus::CommittedSubDag;
use lru::LruCache;
use serde::{Deserialize, Serialize};
use std::{collections::HashSet, num::NonZeroUsize, sync::Arc};
use tokio::sync::mpsc::UnboundedReceiver;
use tracing::{error, info, instrument, warn};
use types::committee::AuthorityIndex;
use types::system_state::EpochStartSystemStateTrait;
use types::{
    base::AuthorityName,
    committee::{Committee, EpochId},
    consensus::{ConsensusTransaction, ConsensusTransactionKey, ConsensusTransactionKind},
    digests::{ConsensusCommitDigest, TransactionDigest},
    execution_indices::ExecutionIndices,
    protocol::ProtocolConfig,
    transaction::{
        TrustedExecutableTransaction, VerifiedExecutableTransaction, VerifiedTransaction,
    },
};

pub struct ConsensusHandlerInitializer {
    state: Arc<AuthorityState>,
    epoch_store: Arc<AuthorityPerEpochStore>,
    throughput_calculator: Arc<ConsensusThroughputCalculator>,
}

impl ConsensusHandlerInitializer {
    pub fn new(
        state: Arc<AuthorityState>,
        epoch_store: Arc<AuthorityPerEpochStore>,
        throughput_calculator: Arc<ConsensusThroughputCalculator>,
    ) -> Self {
        Self {
            state,
            epoch_store,
            throughput_calculator,
        }
    }

    pub fn new_for_testing(state: Arc<AuthorityState>) -> Self {
        Self {
            state: state.clone(),
            epoch_store: state.epoch_store_for_testing().clone(),
            throughput_calculator: Arc::new(ConsensusThroughputCalculator::new(None)),
        }
    }
    pub fn new_consensus_handler(&self) -> ConsensusHandler {
        let new_epoch_start_state = self.epoch_store.epoch_start_state();
        let committee = new_epoch_start_state.get_committee();

        ConsensusHandler::new(
            self.epoch_store.clone(),
            self.state.transaction_manager().clone(),
            committee,
            self.throughput_calculator.clone(),
        )
    }
}

pub struct ConsensusHandler {
    /// A store created for each epoch. ConsensusHandler is recreated each epoch, with the
    /// corresponding store. This store is also used to get the current epoch ID.
    epoch_store: Arc<AuthorityPerEpochStore>,
    // /// Holds the indices, hash and stats after the last consensus commit
    // /// It is used for avoiding replaying already processed transactions,
    // /// checking chain consistency, and accumulating per-epoch consensus output stats.
    last_consensus_stats: ExecutionIndices,
    /// The  committee used to do stake computations
    committee: Committee,
    /// Lru cache to quickly discard transactions processed by consensus
    processed_cache: LruCache<SequencedConsensusTransactionKey, ()>,
    transaction_scheduler: AsyncTransactionScheduler,
    /// Using the throughput calculator to record the current consensus throughput
    throughput_calculator: Arc<ConsensusThroughputCalculator>,
}

const PROCESSED_CACHE_CAP: usize = 1024 * 1024;

impl ConsensusHandler {
    pub fn new(
        epoch_store: Arc<AuthorityPerEpochStore>,
        transaction_manager: Arc<TransactionManager>,
        committee: Committee,
        throughput_calculator: Arc<ConsensusThroughputCalculator>,
    ) -> Self {
        // Recover last_consensus_stats so it is consistent across validators.
        let last_consensus_stats = epoch_store
            .get_last_consensus_stats()
            .expect("Should be able to read last consensus index");

        let transaction_scheduler =
            AsyncTransactionScheduler::start(transaction_manager, epoch_store.clone());
        Self {
            epoch_store,
            committee,
            last_consensus_stats,
            transaction_scheduler,
            processed_cache: LruCache::new(NonZeroUsize::new(PROCESSED_CACHE_CAP).unwrap()),
            throughput_calculator,
        }
    }
}

impl ConsensusHandler {
    #[instrument(level = "debug", skip_all)]
    async fn handle_consensus_output_internal(
        &mut self,
        consensus_output: impl ConsensusOutputAPI,
    ) {
        let last_committed_round = self.last_consensus_stats.last_committed_round;

        let round = consensus_output.leader_round();

        // assert!(round >= last_committed_round);
        if last_committed_round == round {
            // we can receive the same commit twice after restart
            // It is critical that the writes done by this function are atomic - otherwise we can
            // lose the later parts of a commit if we restart midway through processing it.
            warn!(
                "Ignoring consensus output for round {} as it is already committed. NOTE: This is only expected if consensus is running.",
                round
            );
            return;
        }

        /* (serialized, transaction, output_cert) */
        let mut transactions = vec![];
        let timestamp = consensus_output.commit_timestamp_ms();
        let leader_author = consensus_output.leader_author_index();
        let commit_sub_dag_index = consensus_output.commit_sub_dag_index();

        let epoch_start = self
            .epoch_store
            .epoch_start_config()
            .epoch_start_timestamp_ms();
        let timestamp = if timestamp < epoch_start {
            error!(
                "Unexpected commit timestamp {timestamp} less then epoch start time {epoch_start}, author {leader_author}, round {round}",
            );
            epoch_start
        } else {
            timestamp
        };

        info!(
            %consensus_output,
            epoch = ?self.epoch_store.epoch(),
            "Received consensus output"
        );

        // TODO: testing empty commit explicitly.
        // Note that consensus commit batch may contain no transactions, but we still need to record the current
        // round and subdag index in the last_consensus_stats, so that it won't be re-executed in the future.
        self.last_consensus_stats = ExecutionIndices {
            last_committed_round: round,
            sub_dag_index: commit_sub_dag_index,
            transaction_index: 0_u64,
        };

        {
            for (authority_index, authority_transactions) in consensus_output.transactions() {
                // TODO: consider only messages within 1~3 rounds of the leader?
                for (serialized_transaction, transaction) in authority_transactions {
                    let kind = classify(&transaction);
                    let transaction = SequencedConsensusTransactionKind::External(transaction);
                    transactions.push((serialized_transaction, transaction, authority_index));
                }
            }
        }

        let mut all_transactions = Vec::new();
        {
            // We need a set here as well, since the processed_cache is a LRU cache and can drop
            // entries while we're iterating over the sequenced transactions.
            let mut processed_set = HashSet::new();

            for (seq, (serialized, transaction, cert_origin)) in
                transactions.into_iter().enumerate()
            {
                // In process_consensus_transactions_and_commit_boundary(), we will add a system consensus commit
                // prologue transaction, which will be the first transaction in this consensus commit batch.
                // Therefore, the transaction sequence number starts from 1 here.
                let current_tx_index = ExecutionIndices {
                    last_committed_round: round,
                    sub_dag_index: commit_sub_dag_index,
                    transaction_index: (seq + 1) as u64,
                };

                self.last_consensus_stats = current_tx_index;

                let certificate_author = self
                    .committee
                    .authority_by_index(cert_origin.value() as u32)
                    .unwrap();

                let sequenced_transaction = SequencedConsensusTransaction {
                    certificate_author_index: cert_origin,
                    certificate_author: *certificate_author,
                    consensus_index: current_tx_index,
                    transaction,
                };

                let key = sequenced_transaction.key();
                let in_set = !processed_set.insert(key);
                let in_cache = self
                    .processed_cache
                    .put(sequenced_transaction.key(), ())
                    .is_some();

                if in_set || in_cache {
                    continue;
                }

                all_transactions.push(sequenced_transaction);
            }
        }

        let transactions_to_schedule = self
            .epoch_store
            .process_consensus_transactions_and_commit_boundary(
                all_transactions,
                &self.last_consensus_stats,
                &ConsensusCommitInfo::new(self.epoch_store.protocol_config(), &consensus_output),
            )
            .await
            .expect("Unrecoverable error in consensus handler");

        // update the calculated throughput
        self.throughput_calculator
            .add_transactions(timestamp, transactions_to_schedule.len() as u64);

        self.transaction_scheduler
            .schedule(transactions_to_schedule, commit_sub_dag_index)
            .await;
    }
}

struct AsyncTransactionScheduler {
    sender: tokio::sync::mpsc::Sender<(Vec<VerifiedExecutableTransaction>, CommitIndex)>,
}

impl AsyncTransactionScheduler {
    pub fn start(
        transaction_manager: Arc<TransactionManager>,
        epoch_store: Arc<AuthorityPerEpochStore>,
    ) -> Self {
        let (sender, recv) = tokio::sync::mpsc::channel(16);
        tokio::spawn(Self::run(recv, transaction_manager, epoch_store));
        Self { sender }
    }

    pub async fn schedule(
        &self,
        transactions: Vec<VerifiedExecutableTransaction>,
        commit: CommitIndex,
    ) {
        self.sender.send((transactions, commit)).await.ok();
    }

    pub async fn run(
        mut recv: tokio::sync::mpsc::Receiver<(Vec<VerifiedExecutableTransaction>, CommitIndex)>,
        transaction_manager: Arc<TransactionManager>,
        epoch_store: Arc<AuthorityPerEpochStore>,
    ) {
        while let Some((transactions, commit)) = recv.recv().await {
            transaction_manager.enqueue(transactions, &epoch_store, Some(commit));
        }
    }
}

/// Consensus handler used by Mysticeti. Since Mysticeti repo is not yet integrated, we use a
/// channel to receive the consensus output from Mysticeti.
/// During initialization, the sender is passed into Mysticeti which can send consensus output
/// to the channel.
pub struct MysticetiConsensusHandler {
    handle: Option<tokio::task::JoinHandle<()>>,
}

impl MysticetiConsensusHandler {
    pub fn new(
        mut consensus_handler: ConsensusHandler,
        mut receiver: UnboundedReceiver<CommittedSubDag>,
    ) -> Self {
        let handle = tokio::spawn(async move {
            while let Some(consensus_output) = receiver.recv().await {
                consensus_handler
                    .handle_consensus_output_internal(consensus_output)
                    .await;
            }
        });
        Self {
            handle: Some(handle),
        }
    }

    pub async fn abort(&mut self) {
        if let Some(handle) = self.handle.take() {
            handle.abort();
            let _ = handle.await;
        }
    }
}

impl Drop for MysticetiConsensusHandler {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            handle.abort();
        }
    }
}

impl ConsensusHandler {
    fn epoch(&self) -> EpochId {
        self.epoch_store.epoch()
    }

    pub fn last_executed_sub_dag_round(&self) -> u64 {
        self.last_consensus_stats.last_committed_round
    }

    pub fn last_executed_sub_dag_index(&self) -> u64 {
        self.last_consensus_stats.sub_dag_index
    }
}

pub(crate) fn classify(transaction: &ConsensusTransaction) -> &'static str {
    match &transaction.kind {
        ConsensusTransactionKind::UserTransaction(_) => "certificate",
        ConsensusTransactionKind::EndOfPublish(_) => "end_of_publish",
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

    pub fn is_executable_transaction(&self) -> bool {
        match self {
            SequencedConsensusTransactionKind::External(ext) => ext.is_user_certificate(),
            SequencedConsensusTransactionKind::System(_) => true,
        }
    }

    pub fn executable_transaction_digest(&self) -> Option<TransactionDigest> {
        match self {
            SequencedConsensusTransactionKind::External(ext) => {
                if let ConsensusTransactionKind::UserTransaction(txn) = &ext.kind {
                    Some(*txn.digest())
                } else {
                    None
                }
            }
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
        matches!(
            self.transaction,
            SequencedConsensusTransactionKind::System(_)
        )
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
            certificate_author_index: AuthorityIndex::new_for_test(0),
            certificate_author: AuthorityName::ZERO,
            consensus_index: Default::default(),
            transaction: SequencedConsensusTransactionKind::External(transaction),
        }
    }
}

/// Represents the information from the current consensus commit.
pub struct ConsensusCommitInfo {
    pub round: u64,
    pub timestamp: u64,
    pub consensus_commit_digest: ConsensusCommitDigest,

    #[cfg(any(test, feature = "test-utils"))]
    skip_consensus_commit_prologue_in_test: bool,
}

impl ConsensusCommitInfo {
    fn new(protocol_config: &ProtocolConfig, consensus_output: &impl ConsensusOutputAPI) -> Self {
        Self {
            round: consensus_output.leader_round(),
            timestamp: consensus_output.commit_timestamp_ms(),
            consensus_commit_digest: consensus_output.consensus_digest(protocol_config),

            #[cfg(any(test, feature = "test-utils"))]
            skip_consensus_commit_prologue_in_test: false,
        }
    }

    #[cfg(any(test, feature = "test-utils"))]
    pub fn new_for_test(
        commit_round: u64,
        commit_timestamp: u64,
        skip_consensus_commit_prologue_in_test: bool,
    ) -> Self {
        use types::digests::ConsensusCommitDigest;

        Self {
            round: commit_round,
            timestamp: commit_timestamp,
            consensus_commit_digest: ConsensusCommitDigest::default(),
            skip_consensus_commit_prologue_in_test,
        }
    }

    #[cfg(any(test, feature = "test-utils"))]
    pub fn skip_consensus_commit_prologue_in_test(&self) -> bool {
        self.skip_consensus_commit_prologue_in_test
    }

    pub fn create_consensus_commit_prologue_transaction(
        &self,
        epoch: u64,
    ) -> VerifiedExecutableTransaction {
        let transaction = VerifiedTransaction::new_consensus_commit_prologue(
            epoch,
            self.round,
            self.timestamp,
            self.consensus_commit_digest,
        );
        VerifiedExecutableTransaction::new_system(transaction, epoch)
    }
}
