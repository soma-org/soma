use std::collections::{hash_map, BTreeSet, HashMap, VecDeque};

use dashmap::DashMap;
use store::Map as _;
use tracing::{info, trace};
use types::{
    base::{AuthorityName, ConsensusObjectSequenceKey},
    consensus::commit::CommitIndex,
    digests::TransactionDigest,
    error::SomaResult,
    execution_indices::ExecutionIndices,
    object::Version,
    transaction::{TransactionKey, VerifiedExecutableTransaction},
};

use crate::{
    cache::writeback_cache::{do_fallback_lookup, CacheResult},
    consensus_handler::SequencedConsensusTransactionKey,
    epoch_store::{AuthorityEpochTables, AuthorityPerEpochStore},
    reconfiguration::ReconfigState,
    shared_obj_version_manager::AssignedTxAndVersions,
    start_epoch::EpochStartConfiguration,
};

/// The key where the latest consensus index is stored in the database.
// TODO: Make a single table (e.g., called `variables`) storing all our lonely variables in one place.
pub(crate) const LAST_CONSENSUS_STATS_ADDR: u64 = 0;
pub(crate) const RECONFIG_STATE_INDEX: u64 = 0;

#[derive(Default)]
pub(crate) struct ConsensusCommitOutput {
    // Consensus and reconfig state
    consensus_messages_processed: BTreeSet<SequencedConsensusTransactionKey>,
    end_of_publish: BTreeSet<AuthorityName>,
    reconfig_state: Option<ReconfigState>,
    pending_execution: Vec<VerifiedExecutableTransaction>,
    consensus_commit_stats: Option<ExecutionIndices>,

    // transaction scheduling state
    next_shared_object_versions: Option<HashMap<ConsensusObjectSequenceKey, Version>>,
    // TODO: If we delay committing consensus output until after all deferrals have been loaded,
    // we can move deferred_txns to the ConsensusOutputCache and save disk bandwidth.
    // deferred_txns: Vec<(DeferralKey, Vec<VerifiedSequencedConsensusTransaction>)>,
    // deferred txns that have been loaded and can be removed
    // deleted_deferred_txns: BTreeSet<DeferralKey>,

    // congestion control state
    // congestion_control_object_debts: Vec<(ObjectID, u64)>,
    // congestion_control_randomness_object_debts: Vec<(ObjectID, u64)>,
    // execution_time_observations: Vec<(
    //     AuthorityIndex,
    //     u64, /* generation */
    //     Vec<(ExecutionTimeObservationKey, Duration)>,
    // )>,
}

impl ConsensusCommitOutput {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn insert_end_of_publish(&mut self, authority: AuthorityName) {
        self.end_of_publish.insert(authority);
    }

    pub fn insert_pending_execution(&mut self, transactions: &[VerifiedExecutableTransaction]) {
        self.pending_execution.reserve(transactions.len());
        self.pending_execution.extend_from_slice(transactions);
    }

    pub fn store_reconfig_state(&mut self, state: ReconfigState) {
        self.reconfig_state = Some(state);
    }

    pub fn record_consensus_message_processed(&mut self, key: SequencedConsensusTransactionKey) {
        self.consensus_messages_processed.insert(key);
    }

    pub fn record_consensus_commit_stats(&mut self, stats: ExecutionIndices) {
        self.consensus_commit_stats = Some(stats);
    }

    pub fn set_next_shared_object_versions(
        &mut self,
        next_versions: HashMap<ConsensusObjectSequenceKey, Version>,
    ) {
        assert!(self.next_shared_object_versions.is_none());
        self.next_shared_object_versions = Some(next_versions);
    }

    pub fn write_to_batch(self, epoch_store: &AuthorityPerEpochStore) -> SomaResult {
        let tables = epoch_store.tables()?;

        // Create a batch for all operations
        let mut batch = tables.consensus_message_processed.batch();

        // Add consensus messages to batch
        batch.insert_batch(
            &tables.consensus_message_processed,
            self.consensus_messages_processed
                .iter()
                .map(|key| (key.clone(), true)),
        )?;

        // Add end_of_publish to batch
        batch.insert_batch(
            &tables.end_of_publish,
            self.end_of_publish.iter().map(|authority| (*authority, ())),
        )?;

        // Add reconfig state if present
        if let Some(reconfig_state) = &self.reconfig_state {
            batch.insert_batch(
                &tables.reconfig_state,
                [(RECONFIG_STATE_INDEX, reconfig_state.clone())],
            )?;
        }

        // Add consensus commit stats if present
        if let Some(consensus_commit_stats) = &self.consensus_commit_stats {
            batch.insert_batch(
                &tables.last_consensus_stats,
                [(LAST_CONSENSUS_STATS_ADDR, consensus_commit_stats.clone())],
            )?;
        }

        // Add pending execution
        batch.insert_batch(
            &tables.pending_execution,
            self.pending_execution
                .into_iter()
                .map(|tx| (*tx.inner().digest(), tx.serializable())),
        )?;

        // Add next shared object versions if present
        if let Some(next_versions) = self.next_shared_object_versions {
            batch.insert_batch(
                &tables.next_shared_object_versions,
                next_versions.into_iter(),
            )?;
        }

        // Write the batch
        batch.write()?;

        Ok(())
    }
}

/// ConsensusOutputCache holds outputs of consensus processing that do not need to be committed to disk.
/// Data quarantining guarantees that all of this data will be used
/// before the consensus commit from which it originated is marked as processed. Therefore we can rely
/// on replay of consensus commits to recover this data.
pub(crate) struct ConsensusOutputCache {
    // shared version assignments is a DashMap because it is read from execution so we don't
    // want contention.
    shared_version_assignments: DashMap<TransactionKey, Vec<(ConsensusObjectSequenceKey, Version)>>,
    // deferred transactions is only used by consensus handler so there should never be lock contention
    // - hence no need for a DashMap.
    // pub(super) deferred_transactions:
    //     Mutex<BTreeMap<DeferralKey, Vec<VerifiedSequencedConsensusTransaction>>>,
}

impl ConsensusOutputCache {
    pub(crate) fn new(
        epoch_start_configuration: &EpochStartConfiguration,
        tables: &AuthorityEpochTables,
    ) -> Self {
        // let deferred_transactions = tables
        //     .get_all_deferred_transactions()
        //     .expect("load deferred transactions cannot fail");

        Self {
            shared_version_assignments: Default::default(),
            // deferred_transactions: Mutex::new(deferred_transactions),
        }
    }

    pub fn num_shared_version_assignments(&self) -> usize {
        self.shared_version_assignments.len()
    }

    pub fn get_assigned_shared_object_versions(
        &self,
        key: &TransactionKey,
    ) -> Option<Vec<(ConsensusObjectSequenceKey, Version)>> {
        let output = self
            .shared_version_assignments
            .get(key)
            .map(|locks| locks.clone());
        info!(
            "get_assigned_shared_object_versions: {:?} -> {:?}",
            key, output
        );
        output
    }

    pub fn insert_shared_object_assignments(&self, versions: &AssignedTxAndVersions) {
        trace!("insert_shared_object_assignments: {:?}", versions);
        let mut inserted_count = 0;
        for (key, value) in versions {
            if self
                .shared_version_assignments
                .insert(*key, value.clone())
                .is_none()
            {
                inserted_count += 1;
            }
        }
    }

    pub fn set_shared_object_versions_for_testing(
        &self,
        tx_digest: &TransactionDigest,
        assigned_versions: &[(ConsensusObjectSequenceKey, Version)],
    ) {
        self.shared_version_assignments.insert(
            TransactionKey::Digest(*tx_digest),
            assigned_versions.to_owned(),
        );
    }

    pub fn remove_shared_object_assignments<'a>(
        &self,
        keys: impl IntoIterator<Item = &'a TransactionKey>,
    ) {
        let mut removed_count = 0;
        for tx_key in keys {
            if self.shared_version_assignments.remove(tx_key).is_some() {
                removed_count += 1;
            }
        }
    }
}

/// ConsensusOutputQuarantine holds outputs of consensus processing in memory until the checkpoints
/// for the commit have been certified.
pub(crate) struct ConsensusOutputQuarantine {
    // Output from consensus handler
    output_queue: VecDeque<ConsensusCommitOutput>,

    // Highest known commit index
    highest_executed_commit: CommitIndex,

    // Any un-committed next versions are stored here.
    shared_object_next_versions: RefCountedHashMap<ConsensusObjectSequenceKey, Version>,

    // The most recent congestion control debts for objects. Uses a ref-count to track
    // which objects still exist in some element of output_queue.
    // congestion_control_randomness_object_debts:
    //     RefCountedHashMap<ObjectID, CongestionPerObjectDebt>,
    // congestion_control_object_debts: RefCountedHashMap<ObjectID, CongestionPerObjectDebt>,
    processed_consensus_messages: RefCountedHashMap<SequencedConsensusTransactionKey, ()>,
}

impl ConsensusOutputQuarantine {
    pub(super) fn new(highest_executed_commit: CommitIndex) -> Self {
        Self {
            highest_executed_commit,

            output_queue: VecDeque::new(),
            shared_object_next_versions: RefCountedHashMap::new(),
            processed_consensus_messages: RefCountedHashMap::new(),
        }
    }
}

// Write methods - all methods in this block insert new data into the quarantine.
// There are only two sources! ConsensusHandler and CheckpointBuilder.
impl ConsensusOutputQuarantine {
    // Push all data gathered from a consensus commit into the quarantine.
    pub(super) fn push_consensus_output(
        &mut self,
        output: ConsensusCommitOutput,
        epoch_store: &AuthorityPerEpochStore,
    ) -> SomaResult {
        self.insert_shared_object_next_versions(&output);
        // self.insert_congestion_control_debts(&output);
        self.insert_processed_consensus_messages(&output);
        self.output_queue.push_back(output);

        // we may already have observed the certified checkpoint for this round, if state sync is running
        // ahead of consensus, so there may be data to commit right away.
        self.commit(epoch_store)
    }
}

// Commit methods.
impl ConsensusOutputQuarantine {
    /// Update the highest commit checkpoint and commit any data which is now
    /// below the watermark.
    pub(super) fn update_highest_executed_commit(
        &mut self,
        commit: CommitIndex,
        epoch_store: &AuthorityPerEpochStore,
    ) -> SomaResult {
        self.highest_executed_commit = commit;
        self.commit_with_batch(epoch_store)
    }

    pub(super) fn commit(&mut self, epoch_store: &AuthorityPerEpochStore) -> SomaResult {
        self.commit_with_batch(epoch_store)?;
        Ok(())
    }

    /// Commit all data below the watermark.
    fn commit_with_batch(&mut self, epoch_store: &AuthorityPerEpochStore) -> SomaResult {
        // The commit algorithm is simple:
        // 1. First commit all  state which is below the watermark.
        // 3. Commit all consensus output at that height or below.

        let tables = epoch_store.tables()?;

        // Process all items in the queue that are below or at the watermark
        while !self.output_queue.is_empty() {
            // Peek at the next output to see if it should be committed
            // let output_indices = self
            //     .output_queue
            //     .front()
            //     .and_then(|output| output.consensus_commit_stats.as_ref());

            let output = self.output_queue.pop_front().unwrap();
            // self.remove_shared_object_next_versions(&output);
            // self.remove_processed_consensus_messages(&output);
            output.write_to_batch(epoch_store)?;

            // if let Some(indices) = output_indices {
            //     // Compare the output's sub_dag_index (or equivalent) with highest_executed_commit
            //     // Only commit if the output's index is <= highest_executed_commit
            //     if indices.sub_dag_index <= self.highest_executed_commit.into() {
            //         let output = self.output_queue.pop_front().unwrap();

            //         // Remove all tracked state for this output
            //         self.remove_shared_object_next_versions(&output);
            //         self.remove_processed_consensus_messages(&output);
            //         // self.remove_congestion_control_debts(&output);

            //         // Remove shared version assignments if needed
            //         // Note: Adjust this part based on your implementation
            //         // epoch_store.remove_shared_version_assignments(
            //         //     output.pending_execution.iter().map(|tx| tx.inner().digest()),
            //         // );

            //         // Write the output to storage
            //         output.write_to_batch(epoch_store)?;
            //     } else {
            //         // This output is not eligible for commit yet
            //         info!("Not eligble for commit in quarantine yet.");
            //         break;
            //     }
            // } else {
            //     // No consensus indices available, can't make a decision
            //     // Either commit it anyway or break depending on your requirements
            //     let output = self.output_queue.pop_front().unwrap();
            //     self.remove_shared_object_next_versions(&output);
            //     self.remove_processed_consensus_messages(&output);
            //     output.write_to_batch(epoch_store)?;
            // }
        }

        Ok(())
    }
}

impl ConsensusOutputQuarantine {
    fn insert_shared_object_next_versions(&mut self, output: &ConsensusCommitOutput) {
        if let Some(next_versions) = output.next_shared_object_versions.as_ref() {
            for (object_id, next_version) in next_versions {
                self.shared_object_next_versions
                    .insert(*object_id, *next_version);
            }
        }
    }

    // fn insert_congestion_control_debts(&mut self, output: &ConsensusCommitOutput) {
    //     let current_round = output.consensus_round;

    //     for (object_id, debt) in output.congestion_control_object_debts.iter() {
    //         self.congestion_control_object_debts.insert(
    //             *object_id,
    //             CongestionPerObjectDebt::new(current_round, *debt),
    //         );
    //     }

    //     for (object_id, debt) in output.congestion_control_randomness_object_debts.iter() {
    //         self.congestion_control_randomness_object_debts.insert(
    //             *object_id,
    //             CongestionPerObjectDebt::new(current_round, *debt),
    //         );
    //     }
    // }

    // fn remove_congestion_control_debts(&mut self, output: &ConsensusCommitOutput) {
    //     for (object_id, _) in output.congestion_control_object_debts.iter() {
    //         self.congestion_control_object_debts.remove(object_id);
    //     }
    //     for (object_id, _) in output.congestion_control_randomness_object_debts.iter() {
    //         self.congestion_control_randomness_object_debts
    //             .remove(object_id);
    //     }
    // }

    fn insert_processed_consensus_messages(&mut self, output: &ConsensusCommitOutput) {
        for tx_key in output.consensus_messages_processed.iter() {
            self.processed_consensus_messages.insert(tx_key.clone(), ());
        }
    }

    fn remove_processed_consensus_messages(&mut self, output: &ConsensusCommitOutput) {
        for tx_key in output.consensus_messages_processed.iter() {
            self.processed_consensus_messages.remove(tx_key);
        }
    }

    fn remove_shared_object_next_versions(&mut self, output: &ConsensusCommitOutput) {
        if let Some(next_versions) = output.next_shared_object_versions.as_ref() {
            for object_id in next_versions.keys() {
                if !self.shared_object_next_versions.remove(object_id) {
                    panic!(
                        "Shared object next version not found in quarantine: {:?}",
                        object_id
                    );
                }
            }
        }
    }
}

// Read methods - all methods in this block return data from the quarantine which would otherwise
// be found in the database.
impl ConsensusOutputQuarantine {
    pub(super) fn is_consensus_message_processed(
        &self,
        key: &SequencedConsensusTransactionKey,
    ) -> bool {
        self.processed_consensus_messages.contains_key(key)
    }

    pub(super) fn is_empty(&self) -> bool {
        self.output_queue.is_empty()
    }

    pub(super) fn get_next_shared_object_versions(
        &self,
        epoch_start_config: &EpochStartConfiguration,
        tables: &AuthorityEpochTables,
        objects_to_init: &[ConsensusObjectSequenceKey],
    ) -> SomaResult<Vec<Option<Version>>> {
        do_fallback_lookup(
            objects_to_init,
            |object_key| {
                if let Some(next_version) = self.shared_object_next_versions.get(object_key) {
                    Ok(CacheResult::Hit(Some(*next_version)))
                } else {
                    Ok(CacheResult::Miss)
                }
            },
            |object_keys| {
                // Use multi_get for batch lookup from DBMap
                let results = tables.next_shared_object_versions.multi_get(object_keys)?;

                Ok(results.into_iter().collect())
            },
        )
    }
}

// A wrapper around HashMap that uses refcounts to keep entries alive until
// they are no longer needed.
//
// If there are N inserts for the same key, the key will not be removed until
// there are N removes.
//
// It is intended to track the *latest* value for a given key, so duplicate
// inserts are intended to overwrite any prior value.
#[derive(Debug, Default)]
struct RefCountedHashMap<K, V> {
    map: HashMap<K, (usize, V)>,
}

impl<K, V> RefCountedHashMap<K, V>
where
    K: Clone + Eq + std::hash::Hash,
{
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
        }
    }

    pub fn insert(&mut self, key: K, value: V) {
        let entry = self.map.entry(key);
        match entry {
            hash_map::Entry::Occupied(mut entry) => {
                let (ref_count, v) = entry.get_mut();
                *ref_count += 1;
                *v = value;
            }
            hash_map::Entry::Vacant(entry) => {
                entry.insert((1, value));
            }
        }
    }

    // Returns true if the key was present, false otherwise.
    // Note that the key may not be removed if present, as it may have a refcount > 1.
    pub fn remove(&mut self, key: &K) -> bool {
        let entry = self.map.entry(key.clone());
        match entry {
            hash_map::Entry::Occupied(mut entry) => {
                let (ref_count, _) = entry.get_mut();
                *ref_count -= 1;
                if *ref_count == 0 {
                    entry.remove();
                }
                true
            }
            hash_map::Entry::Vacant(_) => false,
        }
    }

    pub fn get(&self, key: &K) -> Option<&V> {
        self.map.get(key).map(|(_, v)| v)
    }

    pub fn contains_key(&self, key: &K) -> bool {
        self.map.contains_key(key)
    }
}
