use std::collections::{hash_map, BTreeMap, BTreeSet, HashMap, VecDeque};

use crate::{
    authority_per_epoch_store::{
        AuthorityEpochTables, AuthorityPerEpochStore, ExecutionIndicesWithStats,
        LAST_CONSENSUS_STATS_ADDR, RECONFIG_STATE_INDEX,
    },
    cache::cache_types::CacheResult,
    checkpoints::{BuilderCheckpointSummary, CheckpointHeight, PendingCheckpoint},
    consensus_handler::SequencedConsensusTransactionKey,
    fallback_fetch::do_fallback_lookup,
    reconfiguration::ReconfigState,
    shared_obj_version_manager::AssignedTxAndVersions,
    start_epoch::EpochStartConfiguration,
};
use dashmap::DashMap;
use moka::policy::EvictionPolicy;
use moka::sync::SegmentedCache as MokaCache;
use parking_lot::{Mutex, RwLock};
use store::{rocks::DBBatch, Map as _};
use tracing::{debug, info, trace};
use types::{
    base::{AuthorityName, ConsensusObjectSequenceKey},
    checkpoints::{CheckpointContents, CheckpointSequenceNumber},
    consensus::{commit::CommitIndex, Round},
    crypto::GenericSignature,
    digests::TransactionDigest,
    error::SomaResult,
    object::Version,
    transaction::{TransactionKey, VerifiedExecutableTransaction},
};

#[derive(Default)]
pub(crate) struct ConsensusCommitOutput {
    // Consensus and reconfig state
    consensus_round: Round,
    consensus_messages_processed: BTreeSet<SequencedConsensusTransactionKey>,
    end_of_publish: BTreeSet<AuthorityName>,
    reconfig_state: Option<ReconfigState>,
    consensus_commit_stats: Option<ExecutionIndicesWithStats>,

    // transaction scheduling state
    next_shared_object_versions: Option<HashMap<ConsensusObjectSequenceKey, Version>>,

    // checkpoint state
    pending_checkpoints: Vec<PendingCheckpoint>,
}

impl ConsensusCommitOutput {
    pub fn new(consensus_round: Round) -> Self {
        Self {
            consensus_round,
            ..Default::default()
        }
    }

    fn get_highest_pending_checkpoint_height(&self) -> Option<CheckpointHeight> {
        self.pending_checkpoints.last().map(|cp| cp.height())
    }

    fn get_pending_checkpoints(
        &self,
        last: Option<CheckpointHeight>,
    ) -> impl Iterator<Item = &PendingCheckpoint> {
        self.pending_checkpoints.iter().filter(move |cp| {
            if let Some(last) = last {
                cp.height() > last
            } else {
                true
            }
        })
    }

    fn pending_checkpoint_exists(&self, index: &CheckpointHeight) -> bool {
        self.pending_checkpoints
            .iter()
            .any(|cp| cp.height() == *index)
    }

    fn get_round(&self) -> Option<u64> {
        self.consensus_commit_stats
            .as_ref()
            .map(|stats| stats.index.last_committed_round)
    }

    pub fn insert_end_of_publish(&mut self, authority: AuthorityName) {
        self.end_of_publish.insert(authority);
    }

    pub(crate) fn record_consensus_commit_stats(&mut self, stats: ExecutionIndicesWithStats) {
        self.consensus_commit_stats = Some(stats);
    }

    // in testing code we often need to write to the db outside of a consensus commit
    pub(crate) fn set_default_commit_stats_for_testing(&mut self) {
        self.record_consensus_commit_stats(Default::default());
    }

    pub fn store_reconfig_state(&mut self, state: ReconfigState) {
        self.reconfig_state = Some(state);
    }

    pub fn record_consensus_message_processed(&mut self, key: SequencedConsensusTransactionKey) {
        self.consensus_messages_processed.insert(key);
    }

    pub fn get_consensus_messages_processed(
        &self,
    ) -> impl Iterator<Item = &SequencedConsensusTransactionKey> {
        self.consensus_messages_processed.iter()
    }

    pub fn set_next_shared_object_versions(
        &mut self,
        next_versions: HashMap<ConsensusObjectSequenceKey, Version>,
    ) {
        assert!(self.next_shared_object_versions.is_none());
        self.next_shared_object_versions = Some(next_versions);
    }

    pub fn insert_pending_checkpoint(&mut self, checkpoint: PendingCheckpoint) {
        self.pending_checkpoints.push(checkpoint);
    }

    pub fn write_to_batch(
        self,
        epoch_store: &AuthorityPerEpochStore,
        batch: &mut DBBatch,
    ) -> SomaResult {
        let tables = epoch_store.tables()?;
        batch.insert_batch(
            &tables.consensus_message_processed,
            self.consensus_messages_processed
                .iter()
                .map(|key| (key, true)),
        )?;

        batch.insert_batch(
            &tables.end_of_publish,
            self.end_of_publish.iter().map(|authority| (authority, ())),
        )?;

        if let Some(reconfig_state) = &self.reconfig_state {
            batch.insert_batch(
                &tables.reconfig_state,
                [(RECONFIG_STATE_INDEX, reconfig_state)],
            )?;
        }

        let consensus_commit_stats = self
            .consensus_commit_stats
            .expect("consensus_commit_stats must be set");
        let round = consensus_commit_stats.index.last_committed_round;

        batch.insert_batch(
            &tables.last_consensus_stats,
            [(LAST_CONSENSUS_STATS_ADDR, consensus_commit_stats)],
        )?;

        if let Some(next_versions) = self.next_shared_object_versions {
            batch.insert_batch(&tables.next_shared_object_versions, next_versions)?;
        }

        Ok(())
    }
}

/// ConsensusOutputCache holds outputs of consensus processing that do not need to be committed to disk.
/// Data quarantining guarantees that all of this data will be used (e.g. for building checkpoints)
/// before the consensus commit from which it originated is marked as processed. Therefore we can rely
/// on replay of consensus commits to recover this data.
pub(crate) struct ConsensusOutputCache {
    // user_signatures_for_checkpoints is written to by consensus handler and read from by checkpoint builder
    // The critical sections are small in both cases so a DashMap is probably not helpful.
    pub(crate) user_signatures_for_checkpoints:
        Mutex<HashMap<TransactionDigest, Vec<GenericSignature>>>,

    executed_in_epoch: RwLock<DashMap<TransactionDigest, ()>>,
    executed_in_epoch_cache: MokaCache<TransactionDigest, ()>,
}

impl ConsensusOutputCache {
    pub(crate) fn new(
        epoch_start_configuration: &EpochStartConfiguration,
        tables: &AuthorityEpochTables,
    ) -> Self {
        let executed_in_epoch_cache_capacity = 50_000;

        Self {
            user_signatures_for_checkpoints: Default::default(),
            executed_in_epoch: RwLock::new(DashMap::with_shard_amount(2048)),
            executed_in_epoch_cache: MokaCache::builder(8)
                // most queries should be for recent transactions
                .max_capacity(executed_in_epoch_cache_capacity)
                .eviction_policy(EvictionPolicy::lru())
                .build(),
        }
    }

    pub fn executed_in_current_epoch(&self, digest: &TransactionDigest) -> bool {
        self.executed_in_epoch
            .read()
            .contains_key(digest) ||
            // we use get instead of contains key to mark the entry as read
            self.executed_in_epoch_cache.get(digest).is_some()
    }

    // Called by execution
    pub fn insert_executed_in_epoch(&self, tx_digest: TransactionDigest) {
        assert!(
            self.executed_in_epoch
                .read()
                .insert(tx_digest, ())
                .is_none(),
            "transaction already executed"
        );
        self.executed_in_epoch_cache.insert(tx_digest, ());
    }

    // CheckpointExecutor calls this (indirectly) in order to prune the in-memory cache of executed
    // transactions. By the time this is called, the transaction digests will have been committed to
    // the `executed_transactions_to_checkpoint` table.
    pub fn remove_executed_in_epoch(&self, tx_digests: &[TransactionDigest]) {
        let executed_in_epoch = self.executed_in_epoch.read();
        for tx_digest in tx_digests {
            executed_in_epoch.remove(tx_digest);
        }
    }
}

/// ConsensusOutputQuarantine holds outputs of consensus processing in memory until the checkpoints
/// for the commit have been certified.
pub(crate) struct ConsensusOutputQuarantine {
    // Output from consensus handler
    output_queue: VecDeque<ConsensusCommitOutput>,

    // Highest known certified checkpoint sequence number
    highest_executed_checkpoint: CheckpointSequenceNumber,

    // Checkpoint Builder output
    builder_checkpoint_summary:
        BTreeMap<CheckpointSequenceNumber, (BuilderCheckpointSummary, CheckpointContents)>,

    builder_digest_to_checkpoint: HashMap<TransactionDigest, CheckpointSequenceNumber>,

    // Any un-committed next versions are stored here.
    shared_object_next_versions: RefCountedHashMap<ConsensusObjectSequenceKey, Version>,

    processed_consensus_messages: RefCountedHashMap<SequencedConsensusTransactionKey, ()>,
}

impl ConsensusOutputQuarantine {
    pub(super) fn new(highest_executed_checkpoint: CheckpointSequenceNumber) -> Self {
        Self {
            highest_executed_checkpoint,

            output_queue: VecDeque::new(),
            builder_checkpoint_summary: BTreeMap::new(),
            builder_digest_to_checkpoint: HashMap::new(),
            shared_object_next_versions: RefCountedHashMap::new(),
            processed_consensus_messages: RefCountedHashMap::new(),
        }
    }
}
// Write methods - all methods in this block insert new data into the quarantine.
// There are only two sources! ConsensusHandler and CheckpointBuilder.
impl ConsensusOutputQuarantine {
    // Push all data gathered from a consensus commit into the quarantine.
    pub(crate) fn push_consensus_output(
        &mut self,
        output: ConsensusCommitOutput,
        epoch_store: &AuthorityPerEpochStore,
    ) -> SomaResult {
        self.insert_shared_object_next_versions(&output);
        self.insert_processed_consensus_messages(&output);
        self.output_queue.push_back(output);

        // we may already have observed the certified checkpoint for this round, if state sync is running
        // ahead of consensus, so there may be data to commit right away.
        self.commit(epoch_store)
    }

    // Record a newly built checkpoint.
    pub(super) fn insert_builder_summary(
        &mut self,
        sequence_number: CheckpointSequenceNumber,
        summary: BuilderCheckpointSummary,
        contents: CheckpointContents,
    ) {
        debug!(?sequence_number, "inserting builder summary {:?}", summary);
        for tx in contents.iter() {
            self.builder_digest_to_checkpoint
                .insert(tx.transaction, sequence_number);
        }
        self.builder_checkpoint_summary
            .insert(sequence_number, (summary, contents));
    }
}

// Commit methods.
impl ConsensusOutputQuarantine {
    /// Update the highest executed checkpoint and commit any data which is now
    /// below the watermark.
    pub(super) fn update_highest_executed_checkpoint(
        &mut self,
        checkpoint: CheckpointSequenceNumber,
        epoch_store: &AuthorityPerEpochStore,
        batch: &mut DBBatch,
    ) -> SomaResult {
        self.highest_executed_checkpoint = checkpoint;
        self.commit_with_batch(epoch_store, batch)
    }

    pub(super) fn commit(&mut self, epoch_store: &AuthorityPerEpochStore) -> SomaResult {
        let mut batch = epoch_store.db_batch()?;
        self.commit_with_batch(epoch_store, &mut batch)?;
        batch.write()?;
        Ok(())
    }

    /// Commit all data below the watermark.
    fn commit_with_batch(
        &mut self,
        epoch_store: &AuthorityPerEpochStore,
        batch: &mut DBBatch,
    ) -> SomaResult {
        // The commit algorithm is simple:
        // 1. First commit all checkpoint builder state which is below the watermark.
        // 2. Determine the consensus commit height that corresponds to the highest committed
        //    checkpoint.
        // 3. Commit all consensus output at that height or below.

        let tables = epoch_store.tables()?;

        let mut highest_committed_height = None;

        while self
            .builder_checkpoint_summary
            .first_key_value()
            .map(|(seq, _)| *seq <= self.highest_executed_checkpoint)
            == Some(true)
        {
            let (seq, (builder_summary, contents)) =
                self.builder_checkpoint_summary.pop_first().unwrap();

            for tx in contents.iter() {
                let digest = &tx.transaction;
                assert_eq!(
                    self.builder_digest_to_checkpoint
                        .remove(digest)
                        .unwrap_or_else(|| {
                            panic!(
                                "transaction {:?} not found in builder_digest_to_checkpoint",
                                digest
                            )
                        }),
                    seq
                );
            }

            batch.insert_batch(
                &tables.builder_digest_to_checkpoint,
                contents.iter().map(|tx| (tx.transaction, seq)),
            )?;

            batch.insert_batch(
                &tables.builder_checkpoint_summary,
                [(seq, &builder_summary)],
            )?;

            let checkpoint_height = builder_summary
                .checkpoint_height
                .expect("non-genesis checkpoint must have height");
            if let Some(highest) = highest_committed_height {
                assert!(
                    checkpoint_height >= highest,
                    "current checkpoint height {} must be no less than highest committed height {}",
                    checkpoint_height,
                    highest
                );
            }

            highest_committed_height = Some(checkpoint_height);
        }

        let Some(highest_committed_height) = highest_committed_height else {
            return Ok(());
        };

        while !self.output_queue.is_empty() {
            // A consensus commit can have more than one pending checkpoint (a regular one and a randomnes one).
            // We can only write the consensus commit if the highest pending checkpoint associated with it has
            // been processed by the builder.
            let Some(highest_in_commit) = self
                .output_queue
                .front()
                .unwrap()
                .get_highest_pending_checkpoint_height()
            else {
                // if highest is none, we have already written the pending checkpoint for the final epoch,
                // so there is no more data that needs to be committed.
                break;
            };

            if highest_in_commit <= highest_committed_height {
                info!(
                    "committing output with highest pending checkpoint height {:?}",
                    highest_in_commit
                );
                let output = self.output_queue.pop_front().unwrap();
                self.remove_shared_object_next_versions(&output);
                self.remove_processed_consensus_messages(&output);

                output.write_to_batch(epoch_store, batch)?;
            } else {
                break;
            }
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
    pub(super) fn last_built_summary(&self) -> Option<&BuilderCheckpointSummary> {
        self.builder_checkpoint_summary
            .values()
            .last()
            .map(|(summary, _)| summary)
    }

    pub(super) fn get_built_summary(
        &self,
        sequence: CheckpointSequenceNumber,
    ) -> Option<&BuilderCheckpointSummary> {
        self.builder_checkpoint_summary
            .get(&sequence)
            .map(|(summary, _)| summary)
    }

    pub(super) fn included_transaction_in_checkpoint(&self, digest: &TransactionDigest) -> bool {
        self.builder_digest_to_checkpoint.contains_key(digest)
    }

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
        tables: &AuthorityEpochTables,
        objects_to_init: &[ConsensusObjectSequenceKey],
    ) -> SomaResult<Vec<Option<Version>>> {
        Ok(do_fallback_lookup(
            objects_to_init,
            |object_key| {
                if let Some(next_version) = self.shared_object_next_versions.get(object_key) {
                    CacheResult::Hit(Some(*next_version))
                } else {
                    CacheResult::Miss
                }
            },
            |object_keys| {
                tables
                    .next_shared_object_versions
                    .multi_get(object_keys)
                    .expect("db error")
            },
        ))
    }

    pub(super) fn get_highest_pending_checkpoint_height(&self) -> Option<CheckpointHeight> {
        self.output_queue
            .back()
            .and_then(|output| output.get_highest_pending_checkpoint_height())
    }

    pub(super) fn get_pending_checkpoints(
        &self,
        last: Option<CheckpointHeight>,
    ) -> Vec<(CheckpointHeight, PendingCheckpoint)> {
        let mut checkpoints = Vec::new();
        for output in &self.output_queue {
            checkpoints.extend(
                output
                    .get_pending_checkpoints(last)
                    .map(|cp| (cp.height(), cp.clone())),
            );
        }
        if cfg!(debug_assertions) {
            let mut prev = None;
            for (height, _) in &checkpoints {
                if let Some(prev) = prev {
                    assert!(prev < *height);
                }
                prev = Some(*height);
            }
        }
        checkpoints
    }

    pub(super) fn pending_checkpoint_exists(&self, index: &CheckpointHeight) -> bool {
        self.output_queue
            .iter()
            .any(|output| output.pending_checkpoint_exists(index))
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
