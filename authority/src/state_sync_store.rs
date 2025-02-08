use crate::cache::ExecutionCacheTraitPointers;
use crate::commit::CommitStore;
use parking_lot::Mutex;
use std::collections::BTreeMap;
use std::sync::Arc;
use types::accumulator::AccumulatorStore;
use types::committee::Authority;
use types::consensus::block::BlockAPI;
use types::consensus::block::EndOfEpochData;
use types::consensus::commit::CommitDigest;
use types::consensus::commit::CommittedSubDag;
use types::dag::dag_state::DagStore;
use types::storage::committee_store::CommitteeStore;
use types::storage::consensus::ConsensusStore;
use types::storage::read_store::ReadCommitteeStore;
use types::storage::storage_error::Error as StorageError;
use types::storage::storage_error::Result;
use types::{
    accumulator::CommitIndex,
    committee::{Committee, EpochId},
    digests::TransactionDigest,
    effects::TransactionEffects,
    object::{Object, ObjectID, Version},
    storage::{
        object_store::ObjectStore, read_store::ReadStore, write_store::WriteStore, ObjectKey,
    },
    transaction::VerifiedTransaction,
};

#[derive(Clone)]
pub struct StateSyncStore {
    cache_traits: ExecutionCacheTraitPointers,

    committee_store: Arc<CommitteeStore>,
    commit_store: Arc<CommitStore>,
    consensus_store: Arc<dyn ConsensusStore>,
    // in memory commit watermark sequence numbers
    highest_synced_commit: Arc<Mutex<Option<CommitIndex>>>,
}

impl StateSyncStore {
    pub fn new(
        cache_traits: ExecutionCacheTraitPointers,
        committee_store: Arc<CommitteeStore>,
        commit_store: Arc<CommitStore>,
        consensus_store: Arc<dyn ConsensusStore>,
    ) -> Self {
        Self {
            cache_traits,
            committee_store,
            commit_store,
            consensus_store,
            highest_synced_commit: Arc::new(Mutex::new(None)),
        }
    }

    pub fn get_objects(&self, object_keys: &[ObjectKey]) -> Result<Vec<Option<Object>>> {
        self.cache_traits
            .object_cache_reader
            .multi_get_objects_by_key(object_keys)
            .map_err(Into::into)
    }

    pub fn get_last_executed_commit(&self) -> Option<CommittedSubDag> {
        self.commit_store
            .get_highest_executed_commit()
            .expect("db error")
    }
}

impl ReadStore for StateSyncStore {
    fn get_commit_by_digest(&self, digest: &CommitDigest) -> Option<CommittedSubDag> {
        self.commit_store
            .get_commit_by_digest(digest)
            .expect("db error")
    }

    fn get_commit_by_index(&self, index: CommitIndex) -> Option<CommittedSubDag> {
        self.commit_store
            .get_commit_by_index(index)
            .expect("db error")
    }

    fn get_highest_synced_commit(&self) -> Result<CommittedSubDag, StorageError> {
        self.commit_store
            .get_highest_synced_commit()
            .map(|maybe_commit| {
                maybe_commit.expect("storage should have been initialized with genesis commit")
            })
            .map_err(Into::into)
    }

    fn get_lowest_available_commit(&self) -> Result<CommitIndex, StorageError> {
        // TODO: update this to work with pruning
        // let highest_pruned_cp = self
        //     .commit_store
        //     .get_highest_pruned_commit_seq_number()
        //     .map_err(Into::<StorageError>::into)?;

        // if highest_pruned_cp == 0 {
        //     Ok(0)
        // } else {
        //     Ok(highest_pruned_cp + 1)
        // }

        Ok(0)
    }

    fn get_transaction(
        &self,
        digest: &TransactionDigest,
    ) -> Result<Option<Arc<VerifiedTransaction>>> {
        self.cache_traits
            .transaction_cache_reader
            .get_transaction_block(digest)
            .map_err(Into::into)
    }

    fn get_transaction_effects(
        &self,
        digest: &TransactionDigest,
    ) -> Result<Option<TransactionEffects>> {
        self.cache_traits
            .transaction_cache_reader
            .get_executed_effects(digest)
            .map_err(Into::into)
    }
}

impl DagStore for StateSyncStore {}

impl ReadCommitteeStore for StateSyncStore {
    fn get_committee(&self, epoch: EpochId) -> Result<Option<Arc<Committee>>> {
        self.committee_store
            .get_committee(&epoch)
            .map_err(Into::into)
    }
}

impl ObjectStore for StateSyncStore {
    fn get_object(&self, object_id: &ObjectID) -> Result<Option<Object>> {
        self.cache_traits.object_store.get_object(object_id)
    }

    fn get_object_by_key(&self, object_id: &ObjectID, version: Version) -> Result<Option<Object>> {
        self.cache_traits
            .object_store
            .get_object_by_key(object_id, version)
    }
}

impl WriteStore for StateSyncStore {
    fn insert_commit(
        &self,
        commit: CommittedSubDag,
    ) -> Result<(), types::storage::storage_error::Error> {
        if let Some(Some(EndOfEpochData {
            next_validator_set, ..
        })) = commit
            .get_end_of_epoch_block()
            .map(|b| b.end_of_epoch_data())
        {
            if let Some(next_validator_set) = next_validator_set {
                let voting_rights: BTreeMap<_, _> = next_validator_set
                    .0
                    .iter()
                    .map(|(name, stake, _)| (*name, *stake))
                    .collect();

                let authorities = next_validator_set
                    .0
                    .iter()
                    .map(|(name, stake, meta)| {
                        (
                            *name,
                            Authority {
                                stake: *stake,
                                address: meta.consensus_address.clone(),
                                hostname: meta.hostname.clone(),
                                protocol_key: meta.protocol_key.clone(),
                                network_key: meta.network_key.clone(),
                                authority_key: meta.authority_key.clone(),
                            },
                        )
                    })
                    .collect();
                let committee = Committee::new(
                    commit
                        .blocks
                        .last()
                        .unwrap()
                        .epoch()
                        .checked_add(1)
                        .unwrap(),
                    voting_rights,
                    authorities,
                );
                self.insert_committee(committee)?;
            }
        }

        self.commit_store.insert_commit(commit).map_err(Into::into)
    }

    fn insert_committee(&self, new_committee: Committee) -> Result<()> {
        self.committee_store
            .insert_new_committee(new_committee)
            .unwrap();
        Ok(())
    }

    fn update_highest_synced_commit(
        &self,
        commit: &CommittedSubDag,
    ) -> Result<(), types::storage::storage_error::Error> {
        let mut locked = self.highest_synced_commit.lock();
        if locked.is_some() && locked.unwrap() >= commit.commit_ref.index {
            return Ok(());
        }
        self.commit_store
            .update_highest_synced_commit(commit)
            .map_err(types::storage::storage_error::Error::custom)?;
        *locked = Some(commit.commit_ref.index);
        Ok(())
    }
}

impl ConsensusStore for StateSyncStore {
    fn write(
        &self,
        write_batch: types::storage::consensus::WriteBatch,
    ) -> types::error::ConsensusResult<()> {
        self.consensus_store.write(write_batch)
    }

    fn read_blocks(
        &self,
        refs: &[types::consensus::block::BlockRef],
    ) -> types::error::ConsensusResult<Vec<Option<types::consensus::block::VerifiedBlock>>> {
        self.consensus_store.read_blocks(refs)
    }

    fn contains_blocks(
        &self,
        refs: &[types::consensus::block::BlockRef],
    ) -> types::error::ConsensusResult<Vec<bool>> {
        self.consensus_store.contains_blocks(refs)
    }

    fn contains_block_at_slot(
        &self,
        slot: types::consensus::block::Slot,
    ) -> types::error::ConsensusResult<bool> {
        self.consensus_store.contains_block_at_slot(slot)
    }

    fn scan_blocks_by_author(
        &self,
        authority: types::committee::AuthorityIndex,
        start_round: types::consensus::block::Round,
    ) -> types::error::ConsensusResult<Vec<types::consensus::block::VerifiedBlock>> {
        self.consensus_store
            .scan_blocks_by_author(authority, start_round)
    }

    fn scan_last_blocks_by_author(
        &self,
        author: types::committee::AuthorityIndex,
        num_of_rounds: u64,
        before_round: Option<types::consensus::block::Round>,
    ) -> types::error::ConsensusResult<Vec<types::consensus::block::VerifiedBlock>> {
        self.consensus_store
            .scan_last_blocks_by_author(author, num_of_rounds, before_round)
    }

    fn read_last_commit(
        &self,
    ) -> types::error::ConsensusResult<Option<types::consensus::commit::TrustedCommit>> {
        self.consensus_store.read_last_commit()
    }

    fn scan_commits(
        &self,
        range: types::consensus::commit::CommitRange,
    ) -> types::error::ConsensusResult<Vec<types::consensus::commit::TrustedCommit>> {
        self.consensus_store.scan_commits(range)
    }

    fn read_commit_votes(
        &self,
        commit_index: types::consensus::commit::CommitIndex,
    ) -> types::error::ConsensusResult<Vec<types::consensus::block::BlockRef>> {
        self.consensus_store.read_commit_votes(commit_index)
    }

    fn read_last_commit_info(
        &self,
    ) -> types::error::ConsensusResult<
        Option<(
            types::consensus::commit::CommitRef,
            types::consensus::commit::CommitInfo,
        )>,
    > {
        self.consensus_store.read_last_commit_info()
    }
}

impl AccumulatorStore for StateSyncStore {
    fn get_root_state_accumulator_for_commit(
        &self,
        commit: CommitIndex,
    ) -> types::error::SomaResult<Option<types::accumulator::Accumulator>> {
        self.cache_traits
            .accumulator_store
            .get_root_state_accumulator_for_commit(commit)
    }

    fn get_root_state_accumulator_for_highest_commit(
        &self,
    ) -> types::error::SomaResult<Option<(CommitIndex, types::accumulator::Accumulator)>> {
        self.cache_traits
            .accumulator_store
            .get_root_state_accumulator_for_highest_commit()
    }

    fn insert_state_accumulator_for_commit(
        &self,
        commit: &CommitIndex,
        acc: &types::accumulator::Accumulator,
    ) -> types::error::SomaResult {
        self.cache_traits
            .accumulator_store
            .insert_state_accumulator_for_commit(commit, acc)
    }

    fn iter_live_object_set(&self) -> Box<dyn Iterator<Item = types::object::LiveObject> + '_> {
        self.cache_traits.accumulator_store.iter_live_object_set()
    }
}
