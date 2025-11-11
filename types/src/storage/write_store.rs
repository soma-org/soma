use std::{collections::HashMap, io::Read, sync::Arc};

use crate::{
    accumulator::{AccumulatorStore, CommitIndex},
    committee::{Committee, Epoch},
    consensus::{
        block::{BlockAPI as _, BlockRef, VerifiedBlock},
        commit::{
            Commit, CommitAPI as _, CommitDigest, CommitInfo, CommittedSubDag, TrustedCommit,
        },
    },
    digests::TransactionDigest,
    effects::TransactionEffects,
    transaction::VerifiedTransaction,
};
use anyhow::anyhow;
use parking_lot::RwLock;
use tap::Pipe as _;

use super::{
    consensus::ConsensusStore,
    object_store::ObjectStore,
    read_store::{ReadCommitteeStore, ReadStore},
    storage_error::{self, Error, Result},
};

pub trait WriteStore: ReadStore {
    fn insert_committee(&self, new_committee: Committee) -> Result<()>;
    fn insert_commit(&self, commit: CommittedSubDag) -> Result<()>;
    fn update_highest_synced_commit(&self, commit: &CommittedSubDag) -> Result<()>;
}

impl<T: WriteStore + ?Sized> WriteStore for &T {
    fn insert_commit(&self, commit: CommittedSubDag) -> Result<()> {
        (*self).insert_commit(commit)
    }

    fn insert_committee(&self, new_committee: Committee) -> Result<()> {
        (*self).insert_committee(new_committee)
    }

    fn update_highest_synced_commit(&self, commit: &CommittedSubDag) -> Result<()> {
        (*self).update_highest_synced_commit(commit)
    }
}

#[derive(Clone, Debug)]
pub struct TestP2pStore {
    inner: Arc<RwLock<TestP2pStoreInner>>,
}

#[derive(Debug, Default)]
struct TestP2pStoreInner {
    commits: HashMap<CommitIndex, CommittedSubDag>,
    commit_by_digest: HashMap<CommitDigest, CommittedSubDag>,
    blocks: HashMap<BlockRef, VerifiedBlock>,
    highest_synced_commit: Option<CommittedSubDag>,
    lowest_available_commit: CommitIndex,
    committees: HashMap<Epoch, Arc<Committee>>,
    transactions: HashMap<TransactionDigest, Arc<VerifiedTransaction>>,
    effects: HashMap<TransactionDigest, TransactionEffects>,

    trusted_commits: HashMap<CommitIndex, TrustedCommit>,
    commit_info: HashMap<CommitIndex, CommitInfo>,
    highest_pruned_commit_index: Option<CommitIndex>,
}

impl TestP2pStore {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(TestP2pStoreInner::default())),
        }
    }

    pub fn insert_genesis_state(&self, genesis_commit: CommittedSubDag, committee: Committee) {
        let mut inner = self.inner.write();

        // Insert committee
        inner
            .committees
            .insert(committee.epoch(), Arc::new(committee));

        // Insert genesis blocks
        for block in &genesis_commit.blocks {
            inner.blocks.insert(block.reference(), block.clone());
        }

        // Insert genesis commit
        inner
            .commits
            .insert(genesis_commit.commit_ref.index, genesis_commit.clone());
        inner
            .commit_by_digest
            .insert(genesis_commit.commit_ref.digest, genesis_commit.clone());
        inner.highest_synced_commit = Some(genesis_commit);
        inner.lowest_available_commit = 0;
    }

    pub fn insert_commit_with_blocks(&self, commit: &CommittedSubDag) {
        let mut inner = self.inner.write();

        // Insert all blocks
        for block in &commit.blocks {
            inner.blocks.insert(block.reference(), block.clone());
        }

        // Insert commit
        inner
            .commits
            .insert(commit.commit_ref.index, commit.clone());
        inner
            .commit_by_digest
            .insert(commit.commit_ref.digest, commit.clone());

        // IMPORTANT: Also create and insert TrustedCommit for scan_commits to work
        let commit_obj = Commit::new(
            commit.commit_ref.index,
            commit.previous_digest,
            commit.timestamp_ms,
            commit.leader,
            commit.blocks.iter().map(|b| b.reference()).collect(),
            commit.epoch(),
        );

        let serialized = commit_obj.serialize().expect("Failed to serialize commit");
        let trusted_commit = TrustedCommit::new_trusted(commit_obj, serialized);

        inner
            .trusted_commits
            .insert(commit.commit_ref.index, trusted_commit);
    }

    pub fn delete_commit_content_test_only(&self, commit_index: CommitIndex) -> Result<()> {
        let mut inner = self.inner.write();

        // Clone the commit to avoid holding a reference while mutating
        let commit = inner.commits.get(&commit_index).cloned();

        if let Some(commit) = commit {
            // Remove blocks but keep commit metadata
            for block in &commit.blocks {
                inner.blocks.remove(&block.reference());
            }
            inner.lowest_available_commit = commit_index + 1;
        }
        Ok(())
    }

    pub fn get_lowest_available_commit_inner(&self) -> CommitIndex {
        self.inner.read().lowest_available_commit
    }

    pub fn get_highest_synced_commit_inner(&self) -> Option<CommittedSubDag> {
        self.inner.read().highest_synced_commit.clone()
    }
}

impl AccumulatorStore for TestP2pStore {
    fn get_root_state_accumulator_for_epoch(
        &self,
        epoch: crate::committee::EpochId,
    ) -> crate::error::SomaResult<Option<(CommitIndex, crate::accumulator::Accumulator)>> {
        todo!()
    }

    fn get_root_state_accumulator_for_highest_epoch(
        &self,
    ) -> crate::error::SomaResult<
        Option<(
            crate::committee::EpochId,
            (CommitIndex, crate::accumulator::Accumulator),
        )>,
    > {
        todo!()
    }

    fn insert_state_accumulator_for_epoch(
        &self,
        epoch: crate::committee::EpochId,
        commit: &CommitIndex,
        acc: &crate::accumulator::Accumulator,
    ) -> crate::error::SomaResult {
        todo!()
    }

    fn iter_live_object_set(&self) -> Box<dyn Iterator<Item = crate::object::LiveObject> + '_> {
        todo!()
    }

    fn iter_cached_live_object_set_for_testing(
        &self,
    ) -> Box<dyn Iterator<Item = crate::object::LiveObject> + '_> {
        todo!()
    }
}

impl WriteStore for TestP2pStore {
    fn insert_commit(&self, commit: CommittedSubDag) -> Result<()> {
        self.insert_commit_with_blocks(&commit);
        Ok(())
    }

    fn update_highest_synced_commit(&self, commit: &CommittedSubDag) -> Result<()> {
        let mut inner = self.inner.write();
        inner.highest_synced_commit = Some(commit.clone());
        Ok(())
    }

    fn insert_committee(&self, new_committee: Committee) -> Result<()> {
        let mut inner = self.inner.write();
        inner
            .committees
            .insert(new_committee.epoch(), Arc::new(new_committee));
        Ok(())
    }
}

impl ObjectStore for TestP2pStore {
    fn get_object(
        &self,
        object_id: &crate::object::ObjectID,
    ) -> Result<Option<crate::object::Object>> {
        todo!()
    }

    fn get_object_by_key(
        &self,
        object_id: &crate::object::ObjectID,
        version: crate::object::Version,
    ) -> Result<Option<crate::object::Object>> {
        todo!()
    }
}

impl ReadCommitteeStore for TestP2pStore {
    fn get_committee(&self, epoch: crate::committee::EpochId) -> Result<Option<Arc<Committee>>> {
        Ok(self.inner.read().committees.get(&epoch).cloned())
    }
}

impl ReadStore for TestP2pStore {
    fn get_highest_synced_commit(&self) -> Result<CommittedSubDag> {
        self.inner
            .read()
            .highest_synced_commit
            .clone()
            .ok_or_else(|| Error::missing(anyhow!("no highest synced commit")))
    }

    fn get_lowest_available_commit(&self) -> Result<CommitIndex> {
        Ok(self.inner.read().lowest_available_commit)
    }

    fn get_commit_by_index(&self, index: CommitIndex) -> Option<CommittedSubDag> {
        self.inner.read().commits.get(&index).cloned()
    }

    fn get_commit_by_digest(&self, digest: &CommitDigest) -> Option<CommittedSubDag> {
        self.inner.read().commit_by_digest.get(digest).cloned()
    }

    fn get_latest_commit(&self) -> Result<CommittedSubDag> {
        let inner = self.inner.read();

        // Find the commit with the highest index
        inner
            .commits
            .values()
            .max_by_key(|c| c.commit_ref.index)
            .cloned()
            .ok_or_else(|| Error::missing(anyhow!("no commits found")))
    }

    fn get_last_commit_index_of_epoch(
        &self,
        epoch: crate::committee::EpochId,
    ) -> Option<CommitIndex> {
        let inner = self.inner.read();

        // Find the highest commit index for the given epoch
        inner
            .commits
            .values()
            .filter(|c| c.epoch() == epoch)
            .map(|c| c.commit_ref.index)
            .max()
    }

    fn get_transaction(
        &self,
        tx_digest: &crate::digests::TransactionDigest,
    ) -> Result<Option<std::sync::Arc<crate::transaction::VerifiedTransaction>>> {
        todo!()
    }

    fn get_transaction_effects(
        &self,
        tx_digest: &crate::digests::TransactionDigest,
    ) -> Result<Option<crate::effects::TransactionEffects>> {
        todo!()
    }

    fn multi_get_transactions(
        &self,
        tx_digests: &[crate::digests::TransactionDigest],
    ) -> Result<Vec<Option<std::sync::Arc<crate::transaction::VerifiedTransaction>>>> {
        tx_digests
            .iter()
            .map(|digest| self.get_transaction(digest))
            .collect::<Result<Vec<_>, _>>()
    }

    fn multi_get_transaction_effects(
        &self,
        tx_digests: &[crate::digests::TransactionDigest],
    ) -> Result<Vec<Option<crate::effects::TransactionEffects>>> {
        tx_digests
            .iter()
            .map(|digest| self.get_transaction_effects(digest))
            .collect::<Result<Vec<_>, _>>()
    }
}

impl ConsensusStore for TestP2pStore {
    fn write(
        &self,
        write_batch: super::consensus::WriteBatch,
    ) -> crate::error::ConsensusResult<()> {
        let mut inner = self.inner.write();

        // Write blocks FIRST
        for block in write_batch.blocks {
            inner.blocks.insert(block.reference(), block);
        }

        // Then write commits
        for commit in write_batch.commits {
            // Store the TrustedCommit
            inner.trusted_commits.insert(commit.index(), commit.clone());

            // Look up blocks for this commit - they should already be in the store
            let commit_blocks: Vec<_> = commit
                .blocks()
                .iter()
                .filter_map(|block_ref| inner.blocks.get(block_ref).cloned())
                .collect();

            // Only create CommittedSubDag if we have all the blocks
            if commit_blocks.len() == commit.blocks().len() {
                let sub_dag = CommittedSubDag::new(
                    commit.leader(),
                    commit_blocks,
                    commit.timestamp_ms(),
                    commit.reference(),
                    commit.previous_digest(),
                );

                inner.commits.insert(commit.index(), sub_dag.clone());
                inner
                    .commit_by_digest
                    .insert(commit.digest(), sub_dag.clone());

                // CRITICAL: Only update highest_synced if this is actually newer
                // AND we're processing commits in order
                if let Some(current_highest) = &inner.highest_synced_commit {
                    if commit.index() == current_highest.commit_ref.index + 1 {
                        inner.highest_synced_commit = Some(sub_dag);
                    }
                } else if commit.index() == 0 {
                    // Genesis case
                    inner.highest_synced_commit = Some(sub_dag);
                }
            }
        }

        Ok(())
    }

    fn read_blocks(
        &self,
        refs: &[crate::consensus::block::BlockRef],
    ) -> crate::error::ConsensusResult<Vec<Option<crate::consensus::block::VerifiedBlock>>> {
        let inner = self.inner.read();

        Ok(refs
            .iter()
            .map(|block_ref| inner.blocks.get(block_ref).cloned())
            .collect())
    }

    fn contains_blocks(
        &self,
        refs: &[crate::consensus::block::BlockRef],
    ) -> crate::error::ConsensusResult<Vec<bool>> {
        let inner = self.inner.read();

        Ok(refs
            .iter()
            .map(|block_ref| inner.blocks.contains_key(block_ref))
            .collect())
    }

    fn scan_blocks_by_author(
        &self,
        authority: crate::committee::AuthorityIndex,
        start_round: crate::consensus::block::Round,
        epoch: Epoch,
    ) -> crate::error::ConsensusResult<Vec<crate::consensus::block::VerifiedBlock>> {
        let inner = self.inner.read();

        let blocks: Vec<_> = inner
            .blocks
            .iter()
            .filter_map(|(ref_key, block)| {
                if ref_key.author == authority
                    && ref_key.round >= start_round
                    && ref_key.epoch == epoch
                {
                    Some(block.clone())
                } else {
                    None
                }
            })
            .collect();

        Ok(blocks)
    }

    fn scan_last_blocks_by_author(
        &self,
        author: crate::committee::AuthorityIndex,
        num_of_rounds: u64,
        before_round: Option<crate::consensus::block::Round>,
        epoch: Epoch,
    ) -> crate::error::ConsensusResult<Vec<crate::consensus::block::VerifiedBlock>> {
        let inner = self.inner.read();
        let before_round = before_round.unwrap_or(u32::MAX);

        let mut blocks: Vec<_> = inner
            .blocks
            .iter()
            .filter_map(|(ref_key, block)| {
                if ref_key.author == author
                    && ref_key.round < before_round
                    && ref_key.epoch == epoch
                {
                    Some(block.clone())
                } else {
                    None
                }
            })
            .collect();

        // Sort by round descending and take the requested number
        blocks.sort_by(|a, b| b.round().cmp(&a.round()));
        blocks.truncate(num_of_rounds as usize);
        blocks.reverse(); // Return in ascending order

        Ok(blocks)
    }

    fn read_last_commit(
        &self,
    ) -> crate::error::ConsensusResult<Option<crate::consensus::commit::TrustedCommit>> {
        let inner = self.inner.read();

        Ok(inner
            .trusted_commits
            .values()
            .max_by_key(|c| c.index())
            .cloned())
    }

    fn scan_commits(
        &self,
        range: crate::consensus::commit::CommitRange,
    ) -> crate::error::ConsensusResult<Vec<crate::consensus::commit::TrustedCommit>> {
        let inner = self.inner.read();

        let commits: Vec<_> = inner
            .trusted_commits
            .iter()
            .filter_map(|(index, commit)| {
                if *index >= range.start() && *index <= range.end() {
                    Some(commit.clone())
                } else {
                    None
                }
            })
            .collect();

        Ok(commits)
    }

    fn read_commit_votes(
        &self,
        commit_index: crate::consensus::commit::CommitIndex,
    ) -> crate::error::ConsensusResult<Vec<crate::consensus::block::BlockRef>> {
        let inner = self.inner.read();

        // For testing, we can return empty or scan blocks for votes
        let mut votes = Vec::new();

        // Look through blocks for votes for this commit
        if let Some(commit) = inner.commits.get(&commit_index) {
            let commit_ref = commit.commit_ref;
            for block in inner.blocks.values() {
                for vote in block.commit_votes() {
                    if *vote == commit_ref {
                        votes.push(block.reference());
                    }
                }
            }
        }

        Ok(votes)
    }

    fn read_last_commit_info(
        &self,
    ) -> crate::error::ConsensusResult<
        Option<(
            crate::consensus::commit::CommitRef,
            crate::consensus::commit::CommitInfo,
        )>,
    > {
        let inner = self.inner.read();

        // Find the highest commit index with info
        inner
            .commit_info
            .iter()
            .max_by_key(|(index, _)| *index)
            .and_then(|(index, info)| {
                inner
                    .commits
                    .get(index)
                    .map(|commit| (commit.commit_ref, info.clone()))
            })
            .pipe(Ok)
    }

    fn prune_epochs_before(&self, epoch: Epoch) -> crate::error::ConsensusResult<()> {
        let mut inner = self.inner.write();

        // Remove blocks from epochs before the given epoch
        inner.blocks.retain(|block_ref, _| block_ref.epoch >= epoch);

        // Remove commits from epochs before the given epoch
        let indices_to_remove: Vec<_> = inner
            .commits
            .iter()
            .filter_map(|(index, commit)| {
                if commit.epoch() < epoch {
                    Some(*index)
                } else {
                    None
                }
            })
            .collect();

        for index in indices_to_remove {
            if let Some(commit) = inner.commits.remove(&index) {
                inner.commit_by_digest.remove(&commit.commit_ref.digest);
                inner.trusted_commits.remove(&index);
                inner.commit_info.remove(&index);

                // Update highest_pruned_commit_index
                inner.highest_pruned_commit_index = Some(
                    inner
                        .highest_pruned_commit_index
                        .map_or(index, |existing| existing.max(index)),
                );
            }
        }

        Ok(())
    }

    fn get_highest_pruned_commit_index(
        &self,
    ) -> crate::error::ConsensusResult<Option<CommitIndex>> {
        Ok(self.inner.read().highest_pruned_commit_index)
    }
}
