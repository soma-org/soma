use parking_lot::RwLock;
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::ops::Bound::Included;
use std::sync::Arc;
use tracing::info;

use crate::consensus::{
    block::{BlockAPI as _, BlockDigest, BlockRef, Round, Slot, VerifiedBlock},
    commit::{
        CommitAPI as _, CommitDigest, CommitIndex, CommitInfo, CommitRange, CommitRef,
        TrustedCommit,
    },
};

use crate::committee::{AuthorityIndex, Committee, Epoch, EpochId};
use crate::error::{ConsensusResult, SomaResult};

use super::{ConsensusStore, WriteBatch};

/// In-memory storage for testing.
pub struct MemStore {
    inner: RwLock<Inner>,
}

struct Inner {
    // Block key needs epoch since blocks from different epochs might have same round/author/digest
    blocks: BTreeMap<EpochId, BTreeMap<(Round, AuthorityIndex, BlockDigest), VerifiedBlock>>,
    digests_by_authorities: BTreeMap<EpochId, BTreeSet<(AuthorityIndex, Round, BlockDigest)>>,

    epochs_by_commits: BTreeMap<CommitIndex, EpochId>,

    // Commit-related storage with epoch separation
    commits: BTreeMap<EpochId, BTreeMap<(CommitIndex, CommitDigest), TrustedCommit>>,
    commit_votes: BTreeMap<EpochId, BTreeSet<(CommitIndex, CommitDigest, BlockRef)>>,
    commit_info: BTreeMap<EpochId, BTreeMap<(CommitIndex, CommitDigest), CommitInfo>>,

    // Track the highest commit index that has been pruned
    highest_pruned_commit_index: Option<CommitIndex>,
}

impl MemStore {
    // #[cfg(test)]
    pub fn new() -> Self {
        use std::collections::BTreeSet;

        MemStore {
            inner: RwLock::new(Inner {
                blocks: BTreeMap::new(),
                digests_by_authorities: BTreeMap::new(),
                epochs_by_commits: BTreeMap::new(),
                commits: BTreeMap::new(),
                commit_votes: BTreeMap::new(),
                commit_info: BTreeMap::new(),
                highest_pruned_commit_index: None,
            }),
        }
    }
}

impl ConsensusStore for MemStore {
    fn write(&self, write_batch: WriteBatch) -> ConsensusResult<()> {
        let mut inner = self.inner.write();

        for block in write_batch.blocks {
            let block_ref = block.reference();
            inner.blocks.entry(block_ref.epoch).or_default().insert(
                (block_ref.round, block_ref.author, block_ref.digest),
                block.clone(),
            );

            inner
                .digests_by_authorities
                .entry(block_ref.epoch)
                .or_default()
                .insert((block_ref.author, block_ref.round, block_ref.digest));

            for vote in block.commit_votes() {
                inner
                    .commit_votes
                    .entry(block_ref.epoch)
                    .or_default()
                    .insert((vote.index, vote.digest, block_ref));
            }
        }

        for commit in write_batch.commits {
            let epoch = commit.epoch();
            let index = commit.index();
            inner.epochs_by_commits.insert(index, epoch);
            inner
                .commits
                .entry(epoch)
                .or_default()
                .insert((index, commit.digest()), commit);
        }

        // TODO: write commit info (and add epoch to commit info)
        // for (commit_ref, commit_info) in write_batch.commit_info {
        //     inner
        //         .commit_info
        //         .entry(commit_ref.epoch())
        //         .or_default()
        //         .insert((commit_ref.index, commit_ref.digest), commit_info);
        // }

        Ok(())
    }

    fn read_blocks(&self, refs: &[BlockRef]) -> ConsensusResult<Vec<Option<VerifiedBlock>>> {
        let inner = self.inner.read();
        let blocks = refs
            .iter()
            .map(|r| {
                inner
                    .blocks
                    .get(&r.epoch)
                    .and_then(|epoch_blocks| epoch_blocks.get(&(r.round, r.author, r.digest)))
                    .cloned()
            })
            .collect();
        Ok(blocks)
    }

    fn contains_blocks(&self, refs: &[BlockRef]) -> ConsensusResult<Vec<bool>> {
        let inner = self.inner.read();
        let exist = refs
            .iter()
            .map(|r| {
                inner
                    .blocks
                    .get(&r.epoch)
                    .map(|epoch_blocks| epoch_blocks.contains_key(&(r.round, r.author, r.digest)))
                    .unwrap_or(false)
            })
            .collect();
        Ok(exist)
    }

    fn scan_blocks_by_author(
        &self,
        author: AuthorityIndex,
        start_round: Round,
        epoch: Epoch,
    ) -> ConsensusResult<Vec<VerifiedBlock>> {
        let inner = self.inner.read();
        let refs = inner
            .digests_by_authorities
            .get(&epoch)
            .map(|epoch_digests| {
                let mut refs = vec![];
                for &(author, round, digest) in epoch_digests.range((
                    Included((author, start_round, BlockDigest::MIN)),
                    Included((author, Round::MAX, BlockDigest::MAX)),
                )) {
                    refs.push(BlockRef::new(round, author, digest, epoch));
                }
                refs
            })
            .unwrap_or_default();

        let results = self.read_blocks(&refs)?;
        let mut blocks = vec![];
        for (r, block) in refs.into_iter().zip(results.into_iter()) {
            if let Some(block) = block {
                blocks.push(block);
            } else {
                panic!("Block {:?} not found!", r);
            }
        }
        Ok(blocks)
    }

    fn scan_last_blocks_by_author(
        &self,
        author: AuthorityIndex,
        num_of_rounds: u64,
        before_round: Option<Round>,
        epoch: EpochId,
    ) -> ConsensusResult<Vec<VerifiedBlock>> {
        let before_round = before_round.unwrap_or(Round::MAX);
        let inner = self.inner.read();

        let mut refs = VecDeque::new();
        if let Some(epoch_digests) = inner.digests_by_authorities.get(&epoch) {
            for &(author, round, digest) in epoch_digests
                .range((
                    Included((author, Round::MIN, BlockDigest::MIN)),
                    Included((author, before_round, BlockDigest::MAX)),
                ))
                .rev()
                .take(num_of_rounds as usize)
            {
                refs.push_front(BlockRef::new(round, author, digest, epoch));
            }
        }

        let results = self.read_blocks(refs.as_slices().0)?;
        let mut blocks = vec![];
        for (r, block) in refs.into_iter().zip(results.into_iter()) {
            blocks.push(
                block.unwrap_or_else(|| panic!("Storage inconsistency: block {:?} not found!", r)),
            );
        }
        Ok(blocks)
    }

    fn read_last_commit(&self) -> ConsensusResult<Option<TrustedCommit>> {
        let inner = self.inner.read();
        // Go through all epochs in reverse order to find the last commit
        for (_, epoch_commits) in inner.commits.iter().rev() {
            if let Some((_, commit)) = epoch_commits.last_key_value() {
                return Ok(Some(commit.clone()));
            }
        }
        Ok(None)
    }

    fn scan_commits(&self, range: CommitRange) -> ConsensusResult<Vec<TrustedCommit>> {
        let inner = self.inner.read();
        let mut commits = vec![];

        // Get the epoch range we need to look through
        let start_epoch = inner.epochs_by_commits.get(&range.start()).copied();
        let end_epoch = inner.epochs_by_commits.get(&range.end()).copied();

        match (start_epoch, end_epoch) {
            (Some(start_epoch), Some(end_epoch)) => {
                // If commits are in the same epoch
                if start_epoch == end_epoch {
                    if let Some(epoch_commits) = inner.commits.get(&start_epoch) {
                        commits.extend(
                            epoch_commits
                                .range((
                                    Included((range.start(), CommitDigest::MIN)),
                                    Included((range.end(), CommitDigest::MAX)),
                                ))
                                .map(|(_, commit)| commit.clone()),
                        );
                    }
                } else {
                    // Need to scan across multiple epochs
                    for (epoch, epoch_commits) in inner.commits.range(start_epoch..=end_epoch) {
                        let start = if epoch == &start_epoch {
                            range.start()
                        } else {
                            0
                        };
                        let end = if epoch == &end_epoch {
                            range.end()
                        } else {
                            CommitIndex::MAX
                        };

                        commits.extend(
                            epoch_commits
                                .range((
                                    Included((start, CommitDigest::MIN)),
                                    Included((end, CommitDigest::MAX)),
                                ))
                                .map(|(_, commit)| commit.clone()),
                        );
                    }
                }
            }
            _ => (), // If we can't find the epochs, return empty vec
        }

        Ok(commits)
    }

    fn read_commit_votes(&self, commit_index: CommitIndex) -> ConsensusResult<Vec<BlockRef>> {
        let inner = self.inner.read();

        // Get the epoch for this commit
        let epoch = inner.epochs_by_commits.get(&commit_index).copied();

        match epoch {
            Some(epoch) => {
                let votes = inner
                    .commit_votes
                    .get(&epoch)
                    .map(|epoch_votes| {
                        epoch_votes
                            .range((
                                Included((commit_index, CommitDigest::MIN, BlockRef::MIN)),
                                Included((commit_index, CommitDigest::MAX, BlockRef::MAX)),
                            ))
                            .map(|(_, _, block_ref)| *block_ref)
                            .collect()
                    })
                    .unwrap_or_default();
                Ok(votes)
            }
            None => Ok(vec![]),
        }
    }

    fn read_last_commit_info(&self) -> ConsensusResult<Option<(CommitRef, CommitInfo)>> {
        let inner = self.inner.read();
        // Go through all epochs in reverse order to find the last commit info
        for (_, epoch_commit_info) in inner.commit_info.iter().rev() {
            if let Some(((index, digest), info)) = epoch_commit_info.last_key_value() {
                return Ok(Some((CommitRef::new(*index, *digest), info.clone())));
            }
        }
        Ok(None)
    }

    fn prune_epochs_before(&self, epoch: Epoch) -> ConsensusResult<()> {
        info!("Starting MemStore pruning for epochs < {}", epoch);

        let mut inner = self.inner.write();

        // Track statistics for logging
        let mut pruned_blocks = 0usize;
        let mut pruned_digests = 0usize;
        let mut pruned_commits = 0usize;
        let mut pruned_votes = 0usize;
        let mut pruned_commit_info = 0usize;
        let mut pruned_epoch_mappings = 0usize;

        let mut max_pruned_commit_index: Option<CommitIndex> = None;

        // Since MemStore is already organized by epoch, we can use split_off for efficiency
        // split_off(epoch) keeps everything >= epoch and removes everything < epoch

        // Count and prune blocks
        for (_, epoch_blocks) in inner.blocks.range(..epoch) {
            pruned_blocks += epoch_blocks.len();
        }
        inner.blocks = inner.blocks.split_off(&epoch);

        // Count and prune digests_by_authorities
        for (_, epoch_digests) in inner.digests_by_authorities.range(..epoch) {
            pruned_digests += epoch_digests.len();
        }
        inner.digests_by_authorities = inner.digests_by_authorities.split_off(&epoch);

        // For commits, also need to clean up epochs_by_commits mapping
        let mut commit_indices_to_remove = Vec::new();
        for (epoch_id, epoch_commits) in inner.commits.range(..epoch) {
            for ((commit_index, _), _) in epoch_commits {
                commit_indices_to_remove.push(*commit_index);
                // Track the maximum commit index being pruned
                max_pruned_commit_index = Some(
                    max_pruned_commit_index.map_or(*commit_index, |max| max.max(*commit_index)),
                );
            }
            pruned_commits += epoch_commits.len();
        }
        inner.commits = inner.commits.split_off(&epoch);

        // Clean up the epochs_by_commits reverse mapping
        for commit_index in commit_indices_to_remove {
            if inner.epochs_by_commits.remove(&commit_index).is_some() {
                pruned_epoch_mappings += 1;
            }
        }

        // Count and prune commit_votes
        for (_, epoch_votes) in inner.commit_votes.range(..epoch) {
            pruned_votes += epoch_votes.len();
        }
        inner.commit_votes = inner.commit_votes.split_off(&epoch);

        // Count and prune commit_info
        for (_, epoch_info) in inner.commit_info.range(..epoch) {
            pruned_commit_info += epoch_info.len();
        }
        inner.commit_info = inner.commit_info.split_off(&epoch);

        // Update the highest pruned commit index
        if let Some(new_max) = max_pruned_commit_index {
            inner.highest_pruned_commit_index = Some(
                inner
                    .highest_pruned_commit_index
                    .map_or(new_max, |existing| existing.max(new_max)),
            );
        }

        info!(
            "Completed MemStore pruning for epochs < {}: pruned {} blocks, {} digests, {} commits, {} votes, {} commit_info entries, {} epoch mappings",
            epoch, pruned_blocks, pruned_digests, pruned_commits, pruned_votes, pruned_commit_info, pruned_epoch_mappings
        );

        Ok(())
    }

    fn get_highest_pruned_commit_index(&self) -> ConsensusResult<Option<CommitIndex>> {
        let inner = self.inner.read();
        Ok(inner.highest_pruned_commit_index)
    }
}
