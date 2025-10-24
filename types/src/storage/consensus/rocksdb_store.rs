use bytes::Bytes;
use std::collections::{BTreeMap, VecDeque};
use std::ops::Bound::Included;
use store::{
    rocks::{default_db_options, DBMap, DBMapTableConfigMap},
    DBMapUtils, Map,
};
use tracing::{debug, info};

use crate::{
    committee::{AuthorityIndex, Epoch},
    consensus::{
        block::{BlockAPI, BlockDigest, BlockRef, Round, SignedBlock, VerifiedBlock},
        commit::{
            CommitAPI, CommitDigest, CommitIndex, CommitInfo, CommitRange, CommitRef, TrustedCommit,
        },
    },
    error::{ConsensusError, ConsensusResult},
    storage::consensus::{ConsensusStore, WriteBatch},
};

/// Persistent storage with RocksDB.
#[derive(DBMapUtils)]
pub struct RocksDBStore {
    /// Stores SignedBlock by refs.
    blocks: DBMap<(Epoch, Round, AuthorityIndex, BlockDigest), Bytes>,
    /// A secondary index that orders refs first by authors.
    #[rename = "digests"]
    digests_by_authorities: DBMap<(Epoch, AuthorityIndex, Round, BlockDigest), ()>,
    /// Maps commit index to Commit.
    commits: DBMap<(CommitIndex, CommitDigest), Bytes>,
    /// Collects votes on commits.
    /// TODO: batch multiple votes into a single row.
    commit_votes: DBMap<(CommitIndex, CommitDigest, BlockRef), ()>,
    /// Stores info related to Commit that helps recovery.
    commit_info: DBMap<(CommitIndex, CommitDigest), CommitInfo>,
}

impl RocksDBStore {
    const BLOCKS_CF: &'static str = "blocks";
    const DIGESTS_BY_AUTHORITIES_CF: &'static str = "digests";
    const COMMITS_CF: &'static str = "commits";
    const COMMIT_VOTES_CF: &'static str = "commit_votes";
    const COMMIT_INFO_CF: &'static str = "commit_info";
    const FINALIZED_COMMITS_CF: &'static str = "finalized_commits";

    /// Creates a new instance of RocksDB storage.
    pub fn new(path: &str) -> Self {
        // Consensus data has high write throughput (all transactions) and is rarely read
        // (only during recovery and when helping peers catch up).
        let db_options = default_db_options().optimize_db_for_write_throughput(2);
        let cf_options = default_db_options().optimize_for_write_throughput();
        let column_family_options = DBMapTableConfigMap::new(BTreeMap::from([
            (
                Self::BLOCKS_CF.to_string(),
                default_db_options()
                    .optimize_for_write_throughput_no_deletion()
                    // Using larger block is ok since there is not much point reads on the cf.
                    .set_block_options(512, 128 << 10),
            ),
            (
                Self::DIGESTS_BY_AUTHORITIES_CF.to_string(),
                cf_options.clone(),
            ),
            (Self::COMMITS_CF.to_string(), cf_options.clone()),
            (Self::COMMIT_VOTES_CF.to_string(), cf_options.clone()),
            (Self::COMMIT_INFO_CF.to_string(), cf_options.clone()),
            (Self::FINALIZED_COMMITS_CF.to_string(), cf_options.clone()),
        ]));
        Self::open_tables_read_write(
            path.into(),
            Some(db_options.options),
            Some(column_family_options),
        )
    }
}

impl ConsensusStore for RocksDBStore {
    fn write(&self, write_batch: WriteBatch) -> ConsensusResult<()> {
        let mut batch = self.blocks.batch();
        for block in write_batch.blocks {
            let block_ref = block.reference();
            batch
                .insert_batch(
                    &self.blocks,
                    [(
                        (
                            block_ref.epoch,
                            block_ref.round,
                            block_ref.author,
                            block_ref.digest,
                        ),
                        block.serialized(),
                    )],
                )
                .map_err(ConsensusError::RocksDBFailure)?;
            batch
                .insert_batch(
                    &self.digests_by_authorities,
                    [(
                        (
                            block_ref.epoch,
                            block_ref.author,
                            block_ref.round,
                            block_ref.digest,
                        ),
                        (),
                    )],
                )
                .map_err(ConsensusError::RocksDBFailure)?;
            for vote in block.commit_votes() {
                batch
                    .insert_batch(
                        &self.commit_votes,
                        [((vote.index, vote.digest, block_ref), ())],
                    )
                    .map_err(ConsensusError::RocksDBFailure)?;
            }
        }

        for commit in write_batch.commits {
            batch
                .insert_batch(
                    &self.commits,
                    [((commit.index(), commit.digest()), commit.serialized())],
                )
                .map_err(ConsensusError::RocksDBFailure)?;
        }

        for (commit_ref, commit_info) in write_batch.commit_info {
            batch
                .insert_batch(
                    &self.commit_info,
                    [((commit_ref.index, commit_ref.digest), commit_info)],
                )
                .map_err(ConsensusError::RocksDBFailure)?;
        }

        batch.write()?;
        Ok(())
    }

    fn read_blocks(&self, refs: &[BlockRef]) -> ConsensusResult<Vec<Option<VerifiedBlock>>> {
        let keys = refs
            .iter()
            .map(|r| (r.epoch, r.round, r.author, r.digest))
            .collect::<Vec<_>>();
        let serialized = self.blocks.multi_get(keys)?;
        let mut blocks = vec![];
        for (key, serialized) in refs.iter().zip(serialized) {
            if let Some(serialized) = serialized {
                let signed_block: SignedBlock =
                    bcs::from_bytes(&serialized).map_err(ConsensusError::MalformedBlock)?;
                // Only accepted blocks should have been written to storage.
                let block = VerifiedBlock::new_verified(signed_block, serialized);
                // Makes sure block data is not corrupted, by comparing digests.
                assert_eq!(*key, block.reference());
                blocks.push(Some(block));
            } else {
                blocks.push(None);
            }
        }
        Ok(blocks)
    }

    fn contains_blocks(&self, refs: &[BlockRef]) -> ConsensusResult<Vec<bool>> {
        let refs = refs
            .iter()
            .map(|r| (r.epoch, r.round, r.author, r.digest))
            .collect::<Vec<_>>();
        let exist = self.blocks.multi_contains_keys(refs)?;
        Ok(exist)
    }

    fn scan_blocks_by_author(
        &self,
        author: AuthorityIndex,
        start_round: Round,
        epoch: Epoch,
    ) -> ConsensusResult<Vec<VerifiedBlock>> {
        let mut refs = vec![];
        for kv in self.digests_by_authorities.safe_range_iter((
            Included((epoch, author, start_round, BlockDigest::MIN)),
            Included((epoch, author, Round::MAX, BlockDigest::MAX)),
        )) {
            let ((epoch, author, round, digest), _) = kv?;
            refs.push(BlockRef::new(round, author, digest, epoch));
        }
        let results = self.read_blocks(refs.as_slice())?;
        let mut blocks = Vec::with_capacity(refs.len());
        for (r, block) in refs.into_iter().zip(results.into_iter()) {
            blocks.push(
                block.unwrap_or_else(|| panic!("Storage inconsistency: block {:?} not found!", r)),
            );
        }
        Ok(blocks)
    }

    // The method returns the last `num_of_rounds` rounds blocks by author in round ascending order.
    // When a `before_round` is defined then the blocks of round `<=before_round` are returned. If not
    // then the max value for round will be used as cut off.
    fn scan_last_blocks_by_author(
        &self,
        author: AuthorityIndex,
        num_of_rounds: u64,
        before_round: Option<Round>,
        epoch: Epoch,
    ) -> ConsensusResult<Vec<VerifiedBlock>> {
        let before_round = before_round.unwrap_or(Round::MAX);
        let mut refs = VecDeque::new();
        for kv in self
            .digests_by_authorities
            .reversed_safe_iter_with_bounds(
                Some((epoch, author, Round::MIN, BlockDigest::MIN)),
                Some((epoch, author, before_round, BlockDigest::MAX)),
            )?
            .take(num_of_rounds as usize)
        {
            let ((epoch, author, round, digest), _) = kv?;
            refs.push_front(BlockRef::new(round, author, digest, epoch));
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
        let Some(result) = self
            .commits
            .reversed_safe_iter_with_bounds(None, None)?
            .next()
        else {
            return Ok(None);
        };
        let ((_index, digest), serialized) = result?;
        let commit = TrustedCommit::new_trusted(
            bcs::from_bytes(&serialized).map_err(ConsensusError::MalformedCommit)?,
            serialized,
        );
        assert_eq!(commit.digest(), digest);
        Ok(Some(commit))
    }

    fn scan_commits(&self, range: CommitRange) -> ConsensusResult<Vec<TrustedCommit>> {
        let mut commits = vec![];
        for result in self.commits.safe_range_iter((
            Included((range.start(), CommitDigest::MIN)),
            Included((range.end(), CommitDigest::MAX)),
        )) {
            let ((_index, digest), serialized) = result?;
            let commit = TrustedCommit::new_trusted(
                bcs::from_bytes(&serialized).map_err(ConsensusError::MalformedCommit)?,
                serialized,
            );
            assert_eq!(commit.digest(), digest);
            commits.push(commit);
        }
        Ok(commits)
    }

    fn read_commit_votes(&self, commit_index: CommitIndex) -> ConsensusResult<Vec<BlockRef>> {
        let mut votes = Vec::new();
        for vote in self.commit_votes.safe_range_iter((
            Included((commit_index, CommitDigest::MIN, BlockRef::MIN)),
            Included((commit_index, CommitDigest::MAX, BlockRef::MAX)),
        )) {
            let ((_, _, block_ref), _) = vote?;
            votes.push(block_ref);
        }
        Ok(votes)
    }

    fn read_last_commit_info(&self) -> ConsensusResult<Option<(CommitRef, CommitInfo)>> {
        let Some(result) = self
            .commit_info
            .reversed_safe_iter_with_bounds(None, None)?
            .next()
        else {
            return Ok(None);
        };
        let (key, commit_info) = result.map_err(ConsensusError::RocksDBFailure)?;
        Ok(Some((CommitRef::new(key.0, key.1), commit_info)))
    }

    fn prune_epochs_before(&self, epoch: Epoch) -> ConsensusResult<()> {
        info!("Starting consensus store pruning for epochs < {}", epoch);

        // Create a single batch for all deletions
        let mut batch = self.blocks.batch();
        let mut pruned_blocks = 0u64;
        let mut pruned_digests = 0u64;
        let mut pruned_commits = 0u64;
        let mut pruned_votes = 0u64;
        let mut pruned_commit_info = 0u64;

        // Prune blocks using range deletion
        // The end key is EXCLUSIVE, so (epoch, MIN, MIN, MIN) means "up to but not including epoch"
        let blocks_start_key = (
            Epoch::MIN,
            Round::MIN,
            AuthorityIndex::MIN,
            BlockDigest::MIN,
        );
        let blocks_end_key = (epoch, Round::MIN, AuthorityIndex::MIN, BlockDigest::MIN);

        // Optional: Count blocks to be deleted for logging
        for result in self
            .blocks
            .safe_iter_with_bounds(Some(blocks_start_key), Some(blocks_end_key))
        {
            if result.is_ok() {
                pruned_blocks += 1;
            }
        }

        if pruned_blocks > 0 {
            debug!("Scheduling deletion of {} blocks", pruned_blocks);
            batch
                .schedule_delete_range(&self.blocks, &blocks_start_key, &blocks_end_key)
                .map_err(ConsensusError::RocksDBFailure)?;
        }

        // Prune digests_by_authorities using range deletion
        let digest_start_key = (
            Epoch::MIN,
            AuthorityIndex::MIN,
            Round::MIN,
            BlockDigest::MIN,
        );
        let digest_end_key = (epoch, AuthorityIndex::MIN, Round::MIN, BlockDigest::MIN);

        // Optional: Count digests to be deleted
        for result in self
            .digests_by_authorities
            .safe_iter_with_bounds(Some(digest_start_key), Some(digest_end_key))
        {
            if result.is_ok() {
                pruned_digests += 1;
            }
        }

        if pruned_digests > 0 {
            debug!("Scheduling deletion of {} digests", pruned_digests);
            batch
                .schedule_delete_range(
                    &self.digests_by_authorities,
                    &digest_start_key,
                    &digest_end_key,
                )
                .map_err(ConsensusError::RocksDBFailure)?;
        }

        // For commits, deserialize and check epoch
        let mut commit_keys_to_delete = Vec::new();
        for result in self.commits.safe_iter_with_bounds(None, None) {
            let ((index, digest), serialized) = result?;
            let commit = TrustedCommit::new_trusted(
                bcs::from_bytes(&serialized).map_err(ConsensusError::MalformedCommit)?,
                serialized,
            );
            if commit.epoch() < epoch {
                commit_keys_to_delete.push((index, digest));
            }
        }
        // Delete in batches
        batch.delete_batch(&self.commits, commit_keys_to_delete.iter())?;

        // For commit_votes, check BlockRef.epoch
        let mut vote_keys_to_delete = Vec::new();
        for result in self.commit_votes.safe_iter_with_bounds(None, None) {
            let ((index, digest, block_ref), _) = result?;
            if block_ref.epoch < epoch {
                vote_keys_to_delete.push((index, digest, block_ref));
            }
        }
        batch.delete_batch(&self.commit_votes, vote_keys_to_delete.iter())?;

        // Delete commit_info entries for pruned commits
        if !commit_keys_to_delete.is_empty() {
            // The commit_info table has the same key structure as commits
            let mut commit_info_keys = Vec::new();
            for (index, digest) in &commit_keys_to_delete {
                if self.commit_info.contains_key(&(*index, *digest))? {
                    commit_info_keys.push((*index, *digest));
                    pruned_commit_info += 1;
                }
            }
            batch.delete_batch(&self.commit_info, commit_info_keys.iter())?;
        }

        // Count commits being deleted
        pruned_commits = commit_keys_to_delete.len() as u64;

        // Count votes being deleted
        pruned_votes = vote_keys_to_delete.len() as u64;

        // Write all deletions atomically
        batch.write()?;

        info!(
            "Completed pruning for epochs < {}: pruned {} blocks, {} digests, {} commits, {} votes, {} commit_info entries",
            epoch, pruned_blocks, pruned_digests, pruned_commits, pruned_votes, pruned_commit_info
        );

        // Trigger compaction to reclaim space
        // This is especially important after large range deletions
        if pruned_blocks > 0 {
            debug!("Triggering compaction for blocks column family");
            self.blocks
                .compact_range(&blocks_start_key, &blocks_end_key)?;
        }

        if pruned_digests > 0 {
            debug!("Triggering compaction for digests column family");
            self.digests_by_authorities
                .compact_range(&digest_start_key, &digest_end_key)?;
        }

        // For commits and commit_votes, we used point deletions, not range deletions
        // So we need to compact the entire table or the specific ranges we deleted
        if pruned_commits > 0 {
            debug!("Triggering compaction for commits column family");
            // Option 1: Compact the entire table (slower but thorough)
            self.commits.compact_range(
                &(CommitIndex::MIN, CommitDigest::MIN),
                &(CommitIndex::MAX, CommitDigest::MAX),
            )?;
        }

        if pruned_votes > 0 {
            debug!("Triggering compaction for commit_votes column family");
            // Compact the entire table since we did point deletions
            self.commit_votes.compact_range(
                &(CommitIndex::MIN, CommitDigest::MIN, BlockRef::MIN),
                &(CommitIndex::MAX, CommitDigest::MAX, BlockRef::MAX),
            )?;
        }
        Ok(())
    }
}
