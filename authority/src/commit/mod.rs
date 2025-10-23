use std::{collections::BTreeMap, path::Path, sync::Arc};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use store::{rocks::DBMap, DBMapUtils, Map as _, TypedStoreError};
use tracing::{debug, info, instrument};
use types::{
    accumulator::CommitIndex,
    committee::EpochId,
    consensus::{
        block::BlockAPI,
        commit::{CommitDigest, CommittedSubDag},
    },
    digests::TransactionEffectsDigest,
    error::SomaResult,
};

pub mod causal_order;
pub mod executor;

#[derive(Copy, Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum CommitWatermark {
    // HighestVerified,
    HighestSynced,
    HighestExecuted,
    HighestPruned,
}

#[derive(DBMapUtils)]
pub struct CommitStore {
    /// Maps commit  digest to commit index
    pub(crate) commit_index_by_digest: DBMap<CommitDigest, CommitIndex>,

    // TODO: delete full commits after state accumulation
    /// Stores entire commit contents from state sync for
    /// efficient reads of full commits. Entries from this table are deleted after state
    /// accumulation has completed.
    // full_commit_content: RwLock<BTreeMap<CommitIndex, FullCommitContents>>,

    /// Stores certified commits (CommittedSubDag)
    pub(crate) certified_commits: DBMap<CommitIndex, CommittedSubDag>,
    /// Map from commit digest to certified commit (CommittedSubDag)
    pub(crate) commit_by_digest: DBMap<CommitDigest, CommittedSubDag>,

    /// Maps commit digest to the effects digests generated from execution
    pub(crate) effects_digests_by_commit_digest: DBMap<CommitDigest, Vec<TransactionEffectsDigest>>,

    /// Watermarks used to determine the highest verified, fully synced, and
    /// fully executed commits
    pub(crate) watermarks: DBMap<CommitWatermark, (CommitIndex, CommitDigest)>,

    /// A map from epoch ID to the index of the last commit in that epoch.
    epoch_last_commit_map: DBMap<EpochId, CommitIndex>,

    /// Store locally computed commit summaries so that we can detect forks and log useful
    /// information. Can be pruned as soon as we verify that we are in agreement with the latest
    /// certified commit.
    pub(crate) locally_computed_commits: DBMap<CommitIndex, CommittedSubDag>,
}

impl CommitStore {
    pub fn new(path: &Path) -> Arc<Self> {
        Arc::new(Self::open_tables_read_write(path.to_path_buf(), None, None))
    }

    #[instrument(level = "info", skip_all)]
    pub fn insert_genesis_commit(&self, commit: CommittedSubDag) {
        // assert_eq!(
        //     commit.commit_ref.leader.b,
        //     0,
        //     "can't call insert_genesis_commit with a commit not in epoch 0"
        // );
        assert_eq!(
            commit.commit_ref.index, 0,
            "can't call insert_genesis_commit with a commit that doesn't have a index of 0"
        );

        // Only insert the genesis commit if the DB is empty and doesn't have it already
        if self
            .get_commit_by_digest(&commit.commit_ref.digest)
            .unwrap()
            .is_none()
        {
            self.insert_commit(commit).unwrap();
        }
    }

    pub fn get_commit_by_digest(
        &self,
        digest: &CommitDigest,
    ) -> Result<Option<CommittedSubDag>, TypedStoreError> {
        self.commit_by_digest
            .get(digest)
            .map(|maybe_commit| maybe_commit.map(|c| c.into()))
    }

    pub fn get_commit_by_index(
        &self,
        index: CommitIndex,
    ) -> Result<Option<CommittedSubDag>, TypedStoreError> {
        self.certified_commits
            .get(&index)
            .map(|maybe_commit| maybe_commit.to_owned().into())
    }

    pub fn get_highest_synced_commit(&self) -> Result<Option<CommittedSubDag>, TypedStoreError> {
        let highest_synced =
            if let Some(highest_synced) = self.watermarks.get(&CommitWatermark::HighestSynced)? {
                highest_synced.clone()
            } else {
                return Ok(None);
            };

        self.get_commit_by_digest(&highest_synced.1)
    }

    pub fn get_highest_executed_commit(&self) -> Result<Option<CommittedSubDag>, TypedStoreError> {
        let highest_executed = if let Some(highest_executed) =
            self.watermarks.get(&CommitWatermark::HighestExecuted)?
        {
            highest_executed.clone()
        } else {
            return Ok(None);
        };
        self.get_commit_by_digest(&highest_executed.1)
    }

    pub fn get_highest_executed_commit_index(
        &self,
    ) -> Result<Option<CommitIndex>, TypedStoreError> {
        if let Some(highest_executed) = self.watermarks.get(&CommitWatermark::HighestExecuted)? {
            Ok(Some(highest_executed.0))
        } else {
            Ok(None)
        }
    }

    pub fn get_highest_pruned_commit_index(&self) -> Result<Option<CommitIndex>, TypedStoreError> {
        self.watermarks
            .get(&CommitWatermark::HighestPruned)
            .map(|watermark| watermark.map(|w| w.0))
    }

    pub fn prune_local_summaries(&self) -> SomaResult {
        if let Some((last_local_summary, _)) = self
            .locally_computed_commits
            .reversed_safe_iter_with_bounds(None, None)?
            .next()
            .transpose()?
        {
            let mut batch = self.locally_computed_commits.batch();
            batch.schedule_delete_range(&self.locally_computed_commits, &0, &last_local_summary)?;
            batch.write()?;
            info!("Pruned local summaries up to {:?}", last_local_summary);
        }
        Ok(())
    }

    fn check_for_commit_fork(
        &self,
        local_commit: &CommittedSubDag,
        verified_commit: &CommittedSubDag,
    ) {
        // if local_commit != verified_commit.data() {
        //     let verified_contents = self
        //         .get_commit_contents(&verified_commit.content_digest)
        //         .map(|opt_contents| {
        //             opt_contents
        //                 .map(|contents| format!("{:?}", contents))
        //                 .unwrap_or_else(|| {
        //                     format!(
        //                         "Verified commit contents not found, digest: {:?}",
        //                         verified_commit.content_digest,
        //                     )
        //                 })
        //         })
        //         .map_err(|e| {
        //             format!(
        //                 "Failed to get verified commit contents, digest: {:?} error: {:?}",
        //                 verified_commit.content_digest, e
        //             )
        //         })
        //         .unwrap_or_else(|err_msg| err_msg);

        //     let local_contents = self
        //         .get_commit_contents(&local_commit.content_digest)
        //         .map(|opt_contents| {
        //             opt_contents
        //                 .map(|contents| format!("{:?}", contents))
        //                 .unwrap_or_else(|| {
        //                     format!(
        //                         "Local commit contents not found, digest: {:?}",
        //                         local_commit.content_digest
        //                     )
        //                 })
        //         })
        //         .map_err(|e| {
        //             format!(
        //                 "Failed to get local commit contents, digest: {:?} error: {:?}",
        //                 local_commit.content_digest, e
        //             )
        //         })
        //         .unwrap_or_else(|err_msg| err_msg);

        //     // commit contents may be too large for panic message.
        //     error!(
        //         verified_commit = ?verified_commit.data(),
        //         ?verified_contents,
        //         ?local_commit,
        //         ?local_contents,
        //         "Local commit fork detected!",
        //     );
        //     panic!(
        //         "Local commit fork detected for index: {}",
        //         local_commit.index()
        //     );
        // }
    }

    // Called by consensus (ConsensusAggregator).
    // Different from `insert_verified_commit`, it does not touch
    // the highest_verified_commit watermark such that state sync
    // will have a chance to process this commit and perform some
    // state-sync only things.
    pub fn insert_certified_commit(&self, commit: &CommittedSubDag) -> Result<(), TypedStoreError> {
        debug!(
            commit_index = commit.commit_ref.index,
            "Inserting certified commit",
        );

        let mut batch = self.certified_commits.batch();
        batch
            .insert_batch(
                &self.certified_commits,
                [(commit.commit_ref.index, commit.clone())],
            )?
            .insert_batch(
                &self.commit_by_digest,
                [(commit.commit_ref.digest, commit.clone())],
            )?;
        if commit.is_last_commit_of_epoch() {
            batch.insert_batch(
                &self.epoch_last_commit_map,
                [(commit.epoch(), commit.commit_ref.index)],
            )?;
        }
        batch.write()?;

        // TODO: check for commit forks
        // if let Some(local_commit) = self.locally_computed_commits.read().get(commit.index()) {
        //     self.check_for_commit_fork(&local_commit, commit);
        // }

        Ok(())
    }

    pub fn update_highest_synced_commit(
        &self,
        commit: &CommittedSubDag,
    ) -> Result<(), TypedStoreError> {
        debug!(
            commit_index = commit.commit_ref.index,
            "Updating highest synced commit",
        );
        self.watermarks.insert(
            &CommitWatermark::HighestSynced,
            &(commit.commit_ref.index, commit.commit_ref.digest),
        )?;

        info!(
            commit_index = commit.commit_ref.index,
            "Updated highest synced commit"
        );
        Ok(())
    }

    pub fn update_highest_executed_commit(
        &self,
        commit: &CommittedSubDag,
    ) -> Result<(), TypedStoreError> {
        if let Some(index) = self.get_highest_executed_commit_index()? {
            if index >= commit.commit_ref.index {
                return Ok(());
            }
            assert_eq!(
                index + 1,
                commit.commit_ref.index,
                "Cannot update highest executed commit to {} when current highest executed commit \
                 is {}",
                commit.commit_ref.index,
                index
            );
        }
        debug!(
            index = commit.commit_ref.index,
            "Updating highest executed commit",
        );
        self.watermarks.insert(
            &CommitWatermark::HighestExecuted,
            &(commit.commit_ref.index, commit.commit_ref.digest),
        )?;

        Ok(())
    }

    pub fn get_index_by_digest(
        &self,
        digest: &CommitDigest,
    ) -> Result<Option<CommitIndex>, TypedStoreError> {
        self.commit_index_by_digest.get(digest)
    }

    pub fn delete_commit(&self, index: CommitIndex) -> Result<(), TypedStoreError> {
        self.certified_commits.remove(&index)
    }

    pub fn delete_digest_index_mapping(
        &self,
        digest: &CommitDigest,
    ) -> Result<(), TypedStoreError> {
        self.commit_index_by_digest.remove(digest)
    }

    pub fn insert_epoch_last_commit(
        &self,
        epoch_id: EpochId,
        commit: &CommittedSubDag,
    ) -> SomaResult {
        self.epoch_last_commit_map
            .insert(&epoch_id, &commit.commit_ref.index)?;
        Ok(())
    }

    pub fn get_epoch_last_commit(&self, epoch_id: EpochId) -> SomaResult<Option<CommittedSubDag>> {
        let index = self.epoch_last_commit_map.get(&epoch_id)?;
        let commit = match index {
            Some(index) => self.get_commit_by_index(index)?,
            None => None,
        };
        Ok(commit)
    }

    pub fn get_last_commit_index_of_epoch(
        &self,
        epoch: EpochId,
    ) -> SomaResult<Option<CommitIndex>> {
        Ok(self.epoch_last_commit_map.get(&epoch)?)
    }

    // Called by state sync, apart from inserting the commit and updating
    // related tables, it also bumps the highest_synced_commit watermark.
    pub fn insert_commit(&self, commit: CommittedSubDag) -> Result<(), TypedStoreError> {
        self.insert_certified_commit(&commit)?;
        self.update_highest_synced_commit(&commit)
    }

    /// Get effects by commit digest
    pub fn get_effects_by_commit_digest(
        &self,
        digest: &CommitDigest,
    ) -> Result<Option<Vec<TransactionEffectsDigest>>, TypedStoreError> {
        self.effects_digests_by_commit_digest.get(digest)
    }

    /// Store effects after commit execution
    pub fn insert_effects_for_commit(
        &self,
        digest: &CommitDigest,
        effects_digests: Vec<TransactionEffectsDigest>,
    ) -> Result<(), TypedStoreError> {
        let mut batch = self.effects_digests_by_commit_digest.batch();
        batch.insert_batch(
            &self.effects_digests_by_commit_digest,
            [(digest.clone(), effects_digests.clone())],
        )?;

        batch.write()?;
        Ok(())
    }
}
