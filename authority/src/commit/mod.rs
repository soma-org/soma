use std::{collections::BTreeMap, sync::Arc};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use store::TypedStoreError;
use tracing::{debug, info, instrument};
use types::{
    accumulator::CommitIndex,
    committee::EpochId,
    consensus::{
        block::BlockAPI,
        commit::{CommitDigest, CommittedSubDag},
    },
    error::SomaResult,
};

pub mod causal_order;
pub mod executor;

#[derive(Copy, Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum CommitWatermark {
    // HighestVerified,
    HighestSynced,
    HighestExecuted,
    // HighestPruned,
}

pub struct CommitStore {
    /// Maps commit  digest to commit index
    pub(crate) commit_index_by_digest: RwLock<BTreeMap<CommitDigest, CommitIndex>>,

    // TODO: delete full commits after state accumulation
    /// Stores entire commit contents from state sync for
    /// efficient reads of full commits. Entries from this table are deleted after state
    /// accumulation has completed.
    // full_commit_content: RwLock<BTreeMap<CommitIndex, FullCommitContents>>,

    /// Stores certified commits (CommittedSubDag)
    pub(crate) certified_commits: RwLock<BTreeMap<CommitIndex, CommittedSubDag>>,
    /// Map from commit digest to certified commit (CommittedSubDag)
    pub(crate) commit_by_digest: RwLock<BTreeMap<CommitDigest, CommittedSubDag>>,

    /// Watermarks used to determine the highest verified, fully synced, and
    /// fully executed commits
    pub(crate) watermarks: RwLock<BTreeMap<CommitWatermark, (CommitIndex, CommitDigest)>>,

    /// A map from epoch ID to the index of the last commit in that epoch.
    epoch_last_commit_map: RwLock<BTreeMap<EpochId, CommitIndex>>,

    /// Store locally computed commit summaries so that we can detect forks and log useful
    /// information. Can be pruned as soon as we verify that we are in agreement with the latest
    /// certified commit.
    pub(crate) locally_computed_commits: RwLock<BTreeMap<CommitIndex, CommittedSubDag>>,
}

impl CommitStore {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            commit_index_by_digest: RwLock::new(BTreeMap::new()),
            certified_commits: RwLock::new(BTreeMap::new()),
            commit_by_digest: RwLock::new(BTreeMap::new()),
            watermarks: RwLock::new(BTreeMap::new()),
            epoch_last_commit_map: RwLock::new(BTreeMap::new()),
            locally_computed_commits: RwLock::new(BTreeMap::new()),
        })
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
        Ok(self
            .commit_by_digest
            .read()
            .get(digest)
            .map(|maybe_commit| maybe_commit.to_owned().into()))
    }

    pub fn get_commit_by_index(
        &self,
        index: CommitIndex,
    ) -> Result<Option<CommittedSubDag>, TypedStoreError> {
        Ok(self
            .certified_commits
            .read()
            .get(&index)
            .map(|maybe_commit| maybe_commit.to_owned().into()))
    }

    pub fn get_highest_synced_commit(&self) -> Result<Option<CommittedSubDag>, TypedStoreError> {
        let highest_synced = if let Some(highest_synced) =
            self.watermarks.read().get(&CommitWatermark::HighestSynced)
        {
            highest_synced.clone()
        } else {
            return Ok(None);
        };

        self.get_commit_by_digest(&highest_synced.1)
    }

    pub fn get_highest_executed_commit(&self) -> Result<Option<CommittedSubDag>, TypedStoreError> {
        let highest_executed = if let Some(highest_executed) = self
            .watermarks
            .read()
            .get(&CommitWatermark::HighestExecuted)
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
        if let Some(highest_executed) = self
            .watermarks
            .read()
            .get(&CommitWatermark::HighestExecuted)
        {
            Ok(Some(highest_executed.0))
        } else {
            Ok(None)
        }
    }

    pub fn prune_local_summaries(&self) -> SomaResult {
        // Get write lock on the BTreeMap
        let mut commits = self.locally_computed_commits.write();

        if let Some((&last_local_summary, _)) = commits.iter().next_back() {
            // Remove all entries up to and including last_local_summary
            commits.retain(|&k, _| k > last_local_summary);

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
        self.certified_commits
            .write()
            .insert(commit.commit_ref.index, commit.clone());

        self.commit_by_digest
            .write()
            .insert(commit.commit_ref.digest, commit.clone());

        if commit.is_last_commit_of_epoch() {
            self.epoch_last_commit_map
                .write()
                .insert(commit.epoch(), commit.commit_ref.index);
        }

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
        self.watermarks.write().insert(
            CommitWatermark::HighestSynced,
            (commit.commit_ref.index, commit.commit_ref.digest),
        );

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
        self.watermarks.write().insert(
            CommitWatermark::HighestExecuted,
            (commit.commit_ref.index, commit.commit_ref.digest),
        );

        Ok(())
    }

    pub fn get_index_by_digest(
        &self,
        digest: &CommitDigest,
    ) -> Result<Option<CommitIndex>, TypedStoreError> {
        Ok(self.commit_index_by_digest.read().get(digest).cloned())
    }

    pub fn delete_commit(&self, index: CommitIndex) -> Result<(), TypedStoreError> {
        self.certified_commits.write().remove(&index);
        Ok(())
    }

    pub fn delete_digest_index_mapping(
        &self,
        digest: &CommitDigest,
    ) -> Result<(), TypedStoreError> {
        self.commit_index_by_digest.write().remove(digest);
        Ok(())
    }

    pub fn insert_epoch_last_commit(
        &self,
        epoch_id: EpochId,
        commit: &CommittedSubDag,
    ) -> SomaResult {
        self.epoch_last_commit_map
            .write()
            .insert(epoch_id, commit.commit_ref.index);
        Ok(())
    }

    pub fn get_epoch_last_commit(&self, epoch_id: EpochId) -> SomaResult<Option<CommittedSubDag>> {
        let guard = self.epoch_last_commit_map.read();
        let index = guard.get(&epoch_id);
        let commit = match index {
            Some(index) => self.get_commit_by_index(*index)?,
            None => None,
        };
        Ok(commit)
    }

    pub fn get_last_commit_index_of_epoch(&self, epoch: EpochId) -> Option<CommitIndex> {
        self.epoch_last_commit_map.read().get(&epoch).cloned()
    }

    // Called by state sync, apart from inserting the commit and updating
    // related tables, it also bumps the highest_synced_commit watermark.
    pub fn insert_commit(&self, commit: CommittedSubDag) -> Result<(), TypedStoreError> {
        self.insert_certified_commit(&commit)?;
        self.update_highest_synced_commit(&commit)
    }
}
