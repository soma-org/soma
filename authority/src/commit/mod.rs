use std::{collections::BTreeMap, path::Path, sync::Arc};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tracing::{debug, error, info, instrument};
use types::{
    accumulator::CommitIndex,
    committee::EpochId,
    digests::{CommitContentsDigest, CommitSummaryDigest},
    error::SomaResult,
    state_sync::{
        CommitContents, CommitSummary, FullCommitContents, TrustedCommitSummary,
        VerifiedCommitContents, VerifiedCommitSummary,
    },
};

use crate::{epoch_store::AuthorityPerEpochStore, store::TypedStoreError};

pub mod builder;
pub mod executor;
pub mod output;

#[derive(Copy, Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum CommitWatermark {
    HighestVerified,
    HighestSynced,
    HighestExecuted,
    // HighestPruned,
}

pub struct CommitStore {
    /// Maps commit contents digest to commit contents
    pub(crate) commit_content: RwLock<BTreeMap<CommitContentsDigest, CommitContents>>,

    /// Maps commit contents digest to commit index
    pub(crate) commit_index_by_contents_digest: RwLock<BTreeMap<CommitContentsDigest, CommitIndex>>,

    /// Stores entire commit contents from state sync for
    /// efficient reads of full commits. Entries from this table are deleted after state
    /// accumulation has completed.
    full_commit_content: RwLock<BTreeMap<CommitIndex, FullCommitContents>>,

    /// Stores certified commits
    pub(crate) certified_commits: RwLock<BTreeMap<CommitIndex, TrustedCommitSummary>>,
    /// Map from commit digest to certified commit
    pub(crate) commit_by_digest: RwLock<BTreeMap<CommitSummaryDigest, TrustedCommitSummary>>,

    /// Watermarks used to determine the highest verified, fully synced, and
    /// fully executed commits
    pub(crate) watermarks: RwLock<BTreeMap<CommitWatermark, (CommitIndex, CommitSummaryDigest)>>,

    /// A map from epoch ID to the index of the last commit in that epoch.
    epoch_last_commit_map: RwLock<BTreeMap<EpochId, CommitIndex>>,

    /// Store locally computed commit summaries so that we can detect forks and log useful
    /// information. Can be pruned as soon as we verify that we are in agreement with the latest
    /// certified commit.
    pub(crate) locally_computed_commits: RwLock<BTreeMap<CommitIndex, CommitSummary>>,
}

impl CommitStore {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            commit_content: RwLock::new(BTreeMap::new()),
            commit_index_by_contents_digest: RwLock::new(BTreeMap::new()),
            full_commit_content: RwLock::new(BTreeMap::new()),
            certified_commits: RwLock::new(BTreeMap::new()),
            commit_by_digest: RwLock::new(BTreeMap::new()),
            watermarks: RwLock::new(BTreeMap::new()),
            epoch_last_commit_map: RwLock::new(BTreeMap::new()),
            locally_computed_commits: RwLock::new(BTreeMap::new()),
        })
    }

    // #[instrument(level = "info", skip_all)]
    // pub fn insert_genesis_commit(
    //     &self,
    //     commit: VerifiedCommitSummary,
    //     contents: CommitContents,
    //     epoch_store: &AuthorityPerEpochStore,
    // ) {
    //     assert_eq!(
    //         commit.epoch(),
    //         0,
    //         "can't call insert_genesis_commit with a commit not in epoch 0"
    //     );
    //     assert_eq!(
    //         *commit.index(),
    //         0,
    //         "can't call insert_genesis_commit with a commit that doesn't have a index of 0"
    //     );

    //     // Only insert the genesis commit if the DB is empty and doesn't have it already
    //     if self
    //         .get_commit_by_digest(commit.digest())
    //         .unwrap()
    //         .is_none()
    //     {
    //         if epoch_store.epoch() == commit.epoch {
    //             epoch_store
    //                 .put_genesis_commit_in_builder(commit.data(), &contents)
    //                 .unwrap();
    //         } else {
    //             debug!(
    //                 validator_epoch =% epoch_store.epoch(),
    //                 genesis_epoch =% commit.epoch(),
    //                 "Not inserting commit builder data for genesis commit",
    //             );
    //         }
    //         self.insert_commit_contents(contents).unwrap();
    //         self.insert_verified_commit(&commit).unwrap();
    //         self.update_highest_synced_commit(&commit).unwrap();
    //     }
    // }

    pub fn get_commit_by_digest(
        &self,
        digest: &CommitSummaryDigest,
    ) -> Result<Option<VerifiedCommitSummary>, TypedStoreError> {
        Ok(self
            .commit_by_digest
            .read()
            .get(digest)
            .map(|maybe_commit| maybe_commit.to_owned().into()))
    }

    pub fn get_commit_by_index(
        &self,
        index: CommitIndex,
    ) -> Result<Option<VerifiedCommitSummary>, TypedStoreError> {
        Ok(self
            .certified_commits
            .read()
            .get(&index)
            .map(|maybe_commit| maybe_commit.to_owned().into()))
    }

    pub fn get_highest_verified_commit(
        &self,
    ) -> Result<Option<VerifiedCommitSummary>, TypedStoreError> {
        let highest_verified = if let Some(highest_verified) = self
            .watermarks
            .read()
            .get(&CommitWatermark::HighestVerified)
        {
            highest_verified.clone()
        } else {
            return Ok(None);
        };
        self.get_commit_by_digest(&highest_verified.1)
    }

    pub fn get_highest_synced_commit(
        &self,
    ) -> Result<Option<VerifiedCommitSummary>, TypedStoreError> {
        let highest_synced = if let Some(highest_synced) =
            self.watermarks.read().get(&CommitWatermark::HighestSynced)
        {
            highest_synced.clone()
        } else {
            return Ok(None);
        };
        self.get_commit_by_digest(&highest_synced.1)
    }

    pub fn get_highest_executed_commit(
        &self,
    ) -> Result<Option<VerifiedCommitSummary>, TypedStoreError> {
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

    pub fn get_commit_contents(
        &self,
        digest: &CommitContentsDigest,
    ) -> Result<Option<CommitContents>, TypedStoreError> {
        Ok(self.commit_content.read().get(digest).cloned())
    }

    pub fn get_full_commit_contents_by_index(
        &self,
        index: CommitIndex,
    ) -> Result<Option<FullCommitContents>, TypedStoreError> {
        Ok(self.full_commit_content.read().get(&index).cloned())
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
        local_commit: &CommitSummary,
        verified_commit: &VerifiedCommitSummary,
    ) {
        if local_commit != verified_commit.data() {
            let verified_contents = self
                .get_commit_contents(&verified_commit.content_digest)
                .map(|opt_contents| {
                    opt_contents
                        .map(|contents| format!("{:?}", contents))
                        .unwrap_or_else(|| {
                            format!(
                                "Verified commit contents not found, digest: {:?}",
                                verified_commit.content_digest,
                            )
                        })
                })
                .map_err(|e| {
                    format!(
                        "Failed to get verified commit contents, digest: {:?} error: {:?}",
                        verified_commit.content_digest, e
                    )
                })
                .unwrap_or_else(|err_msg| err_msg);

            let local_contents = self
                .get_commit_contents(&local_commit.content_digest)
                .map(|opt_contents| {
                    opt_contents
                        .map(|contents| format!("{:?}", contents))
                        .unwrap_or_else(|| {
                            format!(
                                "Local commit contents not found, digest: {:?}",
                                local_commit.content_digest
                            )
                        })
                })
                .map_err(|e| {
                    format!(
                        "Failed to get local commit contents, digest: {:?} error: {:?}",
                        local_commit.content_digest, e
                    )
                })
                .unwrap_or_else(|err_msg| err_msg);

            // commit contents may be too large for panic message.
            error!(
                verified_commit = ?verified_commit.data(),
                ?verified_contents,
                ?local_commit,
                ?local_contents,
                "Local commit fork detected!",
            );
            panic!(
                "Local commit fork detected for index: {}",
                local_commit.index()
            );
        }
    }

    // Called by consensus (ConsensusAggregator).
    // Different from `insert_verified_commit`, it does not touch
    // the highest_verified_commit watermark such that state sync
    // will have a chance to process this commit and perform some
    // state-sync only things.
    pub fn insert_certified_commit(
        &self,
        commit: &VerifiedCommitSummary,
    ) -> Result<(), TypedStoreError> {
        debug!(commit_index = commit.index(), "Inserting certified commit",);
        self.certified_commits
            .write()
            .insert(*commit.index(), commit.serializable_ref().clone());

        self.commit_by_digest
            .write()
            .insert(*commit.digest(), commit.serializable_ref().clone());

        // if commit.next_epoch_committee().is_some() {
        //     batch.insert_batch(
        //         &self.epoch_last_commit_map,
        //         [(&commit.epoch(), commit.index())],
        //     )?;
        // }

        if let Some(local_commit) = self.locally_computed_commits.read().get(commit.index()) {
            self.check_for_commit_fork(&local_commit, commit);
        }

        Ok(())
    }

    // Called by state sync, apart from inserting the commit and updating
    // related tables, it also bumps the highest_verified_commit watermark.
    #[instrument(level = "debug", skip_all)]
    pub fn insert_verified_commit(
        &self,
        commit: &VerifiedCommitSummary,
    ) -> Result<(), TypedStoreError> {
        self.insert_certified_commit(commit)?;
        self.update_highest_verified_commit(commit)
    }

    pub fn update_highest_verified_commit(
        &self,
        commit: &VerifiedCommitSummary,
    ) -> Result<(), TypedStoreError> {
        if Some(*commit.index()) > self.get_highest_verified_commit()?.map(|x| *x.index()) {
            debug!(
                commit_index = commit.index(),
                "Updating highest verified commit",
            );
            self.watermarks.write().insert(
                CommitWatermark::HighestVerified,
                (*commit.index(), *commit.digest()),
            );
        }

        Ok(())
    }

    pub fn update_highest_synced_commit(
        &self,
        commit: &VerifiedCommitSummary,
    ) -> Result<(), TypedStoreError> {
        debug!(
            commit_index = commit.index(),
            "Updating highest synced commit",
        );
        self.watermarks.write().insert(
            CommitWatermark::HighestSynced,
            (*commit.index(), *commit.digest()),
        );
        Ok(())
    }

    pub fn update_highest_executed_commit(
        &self,
        commit: &VerifiedCommitSummary,
    ) -> Result<(), TypedStoreError> {
        if let Some(index) = self.get_highest_executed_commit_index()? {
            if index >= *commit.index() {
                return Ok(());
            }
            assert_eq!(index + 1, *commit.index(),
            "Cannot update highest executed commit to {} when current highest executed commit is {}",
            commit.index(),
            index);
        }
        debug!(index = commit.index(), "Updating highest executed commit",);
        self.watermarks.write().insert(
            CommitWatermark::HighestExecuted,
            (*commit.index(), *commit.digest()),
        );

        Ok(())
    }

    pub fn insert_verified_commit_contents(
        &self,
        commit: &VerifiedCommitSummary,
        full_contents: VerifiedCommitContents,
    ) -> Result<(), TypedStoreError> {
        self.commit_index_by_contents_digest
            .write()
            .insert(commit.content_digest, *commit.index());

        let full_contents = full_contents.into_inner();
        self.full_commit_content
            .write()
            .insert(*commit.index(), full_contents.clone());

        let contents = full_contents.into_commit_contents();
        assert_eq!(&commit.content_digest, contents.digest());
        self.commit_content
            .write()
            .insert(*contents.digest(), contents);

        Ok(())
    }

    pub fn delete_full_commit_contents(&self, index: CommitIndex) -> Result<(), TypedStoreError> {
        self.full_commit_content.write().remove(&index);
        Ok(())
    }

    pub fn get_index_by_contents_digest(
        &self,
        digest: &CommitContentsDigest,
    ) -> Result<Option<CommitIndex>, TypedStoreError> {
        Ok(self
            .commit_index_by_contents_digest
            .read()
            .get(digest)
            .cloned())
    }

    pub fn delete_contents_digest_index_mapping(
        &self,
        digest: &CommitContentsDigest,
    ) -> Result<(), TypedStoreError> {
        self.commit_index_by_contents_digest.write().remove(digest);
        Ok(())
    }

    pub fn get_epoch_last_checkpoint(
        &self,
        epoch_id: EpochId,
    ) -> SomaResult<Option<VerifiedCommitSummary>> {
        let lock = self.epoch_last_commit_map.read();
        let index = lock.get(&epoch_id);
        let commit = match index {
            Some(index) => self.get_commit_by_index(*index)?,
            None => None,
        };
        Ok(commit)
    }

    pub fn insert_epoch_last_commit(
        &self,
        epoch_id: EpochId,
        commit: &VerifiedCommitSummary,
    ) -> SomaResult {
        self.epoch_last_commit_map
            .write()
            .insert(epoch_id, *commit.index());
        Ok(())
    }

    pub fn get_epoch_last_commit(
        &self,
        epoch_id: EpochId,
    ) -> SomaResult<Option<VerifiedCommitSummary>> {
        let guard = self.epoch_last_commit_map.read();
        let index = guard.get(&epoch_id);
        let commit = match index {
            Some(index) => self.get_commit_by_index(*index)?,
            None => None,
        };
        Ok(commit)
    }
}
