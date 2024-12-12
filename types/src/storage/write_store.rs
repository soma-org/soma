use crate::{
    committee::Committee,
    state_sync::{VerifiedCommitContents, VerifiedCommitSummary},
};

use super::{read_store::ReadStore, storage_error::Result};

pub trait WriteStore: ReadStore {
    fn insert_committee(&self, new_committee: Committee) -> Result<()>;
    fn insert_commit(&self, commit: &VerifiedCommitSummary) -> Result<()>;
    fn update_highest_synced_commit(&self, commit: &VerifiedCommitSummary) -> Result<()>;
    fn update_highest_verified_commit(&self, commit: &VerifiedCommitSummary) -> Result<()>;
    fn insert_commit_contents(
        &self,
        commit: &VerifiedCommitSummary,
        contents: VerifiedCommitContents,
    ) -> Result<()>;
}

impl<T: WriteStore + ?Sized> WriteStore for &T {
    fn insert_commit(&self, checkpoint: &VerifiedCommitSummary) -> Result<()> {
        (*self).insert_commit(checkpoint)
    }

    fn update_highest_synced_commit(&self, checkpoint: &VerifiedCommitSummary) -> Result<()> {
        (*self).update_highest_synced_commit(checkpoint)
    }

    fn update_highest_verified_commit(&self, checkpoint: &VerifiedCommitSummary) -> Result<()> {
        (*self).update_highest_verified_commit(checkpoint)
    }

    fn insert_commit_contents(
        &self,
        commit: &VerifiedCommitSummary,
        contents: VerifiedCommitContents,
    ) -> Result<()> {
        (*self).insert_commit_contents(commit, contents)
    }

    fn insert_committee(&self, new_committee: Committee) -> Result<()> {
        (*self).insert_committee(new_committee)
    }
}
