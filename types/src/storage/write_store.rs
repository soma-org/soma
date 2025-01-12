use crate::{
    accumulator::CommitIndex,
    committee::Committee,
};

use super::{read_store::ReadStore, storage_error::Result};

pub trait WriteStore: ReadStore {
    fn insert_committee(&self, new_committee: Committee) -> Result<()>;
    fn update_highest_synced_commit(&self, commit: CommitIndex) -> Result<()>;
}

impl<T: WriteStore + ?Sized> WriteStore for &T {
    fn update_highest_synced_commit(&self, commit: CommitIndex) -> Result<()> {
        (*self).update_highest_synced_commit(commit)
    }

    fn insert_committee(&self, new_committee: Committee) -> Result<()> {
        (*self).insert_committee(new_committee)
    }
}
