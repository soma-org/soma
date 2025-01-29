use crate::{accumulator::CommitIndex, committee::Committee, consensus::commit::CommittedSubDag};

use super::{read_store::ReadStore, storage_error::Result};

pub trait WriteStore: ReadStore {
    fn insert_committee(&self, new_committee: Committee) -> Result<()>;
    fn insert_commit(&self, commit: CommittedSubDag) -> Result<()>;
}

impl<T: WriteStore + ?Sized> WriteStore for &T {
    fn insert_commit(&self, commit: CommittedSubDag) -> Result<()> {
        (*self).insert_commit(commit)
    }

    fn insert_committee(&self, new_committee: Committee) -> Result<()> {
        (*self).insert_committee(new_committee)
    }
}
