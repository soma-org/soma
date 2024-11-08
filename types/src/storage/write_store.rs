use crate::committee::Committee;

use super::{read_store::ReadStore, storage_error::Result};

pub trait WriteStore: ReadStore {
    fn insert_committee(&self, new_committee: Committee) -> Result<()>;
}
