use crate::commit::CommitStore;
use crate::{cache::ExecutionCacheTraitPointers, committee_store::CommitteeStore};
use parking_lot::Mutex;
use std::sync::Arc;
use types::state_sync::VerifiedCommitContents;
use types::storage::storage_error::Error as StorageError;
use types::storage::storage_error::Result;
use types::{
    accumulator::CommitIndex,
    committee::{Committee, EpochId},
    digests::{CommitContentsDigest, CommitSummaryDigest, TransactionDigest},
    effects::TransactionEffects,
    object::{Object, ObjectID, Version},
    state_sync::{CommitContents, FullCommitContents, VerifiedCommitSummary},
    storage::{
        object_store::ObjectStore, read_store::ReadStore, write_store::WriteStore, ObjectKey,
    },
    transaction::VerifiedTransaction,
};

#[derive(Clone)]
pub struct StateSyncStore {
    cache_traits: ExecutionCacheTraitPointers,

    committee_store: Arc<CommitteeStore>,
    commit_store: Arc<CommitStore>,
    // in memory commit watermark sequence numbers
    highest_verified_commit: Arc<Mutex<Option<u64>>>,
    highest_synced_commit: Arc<Mutex<Option<u64>>>,
}

impl StateSyncStore {
    pub fn new(
        cache_traits: ExecutionCacheTraitPointers,
        committee_store: Arc<CommitteeStore>,
        commit_store: Arc<CommitStore>,
    ) -> Self {
        Self {
            cache_traits,
            committee_store,
            commit_store,
            highest_verified_commit: Arc::new(Mutex::new(None)),
            highest_synced_commit: Arc::new(Mutex::new(None)),
        }
    }

    pub fn get_objects(&self, object_keys: &[ObjectKey]) -> Result<Vec<Option<Object>>> {
        self.cache_traits
            .object_cache_reader
            .multi_get_objects_by_key(object_keys)
            .map_err(Into::into)
    }

    pub fn get_last_executed_commit(&self) -> Option<VerifiedCommitSummary> {
        self.commit_store
            .get_highest_executed_commit()
            .expect("db error")
    }
}

impl ReadStore for StateSyncStore {
    fn get_commit_by_digest(&self, digest: &CommitSummaryDigest) -> Option<VerifiedCommitSummary> {
        self.commit_store
            .get_commit_by_digest(digest)
            .expect("db error")
    }

    fn get_commit_by_index(&self, index: CommitIndex) -> Option<VerifiedCommitSummary> {
        self.commit_store
            .get_commit_by_index(index)
            .expect("db error")
    }

    fn get_highest_verified_commit(&self) -> Result<VerifiedCommitSummary, StorageError> {
        self.commit_store
            .get_highest_verified_commit()
            .map(|maybe_commit| {
                maybe_commit.expect("storage should have been initialized with genesis commit")
            })
            .map_err(Into::into)
    }

    fn get_highest_synced_commit(&self) -> Result<VerifiedCommitSummary, StorageError> {
        self.commit_store
            .get_highest_synced_commit()
            .map(|maybe_commit| {
                maybe_commit.expect("storage should have been initialized with genesis commit")
            })
            .map_err(Into::into)
    }

    fn get_lowest_available_commit(&self) -> Result<CommitIndex, StorageError> {
        // let highest_pruned_cp = self
        //     .commit_store
        //     .get_highest_pruned_commit_seq_number()
        //     .map_err(Into::<StorageError>::into)?;

        // if highest_pruned_cp == 0 {
        //     Ok(0)
        // } else {
        //     Ok(highest_pruned_cp + 1)
        // }

        Ok(0)
    }

    fn get_full_commit_contents_by_index(&self, index: CommitIndex) -> Option<FullCommitContents> {
        self.commit_store
            .get_full_commit_contents_by_index(index)
            .expect("db error")
    }

    fn get_full_commit_contents(
        &self,
        digest: &CommitContentsDigest,
    ) -> Option<FullCommitContents> {
        // First look to see if we saved the complete contents already.
        if let Some(seq_num) = self
            .commit_store
            .get_index_by_contents_digest(digest)
            .expect("db error")
        {
            let contents = self
                .commit_store
                .get_full_commit_contents_by_index(seq_num)
                .expect("db error");
            if contents.is_some() {
                return contents;
            }
        }

        // Otherwise gather it from the individual components.
        // Note we can't insert the constructed contents into `full_commit_content`,
        // because it needs to be inserted along with `commit_sequence_by_contents_digest`
        // and `commit_content`. However at this point it's likely we don't know the
        // corresponding sequence number yet.
        self.commit_store
            .get_commit_contents(digest)
            .expect("db error")
            .and_then(|contents| {
                let mut transactions = Vec::with_capacity(contents.size());
                for tx in contents.iter() {
                    if let Ok(Some(t)) = self.get_transaction(&tx) {
                        transactions.push((*t).clone().into_inner())
                    } else {
                        return None;
                    }
                }
                Some(FullCommitContents::from_contents_and_transactions(
                    contents,
                    transactions.into_iter(),
                ))
            })
    }

    fn get_committee(&self, epoch: EpochId) -> Result<Option<Arc<Committee>>> {
        self.committee_store
            .get_committee(&epoch)
            .map_err(Into::into)
    }

    fn get_transaction(
        &self,
        digest: &TransactionDigest,
    ) -> Result<Option<Arc<VerifiedTransaction>>> {
        self.cache_traits
            .transaction_cache_reader
            .get_transaction_block(digest)
            .map_err(Into::into)
    }

    fn get_transaction_effects(
        &self,
        digest: &TransactionDigest,
    ) -> Result<Option<TransactionEffects>> {
        self.cache_traits
            .transaction_cache_reader
            .get_executed_effects(digest)
            .map_err(Into::into)
    }
}

impl ObjectStore for StateSyncStore {
    fn get_object(&self, object_id: &ObjectID) -> Result<Option<Object>> {
        self.cache_traits.object_store.get_object(object_id)
    }

    fn get_object_by_key(&self, object_id: &ObjectID, version: Version) -> Result<Option<Object>> {
        self.cache_traits
            .object_store
            .get_object_by_key(object_id, version)
    }
}

impl WriteStore for StateSyncStore {
    fn insert_commit(&self, commit: &VerifiedCommitSummary) -> Result<()> {
        // TODO: insert committee when inserting commit
        // if let Some(EndOfEpochData {
        //     next_epoch_committee,
        //     ..
        // }) = commit.end_of_epoch_data.as_ref()
        // {
        //     let next_committee = next_epoch_committee.iter().cloned().collect();
        //     let committee = Committee::new(commit.epoch().checked_add(1).unwrap(), next_committee);
        //     self.insert_committee(committee)?;
        // }

        self.commit_store
            .insert_verified_commit(commit)
            .map_err(Into::into)
    }

    fn update_highest_synced_commit(&self, commit: &VerifiedCommitSummary) -> Result<()> {
        let mut locked = self.highest_synced_commit.lock();
        if locked.is_some() && locked.unwrap() >= commit.index {
            return Ok(());
        }
        self.commit_store
            .update_highest_synced_commit(commit)
            .map_err(StorageError::custom)?;
        *locked = Some(commit.index);
        Ok(())
    }

    fn update_highest_verified_commit(&self, commit: &VerifiedCommitSummary) -> Result<()> {
        let mut locked = self.highest_verified_commit.lock();
        if locked.is_some() && locked.unwrap() >= commit.index {
            return Ok(());
        }
        self.commit_store
            .update_highest_verified_commit(commit)
            .map_err(StorageError::custom)?;
        *locked = Some(commit.index);
        Ok(())
    }

    fn insert_commit_contents(
        &self,
        commit: &VerifiedCommitSummary,
        contents: VerifiedCommitContents,
    ) -> Result<()> {
        self.cache_traits
            .state_sync_store
            .multi_insert_transactions(contents.transactions());
        self.commit_store
            .insert_verified_commit_contents(commit, contents)
            .map_err(Into::into)
    }

    fn insert_committee(&self, new_committee: Committee) -> Result<()> {
        self.committee_store
            .insert_new_committee(new_committee)
            .unwrap();
        Ok(())
    }
}
