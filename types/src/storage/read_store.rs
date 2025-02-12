use std::sync::Arc;

use crate::{
    accumulator::CommitIndex,
    committee::{Committee, EpochId},
    consensus::commit::{CommitDigest, CommittedSubDag},
    digests::TransactionDigest,
    effects::TransactionEffects,
    transaction::VerifiedTransaction,
};

use super::{object_store::ObjectStore, storage_error::Result};

pub trait ReadCommitteeStore {
    fn get_committee(&self, epoch: EpochId) -> Result<Option<Arc<Committee>>>;
}

impl<T: ReadCommitteeStore + ?Sized> ReadCommitteeStore for &T {
    fn get_committee(&self, epoch: EpochId) -> Result<Option<Arc<Committee>>> {
        (*self).get_committee(epoch)
    }
}

pub trait ReadStore: ReadCommitteeStore + ObjectStore + Send + Sync {
    //
    // Commit Getters
    //

    /// Get the highest synced commit. This is the highest commit that has been synced from
    /// state-sync.
    fn get_highest_synced_commit(&self) -> Result<CommittedSubDag>;

    /// Lowest available commit for which transaction data can be requested.
    fn get_lowest_available_commit(&self) -> Result<CommitIndex>;

    fn get_commit_by_digest(&self, digest: &CommitDigest) -> Option<CommittedSubDag>;

    fn get_commit_by_index(&self, index: CommitIndex) -> Option<CommittedSubDag>;

    fn get_last_commit_index_of_epoch(&self, epoch: EpochId) -> Option<CommitIndex>;

    //
    // Transaction Getters
    //

    fn get_transaction(
        &self,
        tx_digest: &TransactionDigest,
    ) -> Result<Option<Arc<VerifiedTransaction>>>;

    fn multi_get_transactions(
        &self,
        tx_digests: &[TransactionDigest],
    ) -> Result<Vec<Option<Arc<VerifiedTransaction>>>> {
        tx_digests
            .iter()
            .map(|digest| self.get_transaction(digest))
            .collect::<Result<Vec<_>, _>>()
    }

    fn get_transaction_effects(
        &self,
        tx_digest: &TransactionDigest,
    ) -> Result<Option<TransactionEffects>>;

    fn multi_get_transaction_effects(
        &self,
        tx_digests: &[TransactionDigest],
    ) -> Result<Vec<Option<TransactionEffects>>> {
        tx_digests
            .iter()
            .map(|digest| self.get_transaction_effects(digest))
            .collect::<Result<Vec<_>, _>>()
    }
}

impl<T: ReadStore + ?Sized> ReadStore for &T {
    fn get_highest_synced_commit(&self) -> Result<CommittedSubDag> {
        (*self).get_highest_synced_commit()
    }

    fn get_lowest_available_commit(&self) -> Result<CommitIndex> {
        (*self).get_lowest_available_commit()
    }

    fn get_commit_by_digest(&self, digest: &CommitDigest) -> Option<CommittedSubDag> {
        (*self).get_commit_by_digest(digest)
    }

    fn get_commit_by_index(&self, index: CommitIndex) -> Option<CommittedSubDag> {
        (*self).get_commit_by_index(index)
    }

    fn get_last_commit_index_of_epoch(&self, epoch: EpochId) -> Option<CommitIndex> {
        (*self).get_last_commit_index_of_epoch(epoch)
    }

    fn get_transaction(
        &self,
        tx_digest: &TransactionDigest,
    ) -> Result<Option<Arc<VerifiedTransaction>>> {
        (*self).get_transaction(tx_digest)
    }

    fn multi_get_transactions(
        &self,
        tx_digests: &[TransactionDigest],
    ) -> Result<Vec<Option<Arc<VerifiedTransaction>>>> {
        (*self).multi_get_transactions(tx_digests)
    }

    fn get_transaction_effects(
        &self,
        tx_digest: &TransactionDigest,
    ) -> Result<Option<TransactionEffects>> {
        (*self).get_transaction_effects(tx_digest)
    }

    fn multi_get_transaction_effects(
        &self,
        tx_digests: &[TransactionDigest],
    ) -> Result<Vec<Option<TransactionEffects>>> {
        (*self).multi_get_transaction_effects(tx_digests)
    }
}
