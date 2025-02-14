use std::{io::Read, sync::Arc};

use crate::{
    accumulator::{AccumulatorStore, CommitIndex},
    committee::Committee,
    consensus::commit::CommittedSubDag,
};

use super::{
    consensus::ConsensusStore, object_store::ObjectStore, read_store::ReadStore,
    storage_error::Result,
};

pub trait WriteStore: ReadStore {
    fn insert_committee(&self, new_committee: Committee) -> Result<()>;
    fn insert_commit(&self, commit: CommittedSubDag) -> Result<()>;
    fn update_highest_synced_commit(&self, commit: &CommittedSubDag) -> Result<()>;
}

impl<T: WriteStore + ?Sized> WriteStore for &T {
    fn insert_commit(&self, commit: CommittedSubDag) -> Result<()> {
        (*self).insert_commit(commit)
    }

    fn insert_committee(&self, new_committee: Committee) -> Result<()> {
        (*self).insert_committee(new_committee)
    }

    fn update_highest_synced_commit(&self, commit: &CommittedSubDag) -> Result<()> {
        (*self).update_highest_synced_commit(commit)
    }
}

#[derive(Clone, Debug)]
pub struct TestP2pStore {}

impl TestP2pStore {
    pub fn new() -> Self {
        Self {}
    }
}

impl AccumulatorStore for TestP2pStore {
    fn get_root_state_accumulator_for_epoch(
        &self,
        epoch: crate::committee::EpochId,
    ) -> crate::error::SomaResult<Option<(CommitIndex, crate::accumulator::Accumulator)>> {
        todo!()
    }

    fn get_root_state_accumulator_for_highest_epoch(
        &self,
    ) -> crate::error::SomaResult<
        Option<(
            crate::committee::EpochId,
            (CommitIndex, crate::accumulator::Accumulator),
        )>,
    > {
        todo!()
    }

    fn insert_state_accumulator_for_epoch(
        &self,
        epoch: crate::committee::EpochId,
        commit: &CommitIndex,
        acc: &crate::accumulator::Accumulator,
    ) -> crate::error::SomaResult {
        todo!()
    }

    fn iter_live_object_set(&self) -> Box<dyn Iterator<Item = crate::object::LiveObject> + '_> {
        todo!()
    }
}

impl WriteStore for TestP2pStore {
    fn insert_committee(&self, _new_committee: Committee) -> Result<()> {
        Ok(())
    }

    fn insert_commit(&self, _commit: CommittedSubDag) -> Result<()> {
        Ok(())
    }

    fn update_highest_synced_commit(&self, _commit: &CommittedSubDag) -> Result<()> {
        Ok(())
    }
}

impl ObjectStore for TestP2pStore {
    fn get_object(
        &self,
        object_id: &crate::object::ObjectID,
    ) -> Result<Option<crate::object::Object>> {
        todo!()
    }

    fn get_object_by_key(
        &self,
        object_id: &crate::object::ObjectID,
        version: crate::object::Version,
    ) -> Result<Option<crate::object::Object>> {
        todo!()
    }
}

impl ReadStore for TestP2pStore {
    fn get_committee(&self, epoch: crate::committee::EpochId) -> Result<Option<Arc<Committee>>> {
        todo!()
    }

    fn get_highest_synced_commit(&self) -> Result<CommittedSubDag> {
        todo!()
    }

    fn get_lowest_available_commit(&self) -> Result<CommitIndex> {
        todo!()
    }

    fn get_commit_by_digest(
        &self,
        digest: &crate::consensus::commit::CommitDigest,
    ) -> Option<CommittedSubDag> {
        todo!()
    }

    fn get_commit_by_index(&self, index: CommitIndex) -> Option<CommittedSubDag> {
        todo!()
    }

    fn get_last_commit_index_of_epoch(
        &self,
        epoch: crate::committee::EpochId,
    ) -> Option<CommitIndex> {
        todo!()
    }

    fn get_transaction(
        &self,
        tx_digest: &crate::digests::TransactionDigest,
    ) -> Result<Option<std::sync::Arc<crate::transaction::VerifiedTransaction>>> {
        todo!()
    }

    fn get_transaction_effects(
        &self,
        tx_digest: &crate::digests::TransactionDigest,
    ) -> Result<Option<crate::effects::TransactionEffects>> {
        todo!()
    }

    fn multi_get_transactions(
        &self,
        tx_digests: &[crate::digests::TransactionDigest],
    ) -> Result<Vec<Option<std::sync::Arc<crate::transaction::VerifiedTransaction>>>> {
        tx_digests
            .iter()
            .map(|digest| self.get_transaction(digest))
            .collect::<Result<Vec<_>, _>>()
    }

    fn multi_get_transaction_effects(
        &self,
        tx_digests: &[crate::digests::TransactionDigest],
    ) -> Result<Vec<Option<crate::effects::TransactionEffects>>> {
        tx_digests
            .iter()
            .map(|digest| self.get_transaction_effects(digest))
            .collect::<Result<Vec<_>, _>>()
    }
}

impl ConsensusStore for TestP2pStore {
    fn write(
        &self,
        write_batch: super::consensus::WriteBatch,
    ) -> crate::error::ConsensusResult<()> {
        todo!()
    }

    fn read_blocks(
        &self,
        refs: &[crate::consensus::block::BlockRef],
    ) -> crate::error::ConsensusResult<Vec<Option<crate::consensus::block::VerifiedBlock>>> {
        todo!()
    }

    fn contains_blocks(
        &self,
        refs: &[crate::consensus::block::BlockRef],
    ) -> crate::error::ConsensusResult<Vec<bool>> {
        todo!()
    }

    fn contains_block_at_slot(
        &self,
        slot: crate::consensus::block::Slot,
    ) -> crate::error::ConsensusResult<bool> {
        todo!()
    }

    fn scan_blocks_by_author(
        &self,
        authority: crate::committee::AuthorityIndex,
        start_round: crate::consensus::block::Round,
    ) -> crate::error::ConsensusResult<Vec<crate::consensus::block::VerifiedBlock>> {
        todo!()
    }

    fn scan_last_blocks_by_author(
        &self,
        author: crate::committee::AuthorityIndex,
        num_of_rounds: u64,
        before_round: Option<crate::consensus::block::Round>,
    ) -> crate::error::ConsensusResult<Vec<crate::consensus::block::VerifiedBlock>> {
        todo!()
    }

    fn read_last_commit(
        &self,
    ) -> crate::error::ConsensusResult<Option<crate::consensus::commit::TrustedCommit>> {
        todo!()
    }

    fn scan_commits(
        &self,
        range: crate::consensus::commit::CommitRange,
    ) -> crate::error::ConsensusResult<Vec<crate::consensus::commit::TrustedCommit>> {
        todo!()
    }

    fn read_commit_votes(
        &self,
        commit_index: crate::consensus::commit::CommitIndex,
    ) -> crate::error::ConsensusResult<Vec<crate::consensus::block::BlockRef>> {
        todo!()
    }

    fn read_last_commit_info(
        &self,
    ) -> crate::error::ConsensusResult<
        Option<(
            crate::consensus::commit::CommitRef,
            crate::consensus::commit::CommitInfo,
        )>,
    > {
        todo!()
    }
}
