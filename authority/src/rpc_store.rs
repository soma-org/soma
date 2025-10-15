use crate::rpc_index::{OwnerIndexInfo, OwnerIndexKey};
use crate::{rpc_index::RpcIndexStore, state::AuthorityState, state_sync_store::StateSyncStore};
use std::sync::Arc;
use tap::Pipe;
use types::accumulator::CommitIndex;
use types::base::SomaAddress;
use types::committee::{Committee, EpochId};
use types::consensus::commit::{CommitDigest, CommittedSubDag};
use types::digests::TransactionDigest;
use types::effects::TransactionEffects;
use types::object::ObjectType;
use types::storage::read_store::{
    BalanceInfo, EpochInfo, OwnedObjectInfo, ReadCommitteeStore, ReadStore, RpcIndexes,
    RpcStateReader, TransactionInfo,
};
use types::storage::storage_error::Error as StorageError;
use types::storage::storage_error::Result as StorageResult;
use types::transaction::VerifiedTransaction;
use types::{object::Object, storage::object_store::ObjectStore};

pub struct RestReadStore {
    state: Arc<AuthorityState>,
    store: StateSyncStore,
}

impl RestReadStore {
    pub fn new(state: Arc<AuthorityState>, store: StateSyncStore) -> Self {
        Self { state, store }
    }

    fn index(&self) -> types::storage::storage_error::Result<&RpcIndexStore> {
        self.state.rpc_index.as_deref().ok_or_else(|| {
            types::storage::storage_error::Error::custom("rest index store is disabled")
        })
    }
}

impl ObjectStore for RestReadStore {
    fn get_object(&self, object_id: &types::object::ObjectID) -> StorageResult<Option<Object>> {
        self.store.get_object(object_id)
    }

    fn get_object_by_key(
        &self,
        object_id: &types::object::ObjectID,
        version: types::object::Version,
    ) -> StorageResult<Option<Object>> {
        self.store.get_object_by_key(object_id, version)
    }
}

impl ReadCommitteeStore for RestReadStore {
    fn get_committee(&self, epoch: EpochId) -> StorageResult<Option<Arc<Committee>>> {
        self.store.get_committee(epoch)
    }
}

impl ReadStore for RestReadStore {
    fn get_highest_synced_commit(&self) -> StorageResult<CommittedSubDag> {
        self.store.get_highest_synced_commit()
    }

    fn get_lowest_available_commit(&self) -> StorageResult<CommitIndex> {
        self.store.get_lowest_available_commit()
    }

    fn get_commit_by_digest(&self, digest: &CommitDigest) -> Option<CommittedSubDag> {
        self.store.get_commit_by_digest(digest)
    }

    fn get_commit_by_index(&self, index: CommitIndex) -> Option<CommittedSubDag> {
        self.store.get_commit_by_index(index)
    }

    fn get_last_commit_index_of_epoch(&self, epoch: EpochId) -> Option<CommitIndex> {
        self.store.get_last_commit_index_of_epoch(epoch)
    }

    fn get_transaction(
        &self,
        digest: &TransactionDigest,
    ) -> StorageResult<Option<Arc<VerifiedTransaction>>> {
        self.store.get_transaction(digest)
    }

    fn get_transaction_effects(
        &self,
        digest: &TransactionDigest,
    ) -> StorageResult<Option<TransactionEffects>> {
        self.store.get_transaction_effects(digest)
    }
}

impl RpcStateReader for RestReadStore {
    fn indexes(&self) -> Option<&dyn RpcIndexes> {
        self.index().ok().map(|index| index as _)
    }
}

impl RpcIndexes for RpcIndexStore {
    fn get_epoch_info(&self, epoch: EpochId) -> StorageResult<Option<EpochInfo>> {
        self.get_epoch_info(epoch).map_err(StorageError::custom)
    }

    fn get_transaction_info(
        &self,
        digest: &TransactionDigest,
    ) -> StorageResult<Option<TransactionInfo>> {
        self.get_transaction_info(digest)
            .map_err(StorageError::custom)
    }

    fn owned_objects_iter(
        &self,
        owner: SomaAddress,
        object_type: Option<ObjectType>,
        cursor: Option<OwnedObjectInfo>,
    ) -> StorageResult<
        Box<
            dyn Iterator<Item = Result<OwnedObjectInfo, types::storage::storage_error::Error>> + '_,
        >,
    > {
        let cursor = cursor.map(|cursor| OwnerIndexKey {
            owner: cursor.owner,
            object_type: cursor.object_type,
            inverted_balance: cursor.balance.map(std::ops::Not::not),
            object_id: cursor.object_id,
        });

        let iter = self.owner_iter(owner, object_type, cursor)?.map(|result| {
            result.map(
                |(
                    OwnerIndexKey {
                        owner,
                        object_id,
                        object_type,
                        inverted_balance,
                    },
                    OwnerIndexInfo { version },
                )| {
                    OwnedObjectInfo {
                        owner,
                        object_type,
                        balance: inverted_balance.map(std::ops::Not::not),
                        object_id,
                        version,
                    }
                },
            )
        });

        Ok(Box::new(iter) as _)
    }

    fn get_balance(&self, owner: &SomaAddress) -> StorageResult<Option<u64>> {
        self.get_balance(owner)?
            .map(|info| info.balance_delta.clamp(0, u64::MAX as i128) as u64)
            .pipe(Ok)
    }
}
