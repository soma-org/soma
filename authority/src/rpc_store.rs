use std::sync::Arc;

use crate::{rpc_index::RpcIndexStore, state::AuthorityState, state_sync_store::StateSyncStore};
use types::storage::read_store::{ReadStore, RpcIndexes, RpcStateReader};
use types::storage::storage_error::Result;
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
    fn get_object(&self, object_id: &types::object::ObjectID) -> Result<Option<Object>> {
        self.store.get_object(object_id)
    }

    fn get_object_by_key(
        &self,
        object_id: &types::object::ObjectID,
        version: types::object::Version,
    ) -> Result<Option<Object>> {
        self.store.get_object_by_key(object_id, version)
    }
}

impl ReadStore for RestReadStore {
    fn get_committee(&self, epoch: EpochId) -> Option<Arc<Committee>> {
        self.rocks.get_committee(epoch)
    }

    fn get_latest_checkpoint(&self) -> sui_types::storage::error::Result<VerifiedCheckpoint> {
        self.rocks.get_latest_checkpoint()
    }

    fn get_highest_verified_checkpoint(
        &self,
    ) -> sui_types::storage::error::Result<VerifiedCheckpoint> {
        self.rocks.get_highest_verified_checkpoint()
    }

    fn get_highest_synced_checkpoint(
        &self,
    ) -> sui_types::storage::error::Result<VerifiedCheckpoint> {
        self.rocks.get_highest_synced_checkpoint()
    }

    fn get_lowest_available_checkpoint(
        &self,
    ) -> sui_types::storage::error::Result<CheckpointSequenceNumber> {
        self.rocks.get_lowest_available_checkpoint()
    }

    fn get_checkpoint_by_digest(&self, digest: &CheckpointDigest) -> Option<VerifiedCheckpoint> {
        self.rocks.get_checkpoint_by_digest(digest)
    }

    fn get_checkpoint_by_sequence_number(
        &self,
        sequence_number: CheckpointSequenceNumber,
    ) -> Option<VerifiedCheckpoint> {
        self.rocks
            .get_checkpoint_by_sequence_number(sequence_number)
    }

    fn get_checkpoint_contents_by_digest(
        &self,
        digest: &CheckpointContentsDigest,
    ) -> Option<sui_types::messages_checkpoint::CheckpointContents> {
        self.rocks.get_checkpoint_contents_by_digest(digest)
    }

    fn get_checkpoint_contents_by_sequence_number(
        &self,
        sequence_number: CheckpointSequenceNumber,
    ) -> Option<sui_types::messages_checkpoint::CheckpointContents> {
        self.rocks
            .get_checkpoint_contents_by_sequence_number(sequence_number)
    }

    fn get_transaction(&self, digest: &TransactionDigest) -> Option<Arc<VerifiedTransaction>> {
        self.rocks.get_transaction(digest)
    }

    fn get_transaction_effects(&self, digest: &TransactionDigest) -> Option<TransactionEffects> {
        self.rocks.get_transaction_effects(digest)
    }

    fn get_events(&self, digest: &TransactionDigest) -> Option<TransactionEvents> {
        self.rocks.get_events(digest)
    }

    fn get_full_checkpoint_contents(
        &self,
        sequence_number: Option<CheckpointSequenceNumber>,
        digest: &CheckpointContentsDigest,
    ) -> Option<FullCheckpointContents> {
        self.rocks
            .get_full_checkpoint_contents(sequence_number, digest)
    }

    fn get_unchanged_loaded_runtime_objects(
        &self,
        digest: &TransactionDigest,
    ) -> Option<Vec<ObjectKey>> {
        self.rocks.get_unchanged_loaded_runtime_objects(digest)
    }
}

impl RpcStateReader for RestReadStore {
    fn get_lowest_available_checkpoint_objects(
        &self,
    ) -> sui_types::storage::error::Result<CheckpointSequenceNumber> {
        Ok(self
            .state
            .get_object_cache_reader()
            .get_highest_pruned_checkpoint()
            .map(|cp| cp + 1)
            .unwrap_or(0))
    }

    fn get_chain_identifier(&self) -> Result<sui_types::digests::ChainIdentifier> {
        Ok(self.state.get_chain_identifier())
    }

    fn indexes(&self) -> Option<&dyn RpcIndexes> {
        self.index().ok().map(|index| index as _)
    }

    fn get_struct_layout(
        &self,
        struct_tag: &move_core_types::language_storage::StructTag,
    ) -> Result<Option<move_core_types::annotated_value::MoveTypeLayout>> {
        self.state
            .load_epoch_store_one_call_per_task()
            .executor()
            // TODO(cache) - must read through cache
            .type_layout_resolver(Box::new(self.state.get_backing_package_store().as_ref()))
            .get_annotated_layout(struct_tag)
            .map(|layout| layout.into_layout())
            .map(Some)
            .map_err(StorageError::custom)
    }
}

impl RpcIndexes for RpcIndexStore {
    fn get_epoch_info(&self, epoch: EpochId) -> Result<Option<sui_types::storage::EpochInfo>> {
        self.get_epoch_info(epoch).map_err(StorageError::custom)
    }

    fn get_transaction_info(
        &self,
        digest: &TransactionDigest,
    ) -> sui_types::storage::error::Result<Option<TransactionInfo>> {
        self.get_transaction_info(digest)
            .map_err(StorageError::custom)
    }

    fn owned_objects_iter(
        &self,
        owner: SuiAddress,
        object_type: Option<StructTag>,
        cursor: Option<OwnedObjectInfo>,
    ) -> Result<Box<dyn Iterator<Item = Result<OwnedObjectInfo, TypedStoreError>> + '_>> {
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

    fn get_balance(
        &self,
        owner: &SuiAddress,
        coin_type: &StructTag,
    ) -> sui_types::storage::error::Result<Option<BalanceInfo>> {
        self.get_balance(owner, coin_type)?
            .map(|info| info.into())
            .pipe(Ok)
    }

    fn balance_iter(
        &self,
        owner: &SuiAddress,
        cursor: Option<(SuiAddress, StructTag)>,
    ) -> sui_types::storage::error::Result<BalanceIterator<'_>> {
        let cursor_key =
            cursor.map(|(owner, coin_type)| crate::rpc_index::BalanceKey { owner, coin_type });

        Ok(Box::new(self.balance_iter(*owner, cursor_key)?.map(
            |result| {
                result
                    .map(|(key, info)| (key.coin_type, info.into()))
                    .map_err(Into::into)
            },
        )))
    }
}
