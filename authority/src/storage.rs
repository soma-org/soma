// Portions of this file are derived from Sui (MystenLabs/sui).
// Original source: https://github.com/MystenLabs/sui/tree/main/crates/sui-core/src/
// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

use crate::authority::AuthorityState;
use crate::cache::ExecutionCacheTraitPointers;
use crate::checkpoints::CheckpointStore;
use crate::rpc_index::OwnerIndexInfo;
use crate::rpc_index::OwnerIndexKey;
use crate::rpc_index::RpcIndexStore;
use parking_lot::Mutex;
use store::TypedStoreError;
use tap::Pipe as _;
use tap::TapFallible as _;
use types::base::SomaAddress;
use types::checkpoints::EndOfEpochData;
use types::object::ObjectID;
use types::object::ObjectType;
use types::storage::read_store::BalanceInfo;
use types::storage::read_store::ChallengeInfo;
use types::storage::read_store::OwnedObjectInfo;
use types::storage::read_store::RpcIndexes;
use types::storage::read_store::RpcStateReader;
use types::storage::read_store::TargetInfo;

use std::sync::Arc;
use tracing::error;
use types::checkpoints::CheckpointSequenceNumber;
use types::checkpoints::FullCheckpointContents;
use types::checkpoints::VerifiedCheckpoint;
use types::checkpoints::VerifiedCheckpointContents;

use types::digests::CheckpointContentsDigest;
use types::digests::CheckpointDigest;

use types::storage::committee_store::CommitteeStore;

use types::storage::storage_error::Error as StorageError;
use types::storage::storage_error::Result;
use types::{
    committee::{Committee, EpochId},
    digests::TransactionDigest,
    effects::TransactionEffects,
    object::Object,
    storage::{
        ObjectKey, object_store::ObjectStore, read_store::ReadStore, write_store::WriteStore,
    },
    transaction::VerifiedTransaction,
};
#[derive(Clone)]
pub struct RocksDbStore {
    cache_traits: ExecutionCacheTraitPointers,

    committee_store: Arc<CommitteeStore>,
    checkpoint_store: Arc<CheckpointStore>,
    // in memory checkpoint watermark sequence numbers
    highest_verified_checkpoint: Arc<Mutex<Option<u64>>>,
    highest_synced_checkpoint: Arc<Mutex<Option<u64>>>,
}

impl RocksDbStore {
    pub fn new(
        cache_traits: ExecutionCacheTraitPointers,
        committee_store: Arc<CommitteeStore>,
        checkpoint_store: Arc<CheckpointStore>,
    ) -> Self {
        Self {
            cache_traits,
            committee_store,
            checkpoint_store,
            highest_verified_checkpoint: Arc::new(Mutex::new(None)),
            highest_synced_checkpoint: Arc::new(Mutex::new(None)),
        }
    }

    pub fn get_objects(&self, object_keys: &[ObjectKey]) -> Vec<Option<Object>> {
        self.cache_traits.object_cache_reader.multi_get_objects_by_key(object_keys)
    }

    pub fn get_last_executed_checkpoint(&self) -> Option<VerifiedCheckpoint> {
        self.checkpoint_store.get_highest_executed_checkpoint().expect("db error")
    }
}

impl ReadStore for RocksDbStore {
    fn get_checkpoint_by_digest(&self, digest: &CheckpointDigest) -> Option<VerifiedCheckpoint> {
        self.checkpoint_store.get_checkpoint_by_digest(digest).expect("db error")
    }

    fn get_checkpoint_by_sequence_number(
        &self,
        sequence_number: CheckpointSequenceNumber,
    ) -> Option<VerifiedCheckpoint> {
        self.checkpoint_store.get_checkpoint_by_sequence_number(sequence_number).expect("db error")
    }

    fn get_highest_verified_checkpoint(&self) -> Result<VerifiedCheckpoint, StorageError> {
        self.checkpoint_store
            .get_highest_verified_checkpoint()
            .map(|maybe_checkpoint| {
                maybe_checkpoint
                    .expect("storage should have been initialized with genesis checkpoint")
            })
            .map_err(Into::into)
    }

    fn get_highest_synced_checkpoint(&self) -> Result<VerifiedCheckpoint, StorageError> {
        self.checkpoint_store
            .get_highest_synced_checkpoint()
            .map(|maybe_checkpoint| {
                maybe_checkpoint
                    .expect("storage should have been initialized with genesis checkpoint")
            })
            .map_err(Into::into)
    }

    fn get_lowest_available_checkpoint(&self) -> Result<CheckpointSequenceNumber, StorageError> {
        if let Some(highest_pruned_cp) = self
            .checkpoint_store
            .get_highest_pruned_checkpoint_seq_number()
            .map_err(Into::<StorageError>::into)?
        {
            Ok(highest_pruned_cp + 1)
        } else {
            Ok(0)
        }
    }

    fn get_full_checkpoint_contents(
        &self,
        sequence_number: Option<CheckpointSequenceNumber>,
        digest: &CheckpointContentsDigest,
    ) -> Option<FullCheckpointContents> {
        #[cfg(debug_assertions)]
        if let Some(sequence_number) = sequence_number {
            // When sequence_number is provided as an optimization, we want to ensure that
            // the sequence number we get from the db matches the one we provided.
            // Only check this in debug mode though.
            if let Some(loaded_sequence_number) = self
                .checkpoint_store
                .get_sequence_number_by_contents_digest(digest)
                .expect("db error")
            {
                assert_eq!(loaded_sequence_number, sequence_number);
            }
        }

        let sequence_number = sequence_number.or_else(|| {
            self.checkpoint_store.get_sequence_number_by_contents_digest(digest).expect("db error")
        });
        if let Some(sequence_number) = sequence_number {
            // Note: We don't use `?` here because we want to tolerate
            // potential db errors due to data corruption.
            // In that case, we will fallback and construct the contents
            // from the individual components as if we could not find the
            // cached full contents.
            if let Ok(Some(contents)) = self
                .checkpoint_store
                .get_full_checkpoint_contents_by_sequence_number(sequence_number)
                .tap_err(|e| {
                    error!(
                        "error getting full checkpoint contents for checkpoint {:?}: {:?}",
                        sequence_number, e
                    )
                })
            {
                return Some(contents);
            }
        }

        // Otherwise gather it from the individual components.
        // Note we can't insert the constructed contents into `full_checkpoint_content`,
        // because it needs to be inserted along with `checkpoint_sequence_by_contents_digest`
        // and `checkpoint_content`. However at this point it's likely we don't know the
        // corresponding sequence number yet.
        self.checkpoint_store.get_checkpoint_contents(digest).expect("db error").and_then(
            |contents| {
                let mut transactions = Vec::with_capacity(contents.size());
                for tx in contents.iter() {
                    if let (Some(t), Some(e)) = (
                        self.get_transaction(&tx.transaction),
                        self.cache_traits.transaction_cache_reader.get_effects(&tx.effects),
                    ) {
                        transactions
                            .push(types::base::ExecutionData::new((*t).clone().into_inner(), e))
                    } else {
                        return None;
                    }
                }
                Some(FullCheckpointContents::from_contents_and_execution_data(
                    contents,
                    transactions.into_iter(),
                ))
            },
        )
    }

    fn get_committee(&self, epoch: EpochId) -> Option<Arc<Committee>> {
        self.committee_store.get_committee(&epoch).unwrap()
    }

    fn get_transaction(&self, digest: &TransactionDigest) -> Option<Arc<VerifiedTransaction>> {
        self.cache_traits.transaction_cache_reader.get_transaction_block(digest)
    }

    fn get_transaction_effects(&self, digest: &TransactionDigest) -> Option<TransactionEffects> {
        self.cache_traits.transaction_cache_reader.get_executed_effects(digest)
    }

    fn get_latest_checkpoint(&self) -> types::storage::storage_error::Result<VerifiedCheckpoint> {
        self.checkpoint_store.get_highest_executed_checkpoint().expect("db error").ok_or_else(
            || types::storage::storage_error::Error::missing("unable to get latest checkpoint"),
        )
    }

    fn get_checkpoint_contents_by_digest(
        &self,
        digest: &CheckpointContentsDigest,
    ) -> Option<types::checkpoints::CheckpointContents> {
        self.checkpoint_store.get_checkpoint_contents(digest).expect("db error")
    }

    fn get_checkpoint_contents_by_sequence_number(
        &self,
        sequence_number: CheckpointSequenceNumber,
    ) -> Option<types::checkpoints::CheckpointContents> {
        match self.get_checkpoint_by_sequence_number(sequence_number) {
            Some(checkpoint) => self.get_checkpoint_contents_by_digest(&checkpoint.content_digest),
            None => None,
        }
    }
}

impl ObjectStore for RocksDbStore {
    fn get_object(&self, object_id: &types::object::ObjectID) -> Option<Object> {
        self.cache_traits.object_store.get_object(object_id)
    }

    fn get_object_by_key(
        &self,
        object_id: &types::object::ObjectID,
        version: types::object::Version,
    ) -> Option<Object> {
        self.cache_traits.object_store.get_object_by_key(object_id, version)
    }
}

impl WriteStore for RocksDbStore {
    fn insert_checkpoint(
        &self,
        checkpoint: &VerifiedCheckpoint,
    ) -> Result<(), types::storage::storage_error::Error> {
        if let Some(EndOfEpochData { next_epoch_validator_committee, .. }) =
            checkpoint.end_of_epoch_data.as_ref()
        {
            self.insert_committee(next_epoch_validator_committee.clone())?;
        }

        self.checkpoint_store.insert_verified_checkpoint(checkpoint).map_err(Into::into)
    }

    fn update_highest_synced_checkpoint(
        &self,
        checkpoint: &VerifiedCheckpoint,
    ) -> Result<(), types::storage::storage_error::Error> {
        let mut locked = self.highest_synced_checkpoint.lock();
        if locked.is_some() && locked.unwrap() >= checkpoint.sequence_number {
            return Ok(());
        }
        self.checkpoint_store
            .update_highest_synced_checkpoint(checkpoint)
            .map_err(types::storage::storage_error::Error::custom)?;
        *locked = Some(checkpoint.sequence_number);
        Ok(())
    }

    fn update_highest_verified_checkpoint(
        &self,
        checkpoint: &VerifiedCheckpoint,
    ) -> Result<(), types::storage::storage_error::Error> {
        let mut locked = self.highest_verified_checkpoint.lock();
        if locked.is_some() && locked.unwrap() >= checkpoint.sequence_number {
            return Ok(());
        }
        self.checkpoint_store
            .update_highest_verified_checkpoint(checkpoint)
            .map_err(types::storage::storage_error::Error::custom)?;
        *locked = Some(checkpoint.sequence_number);
        Ok(())
    }

    fn insert_checkpoint_contents(
        &self,
        checkpoint: &VerifiedCheckpoint,
        contents: VerifiedCheckpointContents,
    ) -> Result<(), types::storage::storage_error::Error> {
        self.cache_traits
            .state_sync_store
            .multi_insert_transaction_and_effects(contents.transactions());
        self.checkpoint_store
            .insert_verified_checkpoint_contents(checkpoint, contents)
            .map_err(Into::into)
    }

    fn insert_committee(
        &self,
        new_committee: Committee,
    ) -> Result<(), types::storage::storage_error::Error> {
        self.committee_store.insert_new_committee(&new_committee).unwrap();
        Ok(())
    }
}

pub struct RestReadStore {
    state: Arc<AuthorityState>,
    rocks: RocksDbStore,
}

impl RestReadStore {
    pub fn new(state: Arc<AuthorityState>, rocks: RocksDbStore) -> Self {
        Self { state, rocks }
    }

    fn index(&self) -> types::storage::storage_error::Result<&RpcIndexStore> {
        self.state.rpc_index.as_deref().ok_or_else(|| {
            types::storage::storage_error::Error::custom("rest index store is disabled")
        })
    }
}

impl ObjectStore for RestReadStore {
    fn get_object(&self, object_id: &types::object::ObjectID) -> Option<Object> {
        self.rocks.get_object(object_id)
    }

    fn get_object_by_key(
        &self,
        object_id: &types::object::ObjectID,
        version: types::object::Version,
    ) -> Option<Object> {
        self.rocks.get_object_by_key(object_id, version)
    }
}

impl ReadStore for RestReadStore {
    fn get_committee(&self, epoch: EpochId) -> Option<Arc<Committee>> {
        self.rocks.get_committee(epoch)
    }

    fn get_latest_checkpoint(&self) -> types::storage::storage_error::Result<VerifiedCheckpoint> {
        self.rocks.get_latest_checkpoint()
    }

    fn get_highest_verified_checkpoint(
        &self,
    ) -> types::storage::storage_error::Result<VerifiedCheckpoint> {
        self.rocks.get_highest_verified_checkpoint()
    }

    fn get_highest_synced_checkpoint(
        &self,
    ) -> types::storage::storage_error::Result<VerifiedCheckpoint> {
        self.rocks.get_highest_synced_checkpoint()
    }

    fn get_lowest_available_checkpoint(
        &self,
    ) -> types::storage::storage_error::Result<CheckpointSequenceNumber> {
        self.rocks.get_lowest_available_checkpoint()
    }

    fn get_checkpoint_by_digest(&self, digest: &CheckpointDigest) -> Option<VerifiedCheckpoint> {
        self.rocks.get_checkpoint_by_digest(digest)
    }

    fn get_checkpoint_by_sequence_number(
        &self,
        sequence_number: CheckpointSequenceNumber,
    ) -> Option<VerifiedCheckpoint> {
        self.rocks.get_checkpoint_by_sequence_number(sequence_number)
    }

    fn get_checkpoint_contents_by_digest(
        &self,
        digest: &CheckpointContentsDigest,
    ) -> Option<types::checkpoints::CheckpointContents> {
        self.rocks.get_checkpoint_contents_by_digest(digest)
    }

    fn get_checkpoint_contents_by_sequence_number(
        &self,
        sequence_number: CheckpointSequenceNumber,
    ) -> Option<types::checkpoints::CheckpointContents> {
        self.rocks.get_checkpoint_contents_by_sequence_number(sequence_number)
    }

    fn get_transaction(&self, digest: &TransactionDigest) -> Option<Arc<VerifiedTransaction>> {
        self.rocks.get_transaction(digest)
    }

    fn get_transaction_effects(&self, digest: &TransactionDigest) -> Option<TransactionEffects> {
        self.rocks.get_transaction_effects(digest)
    }

    fn get_full_checkpoint_contents(
        &self,
        sequence_number: Option<CheckpointSequenceNumber>,
        digest: &CheckpointContentsDigest,
    ) -> Option<FullCheckpointContents> {
        self.rocks.get_full_checkpoint_contents(sequence_number, digest)
    }
}

impl RpcStateReader for RestReadStore {
    fn get_lowest_available_checkpoint_objects(
        &self,
    ) -> types::storage::storage_error::Result<CheckpointSequenceNumber> {
        Ok(self
            .state
            .get_object_cache_reader()
            .get_highest_pruned_checkpoint()
            .map(|cp| cp + 1)
            .unwrap_or(0))
    }

    fn get_chain_identifier(&self) -> Result<types::digests::ChainIdentifier> {
        Ok(self.state.get_chain_identifier())
    }

    fn indexes(&self) -> Option<&dyn RpcIndexes> {
        Some(self)
    }
}

impl RpcIndexes for RestReadStore {
    fn get_epoch_info(
        &self,
        epoch: EpochId,
    ) -> Result<Option<types::storage::read_store::EpochInfo>> {
        self.index()?.get_epoch_info(epoch).map_err(StorageError::custom)
    }

    fn get_transaction_info(
        &self,
        digest: &TransactionDigest,
    ) -> types::storage::storage_error::Result<Option<types::storage::read_store::TransactionInfo>>
    {
        self.index()?.get_transaction_info(digest).map_err(StorageError::custom)
    }

    fn owned_objects_iter(
        &self,
        owner: SomaAddress,
        object_type: Option<ObjectType>,
        cursor: Option<OwnedObjectInfo>,
    ) -> Result<
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

        let iter = self.index()?.owner_iter(owner, object_type, cursor)?.map(|result| {
            result
                .map(
                    |(
                        OwnerIndexKey { owner, object_id, object_type, inverted_balance },
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
                .map_err(Into::into)
        });

        Ok(Box::new(iter) as _)
    }

    fn get_balance(
        &self,
        owner: &SomaAddress,
    ) -> types::storage::storage_error::Result<Option<BalanceInfo>> {
        self.index()?.get_balance(owner)?.map(|info| info.into()).pipe(Ok)
    }

    fn get_highest_indexed_checkpoint_seq_number(
        &self,
    ) -> types::storage::storage_error::Result<Option<CheckpointSequenceNumber>> {
        self.index()?.get_highest_indexed_checkpoint_seq_number().map_err(Into::into)
    }

    fn targets_iter(
        &self,
        status_filter: Option<String>,
        epoch_filter: Option<u64>,
        cursor: Option<TargetInfo>,
    ) -> Result<
        Box<dyn Iterator<Item = Result<TargetInfo, types::storage::storage_error::Error>> + '_>,
    > {
        let iter = self
            .index()?
            .targets_iter(status_filter, epoch_filter, cursor)?
            .map(|r| r.map_err(Into::into));
        Ok(Box::new(iter))
    }

    fn challenges_iter(
        &self,
        status_filter: Option<String>,
        epoch_filter: Option<u64>,
        target_filter: Option<ObjectID>,
        cursor: Option<ChallengeInfo>,
    ) -> Result<
        Box<dyn Iterator<Item = Result<ChallengeInfo, types::storage::storage_error::Error>> + '_>,
    > {
        let iter = self
            .index()?
            .challenges_iter(status_filter, epoch_filter, target_filter, cursor)?
            .map(|r| r.map_err(Into::into));
        Ok(Box::new(iter))
    }
}
