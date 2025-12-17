use std::{
    collections::{BTreeSet, HashMap, HashSet},
    sync::Arc,
};

use serde::{Deserialize, Serialize};
use store::TypedStoreError;

use crate::{
    balance_change::{derive_balance_changes, BalanceChange},
    base::SomaAddress,
    checkpoints::{
        CheckpointContents, CheckpointSequenceNumber, FullCheckpointContents, VerifiedCheckpoint,
    },
    committee::{Committee, EpochId},
    digests::{ChainIdentifier, CheckpointContentsDigest, CheckpointDigest, TransactionDigest},
    effects::TransactionEffects,
    full_checkpoint_content::{Checkpoint, ExecutedTransaction, ObjectSet},
    object::{Object, ObjectID, ObjectRef, ObjectType, Version},
    shard_crypto::keys::EncoderPublicKey,
    storage::ObjectKey,
    system_state::shard::{Shard, Target, TargetOrigin},
    transaction::VerifiedTransaction,
};

use super::{object_store::ObjectStore, storage_error::Result};

pub trait ReadStore: ObjectStore {
    //
    // Committee Getters
    //

    fn get_committee(&self, epoch: EpochId) -> Option<Arc<Committee>>;

    //
    // Checkpoint Getters
    //

    /// Get the latest available checkpoint. This is the latest executed checkpoint.
    ///
    /// All transactions, effects, objects and events are guaranteed to be available for the
    /// returned checkpoint.
    fn get_latest_checkpoint(&self) -> Result<VerifiedCheckpoint>;

    /// Get the latest available checkpoint sequence number. This is the sequence number of the latest executed checkpoint.
    fn get_latest_checkpoint_sequence_number(&self) -> Result<CheckpointSequenceNumber> {
        let latest_checkpoint = self.get_latest_checkpoint()?;
        Ok(*latest_checkpoint.sequence_number())
    }

    /// Get the epoch of the latest checkpoint
    fn get_latest_epoch_id(&self) -> Result<EpochId> {
        let latest_checkpoint = self.get_latest_checkpoint()?;
        Ok(latest_checkpoint.epoch())
    }

    /// Get the highest verified checkpint. This is the highest checkpoint summary that has been
    /// verified, generally by state-sync. Only the checkpoint header is guaranteed to be present in
    /// the store.
    fn get_highest_verified_checkpoint(&self) -> Result<VerifiedCheckpoint>;

    /// Get the highest synced checkpint. This is the highest checkpoint that has been synced from
    /// state-synce. The checkpoint header, contents, transactions, and effects of this checkpoint
    /// are guaranteed to be present in the store
    fn get_highest_synced_checkpoint(&self) -> Result<VerifiedCheckpoint>;

    /// Lowest available checkpoint for which transaction and checkpoint data can be requested.
    ///
    /// Specifically this is the lowest checkpoint for which the following data can be requested:
    ///  - checkpoints
    ///  - transactions
    ///  - effects
    ///
    /// For object availability see `get_lowest_available_checkpoint_objects`.
    fn get_lowest_available_checkpoint(&self) -> Result<CheckpointSequenceNumber>;

    fn get_checkpoint_by_digest(&self, digest: &CheckpointDigest) -> Option<VerifiedCheckpoint>;

    fn get_checkpoint_by_sequence_number(
        &self,
        sequence_number: CheckpointSequenceNumber,
    ) -> Option<VerifiedCheckpoint>;

    fn get_checkpoint_contents_by_digest(
        &self,
        digest: &CheckpointContentsDigest,
    ) -> Option<CheckpointContents>;

    fn get_checkpoint_contents_by_sequence_number(
        &self,
        sequence_number: CheckpointSequenceNumber,
    ) -> Option<CheckpointContents>;

    //
    // Transaction Getters
    //

    fn get_transaction(&self, tx_digest: &TransactionDigest) -> Option<Arc<VerifiedTransaction>>;

    fn multi_get_transactions(
        &self,
        tx_digests: &[TransactionDigest],
    ) -> Vec<Option<Arc<VerifiedTransaction>>> {
        tx_digests
            .iter()
            .map(|digest| self.get_transaction(digest))
            .collect()
    }

    fn get_transaction_effects(&self, tx_digest: &TransactionDigest) -> Option<TransactionEffects>;

    fn multi_get_transaction_effects(
        &self,
        tx_digests: &[TransactionDigest],
    ) -> Vec<Option<TransactionEffects>> {
        tx_digests
            .iter()
            .map(|digest| self.get_transaction_effects(digest))
            .collect()
    }

    //
    // Extra Checkpoint fetching apis
    //

    /// Get a "full" checkpoint for purposes of state-sync
    /// "full" checkpoints include: header, contents, transactions, effects.
    /// sequence_number is optional since we can always query it using the digest.
    /// However if it is provided, we can avoid an extra db lookup.
    fn get_full_checkpoint_contents(
        &self,
        sequence_number: Option<CheckpointSequenceNumber>,
        digest: &CheckpointContentsDigest,
    ) -> Option<FullCheckpointContents>;

    // Fetch all checkpoint data
    fn get_checkpoint_data(
        &self,
        checkpoint: VerifiedCheckpoint,
        checkpoint_contents: CheckpointContents,
    ) -> Result<Checkpoint> {
        use crate::effects::TransactionEffectsAPI;
        use crate::storage::storage_error::Error;
        use std::collections::HashMap;

        let transaction_digests = checkpoint_contents
            .iter()
            .map(|execution_digests| execution_digests.transaction)
            .collect::<Vec<_>>();
        let txns = self
            .multi_get_transactions(&transaction_digests)
            .into_iter()
            .map(|maybe_transaction| {
                maybe_transaction.ok_or_else(|| Error::missing("missing transaction"))
            })
            .collect::<Result<Vec<_>>>()?;

        let effects = self
            .multi_get_transaction_effects(&transaction_digests)
            .into_iter()
            .map(|maybe_effects| maybe_effects.ok_or_else(|| Error::missing("missing effects")))
            .collect::<Result<Vec<_>>>()?;

        let mut transactions = Vec::with_capacity(txns.len());
        for (tx, fx) in txns.into_iter().zip(effects) {
            let transaction = ExecutedTransaction {
                transaction: tx.transaction_data().clone(),
                signatures: tx.tx_signatures().to_vec(),
                effects: fx.clone(),
            };
            transactions.push(transaction);
        }

        let object_set = {
            let refs = transactions
                .iter()
                .flat_map(|tx| {
                    crate::storage::get_transaction_object_set(&tx.transaction, &tx.effects)
                })
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect::<Vec<_>>();

            let objects = self.multi_get_objects_by_key(&refs);

            let mut object_set = ObjectSet::default();
            for (idx, object) in objects.into_iter().enumerate() {
                object_set.insert(object.ok_or_else(|| {
                    Error::missing(format!("unable to load object {:?}", refs[idx]))
                })?);
            }
            object_set
        };

        let checkpoint_data = Checkpoint {
            summary: checkpoint.into(),
            contents: checkpoint_contents,
            transactions,
            object_set,
        };

        Ok(checkpoint_data)
    }
}
impl<T: ReadStore + ?Sized> ReadStore for &T {
    fn get_committee(&self, epoch: EpochId) -> Option<Arc<Committee>> {
        (*self).get_committee(epoch)
    }

    fn get_latest_checkpoint(&self) -> Result<VerifiedCheckpoint> {
        (*self).get_latest_checkpoint()
    }

    fn get_latest_checkpoint_sequence_number(&self) -> Result<CheckpointSequenceNumber> {
        (*self).get_latest_checkpoint_sequence_number()
    }

    fn get_latest_epoch_id(&self) -> Result<EpochId> {
        (*self).get_latest_epoch_id()
    }

    fn get_highest_verified_checkpoint(&self) -> Result<VerifiedCheckpoint> {
        (*self).get_highest_verified_checkpoint()
    }

    fn get_highest_synced_checkpoint(&self) -> Result<VerifiedCheckpoint> {
        (*self).get_highest_synced_checkpoint()
    }

    fn get_lowest_available_checkpoint(&self) -> Result<CheckpointSequenceNumber> {
        (*self).get_lowest_available_checkpoint()
    }

    fn get_checkpoint_by_digest(&self, digest: &CheckpointDigest) -> Option<VerifiedCheckpoint> {
        (*self).get_checkpoint_by_digest(digest)
    }

    fn get_checkpoint_by_sequence_number(
        &self,
        sequence_number: CheckpointSequenceNumber,
    ) -> Option<VerifiedCheckpoint> {
        (*self).get_checkpoint_by_sequence_number(sequence_number)
    }

    fn get_checkpoint_contents_by_digest(
        &self,
        digest: &CheckpointContentsDigest,
    ) -> Option<CheckpointContents> {
        (*self).get_checkpoint_contents_by_digest(digest)
    }

    fn get_checkpoint_contents_by_sequence_number(
        &self,
        sequence_number: CheckpointSequenceNumber,
    ) -> Option<CheckpointContents> {
        (*self).get_checkpoint_contents_by_sequence_number(sequence_number)
    }

    fn get_transaction(&self, tx_digest: &TransactionDigest) -> Option<Arc<VerifiedTransaction>> {
        (*self).get_transaction(tx_digest)
    }

    fn multi_get_transactions(
        &self,
        tx_digests: &[TransactionDigest],
    ) -> Vec<Option<Arc<VerifiedTransaction>>> {
        (*self).multi_get_transactions(tx_digests)
    }

    fn get_transaction_effects(&self, tx_digest: &TransactionDigest) -> Option<TransactionEffects> {
        (*self).get_transaction_effects(tx_digest)
    }

    fn multi_get_transaction_effects(
        &self,
        tx_digests: &[TransactionDigest],
    ) -> Vec<Option<TransactionEffects>> {
        (*self).multi_get_transaction_effects(tx_digests)
    }

    fn get_full_checkpoint_contents(
        &self,
        sequence_number: Option<CheckpointSequenceNumber>,
        digest: &CheckpointContentsDigest,
    ) -> Option<FullCheckpointContents> {
        (*self).get_full_checkpoint_contents(sequence_number, digest)
    }

    fn get_checkpoint_data(
        &self,
        checkpoint: VerifiedCheckpoint,
        checkpoint_contents: CheckpointContents,
    ) -> Result<Checkpoint> {
        (*self).get_checkpoint_data(checkpoint, checkpoint_contents)
    }
}

impl<T: ReadStore + ?Sized> ReadStore for Box<T> {
    fn get_committee(&self, epoch: EpochId) -> Option<Arc<Committee>> {
        (**self).get_committee(epoch)
    }

    fn get_latest_checkpoint(&self) -> Result<VerifiedCheckpoint> {
        (**self).get_latest_checkpoint()
    }

    fn get_latest_checkpoint_sequence_number(&self) -> Result<CheckpointSequenceNumber> {
        (**self).get_latest_checkpoint_sequence_number()
    }

    fn get_latest_epoch_id(&self) -> Result<EpochId> {
        (**self).get_latest_epoch_id()
    }

    fn get_highest_verified_checkpoint(&self) -> Result<VerifiedCheckpoint> {
        (**self).get_highest_verified_checkpoint()
    }

    fn get_highest_synced_checkpoint(&self) -> Result<VerifiedCheckpoint> {
        (**self).get_highest_synced_checkpoint()
    }

    fn get_lowest_available_checkpoint(&self) -> Result<CheckpointSequenceNumber> {
        (**self).get_lowest_available_checkpoint()
    }

    fn get_checkpoint_by_digest(&self, digest: &CheckpointDigest) -> Option<VerifiedCheckpoint> {
        (**self).get_checkpoint_by_digest(digest)
    }

    fn get_checkpoint_by_sequence_number(
        &self,
        sequence_number: CheckpointSequenceNumber,
    ) -> Option<VerifiedCheckpoint> {
        (**self).get_checkpoint_by_sequence_number(sequence_number)
    }

    fn get_checkpoint_contents_by_digest(
        &self,
        digest: &CheckpointContentsDigest,
    ) -> Option<CheckpointContents> {
        (**self).get_checkpoint_contents_by_digest(digest)
    }

    fn get_checkpoint_contents_by_sequence_number(
        &self,
        sequence_number: CheckpointSequenceNumber,
    ) -> Option<CheckpointContents> {
        (**self).get_checkpoint_contents_by_sequence_number(sequence_number)
    }

    fn get_transaction(&self, tx_digest: &TransactionDigest) -> Option<Arc<VerifiedTransaction>> {
        (**self).get_transaction(tx_digest)
    }

    fn multi_get_transactions(
        &self,
        tx_digests: &[TransactionDigest],
    ) -> Vec<Option<Arc<VerifiedTransaction>>> {
        (**self).multi_get_transactions(tx_digests)
    }

    fn get_transaction_effects(&self, tx_digest: &TransactionDigest) -> Option<TransactionEffects> {
        (**self).get_transaction_effects(tx_digest)
    }

    fn multi_get_transaction_effects(
        &self,
        tx_digests: &[TransactionDigest],
    ) -> Vec<Option<TransactionEffects>> {
        (**self).multi_get_transaction_effects(tx_digests)
    }

    fn get_full_checkpoint_contents(
        &self,
        sequence_number: Option<CheckpointSequenceNumber>,
        digest: &CheckpointContentsDigest,
    ) -> Option<FullCheckpointContents> {
        (**self).get_full_checkpoint_contents(sequence_number, digest)
    }

    fn get_checkpoint_data(
        &self,
        checkpoint: VerifiedCheckpoint,
        checkpoint_contents: CheckpointContents,
    ) -> Result<Checkpoint> {
        (**self).get_checkpoint_data(checkpoint, checkpoint_contents)
    }
}

impl<T: ReadStore + ?Sized> ReadStore for Arc<T> {
    fn get_committee(&self, epoch: EpochId) -> Option<Arc<Committee>> {
        (**self).get_committee(epoch)
    }

    fn get_latest_checkpoint(&self) -> Result<VerifiedCheckpoint> {
        (**self).get_latest_checkpoint()
    }

    fn get_latest_checkpoint_sequence_number(&self) -> Result<CheckpointSequenceNumber> {
        (**self).get_latest_checkpoint_sequence_number()
    }

    fn get_latest_epoch_id(&self) -> Result<EpochId> {
        (**self).get_latest_epoch_id()
    }

    fn get_highest_verified_checkpoint(&self) -> Result<VerifiedCheckpoint> {
        (**self).get_highest_verified_checkpoint()
    }

    fn get_highest_synced_checkpoint(&self) -> Result<VerifiedCheckpoint> {
        (**self).get_highest_synced_checkpoint()
    }

    fn get_lowest_available_checkpoint(&self) -> Result<CheckpointSequenceNumber> {
        (**self).get_lowest_available_checkpoint()
    }

    fn get_checkpoint_by_digest(&self, digest: &CheckpointDigest) -> Option<VerifiedCheckpoint> {
        (**self).get_checkpoint_by_digest(digest)
    }

    fn get_checkpoint_by_sequence_number(
        &self,
        sequence_number: CheckpointSequenceNumber,
    ) -> Option<VerifiedCheckpoint> {
        (**self).get_checkpoint_by_sequence_number(sequence_number)
    }

    fn get_checkpoint_contents_by_digest(
        &self,
        digest: &CheckpointContentsDigest,
    ) -> Option<CheckpointContents> {
        (**self).get_checkpoint_contents_by_digest(digest)
    }

    fn get_checkpoint_contents_by_sequence_number(
        &self,
        sequence_number: CheckpointSequenceNumber,
    ) -> Option<CheckpointContents> {
        (**self).get_checkpoint_contents_by_sequence_number(sequence_number)
    }

    fn get_transaction(&self, tx_digest: &TransactionDigest) -> Option<Arc<VerifiedTransaction>> {
        (**self).get_transaction(tx_digest)
    }

    fn multi_get_transactions(
        &self,
        tx_digests: &[TransactionDigest],
    ) -> Vec<Option<Arc<VerifiedTransaction>>> {
        (**self).multi_get_transactions(tx_digests)
    }

    fn get_transaction_effects(&self, tx_digest: &TransactionDigest) -> Option<TransactionEffects> {
        (**self).get_transaction_effects(tx_digest)
    }

    fn multi_get_transaction_effects(
        &self,
        tx_digests: &[TransactionDigest],
    ) -> Vec<Option<TransactionEffects>> {
        (**self).multi_get_transaction_effects(tx_digests)
    }

    fn get_full_checkpoint_contents(
        &self,
        sequence_number: Option<CheckpointSequenceNumber>,
        digest: &CheckpointContentsDigest,
    ) -> Option<FullCheckpointContents> {
        (**self).get_full_checkpoint_contents(sequence_number, digest)
    }

    fn get_checkpoint_data(
        &self,
        checkpoint: VerifiedCheckpoint,
        checkpoint_contents: CheckpointContents,
    ) -> Result<Checkpoint> {
        (**self).get_checkpoint_data(checkpoint, checkpoint_contents)
    }
}

/// Trait used to provide functionality to the REST API service.
///
/// It extends both ObjectStore and ReadStore by adding functionality that may require more
/// detailed underlying databases or indexes to support.
pub trait RpcStateReader: ObjectStore + ReadStore + Send + Sync {
    /// Lowest available checkpoint for which object data can be requested.
    ///
    /// Specifically this is the lowest checkpoint for which input/output object data will be
    /// available.
    fn get_lowest_available_checkpoint_objects(&self) -> Result<CheckpointSequenceNumber>;

    fn get_chain_identifier(&self) -> Result<ChainIdentifier>;

    // Get a handle to an instance of the RpcIndexes
    fn indexes(&self) -> Option<&dyn RpcIndexes>;
}

pub trait RpcIndexes: Send + Sync {
    fn get_epoch_info(&self, epoch: EpochId) -> Result<Option<EpochInfo>>;

    fn get_transaction_info(&self, digest: &TransactionDigest) -> Result<Option<TransactionInfo>>;

    fn owned_objects_iter(
        &self,
        owner: SomaAddress,
        object_type: Option<ObjectType>,
        cursor: Option<OwnedObjectInfo>,
    ) -> Result<Box<dyn Iterator<Item = Result<OwnedObjectInfo, TypedStoreError>> + '_>>;

    fn get_balance(&self, owner: &SomaAddress) -> Result<Option<BalanceInfo>>;

    fn get_highest_indexed_checkpoint_seq_number(&self)
        -> Result<Option<CheckpointSequenceNumber>>;

    // =========================================================================
    // SHARD QUERY METHODS
    // =========================================================================

    /// Get all shards created in a specific epoch
    fn get_shards_by_epoch(&self, epoch: EpochId) -> Result<Vec<ShardIndexInfo>>;

    /// Get shards submitted by a specific address, optionally filtered by epoch
    fn get_shards_by_submitter(
        &self,
        submitter: SomaAddress,
        epoch: Option<EpochId>,
    ) -> Result<Vec<ShardIndexInfo>>;

    /// Get shards won by a specific encoder
    fn get_shards_by_encoder(&self, encoder: &EncoderPublicKey) -> Result<Vec<ShardIndexInfo>>;

    /// Get claimable escrows (shards where created_epoch + 2 <= current_epoch)
    fn get_claimable_escrows(&self, current_epoch: EpochId) -> Result<Vec<ShardIndexInfo>>;

    // =========================================================================
    // TARGET QUERY METHODS
    // =========================================================================

    /// Get all targets valid for competition in the given epoch
    fn get_valid_targets(&self, epoch: EpochId) -> Result<Vec<TargetIndexInfo>>;

    /// Get claimable rewards (targets where created_epoch + 2 <= current_epoch)
    fn get_claimable_rewards(&self, current_epoch: EpochId) -> Result<Vec<TargetIndexInfo>>;
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct OwnedObjectInfo {
    pub owner: SomaAddress,
    pub object_type: ObjectType,
    pub balance: Option<u64>,
    pub object_id: ObjectID,
    pub version: Version,
}

#[derive(Clone, Serialize, Deserialize, Eq, PartialEq, Debug)]
pub struct TransactionInfo {
    pub checkpoint: u64,
    pub balance_changes: Vec<BalanceChange>,
    pub object_types: HashMap<ObjectID, ObjectType>,
}

impl TransactionInfo {
    pub fn new(
        effects: &TransactionEffects,
        input_objects: &[Object],
        output_objects: &[Object],
        checkpoint: u64,
    ) -> TransactionInfo {
        let balance_changes = derive_balance_changes(effects, input_objects, output_objects);

        let object_types = input_objects
            .iter()
            .chain(output_objects)
            .map(|object| (object.id(), ObjectType::from(object)))
            .collect();

        TransactionInfo {
            checkpoint,
            balance_changes,
            object_types,
        }
    }
}

#[derive(Clone, Default, Serialize, Deserialize, Eq, PartialEq, Debug)]
pub struct EpochInfo {
    pub epoch: u64,
    pub protocol_version: Option<u64>,
    pub start_timestamp_ms: Option<u64>,
    pub end_timestamp_ms: Option<u64>,
    pub start_checkpoint: Option<u64>,
    pub end_checkpoint: Option<u64>,
    // TODO: pub reference_byte_price: Option<u64>,
    pub system_state: Option<crate::system_state::SystemState>,
}

#[derive(Default, Copy, Clone, Debug, Eq, PartialEq)]
pub struct BalanceInfo {
    pub balance: u64,
}

/// Cached info about a shard
#[derive(Clone, Serialize, Deserialize, PartialEq, Eq, Debug)]
pub struct ShardIndexInfo {
    pub shard_id: ObjectID,
    pub created_epoch: EpochId,
    pub amount: u64,
    pub data_submitter: SomaAddress,
    pub target: Option<ObjectRef>,
    pub has_winner: bool,
    pub winning_encoder: Option<EncoderPublicKey>,
}

impl ShardIndexInfo {
    pub fn from_shard(shard_id: ObjectID, shard: &Shard) -> Self {
        Self {
            shard_id,
            created_epoch: shard.created_epoch,
            amount: shard.amount,
            data_submitter: shard.data_submitter,
            target: shard.target.clone(),
            has_winner: shard.winning_encoder.is_some(),
            winning_encoder: shard.winning_encoder.clone(),
        }
    }

    /// Check if escrow can be claimed (current_epoch >= created_epoch + 2)
    pub fn is_escrow_claimable(&self, current_epoch: EpochId) -> bool {
        current_epoch >= self.created_epoch + 2
    }
}

/// Cached info about a target
#[derive(Clone, Serialize, Deserialize, PartialEq, Eq, Debug)]
pub struct TargetIndexInfo {
    pub target_id: ObjectID,
    pub created_epoch: EpochId,
    pub valid_epoch: EpochId,
    pub origin: TargetOriginType,
    pub creator: Option<SomaAddress>,
    pub reward_amount: Option<u64>,
    pub has_winner: bool,
}

#[derive(Clone, Serialize, Deserialize, PartialEq, Eq, Debug)]
pub enum TargetOriginType {
    System,
    User,
    Genesis,
}

impl TargetIndexInfo {
    pub fn from_target(target_id: ObjectID, target: &Target) -> Self {
        let (origin, creator, reward_amount, valid_epoch) = match &target.origin {
            TargetOrigin::System => (
                TargetOriginType::System,
                None,
                None,
                target.created_epoch + 1,
            ),
            TargetOrigin::User {
                creator,
                reward_amount,
            } => (
                TargetOriginType::User,
                Some(*creator),
                Some(*reward_amount),
                target.created_epoch + 1,
            ),
            TargetOrigin::Genesis { reward_amount } => (
                TargetOriginType::Genesis,
                None,
                Some(*reward_amount),
                target.created_epoch,
            ),
        };

        Self {
            target_id,
            created_epoch: target.created_epoch,
            valid_epoch,
            origin,
            creator,
            reward_amount,
            has_winner: target.winning_shard.is_some(),
        }
    }

    /// Check if reward can be claimed (current_epoch >= created_epoch + 2)
    pub fn is_reward_claimable(&self, current_epoch: EpochId) -> bool {
        current_epoch >= self.created_epoch + 2
    }
}
