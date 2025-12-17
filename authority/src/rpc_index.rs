use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::authority_store::AuthorityStore;
use crate::checkpoints::CheckpointStore;
use serde::{Deserialize, Serialize};
use std::sync::Mutex;
use store::rocks::{DBMap, DBMapTableConfigMap};
use store::rocksdb::{compaction_filter::Decision, MergeOperands, WriteOptions};
use store::{DBMapUtils, Map as _, TypedStoreError};
use sysinfo::{MemoryRefreshKind, RefreshKind, System};
use tracing::{debug, info};
use types::checkpoints::{CheckpointContents, CheckpointSequenceNumber};
use types::effects::{TransactionEffects, TransactionEffectsAPI as _};
use types::full_checkpoint_content::{CheckpointData, CheckpointTransaction};
use types::shard_crypto::keys::EncoderPublicKey;
use types::storage::WriteKind;
use types::system_state::shard::{Shard, Target, TargetOrigin};
use types::transaction_outputs::WrittenObjects;

use types::committee::EpochId;
use types::consensus::ConsensusTransactionKind;
use types::digests::TransactionDigest;
use types::object::{LiveObject, ObjectRef};
use types::storage::read_store::{EpochInfo, ShardIndexInfo, TargetIndexInfo, TransactionInfo};
use types::storage::storage_error::Error as StorageError;
use types::system_state::SystemStateTrait;
use types::{
    base::SomaAddress,
    object::{Object, ObjectID, ObjectType, Owner, Version},
};

const CURRENT_DB_VERSION: u64 = 1;
const BALANCE_FLUSH_THRESHOLD: usize = 10_000;

fn bulk_ingestion_write_options() -> WriteOptions {
    let mut opts = WriteOptions::default();
    opts.disable_wal(true);
    opts
}

/// Get available memory, respecting cgroup limits in containerized environments
fn get_available_memory() -> u64 {
    // RefreshKind::nothing().with_memory() avoids collecting other, slower stats
    let mut sys = System::new_with_specifics(
        RefreshKind::nothing().with_memory(MemoryRefreshKind::everything()),
    );
    sys.refresh_memory();

    // Check if we have cgroup limits
    if let Some(cgroup_limits) = sys.cgroup_limits() {
        let memory_limit = cgroup_limits.total_memory;
        // cgroup_limits.total_memory is 0 when there's no limit
        if memory_limit > 0 {
            debug!("Using cgroup memory limit: {} bytes", memory_limit);
            return memory_limit;
        }
    }

    // Fall back to system memory if no cgroup limits found
    // sysinfo 0.35 already reports bytes (not KiB like older versions)
    let total_memory_bytes = sys.total_memory();
    debug!("Using system memory: {} bytes", total_memory_bytes);
    total_memory_bytes
}

fn default_table_options() -> store::rocks::DBOptions {
    store::rocks::default_db_options().disable_write_throttling()
}

fn balance_delta_merge_operator(
    _key: &[u8],
    existing_val: Option<&[u8]>,
    operands: &MergeOperands,
) -> Option<Vec<u8>> {
    let mut result = existing_val
        .map(|v| {
            bcs::from_bytes::<BalanceIndexInfo>(v)
                .expect("Failed to deserialize BalanceIndexInfo from RocksDB - data corruption.")
        })
        .unwrap_or_default();

    for operand in operands.iter() {
        let delta = bcs::from_bytes::<BalanceIndexInfo>(operand)
            .expect("Failed to deserialize BalanceIndexInfo from RocksDB - data corruption.");
        result.merge_delta(&delta);
    }
    Some(
        bcs::to_bytes(&result)
            .expect("Failed to deserialize BalanceIndexInfo from RocksDB - data corruption."),
    )
}

fn balance_compaction_filter(_level: u32, _key: &[u8], value: &[u8]) -> Decision {
    let balance_info = match bcs::from_bytes::<BalanceIndexInfo>(value) {
        Ok(info) => info,
        Err(_) => return Decision::Keep,
    };

    if balance_info.balance_delta == 0 {
        Decision::Remove
    } else {
        Decision::Keep
    }
}

fn balance_table_options() -> store::rocks::DBOptions {
    default_table_options()
        .set_merge_operator_associative("balance_merge", balance_delta_merge_operator)
        .set_compaction_filter("balance_zero_filter", balance_compaction_filter)
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
struct MetadataInfo {
    /// Version of the Database
    version: u64,
}

/// Checkpoint watermark type
#[derive(Copy, Clone, Debug, Serialize, Deserialize, PartialOrd, Ord, PartialEq, Eq)]
pub enum Watermark {
    Indexed,
    Pruned,
}

#[derive(Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct OwnerIndexKey {
    pub owner: SomaAddress,

    pub object_type: ObjectType,

    pub inverted_balance: Option<u64>,

    pub object_id: ObjectID,
}

impl OwnerIndexKey {
    // Creates a key from the provided object.
    // Panics if the provided object is not an Address owned object
    fn from_object(object: &Object) -> Self {
        let owner = match object.owner() {
            Owner::AddressOwner(owner) => owner,

            _ => panic!("cannot create OwnerIndexKey if object is not address-owned"),
        };
        let object_type = object.type_().clone();

        let inverted_balance = object.as_coin();

        Self {
            owner: *owner,
            object_type,
            inverted_balance,
            object_id: object.id(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct OwnerIndexInfo {
    // object_id and type of this object are a part of the key
    pub version: Version,
}

impl OwnerIndexInfo {
    pub fn new(object: &Object) -> Self {
        Self {
            version: object.version(),
        }
    }
}

#[derive(Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct BalanceKey {
    pub owner: SomaAddress,
}

#[derive(Clone, Serialize, Deserialize, PartialEq, Eq, Debug, Default)]
pub struct BalanceIndexInfo {
    pub balance_delta: i128,
}

impl From<u64> for BalanceIndexInfo {
    fn from(coin_value: u64) -> Self {
        Self {
            balance_delta: coin_value as i128,
        }
    }
}

impl BalanceIndexInfo {
    fn invert(self) -> Self {
        // Check for potential overflow when negating i128::MIN
        assert!(
            self.balance_delta != i128::MIN,
            "Cannot invert balance_delta: would overflow i128"
        );

        Self {
            balance_delta: -self.balance_delta,
        }
    }

    fn merge_delta(&mut self, other: &Self) {
        self.balance_delta += other.balance_delta;
    }
}

impl From<BalanceIndexInfo> for types::storage::read_store::BalanceInfo {
    fn from(index_info: BalanceIndexInfo) -> Self {
        // Note: We represent balance deltas as i128 to simplify merging positive and negative updates.
        // Be aware: Move doesn’t enforce a one-time-witness (OTW) pattern when creating a Supply<T>.
        // Anyone can call `sui::balance::create_supply` and mint unbounded supply, potentially pushing
        // total balances over u64::MAX. To avoid crashing the indexer, we clamp the merged value instead
        // of panicking on overflow. This has the unfortunate consequence of making bugs in the index
        // harder to detect, but is a necessary trade-off to avoid creating a DOS attack vector.
        let balance = index_info.balance_delta.clamp(0, u64::MAX as i128) as u64;
        types::storage::read_store::BalanceInfo { balance }
    }
}

/// Key for indexing shards by their creation epoch
#[derive(Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct ShardEpochKey {
    pub epoch: EpochId,
    pub shard_id: ObjectID,
}

/// Key for indexing shards by data submitter
#[derive(Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct ShardSubmitterKey {
    pub submitter: SomaAddress,
    pub epoch: EpochId,
    pub shard_id: ObjectID,
}

/// Key for indexing shards by winning encoder
#[derive(Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct ShardEncoderKey {
    pub encoder: EncoderPublicKey,
    pub epoch: EpochId,
    pub shard_id: ObjectID,
}

// =============================================================================
// TARGET INDEX TYPES
// =============================================================================

/// Key for indexing targets by their valid epoch
#[derive(Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct TargetValidEpochKey {
    pub valid_epoch: EpochId,
    pub target_id: ObjectID,
}

/// RocksDB tables for the RpcIndexStore
///
/// Anytime a new table is added, or and existing one has it's schema changed, make sure to also
/// update the value of `CURRENT_DB_VERSION`.
///
/// NOTE: Authors and Reviewers before adding any new tables ensure that they are either:
/// - bounded in size by the live object set
/// - are prune-able and have corresponding logic in the `prune` function
#[derive(DBMapUtils)]
struct IndexStoreTables {
    /// A singleton that store metadata information on the DB.
    ///
    /// A few uses for this singleton:
    /// - determining if the DB has been initialized (as some tables will still be empty post
    ///     initialization)
    /// - version of the DB. Everytime a new table or schema is changed the version number needs to
    ///     be incremented.
    meta: DBMap<(), MetadataInfo>,

    /// Table used to track watermark for the highest indexed checkpoint
    ///
    /// This is useful to help know the highest checkpoint that was indexed in the event that the
    /// node was running with indexes enabled, then run for a period of time with indexes disabled,
    /// and then run with them enabled again so that the tables can be reinitialized.
    watermark: DBMap<Watermark, CheckpointSequenceNumber>,

    /// An index of extra metadata for Epochs.
    ///
    /// Only contains entries for transactions which have yet to be pruned from the main database.
    epochs: DBMap<EpochId, EpochInfo>,

    /// An index of extra metadata for Transactions.
    ///
    /// Only contains entries for transactions which have yet to be pruned from the main database.
    transactions: DBMap<TransactionDigest, TransactionInfo>,

    /// An index of object ownership.
    ///
    /// Allows an efficient iterator to list all objects currently owned by a specific user
    /// account.
    owner: DBMap<OwnerIndexKey, OwnerIndexInfo>,

    /// An index of Balances.
    ///
    /// Allows looking up balances by owner address and coin type.
    balance: DBMap<BalanceKey, BalanceIndexInfo>,

    /// Shards indexed by creation epoch
    shard_by_epoch: DBMap<ShardEpochKey, ShardIndexInfo>,

    /// Shards indexed by data submitter
    shard_by_submitter: DBMap<ShardSubmitterKey, ShardIndexInfo>,

    /// Shards indexed by winning encoder
    shard_by_encoder: DBMap<ShardEncoderKey, ShardIndexInfo>,

    // ----- Target Tables -----
    /// Targets indexed by their valid epoch
    target_by_valid_epoch: DBMap<TargetValidEpochKey, TargetIndexInfo>,
}

impl IndexStoreTables {
    fn track_coin_balance_change(
        object: &Object,
        owner: &SomaAddress,
        is_removal: bool,
        balance_changes: &mut HashMap<BalanceKey, BalanceIndexInfo>,
    ) -> Result<(), StorageError> {
        if let Some(value) = object.as_coin() {
            let key = BalanceKey { owner: *owner };

            let mut delta = BalanceIndexInfo::from(value);
            if is_removal {
                delta = delta.invert();
            }

            balance_changes.entry(key).or_default().merge_delta(&delta);
        }
        Ok(())
    }

    fn open_with_index_options<P: Into<PathBuf>>(
        path: P,
        // index_options: IndexStoreOptions,
    ) -> Self {
        let mut table_options = std::collections::BTreeMap::new();
        table_options.insert("balance".to_string(), balance_table_options());

        IndexStoreTables::open_tables_read_write(
            path.into(),
            None,
            Some(DBMapTableConfigMap::new(table_options)),
        )
    }

    fn open_with_options<P: Into<PathBuf>>(
        path: P,
        options: store::rocksdb::Options,
        table_options: Option<DBMapTableConfigMap>,
    ) -> Self {
        IndexStoreTables::open_tables_read_write(path.into(), Some(options), table_options)
    }

    fn needs_to_do_initialization(&self, checkpoint_store: &CheckpointStore) -> bool {
        (match self.meta.get(&()) {
            Ok(Some(metadata)) => metadata.version != CURRENT_DB_VERSION,
            Ok(None) => true,
            Err(_) => true,
        }) || self.is_indexed_watermark_out_of_date(checkpoint_store)
    }

    // Check if the index watermark is behind the highets_executed watermark.
    fn is_indexed_watermark_out_of_date(&self, checkpoint_store: &CheckpointStore) -> bool {
        let highest_executed_checkpint = checkpoint_store
            .get_highest_executed_checkpoint_seq_number()
            .ok()
            .flatten();
        let watermark = self.watermark.get(&Watermark::Indexed).ok().flatten();
        watermark < highest_executed_checkpint
    }

    #[tracing::instrument(skip_all)]
    fn init(
        &mut self,
        authority_store: &AuthorityStore,
        checkpoint_store: &CheckpointStore,
        batch_size_limit: usize,
    ) -> Result<(), StorageError> {
        info!("Initializing RPC indexes");

        let highest_executed_checkpoint =
            checkpoint_store.get_highest_executed_checkpoint_seq_number()?;
        let lowest_available_checkpoint = checkpoint_store
            .get_highest_pruned_checkpoint_seq_number()?
            .map(|c| c.saturating_add(1))
            .unwrap_or(0);
        let lowest_available_checkpoint_objects = authority_store
            .perpetual_tables
            .get_highest_pruned_checkpoint()?
            .map(|c| c.saturating_add(1))
            .unwrap_or(0);
        // Doing backfill requires processing objects so we have to restrict our backfill range
        // to the range of checkpoints that we have objects for.
        let lowest_available_checkpoint =
            lowest_available_checkpoint.max(lowest_available_checkpoint_objects);

        let checkpoint_range = highest_executed_checkpoint.map(|highest_executed_checkpint| {
            lowest_available_checkpoint..=highest_executed_checkpint
        });

        if let Some(checkpoint_range) = checkpoint_range {
            self.index_existing_transactions(authority_store, checkpoint_store, checkpoint_range)?;
        }

        self.initialize_current_epoch(authority_store, checkpoint_store)?;

        // Only index live objects if genesis checkpoint has been executed.
        // If genesis hasn't been executed yet, the objects will be properly indexed
        // as checkpoints are processed through the normal checkpoint execution path.
        if highest_executed_checkpoint.is_some() {
            let make_live_object_indexer = RpcParLiveObjectSetIndexer {
                tables: self,
                batch_size_limit,
            };

            par_index_live_object_set(authority_store, &make_live_object_indexer)?;
        }

        self.watermark.insert(
            &Watermark::Indexed,
            &highest_executed_checkpoint.unwrap_or(0),
        )?;

        self.meta.insert(
            &(),
            &MetadataInfo {
                version: CURRENT_DB_VERSION,
            },
        )?;

        info!("Finished initializing RPC indexes");

        Ok(())
    }

    #[tracing::instrument(skip(self, authority_store, checkpoint_store))]
    fn index_existing_transactions(
        &mut self,
        authority_store: &AuthorityStore,
        checkpoint_store: &CheckpointStore,
        checkpoint_range: std::ops::RangeInclusive<u64>,
    ) -> Result<(), StorageError> {
        info!(
            "Indexing {} checkpoints in range {checkpoint_range:?}",
            checkpoint_range.size_hint().0
        );
        let start_time = Instant::now();

        checkpoint_range.into_iter().try_for_each(|seq| {
            let checkpoint_data =
                sparse_checkpoint_data_for_backfill(authority_store, checkpoint_store, seq)?;

            let mut batch = self.transactions.batch();

            self.index_epoch(&checkpoint_data, &mut batch)?;
            self.index_transactions(&checkpoint_data, &mut batch)?;

            batch
                .write_opt(&(bulk_ingestion_write_options()))
                .map_err(StorageError::from)
        })?;

        info!(
            "Indexing checkpoints took {} seconds",
            start_time.elapsed().as_secs()
        );
        Ok(())
    }

    /// Prune data from this Index
    fn prune(
        &self,
        pruned_checkpoint_watermark: u64,
        checkpoint_contents_to_prune: &[CheckpointContents],
    ) -> Result<(), TypedStoreError> {
        let mut batch = self.transactions.batch();

        let transactions_to_prune = checkpoint_contents_to_prune
            .iter()
            .flat_map(|contents| contents.iter().map(|digests| digests.transaction));

        batch.delete_batch(&self.transactions, transactions_to_prune)?;
        batch.insert_batch(
            &self.watermark,
            [(Watermark::Pruned, pruned_checkpoint_watermark)],
        )?;

        batch.write()
    }

    /// Index a Checkpoint
    fn index_checkpoint(
        &self,
        checkpoint: &CheckpointData,
    ) -> Result<store::rocks::DBBatch, StorageError> {
        debug!(
            checkpoint = checkpoint.checkpoint_summary.sequence_number,
            "indexing checkpoint"
        );

        let mut batch = self.transactions.batch();

        self.index_epoch(checkpoint, &mut batch)?;
        self.index_transactions(checkpoint, &mut batch)?;
        self.index_objects(checkpoint, &mut batch)?;

        batch.insert_batch(
            &self.watermark,
            [(
                Watermark::Indexed,
                checkpoint.checkpoint_summary.sequence_number,
            )],
        )?;

        debug!(
            checkpoint = checkpoint.checkpoint_summary.sequence_number,
            "finished indexing checkpoint"
        );

        Ok(batch)
    }

    fn index_epoch(
        &self,
        checkpoint: &CheckpointData,
        batch: &mut store::rocks::DBBatch,
    ) -> Result<(), StorageError> {
        let Some(epoch_info) = checkpoint.epoch_info()? else {
            return Ok(());
        };
        if epoch_info.epoch > 0 {
            let prev_epoch = epoch_info.epoch - 1;
            let mut current_epoch = self.epochs.get(&prev_epoch)?.unwrap_or_default();
            current_epoch.epoch = prev_epoch; // set this incase there wasn't an entry
            current_epoch.end_timestamp_ms = epoch_info.start_timestamp_ms;
            // current_epoch.end_checkpoint = epoch_info.start_checkpoint.map(|sq| sq - 1);
            batch.insert_batch(&self.epochs, [(prev_epoch, current_epoch)])?;
        }
        batch.insert_batch(&self.epochs, [(epoch_info.epoch, epoch_info)])?;

        Ok(())
    }

    // After attempting to reindex past epochs, ensure that the current epoch is at least partially
    // initalized
    fn initialize_current_epoch(
        &mut self,
        authority_store: &AuthorityStore,
        checkpoint_store: &CheckpointStore,
    ) -> Result<(), StorageError> {
        let Some(checkpoint) = checkpoint_store.get_highest_executed_checkpoint()? else {
            return Ok(());
        };
        let system_state = types::system_state::get_system_state(authority_store)
            .map_err(|e| StorageError::custom(format!("Failed to find system state: {e}")))?;

        let mut epoch = self.epochs.get(&checkpoint.epoch())?.unwrap_or_default();
        epoch.epoch = checkpoint.epoch();

        if epoch.start_timestamp_ms.is_none() {
            epoch.start_timestamp_ms = Some(system_state.epoch_start_timestamp_ms());
        }

        if epoch.system_state.is_none() {
            epoch.system_state = Some(system_state);
        }

        self.epochs.insert(&epoch.epoch, &epoch)?;

        Ok(())
    }

    fn index_transactions(
        &self,
        checkpoint: &CheckpointData,
        batch: &mut store::rocks::DBBatch,
    ) -> Result<(), StorageError> {
        for tx in &checkpoint.transactions {
            let info = TransactionInfo::new(
                &tx.effects,
                &tx.input_objects,
                &tx.output_objects,
                checkpoint.checkpoint_summary.sequence_number,
            );

            let digest = tx.transaction.digest();
            batch.insert_batch(&self.transactions, [(digest, info)])?;
        }

        Ok(())
    }

    fn index_objects(
        &self,
        checkpoint: &CheckpointData,
        batch: &mut store::rocks::DBBatch,
    ) -> Result<(), StorageError> {
        let mut balance_changes: HashMap<BalanceKey, BalanceIndexInfo> = HashMap::new();

        for tx in &checkpoint.transactions {
            // determine changes from removed objects
            for removed_object in tx.removed_objects_pre_version() {
                match removed_object.owner() {
                    Owner::AddressOwner(owner) => {
                        Self::track_coin_balance_change(
                            removed_object,
                            &owner,
                            true,
                            &mut balance_changes,
                        )?;

                        let owner_key = OwnerIndexKey::from_object(removed_object);
                        batch.delete_batch(&self.owner, [owner_key])?;
                    }

                    Owner::Shared { .. } | Owner::Immutable => {}
                }

                // Remove shard/target from indexes
                self.remove_shard_target_index(removed_object, batch)?;
            }

            // determine changes from changed objects
            for (object, old_object) in tx.changed_objects() {
                if let Some(old_object) = old_object {
                    match old_object.owner() {
                        Owner::AddressOwner(owner) => {
                            Self::track_coin_balance_change(
                                old_object,
                                &owner,
                                true,
                                &mut balance_changes,
                            )?;

                            let owner_key = OwnerIndexKey::from_object(old_object);
                            batch.delete_batch(&self.owner, [owner_key])?;
                        }

                        Owner::Shared { .. } | Owner::Immutable => {}
                    }
                }

                match object.owner() {
                    Owner::AddressOwner(owner) => {
                        Self::track_coin_balance_change(
                            object,
                            &owner,
                            false,
                            &mut balance_changes,
                        )?;
                        let owner_key = OwnerIndexKey::from_object(object);
                        let owner_info = OwnerIndexInfo::new(object);
                        batch.insert_batch(&self.owner, [(owner_key, owner_info)])?;
                    }

                    Owner::Shared { .. } | Owner::Immutable => {}
                }

                // Index shard/target objects
                self.index_shard_target_object(object, batch)?;
            }
        }

        batch.partial_merge_batch(&self.balance, balance_changes)?;

        Ok(())
    }

    fn index_shard_target_object(
        &self,
        object: &Object,
        batch: &mut store::rocks::DBBatch,
    ) -> Result<(), StorageError> {
        match object.type_() {
            ObjectType::Shard => {
                if let Some(shard) = object.as_shard() {
                    self.index_shard(object.id(), &shard, batch)?;
                }
            }
            ObjectType::Target => {
                if let Some(target) = object.as_target() {
                    self.index_target(object.id(), &target, batch)?;
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn remove_shard_target_index(
        &self,
        object: &Object,
        batch: &mut store::rocks::DBBatch,
    ) -> Result<(), StorageError> {
        match object.type_() {
            ObjectType::Shard => {
                if let Some(shard) = object.as_shard() {
                    self.remove_shard(object.id(), &shard, batch)?;
                }
            }
            ObjectType::Target => {
                if let Some(target) = object.as_target() {
                    self.remove_target(object.id(), &target, batch)?;
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn index_shard(
        &self,
        shard_id: ObjectID,
        shard: &Shard,
        batch: &mut store::rocks::DBBatch,
    ) -> Result<(), StorageError> {
        let info = ShardIndexInfo::from_shard(shard_id, shard);

        // Index by epoch
        let epoch_key = ShardEpochKey {
            epoch: shard.created_epoch,
            shard_id,
        };
        batch.insert_batch(&self.shard_by_epoch, [(epoch_key, info.clone())])?;

        // Index by submitter
        let submitter_key = ShardSubmitterKey {
            submitter: shard.data_submitter,
            epoch: shard.created_epoch,
            shard_id,
        };
        batch.insert_batch(&self.shard_by_submitter, [(submitter_key, info.clone())])?;

        // Index by winning encoder if present
        if let Some(encoder) = &shard.winning_encoder {
            let encoder_key = ShardEncoderKey {
                encoder: encoder.clone(),
                epoch: shard.created_epoch,
                shard_id,
            };
            batch.insert_batch(&self.shard_by_encoder, [(encoder_key, info)])?;
        }

        Ok(())
    }

    fn remove_shard(
        &self,
        shard_id: ObjectID,
        shard: &Shard,
        batch: &mut store::rocks::DBBatch,
    ) -> Result<(), StorageError> {
        // Remove from epoch index
        let epoch_key = ShardEpochKey {
            epoch: shard.created_epoch,
            shard_id,
        };
        batch.delete_batch(&self.shard_by_epoch, [epoch_key])?;

        // Remove from submitter index
        let submitter_key = ShardSubmitterKey {
            submitter: shard.data_submitter,
            epoch: shard.created_epoch,
            shard_id,
        };
        batch.delete_batch(&self.shard_by_submitter, [submitter_key])?;

        // Remove from encoder index if present
        if let Some(encoder) = &shard.winning_encoder {
            let encoder_key = ShardEncoderKey {
                encoder: encoder.clone(),
                epoch: shard.created_epoch,
                shard_id,
            };
            batch.delete_batch(&self.shard_by_encoder, [encoder_key])?;
        }

        Ok(())
    }

    fn index_target(
        &self,
        target_id: ObjectID,
        target: &Target,
        batch: &mut store::rocks::DBBatch,
    ) -> Result<(), StorageError> {
        let info = TargetIndexInfo::from_target(target_id, target);

        // Index by valid epoch
        let epoch_key = TargetValidEpochKey {
            valid_epoch: info.valid_epoch,
            target_id,
        };
        batch.insert_batch(&self.target_by_valid_epoch, [(epoch_key, info)])?;

        Ok(())
    }

    fn remove_target(
        &self,
        target_id: ObjectID,
        target: &Target,
        batch: &mut store::rocks::DBBatch,
    ) -> Result<(), StorageError> {
        let info = TargetIndexInfo::from_target(target_id, target);

        // Remove from epoch index
        let epoch_key = TargetValidEpochKey {
            valid_epoch: info.valid_epoch,
            target_id,
        };
        batch.delete_batch(&self.target_by_valid_epoch, [epoch_key])?;

        Ok(())
    }

    // =========================================================================
    // SHARD QUERY METHODS
    // =========================================================================

    /// Get all shards created in a specific epoch
    pub fn get_shards_by_epoch(
        &self,
        epoch: EpochId,
    ) -> Result<Vec<ShardIndexInfo>, TypedStoreError> {
        let lower_bound = ShardEpochKey {
            epoch,
            shard_id: ObjectID::ZERO,
        };

        self.shard_by_epoch
            .safe_iter_with_bounds(Some(lower_bound), None)
            .take_while(|item| match item {
                Ok((key, _)) => key.epoch == epoch,
                Err(_) => true,
            })
            .map(|item| item.map(|(_, info)| info))
            .collect()
    }

    /// Get shards submitted by a specific address
    pub fn get_shards_by_submitter(
        &self,
        submitter: SomaAddress,
        epoch: Option<EpochId>,
    ) -> Result<Vec<ShardIndexInfo>, TypedStoreError> {
        let lower_bound = ShardSubmitterKey {
            submitter,
            epoch: epoch.unwrap_or(0),
            shard_id: ObjectID::ZERO,
        };

        self.shard_by_submitter
            .safe_iter_with_bounds(Some(lower_bound), None)
            .take_while(|item| match item {
                Ok((key, _)) => {
                    key.submitter == submitter && epoch.map(|e| key.epoch == e).unwrap_or(true)
                }
                Err(_) => true,
            })
            .map(|item| item.map(|(_, info)| info))
            .collect()
    }

    /// Get shards won by a specific encoder
    pub fn get_shards_by_encoder(
        &self,
        encoder: &EncoderPublicKey,
    ) -> Result<Vec<ShardIndexInfo>, TypedStoreError> {
        let lower_bound = ShardEncoderKey {
            encoder: encoder.clone(),
            epoch: 0,
            shard_id: ObjectID::ZERO,
        };

        self.shard_by_encoder
            .safe_iter_with_bounds(Some(lower_bound), None)
            .take_while(|item| match item {
                Ok((key, _)) => &key.encoder == encoder,
                Err(_) => true,
            })
            .map(|item| item.map(|(_, info)| info))
            .collect()
    }

    /// Get claimable escrows (shards where created_epoch + 2 <= current_epoch)
    pub fn get_claimable_escrows(
        &self,
        current_epoch: EpochId,
    ) -> Result<Vec<ShardIndexInfo>, TypedStoreError> {
        if current_epoch < 2 {
            return Ok(Vec::new());
        }

        let max_epoch = current_epoch - 2;
        let mut results = Vec::new();

        for epoch in 0..=max_epoch {
            let shards = self.get_shards_by_epoch(epoch)?;
            results.extend(shards);
        }

        Ok(results)
    }

    // =========================================================================
    // TARGET QUERY METHODS
    // =========================================================================

    /// Get all targets valid for competition in the given epoch
    pub fn get_valid_targets(
        &self,
        epoch: EpochId,
    ) -> Result<Vec<TargetIndexInfo>, TypedStoreError> {
        let lower_bound = TargetValidEpochKey {
            valid_epoch: epoch,
            target_id: ObjectID::ZERO,
        };

        self.target_by_valid_epoch
            .safe_iter_with_bounds(Some(lower_bound), None)
            .take_while(|item| match item {
                Ok((key, _)) => key.valid_epoch == epoch,
                Err(_) => true,
            })
            .map(|item| item.map(|(_, info)| info))
            .collect()
    }

    /// Get claimable rewards (targets where created_epoch + 2 <= current_epoch)
    pub fn get_claimable_rewards(
        &self,
        current_epoch: EpochId,
    ) -> Result<Vec<TargetIndexInfo>, TypedStoreError> {
        if current_epoch < 2 {
            return Ok(Vec::new());
        }

        let max_valid_epoch = current_epoch - 1;
        let mut results = Vec::new();

        for epoch in 0..=max_valid_epoch {
            let targets = self.get_valid_targets(epoch)?;
            for target in targets {
                if target.is_reward_claimable(current_epoch) {
                    results.push(target);
                }
            }
        }

        Ok(results)
    }

    fn get_epoch_info(&self, epoch: EpochId) -> Result<Option<EpochInfo>, TypedStoreError> {
        self.epochs.get(&epoch)
    }

    fn get_transaction_info(
        &self,
        digest: &TransactionDigest,
    ) -> Result<Option<TransactionInfo>, TypedStoreError> {
        self.transactions.get(digest)
    }

    fn owner_iter(
        &self,
        owner: SomaAddress,
        object_type: Option<ObjectType>,
        cursor: Option<OwnerIndexKey>,
    ) -> Result<
        impl Iterator<Item = Result<(OwnerIndexKey, OwnerIndexInfo), TypedStoreError>> + '_,
        TypedStoreError,
    > {
        // TODO can we figure out how to pass a raw byte array as a cursor?
        let lower_bound = cursor.unwrap_or_else(|| OwnerIndexKey {
            owner,
            object_type: object_type.clone().unwrap(),
            inverted_balance: None,
            object_id: ObjectID::ZERO,
        });

        Ok(self
            .owner
            .safe_iter_with_bounds(Some(lower_bound), None)
            .take_while(move |item| {
                // If there's an error let if flow through
                let Ok((key, _)) = item else {
                    return true;
                };

                // Only take if owner matches
                key.owner == owner
                    // and if an object type was supplied that the type matches
                    && object_type
                        .as_ref()
                        .map(|ty| ty == &key.object_type)
                        .unwrap_or(true)
            }))
    }

    fn get_balance(
        &self,
        owner: &SomaAddress,
    ) -> Result<Option<BalanceIndexInfo>, TypedStoreError> {
        let key = BalanceKey {
            owner: owner.to_owned(),
        };
        self.balance.get(&key)
    }

    fn balance_iter(
        &self,
        owner: SomaAddress,
        cursor: Option<BalanceKey>,
    ) -> Result<
        impl Iterator<Item = Result<(BalanceKey, BalanceIndexInfo), TypedStoreError>> + '_,
        TypedStoreError,
    > {
        let lower_bound = cursor.unwrap_or_else(|| BalanceKey { owner });

        Ok(self
            .balance
            .safe_iter_with_bounds(Some(lower_bound), None)
            .scan((), move |_, item| {
                match item {
                    Ok((key, value)) if key.owner == owner => Some(Ok((key, value))),
                    Ok(_) => None,          // Different owner, stop iteration
                    Err(e) => Some(Err(e)), // Propagate error
                }
            }))
    }

    /// Index a single executed transaction's object changes immediately after execution.
    /// This mirrors the checkpoint-based `index_objects` but for a single transaction.
    fn index_executed_tx_objects(
        &self,
        effects: &TransactionEffects,
        written: &WrittenObjects,
        input_objects: &BTreeMap<ObjectID, Object>,
    ) -> Result<(), StorageError> {
        let mut batch = self.owner.batch();
        let mut balance_changes: HashMap<BalanceKey, BalanceIndexInfo> = HashMap::new();

        let modified_at_versions: HashMap<ObjectID, Version> =
            effects.modified_at_versions().into_iter().collect();

        // Process deleted (removals)
        for (id, _, _) in effects.deleted().into_iter() {
            if let Some(old_version) = modified_at_versions.get(&id) {
                if let Some(old_object) = input_objects.get(&id) {
                    // Verify version matches what we expect
                    if old_object.version() == *old_version {
                        match old_object.owner() {
                            Owner::AddressOwner(owner) => {
                                Self::track_coin_balance_change(
                                    old_object,
                                    owner,
                                    true, // is_removal
                                    &mut balance_changes,
                                )?;
                                let owner_key = OwnerIndexKey::from_object(old_object);
                                batch.delete_batch(&self.owner, [owner_key])?;
                            }
                            Owner::Shared { .. } | Owner::Immutable => {}
                        }
                    }
                }
            }
        }

        // Process all changed objects
        for (oref, owner, kind) in effects.all_changed_objects() {
            let id = &oref.0;

            // For mutated objects, handle old owner removal if owner changed
            if matches!(kind, WriteKind::Mutate) {
                if let Some(old_version) = modified_at_versions.get(id) {
                    if let Some(old_object) = input_objects.get(id) {
                        if old_object.version() == *old_version {
                            // Check if owner changed
                            let old_owner_changed = match old_object.owner() {
                                Owner::AddressOwner(old_addr) => {
                                    !matches!(owner, Owner::AddressOwner(new_addr) if *old_addr == new_addr)
                                }
                                _ => false,
                            };

                            if old_owner_changed {
                                if let Owner::AddressOwner(old_owner) = old_object.owner() {
                                    Self::track_coin_balance_change(
                                        old_object,
                                        old_owner,
                                        true,
                                        &mut balance_changes,
                                    )?;
                                    let owner_key = OwnerIndexKey::from_object(old_object);
                                    batch.delete_batch(&self.owner, [owner_key])?;
                                }
                            }
                        }
                    }
                }
            }

            // Handle new/updated owner entry
            match owner {
                Owner::AddressOwner(addr) => {
                    if let Some(new_object) = written.get(id) {
                        Self::track_coin_balance_change(
                            new_object,
                            &addr,
                            false, // not a removal
                            &mut balance_changes,
                        )?;
                        let owner_key = OwnerIndexKey::from_object(new_object);
                        let owner_info = OwnerIndexInfo::new(new_object);
                        batch.insert_batch(&self.owner, [(owner_key, owner_info)])?;
                    }
                }
                Owner::Shared { .. } | Owner::Immutable => {}
            }
        }

        // Apply balance changes via merge operator
        batch.partial_merge_batch(&self.balance, balance_changes)?;
        batch.write()?;

        Ok(())
    }
}

fn sparse_checkpoint_data_for_backfill(
    authority_store: &AuthorityStore,
    checkpoint_store: &CheckpointStore,
    checkpoint: u64,
) -> Result<CheckpointData, StorageError> {
    let summary = checkpoint_store
        .get_checkpoint_by_sequence_number(checkpoint)?
        .ok_or_else(|| StorageError::missing(format!("missing checkpoint {checkpoint}")))?;
    let contents = checkpoint_store
        .get_checkpoint_contents(&summary.content_digest)?
        .ok_or_else(|| StorageError::missing(format!("missing checkpoint {checkpoint}")))?;

    let transaction_digests = contents
        .iter()
        .map(|execution_digests| execution_digests.transaction)
        .collect::<Vec<_>>();
    let transactions = authority_store
        .multi_get_transaction_blocks(&transaction_digests)?
        .into_iter()
        .map(|maybe_transaction| {
            maybe_transaction.ok_or_else(|| StorageError::custom("missing transaction"))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let effects = authority_store
        .multi_get_executed_effects(&transaction_digests)?
        .into_iter()
        .map(|maybe_effects| maybe_effects.ok_or_else(|| StorageError::custom("missing effects")))
        .collect::<Result<Vec<_>, _>>()?;

    let mut full_transactions = Vec::with_capacity(transactions.len());
    for (tx, fx) in transactions.into_iter().zip(effects) {
        let input_objects = types::storage::get_transaction_input_objects(authority_store, &fx)?;
        let output_objects = types::storage::get_transaction_output_objects(authority_store, &fx)?;

        let full_transaction = CheckpointTransaction {
            transaction: tx.into(),
            effects: fx,

            input_objects,
            output_objects,
        };

        full_transactions.push(full_transaction);
    }

    let checkpoint_data = CheckpointData {
        checkpoint_summary: summary.into(),
        checkpoint_contents: contents,
        transactions: full_transactions,
    };

    Ok(checkpoint_data)
}

pub struct RpcIndexStore {
    tables: IndexStoreTables,
    pending_updates: Mutex<BTreeMap<u64, store::rocks::DBBatch>>,
}

impl RpcIndexStore {
    /// Given the provided directory, construct the path to the db
    fn db_path(dir: &Path) -> PathBuf {
        dir.join("rpc-index")
    }

    pub async fn new(
        dir: &Path,
        authority_store: &AuthorityStore,
        checkpoint_store: &CheckpointStore,
    ) -> Self {
        let path = Self::db_path(dir);

        let tables = {
            let tables = IndexStoreTables::open_with_index_options(&path);

            // If the index tables are uninitialized or on an older version then we need to
            // populate them
            if tables.needs_to_do_initialization(checkpoint_store) {
                let batch_size_limit;

                let mut tables = {
                    drop(tables);
                    store::rocks::safe_drop_db(path.clone(), Duration::from_secs(30))
                        .await
                        .expect("unable to destroy old rpc-index db");

                    // Open the empty DB with `unordered_write`s enabled in order to get a ~3x
                    // speedup when indexing
                    let mut options = store::rocksdb::Options::default();
                    options.set_unordered_write(true);

                    // Allow CPU-intensive flushing operations to use all CPUs.
                    let max_background_jobs = if let Some(jobs) = None
                    //TODO: index_config.as_ref().and_then(|c| c.max_background_jobs)
                    {
                        debug!("Using config override for max_background_jobs: {}", jobs);
                        jobs
                    } else {
                        let jobs = num_cpus::get() as i32;
                        debug!(
                            "Calculated max_background_jobs: {} (based on CPU count)",
                            jobs
                        );
                        jobs
                    };
                    options.set_max_background_jobs(max_background_jobs);

                    // We are disabling compaction for all column families below. This means we can
                    // also disable the backpressure that slows down writes when the number of L0
                    // files builds up since we will never compact them anyway.
                    options.set_level_zero_file_num_compaction_trigger(0);
                    options.set_level_zero_slowdown_writes_trigger(-1);
                    options.set_level_zero_stop_writes_trigger(i32::MAX);

                    let total_memory_bytes = get_available_memory();
                    // This is an upper bound on the amount to of ram the memtables can use across
                    // all column families.
                    let db_buffer_size = if let Some(size) = None
                    //TODO:index_config.as_ref().and_then(|c| c.db_write_buffer_size)
                    {
                        debug!(
                            "Using config override for db_write_buffer_size: {} bytes",
                            size
                        );
                        size
                    } else {
                        // Default to 80% of system RAM
                        let size = (total_memory_bytes as f64 * 0.8) as usize;
                        debug!(
                            "Calculated db_write_buffer_size: {} bytes (80% of {} total bytes)",
                            size, total_memory_bytes
                        );
                        size
                    };
                    options.set_db_write_buffer_size(db_buffer_size);

                    // Create column family specific options.
                    let mut table_config_map = BTreeMap::new();

                    // Create options with compactions disabled and large write buffers.
                    // Each CF can use up to 25% of system RAM, but total is still limited by
                    // set_db_write_buffer_size configured above.
                    let mut cf_options = store::rocks::default_db_options();
                    cf_options.options.set_disable_auto_compactions(true);

                    let (buffer_size, buffer_count) = match (
                        None,
                        None, //TODO: index_config.as_ref().and_then(|c| c.cf_write_buffer_size),
                             // index_config
                             //     .as_ref()
                             //     .and_then(|c| c.cf_max_write_buffer_number),
                    ) {
                        (Some(size), Some(count)) => {
                            debug!(
                                "Using config overrides - buffer_size: {} bytes, buffer_count: {}",
                                size, count
                            );
                            (size, count)
                        }
                        (None, None) => {
                            // Calculate buffer configuration: 25% of RAM split across buffers
                            let cf_memory_budget = (total_memory_bytes as f64 * 0.25) as usize;
                            debug!(
                                "Column family memory budget: {} bytes (25% of {} total bytes)",
                                cf_memory_budget, total_memory_bytes
                            );
                            const MIN_BUFFER_SIZE: usize = 64 * 1024 * 1024; // 64MB minimum

                            // Target number of buffers based on CPU count
                            // More CPUs = more parallel flushing capability
                            let target_buffer_count = num_cpus::get().max(2);

                            // Aim for CPU-based buffer count, but reduce if it would make buffers too small
                            //   For example:
                            // - 128GB RAM, 32 CPUs: 32GB per CF / 32 buffers = 1GB each
                            // - 16GB RAM, 8 CPUs: 4GB per CF / 8 buffers = 512MB each
                            // - 4GB RAM, 8 CPUs: 1GB per CF / 64MB min = ~16 buffers of 64MB each
                            let buffer_size =
                                (cf_memory_budget / target_buffer_count).max(MIN_BUFFER_SIZE);
                            let buffer_count = (cf_memory_budget / buffer_size)
                                .clamp(2, target_buffer_count)
                                as i32;
                            debug!("Calculated buffer_size: {} bytes, buffer_count: {} (based on {} CPUs)", 
                                buffer_size, buffer_count, target_buffer_count);
                            (buffer_size, buffer_count)
                        }
                        _ => {
                            panic!("indexing-cf-write-buffer-size and indexing-cf-max-write-buffer-number must both be specified or both be omitted");
                        }
                    };

                    cf_options.options.set_write_buffer_size(buffer_size);
                    cf_options.options.set_max_write_buffer_number(buffer_count);

                    // TODO: Calculate batch size limit: default to half the buffer size or 128MB, whichever is smaller
                    // batch_size_limit = if let Some(limit) =
                    //     index_config.as_ref().and_then(|c| c.batch_size_limit)
                    // {
                    //     debug!(
                    //         "Using config override for batch_size_limit: {} bytes",
                    //         limit
                    //     );
                    //     limit
                    // } else {
                    batch_size_limit = {
                        let half_buffer = buffer_size / 2;
                        let default_limit = 1 << 27; // 128MB
                        let limit = half_buffer.min(default_limit);
                        debug!("Calculated batch_size_limit: {} bytes (min of half_buffer={} and default_limit={})", 
                            limit, half_buffer, default_limit);
                        limit
                    };

                    // Apply cf_options to all tables
                    for (table_name, _) in IndexStoreTables::describe_tables() {
                        table_config_map.insert(table_name, cf_options.clone());
                    }

                    // Override Balance options with the merge operator
                    let mut balance_options = cf_options.clone();
                    balance_options = balance_options.set_merge_operator_associative(
                        "balance_merge",
                        balance_delta_merge_operator,
                    );
                    table_config_map.insert("balance".to_string(), balance_options);

                    IndexStoreTables::open_with_options(
                        &path,
                        options,
                        Some(DBMapTableConfigMap::new(table_config_map)),
                    )
                };

                tables
                    .init(authority_store, checkpoint_store, batch_size_limit)
                    .expect("unable to initialize rpc index from live object set");

                // Flush all data to disk before dropping tables.
                // This is critical because WAL is disabled during bulk indexing.
                // Note we only need to call flush on one table because all tables share the same
                // underlying database.
                tables
                    .meta
                    .flush()
                    .expect("Failed to flush RPC index tables to disk");

                let weak_db = Arc::downgrade(&tables.meta.db);
                drop(tables);

                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
                loop {
                    if weak_db.strong_count() == 0 {
                        break;
                    }
                    if std::time::Instant::now() > deadline {
                        panic!("unable to reopen DB after indexing");
                    }
                    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                }

                // Reopen the DB with default options (eg without `unordered_write`s enabled)
                let reopened_tables = IndexStoreTables::open_with_index_options(&path);

                // Sanity check: verify the database version was persisted correctly
                let stored_version = reopened_tables
                    .meta
                    .get(&())
                    .expect("Failed to read metadata from reopened database")
                    .expect("Metadata not found in reopened database");
                assert_eq!(
                    stored_version.version, CURRENT_DB_VERSION,
                    "Database version mismatch after flush and reopen: expected {}, found {}",
                    CURRENT_DB_VERSION, stored_version.version
                );

                reopened_tables
            } else {
                tables
            }
        };

        Self {
            tables,
            pending_updates: Default::default(),
        }
    }

    pub fn new_without_init(dir: &Path) -> Self {
        let path = Self::db_path(dir);
        let tables = IndexStoreTables::open_with_index_options(path);

        Self {
            tables,
            pending_updates: Default::default(),
        }
    }

    pub fn prune(
        &self,
        pruned_checkpoint_watermark: u64,
        checkpoint_contents_to_prune: &[CheckpointContents],
    ) -> Result<(), TypedStoreError> {
        self.tables
            .prune(pruned_checkpoint_watermark, checkpoint_contents_to_prune)
    }

    /// Index a checkpoint and stage the index updated in `pending_updates`.
    ///
    /// Updates will not be committed to the database until `commit_update_for_checkpoint` is
    /// called.
    #[tracing::instrument(
        skip_all,
        fields(checkpoint = checkpoint.checkpoint_summary.sequence_number)
    )]
    pub fn index_checkpoint(&self, checkpoint: &CheckpointData) {
        let sequence_number = checkpoint.checkpoint_summary.sequence_number;
        let batch = self.tables.index_checkpoint(checkpoint).expect("db error");

        self.pending_updates
            .lock()
            .unwrap()
            .insert(sequence_number.into(), batch);
    }

    /// Commits the pending updates for the provided checkpoint number.
    ///
    /// Invariants:
    /// - `index_checkpoint` must have been called for the provided checkpoint
    /// - Callers of this function must ensure that it is called for each checkpoint in sequential
    ///   order. This will panic if the provided checkpoint does not match the expected next
    ///   checkpoint to commit.
    #[tracing::instrument(skip(self))]
    pub fn commit_update_for_checkpoint(&self, checkpoint: u64) -> Result<(), StorageError> {
        let next_batch = self.pending_updates.lock().unwrap().pop_first();

        // Its expected that the next batch exists
        let (next_sequence_number, batch) = next_batch.unwrap();
        assert_eq!(
            checkpoint, next_sequence_number,
            "commit_update_for_checkpoint must be called in order"
        );

        Ok(batch.write()?)
    }

    pub fn get_epoch_info(&self, epoch: EpochId) -> Result<Option<EpochInfo>, TypedStoreError> {
        self.tables.get_epoch_info(epoch)
    }

    pub fn get_transaction_info(
        &self,
        digest: &TransactionDigest,
    ) -> Result<Option<TransactionInfo>, TypedStoreError> {
        self.tables.get_transaction_info(digest)
    }

    pub fn owner_iter(
        &self,
        owner: SomaAddress,
        object_type: Option<ObjectType>,
        cursor: Option<OwnerIndexKey>,
    ) -> Result<
        impl Iterator<Item = Result<(OwnerIndexKey, OwnerIndexInfo), TypedStoreError>> + '_,
        TypedStoreError,
    > {
        self.tables.owner_iter(owner, object_type, cursor)
    }

    pub fn get_balance(
        &self,
        owner: &SomaAddress,
    ) -> Result<Option<BalanceIndexInfo>, TypedStoreError> {
        self.tables.get_balance(owner)
    }

    pub fn balance_iter(
        &self,
        owner: SomaAddress,
        cursor: Option<BalanceKey>,
    ) -> Result<
        impl Iterator<Item = Result<(BalanceKey, BalanceIndexInfo), TypedStoreError>> + '_,
        TypedStoreError,
    > {
        self.tables.balance_iter(owner, cursor)
    }

    pub fn get_highest_indexed_checkpoint_seq_number(
        &self,
    ) -> Result<Option<CheckpointSequenceNumber>, TypedStoreError> {
        self.tables.watermark.get(&Watermark::Indexed)
    }

    /// Index a transaction immediately after execution (before checkpoint finalization).
    /// This updates the owner and balance indexes in real-time.
    pub fn index_executed_tx(
        &self,
        effects: &TransactionEffects,
        written: &WrittenObjects,
        input_objects: &BTreeMap<ObjectID, Object>,
    ) -> Result<(), StorageError> {
        self.tables
            .index_executed_tx_objects(effects, written, input_objects)
    }

    // =========================================================================
    // SHARD QUERY METHODS
    // =========================================================================

    /// Get all shards created in a specific epoch
    pub fn get_shards_by_epoch(
        &self,
        epoch: EpochId,
    ) -> Result<Vec<ShardIndexInfo>, TypedStoreError> {
        self.tables.get_shards_by_epoch(epoch)
    }

    /// Get shards submitted by a specific address
    pub fn get_shards_by_submitter(
        &self,
        submitter: SomaAddress,
        epoch: Option<EpochId>,
    ) -> Result<Vec<ShardIndexInfo>, TypedStoreError> {
        self.tables.get_shards_by_submitter(submitter, epoch)
    }

    /// Get shards won by a specific encoder
    pub fn get_shards_by_encoder(
        &self,
        encoder: &EncoderPublicKey,
    ) -> Result<Vec<ShardIndexInfo>, TypedStoreError> {
        self.tables.get_shards_by_encoder(encoder)
    }

    /// Get claimable escrows
    pub fn get_claimable_escrows(
        &self,
        current_epoch: EpochId,
    ) -> Result<Vec<ShardIndexInfo>, TypedStoreError> {
        self.tables.get_claimable_escrows(current_epoch)
    }

    // =========================================================================
    // TARGET QUERY METHODS
    // =========================================================================

    /// Get all targets valid for competition in the given epoch
    pub fn get_valid_targets(
        &self,
        epoch: EpochId,
    ) -> Result<Vec<TargetIndexInfo>, TypedStoreError> {
        self.tables.get_valid_targets(epoch)
    }

    /// Get claimable rewards
    pub fn get_claimable_rewards(
        &self,
        current_epoch: EpochId,
    ) -> Result<Vec<TargetIndexInfo>, TypedStoreError> {
        self.tables.get_claimable_rewards(current_epoch)
    }
}

struct RpcParLiveObjectSetIndexer<'a> {
    tables: &'a IndexStoreTables,
    batch_size_limit: usize,
}

struct RpcLiveObjectIndexer<'a> {
    tables: &'a IndexStoreTables,
    batch: store::rocks::DBBatch,
    balance_changes: HashMap<BalanceKey, BalanceIndexInfo>,
    batch_size_limit: usize,
}

impl<'a> ParMakeLiveObjectIndexer for RpcParLiveObjectSetIndexer<'a> {
    type ObjectIndexer = RpcLiveObjectIndexer<'a>;

    fn make_live_object_indexer(&self) -> Self::ObjectIndexer {
        RpcLiveObjectIndexer {
            tables: self.tables,
            batch: self.tables.owner.batch(),
            balance_changes: HashMap::new(),
            batch_size_limit: self.batch_size_limit,
        }
    }
}

impl LiveObjectIndexer for RpcLiveObjectIndexer<'_> {
    fn index_object(&mut self, object: Object) -> Result<(), StorageError> {
        match object.owner {
            // Owner Index
            Owner::AddressOwner(owner) => {
                let owner_key = OwnerIndexKey::from_object(&object);
                let owner_info = OwnerIndexInfo::new(&object);
                self.batch
                    .insert_batch(&self.tables.owner, [(owner_key, owner_info)])?;

                if let Some(value) = object.as_coin() {
                    let balance_key = BalanceKey { owner };
                    let balance_info = BalanceIndexInfo::from(value);
                    self.balance_changes
                        .entry(balance_key)
                        .or_default()
                        .merge_delta(&balance_info);

                    if self.balance_changes.len() >= BALANCE_FLUSH_THRESHOLD {
                        self.batch.partial_merge_batch(
                            &self.tables.balance,
                            std::mem::take(&mut self.balance_changes),
                        )?;
                    }
                }
            }

            Owner::Shared { .. } | Owner::Immutable => {}
        }

        // If the batch size grows to greater than the limit then write out to the DB so that the
        // data we need to hold in memory doesn't grow unbounded.
        if self.batch.size_in_bytes() >= self.batch_size_limit {
            std::mem::replace(&mut self.batch, self.tables.owner.batch())
                .write_opt(&bulk_ingestion_write_options())?;
        }

        Ok(())
    }

    fn finish(mut self) -> Result<(), StorageError> {
        self.batch.partial_merge_batch(
            &self.tables.balance,
            std::mem::take(&mut self.balance_changes),
        )?;
        self.batch.write_opt(&bulk_ingestion_write_options())?;

        Ok(())
    }
}

/// Make `LiveObjectIndexer`s for parallel indexing of the live object set
pub trait ParMakeLiveObjectIndexer: Sync {
    type ObjectIndexer: LiveObjectIndexer;

    fn make_live_object_indexer(&self) -> Self::ObjectIndexer;
}

/// Represents an instance of a indexer that operates on a subset of the live object set
pub trait LiveObjectIndexer {
    /// Called on each object in the range of the live object set this indexer task is responsible
    /// for.
    fn index_object(&mut self, object: Object) -> Result<(), StorageError>;

    /// Called once the range of objects this indexer task is responsible for have been processed
    /// by calling `index_object`.
    fn finish(self) -> Result<(), StorageError>;
}

/// Utility for iterating over, and indexing, the live object set in parallel
///
/// This is done by dividing the addressable ObjectID space into smaller, disjoint sets and
/// operating on each set in parallel in a separate thread. User's will need to implement the
/// `ParMakeLiveObjectIndexer` trait which will be used to make N `LiveObjectIndexer`s which will
/// then process one of the disjoint parts of the live object set.
#[tracing::instrument(skip_all)]
pub fn par_index_live_object_set<T: ParMakeLiveObjectIndexer>(
    authority_store: &AuthorityStore,
    make_indexer: &T,
) -> Result<(), StorageError> {
    info!("Indexing Live Object Set");
    let start_time = Instant::now();
    std::thread::scope(|s| -> Result<(), StorageError> {
        let mut threads = Vec::new();
        const BITS: u8 = 5;
        for index in 0u8..(1 << BITS) {
            threads.push(s.spawn(move || {
                let object_indexer = make_indexer.make_live_object_indexer();
                live_object_set_index_task(index, BITS, authority_store, object_indexer)
            }));
        }

        // join threads
        for thread in threads {
            thread.join().unwrap()?;
        }

        Ok(())
    })?;

    info!(
        "Indexing Live Object Set took {} seconds",
        start_time.elapsed().as_secs()
    );

    Ok(())
}

#[tracing::instrument(skip(authority_store, object_indexer))]
fn live_object_set_index_task<T: LiveObjectIndexer>(
    task_id: u8,
    bits: u8,
    authority_store: &AuthorityStore,
    mut object_indexer: T,
) -> Result<(), StorageError> {
    let mut id_bytes = [0; ObjectID::LENGTH];
    id_bytes[0] = task_id << (8 - bits);
    let start_id = ObjectID::new(id_bytes);

    id_bytes[0] |= (1 << (8 - bits)) - 1;
    for element in id_bytes.iter_mut().skip(1) {
        *element = u8::MAX;
    }
    let end_id = ObjectID::new(id_bytes);

    let mut object_scanned: u64 = 0;
    for object in authority_store
        .perpetual_tables
        .range_iter_live_object_set(Some(start_id), Some(end_id), false)
        .filter_map(LiveObject::to_normal)
    {
        object_scanned += 1;
        if object_scanned % 2_000_000 == 0 {
            info!(
                "[Index] Task {}: object scanned: {}",
                task_id, object_scanned
            );
        }

        object_indexer.index_object(object)?
    }

    object_indexer.finish()?;

    Ok(())
}
