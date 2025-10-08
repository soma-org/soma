use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use tracing::{debug, info};
use types::checkpoint::{CheckpointData, CheckpointTransaction};
use types::committee::EpochId;
use types::consensus::ConsensusTransactionKind;
use types::digests::TransactionDigest;
use types::object::LiveObject;
use types::storage::read_store::{EpochInfo, TransactionInfo};
use types::storage::storage_error::Error as StorageError;
use types::system_state::SystemStateTrait;
use types::{
    accumulator::CommitIndex,
    base::SomaAddress,
    object::{Object, ObjectID, ObjectType, Owner, Version},
};

use crate::commit::CommitStore;
use crate::output::ConsensusOutputAPI;
use crate::store::AuthorityStore;

const CURRENT_DB_VERSION: u64 = 1;
const BALANCE_FLUSH_THRESHOLD: usize = 10_000;

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

/// RocksDB tables for the RpcIndexStore
///
/// Anytime a new table is added, or and existing one has it's schema changed, make sure to also
/// update the value of `CURRENT_DB_VERSION`.
///
/// NOTE: Authors and Reviewers before adding any new tables ensure that they are either:
/// - bounded in size by the live object set
/// - are prune-able and have corresponding logic in the `prune` function
// #[derive(DBMapUtils)]
struct IndexStoreTables {
    /// A singleton that store metadata information on the DB.
    ///
    /// A few uses for this singleton:
    /// - determining if the DB has been initialized (as some tables will still be empty post
    ///     initialization)
    /// - version of the DB. Everytime a new table or schema is changed the version number needs to
    ///     be incremented.
    meta: RwLock<BTreeMap<(), MetadataInfo>>,

    /// Table used to track watermark for the highest indexed checkpoint
    ///
    /// This is useful to help know the highest checkpoint that was indexed in the event that the
    /// node was running with indexes enabled, then run for a period of time with indexes disabled,
    /// and then run with them enabled again so that the tables can be reinitialized.
    watermark: RwLock<BTreeMap<Watermark, CommitIndex>>,

    /// An index of extra metadata for Epochs.
    ///
    /// Only contains entries for transactions which have yet to be pruned from the main database.
    epochs: RwLock<BTreeMap<EpochId, EpochInfo>>,

    /// An index of extra metadata for Transactions.
    ///
    /// Only contains entries for transactions which have yet to be pruned from the main database.
    transactions: RwLock<BTreeMap<TransactionDigest, TransactionInfo>>,

    /// An index of object ownership.
    ///
    /// Allows an efficient iterator to list all objects currently owned by a specific user
    /// account.
    owner: RwLock<BTreeMap<OwnerIndexKey, OwnerIndexInfo>>,

    /// An index of Balances.
    ///
    /// Allows looking up balances by owner address and coin type.
    balance: RwLock<BTreeMap<BalanceKey, BalanceIndexInfo>>,
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

    fn open<P: Into<PathBuf>>(path: P) -> Self {
        Self {
            meta: RwLock::new(BTreeMap::new()),
            watermark: RwLock::new(BTreeMap::new()),
            epochs: RwLock::new(BTreeMap::new()),
            transactions: RwLock::new(BTreeMap::new()),
            owner: RwLock::new(BTreeMap::new()),
            balance: RwLock::new(BTreeMap::new()),
        }
    }

    fn needs_to_do_initialization(&self, commit_store: &CommitStore) -> bool {
        (match self.meta.read().get(&()) {
            Some(metadata) => metadata.version != CURRENT_DB_VERSION,
            None => true,
        }) || self.is_indexed_watermark_out_of_date(commit_store)
    }

    // Check if the index watermark is behind the highets_executed watermark.
    fn is_indexed_watermark_out_of_date(&self, commit_store: &CommitStore) -> bool {
        let highest_executed_checkpoint = commit_store
            .get_highest_executed_commit_index()
            .ok()
            .flatten();
        let lock = self.watermark.read();
        let watermark = lock.get(&Watermark::Indexed);
        watermark < highest_executed_checkpoint.as_ref()
    }

    #[tracing::instrument(skip_all)]
    fn init(
        &mut self,
        authority_store: &AuthorityStore,
        commit_store: &CommitStore,
        batch_size_limit: usize,
    ) -> Result<(), StorageError> {
        info!("Initializing RPC indexes");

        let highest_executed_checkpoint = commit_store.get_highest_executed_commit_index()?;

        // TODO: get lowest available checkpoint after implementing pruning
        // let lowest_available_checkpoint = commit_store
        //     .get_highest_synced_commit_index()?
        //     .map(|c| c.saturating_add(1))
        //     .unwrap_or(0);
        // let lowest_available_checkpoint_objects = authority_store
        //     .perpetual_tables
        //     .get_highest_pruned_checkpoint()?
        //     .map(|c| c.saturating_add(1))
        //     .unwrap_or(0);
        // // Doing backfill requires processing objects so we have to restrict our backfill range
        // // to the range of checkpoints that we have objects for.
        // let lowest_available_checkpoint =
        //     lowest_available_checkpoint.max(lowest_available_checkpoint_objects);

        let checkpoint_range = highest_executed_checkpoint
            .map(|highest_executed_checkpint| 0..=highest_executed_checkpint.into());

        if let Some(checkpoint_range) = checkpoint_range {
            self.index_existing_transactions(authority_store, commit_store, checkpoint_range)?;
        }

        self.initialize_current_epoch(authority_store, commit_store)?;

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

        self.watermark
            .write()
            .insert(Watermark::Indexed, highest_executed_checkpoint.unwrap_or(0));

        self.meta.write().insert(
            (),
            MetadataInfo {
                version: CURRENT_DB_VERSION,
            },
        );

        info!("Finished initializing RPC indexes");

        Ok(())
    }

    #[tracing::instrument(skip(self, authority_store, commit_store))]
    fn index_existing_transactions(
        &mut self,
        authority_store: &AuthorityStore,
        commit_store: &CommitStore,
        checkpoint_range: std::ops::RangeInclusive<u64>,
    ) -> Result<(), StorageError> {
        info!(
            "Indexing {} checkpoints in range {checkpoint_range:?}",
            checkpoint_range.size_hint().0
        );
        let start_time = Instant::now();

        checkpoint_range.into_iter().try_for_each(|seq| {
            let checkpoint_data =
                sparse_checkpoint_data_for_backfill(authority_store, commit_store, seq)?;

            self.index_epoch(&checkpoint_data);
            self.index_transactions(&checkpoint_data);

            Ok::<(), StorageError>(())
        })?;

        info!(
            "Indexing checkpoints took {} seconds",
            start_time.elapsed().as_secs()
        );
        Ok(())
    }

    // TODO: Prune data from this Index
    // fn prune(
    //     &self,
    //     pruned_checkpoint_watermark: u64,
    //     checkpoint_contents_to_prune: &[CheckpointContents],
    // ) -> Result<(), StorageError> {
    //     let transactions_to_prune = checkpoint_contents_to_prune
    //         .iter()
    //         .flat_map(|contents| contents.iter().map(|digests| digests.transaction));

    //     batch.delete_batch(&self.transactions, transactions_to_prune)?;
    //     batch.insert_batch(
    //         &self.watermark,
    //         [(Watermark::Pruned, pruned_checkpoint_watermark)],
    //     )?;

    //     batch.write()
    // }

    /// Index a Checkpoint
    fn index_checkpoint(&self, checkpoint: &CheckpointData) -> Result<(), StorageError> {
        debug!(
            checkpoint = checkpoint.committed_subdag.commit_ref.index,
            "indexing checkpoint"
        );

        self.index_epoch(checkpoint)?;
        self.index_transactions(checkpoint)?;
        self.index_objects(checkpoint)?;

        self.watermark.write().insert(
            Watermark::Indexed,
            checkpoint.committed_subdag.commit_ref.index,
        );

        debug!(
            checkpoint = checkpoint.committed_subdag.commit_ref.index,
            "finished indexing checkpoint"
        );

        Ok(())
    }

    fn index_epoch(&self, checkpoint: &CheckpointData) -> Result<(), StorageError> {
        let Some(epoch_info) = checkpoint.epoch_info()? else {
            return Ok(());
        };
        if epoch_info.epoch > 0 {
            let prev_epoch = epoch_info.epoch - 1;
            let mut current_epoch = self
                .epochs
                .read()
                .get(&prev_epoch)
                .cloned()
                .unwrap_or_default();
            current_epoch.epoch = prev_epoch; // set this incase there wasn't an entry
            current_epoch.end_timestamp_ms = epoch_info.start_timestamp_ms;
            // current_epoch.end_checkpoint = epoch_info.start_checkpoint.map(|sq| sq - 1);
            self.epochs.write().insert(prev_epoch, current_epoch);
        }
        self.epochs.write().insert(epoch_info.epoch, epoch_info);

        Ok(())
    }

    // After attempting to reindex past epochs, ensure that the current epoch is at least partially
    // initalized
    fn initialize_current_epoch(
        &mut self,
        authority_store: &AuthorityStore,
        commit_store: &CommitStore,
    ) -> Result<(), StorageError> {
        let Some(checkpoint) = commit_store.get_highest_executed_commit()? else {
            return Ok(());
        };

        let system_state = types::system_state::get_system_state(authority_store)
            .map_err(|e| StorageError::custom(format!("Failed to find system state: {e}")))?;

        let mut epoch = self
            .epochs
            .read()
            .get(&checkpoint.epoch())
            .cloned()
            .unwrap_or_default();
        epoch.epoch = checkpoint.epoch();

        if epoch.start_timestamp_ms.is_none() {
            epoch.start_timestamp_ms = Some(system_state.epoch_start_timestamp_ms());
        }

        if epoch.system_state.is_none() {
            epoch.system_state = Some(system_state);
        }

        self.epochs.write().insert(epoch.epoch, epoch);

        Ok(())
    }

    fn index_transactions(&self, checkpoint: &CheckpointData) -> Result<(), StorageError> {
        for tx in &checkpoint.transactions {
            let info = TransactionInfo::new(
                &tx.effects,
                &tx.input_objects,
                &tx.output_objects,
                checkpoint.committed_subdag.commit_ref.index.into(),
            );

            let digest = tx.transaction.digest();
            self.transactions.write().insert(*digest, info);
        }

        Ok(())
    }

    fn index_objects(&self, checkpoint: &CheckpointData) -> Result<(), StorageError> {
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
                        self.owner.write().remove(&owner_key);
                    }

                    Owner::Shared { .. } | Owner::Immutable => {}
                }
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
                            self.owner.write().remove(&owner_key);
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
                        self.owner.write().insert(owner_key, owner_info);
                    }

                    Owner::Shared { .. } | Owner::Immutable => {}
                }
            }
        }

        self.apply_balance_changes(balance_changes)?;

        Ok(())
    }

    fn apply_balance_changes(
        &self,
        balance_changes: HashMap<BalanceKey, BalanceIndexInfo>,
    ) -> Result<(), StorageError> {
        let mut balance_table = self.balance.write();

        for (key, delta) in balance_changes {
            match balance_table.get_mut(&key) {
                Some(existing) => {
                    // Merge the delta into the existing balance
                    existing.merge_delta(&delta);
                }
                None => {
                    // Insert new balance entry
                    balance_table.insert(key, delta);
                }
            }
        }

        Ok(())
    }

    fn get_epoch_info(&self, epoch: EpochId) -> Option<EpochInfo> {
        self.epochs.read().get(&epoch).cloned()
    }

    fn get_transaction_info(&self, digest: &TransactionDigest) -> Option<TransactionInfo> {
        self.transactions.read().get(digest).cloned()
    }

    fn owner_iter(
        &self,
        owner: SomaAddress,
        object_type: Option<ObjectType>,
        cursor: Option<OwnerIndexKey>,
    ) -> Result<
        impl Iterator<Item = Result<(OwnerIndexKey, OwnerIndexInfo), StorageError>> + '_,
        StorageError,
    > {
        // TODO can we figure out how to pass a raw byte array as a cursor?
        let lower_bound = cursor.unwrap_or_else(|| OwnerIndexKey {
            owner,
            object_type: object_type.clone().unwrap(),
            inverted_balance: None,
            object_id: ObjectID::ZERO,
        });
        let guard = self.owner.read();
        let results: Vec<_> = guard
            .range(lower_bound..)
            .take_while(|(key, _)| {
                key.owner == owner
                    && object_type
                        .as_ref()
                        .map(|ty| ty == &key.object_type)
                        .unwrap_or(true)
            })
            .map(|(k, v)| Ok((k.clone(), v.clone())))
            .collect();

        Ok(results.into_iter())
    }

    fn get_balance(&self, owner: &SomaAddress) -> Option<BalanceIndexInfo> {
        let key = BalanceKey {
            owner: owner.to_owned(),
        };
        self.balance.read().get(&key).cloned()
    }

    fn balance_iter(
        &self,
        owner: SomaAddress,
        cursor: Option<BalanceKey>,
    ) -> Result<
        impl Iterator<Item = Result<(BalanceKey, BalanceIndexInfo), StorageError>> + '_,
        StorageError,
    > {
        let lower_bound = cursor.unwrap_or_else(|| BalanceKey { owner });

        let guard = self.balance.read();
        let results: Vec<_> = guard
            .range(lower_bound..)
            .take_while(|(key, _)| key.owner == owner)
            .map(|(k, v)| Ok((k.clone(), v.clone())))
            .collect();

        Ok(results.into_iter())
    }
}

fn sparse_checkpoint_data_for_backfill(
    authority_store: &AuthorityStore,
    commit_store: &CommitStore,
    checkpoint: u64,
) -> Result<CheckpointData, StorageError> {
    let committed_subdag = commit_store
        .get_commit_by_index(checkpoint.try_into().unwrap())? //TODO: fix this
        .ok_or_else(|| StorageError::missing(format!("missing checkpoint {checkpoint}")))?;
    // let contents = commit_store
    //     .get_checkpoint_contents(&summary.content_digest)?
    //     .ok_or_else(|| StorageError::missing(format!("missing checkpoint {checkpoint}")))?;

    let all_tx_digests: HashSet<_> = committed_subdag
        .transactions()
        .iter()
        .flat_map(|(_, authority_transactions)| {
            authority_transactions
                .iter()
                .filter_map(|(_, transaction)| {
                    if let ConsensusTransactionKind::UserTransaction(cert_tx) = &transaction.kind {
                        Some(*cert_tx.digest())
                    } else {
                        None
                    }
                })
        })
        .collect();

    let all_tx_digests: Vec<_> = all_tx_digests.into_iter().collect();
    let transactions = authority_store
        .multi_get_transaction_blocks(&all_tx_digests)?
        .into_iter()
        .map(|maybe_transaction| {
            maybe_transaction.ok_or_else(|| StorageError::custom("missing transaction"))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let effects = authority_store
        .multi_get_executed_effects(&all_tx_digests)?
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
        committed_subdag: committed_subdag,
        // checkpoint_contents: contents,
        transactions: full_transactions,
    };

    Ok(checkpoint_data)
}

pub struct RpcIndexStore {
    tables: IndexStoreTables,
    // pending_updates: Mutex<BTreeMap<u64, typed_store::rocks::DBBatch>>,
}

impl RpcIndexStore {
    /// Given the provided directory, construct the path to the db
    fn db_path(dir: &Path) -> PathBuf {
        dir.join("rpc-index")
    }

    pub async fn new(
        dir: &Path,
        authority_store: &AuthorityStore,
        commit_store: &CommitStore,
        // index_config: Option<&RpcIndexInitConfig>,
    ) -> Self {
        let path = Self::db_path(dir);

        let tables = {
            let tables = IndexStoreTables::open(&path);

            tables
        };

        Self {
            tables,
            // pending_updates: Default::default(),
        }
    }

    pub fn new_without_init(dir: &Path) -> Self {
        let path = Self::db_path(dir);
        let tables = IndexStoreTables::open(path);

        Self {
            tables,
            // pending_updates: Default::default(),
        }
    }

    // TODO: prune
    // pub fn prune(
    //     &self,
    //     pruned_checkpoint_watermark: u64,
    //     checkpoint_contents_to_prune: &[CheckpointContents],
    // ) -> Result<(), StorageError> {
    //     self.tables
    //         .prune(pruned_checkpoint_watermark, checkpoint_contents_to_prune)
    // }

    /// Index a checkpoint and stage the index updated in `pending_updates`.
    ///
    /// Updates will not be committed to the database until `commit_update_for_checkpoint` is
    /// called.
    #[tracing::instrument(
        skip_all,
        fields(checkpoint = checkpoint.committed_subdag.commit_ref.index)
    )]
    pub fn index_checkpoint(&self, checkpoint: &CheckpointData) {
        let sequence_number = checkpoint.committed_subdag.commit_ref.index;
        self.tables.index_checkpoint(checkpoint).expect("db error");
    }

    pub fn get_epoch_info(&self, epoch: EpochId) -> Result<Option<EpochInfo>, StorageError> {
        Ok(self.tables.get_epoch_info(epoch))
    }

    pub fn get_transaction_info(
        &self,
        digest: &TransactionDigest,
    ) -> Result<Option<TransactionInfo>, StorageError> {
        Ok(self.tables.get_transaction_info(digest))
    }

    pub fn owner_iter(
        &self,
        owner: SomaAddress,
        object_type: Option<ObjectType>,
        cursor: Option<OwnerIndexKey>,
    ) -> Result<
        impl Iterator<Item = Result<(OwnerIndexKey, OwnerIndexInfo), StorageError>> + '_,
        StorageError,
    > {
        self.tables.owner_iter(owner, object_type, cursor)
    }

    pub fn get_balance(
        &self,
        owner: &SomaAddress,
    ) -> Result<Option<BalanceIndexInfo>, StorageError> {
        Ok(self.tables.get_balance(owner))
    }

    pub fn balance_iter(
        &self,
        owner: SomaAddress,
        cursor: Option<BalanceKey>,
    ) -> Result<
        impl Iterator<Item = Result<(BalanceKey, BalanceIndexInfo), StorageError>> + '_,
        StorageError,
    > {
        self.tables.balance_iter(owner, cursor)
    }
}

struct RpcParLiveObjectSetIndexer<'a> {
    tables: &'a IndexStoreTables,
    batch_size_limit: usize,
}

struct RpcLiveObjectIndexer<'a> {
    tables: &'a IndexStoreTables,

    balance_changes: HashMap<BalanceKey, BalanceIndexInfo>,
    batch_size_limit: usize,
}

impl<'a> ParMakeLiveObjectIndexer for RpcParLiveObjectSetIndexer<'a> {
    type ObjectIndexer = RpcLiveObjectIndexer<'a>;

    fn make_live_object_indexer(&self) -> Self::ObjectIndexer {
        RpcLiveObjectIndexer {
            tables: self.tables,
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
                self.tables.owner.write().insert(owner_key, owner_info);

                if let Some(value) = object.as_coin() {
                    let balance_key = BalanceKey { owner };
                    let balance_info = BalanceIndexInfo::from(value);
                    self.balance_changes
                        .entry(balance_key)
                        .or_default()
                        .merge_delta(&balance_info);

                    if self.balance_changes.len() >= BALANCE_FLUSH_THRESHOLD {
                        self.tables
                            .apply_balance_changes(std::mem::take(&mut self.balance_changes))?;
                    }
                }
            }

            Owner::Shared { .. } | Owner::Immutable => {}
        }

        Ok(())
    }

    fn finish(mut self) -> Result<(), StorageError> {
        self.tables
            .apply_balance_changes(std::mem::take(&mut self.balance_changes))?;

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
