// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use crate::authority_store_tables::{AuthorityPerpetualTables, AuthorityPrunerTables, StoreObject};
use crate::checkpoints::{CheckpointStore, CheckpointWatermark};
use crate::rpc_index::RpcIndexStore;
use anyhow::anyhow;
use bincode::Options;
use once_cell::sync::Lazy;
use std::cmp::{max, min};
use std::collections::{BTreeSet, HashMap};
use std::sync::atomic::AtomicU64;
use std::sync::{Mutex, Weak};
use std::time::{SystemTime, UNIX_EPOCH};
use std::{sync::Arc, time::Duration};
use store::rocksdb::LiveFile;
use store::rocksdb::compaction_filter::Decision;
use store::{Map, TypedStoreError};
use tokio::sync::oneshot::{self, Sender};
use tokio::time::Instant;
use tracing::{debug, error, info, warn};
use types::checkpoints::{CheckpointContents, CheckpointSequenceNumber};
use types::committee::EpochId;
use types::config::node_config::AuthorityStorePruningConfig;
use types::digests::CheckpointDigest;
use types::effects::{TransactionEffects, TransactionEffectsAPI as _};
use types::envelope::Message as _;
use types::object::{ObjectID, Version};
use types::storage::ObjectKey;

static PERIODIC_PRUNING_TABLES: Lazy<BTreeSet<String>> = Lazy::new(|| {
    ["objects", "effects", "transactions", "executed_effects", "executed_transactions_to_commit"]
        .into_iter()
        .map(|cf| cf.to_string())
        .collect()
});
pub const EPOCH_DURATION_MS_FOR_TESTING: u64 = 24 * 60 * 60 * 1000;
pub struct AuthorityStorePruner {
    _objects_pruner_cancel_handle: oneshot::Sender<()>,
}

#[derive(Default)]
pub struct PrunerWatermarks {
    pub epoch_id: Arc<AtomicU64>,
    pub checkpoint_id: Arc<AtomicU64>,
}

static MIN_PRUNING_TICK_DURATION_MS: u64 = 10 * 1000;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PruningMode {
    Objects,
    Checkpoints,
}

impl AuthorityStorePruner {
    /// prunes old versions of objects based on transaction effects
    async fn prune_objects(
        transaction_effects: Vec<TransactionEffects>,
        perpetual_db: &Arc<AuthorityPerpetualTables>,
        pruner_db: Option<&Arc<AuthorityPrunerTables>>,
        checkpoint_number: CheckpointSequenceNumber,
        enable_pruning_tombstones: bool,
    ) -> anyhow::Result<()> {
        let mut wb = perpetual_db.objects.batch();
        let mut pruner_db_wb = pruner_db.map(|db| db.object_tombstones.batch());

        // Collect objects keys that need to be deleted from `transaction_effects`.
        let mut live_object_keys_to_prune = vec![];
        let mut object_tombstones_to_prune = vec![];
        for effects in &transaction_effects {
            for (object_id, seq_number) in effects.modified_at_versions() {
                live_object_keys_to_prune.push(ObjectKey(object_id, seq_number));
            }

            if enable_pruning_tombstones {
                for deleted_object_key in effects.all_tombstones() {
                    object_tombstones_to_prune
                        .push(ObjectKey(deleted_object_key.0, deleted_object_key.1));
                }
            }
        }

        let mut updates: HashMap<ObjectID, (Version, Version)> = HashMap::new();
        for ObjectKey(object_id, seq_number) in live_object_keys_to_prune {
            updates
                .entry(object_id)
                .and_modify(|range| *range = (min(range.0, seq_number), max(range.1, seq_number)))
                .or_insert((seq_number, seq_number));
        }

        for (object_id, (min_version, max_version)) in updates {
            debug!("Pruning object {:?} versions {:?} - {:?}", object_id, min_version, max_version);
            match pruner_db_wb {
                Some(ref mut batch) => {
                    batch.insert_batch(
                        &pruner_db.expect("invariant checked").object_tombstones,
                        std::iter::once((object_id, max_version)),
                    )?;
                }
                None => {
                    let start_range = ObjectKey(object_id, min_version);
                    let end_range = ObjectKey(object_id, (max_version.value() + 1).into());
                    wb.schedule_delete_range(&perpetual_db.objects, &start_range, &end_range)?;
                }
            }
        }

        // When enable_pruning_tombstones is enabled, instead of using range deletes, we need to do a scan of all the keys
        // for the deleted objects and then do point deletes to delete all the existing keys. This is because to improve read
        // performance, we set `ignore_range_deletions` on all read options, and using range delete to delete tombstones
        // may leak object (imagine a tombstone is compacted away, but earlier version is still not). Using point deletes
        // guarantees that all earlier versions are deleted in the database.
        if !object_tombstones_to_prune.is_empty() {
            let mut object_keys_to_delete = vec![];
            for ObjectKey(object_id, seq_number) in object_tombstones_to_prune {
                for result in perpetual_db.objects.safe_iter_with_bounds(
                    Some(ObjectKey(object_id, Version::MIN)),
                    Some(ObjectKey(object_id, seq_number.next())),
                ) {
                    let (object_key, _) = result?;
                    assert_eq!(object_key.0, object_id);
                    object_keys_to_delete.push(object_key);
                }
            }

            wb.delete_batch(&perpetual_db.objects, object_keys_to_delete)?;
        }

        perpetual_db.set_highest_pruned_checkpoint(&mut wb, checkpoint_number)?;

        if let Some(batch) = pruner_db_wb {
            batch.write()?;
        }
        wb.write()?;
        Ok(())
    }

    fn prune_checkpoints(
        perpetual_db: &Arc<AuthorityPerpetualTables>,
        checkpoint_db: &Arc<CheckpointStore>,
        rpc_index: Option<&RpcIndexStore>,
        checkpoint_number: CheckpointSequenceNumber,
        checkpoints_to_prune: Vec<CheckpointDigest>,
        checkpoint_content_to_prune: Vec<CheckpointContents>,
        effects_to_prune: &Vec<TransactionEffects>,
    ) -> anyhow::Result<()> {
        let mut perpetual_batch = perpetual_db.objects.batch();
        let transactions: Vec<_> = checkpoint_content_to_prune
            .iter()
            .flat_map(|content| content.iter().map(|tx| tx.transaction))
            .collect();

        perpetual_batch.delete_batch(&perpetual_db.transactions, transactions.iter())?;
        perpetual_batch.delete_batch(&perpetual_db.executed_effects, transactions.iter())?;

        let mut effect_digests = vec![];
        for effects in effects_to_prune {
            let effects_digest = effects.digest();
            debug!("Pruning effects {:?}", effects_digest);
            effect_digests.push(effects_digest);
        }

        perpetual_batch.delete_batch(&perpetual_db.effects, effect_digests)?;

        let mut checkpoints_batch = checkpoint_db.tables.certified_checkpoints.batch();

        let checkpoint_content_digests =
            checkpoint_content_to_prune.iter().map(|ckpt| ckpt.digest());
        checkpoints_batch.delete_batch(
            &checkpoint_db.tables.checkpoint_content,
            checkpoint_content_digests.clone(),
        )?;
        checkpoints_batch.delete_batch(
            &checkpoint_db.tables.checkpoint_sequence_by_contents_digest,
            checkpoint_content_digests,
        )?;

        checkpoints_batch
            .delete_batch(&checkpoint_db.tables.checkpoint_by_digest, checkpoints_to_prune)?;

        checkpoints_batch.insert_batch(
            &checkpoint_db.tables.watermarks,
            [(
                &CheckpointWatermark::HighestPruned,
                &(checkpoint_number, CheckpointDigest::random()),
            )],
        )?;

        if let Some(rpc_index) = rpc_index {
            rpc_index.prune(checkpoint_number, &checkpoint_content_to_prune)?;
        }
        perpetual_batch.write()?;
        checkpoints_batch.write()?;

        Ok(())
    }

    /// Prunes old data based on effects from all checkpoints from epochs eligible for pruning
    pub async fn prune_objects_for_eligible_epochs(
        perpetual_db: &Arc<AuthorityPerpetualTables>,
        checkpoint_store: &Arc<CheckpointStore>,
        rpc_index: Option<&RpcIndexStore>,
        pruner_db: Option<&Arc<AuthorityPrunerTables>>,
        config: AuthorityStorePruningConfig,
        epoch_duration_ms: u64,
    ) -> anyhow::Result<()> {
        let (mut max_eligible_checkpoint_number, epoch_id) = checkpoint_store
            .get_highest_executed_checkpoint()?
            .map(|c| (*c.sequence_number(), c.epoch))
            .unwrap_or_default();
        let pruned_checkpoint_number =
            perpetual_db.get_highest_pruned_checkpoint()?.unwrap_or_default();
        if config.smooth && config.num_epochs_to_retain > 0 {
            max_eligible_checkpoint_number = Self::smoothed_max_eligible_checkpoint_number(
                checkpoint_store,
                max_eligible_checkpoint_number,
                pruned_checkpoint_number,
                epoch_id,
                epoch_duration_ms,
                config.num_epochs_to_retain,
            )?;
        }
        Self::prune_for_eligible_epochs(
            perpetual_db,
            checkpoint_store,
            rpc_index,
            pruner_db,
            PruningMode::Objects,
            config.num_epochs_to_retain,
            pruned_checkpoint_number,
            max_eligible_checkpoint_number,
            config,
        )
        .await
    }

    pub async fn prune_checkpoints_for_eligible_epochs(
        perpetual_db: &Arc<AuthorityPerpetualTables>,
        checkpoint_store: &Arc<CheckpointStore>,
        rpc_index: Option<&RpcIndexStore>,
        pruner_db: Option<&Arc<AuthorityPrunerTables>>,
        config: AuthorityStorePruningConfig,

        epoch_duration_ms: u64,
        pruner_watermarks: &Arc<PrunerWatermarks>,
    ) -> anyhow::Result<()> {
        let pruned_checkpoint_number =
            checkpoint_store.get_highest_pruned_checkpoint_seq_number()?.unwrap_or(0);
        let (mut max_eligible_checkpoint, epoch_id) = checkpoint_store
            .get_highest_executed_checkpoint()?
            .map(|c| (*c.sequence_number(), c.epoch))
            .unwrap_or_default();
        if config.num_epochs_to_retain != u64::MAX {
            max_eligible_checkpoint = min(
                max_eligible_checkpoint,
                perpetual_db.get_highest_pruned_checkpoint()?.unwrap_or_default(),
            );
        }
        if config.smooth {
            if let Some(num_epochs_to_retain) = config.num_epochs_to_retain_for_checkpoints {
                max_eligible_checkpoint = Self::smoothed_max_eligible_checkpoint_number(
                    checkpoint_store,
                    max_eligible_checkpoint,
                    pruned_checkpoint_number,
                    epoch_id,
                    epoch_duration_ms,
                    num_epochs_to_retain,
                )?;
            }
        }
        debug!("Max eligible checkpoint {}", max_eligible_checkpoint);
        Self::prune_for_eligible_epochs(
            perpetual_db,
            checkpoint_store,
            rpc_index,
            pruner_db,
            PruningMode::Checkpoints,
            config
                .num_epochs_to_retain_for_checkpoints()
                .ok_or_else(|| anyhow!("config value not set"))?,
            pruned_checkpoint_number,
            max_eligible_checkpoint,
            config.clone(),
        )
        .await?;

        if let Some(num_epochs_to_retain) = config.num_epochs_to_retain_for_checkpoints() {
            Self::update_pruning_watermarks(
                checkpoint_store,
                num_epochs_to_retain,
                pruner_watermarks,
            )?;
        }
        Ok(())
    }

    /// Prunes old object versions based on effects from all checkpoints from epochs eligible for pruning
    pub async fn prune_for_eligible_epochs(
        perpetual_db: &Arc<AuthorityPerpetualTables>,
        checkpoint_store: &Arc<CheckpointStore>,
        rpc_index: Option<&RpcIndexStore>,
        pruner_db: Option<&Arc<AuthorityPrunerTables>>,
        mode: PruningMode,
        num_epochs_to_retain: u64,
        starting_checkpoint_number: CheckpointSequenceNumber,
        max_eligible_checkpoint: CheckpointSequenceNumber,
        config: AuthorityStorePruningConfig,
    ) -> anyhow::Result<()> {
        let mut checkpoint_number = starting_checkpoint_number;
        let current_epoch = checkpoint_store
            .get_highest_executed_checkpoint()?
            .map(|c| c.epoch())
            .unwrap_or_default();

        let mut checkpoints_to_prune = vec![];
        let mut checkpoint_content_to_prune = vec![];
        let mut effects_to_prune = vec![];

        loop {
            let Some(ckpt) =
                checkpoint_store.tables.certified_checkpoints.get(&(checkpoint_number + 1))?
            else {
                break;
            };
            let checkpoint = ckpt.into_inner();
            // Skipping because  checkpoint's epoch or checkpoint number is too new.
            // We have to respect the highest executed checkpoint watermark (including the watermark itself)
            // because there might be parts of the system that still require access to old object versions
            // (i.e. state accumulator).
            if (current_epoch < checkpoint.epoch() + num_epochs_to_retain)
                || (*checkpoint.sequence_number() >= max_eligible_checkpoint)
            {
                break;
            }
            checkpoint_number = *checkpoint.sequence_number();

            let content = checkpoint_store
                .get_checkpoint_contents(&checkpoint.content_digest)?
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "checkpoint content data is missing: {}",
                        checkpoint.sequence_number
                    )
                })?;
            let effects = perpetual_db.effects.multi_get(content.iter().map(|tx| tx.effects))?;

            info!("scheduling pruning for checkpoint {:?}", checkpoint_number);
            checkpoints_to_prune.push(*checkpoint.digest());
            checkpoint_content_to_prune.push(content);
            effects_to_prune.extend(effects.into_iter().flatten());

            if effects_to_prune.len() >= config.max_transactions_in_batch
                || checkpoints_to_prune.len() >= config.max_checkpoints_in_batch
            {
                match mode {
                    PruningMode::Objects => {
                        Self::prune_objects(
                            effects_to_prune,
                            perpetual_db,
                            pruner_db,
                            checkpoint_number,
                            !config.killswitch_tombstone_pruning,
                        )
                        .await?
                    }
                    PruningMode::Checkpoints => Self::prune_checkpoints(
                        perpetual_db,
                        checkpoint_store,
                        rpc_index,
                        checkpoint_number,
                        checkpoints_to_prune,
                        checkpoint_content_to_prune,
                        &effects_to_prune,
                    )?,
                };
                checkpoints_to_prune = vec![];
                checkpoint_content_to_prune = vec![];
                effects_to_prune = vec![];
                // yield back to the tokio runtime. Prevent potential halt of other tasks
                tokio::task::yield_now().await;
            }
        }

        if !checkpoints_to_prune.is_empty() {
            match mode {
                PruningMode::Objects => {
                    Self::prune_objects(
                        effects_to_prune,
                        perpetual_db,
                        pruner_db,
                        checkpoint_number,
                        !config.killswitch_tombstone_pruning,
                    )
                    .await?
                }
                PruningMode::Checkpoints => Self::prune_checkpoints(
                    perpetual_db,
                    checkpoint_store,
                    rpc_index,
                    checkpoint_number,
                    checkpoints_to_prune,
                    checkpoint_content_to_prune,
                    &effects_to_prune,
                )?,
            };
        }
        Ok(())
    }

    fn update_pruning_watermarks(
        checkpoint_store: &Arc<CheckpointStore>,
        num_epochs_to_retain: u64,
        pruning_watermark: &Arc<PrunerWatermarks>,
    ) -> anyhow::Result<bool> {
        use std::sync::atomic::Ordering;
        let current_watermark = pruning_watermark.epoch_id.load(Ordering::Relaxed);
        let current_epoch_id = checkpoint_store
            .get_highest_executed_checkpoint()?
            .map(|c| c.epoch)
            .unwrap_or_default();
        if current_epoch_id < num_epochs_to_retain {
            return Ok(false);
        }
        let target_epoch_id = current_epoch_id - num_epochs_to_retain;
        let checkpoint_id =
            checkpoint_store.get_epoch_last_checkpoint_seq_number(target_epoch_id)?;

        let new_watermark = target_epoch_id + 1;
        if current_watermark == new_watermark {
            return Ok(false);
        }
        info!("relocation: setting epoch watermark to {}", new_watermark);
        pruning_watermark.epoch_id.store(new_watermark, Ordering::Relaxed);
        if let Some(checkpoint_id) = checkpoint_id {
            info!("relocation: setting checkpoint watermark to {}", checkpoint_id);
            pruning_watermark.checkpoint_id.store(checkpoint_id, Ordering::Relaxed);
        }
        Ok(true)
    }

    fn compact_next_sst_file(
        perpetual_db: Arc<AuthorityPerpetualTables>,
        delay_days: usize,
        last_processed: Arc<Mutex<HashMap<String, SystemTime>>>,
    ) -> anyhow::Result<Option<LiveFile>> {
        let db_path = perpetual_db.objects.db.path_for_pruning();
        let mut state =
            last_processed.lock().expect("failed to obtain a lock for last processed SST files");
        let mut sst_file_for_compaction: Option<LiveFile> = None;
        let time_threshold =
            SystemTime::now() - Duration::from_secs(delay_days as u64 * 24 * 60 * 60);
        for sst_file in perpetual_db.objects.db.live_files()? {
            let file_path = db_path.join(sst_file.name.clone().trim_matches('/'));
            let last_modified = std::fs::metadata(file_path)?.modified()?;
            if !PERIODIC_PRUNING_TABLES.contains(&sst_file.column_family_name)
                || sst_file.level < 1
                || sst_file.start_key.is_none()
                || sst_file.end_key.is_none()
                || last_modified > time_threshold
                || state.get(&sst_file.name).unwrap_or(&UNIX_EPOCH) > &time_threshold
            {
                continue;
            }
            if let Some(candidate) = &sst_file_for_compaction {
                if candidate.size > sst_file.size {
                    continue;
                }
            }
            sst_file_for_compaction = Some(sst_file);
        }
        let Some(sst_file) = sst_file_for_compaction else {
            return Ok(None);
        };
        info!(
            "Manual compaction of sst file {:?}. Size: {:?}, level: {:?}",
            sst_file.name, sst_file.size, sst_file.level
        );
        perpetual_db.objects.compact_range_raw(
            &sst_file.column_family_name,
            sst_file.start_key.clone().unwrap(),
            sst_file.end_key.clone().unwrap(),
        )?;
        state.insert(sst_file.name.clone(), SystemTime::now());
        Ok(Some(sst_file))
    }

    fn pruning_tick_duration_ms(epoch_duration_ms: u64) -> u64 {
        min(epoch_duration_ms / 2, MIN_PRUNING_TICK_DURATION_MS)
    }

    fn smoothed_max_eligible_checkpoint_number(
        checkpoint_store: &Arc<CheckpointStore>,
        mut max_eligible_checkpoint: CheckpointSequenceNumber,
        pruned_checkpoint: CheckpointSequenceNumber,
        epoch_id: EpochId,
        epoch_duration_ms: u64,
        num_epochs_to_retain: u64,
    ) -> anyhow::Result<CheckpointSequenceNumber> {
        if epoch_id < num_epochs_to_retain {
            return Ok(0);
        }
        let last_checkpoint_in_epoch = checkpoint_store
            .get_epoch_last_checkpoint(epoch_id - num_epochs_to_retain)?
            .map(|checkpoint| checkpoint.sequence_number)
            .unwrap_or_default();
        max_eligible_checkpoint = max_eligible_checkpoint.min(last_checkpoint_in_epoch);
        if max_eligible_checkpoint == 0 {
            return Ok(max_eligible_checkpoint);
        }
        let num_intervals = epoch_duration_ms
            .checked_div(Self::pruning_tick_duration_ms(epoch_duration_ms))
            .unwrap_or(1);
        let delta = max_eligible_checkpoint
            .checked_sub(pruned_checkpoint)
            .unwrap_or_default()
            .checked_div(num_intervals)
            .unwrap_or(1);
        Ok(pruned_checkpoint + delta)
    }

    fn setup_pruning(
        config: AuthorityStorePruningConfig,
        epoch_duration_ms: u64,
        perpetual_db: Arc<AuthorityPerpetualTables>,
        checkpoint_store: Arc<CheckpointStore>,
        rpc_index: Option<Arc<RpcIndexStore>>,

        pruner_db: Option<Arc<AuthorityPrunerTables>>,

        pruner_watermarks: Arc<PrunerWatermarks>,
    ) -> Sender<()> {
        let (sender, mut recv) = tokio::sync::oneshot::channel();
        debug!(
            "Starting object pruning service with num_epochs_to_retain={}",
            config.num_epochs_to_retain
        );

        let tick_duration =
            Duration::from_millis(Self::pruning_tick_duration_ms(epoch_duration_ms));
        let pruning_initial_delay = if cfg!(msim) {
            Duration::from_millis(1)
        } else {
            Duration::from_secs(config.pruning_run_delay_seconds.unwrap_or(60 * 60))
        };
        let mut objects_prune_interval =
            tokio::time::interval_at(Instant::now() + pruning_initial_delay, tick_duration);

        let mut checkpoints_prune_interval =
            tokio::time::interval_at(Instant::now() + pruning_initial_delay, tick_duration);

        let perpetual_db_for_compaction = perpetual_db.clone();
        if let Some(delay_days) = config.periodic_compaction_threshold_days {
            tokio::spawn(async move {
                let last_processed = Arc::new(Mutex::new(HashMap::new()));
                loop {
                    let db = perpetual_db_for_compaction.clone();
                    let state = Arc::clone(&last_processed);
                    let result = tokio::task::spawn_blocking(move || {
                        Self::compact_next_sst_file(db, delay_days, state)
                    })
                    .await;
                    let mut sleep_interval_secs = 1;
                    match result {
                        Err(err) => error!("Failed to compact sst file: {:?}", err),
                        Ok(Err(err)) => error!("Failed to compact sst file: {:?}", err),
                        Ok(Ok(None)) => {
                            sleep_interval_secs = 3600;
                        }
                        _ => {}
                    }
                    tokio::time::sleep(Duration::from_secs(sleep_interval_secs)).await;
                }
            });
        }
        tokio::task::spawn(async move {
            loop {
                tokio::select! {
                    _ = objects_prune_interval.tick(), if config.num_epochs_to_retain != u64::MAX => {
                        if let Err(err) = Self::prune_objects_for_eligible_epochs(&perpetual_db, &checkpoint_store, rpc_index.as_deref(), pruner_db.as_ref(), config.clone(),  epoch_duration_ms).await {
                            error!("Failed to prune objects: {:?}", err);
                        }
                    },
                    _ = checkpoints_prune_interval.tick(), if !matches!(config.num_epochs_to_retain_for_checkpoints(), None | Some(u64::MAX) | Some(0)) => {
                        if let Err(err) = Self::prune_checkpoints_for_eligible_epochs(&perpetual_db, &checkpoint_store, rpc_index.as_deref(), pruner_db.as_ref(), config.clone(),  epoch_duration_ms, &pruner_watermarks).await {
                            error!("Failed to prune checkpoints: {:?}", err);
                        }
                    },

                    _ = &mut recv => break,
                }
            }
        });

        sender
    }

    pub fn new(
        perpetual_db: Arc<AuthorityPerpetualTables>,
        checkpoint_store: Arc<CheckpointStore>,
        rpc_index: Option<Arc<RpcIndexStore>>,
        mut pruning_config: AuthorityStorePruningConfig,
        is_validator: bool,
        epoch_duration_ms: u64,
        pruner_db: Option<Arc<AuthorityPrunerTables>>,
        pruner_watermarks: Arc<PrunerWatermarks>, // used by tidehunter relocation filters
    ) -> Self {
        if pruning_config.num_epochs_to_retain > 0 && pruning_config.num_epochs_to_retain < u64::MAX
        {
            warn!(
                "Using objects pruner with num_epochs_to_retain = {} can lead to performance issues",
                pruning_config.num_epochs_to_retain
            );
            if is_validator {
                warn!("Resetting to aggressive pruner.");
                pruning_config.num_epochs_to_retain = 0;
            } else {
                warn!("Consider using an aggressive pruner (num_epochs_to_retain = 0)");
            }
        }
        AuthorityStorePruner {
            _objects_pruner_cancel_handle: Self::setup_pruning(
                pruning_config,
                epoch_duration_ms,
                perpetual_db,
                checkpoint_store,
                rpc_index,
                pruner_db,
                pruner_watermarks,
            ),
        }
    }

    pub fn compact(perpetual_db: &Arc<AuthorityPerpetualTables>) -> Result<(), TypedStoreError> {
        perpetual_db.objects.compact_range(
            &ObjectKey(ObjectID::ZERO, Version::MIN),
            &ObjectKey(ObjectID::MAX, Version::MAX),
        )
    }
}

#[derive(Clone)]
pub struct ObjectsCompactionFilter {
    db: Weak<AuthorityPrunerTables>,
}

impl ObjectsCompactionFilter {
    pub fn new(db: Arc<AuthorityPrunerTables>) -> Self {
        Self { db: Arc::downgrade(&db) }
    }
    pub fn filter(&mut self, key: &[u8], value: &[u8]) -> anyhow::Result<Decision> {
        let ObjectKey(object_id, version) = bincode::DefaultOptions::new()
            .with_big_endian()
            .with_fixint_encoding()
            .deserialize(key)?;
        let object: StoreObject = bcs::from_bytes(value)?;
        if matches!(object, StoreObject::Value(_)) {
            if let Some(db) = self.db.upgrade() {
                if let Some(gc_version) = db.object_tombstones.get(&object_id)? {
                    if version <= gc_version {
                        return Ok(Decision::Remove);
                    }
                }
            }
        }
        Ok(Decision::Keep)
    }
}
