use crate::rpc_index::RpcIndexStore;
use crate::store_tables::{AuthorityPerpetualTables, AuthorityPrunerTables, StoreObject};
use anyhow::anyhow;
use bincode::Options;
use once_cell::sync::Lazy;
use std::cmp::{max, min};
use std::collections::{BTreeSet, HashMap};
use std::sync::atomic::AtomicU64;
use std::sync::{Mutex, Weak};
use std::time::{SystemTime, UNIX_EPOCH};
use std::{sync::Arc, time::Duration};
use store::rocksdb::compaction_filter::Decision;
use store::rocksdb::LiveFile;
use store::{Map, TypedStoreError};
use tokio::sync::oneshot::{self, Sender};
use tokio::time::Instant;
use tracing::{debug, error, info, warn};
use types::committee::EpochId;
use types::config::node_config::AuthorityStorePruningConfig;
use types::consensus::commit::{CommitDigest, CommitIndex, CommittedSubDag};
use types::consensus::output::ConsensusOutputAPI;
use types::consensus::ConsensusTransactionKind;
use types::effects::{TransactionEffects, TransactionEffectsAPI as _};
use types::envelope::Message as _;
use types::object::{ObjectID, Version};
use types::storage::ObjectKey;

static PERIODIC_PRUNING_TABLES: Lazy<BTreeSet<String>> = Lazy::new(|| {
    [
        "objects",
        "effects",
        "transactions",
        "executed_effects",
        "executed_transactions_to_commit",
    ]
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
    pub commit_id: Arc<AtomicU64>,
}

static MIN_PRUNING_TICK_DURATION_MS: u64 = 10 * 1000;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PruningMode {
    Objects,
    Commits,
}

impl AuthorityStorePruner {
    /// prunes old versions of objects based on transaction effects
    async fn prune_objects(
        transaction_effects: Vec<TransactionEffects>,
        perpetual_db: &Arc<AuthorityPerpetualTables>,
        pruner_db: Option<&Arc<AuthorityPrunerTables>>,
        commit_index: CommitIndex,
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
            debug!(
                "Pruning object {:?} versions {:?} - {:?}",
                object_id, min_version, max_version
            );
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

        perpetual_db.set_highest_pruned_commit(&mut wb, commit_index)?;

        if let Some(batch) = pruner_db_wb {
            batch.write()?;
        }
        wb.write()?;
        Ok(())
    }

    fn prune_commits(
        perpetual_db: &Arc<AuthorityPerpetualTables>,
        commit_db: &Arc<CommitStore>,
        rpc_index: Option<&RpcIndexStore>,
        commit_index: CommitIndex,
        commits_to_prune: Vec<CommitDigest>,
        commit_content_to_prune: Vec<CommittedSubDag>,
        effects_to_prune: &Vec<TransactionEffects>,
    ) -> anyhow::Result<()> {
        let mut perpetual_batch = perpetual_db.objects.batch();
        let transactions: Vec<_> = commit_content_to_prune
            .iter()
            .flat_map(|content| {
                content
                    .transactions()
                    .iter()
                    .flat_map(|(_, authority_transactions)| {
                        authority_transactions
                            .iter()
                            .filter_map(|(_, transaction)| {
                                if let ConsensusTransactionKind::UserTransaction(cert_tx) =
                                    &transaction.kind
                                {
                                    Some(*cert_tx.digest())
                                } else {
                                    None
                                }
                            })
                    })
                    .collect::<Vec<_>>()
            })
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

        let mut commits_batch = commit_db.certified_commits.batch();

        let commit_content_digests = commit_content_to_prune
            .iter()
            .map(|ckpt| ckpt.commit_ref.digest);
        let commit_content_indices = commit_content_to_prune
            .iter()
            .map(|ckpt| ckpt.commit_ref.index);
        commits_batch.delete_batch(&commit_db.certified_commits, commit_content_indices)?;
        commits_batch.delete_batch(
            &commit_db.commit_index_by_digest,
            commit_content_digests.clone(),
        )?;
        commits_batch.delete_batch(&commit_db.commit_by_digest, commits_to_prune)?;
        commits_batch.delete_batch(
            &commit_db.effects_digests_by_commit_digest,
            commit_content_digests,
        )?;

        commits_batch.insert_batch(
            &commit_db.watermarks,
            [(
                &CommitWatermark::HighestPruned,
                &(commit_index, CommitDigest::random()),
            )],
        )?;

        if let Some(rpc_index) = rpc_index {
            rpc_index.prune(commit_index.into(), &transactions)?;
        }
        perpetual_batch.write()?;
        commits_batch.write()?;

        Ok(())
    }

    /// Prunes old data based on effects from all commits from epochs eligible for pruning
    pub async fn prune_objects_for_eligible_epochs(
        perpetual_db: &Arc<AuthorityPerpetualTables>,
        commit_store: &Arc<CommitStore>,
        rpc_index: Option<&RpcIndexStore>,
        pruner_db: Option<&Arc<AuthorityPrunerTables>>,
        config: AuthorityStorePruningConfig,
        epoch_duration_ms: u64,
    ) -> anyhow::Result<()> {
        let (mut max_eligible_commit_index, epoch_id) = commit_store
            .get_highest_executed_commit()?
            .map(|c| (c.commit_ref.index, c.epoch()))
            .unwrap_or_default();
        let pruned_commit_index = perpetual_db
            .get_highest_pruned_commit()?
            .unwrap_or_default();
        if config.smooth && config.num_epochs_to_retain > 0 {
            max_eligible_commit_index = Self::smoothed_max_eligible_commit_index(
                commit_store,
                max_eligible_commit_index,
                pruned_commit_index,
                epoch_id,
                epoch_duration_ms,
                config.num_epochs_to_retain,
            )?;
        }
        Self::prune_for_eligible_epochs(
            perpetual_db,
            commit_store,
            rpc_index,
            pruner_db,
            PruningMode::Objects,
            config.num_epochs_to_retain,
            pruned_commit_index,
            max_eligible_commit_index,
            config,
        )
        .await
    }

    pub async fn prune_commits_for_eligible_epochs(
        perpetual_db: &Arc<AuthorityPerpetualTables>,
        commit_store: &Arc<CommitStore>,
        rpc_index: Option<&RpcIndexStore>,
        pruner_db: Option<&Arc<AuthorityPrunerTables>>,
        config: AuthorityStorePruningConfig,
        epoch_duration_ms: u64,
        pruner_watermarks: &Arc<PrunerWatermarks>,
    ) -> anyhow::Result<()> {
        let pruned_commit_index = commit_store.get_highest_pruned_commit_index()?.unwrap_or(0);
        let (mut max_eligible_commit, epoch_id) = commit_store
            .get_highest_executed_commit()?
            .map(|c| (c.commit_ref.index, c.epoch()))
            .unwrap_or_default();
        if config.num_epochs_to_retain != u64::MAX {
            max_eligible_commit = min(
                max_eligible_commit,
                perpetual_db
                    .get_highest_pruned_commit()?
                    .unwrap_or_default(),
            );
        }
        if config.smooth {
            if let Some(num_epochs_to_retain) = config.num_epochs_to_retain_for_commits {
                max_eligible_commit = Self::smoothed_max_eligible_commit_index(
                    commit_store,
                    max_eligible_commit,
                    pruned_commit_index,
                    epoch_id,
                    epoch_duration_ms,
                    num_epochs_to_retain,
                )?;
            }
        }
        debug!("Max eligible commit {}", max_eligible_commit);
        Self::prune_for_eligible_epochs(
            perpetual_db,
            commit_store,
            rpc_index,
            pruner_db,
            PruningMode::Commits,
            config
                .num_epochs_to_retain_for_commits
                .ok_or_else(|| anyhow!("config value not set"))?,
            pruned_commit_index,
            max_eligible_commit,
            config.clone(),
        )
        .await?;

        if let Some(num_epochs_to_retain) = config.num_epochs_to_retain_for_commits {
            Self::update_pruning_watermarks(commit_store, num_epochs_to_retain, pruner_watermarks)?;
        }
        Ok(())
    }

    /// Prunes old object versions based on effects from all commits from epochs eligible for pruning
    pub async fn prune_for_eligible_epochs(
        perpetual_db: &Arc<AuthorityPerpetualTables>,
        commit_store: &Arc<CommitStore>,
        rpc_index: Option<&RpcIndexStore>,
        pruner_db: Option<&Arc<AuthorityPrunerTables>>,
        mode: PruningMode,
        num_epochs_to_retain: u64,
        starting_commit_index: CommitIndex,
        max_eligible_commit: CommitIndex,
        config: AuthorityStorePruningConfig,
    ) -> anyhow::Result<()> {
        let mut commit_index = starting_commit_index;
        let current_epoch = commit_store
            .get_highest_executed_commit()?
            .map(|c| c.epoch())
            .unwrap_or_default();

        let mut commits_to_prune = vec![];
        let mut commit_content_to_prune = vec![];
        let mut effects_to_prune = vec![];

        loop {
            let Some(commit) = commit_store.certified_commits.get(&(commit_index + 1))? else {
                break;
            };
            // Skipping because  commit's epoch or commit number is too new.
            // We have to respect the highest executed commit watermark (including the watermark itself)
            // because there might be parts of the system that still require access to old object versions
            // (i.e. state accumulator).
            if (current_epoch < commit.epoch() + num_epochs_to_retain)
                || (commit.commit_ref.index >= max_eligible_commit)
            {
                break;
            }
            commit_index = commit.commit_ref.index;

            let content = commit_store
                .get_commit_by_digest(&commit.commit_ref.digest)?
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "commit content data is missing: {}",
                        commit.commit_ref.index
                    )
                })?;
            let content_effects = commit_store
                .get_effects_by_commit_digest(&commit.commit_ref.digest)?
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "commit effects data is missing: {}",
                        commit.commit_ref.index
                    )
                })?;
            let effects = perpetual_db.effects.multi_get(content_effects)?;

            info!("scheduling pruning for commit {:?}", commit_index);
            commits_to_prune.push(commit.commit_ref.digest);
            commit_content_to_prune.push(content);
            effects_to_prune.extend(effects.into_iter().flatten());

            if effects_to_prune.len() >= config.max_transactions_in_batch
                || commits_to_prune.len() >= config.max_commits_in_batch
            {
                match mode {
                    PruningMode::Objects => {
                        Self::prune_objects(
                            effects_to_prune,
                            perpetual_db,
                            pruner_db,
                            commit_index,
                            !config.killswitch_tombstone_pruning,
                        )
                        .await?
                    }
                    PruningMode::Commits => Self::prune_commits(
                        perpetual_db,
                        commit_store,
                        rpc_index,
                        commit_index,
                        commits_to_prune,
                        commit_content_to_prune,
                        &effects_to_prune,
                    )?,
                };
                commits_to_prune = vec![];
                commit_content_to_prune = vec![];
                effects_to_prune = vec![];
                // yield back to the tokio runtime. Prevent potential halt of other tasks
                tokio::task::yield_now().await;
            }
        }

        if !commits_to_prune.is_empty() {
            match mode {
                PruningMode::Objects => {
                    Self::prune_objects(
                        effects_to_prune,
                        perpetual_db,
                        pruner_db,
                        commit_index,
                        !config.killswitch_tombstone_pruning,
                    )
                    .await?
                }
                PruningMode::Commits => Self::prune_commits(
                    perpetual_db,
                    commit_store,
                    rpc_index,
                    commit_index,
                    commits_to_prune,
                    commit_content_to_prune,
                    &effects_to_prune,
                )?,
            };
        }
        Ok(())
    }

    fn update_pruning_watermarks(
        commit_store: &Arc<CommitStore>,
        num_epochs_to_retain: u64,
        pruning_watermark: &Arc<PrunerWatermarks>,
    ) -> anyhow::Result<bool> {
        use std::sync::atomic::Ordering;
        let current_watermark = pruning_watermark.epoch_id.load(Ordering::Relaxed);
        let current_epoch_id = commit_store
            .get_highest_executed_commit()?
            .map(|c| c.epoch())
            .unwrap_or_default();
        if current_epoch_id < num_epochs_to_retain {
            return Ok(false);
        }
        let target_epoch_id = current_epoch_id - num_epochs_to_retain;
        let commit = commit_store.get_epoch_last_commit(target_epoch_id)?;

        let new_watermark = target_epoch_id + 1;
        if current_watermark == new_watermark {
            return Ok(false);
        }
        info!("relocation: setting epoch watermark to {}", new_watermark);
        pruning_watermark
            .epoch_id
            .store(new_watermark, Ordering::Relaxed);
        if let Some(commit) = commit {
            let commit_id = commit.commit_ref.index;
            info!("relocation: setting commit watermark to {}", commit_id);
            pruning_watermark
                .commit_id
                .store(commit_id.into(), Ordering::Relaxed);
        }
        Ok(true)
    }

    fn compact_next_sst_file(
        perpetual_db: Arc<AuthorityPerpetualTables>,
        delay_days: usize,
        last_processed: Arc<Mutex<HashMap<String, SystemTime>>>,
    ) -> anyhow::Result<Option<LiveFile>> {
        let db_path = perpetual_db.objects.db.path_for_pruning();
        let mut state = last_processed
            .lock()
            .expect("failed to obtain a lock for last processed SST files");
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

    fn smoothed_max_eligible_commit_index(
        commit_store: &Arc<CommitStore>,
        mut max_eligible_commit: CommitIndex,
        pruned_commit: CommitIndex,
        epoch_id: EpochId,
        epoch_duration_ms: u64,
        num_epochs_to_retain: u64,
    ) -> anyhow::Result<CommitIndex> {
        if epoch_id < num_epochs_to_retain {
            return Ok(0);
        }
        let last_commit_in_epoch = commit_store
            .get_epoch_last_commit(epoch_id - num_epochs_to_retain)?
            .map(|commit| commit.commit_ref.index)
            .unwrap_or_default();
        max_eligible_commit = max_eligible_commit.min(last_commit_in_epoch);
        if max_eligible_commit == 0 {
            return Ok(max_eligible_commit);
        }
        let num_intervals: u32 = epoch_duration_ms
            .checked_div(Self::pruning_tick_duration_ms(epoch_duration_ms))
            .unwrap_or(1)
            .try_into()
            .unwrap_or(1);
        let delta = max_eligible_commit
            .checked_sub(pruned_commit)
            .unwrap_or_default()
            .checked_div(num_intervals)
            .unwrap_or(1);
        Ok(pruned_commit + delta)
    }

    fn setup_pruning(
        config: AuthorityStorePruningConfig,
        epoch_duration_ms: u64,
        perpetual_db: Arc<AuthorityPerpetualTables>,
        commit_store: Arc<CommitStore>,
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

        {
            let mut commits_prune_interval =
                tokio::time::interval_at(Instant::now() + pruning_initial_delay, tick_duration);
            let mut indexes_prune_interval =
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
                            if let Err(err) = Self::prune_objects_for_eligible_epochs(&perpetual_db, &commit_store, rpc_index.as_deref(), pruner_db.as_ref(), config.clone(), epoch_duration_ms).await {
                                error!("Failed to prune objects: {:?}", err);
                            }
                        },
                        _ = commits_prune_interval.tick(), if !matches!(config.num_epochs_to_retain_for_commits, None | Some(u64::MAX) | Some(0)) => {
                            if let Err(err) = Self::prune_commits_for_eligible_epochs(&perpetual_db, &commit_store, rpc_index.as_deref(), pruner_db.as_ref(), config.clone(), epoch_duration_ms, &pruner_watermarks).await {
                                error!("Failed to prune commits: {:?}", err);
                            }
                        },
                        _ = &mut recv => break,
                    }
                }
            });
        }
        sender
    }

    pub fn new(
        perpetual_db: Arc<AuthorityPerpetualTables>,
        commit_store: Arc<CommitStore>,
        rpc_index: Option<Arc<RpcIndexStore>>,
        mut pruning_config: AuthorityStorePruningConfig,
        is_validator: bool,
        epoch_duration_ms: u64,
        pruner_db: Option<Arc<AuthorityPrunerTables>>,
        pruner_watermarks: Arc<PrunerWatermarks>, // used by tidehunter relocation filters
    ) -> Self {
        if pruning_config.num_epochs_to_retain > 0 && pruning_config.num_epochs_to_retain < u64::MAX
        {
            warn!("Using objects pruner with num_epochs_to_retain = {} can lead to performance issues", pruning_config.num_epochs_to_retain);
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
                commit_store,
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
        Self {
            db: Arc::downgrade(&db),
        }
    }
    pub fn filter(&mut self, key: &[u8], value: &[u8]) -> anyhow::Result<Decision> {
        let ObjectKey(object_id, version) = bincode::DefaultOptions::new()
            .with_big_endian()
            .with_fixint_encoding()
            .deserialize(key)?;
        let object: StoreObject = bcs::from_bytes(value)?;
        if matches!(object, StoreObject::Value(_)) {
            if let Some(db) = self.db.upgrade() {
                match db.object_tombstones.get(&object_id)? {
                    Some(gc_version) => {
                        if version <= gc_version {
                            return Ok(Decision::Remove);
                        }
                    }
                    None => {}
                }
            }
        }
        Ok(Decision::Keep)
    }
}
