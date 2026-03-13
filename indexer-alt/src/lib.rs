// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use anyhow::Context;
use anyhow::Result;
use indexer_framework::Indexer;
use indexer_framework::postgres::Db;
use indexer_framework::pipeline::concurrent::ConcurrentConfig;
use indexer_framework::pipeline::concurrent::PrunerConfig;
use tokio::task::JoinHandle;
use tracing::warn;

pub mod handlers;
#[cfg(test)]
pub(crate) mod test_utils;

/// Configuration for which pipeline tiers get pruned.
///
/// Pipelines are classified into three tiers:
/// - **Tier A (KV content)**: `kv_checkpoints`, `kv_transactions`, `kv_objects`,
///   `kv_epoch_starts`, `kv_epoch_ends`. Data is duplicated in BigTable, so pruning
///   is safe once the KvLoader fallback is configured in GraphQL.
/// - **Tier B (Indexes)**: `tx_digests`, `tx_affected_*`, `tx_balance_changes`,
///   `tx_kinds`, `tx_calls`. No BigTable equivalent — pruning shrinks the queryable
///   window for historical index-backed queries.
/// - **Tier C (Never prune)**: `cp_sequence_numbers`, `obj_versions`, `obj_info`,
///   `coin_balance_buckets`, and all `soma_*` tables.
#[derive(Debug, Clone, Default)]
pub struct PruningConfig {
    /// Tier A: KV content tables (data duplicated in BigTable). Default: None (disabled).
    pub kv_pruner: Option<PrunerConfig>,

    /// Tier B: Index tables. Default: None (disabled).
    pub index_pruner: Option<PrunerConfig>,

    /// Optional BigTable watermark floor for zero-gap guarantee. When set, Tier A
    /// pipelines will not prune past this checkpoint, ensuring data exists in BigTable
    /// before it is removed from Postgres.
    pub bigtable_watermark: Option<Arc<AtomicU64>>,
}

/// Register all indexer pipelines on the given indexer instance.
///
/// Each handler implements the `Processor` trait (for transforming checkpoint data into rows)
/// and the postgres `Handler` trait (for committing rows to the database).
///
/// The `pruning` config controls which pipeline tiers have pruning enabled. By default
/// all pruning is disabled (Postgres grows unbounded).
pub async fn setup_indexer(indexer: &mut Indexer<Db>, pruning: PruningConfig) -> Result<()> {
    // Tier A: KV content tables — prunable, data in BigTable.
    // Inject external_watermark_floor for zero-gap guarantee.
    let kv_config = ConcurrentConfig {
        pruner: pruning.kv_pruner.map(|mut p| {
            p.external_watermark_floor = pruning.bigtable_watermark.clone();
            p
        }),
        ..Default::default()
    };

    // Tier B: Index tables — prunable with longer retention, no BigTable backup.
    let index_config = ConcurrentConfig {
        pruner: pruning.index_pruner,
        ..Default::default()
    };

    // Tier C: Never prune.
    let no_prune = ConcurrentConfig::default();

    // --- Tier C: Core mapping tables (never pruned) ---
    indexer
        .concurrent_pipeline(handlers::cp_sequence_numbers::CpSequenceNumbers, no_prune.clone())
        .await
        .context("Failed to register cp_sequence_numbers pipeline")?;

    // --- Tier A: KV content tables ---
    indexer
        .concurrent_pipeline(handlers::kv_checkpoints::KvCheckpoints, kv_config.clone())
        .await
        .context("Failed to register kv_checkpoints pipeline")?;

    indexer
        .concurrent_pipeline(handlers::kv_transactions::KvTransactions, kv_config.clone())
        .await
        .context("Failed to register kv_transactions pipeline")?;

    indexer
        .concurrent_pipeline(handlers::kv_objects::KvObjects, kv_config.clone())
        .await
        .context("Failed to register kv_objects pipeline")?;

    indexer
        .concurrent_pipeline(handlers::kv_epoch_starts::KvEpochStarts, kv_config.clone())
        .await
        .context("Failed to register kv_epoch_starts pipeline")?;

    indexer
        .concurrent_pipeline(handlers::kv_epoch_ends::KvEpochEnds, kv_config)
        .await
        .context("Failed to register kv_epoch_ends pipeline")?;

    // --- Tier B: Index tables ---
    indexer
        .concurrent_pipeline(handlers::tx_digests::TxDigests, index_config.clone())
        .await
        .context("Failed to register tx_digests pipeline")?;

    indexer
        .concurrent_pipeline(handlers::tx_affected_addresses::TxAffectedAddresses, index_config.clone())
        .await
        .context("Failed to register tx_affected_addresses pipeline")?;

    indexer
        .concurrent_pipeline(handlers::tx_affected_objects::TxAffectedObjects, index_config.clone())
        .await
        .context("Failed to register tx_affected_objects pipeline")?;

    indexer
        .concurrent_pipeline(handlers::tx_balance_changes::TxBalanceChanges, index_config.clone())
        .await
        .context("Failed to register tx_balance_changes pipeline")?;

    indexer
        .concurrent_pipeline(handlers::tx_kinds::TxKinds, index_config.clone())
        .await
        .context("Failed to register tx_kinds pipeline")?;

    indexer
        .concurrent_pipeline(handlers::tx_calls::TxCalls, index_config)
        .await
        .context("Failed to register tx_calls pipeline")?;

    // --- Tier C: State tables (never pruned) ---
    indexer
        .concurrent_pipeline(handlers::obj_versions::ObjVersions, no_prune.clone())
        .await
        .context("Failed to register obj_versions pipeline")?;

    indexer
        .concurrent_pipeline(handlers::obj_info::ObjInfo, no_prune.clone())
        .await
        .context("Failed to register obj_info pipeline")?;

    indexer
        .concurrent_pipeline(handlers::coin_balance_buckets::CoinBalanceBuckets, no_prune.clone())
        .await
        .context("Failed to register coin_balance_buckets pipeline")?;

    // --- Tier C: Soma-specific pipelines (never pruned) ---
    indexer
        .concurrent_pipeline(handlers::soma_targets::SomaTargets, no_prune.clone())
        .await
        .context("Failed to register soma_targets pipeline")?;

    indexer
        .concurrent_pipeline(handlers::soma_models::SomaModels, no_prune.clone())
        .await
        .context("Failed to register soma_models pipeline")?;

    indexer
        .concurrent_pipeline(handlers::soma_rewards::SomaRewards, no_prune.clone())
        .await
        .context("Failed to register soma_rewards pipeline")?;

    indexer
        .concurrent_pipeline(handlers::soma_reward_balances::SomaRewardBalances, no_prune.clone())
        .await
        .context("Failed to register soma_reward_balances pipeline")?;

    indexer
        .concurrent_pipeline(handlers::soma_target_models::SomaTargetModels, no_prune.clone())
        .await
        .context("Failed to register soma_target_models pipeline")?;

    indexer
        .concurrent_pipeline(handlers::soma_staked_soma::SomaStakedSoma, no_prune.clone())
        .await
        .context("Failed to register soma_staked_soma pipeline")?;

    indexer
        .concurrent_pipeline(handlers::soma_epoch_state::SomaEpochState, no_prune.clone())
        .await
        .context("Failed to register soma_epoch_state pipeline")?;

    indexer
        .concurrent_pipeline(handlers::soma_target_reports::SomaTargetReports, no_prune)
        .await
        .context("Failed to register soma_target_reports pipeline")?;

    Ok(())
}

/// Spawns a background task that periodically polls BigTable for its watermark
/// and updates the shared `floor` AtomicU64. The pruner's reader_watermark task
/// uses this floor to ensure data is not deleted from Postgres before BigTable
/// has confirmed indexing it (zero-gap guarantee).
pub fn spawn_bigtable_watermark_poller(
    client: indexer_kvstore::BigTableClient,
    floor: Arc<AtomicU64>,
) -> JoinHandle<()> {
    use indexer_kvstore::KeyValueStoreReader;
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(30));
        loop {
            interval.tick().await;
            let mut c = client.clone();
            match KeyValueStoreReader::get_watermark(&mut c).await {
                Ok(Some(wm)) => {
                    floor.store(wm.checkpoint_hi_inclusive, Ordering::Relaxed);
                }
                Ok(None) => {} // BigTable empty, keep floor at 0
                Err(e) => warn!("Failed to poll BigTable watermark: {e}"),
            }
        }
    })
}
