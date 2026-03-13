// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use anyhow::Context;
use anyhow::Result;
use indexer_framework::Indexer;
use indexer_framework::postgres::Db;
use indexer_framework::pipeline::concurrent::ConcurrentConfig;

pub mod handlers;
#[cfg(test)]
pub(crate) mod test_utils;

/// Register all indexer pipelines on the given indexer instance.
///
/// Each handler implements the `Processor` trait (for transforming checkpoint data into rows)
/// and the postgres `Handler` trait (for committing rows to the database).
pub async fn setup_indexer(indexer: &mut Indexer<Db>) -> Result<()> {
    let config = ConcurrentConfig::default();

    // High priority: core data pipelines
    indexer
        .concurrent_pipeline(handlers::cp_sequence_numbers::CpSequenceNumbers, config.clone())
        .await
        .context("Failed to register cp_sequence_numbers pipeline")?;

    indexer
        .concurrent_pipeline(handlers::tx_digests::TxDigests, config.clone())
        .await
        .context("Failed to register tx_digests pipeline")?;

    indexer
        .concurrent_pipeline(handlers::kv_checkpoints::KvCheckpoints, config.clone())
        .await
        .context("Failed to register kv_checkpoints pipeline")?;

    indexer
        .concurrent_pipeline(handlers::kv_transactions::KvTransactions, config.clone())
        .await
        .context("Failed to register kv_transactions pipeline")?;

    indexer
        .concurrent_pipeline(handlers::kv_objects::KvObjects, config.clone())
        .await
        .context("Failed to register kv_objects pipeline")?;

    indexer
        .concurrent_pipeline(handlers::kv_epoch_starts::KvEpochStarts, config.clone())
        .await
        .context("Failed to register kv_epoch_starts pipeline")?;

    indexer
        .concurrent_pipeline(handlers::kv_epoch_ends::KvEpochEnds, config.clone())
        .await
        .context("Failed to register kv_epoch_ends pipeline")?;

    // Medium priority: auxiliary indexes
    indexer
        .concurrent_pipeline(handlers::tx_affected_addresses::TxAffectedAddresses, config.clone())
        .await
        .context("Failed to register tx_affected_addresses pipeline")?;

    indexer
        .concurrent_pipeline(handlers::tx_affected_objects::TxAffectedObjects, config.clone())
        .await
        .context("Failed to register tx_affected_objects pipeline")?;

    indexer
        .concurrent_pipeline(handlers::tx_balance_changes::TxBalanceChanges, config.clone())
        .await
        .context("Failed to register tx_balance_changes pipeline")?;

    indexer
        .concurrent_pipeline(handlers::tx_kinds::TxKinds, config.clone())
        .await
        .context("Failed to register tx_kinds pipeline")?;

    indexer
        .concurrent_pipeline(handlers::tx_calls::TxCalls, config.clone())
        .await
        .context("Failed to register tx_calls pipeline")?;

    indexer
        .concurrent_pipeline(handlers::obj_versions::ObjVersions, config.clone())
        .await
        .context("Failed to register obj_versions pipeline")?;

    indexer
        .concurrent_pipeline(handlers::obj_info::ObjInfo, config.clone())
        .await
        .context("Failed to register obj_info pipeline")?;

    indexer
        .concurrent_pipeline(handlers::coin_balance_buckets::CoinBalanceBuckets, config.clone())
        .await
        .context("Failed to register coin_balance_buckets pipeline")?;

    // Soma-specific pipelines
    indexer
        .concurrent_pipeline(handlers::soma_targets::SomaTargets, config.clone())
        .await
        .context("Failed to register soma_targets pipeline")?;

    indexer
        .concurrent_pipeline(handlers::soma_models::SomaModels, config.clone())
        .await
        .context("Failed to register soma_models pipeline")?;

    indexer
        .concurrent_pipeline(handlers::soma_rewards::SomaRewards, config.clone())
        .await
        .context("Failed to register soma_rewards pipeline")?;

    indexer
        .concurrent_pipeline(handlers::soma_reward_balances::SomaRewardBalances, config.clone())
        .await
        .context("Failed to register soma_reward_balances pipeline")?;

    indexer
        .concurrent_pipeline(handlers::soma_target_models::SomaTargetModels, config.clone())
        .await
        .context("Failed to register soma_target_models pipeline")?;

    indexer
        .concurrent_pipeline(handlers::soma_staked_soma::SomaStakedSoma, config.clone())
        .await
        .context("Failed to register soma_staked_soma pipeline")?;

    indexer
        .concurrent_pipeline(handlers::soma_epoch_state::SomaEpochState, config.clone())
        .await
        .context("Failed to register soma_epoch_state pipeline")?;

    indexer
        .concurrent_pipeline(handlers::soma_target_reports::SomaTargetReports, config)
        .await
        .context("Failed to register soma_target_reports pipeline")?;

    Ok(())
}
