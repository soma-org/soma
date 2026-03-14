// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

mod bigtable;
pub mod config;
mod handlers;
pub mod kv_rpc;
mod rate_limiter;
pub mod tables;

use std::sync::Arc;
use std::sync::OnceLock;

use anyhow::Result;
use async_trait::async_trait;
use indexer_framework::Indexer;
use indexer_framework::IndexerArgs;
use indexer_framework::ingestion::ClientArgs;
use indexer_framework::pipeline::CommitterConfig;
use indexer_framework::pipeline::concurrent::ConcurrentConfig;
use prometheus::Registry;
use serde::Deserialize;
use serde::Serialize;

use crate::rate_limiter::CompositeRateLimiter;
use crate::rate_limiter::RateLimiter;
use bytes::Bytes;
use types::checkpoints::CheckpointContents;
use types::checkpoints::CheckpointSequenceNumber;
use types::checkpoints::CheckpointSummary;
use types::committee::EpochId;
use types::crypto::AuthorityStrongQuorumSignInfo;
use types::digests::CheckpointDigest;
use types::digests::TransactionDigest;
use types::effects::TransactionEffects;
use types::object::Object;
use types::object::ObjectID;
use types::storage::ObjectKey;
use types::transaction::Transaction;

pub use crate::bigtable::client::BigTableClient;
pub use crate::bigtable::store::BigTableConnection;
pub use crate::bigtable::store::BigTableStore;
pub use crate::handlers::BigTableHandler;
pub use crate::handlers::CheckpointsByDigestPipeline;
pub use crate::handlers::CheckpointsPipeline;
pub use crate::handlers::EpochEndPipeline;
pub use crate::handlers::EpochStartPipeline;
pub use crate::handlers::ObjectsPipeline;
pub use crate::handlers::SomaModelsPipeline;
pub use crate::handlers::SomaRewardsPipeline;
pub use crate::handlers::SomaTargetsPipeline;
pub use crate::handlers::TransactionsPipeline;
pub use config::CommitterLayer;
pub use config::ConcurrentLayer;
pub use config::IndexerConfig;
pub use config::IngestionConfig;
pub use config::PipelineLayer;

pub const CHECKPOINTS_PIPELINE: &str =
    <BigTableHandler<CheckpointsPipeline> as indexer_framework::pipeline::Processor>::NAME;
pub const CHECKPOINTS_BY_DIGEST_PIPELINE: &str =
    <BigTableHandler<CheckpointsByDigestPipeline> as indexer_framework::pipeline::Processor>::NAME;
pub const TRANSACTIONS_PIPELINE: &str =
    <BigTableHandler<TransactionsPipeline> as indexer_framework::pipeline::Processor>::NAME;
pub const OBJECTS_PIPELINE: &str =
    <BigTableHandler<ObjectsPipeline> as indexer_framework::pipeline::Processor>::NAME;
pub const EPOCH_START_PIPELINE: &str =
    <BigTableHandler<EpochStartPipeline> as indexer_framework::pipeline::Processor>::NAME;
pub const EPOCH_END_PIPELINE: &str =
    <BigTableHandler<EpochEndPipeline> as indexer_framework::pipeline::Processor>::NAME;
pub const SOMA_TARGETS_PIPELINE: &str =
    <BigTableHandler<SomaTargetsPipeline> as indexer_framework::pipeline::Processor>::NAME;
pub const SOMA_MODELS_PIPELINE: &str =
    <BigTableHandler<SomaModelsPipeline> as indexer_framework::pipeline::Processor>::NAME;
pub const SOMA_REWARDS_PIPELINE: &str =
    <BigTableHandler<SomaRewardsPipeline> as indexer_framework::pipeline::Processor>::NAME;

/// All pipeline names registered by the indexer.
pub const ALL_PIPELINE_NAMES: [&str; 9] = [
    CHECKPOINTS_PIPELINE,
    CHECKPOINTS_BY_DIGEST_PIPELINE,
    TRANSACTIONS_PIPELINE,
    OBJECTS_PIPELINE,
    EPOCH_START_PIPELINE,
    EPOCH_END_PIPELINE,
    SOMA_TARGETS_PIPELINE,
    SOMA_MODELS_PIPELINE,
    SOMA_REWARDS_PIPELINE,
];

/// Non-legacy pipeline names used for the default `get_watermark` implementation.
const WATERMARK_PIPELINES: [&str; 9] = ALL_PIPELINE_NAMES;

static WRITE_LEGACY_DATA: OnceLock<bool> = OnceLock::new();

pub fn set_write_legacy_data(value: bool) {
    WRITE_LEGACY_DATA.set(value).expect("write_legacy_data already set");
}

pub fn write_legacy_data() -> bool {
    *WRITE_LEGACY_DATA.get_or_init(|| false)
}

pub struct BigTableIndexer {
    pub indexer: Indexer<BigTableStore>,
}

#[derive(Clone, Debug)]
pub struct CheckpointData {
    pub summary: CheckpointSummary,
    pub contents: CheckpointContents,
    pub signatures: AuthorityStrongQuorumSignInfo,
}

#[derive(Clone, Debug)]
pub struct TransactionData {
    pub transaction: Transaction,
    pub effects: TransactionEffects,
    pub timestamp: u64,
    pub checkpoint_number: CheckpointSequenceNumber,
}

/// Epoch data returned by reader methods.
#[derive(Clone, Debug, Default)]
pub struct EpochData {
    pub epoch: Option<u64>,
    pub protocol_version: Option<u64>,
    pub start_timestamp_ms: Option<u64>,
    pub start_checkpoint: Option<u64>,
    pub reference_gas_price: Option<u64>,
    pub system_state_bcs: Option<Vec<u8>>,
    pub end_timestamp_ms: Option<u64>,
    pub end_checkpoint: Option<u64>,
    pub cp_hi: Option<u64>,
    pub tx_hi: Option<u64>,
    pub safe_mode: Option<bool>,
    pub epoch_commitments: Option<Vec<u8>>,
}

/// Serializable watermark for per-pipeline tracking in BigTable.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Watermark {
    pub epoch_hi_inclusive: u64,
    pub checkpoint_hi_inclusive: u64,
    pub tx_hi: u64,
    pub timestamp_ms_hi_inclusive: u64,
}

#[async_trait]
pub trait KeyValueStoreReader {
    async fn get_checkpoints(
        &mut self,
        sequence_numbers: &[CheckpointSequenceNumber],
    ) -> Result<Vec<CheckpointData>>;
    async fn get_checkpoint_by_digest(
        &mut self,
        digest: CheckpointDigest,
    ) -> Result<Option<CheckpointData>>;
    async fn get_transactions(
        &mut self,
        transactions: &[TransactionDigest],
    ) -> Result<Vec<TransactionData>>;
    async fn get_objects(&mut self, keys: &[ObjectKey]) -> Result<Vec<Object>>;
    async fn get_latest_object(&mut self, object_id: &ObjectID) -> Result<Option<Object>>;
    async fn get_epoch(&mut self, epoch_id: EpochId) -> Result<Option<EpochData>>;
    async fn get_latest_epoch(&mut self) -> Result<Option<EpochData>>;
    async fn get_watermark(&mut self) -> Result<Option<Watermark>> {
        self.get_watermark_for_pipelines(&WATERMARK_PIPELINES).await
    }
    async fn get_watermark_for_pipelines(
        &mut self,
        pipelines: &[&str],
    ) -> Result<Option<Watermark>>;
}

#[async_trait]
impl KeyValueStoreReader for BigTableClient {
    async fn get_checkpoints(
        &mut self,
        sequence_numbers: &[CheckpointSequenceNumber],
    ) -> Result<Vec<CheckpointData>> {
        let keys = sequence_numbers.iter().copied().map(tables::checkpoints::encode_key).collect();
        let mut checkpoints = vec![];
        for (_, row) in self.multi_get(tables::checkpoints::NAME, keys, None).await? {
            checkpoints.push(tables::checkpoints::decode(&row)?);
        }
        Ok(checkpoints)
    }

    async fn get_checkpoint_by_digest(
        &mut self,
        digest: CheckpointDigest,
    ) -> Result<Option<CheckpointData>> {
        let key = tables::checkpoints_by_digest::encode_key(&digest);
        let rows = self.multi_get(tables::checkpoints_by_digest::NAME, vec![key], None).await?;
        let Some((_, row)) = rows.into_iter().next() else {
            return Ok(None);
        };
        let seq = tables::checkpoints_by_digest::decode(&row)?;
        let mut checkpoints = self.get_checkpoints(&[seq]).await?;
        Ok(checkpoints.pop())
    }

    async fn get_transactions(
        &mut self,
        transactions: &[TransactionDigest],
    ) -> Result<Vec<TransactionData>> {
        let keys = transactions.iter().map(tables::transactions::encode_key).collect();
        let mut result = vec![];
        for (_, row) in self.multi_get(tables::transactions::NAME, keys, None).await? {
            result.push(tables::transactions::decode(&row)?);
        }
        Ok(result)
    }

    async fn get_objects(&mut self, keys: &[ObjectKey]) -> Result<Vec<Object>> {
        let bt_keys = keys.iter().map(tables::objects::encode_key).collect();
        let mut result = vec![];
        for (_, row) in self.multi_get(tables::objects::NAME, bt_keys, None).await? {
            result.push(tables::objects::decode(&row)?);
        }
        Ok(result)
    }

    async fn get_latest_object(&mut self, object_id: &ObjectID) -> Result<Option<Object>> {
        // Objects are keyed by (object_id, version) with version as big-endian u64.
        // To find the latest version, reverse scan with the object_id prefix.
        let mut start_key = object_id.to_vec();
        start_key.extend(0u64.to_be_bytes());
        let mut end_key = object_id.to_vec();
        end_key.extend(u64::MAX.to_be_bytes());

        let rows = self
            .range_scan(
                tables::objects::NAME,
                Some(Bytes::from(start_key)),
                Some(Bytes::from(end_key)),
                1,
                true, // reversed — get latest version first
            )
            .await?;

        match rows.into_iter().next() {
            Some((_, row)) => Ok(Some(tables::objects::decode(&row)?)),
            None => Ok(None),
        }
    }

    async fn get_epoch(&mut self, epoch_id: EpochId) -> Result<Option<EpochData>> {
        let key = tables::epochs::encode_key(epoch_id);
        let rows = self.multi_get(tables::epochs::NAME, vec![key], None).await?;
        match rows.into_iter().next() {
            Some((_, row)) => Ok(Some(tables::epochs::decode(&row)?)),
            None => Ok(None),
        }
    }

    async fn get_latest_epoch(&mut self) -> Result<Option<EpochData>> {
        let rows = self
            .range_scan(
                tables::epochs::NAME,
                None,
                Some(tables::epochs::encode_key_upper_bound()),
                1,
                true, // reversed — get latest epoch first
            )
            .await?;

        match rows.into_iter().next() {
            Some((_, row)) => Ok(Some(tables::epochs::decode(&row)?)),
            None => Ok(None),
        }
    }

    async fn get_watermark_for_pipelines(
        &mut self,
        pipelines: &[&str],
    ) -> Result<Option<Watermark>> {
        let keys: Vec<Vec<u8>> =
            pipelines.iter().map(|name| tables::watermarks::encode_key(name)).collect();

        let rows = self.multi_get(tables::watermark_alt_legacy::NAME, keys, None).await?;

        if rows.len() != pipelines.len() {
            return Ok(None);
        }

        let mut min_wm: Option<Watermark> = None;
        for (_, row) in &rows {
            let wm = tables::watermarks::decode(row)?;
            min_wm = Some(match min_wm {
                Some(prev) if prev.checkpoint_hi_inclusive <= wm.checkpoint_hi_inclusive => prev,
                _ => wm,
            });
        }

        Ok(min_wm)
    }
}

impl BigTableIndexer {
    pub async fn new(
        store: BigTableStore,
        indexer_args: IndexerArgs,
        client_args: ClientArgs,
        ingestion_config: IngestionConfig,
        committer: CommitterConfig,
        config: IndexerConfig,
        pipeline: PipelineLayer,
        registry: &Registry,
    ) -> Result<Self> {
        let mut indexer =
            Indexer::new(store, indexer_args, client_args, ingestion_config.into(), None, registry)
                .await?;

        let global = config.total_max_rows_per_second.map(RateLimiter::new);
        let base_rps = config.max_rows_per_second;

        fn build_rate_limiter(
            layer: &ConcurrentLayer,
            base_rps: Option<u64>,
            global: &Option<Arc<RateLimiter>>,
        ) -> Arc<CompositeRateLimiter> {
            let mut limiters = Vec::new();
            if let Some(rps) = layer.max_rows_per_second.or(base_rps) {
                limiters.push(RateLimiter::new(rps));
            }
            if let Some(g) = global {
                limiters.push(g.clone());
            }
            Arc::new(CompositeRateLimiter::new(limiters))
        }

        let base = ConcurrentConfig { committer, pruner: None, ..Default::default() };

        indexer
            .concurrent_pipeline(
                BigTableHandler::new(
                    CheckpointsPipeline,
                    &pipeline.checkpoints,
                    build_rate_limiter(&pipeline.checkpoints, base_rps, &global),
                ),
                pipeline.checkpoints.finish(base.clone()),
            )
            .await?;
        indexer
            .concurrent_pipeline(
                BigTableHandler::new(
                    CheckpointsByDigestPipeline,
                    &pipeline.checkpoints_by_digest,
                    build_rate_limiter(&pipeline.checkpoints_by_digest, base_rps, &global),
                ),
                pipeline.checkpoints_by_digest.finish(base.clone()),
            )
            .await?;
        indexer
            .concurrent_pipeline(
                BigTableHandler::new(
                    TransactionsPipeline,
                    &pipeline.transactions,
                    build_rate_limiter(&pipeline.transactions, base_rps, &global),
                ),
                pipeline.transactions.finish(base.clone()),
            )
            .await?;
        indexer
            .concurrent_pipeline(
                BigTableHandler::new(
                    ObjectsPipeline,
                    &pipeline.objects,
                    build_rate_limiter(&pipeline.objects, base_rps, &global),
                ),
                pipeline.objects.finish(base.clone()),
            )
            .await?;
        indexer
            .concurrent_pipeline(
                BigTableHandler::new(
                    EpochStartPipeline,
                    &pipeline.epoch_start,
                    build_rate_limiter(&pipeline.epoch_start, base_rps, &global),
                ),
                pipeline.epoch_start.finish(base.clone()),
            )
            .await?;
        indexer
            .concurrent_pipeline(
                BigTableHandler::new(
                    EpochEndPipeline,
                    &pipeline.epoch_end,
                    build_rate_limiter(&pipeline.epoch_end, base_rps, &global),
                ),
                pipeline.epoch_end.finish(base.clone()),
            )
            .await?;
        // Soma-specific pipelines
        indexer
            .concurrent_pipeline(
                BigTableHandler::new(
                    SomaTargetsPipeline,
                    &pipeline.soma_targets,
                    build_rate_limiter(&pipeline.soma_targets, base_rps, &global),
                ),
                pipeline.soma_targets.finish(base.clone()),
            )
            .await?;
        indexer
            .concurrent_pipeline(
                BigTableHandler::new(
                    SomaModelsPipeline,
                    &pipeline.soma_models,
                    build_rate_limiter(&pipeline.soma_models, base_rps, &global),
                ),
                pipeline.soma_models.finish(base.clone()),
            )
            .await?;
        indexer
            .concurrent_pipeline(
                BigTableHandler::new(
                    SomaRewardsPipeline,
                    &pipeline.soma_rewards,
                    build_rate_limiter(&pipeline.soma_rewards, base_rps, &global),
                ),
                pipeline.soma_rewards.finish(base),
            )
            .await?;

        Ok(Self { indexer })
    }
}

impl From<indexer_store_traits::CommitterWatermark> for Watermark {
    fn from(w: indexer_store_traits::CommitterWatermark) -> Self {
        Self {
            epoch_hi_inclusive: w.epoch,
            checkpoint_hi_inclusive: w.checkpoint_hi_inclusive,
            tx_hi: w.tx_hi,
            timestamp_ms_hi_inclusive: w.timestamp_ms_hi_inclusive,
        }
    }
}

impl From<Watermark> for indexer_store_traits::CommitterWatermark {
    fn from(w: Watermark) -> Self {
        Self {
            epoch: w.epoch_hi_inclusive,
            checkpoint_hi_inclusive: w.checkpoint_hi_inclusive,
            tx_hi: w.tx_hi,
            timestamp_ms_hi_inclusive: w.timestamp_ms_hi_inclusive,
        }
    }
}

// ---------------------------------------------------------------------------
// KvLoader — thread-safe, `&self` interface for GraphQL BCS content lookups
// ---------------------------------------------------------------------------

/// A thread-safe interface for loading BCS content from a KV store (BigTable).
///
/// Unlike [`KeyValueStoreReader`] (which takes `&mut self`), this trait takes `&self`
/// and is intended for use from async-graphql resolvers that share data via `Arc`.
/// Implementations must handle interior mutability / cloning internally.
#[async_trait]
pub trait KvLoader: Send + Sync {
    async fn get_checkpoint(&self, seq: CheckpointSequenceNumber)
    -> Result<Option<CheckpointData>>;
    async fn get_transaction(&self, digest: &TransactionDigest) -> Result<Option<TransactionData>>;
    async fn get_object(&self, id: &ObjectID, version: u64) -> Result<Option<Object>>;
    async fn get_epoch(&self, epoch_id: EpochId) -> Result<Option<EpochData>>;
    async fn get_watermark(&self) -> Result<Option<Watermark>>;
}

/// `BigTableClient`-backed [`KvLoader`] implementation.
///
/// Each method clones the inner `BigTableClient` (cheap — just a tonic `Channel`)
/// and calls the corresponding `KeyValueStoreReader` method on the clone.
pub struct BigTableKvLoader {
    client: BigTableClient,
}

impl BigTableKvLoader {
    pub fn new(client: BigTableClient) -> Self {
        Self { client }
    }
}

#[async_trait]
impl KvLoader for BigTableKvLoader {
    async fn get_checkpoint(
        &self,
        seq: CheckpointSequenceNumber,
    ) -> Result<Option<CheckpointData>> {
        let mut c = self.client.clone();
        let results = c.get_checkpoints(&[seq]).await?;
        Ok(results.into_iter().next())
    }

    async fn get_transaction(&self, digest: &TransactionDigest) -> Result<Option<TransactionData>> {
        let mut c = self.client.clone();
        let results = c.get_transactions(&[*digest]).await?;
        Ok(results.into_iter().next())
    }

    async fn get_object(&self, id: &ObjectID, version: u64) -> Result<Option<Object>> {
        let mut c = self.client.clone();
        let key = ObjectKey(*id, types::object::Version::from_u64(version));
        let results = c.get_objects(&[key]).await?;
        Ok(results.into_iter().next())
    }

    async fn get_epoch(&self, epoch_id: EpochId) -> Result<Option<EpochData>> {
        let mut c = self.client.clone();
        c.get_epoch(epoch_id).await
    }

    async fn get_watermark(&self) -> Result<Option<Watermark>> {
        let mut c = self.client.clone();
        KeyValueStoreReader::get_watermark(&mut c).await
    }
}
