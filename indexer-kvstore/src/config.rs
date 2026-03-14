// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use indexer_framework::config::ConcurrencyConfig;
use indexer_framework::pipeline::CommitterConfig;
use indexer_framework::pipeline::concurrent::ConcurrentConfig;
use indexer_framework::{self as framework};
use serde::Deserialize;

#[derive(Clone, Default, Debug, Deserialize)]
#[serde(default)]
pub struct IndexerConfig {
    pub ingestion: IngestionConfig,
    pub committer: CommitterLayer,
    pub pipeline: PipelineLayer,
    pub total_max_rows_per_second: Option<u64>,
    pub max_rows_per_second: Option<u64>,
    pub bigtable_connection_pool_size: Option<usize>,
    pub bigtable_channel_timeout_ms: Option<u64>,
}

#[derive(Clone, Default, Debug, Deserialize)]
#[serde(default)]
pub struct CommitterLayer {
    pub write_concurrency: Option<usize>,
    pub collect_interval_ms: Option<u64>,
    pub watermark_interval_ms: Option<u64>,
    pub watermark_interval_jitter_ms: Option<u64>,
}

impl CommitterLayer {
    pub fn finish(self, base: CommitterConfig) -> CommitterConfig {
        CommitterConfig {
            write_concurrency: self.write_concurrency.unwrap_or(base.write_concurrency),
            collect_interval_ms: self.collect_interval_ms.unwrap_or(base.collect_interval_ms),
            watermark_interval_ms: self.watermark_interval_ms.unwrap_or(base.watermark_interval_ms),
            watermark_interval_jitter_ms: self
                .watermark_interval_jitter_ms
                .unwrap_or(base.watermark_interval_jitter_ms),
        }
    }
}

#[derive(Clone, Default, Debug, Deserialize)]
#[serde(default)]
pub struct ConcurrentLayer {
    pub committer: Option<CommitterLayer>,
    pub max_rows: Option<usize>,
    pub max_rows_per_second: Option<u64>,
    pub fanout: Option<ConcurrencyConfig>,
    pub min_eager_rows: Option<usize>,
    pub max_pending_rows: Option<usize>,
    pub max_watermark_updates: Option<usize>,
    pub processor_channel_size: Option<usize>,
    pub collector_channel_size: Option<usize>,
    pub committer_channel_size: Option<usize>,
}

impl ConcurrentLayer {
    pub fn finish(self, base: ConcurrentConfig) -> ConcurrentConfig {
        ConcurrentConfig {
            committer: if let Some(c) = self.committer {
                c.finish(base.committer)
            } else {
                base.committer
            },
            pruner: None,
            fanout: self.fanout.or(base.fanout),
            min_eager_rows: self.min_eager_rows.or(base.min_eager_rows),
            max_pending_rows: self.max_pending_rows.or(base.max_pending_rows),
            max_watermark_updates: self.max_watermark_updates.or(base.max_watermark_updates),
            processor_channel_size: self.processor_channel_size.or(base.processor_channel_size),
            collector_channel_size: self.collector_channel_size.or(base.collector_channel_size),
            committer_channel_size: self.committer_channel_size.or(base.committer_channel_size),
        }
    }
}

#[derive(Clone, Default, Debug, Deserialize)]
#[serde(default)]
pub struct PipelineLayer {
    pub checkpoints: ConcurrentLayer,
    pub checkpoints_by_digest: ConcurrentLayer,
    pub transactions: ConcurrentLayer,
    pub objects: ConcurrentLayer,
    pub epoch_start: ConcurrentLayer,
    pub epoch_end: ConcurrentLayer,
    // Soma-specific
    pub soma_targets: ConcurrentLayer,
    pub soma_models: ConcurrentLayer,
    pub soma_rewards: ConcurrentLayer,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct IngestionConfig {
    pub checkpoint_buffer_size: usize,
    pub ingest_concurrency: ConcurrencyConfig,
    pub retry_interval_ms: u64,
}

impl Default for IngestionConfig {
    fn default() -> Self {
        let base = framework::ingestion::IngestionConfig::default();
        Self {
            checkpoint_buffer_size: base.checkpoint_buffer_size,
            ingest_concurrency: base.ingest_concurrency,
            retry_interval_ms: base.retry_interval_ms,
        }
    }
}

impl From<IngestionConfig> for framework::ingestion::IngestionConfig {
    fn from(config: IngestionConfig) -> Self {
        framework::ingestion::IngestionConfig {
            checkpoint_buffer_size: config.checkpoint_buffer_size,
            ingest_concurrency: config.ingest_concurrency,
            retry_interval_ms: config.retry_interval_ms,
        }
    }
}
