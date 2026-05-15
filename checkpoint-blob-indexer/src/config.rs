// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use indexer_framework::config::ConcurrencyConfig;
use indexer_framework::pipeline::CommitterConfig;
use indexer_framework::pipeline::concurrent::ConcurrentConfig;
use soma_default_config::DefaultConfig;

#[DefaultConfig]
#[derive(Clone, Default, Debug)]
pub struct IndexerConfig {
    pub ingestion: IngestionConfig,
    pub committer: CommitterLayer,
    pub pipeline: PipelineLayer,
}

#[DefaultConfig]
#[derive(Clone, Default, Debug)]
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

#[DefaultConfig]
#[derive(Clone, Default, Debug)]
pub struct ConcurrentLayer {
    pub committer: Option<CommitterLayer>,
    pub fanout: Option<ConcurrencyConfig>,
}

impl ConcurrentLayer {
    pub fn finish(self, base: ConcurrentConfig) -> ConcurrentConfig {
        ConcurrentConfig {
            committer: if let Some(c) = self.committer {
                c.finish(base.committer)
            } else {
                base.committer
            },
            ..base
        }
    }
}

#[DefaultConfig]
#[derive(Clone, Default, Debug)]
pub struct PipelineLayer {
    pub checkpoint_blob: ConcurrentLayer,
    pub epochs: ConcurrentLayer,
}

/// Mirror of [`indexer_framework::ingestion::IngestionConfig`] with serde-friendly defaults.
#[DefaultConfig]
#[derive(Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct IngestionConfig {
    pub ingest_concurrency: ConcurrencyConfig,
    pub retry_interval_ms: u64,
    pub checkpoint_buffer_size: usize,
}

impl Default for IngestionConfig {
    fn default() -> Self {
        indexer_framework::ingestion::IngestionConfig::default().into()
    }
}

impl From<indexer_framework::ingestion::IngestionConfig> for IngestionConfig {
    fn from(config: indexer_framework::ingestion::IngestionConfig) -> Self {
        Self {
            ingest_concurrency: config.ingest_concurrency,
            retry_interval_ms: config.retry_interval_ms,
            checkpoint_buffer_size: config.checkpoint_buffer_size,
        }
    }
}

impl From<IngestionConfig> for indexer_framework::ingestion::IngestionConfig {
    fn from(config: IngestionConfig) -> Self {
        Self {
            ingest_concurrency: config.ingest_concurrency,
            retry_interval_ms: config.retry_interval_ms,
            checkpoint_buffer_size: config.checkpoint_buffer_size,
        }
    }
}
