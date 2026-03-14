// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::time::Duration;

use async_trait::async_trait;
use serde::Deserialize;
use serde::Serialize;
use soma_futures::service::Service;
use tokio::sync::SetOnce;
use tokio::sync::mpsc;
use tracing::info;

use crate::Task;
use crate::config::ConcurrencyConfig;
use crate::metrics::IndexerMetrics;
use crate::pipeline::CommitterConfig;
use crate::pipeline::Processor;
use crate::pipeline::WatermarkPart;
use crate::pipeline::concurrent::collector::collector;
use crate::pipeline::concurrent::commit_watermark::commit_watermark;
use crate::pipeline::concurrent::committer::committer;
use crate::pipeline::concurrent::main_reader_lo::track_main_reader_lo;
use crate::pipeline::concurrent::pruner::pruner;
use crate::pipeline::concurrent::reader_watermark::reader_watermark;
use crate::pipeline::processor::processor;
use crate::store::Store;
use crate::types::full_checkpoint_content::Checkpoint;

mod collector;
mod commit_watermark;
mod committer;
mod main_reader_lo;
mod pruner;
mod reader_watermark;

/// Status returned by `Handler::batch` to indicate whether the batch is ready to be committed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchStatus {
    /// The batch can accept more values.
    Pending,
    /// The batch is full and should be committed.
    Ready,
}

/// Handlers implement the logic for a given indexing pipeline.
///
/// Concurrent handlers can only be used in concurrent pipelines, where checkpoint data is
/// processed and committed out-of-order and a watermark table is kept up-to-date with the latest
/// checkpoint below which all data has been committed.
#[async_trait]
pub trait Handler: Processor {
    type Store: Store;
    type Batch: Default + Send + Sync + 'static;

    /// If at least this many rows are pending, the committer will commit them eagerly.
    const MIN_EAGER_ROWS: usize = 50;

    /// If there are more than this many rows pending, the committer applies backpressure.
    const MAX_PENDING_ROWS: usize = 5000;

    /// The maximum number of watermarks that can show up in a single batch.
    const MAX_WATERMARK_UPDATES: usize = 10_000;

    /// Add values from the iterator to the batch.
    fn batch(
        &self,
        batch: &mut Self::Batch,
        values: &mut std::vec::IntoIter<Self::Value>,
    ) -> BatchStatus;

    /// Commit the batch to the database, returning the number of rows affected.
    async fn commit<'a>(
        &self,
        batch: &Self::Batch,
        conn: &mut <Self::Store as Store>::Connection<'a>,
    ) -> anyhow::Result<usize>;

    /// Clean up data between checkpoints `_from` and `_to_exclusive` (exclusive) in the database.
    async fn prune<'a>(
        &self,
        _from: u64,
        _to_exclusive: u64,
        _conn: &mut <Self::Store as Store>::Connection<'a>,
    ) -> anyhow::Result<usize> {
        Ok(0)
    }
}

/// Configuration for a concurrent pipeline
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct ConcurrentConfig {
    /// Configuration for the writer, that makes forward progress.
    pub committer: CommitterConfig,

    /// Configuration for the pruner, that deletes old data.
    pub pruner: Option<PrunerConfig>,

    /// Processor concurrency.
    pub fanout: Option<ConcurrencyConfig>,

    /// Override for `Handler::MIN_EAGER_ROWS`.
    pub min_eager_rows: Option<usize>,

    /// Override for `Handler::MAX_PENDING_ROWS`.
    pub max_pending_rows: Option<usize>,

    /// Override for `Handler::MAX_WATERMARK_UPDATES`.
    pub max_watermark_updates: Option<usize>,

    /// Size of the channel between the processor and collector.
    pub processor_channel_size: Option<usize>,

    /// Size of the channel between the collector and committer.
    pub collector_channel_size: Option<usize>,

    /// Size of the channel between the committer and the watermark updater.
    pub committer_channel_size: Option<usize>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PrunerConfig {
    /// How often the pruner should check whether there is any data to prune, in milliseconds.
    pub interval_ms: u64,

    /// How long to wait after the reader low watermark was set, in milliseconds.
    pub delay_ms: u64,

    /// How much data to keep, measured in checkpoints.
    pub retention: u64,

    /// The maximum range to try and prune in one request, measured in checkpoints.
    pub max_chunk_size: u64,

    /// The max number of tasks to run in parallel for pruning.
    pub prune_concurrency: u64,

    /// Optional external watermark floor (e.g., BigTable's checkpoint_hi). When set,
    /// `reader_lo` will never exceed `floor + 1`, ensuring data is not pruned from
    /// Postgres before the external store has confirmed indexing it.
    #[serde(skip)]
    pub external_watermark_floor: Option<Arc<AtomicU64>>,
}

/// Values ready to be written to the database.
struct BatchedRows<H: Handler> {
    /// The batch to write
    batch: H::Batch,
    /// Number of rows in the batch
    batch_len: usize,
    /// Proportions of all the watermarks that are represented in this chunk
    watermark: Vec<WatermarkPart>,
}

impl PrunerConfig {
    pub fn interval(&self) -> Duration {
        Duration::from_millis(self.interval_ms)
    }

    pub fn delay(&self) -> Duration {
        Duration::from_millis(self.delay_ms)
    }
}

impl Default for PrunerConfig {
    fn default() -> Self {
        Self {
            interval_ms: 300_000,
            delay_ms: 120_000,
            retention: 4_000_000,
            max_chunk_size: 2_000,
            prune_concurrency: 1,
            external_watermark_floor: None,
        }
    }
}

fn default_concurrency() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

/// Start a new concurrent (out-of-order) indexing pipeline served by the handler, `H`.
pub(crate) fn pipeline<H: Handler + Send + Sync + 'static>(
    handler: H,
    next_checkpoint: u64,
    config: ConcurrentConfig,
    store: H::Store,
    task: Option<Task>,
    checkpoint_rx: mpsc::Receiver<Arc<Checkpoint>>,
    metrics: Arc<IndexerMetrics>,
) -> Service {
    info!(
        pipeline = H::NAME,
        "Starting pipeline with config: {config:#?}",
    );

    let ConcurrentConfig {
        committer: committer_config,
        pruner: pruner_config,
        fanout,
        min_eager_rows,
        max_pending_rows,
        max_watermark_updates,
        processor_channel_size,
        collector_channel_size,
        committer_channel_size,
    } = config;

    let num_cpus = default_concurrency();

    let concurrency = fanout.unwrap_or(ConcurrencyConfig::Adaptive {
        initial: 1,
        min: 1,
        max: num_cpus,
        dead_band: None,
    });
    let min_eager_rows = min_eager_rows.unwrap_or(H::MIN_EAGER_ROWS);
    let max_pending_rows = max_pending_rows.unwrap_or(H::MAX_PENDING_ROWS);
    let max_watermark_updates = max_watermark_updates.unwrap_or(H::MAX_WATERMARK_UPDATES);

    let processor_channel_size = processor_channel_size.unwrap_or(num_cpus / 2);
    let (processor_tx, collector_rx) = mpsc::channel(processor_channel_size);

    let collector_channel_size = collector_channel_size.unwrap_or(num_cpus / 2);
    let (collector_tx, committer_rx) = mpsc::channel(collector_channel_size);
    let committer_channel_size = committer_channel_size.unwrap_or(num_cpus);
    let (committer_tx, watermark_rx) = mpsc::channel(committer_channel_size);
    let main_reader_lo = Arc::new(SetOnce::new());

    let handler = Arc::new(handler);

    let s_processor = processor(
        handler.clone(),
        checkpoint_rx,
        processor_tx,
        metrics.clone(),
        concurrency,
    );

    let s_collector = collector::<H>(
        handler.clone(),
        committer_config.clone(),
        collector_rx,
        collector_tx,
        main_reader_lo.clone(),
        metrics.clone(),
        min_eager_rows,
        max_pending_rows,
        max_watermark_updates,
    );

    let s_committer = committer::<H>(
        handler.clone(),
        committer_config.clone(),
        committer_rx,
        committer_tx,
        store.clone(),
        metrics.clone(),
    );

    let s_commit_watermark = commit_watermark::<H>(
        next_checkpoint,
        committer_config,
        watermark_rx,
        store.clone(),
        task.as_ref().map(|t| t.task.clone()),
        metrics.clone(),
    );

    let s_track_reader_lo = track_main_reader_lo::<H>(
        main_reader_lo.clone(),
        task.as_ref().map(|t| t.reader_interval),
        store.clone(),
    );

    let s_reader_watermark =
        reader_watermark::<H>(pruner_config.clone(), store.clone(), metrics.clone());

    let s_pruner = pruner(handler, pruner_config, store, metrics);

    s_processor
        .merge(s_collector)
        .merge(s_committer)
        .merge(s_commit_watermark)
        .attach(s_track_reader_lo)
        .attach(s_reader_watermark)
        .attach(s_pruner)
}
