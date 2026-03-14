// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::sync::Arc;

use futures::StreamExt;
use futures::stream::FuturesUnordered;
use soma_futures::service::Service;
use tokio::sync::Semaphore;
use tokio::time::MissedTickBehavior;
use tokio::time::interval;
use tracing::debug;
use tracing::error;
use tracing::info;
use tracing::warn;

use crate::metrics::IndexerMetrics;
use crate::pipeline::concurrent::Handler;
use crate::pipeline::concurrent::PrunerConfig;
use crate::pipeline::logging::LoggerWatermark;
use crate::pipeline::logging::WatermarkLogger;
use crate::store::Connection;
use crate::store::Store;

#[derive(Default)]
struct PendingRanges {
    /// Maps from `from` to `to_exclusive` for all the ranges that are ready to be pruned.
    ranges: BTreeMap<u64, u64>,
    /// The last range that has been scheduled for pruning.
    last_scheduled_range: Option<(u64, u64)>,
}

impl PendingRanges {
    /// Schedule a new range to be pruned.
    /// Using the last scheduled range to avoid double pruning of the same range.
    /// This is important because double pruning will not always work since pruning
    /// may not be idempotent for some pipelines.
    fn schedule(&mut self, mut from: u64, to_exclusive: u64) {
        let last_scheduled_range = self.last_scheduled_range.unwrap_or((0, 0));
        // If the end of the last scheduled range is greater than the end of the new range,
        // it means the entire new range was already scheduled before.
        if to_exclusive <= last_scheduled_range.1 {
            return;
        }
        // Otherwise, we make sure the new range starts after the last scheduled range.
        from = from.max(last_scheduled_range.1);
        self.ranges.insert(from, to_exclusive);
        self.last_scheduled_range = Some((from, to_exclusive));
    }

    fn len(&self) -> usize {
        self.ranges.len()
    }

    fn iter(&self) -> impl Iterator<Item = (u64, u64)> + '_ {
        self.ranges
            .iter()
            .map(|(from, to_exclusive)| (*from, *to_exclusive))
    }

    /// Remove the range from the pending_prune_ranges.
    fn remove(&mut self, from: &u64) {
        self.ranges.remove(from).unwrap();
    }

    /// Returns the current pruner_hi watermark, i.e. the first checkpoint that has not yet been
    /// pruned. This will be the first key in the pending_prune_ranges map. If the map is empty,
    /// then it is the last checkpoint that has been scheduled for pruning.
    fn get_pruner_hi(&self) -> u64 {
        self.ranges.keys().next().cloned().unwrap_or(
            self.last_scheduled_range
                .map(|(_, t)| t)
                // get_pruner_hi will generally not be called until we have scheduled something.
                // But return 0 just in case we called it earlier.
                .unwrap_or_default(),
        )
    }
}

/// The pruner task is responsible for deleting old data from the database. It will periodically
/// check the `watermarks` table to see if there is any data that should be pruned between the
/// `pruner_hi` (inclusive), and `reader_lo` (exclusive) checkpoints.
///
/// To ensure that the pruner does not interfere with reads that are still in flight, it respects
/// the watermark's `pruner_timestamp`, which records the time that `reader_lo` was last updated.
/// The task will not prune data until at least `config.delay()` has passed since `pruner_timestamp`
/// to give in-flight reads time to land.
///
/// If the `config` is `None`, the task will shutdown immediately.
pub(super) fn pruner<H: Handler + Send + Sync + 'static>(
    handler: Arc<H>,
    config: Option<PrunerConfig>,
    store: H::Store,
    metrics: Arc<IndexerMetrics>,
) -> Service {
    Service::new().spawn_aborting(async move {
        let Some(config) = config else {
            info!(pipeline = H::NAME, "Skipping pruner task");
            return Ok(());
        };

        info!(
            pipeline = H::NAME,
            "Starting pruner with config: {:?}", config
        );

        // The pruner can pause for a while, waiting for the delay imposed by the
        // `pruner_timestamp` to expire. In that case, the period between ticks should not be
        // compressed to make up for missed ticks.
        let mut poll = interval(config.interval());
        poll.set_missed_tick_behavior(MissedTickBehavior::Delay);

        // The pruner task will periodically output a log message at a higher log level to
        // demonstrate that it is making progress.
        let mut logger = WatermarkLogger::new("pruner");

        // Maintains the list of chunks that are ready to be pruned but not yet pruned.
        let mut pending_prune_ranges = PendingRanges::default();

        loop {
            poll.tick().await;

            // (1) Get the latest pruning bounds from the database.
            let watermark = {
                let guard = metrics
                    .watermark_pruner_read_latency
                    .with_label_values(&[H::NAME])
                    .start_timer();

                let Ok(mut conn) = store.connect().await else {
                    warn!(
                        pipeline = H::NAME,
                        "Pruner failed to connect, while fetching watermark"
                    );
                    continue;
                };

                match conn.pruner_watermark(H::NAME, config.delay()).await {
                    Ok(Some(current)) => {
                        guard.stop_and_record();
                        current
                    }

                    Ok(None) => {
                        guard.stop_and_record();
                        warn!(pipeline = H::NAME, "No watermark for pipeline, skipping");
                        continue;
                    }

                    Err(e) => {
                        guard.stop_and_record();
                        warn!(pipeline = H::NAME, "Failed to get watermark: {e}");
                        continue;
                    }
                }
            };

            // (2) Wait until this information can be acted upon.
            if !watermark.wait_for.is_zero() {
                debug!(pipeline = H::NAME, wait_for = ?watermark.wait_for, "Waiting to prune");
                tokio::time::sleep(watermark.wait_for).await;
            }

            // Tracks the current highest `pruner_hi` not yet written to db.
            let mut highest_pruned = watermark.pruner_hi;
            // Tracks the `pruner_hi` that has been written to the db.
            let mut highest_watermarked = watermark.pruner_hi;

            // (3) Collect all the new chunks that are ready to be pruned using the
            // prunable_range() API, then break it into chunks of max_chunk_size.
            if let Some(range) = watermark.prunable_range() {
                let mut from = range.start;
                let to = range.end;
                while from < to {
                    let chunk_end = (from + config.max_chunk_size).min(to);
                    pending_prune_ranges.schedule(from, chunk_end);
                    from = chunk_end;
                }
            }

            debug!(
                pipeline = H::NAME,
                "Number of chunks to prune: {}",
                pending_prune_ranges.len()
            );

            // (4) Prune chunk by chunk to avoid the task waiting on a long-running database
            // transaction. Spawn all tasks in parallel, but limit the number of concurrent tasks.
            let semaphore = Arc::new(Semaphore::new(config.prune_concurrency as usize));
            let mut tasks = FuturesUnordered::new();
            for (from, to_exclusive) in pending_prune_ranges.iter() {
                let semaphore = semaphore.clone();
                let metrics = metrics.clone();
                let handler = handler.clone();
                let db = store.clone();

                tasks.push(tokio::spawn(async move {
                    let _permit = semaphore.acquire().await.unwrap();
                    let result = prune_task_impl(metrics, db, handler, from, to_exclusive).await;
                    ((from, to_exclusive), result)
                }));
            }

            // (5) Wait for all tasks to finish. For each task, if it succeeds, remove the range
            // from the pending_prune_ranges. Otherwise the range will remain in the map and will be
            // retried in the next iteration.
            while let Some(r) = tasks.next().await {
                let ((from, to_exclusive), result) = r.unwrap();
                match result {
                    Ok(()) => {
                        pending_prune_ranges.remove(&from);
                        let pruner_hi = pending_prune_ranges.get_pruner_hi();
                        highest_pruned = highest_pruned.max(pruner_hi);
                    }
                    Err(e) => {
                        error!(
                            pipeline = H::NAME,
                            "Failed to prune data for range: {from} to {to_exclusive}: {e}"
                        );
                    }
                }

                if highest_pruned > highest_watermarked {
                    metrics
                        .watermark_pruner_hi
                        .with_label_values(&[H::NAME])
                        .set(highest_pruned as i64);

                    let guard = metrics
                        .watermark_pruner_write_latency
                        .with_label_values(&[H::NAME])
                        .start_timer();

                    let Ok(mut conn) = store.connect().await else {
                        warn!(
                            pipeline = H::NAME,
                            "Pruner failed to connect while updating watermark"
                        );
                        continue;
                    };

                    match conn.set_pruner_watermark(H::NAME, highest_pruned).await {
                        Err(e) => {
                            let elapsed = guard.stop_and_record();
                            error!(
                                pipeline = H::NAME,
                                elapsed_ms = elapsed * 1000.0,
                                "Failed to update pruner watermark: {e}"
                            )
                        }
                        Ok(true) => {
                            highest_watermarked = highest_pruned;
                            let elapsed = guard.stop_and_record();
                            logger.log::<H>(
                                LoggerWatermark::checkpoint(highest_watermarked),
                                elapsed,
                            );

                            metrics
                                .watermark_pruner_hi_in_db
                                .with_label_values(&[H::NAME])
                                .set(highest_watermarked as i64);
                        }
                        Ok(false) => {}
                    }
                }
            }
        }
    })
}

async fn prune_task_impl<H: Handler + Send + Sync + 'static>(
    metrics: Arc<IndexerMetrics>,
    db: H::Store,
    handler: Arc<H>,
    from: u64,
    to_exclusive: u64,
) -> Result<(), anyhow::Error> {
    metrics
        .total_pruner_chunks_attempted
        .with_label_values(&[H::NAME])
        .inc();

    let guard = metrics
        .pruner_delete_latency
        .with_label_values(&[H::NAME])
        .start_timer();

    let mut conn = db.connect().await?;

    debug!(pipeline = H::NAME, "Pruning from {from} to {to_exclusive}");

    let affected = match handler.prune(from, to_exclusive, &mut conn).await {
        Ok(affected) => {
            guard.stop_and_record();
            affected
        }

        Err(e) => {
            guard.stop_and_record();
            return Err(e);
        }
    };

    metrics
        .total_pruner_chunks_deleted
        .with_label_values(&[H::NAME])
        .inc();

    metrics
        .total_pruner_rows_deleted
        .with_label_values(&[H::NAME])
        .inc_by(affected as u64);

    Ok(())
}
