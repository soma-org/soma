// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::collections::btree_map::Entry;
use std::sync::Arc;

use indexer_store_traits::pipeline_task;
use soma_futures::service::Service;
use tokio::sync::mpsc;
use tracing::debug;
use tracing::error;
use tracing::info;
use tracing::warn;

use crate::metrics::CheckpointLagMetricReporter;
use crate::metrics::IndexerMetrics;
use crate::pipeline::CommitterConfig;
use crate::pipeline::WARN_PENDING_WATERMARKS;
use crate::pipeline::WatermarkPart;
use crate::pipeline::concurrent::Handler;
use crate::pipeline::logging::WatermarkLogger;
use crate::store::CommitterWatermark;
use crate::store::Connection;
use crate::store::Store;

/// The watermark task is responsible for keeping track of a pipeline's out-of-order commits and
/// updating its row in the `watermarks` table when a continuous run of checkpoints have landed
/// since the last watermark update.
///
/// It receives watermark "parts" that detail the proportion of each checkpoint's data that has been
/// written out by the committer and periodically (on a configurable interval) checks if the
/// watermark for the pipeline can be pushed forward. The watermark can be pushed forward if there
/// is one or more complete (all data for that checkpoint written out) watermarks spanning
/// contiguously from the current high watermark into the future.
///
/// If it detects that more than [WARN_PENDING_WATERMARKS] watermarks have built up, it will issue a
/// warning, as this could be the indication of a memory leak, and the caller probably intended to
/// run the indexer with watermarking disabled (e.g. if they are running a backfill).
///
/// The task will shutdown if the `rx` channel closes and the watermark cannot be progressed.
pub(super) fn commit_watermark<H: Handler + 'static>(
    mut next_checkpoint: u64,
    config: CommitterConfig,
    mut rx: mpsc::Receiver<Vec<WatermarkPart>>,
    store: H::Store,
    task: Option<String>,
    metrics: Arc<IndexerMetrics>,
) -> Service {
    let pipeline_task =
        pipeline_task(H::NAME, task.as_deref(), <H::Store as Store>::DELIMITER);
    Service::new().spawn_aborting(async move {
        // To correctly update the watermark, the task tracks the watermark it last tried to write
        // and the watermark parts for any checkpoints that have been written since then
        // ("pre-committed"). After each batch is written, the task will try to progress the
        // watermark as much as possible without going over any holes in the sequence of
        // checkpoints (entirely missing watermarks, or incomplete watermarks).
        let mut precommitted: BTreeMap<u64, WatermarkPart> = BTreeMap::new();

        // The watermark task will periodically output a log message at a higher log level to
        // demonstrate that the pipeline is making progress.
        let mut logger = WatermarkLogger::new("concurrent_committer");

        let checkpoint_lag_reporter = CheckpointLagMetricReporter::new_for_pipeline::<H>(
            &metrics.watermarked_checkpoint_timestamp_lag,
            &metrics.latest_watermarked_checkpoint_timestamp_lag_ms,
            &metrics.watermark_checkpoint_in_db,
        );

        info!(
            pipeline = H::NAME,
            next_checkpoint, "Starting commit watermark task"
        );

        let mut next_wake = tokio::time::Instant::now();
        let mut pending_watermark = None;

        loop {
            let mut should_write_db = false;

            tokio::select! {
                () = tokio::time::sleep_until(next_wake) => {
                    // Schedule next wake immediately, so the timer effectively runs in parallel
                    // with the commit logic below.
                    next_wake = config.watermark_interval_with_jitter();
                    should_write_db = true;
                }
                Some(parts) = rx.recv() => {
                    for part in parts {
                        match precommitted.entry(part.checkpoint()) {
                            Entry::Vacant(entry) => {
                                entry.insert(part);
                            }

                            Entry::Occupied(mut entry) => {
                                entry.get_mut().add(part);
                            }
                        }
                    }
                }
            }

            // Advance the watermark through contiguous precommitted entries on every
            // iteration, not just when the DB write timer fires. This ensures commit_hi
            // feedback reaches the broadcaster immediately.
            let guard = metrics
                .watermark_gather_latency
                .with_label_values(&[H::NAME])
                .start_timer();

            while let Some(pending) = precommitted.first_entry() {
                let part = pending.get();

                // Some rows from the next watermark have not landed yet.
                if !part.is_complete() {
                    break;
                }

                match next_checkpoint.cmp(&part.watermark.checkpoint_hi_inclusive) {
                    // Next pending checkpoint is from the future.
                    Ordering::Less => break,

                    // This is the next checkpoint -- include it.
                    Ordering::Equal => {
                        pending_watermark = Some(pending.remove().watermark);
                        next_checkpoint += 1;
                    }

                    // Next pending checkpoint is in the past. Out of order watermarks can
                    // be encountered when a pipeline is starting up, because ingestion
                    // must start at the lowest checkpoint across all pipelines, or because
                    // of a backfill, where the initial checkpoint has been overridden.
                    Ordering::Greater => {
                        // Track how many we see to make sure it doesn't grow without bound.
                        metrics
                            .total_watermarks_out_of_order
                            .with_label_values(&[H::NAME])
                            .inc();

                        pending.remove();
                    }
                }
            }

            let elapsed = guard.stop_and_record();

            if let Some(ref watermark) = pending_watermark {
                metrics
                    .watermark_epoch
                    .with_label_values(&[H::NAME])
                    .set(watermark.epoch as i64);

                metrics
                    .watermark_checkpoint
                    .with_label_values(&[H::NAME])
                    .set(watermark.checkpoint_hi_inclusive as i64);

                metrics
                    .watermark_transaction
                    .with_label_values(&[H::NAME])
                    .set(watermark.tx_hi as i64);

                metrics
                    .watermark_timestamp_ms
                    .with_label_values(&[H::NAME])
                    .set(watermark.timestamp_ms_hi_inclusive as i64);

                debug!(
                    pipeline = H::NAME,
                    elapsed_ms = elapsed * 1000.0,
                    watermark = watermark.checkpoint_hi_inclusive,
                    timestamp = %watermark.timestamp(),
                    pending = precommitted.len(),
                    "Gathered watermarks",
                );
            }

            if precommitted.len() > WARN_PENDING_WATERMARKS {
                warn!(
                    pipeline = H::NAME,
                    pending = precommitted.len(),
                    "Pipeline has a large number of pending commit watermarks",
                );
            }

            // DB writes are deferred to the timer interval to avoid excessive DB load.
            if should_write_db {
                if let Some(watermark) = pending_watermark.take() {
                    if write_watermark::<H>(
                        &store,
                        &pipeline_task,
                        &watermark,
                        &mut logger,
                        &checkpoint_lag_reporter,
                        &metrics,
                    )
                    .await
                    .is_err()
                    {
                        pending_watermark = Some(watermark);
                    }
                }
            }

            if rx.is_closed() && rx.is_empty() {
                info!(pipeline = H::NAME, "Committer closed channel");
                break;
            }
        }

        if let Some(watermark) = pending_watermark {
            if write_watermark::<H>(
                &store,
                &pipeline_task,
                &watermark,
                &mut logger,
                &checkpoint_lag_reporter,
                &metrics,
            )
            .await
            .is_err()
            {
                warn!(
                    pipeline = H::NAME,
                    ?watermark,
                    "Failed to write final watermark on shutdown, will not retry",
                );
            }
        }

        info!(pipeline = H::NAME, "Stopping committer watermark task");
        Ok(())
    })
}

/// Write the watermark to DB and update metrics. Returns `Err` on failure so the caller can
/// preserve the watermark for retry on the next tick.
async fn write_watermark<H: Handler>(
    store: &H::Store,
    pipeline_task: &str,
    watermark: &CommitterWatermark,
    logger: &mut WatermarkLogger,
    checkpoint_lag_reporter: &CheckpointLagMetricReporter,
    metrics: &IndexerMetrics,
) -> Result<(), ()> {
    let Ok(mut conn) = store.connect().await else {
        warn!(
            pipeline = H::NAME,
            "Commit watermark task failed to get connection for DB"
        );
        return Err(());
    };

    let guard = metrics
        .watermark_commit_latency
        .with_label_values(&[H::NAME])
        .start_timer();

    // TODO: If initial_watermark is empty, when we update watermark
    // for the first time, we should also update the low watermark.
    match conn
        .set_committer_watermark(pipeline_task, *watermark)
        .await
    {
        Err(e) => {
            let elapsed = guard.stop_and_record();
            error!(
                pipeline = H::NAME,
                elapsed_ms = elapsed * 1000.0,
                ?watermark,
                "Error updating commit watermark: {e}",
            );
            Err(())
        }

        Ok(true) => {
            let elapsed = guard.stop_and_record();

            logger.log::<H>(watermark, elapsed);

            checkpoint_lag_reporter.report_lag(
                watermark.checkpoint_hi_inclusive,
                watermark.timestamp_ms_hi_inclusive,
            );

            metrics
                .watermark_epoch_in_db
                .with_label_values(&[H::NAME])
                .set(watermark.epoch as i64);

            metrics
                .watermark_transaction_in_db
                .with_label_values(&[H::NAME])
                .set(watermark.tx_hi as i64);

            metrics
                .watermark_timestamp_in_db_ms
                .with_label_values(&[H::NAME])
                .set(watermark.timestamp_ms_hi_inclusive as i64);

            Ok(())
        }
        Ok(false) => Ok(()),
    }
}
