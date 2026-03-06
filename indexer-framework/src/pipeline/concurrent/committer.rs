// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::time::Duration;

use backoff::ExponentialBackoff;
use soma_futures::service::Service;
use soma_futures::stream::Break;
use soma_futures::stream::TrySpawnStreamExt;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tracing::debug;
use tracing::error;
use tracing::info;
use tracing::warn;

use crate::metrics::CheckpointLagMetricReporter;
use crate::metrics::IndexerMetrics;
use crate::pipeline::CommitterConfig;
use crate::pipeline::WatermarkPart;
use crate::pipeline::concurrent::BatchedRows;
use crate::pipeline::concurrent::Handler;
use crate::store::Store;

/// If the committer needs to retry a commit, it will wait this long initially.
const INITIAL_RETRY_INTERVAL: Duration = Duration::from_millis(100);

/// If the committer needs to retry a commit, it will wait at most this long between retries.
const MAX_RETRY_INTERVAL: Duration = Duration::from_secs(1);

/// The committer task is responsible for writing batches of rows to the database. It receives
/// batches on `rx` and writes them out to the `db` concurrently (`config.write_concurrency`
/// controls the degree of fan-out).
///
/// The writing of each batch will be repeatedly retried on an exponential back-off until it
/// succeeds. Once the write succeeds, the [WatermarkPart]s for that batch are sent on `tx` to the
/// watermark task.
///
/// This task will shutdown if its receiver or sender channels are closed.
pub(super) fn committer<H: Handler + 'static>(
    handler: Arc<H>,
    config: CommitterConfig,
    rx: mpsc::Receiver<BatchedRows<H>>,
    tx: mpsc::Sender<Vec<WatermarkPart>>,
    db: H::Store,
    metrics: Arc<IndexerMetrics>,
) -> Service {
    Service::new().spawn_aborting(async move {
        info!(pipeline = H::NAME, "Starting committer");
        let checkpoint_lag_reporter = CheckpointLagMetricReporter::new_for_pipeline::<H>(
            &metrics.partially_committed_checkpoint_timestamp_lag,
            &metrics.latest_partially_committed_checkpoint_timestamp_lag_ms,
            &metrics.latest_partially_committed_checkpoint,
        );

        match ReceiverStream::new(rx)
            .try_for_each_spawned(
                config.write_concurrency,
                |BatchedRows {
                     batch,
                     batch_len,
                     watermark,
                 }| {
                    let batch = Arc::new(batch);
                    let handler = handler.clone();
                    let tx = tx.clone();
                    let db = db.clone();
                    let metrics = metrics.clone();
                    let checkpoint_lag_reporter = checkpoint_lag_reporter.clone();

                    // Repeatedly try to get a connection to the DB and write the batch. Use an
                    // exponential backoff in case the failure is due to contention over the DB
                    // connection pool.
                    let backoff = ExponentialBackoff {
                        initial_interval: INITIAL_RETRY_INTERVAL,
                        current_interval: INITIAL_RETRY_INTERVAL,
                        max_interval: MAX_RETRY_INTERVAL,
                        max_elapsed_time: None,
                        ..Default::default()
                    };

                    let highest_checkpoint = watermark.iter().map(|w| w.checkpoint()).max();
                    let highest_checkpoint_timestamp =
                        watermark.iter().map(|w| w.timestamp_ms()).max();

                    use backoff::Error as BE;
                    let commit = move || {
                        let batch = batch.clone();
                        let handler = handler.clone();
                        let db = db.clone();
                        let metrics = metrics.clone();
                        let checkpoint_lag_reporter = checkpoint_lag_reporter.clone();
                        async move {
                            if batch_len == 0 {
                                return Ok(());
                            }

                            metrics
                                .total_committer_batches_attempted
                                .with_label_values(&[H::NAME])
                                .inc();

                            let guard = metrics
                                .committer_commit_latency
                                .with_label_values(&[H::NAME])
                                .start_timer();

                            let mut conn = db.connect().await.map_err(|e| {
                                warn!(
                                    pipeline = H::NAME,
                                    "Committer failed to get connection for DB"
                                );

                                metrics
                                    .total_committer_batches_failed
                                    .with_label_values(&[H::NAME])
                                    .inc();

                                BE::transient(Break::Err(e))
                            })?;

                            let affected = handler.commit(&batch, &mut conn).await;
                            let elapsed = guard.stop_and_record();

                            match affected {
                                Ok(affected) => {
                                    debug!(
                                        pipeline = H::NAME,
                                        elapsed_ms = elapsed * 1000.0,
                                        affected,
                                        committed = batch_len,
                                        "Wrote batch",
                                    );

                                    checkpoint_lag_reporter.report_lag(
                                        // unwrap is safe because we would have returned if
                                        // values is empty.
                                        highest_checkpoint.unwrap(),
                                        highest_checkpoint_timestamp.unwrap(),
                                    );

                                    metrics
                                        .total_committer_batches_succeeded
                                        .with_label_values(&[H::NAME])
                                        .inc();

                                    metrics
                                        .total_committer_rows_committed
                                        .with_label_values(&[H::NAME])
                                        .inc_by(batch_len as u64);

                                    metrics
                                        .total_committer_rows_affected
                                        .with_label_values(&[H::NAME])
                                        .inc_by(affected as u64);

                                    metrics
                                        .committer_tx_rows
                                        .with_label_values(&[H::NAME])
                                        .observe(affected as f64);

                                    Ok(())
                                }

                                Err(e) => {
                                    warn!(
                                        pipeline = H::NAME,
                                        elapsed_ms = elapsed * 1000.0,
                                        committed = batch_len,
                                        "Error writing batch: {e}",
                                    );

                                    metrics
                                        .total_committer_batches_failed
                                        .with_label_values(&[H::NAME])
                                        .inc();

                                    Err(BE::transient(Break::Err(e)))
                                }
                            }
                        }
                    };

                    async move {
                        // Double check that the commit actually went through, (this backoff should
                        // not produce any permanent errors, but if it does, we need to shutdown
                        // the pipeline).
                        backoff::future::retry(backoff, commit).await?;
                        if tx.send(watermark).await.is_err() {
                            info!(pipeline = H::NAME, "Watermark closed channel");
                            return Err(Break::<anyhow::Error>::Break);
                        }

                        Ok(())
                    }
                },
            )
            .await
        {
            Ok(()) => {
                info!(pipeline = H::NAME, "Batches done, stopping committer");
                Ok(())
            }

            Err(Break::Break) => {
                info!(pipeline = H::NAME, "Channels closed, stopping committer");
                Ok(())
            }

            Err(Break::Err(e)) => {
                error!(pipeline = H::NAME, "Error from committer: {e}");
                Err(e.context(format!("Error from committer {}", H::NAME)))
            }
        }
    })
}
