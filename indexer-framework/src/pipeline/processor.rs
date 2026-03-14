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

use async_trait::async_trait;

use crate::config::ConcurrencyConfig;
use crate::metrics::CheckpointLagMetricReporter;
use crate::metrics::IndexerMetrics;
use crate::pipeline::IndexedCheckpoint;
use crate::types::full_checkpoint_content::Checkpoint;

/// If the processor needs to retry processing a checkpoint, it will wait this long initially.
const INITIAL_RETRY_INTERVAL: Duration = Duration::from_millis(100);

/// If the processor needs to retry processing a checkpoint, it will wait at most this long.
const MAX_RETRY_INTERVAL: Duration = Duration::from_secs(5);

/// Implementors of this trait are responsible for transforming checkpoint into rows for their
/// table.
#[async_trait]
pub trait Processor: Send + Sync + 'static {
    /// Used to identify the pipeline in logs and metrics.
    const NAME: &'static str;

    /// The type of value being inserted by the handler.
    type Value: Send + Sync + 'static;

    /// The processing logic for turning a checkpoint into rows of the table.
    ///
    /// All errors returned from this method are treated as transient and will be retried
    /// indefinitely with exponential backoff.
    async fn process(&self, checkpoint: &Arc<Checkpoint>) -> anyhow::Result<Vec<Self::Value>>;
}

/// The processor task is responsible for taking checkpoint data and breaking it down into rows
/// ready to commit.
pub(super) fn processor<P: Processor>(
    processor: Arc<P>,
    rx: mpsc::Receiver<Arc<Checkpoint>>,
    tx: mpsc::Sender<IndexedCheckpoint<P>>,
    metrics: Arc<IndexerMetrics>,
    concurrency: ConcurrencyConfig,
) -> Service {
    Service::new().spawn_aborting(async move {
        info!(pipeline = P::NAME, "Starting processor");
        let checkpoint_lag_reporter = CheckpointLagMetricReporter::new_for_pipeline::<P>(
            &metrics.processed_checkpoint_timestamp_lag,
            &metrics.latest_processed_checkpoint_timestamp_lag_ms,
            &metrics.latest_processed_checkpoint,
        );

        let report_metrics = metrics.clone();
        match ReceiverStream::new(rx)
            .try_for_each_send_spawned(
                concurrency.into(),
                |checkpoint| {
                    let metrics = metrics.clone();
                    let checkpoint_lag_reporter = checkpoint_lag_reporter.clone();
                    let processor = processor.clone();

                    async move {
                        metrics
                            .total_handler_checkpoints_received
                            .with_label_values(&[P::NAME])
                            .inc();

                        let guard = metrics
                            .handler_checkpoint_latency
                            .with_label_values(&[P::NAME])
                            .start_timer();

                        // Retry processing with exponential backoff
                        let backoff = ExponentialBackoff {
                            initial_interval: INITIAL_RETRY_INTERVAL,
                            current_interval: INITIAL_RETRY_INTERVAL,
                            max_interval: MAX_RETRY_INTERVAL,
                            max_elapsed_time: None,
                            ..Default::default()
                        };

                        let values = backoff::future::retry(backoff, || async {
                            processor.process(&checkpoint).await.map_err(backoff::Error::transient)
                        })
                        .await?;

                        let elapsed = guard.stop_and_record();

                        let epoch = checkpoint.summary.epoch;
                        let cp_sequence_number = checkpoint.summary.sequence_number;
                        let tx_hi = checkpoint.summary.network_total_transactions;
                        let timestamp_ms = checkpoint.summary.timestamp_ms;

                        debug!(
                            pipeline = P::NAME,
                            checkpoint = cp_sequence_number,
                            elapsed_ms = elapsed * 1000.0,
                            "Processed checkpoint",
                        );

                        checkpoint_lag_reporter.report_lag(cp_sequence_number, timestamp_ms);

                        metrics
                            .total_handler_checkpoints_processed
                            .with_label_values(&[P::NAME])
                            .inc();

                        metrics
                            .total_handler_rows_created
                            .with_label_values(&[P::NAME])
                            .inc_by(values.len() as u64);

                        Ok(IndexedCheckpoint::new(
                            epoch,
                            cp_sequence_number,
                            tx_hi,
                            timestamp_ms,
                            values,
                        ))
                    }
                },
                tx,
                move |stats| {
                    report_metrics
                        .processor_concurrency_limit
                        .with_label_values(&[P::NAME])
                        .set(stats.limit as i64);
                    report_metrics
                        .processor_concurrency_inflight
                        .with_label_values(&[P::NAME])
                        .set(stats.inflight as i64);
                },
            )
            .await
        {
            Ok(()) => {
                info!(pipeline = P::NAME, "Checkpoints done, stopping processor");
            }

            Err(Break::Break) => {
                info!(pipeline = P::NAME, "Channel closed, stopping processor");
            }

            Err(Break::Err(e)) => {
                error!(pipeline = P::NAME, "Error from handler: {e}");
                return Err(e.context(format!("Error from processor {}", P::NAME)));
            }
        };

        Ok(())
    })
}
