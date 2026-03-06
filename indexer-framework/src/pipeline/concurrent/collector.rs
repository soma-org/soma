// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use soma_futures::service::Service;
use tokio::sync::SetOnce;
use tokio::sync::mpsc;
use tokio::time::MissedTickBehavior;
use tokio::time::interval;
use tracing::debug;
use tracing::info;

use crate::metrics::CheckpointLagMetricReporter;
use crate::metrics::IndexerMetrics;
use crate::pipeline::CommitterConfig;
use crate::pipeline::IndexedCheckpoint;
use crate::pipeline::WatermarkPart;
use crate::pipeline::concurrent::BatchStatus;
use crate::pipeline::concurrent::BatchedRows;
use crate::pipeline::concurrent::Handler;

/// Processed values that are waiting to be written to the database.
struct PendingCheckpoint<H: Handler> {
    values: std::vec::IntoIter<H::Value>,
    watermark: WatermarkPart,
}

impl<H: Handler> PendingCheckpoint<H> {
    fn is_empty(&self) -> bool {
        let empty = self.values.len() == 0;
        debug_assert!(!empty || self.watermark.batch_rows == 0);
        empty
    }
}

impl<H: Handler> From<IndexedCheckpoint<H>> for PendingCheckpoint<H> {
    fn from(indexed: IndexedCheckpoint<H>) -> Self {
        let total_rows = indexed.values.len();
        Self {
            watermark: WatermarkPart {
                watermark: indexed.watermark,
                batch_rows: total_rows,
                total_rows,
            },
            values: indexed.values.into_iter(),
        }
    }
}

/// The collector task gathers rows into batches and sends them to the committer.
pub(super) fn collector<H: Handler + 'static>(
    handler: Arc<H>,
    config: CommitterConfig,
    mut rx: mpsc::Receiver<IndexedCheckpoint<H>>,
    tx: mpsc::Sender<BatchedRows<H>>,
    main_reader_lo: Arc<SetOnce<AtomicU64>>,
    metrics: Arc<IndexerMetrics>,
    min_eager_rows: usize,
    max_pending_rows: usize,
    max_watermark_updates: usize,
) -> Service {
    Service::new().spawn_aborting(async move {
        let mut poll = interval(config.collect_interval());
        poll.set_missed_tick_behavior(MissedTickBehavior::Delay);

        let checkpoint_lag_reporter = CheckpointLagMetricReporter::new_for_pipeline::<H>(
            &metrics.collected_checkpoint_timestamp_lag,
            &metrics.latest_collected_checkpoint_timestamp_lag_ms,
            &metrics.latest_collected_checkpoint,
        );

        let mut pending: BTreeMap<u64, PendingCheckpoint<H>> = BTreeMap::new();
        let mut pending_rows = 0;

        info!(pipeline = H::NAME, "Starting collector");

        // Wait for main_reader_lo to be initialized before processing any checkpoints.
        let reader_lo_atomic = main_reader_lo.wait().await;

        loop {
            // === IDLE: block until timer fires or enough data accumulates ===
            tokio::select! {
                biased;

                Some(mut indexed) = rx.recv(), if pending_rows < max_pending_rows => {
                    let reader_lo = reader_lo_atomic.load(Ordering::Relaxed);

                    metrics
                        .collector_reader_lo
                        .with_label_values(&[H::NAME])
                        .set(reader_lo as i64);

                    let mut recv_cps = 0usize;
                    let mut recv_rows = 0usize;
                    loop {
                        if indexed.checkpoint() < reader_lo {
                            indexed.values.clear();
                            metrics
                                .total_collector_skipped_checkpoints
                                .with_label_values(&[H::NAME])
                                .inc();
                        }

                        recv_cps += 1;
                        recv_rows += indexed.len();
                        pending_rows += indexed.len();
                        pending.insert(indexed.checkpoint(), indexed.into());

                        if pending_rows >= max_pending_rows {
                            break;
                        }

                        match rx.try_recv() {
                            Ok(next) => indexed = next,
                            Err(_) => break,
                        }
                    }

                    metrics
                        .total_collector_rows_received
                        .with_label_values(&[H::NAME])
                        .inc_by(recv_rows as u64);
                    metrics
                        .total_collector_checkpoints_received
                        .with_label_values(&[H::NAME])
                        .inc_by(recv_cps as u64);

                    if pending_rows < min_eager_rows {
                        continue;
                    }
                }

                _ = poll.tick() => {}
            }

            // === FLUSHING: send batches until pending is drained ===
            loop {
                let guard = metrics
                    .collector_gather_latency
                    .with_label_values(&[H::NAME])
                    .start_timer();

                let mut batch = H::Batch::default();
                let mut watermark = Vec::new();
                let mut batch_len = 0;

                loop {
                    let Some(mut entry) = pending.first_entry() else {
                        break;
                    };

                    if watermark.len() >= max_watermark_updates {
                        break;
                    }

                    let indexed = entry.get_mut();
                    let before = indexed.values.len();
                    let status = handler.batch(&mut batch, &mut indexed.values);
                    let taken = before - indexed.values.len();

                    batch_len += taken;
                    watermark.push(indexed.watermark.take(taken));
                    if indexed.is_empty() {
                        checkpoint_lag_reporter.report_lag(
                            indexed.watermark.checkpoint(),
                            indexed.watermark.timestamp_ms(),
                        );
                        entry.remove();
                    }

                    if status == BatchStatus::Ready {
                        break;
                    }
                }

                let elapsed = guard.stop_and_record();
                debug!(
                    pipeline = H::NAME,
                    elapsed_ms = elapsed * 1000.0,
                    rows = batch_len,
                    "Gathered batch",
                );

                metrics
                    .total_collector_batches_created
                    .with_label_values(&[H::NAME])
                    .inc();

                metrics
                    .collector_batch_size
                    .with_label_values(&[H::NAME])
                    .observe(batch_len as f64);

                pending_rows -= batch_len;

                let batched_rows = BatchedRows {
                    batch,
                    batch_len,
                    watermark,
                };
                if tx.send(batched_rows).await.is_err() {
                    info!(
                        pipeline = H::NAME,
                        "Committer closed channel, stopping collector"
                    );
                    return Ok(());
                }

                if pending.is_empty() {
                    break;
                }
            }

            if rx.is_closed() && rx.is_empty() && pending_rows == 0 {
                info!(
                    pipeline = H::NAME,
                    "Processor closed channel, pending rows empty, stopping collector",
                );
                break;
            }
        }

        Ok(())
    })
}
