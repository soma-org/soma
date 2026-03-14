// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use soma_futures::service::Service;
use tokio::time::interval;
use tracing::debug;
use tracing::info;
use tracing::warn;

use crate::metrics::IndexerMetrics;
use crate::pipeline::concurrent::Handler;
use crate::pipeline::concurrent::PrunerConfig;
use crate::store::Connection;
use crate::store::Store;

/// The reader watermark task is responsible for updating the `reader_lo` and `pruner_timestamp`
/// values for a pipeline's row in the watermark table, based on the pruner configuration, and the
/// committer's progress.
///
/// `reader_lo` is the lowest checkpoint that readers are allowed to read from with a guarantee of
/// data availability for this pipeline, and `pruner_timestamp` is the timestamp at which this task
/// last updated that watermark. The timestamp is always fetched from the database (not from the
/// indexer or the reader), to avoid issues with drift between clocks.
///
/// If there is no pruner configuration, this task will immediately exit.
pub(super) fn reader_watermark<H: Handler + 'static>(
    config: Option<PrunerConfig>,
    store: H::Store,
    metrics: Arc<IndexerMetrics>,
) -> Service {
    Service::new().spawn_aborting(async move {
        let Some(config) = config else {
            info!(pipeline = H::NAME, "Skipping reader watermark task");
            return Ok(());
        };

        let mut poll = interval(config.interval());

        loop {
            poll.tick().await;

            let Ok(mut conn) = store.connect().await else {
                warn!(pipeline = H::NAME, "Reader watermark task failed to get connection for DB");
                continue;
            };

            let current = match conn.reader_watermark(H::NAME).await {
                Ok(Some(current)) => current,

                Ok(None) => {
                    warn!(pipeline = H::NAME, "No watermark for pipeline, skipping");
                    continue;
                }

                Err(e) => {
                    warn!(pipeline = H::NAME, "Failed to get current watermark: {e}");
                    continue;
                }
            };

            // Calculate the new reader watermark based on the current high watermark.
            let mut new_reader_lo =
                (current.checkpoint_hi_inclusive as u64 + 1).saturating_sub(config.retention);

            // Zero-gap guarantee: clamp reader_lo against the external watermark floor
            // (e.g., BigTable's checkpoint_hi) so we never prune data that the external
            // store hasn't confirmed indexing yet.
            if let Some(floor) = &config.external_watermark_floor {
                let bt_hi = floor.load(std::sync::atomic::Ordering::Relaxed);
                if bt_hi > 0 {
                    new_reader_lo = new_reader_lo.min(bt_hi + 1);
                } else {
                    // External store hasn't reported yet — don't allow any pruning.
                    new_reader_lo = 0;
                }
            }

            if new_reader_lo <= current.reader_lo as u64 {
                debug!(
                    pipeline = H::NAME,
                    current = current.reader_lo,
                    new = new_reader_lo,
                    "No change to reader watermark",
                );
                continue;
            }

            metrics.watermark_reader_lo.with_label_values(&[H::NAME]).set(new_reader_lo as i64);

            let Ok(updated) = conn.set_reader_watermark(H::NAME, new_reader_lo).await else {
                warn!(pipeline = H::NAME, "Failed to update reader watermark");
                continue;
            };

            if updated {
                info!(pipeline = H::NAME, new_reader_lo, "Watermark");

                metrics
                    .watermark_reader_lo_in_db
                    .with_label_values(&[H::NAME])
                    .set(new_reader_lo as i64);
            }
        }
    })
}
