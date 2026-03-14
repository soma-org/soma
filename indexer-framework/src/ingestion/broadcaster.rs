// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use futures::Stream;
use soma_futures::service::Service;
use soma_futures::stream::Break;
use soma_futures::stream::TrySpawnStreamExt;
use soma_futures::task::TaskGuard;
use tokio::sync::mpsc;
use tokio::sync::watch;
use tokio_stream::StreamExt;
use tracing::debug;
use tracing::info;

use crate::config::ConcurrencyConfig;
use crate::ingestion::IngestionConfig;
use crate::ingestion::ingestion_client::IngestionClient;
use crate::metrics::IngestionMetrics;
use crate::types::full_checkpoint_content::Checkpoint;

/// Broadcaster task that manages checkpoint flow and spawns broadcast tasks for ranges.
///
/// This task:
/// 1. Maintains an ingest_hi based on subscriber feedback.
/// 2. Spawns ingestion tasks for the requested checkpoint range.
/// 3. The task will shut down if the `checkpoints` range completes.
pub(super) fn broadcaster<R>(
    checkpoints: R,
    next_sequential_checkpoint: Option<u64>,
    config: IngestionConfig,
    client: IngestionClient,
    mut commit_hi_rx: mpsc::UnboundedReceiver<(&'static str, u64)>,
    subscribers: Vec<mpsc::Sender<Arc<Checkpoint>>>,
    metrics: Arc<IngestionMetrics>,
) -> Service
where
    R: std::ops::RangeBounds<u64> + Send + 'static,
{
    Service::new().spawn_aborting(async move {
        info!("Starting broadcaster");

        // Extract start and end from the range bounds
        let start_cp = match checkpoints.start_bound() {
            std::ops::Bound::Included(&n) => n,
            std::ops::Bound::Excluded(&n) => n + 1,
            std::ops::Bound::Unbounded => 0,
        };

        let end_cp = match checkpoints.end_bound() {
            std::ops::Bound::Included(&n) => n.saturating_add(1),
            std::ops::Bound::Excluded(&n) => n,
            std::ops::Bound::Unbounded => u64::MAX,
        };

        let buffer_size = config.checkpoint_buffer_size as u64;

        // Track subscriber watermarks
        let mut subscribers_hi = HashMap::<&'static str, u64>::new();

        // Initialize ingest_hi watch channel, seeded from the sequential checkpoint if present.
        let initial_hi = next_sequential_checkpoint.unwrap_or(start_cp);
        let (ingest_hi_tx, ingest_hi_rx) = watch::channel(initial_hi.saturating_add(buffer_size));
        let ingest_hi_rx = next_sequential_checkpoint.is_some().then_some(&ingest_hi_rx);

        // Move subscribers directly into the ingestion task so that subscriber channels
        // close as soon as all checkpoints have been broadcast. Previously, wrapping in
        // Arc kept the original senders alive in this scope, which prevented processors
        // (especially 0-row pipelines) from detecting stream closure promptly.
        let ingest_guard = ingest_and_broadcast_range(
            start_cp,
            end_cp,
            config.retry_interval(),
            config.ingest_concurrency.clone(),
            ingest_hi_rx.cloned(),
            client,
            subscribers,
            metrics.clone(),
        );

        let mut ingest_future = ingest_guard;

        loop {
            tokio::select! {
                // Subscriber watermark update
                Some((name, hi)) = commit_hi_rx.recv() => {
                    subscribers_hi.insert(name, hi);

                    if let Some(min_hi) = subscribers_hi.values().copied().min() {
                        let new_ingest_hi = min_hi.saturating_add(buffer_size);
                        let _ = ingest_hi_tx.send(new_ingest_hi);
                    }
                }

                // Handle ingestion completion
                ingestion_result = &mut ingest_future => {
                    match ingestion_result
                        .context("Ingestion task panicked, stopping broadcaster")?
                    {
                        Ok(()) => {},
                        Err(Break::Break) => {
                            break;
                        }
                        Err(Break::Err(e)) => {
                            return Err(anyhow::anyhow!(e).context("Ingestion task failed, stopping broadcaster"));
                        }
                    }
                    break;
                }
            }
        }

        info!("Checkpoints done, stopping broadcaster");
        Ok(())
    })
}

/// Wraps a checkpoint range in an async stream gated by `ingest_hi`. Items are only yielded
/// when the backpressure window allows.
fn backpressured_checkpoint_stream(
    start: u64,
    end: u64,
    ingest_hi_rx: Option<watch::Receiver<u64>>,
) -> impl Stream<Item = u64> {
    futures::stream::unfold((start, ingest_hi_rx), move |(cp, rx)| async move {
        if cp >= end {
            return None;
        }

        let Some(mut rx) = rx else {
            // No backpressure, just yield checkpoints as fast as possible.
            return Some((cp, (cp + 1, None)));
        };

        if rx.wait_for(|hi| cp < *hi).await.is_err() {
            return None;
        }

        Some((cp, (cp + 1, Some(rx))))
    })
}

/// Fetch and broadcasts checkpoints from a range [start..end) to subscribers.
fn ingest_and_broadcast_range(
    start: u64,
    end: u64,
    retry_interval: Duration,
    ingest_concurrency: ConcurrencyConfig,
    ingest_hi_rx: Option<watch::Receiver<u64>>,
    client: IngestionClient,
    subscribers: Vec<mpsc::Sender<Arc<Checkpoint>>>,
    metrics: Arc<IngestionMetrics>,
) -> TaskGuard<Result<(), Break<super::error::Error>>> {
    TaskGuard::new(tokio::spawn(async move {
        let report_metrics = metrics.clone();
        backpressured_checkpoint_stream(start, end, ingest_hi_rx)
            .try_for_each_broadcast_spawned(
                ingest_concurrency.into(),
                |cp| {
                    let client = client.clone();
                    async move {
                        let checkpoint = client.wait_for(cp, retry_interval).await?;
                        debug!(checkpoint = cp, "Fetched checkpoint");
                        Ok(checkpoint)
                    }
                },
                subscribers,
                move |stats| {
                    report_metrics
                        .ingestion_concurrency_limit
                        .set(stats.limit as i64);
                    report_metrics
                        .ingestion_concurrency_inflight
                        .set(stats.inflight as i64);
                },
            )
            .await
    }))
}

