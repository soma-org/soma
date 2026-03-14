// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

// Allow use of `unbounded_channel` in `ingestion` -- it is used by the regulator task to receive
// feedback. Traffic through this task should be minimal, but if a bound is applied to it and that
// bound is hit, the indexer could deadlock.
#![allow(clippy::disallowed_methods)]

use std::sync::Arc;
use std::time::Duration;

use prometheus::Registry;
use serde::Deserialize;
use serde::Serialize;
use soma_futures::service::Service;
use tokio::sync::mpsc;

pub use crate::config::ConcurrencyConfig as IngestConcurrencyConfig;
use crate::ingestion::broadcaster::broadcaster;
use crate::ingestion::error::Error;
use crate::ingestion::error::Result;
use crate::ingestion::ingestion_client::IngestionClient;
use crate::ingestion::ingestion_client::IngestionClientArgs;
use crate::metrics::IngestionMetrics;
use crate::types::full_checkpoint_content::Checkpoint;

mod broadcaster;
pub(crate) mod decode;
pub mod error;
pub mod ingestion_client;
#[cfg(test)]
pub(crate) mod test_utils;

/// Combined arguments for ingestion clients.
#[derive(clap::Args, Clone, Debug, Default)]
pub struct ClientArgs {
    #[clap(flatten)]
    pub ingestion: IngestionClientArgs,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct IngestionConfig {
    /// Maximum size of checkpoint backlog across all workers downstream of the ingestion service.
    pub checkpoint_buffer_size: usize,

    /// Concurrency control for checkpoint ingestion.
    pub ingest_concurrency: IngestConcurrencyConfig,

    /// Polling interval to retry fetching checkpoints that do not exist, in milliseconds.
    pub retry_interval_ms: u64,
}

pub struct IngestionService {
    config: IngestionConfig,
    ingestion_client: IngestionClient,
    commit_hi_tx: mpsc::UnboundedSender<(&'static str, u64)>,
    commit_hi_rx: mpsc::UnboundedReceiver<(&'static str, u64)>,
    subscribers: Vec<mpsc::Sender<Arc<Checkpoint>>>,
    metrics: Arc<IngestionMetrics>,
}

impl IngestionConfig {
    pub fn retry_interval(&self) -> Duration {
        Duration::from_millis(self.retry_interval_ms)
    }
}

impl IngestionService {
    /// Create a new instance of the ingestion service, responsible for fetching checkpoints and
    /// disseminating them to subscribers.
    pub fn new(
        args: ClientArgs,
        config: IngestionConfig,
        metrics_prefix: Option<&str>,
        registry: &Registry,
    ) -> Result<Self> {
        let metrics = IngestionMetrics::new(metrics_prefix, registry);
        let ingestion_client = IngestionClient::new(args.ingestion, metrics.clone())?;

        let subscribers = Vec::new();
        let (commit_hi_tx, commit_hi_rx) = mpsc::unbounded_channel();
        Ok(Self { config, ingestion_client, commit_hi_tx, commit_hi_rx, subscribers, metrics })
    }

    /// The ingestion client this service uses to fetch checkpoints.
    pub(crate) fn ingestion_client(&self) -> &IngestionClient {
        &self.ingestion_client
    }

    /// Access to the ingestion metrics.
    pub(crate) fn metrics(&self) -> &Arc<IngestionMetrics> {
        &self.metrics
    }

    /// Add a new subscription to the ingestion service.
    ///
    /// Returns the channel to receive checkpoints from and the channel to send commit_hi values to.
    pub fn subscribe(
        &mut self,
    ) -> (mpsc::Receiver<Arc<Checkpoint>>, mpsc::UnboundedSender<(&'static str, u64)>) {
        let (sender, receiver) = mpsc::channel(self.config.checkpoint_buffer_size);
        self.subscribers.push(sender);
        (receiver, self.commit_hi_tx.clone())
    }

    /// Start the ingestion service as a background task, consuming it in the process.
    pub async fn run<R>(
        self,
        checkpoints: R,
        next_sequential_checkpoint: Option<u64>,
    ) -> Result<Service>
    where
        R: std::ops::RangeBounds<u64> + Send + 'static,
    {
        let IngestionService {
            config,
            ingestion_client,
            commit_hi_tx: _,
            commit_hi_rx,
            subscribers,
            metrics,
        } = self;

        if subscribers.is_empty() {
            return Err(Error::NoSubscribers);
        }

        Ok(broadcaster(
            checkpoints,
            next_sequential_checkpoint,
            config,
            ingestion_client,
            commit_hi_rx,
            subscribers,
            metrics,
        ))
    }
}

impl Default for IngestionConfig {
    fn default() -> Self {
        Self {
            checkpoint_buffer_size: 50,
            ingest_concurrency: IngestConcurrencyConfig::Adaptive {
                initial: 1,
                min: 1,
                max: 500,
                dead_band: None,
            },
            retry_interval_ms: 200,
        }
    }
}
