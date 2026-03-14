// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use anyhow::bail;
use anyhow::ensure;
use indexer_store_traits::Connection as _;
use indexer_store_traits::pipeline_task;
use ingestion::ClientArgs;
use ingestion::IngestionConfig;
use ingestion::IngestionService;
use ingestion::ingestion_client::IngestionClient;
use metrics::IndexerMetrics;
use prometheus::Registry;
use tracing::info;

use crate::metrics::IngestionMetrics;
use crate::pipeline::Processor;
use crate::pipeline::concurrent::ConcurrentConfig;
use crate::pipeline::concurrent::{self};
use crate::pipeline::sequential::Handler;
use crate::pipeline::sequential::SequentialConfig;
use crate::pipeline::sequential::{self};

pub use anyhow::Result;
/// External users access the store trait through framework::store
pub use indexer_store_traits as store;
pub use soma_field_count::FieldCount;
pub use soma_futures::service;
pub use types;

pub mod config;
pub mod ingestion;
pub mod metrics;
pub mod pipeline;
#[cfg(feature = "postgres")]
pub mod postgres;

#[cfg(test)]
pub mod mocks;

/// Command-line arguments for the indexer
#[derive(clap::Args, Default, Debug, Clone)]
pub struct IndexerArgs {
    /// Override the next checkpoint for all pipelines without a committer watermark to start
    /// processing from, which is 0 by default. Pipelines with existing watermarks will ignore this
    /// setting and always resume from their committer watermark + 1.
    ///
    /// Setting this value indirectly affects ingestion, as the checkpoint to start ingesting from
    /// is the minimum across all pipelines' next checkpoints.
    #[arg(long)]
    pub first_checkpoint: Option<u64>,

    /// Override for the checkpoint to end ingestion at (inclusive) -- useful for backfills. By
    /// default, ingestion will not stop, and will continue to poll for new checkpoints.
    #[arg(long)]
    pub last_checkpoint: Option<u64>,

    /// Only run the following pipelines. If not provided, all pipelines found in the
    /// configuration file will be run.
    #[arg(long, action = clap::ArgAction::Append)]
    pub pipeline: Vec<String>,

    /// Additional configurations for running a tasked indexer.
    #[clap(flatten)]
    pub task: TaskArgs,
}

/// Command-line arguments for configuring a tasked indexer.
#[derive(clap::Parser, Default, Debug, Clone)]
pub struct TaskArgs {
    /// An optional task name for this indexer. When set, pipelines will record watermarks using the
    /// delimiter defined on the store. This allows the same pipelines to run under multiple
    /// indexers (e.g. for backfills or temporary workflows) while maintaining separate watermark
    /// entries in the database.
    ///
    /// By default there is no task name, and watermarks are keyed only by `pipeline`.
    ///
    /// Sequential pipelines cannot be attached to a tasked indexer.
    ///
    /// The framework ensures that tasked pipelines never commit checkpoints below the main
    /// pipeline's pruner watermark. Requires `--reader-interval-ms`.
    #[arg(long, requires = "reader_interval_ms")]
    task: Option<String>,

    /// The interval in milliseconds at which each of the pipelines on a tasked indexer should
    /// refetch its main pipeline's reader watermark.
    ///
    /// This is required when `--task` is set and should ideally be set to a value that is
    /// an order of magnitude smaller than the main pipeline's pruning interval, to ensure this
    /// task pipeline can pick up the new reader watermark before the main pipeline prunes up to
    /// it.
    ///
    /// If the main pipeline does not have pruning enabled, this value can be set to some high
    /// value, as the tasked pipeline will never see an updated reader watermark.
    #[arg(long, requires = "task")]
    reader_interval_ms: Option<u64>,
}

pub struct Indexer<S: store::Store> {
    /// The storage backend that the indexer uses to write and query indexed data.
    store: S,

    /// Prometheus Metrics.
    metrics: Arc<IndexerMetrics>,

    /// Service for downloading and disseminating checkpoint data.
    ingestion_service: IngestionService,

    /// The next checkpoint for a pipeline without a committer watermark to start processing from,
    /// which will be 0 by default. Pipelines with existing watermarks will ignore this setting and
    /// always resume from their committer watermark + 1.
    ///
    /// Setting this value indirectly affects ingestion, as the checkpoint to start ingesting from
    /// is the minimum across all pipelines' next checkpoints.
    default_next_checkpoint: u64,

    /// Optional override of the checkpoint upperbound. When set, the indexer will stop ingestion at
    /// this checkpoint.
    last_checkpoint: Option<u64>,

    /// An optional task name for this indexer. When set, pipelines will record watermarks using the
    /// delimiter defined on the store.
    task: Option<Task>,

    /// Optional filter for pipelines to run. If `None`, all pipelines added to the indexer will
    /// run.
    enabled_pipelines: Option<BTreeSet<String>>,

    /// Pipelines that have already been registered with the indexer. Used to make sure a pipeline
    /// with the same name isn't added twice.
    added_pipelines: BTreeSet<&'static str>,

    /// The checkpoint for the indexer to start ingesting from. This is derived from the committer
    /// watermarks of pipelines added to the indexer.
    first_ingestion_checkpoint: u64,

    /// The minimum next_checkpoint across all sequential pipelines. This is used to initialize
    /// the regulator to prevent ingestion from running too far ahead of sequential pipelines.
    next_sequential_checkpoint: Option<u64>,

    /// The service handles for every pipeline, used to manage lifetimes and graceful shutdown.
    pipelines: Vec<service::Service>,
}

/// Configuration for a tasked indexer.
#[derive(Clone)]
pub(crate) struct Task {
    /// Name of the tasked indexer, to be used with the delimiter defined on the indexer's store to
    /// record pipeline watermarks.
    task: String,
    /// The interval at which each of the pipelines on a tasked indexer should refetch its main
    /// pipeline's reader watermark.
    reader_interval: Duration,
}

impl TaskArgs {
    pub fn tasked(task: String, reader_interval_ms: u64) -> Self {
        Self { task: Some(task), reader_interval_ms: Some(reader_interval_ms) }
    }

    fn into_task(self) -> Option<Task> {
        Some(Task {
            task: self.task?,
            reader_interval: Duration::from_millis(self.reader_interval_ms?),
        })
    }
}

impl<S: store::Store> Indexer<S> {
    /// Create a new instance of the indexer framework from a store that implements the `Store`
    /// trait, along with `indexer_args`, `client_args`, and `ingestion_config`. Together, these
    /// arguments configure the following:
    ///
    /// - What is indexed (which checkpoints, which pipelines, whether to track and update
    ///   watermarks) and where to serve metrics from,
    /// - Where to download checkpoints from,
    /// - Concurrency and buffering parameters for downloading checkpoints.
    ///
    /// After initialization, at least one pipeline must be added using [Self::concurrent_pipeline]
    /// or [Self::sequential_pipeline], before the indexer is started using [Self::run].
    pub async fn new(
        store: S,
        indexer_args: IndexerArgs,
        client_args: ClientArgs,
        ingestion_config: IngestionConfig,
        metrics_prefix: Option<&str>,
        registry: &Registry,
    ) -> Result<Self> {
        let IndexerArgs { first_checkpoint, last_checkpoint, pipeline, task } = indexer_args;

        let metrics = IndexerMetrics::new(metrics_prefix, registry);

        let ingestion_service =
            IngestionService::new(client_args, ingestion_config, metrics_prefix, registry)?;

        Ok(Self {
            store,
            metrics,
            ingestion_service,
            default_next_checkpoint: first_checkpoint.unwrap_or_default(),
            last_checkpoint,
            task: task.into_task(),
            enabled_pipelines: if pipeline.is_empty() {
                None
            } else {
                Some(pipeline.into_iter().collect())
            },
            added_pipelines: BTreeSet::new(),
            first_ingestion_checkpoint: u64::MAX,
            next_sequential_checkpoint: None,
            pipelines: vec![],
        })
    }

    /// The store used by the indexer.
    pub fn store(&self) -> &S {
        &self.store
    }

    /// The ingestion client used by the indexer to fetch checkpoints.
    pub fn ingestion_client(&self) -> &IngestionClient {
        self.ingestion_service.ingestion_client()
    }

    /// The indexer's metrics.
    pub fn indexer_metrics(&self) -> &Arc<IndexerMetrics> {
        &self.metrics
    }

    /// The ingestion service's metrics.
    pub fn ingestion_metrics(&self) -> &Arc<IngestionMetrics> {
        self.ingestion_service.metrics()
    }

    /// The pipelines that this indexer will run.
    pub fn pipelines(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.added_pipelines
            .iter()
            .copied()
            .filter(|p| self.enabled_pipelines.as_ref().is_none_or(|e| e.contains(*p)))
    }

    /// The minimum next checkpoint across all sequential pipelines. This value is used to
    /// initialize the ingestion regulator's high watermark to prevent ingestion from running
    /// too far ahead of sequential pipelines.
    pub fn next_sequential_checkpoint(&self) -> Option<u64> {
        self.next_sequential_checkpoint
    }

    /// Adds a new pipeline to this indexer and starts it up. Although their tasks have started,
    /// they will be idle until the ingestion service starts, and serves it checkpoint data.
    ///
    /// Concurrent pipelines commit checkpoint data out-of-order to maximise throughput, and they
    /// keep the watermark table up-to-date with the highest point they can guarantee all data
    /// exists for, for their pipeline.
    pub async fn concurrent_pipeline<H>(
        &mut self,
        handler: H,
        config: ConcurrentConfig,
    ) -> Result<()>
    where
        H: concurrent::Handler<Store = S> + Send + Sync + 'static,
    {
        let Some(next_checkpoint) = self.add_pipeline::<H>().await? else {
            return Ok(());
        };

        self.pipelines.push(concurrent::pipeline::<H>(
            handler,
            next_checkpoint,
            config,
            self.store.clone(),
            self.task.clone(),
            self.ingestion_service.subscribe().0,
            self.metrics.clone(),
        ));

        Ok(())
    }

    /// Start ingesting checkpoints from `first_ingestion_checkpoint`. Individual pipelines
    /// will start processing and committing once the ingestion service has caught up to their
    /// respective watermarks.
    ///
    /// Ingestion will stop after consuming the configured `last_checkpoint` if one is provided.
    pub async fn run(self) -> Result<service::Service> {
        if let Some(enabled_pipelines) = self.enabled_pipelines {
            ensure!(
                enabled_pipelines.is_empty(),
                "Tried to enable pipelines that this indexer does not know about: \
                {enabled_pipelines:#?}",
            );
        }

        let last_checkpoint = self.last_checkpoint.unwrap_or(u64::MAX);

        info!(self.first_ingestion_checkpoint, last_checkpoint = ?self.last_checkpoint, "Ingestion range");

        let mut service = self
            .ingestion_service
            .run(self.first_ingestion_checkpoint..=last_checkpoint, self.next_sequential_checkpoint)
            .await
            .context("Failed to start ingestion service")?;

        for pipeline in self.pipelines {
            service = service.merge(pipeline);
        }

        Ok(service)
    }

    /// Determine the checkpoint for the pipeline to resume processing from. This is either the
    /// checkpoint after its watermark, or if that doesn't exist, then the provided
    /// [Self::default_next_checkpoint], and if that is not set, then 0 (genesis).
    ///
    /// Update the starting ingestion checkpoint as the minimum across all the next checkpoints
    /// calculated above.
    ///
    /// Returns `Ok(None)` if the pipeline is disabled.
    async fn add_pipeline<P: Processor + 'static>(&mut self) -> Result<Option<u64>> {
        ensure!(self.added_pipelines.insert(P::NAME), "Pipeline {:?} already added", P::NAME,);

        if let Some(enabled_pipelines) = &mut self.enabled_pipelines {
            if !enabled_pipelines.remove(P::NAME) {
                info!(pipeline = P::NAME, "Skipping");
                return Ok(None);
            }
        }

        let mut conn =
            self.store.connect().await.context("Failed to establish connection to store")?;

        let pt = pipeline_task(P::NAME, self.task.as_ref().map(|t| t.task.as_str()), S::DELIMITER);

        let checkpoint_hi_inclusive = conn
            .init_watermark(&pt, self.default_next_checkpoint)
            .await
            .with_context(|| format!("Failed to init watermark for {pt}"))?;

        let next_checkpoint =
            checkpoint_hi_inclusive.map_or(self.default_next_checkpoint, |c| c + 1);

        self.first_ingestion_checkpoint = next_checkpoint.min(self.first_ingestion_checkpoint);

        Ok(Some(next_checkpoint))
    }
}

impl<T: store::TransactionalStore> Indexer<T> {
    /// Adds a new pipeline to this indexer and starts it up. Although their tasks have started,
    /// they will be idle until the ingestion service starts, and serves it checkpoint data.
    ///
    /// Sequential pipelines commit checkpoint data in-order which sacrifices throughput, but may be
    /// required to handle pipelines that modify data in-place (where each update is not an insert,
    /// but could be a modification of an existing row, where ordering between updates is
    /// important).
    ///
    /// The pipeline can optionally be configured to lag behind the ingestion service by a fixed
    /// number of checkpoints (configured by `checkpoint_lag`).
    pub async fn sequential_pipeline<H>(
        &mut self,
        handler: H,
        config: SequentialConfig,
    ) -> Result<()>
    where
        H: Handler<Store = T> + Send + Sync + 'static,
    {
        let Some(next_checkpoint) = self.add_pipeline::<H>().await? else {
            return Ok(());
        };

        if self.task.is_some() {
            bail!(
                "Sequential pipelines do not support pipeline tasks. \
                These pipelines guarantee that each checkpoint is committed exactly once and in order. \
                Running the same pipeline under a different task would violate these guarantees."
            );
        }

        // Track the minimum next_checkpoint across all sequential pipelines
        self.next_sequential_checkpoint = Some(
            self.next_sequential_checkpoint.map_or(next_checkpoint, |n| n.min(next_checkpoint)),
        );

        let (checkpoint_rx, commit_hi_tx) = self.ingestion_service.subscribe();

        self.pipelines.push(sequential::pipeline::<H>(
            handler,
            next_checkpoint,
            config,
            self.store.clone(),
            checkpoint_rx,
            commit_hi_tx,
            self.metrics.clone(),
        ));

        Ok(())
    }
}
