// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use backoff::Error as BE;
use backoff::ExponentialBackoff;
use backoff::backoff::Constant;
use bytes::Bytes;
use clap::ArgGroup;
use object_store::ClientOptions;
use object_store::ObjectStore;
use object_store::aws::AmazonS3Builder;
use object_store::gcp::GoogleCloudStorageBuilder;
use object_store::http::HttpBuilder;
use object_store::local::LocalFileSystem;
use soma_futures::future::with_slow_future_monitor;
use tracing::debug;
use tracing::error;
use tracing::warn;
use url::Url;

use crate::ingestion::Error as IngestionError;
use crate::ingestion::Result as IngestionResult;
use crate::ingestion::decode;
use crate::metrics::CheckpointLagMetricReporter;
use crate::metrics::IngestionMetrics;
use crate::types::full_checkpoint_content::Checkpoint;

/// Wait at most this long between retries for transient errors.
const MAX_TRANSIENT_RETRY_INTERVAL: Duration = Duration::from_secs(60);

/// Threshold for logging warnings about slow HTTP operations during checkpoint fetching.
const SLOW_OPERATION_WARNING_THRESHOLD: Duration = Duration::from_secs(60);

#[async_trait]
pub(crate) trait IngestionClientTrait: Send + Sync {
    async fn fetch(&self, checkpoint: u64) -> FetchResult;
}

#[derive(clap::Args, Clone, Debug)]
#[command(group(ArgGroup::new("source").required(true).multiple(false)))]
pub struct IngestionClientArgs {
    /// Remote Store to fetch checkpoints from over HTTP.
    #[arg(long, group = "source")]
    pub remote_store_url: Option<Url>,

    /// Fetch checkpoints from AWS S3. Provide the bucket name or endpoint-and-bucket.
    /// (env: AWS_ENDPOINT, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION)
    #[arg(long, group = "source")]
    pub remote_store_s3: Option<String>,

    /// Fetch checkpoints from Google Cloud Storage. Provide the bucket name.
    /// (env: GOOGLE_SERVICE_ACCOUNT_PATH)
    #[arg(long, group = "source")]
    pub remote_store_gcs: Option<String>,

    /// Path to the local ingestion directory.
    #[arg(long, group = "source")]
    pub local_ingestion_path: Option<PathBuf>,

    /// How long to wait for a checkpoint file to be downloaded (milliseconds). Set to 0 to disable
    /// the timeout.
    #[arg(long, default_value_t = Self::default().checkpoint_timeout_ms)]
    pub checkpoint_timeout_ms: u64,

    /// How long to wait while establishing a connection to the checkpoint store (milliseconds).
    /// Set to 0 to disable the timeout.
    #[arg(long, default_value_t = Self::default().checkpoint_connection_timeout_ms)]
    pub checkpoint_connection_timeout_ms: u64,
}

impl Default for IngestionClientArgs {
    fn default() -> Self {
        Self {
            remote_store_url: None,
            remote_store_s3: None,
            remote_store_gcs: None,
            local_ingestion_path: None,
            checkpoint_timeout_ms: 120_000,
            checkpoint_connection_timeout_ms: 120_000,
        }
    }
}

impl IngestionClientArgs {
    fn client_options(&self) -> ClientOptions {
        let mut options = ClientOptions::default();
        options = if self.checkpoint_timeout_ms == 0 {
            options.with_timeout_disabled()
        } else {
            let timeout = Duration::from_millis(self.checkpoint_timeout_ms);
            options.with_timeout(timeout)
        };
        options = if self.checkpoint_connection_timeout_ms == 0 {
            options.with_connect_timeout_disabled()
        } else {
            let timeout = Duration::from_millis(self.checkpoint_connection_timeout_ms);
            options.with_connect_timeout(timeout)
        };
        options
    }
}

#[derive(thiserror::Error, Debug)]
pub enum FetchError {
    #[error("Checkpoint not found")]
    NotFound,
    #[error("Failed to fetch checkpoint due to {reason}: {error}")]
    Transient {
        reason: &'static str,
        #[source]
        error: anyhow::Error,
    },
    #[error("Permanent error in {reason}: {error}")]
    Permanent {
        reason: &'static str,
        #[source]
        error: anyhow::Error,
    },
}

pub type FetchResult = Result<FetchData, FetchError>;

#[derive(Clone)]
#[allow(clippy::large_enum_variant)]
pub enum FetchData {
    Raw(Bytes),
    Checkpoint(Checkpoint),
}

/// An object store-backed ingestion client that fetches checkpoint files from a remote store.
struct StoreIngestionClient {
    store: Arc<dyn ObjectStore>,
}

impl StoreIngestionClient {
    fn new(store: Arc<dyn ObjectStore>) -> Self {
        Self { store }
    }
}

#[async_trait]
impl IngestionClientTrait for StoreIngestionClient {
    async fn fetch(&self, checkpoint: u64) -> FetchResult {
        let path = object_store::path::Path::from(format!("{}.binpb.zst", checkpoint));

        match self.store.get(&path).await {
            Ok(result) => {
                let bytes = result
                    .bytes()
                    .await
                    .map_err(|e| FetchError::Transient { reason: "read_bytes", error: e.into() })?;
                Ok(FetchData::Raw(bytes))
            }
            Err(object_store::Error::NotFound { .. }) => Err(FetchError::NotFound),
            Err(e) => Err(FetchError::Transient { reason: "object_store", error: e.into() }),
        }
    }
}

#[derive(Clone)]
pub struct IngestionClient {
    client: Arc<dyn IngestionClientTrait>,
    /// Wrap the metrics in an `Arc` to keep copies of the client cheap.
    metrics: Arc<IngestionMetrics>,
    checkpoint_lag_reporter: Arc<CheckpointLagMetricReporter>,
}

impl IngestionClient {
    /// Construct a new ingestion client. Its source is determined by `args`.
    pub fn new(args: IngestionClientArgs, metrics: Arc<IngestionMetrics>) -> IngestionResult<Self> {
        let retry = object_store::RetryConfig::default();
        let client = if let Some(url) = args.remote_store_url.as_ref() {
            let store = HttpBuilder::new()
                .with_url(url.to_string())
                .with_client_options(args.client_options().with_allow_http(true))
                .with_retry(retry)
                .build()
                .map(Arc::new)?;
            IngestionClient::with_store(store, metrics.clone())?
        } else if let Some(bucket) = args.remote_store_s3.as_ref() {
            let store = AmazonS3Builder::from_env()
                .with_client_options(args.client_options())
                .with_retry(retry)
                .with_bucket_name(bucket)
                .build()
                .map(Arc::new)?;
            IngestionClient::with_store(store, metrics.clone())?
        } else if let Some(bucket) = args.remote_store_gcs.as_ref() {
            let store = GoogleCloudStorageBuilder::from_env()
                .with_client_options(args.client_options())
                .with_retry(retry)
                .with_bucket_name(bucket)
                .build()
                .map(Arc::new)?;
            IngestionClient::with_store(store, metrics.clone())?
        } else if let Some(path) = args.local_ingestion_path.as_ref() {
            let store = LocalFileSystem::new_with_prefix(path).map(Arc::new)?;
            IngestionClient::with_store(store, metrics.clone())?
        } else {
            panic!(
                "One of remote_store_url, remote_store_s3, remote_store_gcs, or \
                local_ingestion_path must be provided"
            );
        };

        Ok(client)
    }

    /// An ingestion client that fetches checkpoints from a remote object store.
    pub fn with_store(
        store: Arc<dyn ObjectStore>,
        metrics: Arc<IngestionMetrics>,
    ) -> IngestionResult<Self> {
        let client = Arc::new(StoreIngestionClient::new(store));
        Ok(Self::new_impl(client, metrics))
    }

    pub(crate) fn new_impl(
        client: Arc<dyn IngestionClientTrait>,
        metrics: Arc<IngestionMetrics>,
    ) -> Self {
        let checkpoint_lag_reporter = CheckpointLagMetricReporter::new(
            metrics.ingested_checkpoint_timestamp_lag.clone(),
            metrics.latest_ingested_checkpoint_timestamp_lag_ms.clone(),
            metrics.latest_ingested_checkpoint.clone(),
        );
        IngestionClient { client, metrics, checkpoint_lag_reporter }
    }

    /// Fetch checkpoint data by sequence number.
    ///
    /// This function behaves like `IngestionClient::fetch`, but will repeatedly retry the fetch if
    /// the checkpoint is not found, on a constant back-off.
    pub async fn wait_for(
        &self,
        checkpoint: u64,
        retry_interval: Duration,
    ) -> IngestionResult<Arc<Checkpoint>> {
        let backoff = Constant::new(retry_interval);
        let fetch = || async {
            use backoff::Error as BE;
            self.fetch(checkpoint).await.map_err(|e| match e {
                IngestionError::NotFound(checkpoint) => {
                    debug!(checkpoint, "Checkpoint not found, retrying...");
                    self.metrics.total_ingested_not_found_retries.inc();
                    BE::transient(e)
                }
                e => BE::permanent(e),
            })
        };

        backoff::future::retry(backoff, fetch).await
    }

    /// Fetch checkpoint data by sequence number.
    ///
    /// Repeatedly retries transient errors with an exponential backoff (up to
    /// `MAX_TRANSIENT_RETRY_INTERVAL`). The function will immediately return if the checkpoint
    /// is not found.
    pub async fn fetch(&self, checkpoint: u64) -> IngestionResult<Arc<Checkpoint>> {
        let client = self.client.clone();
        let request = move || {
            let client = client.clone();
            async move {
                let fetch_data = with_slow_future_monitor(
                    client.fetch(checkpoint),
                    SLOW_OPERATION_WARNING_THRESHOLD,
                    || {
                        warn!(
                            checkpoint,
                            threshold_ms = SLOW_OPERATION_WARNING_THRESHOLD.as_millis(),
                            "Slow checkpoint fetch operation detected"
                        );
                    },
                )
                .await
                .map_err(|err| match err {
                    FetchError::NotFound => BE::permanent(IngestionError::NotFound(checkpoint)),
                    FetchError::Transient { reason, error } => self.metrics.inc_retry(
                        checkpoint,
                        reason,
                        IngestionError::FetchError(checkpoint, error),
                    ),
                    FetchError::Permanent { reason, error } => {
                        error!(checkpoint, reason, "Permanent fetch error: {error}");
                        self.metrics
                            .total_ingested_permanent_errors
                            .with_label_values(&[reason])
                            .inc();
                        BE::permanent(IngestionError::FetchError(checkpoint, error))
                    }
                })?;

                Ok::<Checkpoint, backoff::Error<IngestionError>>(match fetch_data {
                    FetchData::Raw(bytes) => {
                        self.metrics.total_ingested_bytes.inc_by(bytes.len() as u64);

                        decode::checkpoint(&bytes).map_err(|e| {
                            self.metrics.inc_retry(
                                checkpoint,
                                e.reason(),
                                IngestionError::DeserializationError(checkpoint, e.into()),
                            )
                        })?
                    }
                    FetchData::Checkpoint(data) => data,
                })
            }
        };

        // Keep backing off until we are waiting for the max interval, but don't give up.
        let backoff = ExponentialBackoff {
            max_interval: MAX_TRANSIENT_RETRY_INTERVAL,
            max_elapsed_time: None,
            ..Default::default()
        };

        let guard = self.metrics.ingested_checkpoint_latency.start_timer();
        let data = backoff::future::retry(backoff, request).await?;
        let elapsed = guard.stop_and_record();

        debug!(checkpoint, elapsed_ms = elapsed * 1000.0, "Fetched checkpoint");

        self.checkpoint_lag_reporter.report_lag(checkpoint, data.summary.timestamp_ms);

        self.metrics.total_ingested_checkpoints.inc();

        self.metrics.total_ingested_transactions.inc_by(data.transactions.len() as u64);

        self.metrics.total_ingested_objects.inc_by(data.object_set.len() as u64);

        Ok(Arc::new(data))
    }
}
