// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
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
use tokio::sync::Notify;
use tracing::debug;
use tracing::error;
use tracing::info;
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

    /// Fetch checkpoints directly from a fullnode's gRPC `LedgerService` (unary
    /// `get_checkpoint`). Provide the gRPC base URL, e.g. `http://fullnode:9000`.
    #[arg(long, group = "source")]
    pub rpc_api_url: Option<Url>,

    /// Stream checkpoints live from a fullnode's gRPC `SubscriptionService`
    /// (`subscribe_checkpoints`), with unary `get_checkpoint` on the same URL used to backfill
    /// the gap between the resume point and the live tip. Provide the gRPC base URL, e.g.
    /// `http://fullnode:9000`.
    #[arg(long, group = "source")]
    pub streaming_url: Option<Url>,

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
            rpc_api_url: None,
            streaming_url: None,
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

/// A gRPC-backed ingestion client that fetches checkpoints directly from a fullnode's
/// `LedgerService` via the unary `get_checkpoint` RPC.
///
/// This is the soma equivalent of Sui's `rpc_client.rs` ingestion source. It does not stream;
/// each call opens (or reuses) a channel and issues one `get_checkpoint`. Streaming via
/// `SubscriptionService::subscribe_checkpoints` is a separate, larger piece of work that also
/// requires the broadcaster's fallback ladder.
#[derive(Clone)]
struct RpcIngestionClient {
    /// A lazily-connected channel — established on first use and transparently reconnected by
    /// tonic. Cloning a `Channel` is cheap (internally reference-counted), so each `fetch`
    /// clones rather than dialing afresh.
    channel: tonic::transport::Channel,
}

impl RpcIngestionClient {
    fn new(url: &Url) -> IngestionResult<Self> {
        let channel = tonic::transport::Endpoint::from_shared(url.to_string())
            .map_err(|e| IngestionError::FetchError(0, e.into()))?
            .connect_lazy();
        Ok(Self { channel })
    }
}

#[async_trait]
impl IngestionClientTrait for RpcIngestionClient {
    async fn fetch(&self, checkpoint: u64) -> FetchResult {
        use rpc::proto::soma::get_checkpoint_request::CheckpointId;
        use rpc::proto::soma::ledger_service_client::LedgerServiceClient;
        use rpc::proto::soma::GetCheckpointRequest;

        let mut client = LedgerServiceClient::new(self.channel.clone())
            .max_decoding_message_size(256 * 1024 * 1024);

        // Request every top-level field so the response carries a full checkpoint. The server's
        // default mask is just `sequence_number,digest`, which would not be ingestable.
        // GetCheckpointRequest is #[non_exhaustive] — build it field-by-field.
        let mut request = GetCheckpointRequest::default();
        request.read_mask = Some(prost_types::FieldMask {
            paths: [
                "sequence_number",
                "digest",
                "summary",
                "signature",
                "contents",
                "transactions",
                "objects",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        });
        request.checkpoint_id = Some(CheckpointId::SequenceNumber(checkpoint));

        let response = client.get_checkpoint(request).await.map_err(|status| {
            match status.code() {
                tonic::Code::NotFound => FetchError::NotFound,
                _ => FetchError::Transient {
                    reason: "grpc",
                    error: anyhow::anyhow!(status),
                },
            }
        })?;

        let proto_checkpoint = response.into_inner().checkpoint.ok_or_else(|| {
            FetchError::Permanent {
                reason: "missing_checkpoint",
                error: anyhow::anyhow!("get_checkpoint response had no checkpoint"),
            }
        })?;

        // Proto→Checkpoint conversion is multi-ms of CPU; offload to the blocking pool so it
        // doesn't stall the reactor. A conversion failure is permanent — the bytes are what
        // they are (mirrors the decode-as-permanent fix on the object-store path).
        let checkpoint = tokio::task::spawn_blocking(move || Checkpoint::try_from(&proto_checkpoint))
            .await
            .map_err(|e| FetchError::Transient {
                reason: "decode_task",
                error: anyhow::anyhow!("proto conversion task panicked: {e}"),
            })?
            .map_err(|e| FetchError::Permanent {
                reason: "proto_conversion",
                error: e.into(),
            })?;

        Ok(FetchData::Checkpoint(checkpoint))
    }
}

/// Maximum number of streamed checkpoints held in memory ahead of the broadcaster. When the
/// broadcaster falls further behind than this, the oldest buffered checkpoints are evicted and
/// served from the unary RPC backfill path instead.
const STREAM_BUFFER_CAP: usize = 128;

/// How long a `fetch` for a not-yet-streamed checkpoint waits on the stream before re-checking.
const STREAM_WAIT_TICK: Duration = Duration::from_secs(5);

/// Number of `STREAM_WAIT_TICK` cycles a `fetch` waits for the stream before falling back to a
/// unary RPC fetch. Guards against a silently stalled subscription wedging ingestion.
const STREAM_WAIT_TICKS_BEFORE_FALLBACK: u32 = 3;

/// Shared state between the background subscription task and `fetch` callers.
struct StreamBuffer {
    /// Streamed checkpoints keyed by sequence number, capped at `STREAM_BUFFER_CAP` (evict
    /// oldest on insert).
    checkpoints: Mutex<BTreeMap<u64, Arc<Checkpoint>>>,
    /// Highest cursor observed on the stream; 0 until the first checkpoint arrives.
    tip: AtomicU64,
    /// Notifies `fetch` waiters whenever a checkpoint lands in `checkpoints`.
    notify: Notify,
}

/// A gRPC streaming ingestion client. A background task subscribes to the fullnode's
/// `SubscriptionService::subscribe_checkpoints` and feeds a bounded buffer; `fetch` serves
/// live checkpoints from that buffer and falls back to unary `get_checkpoint` for any
/// checkpoint behind the streamed window (backfill).
///
/// This is the soma equivalent of Sui's `streaming_client.rs` + the broadcaster's
/// streaming/RPC fallback ladder, collapsed into a single `IngestionClientTrait` impl so it
/// drops into the existing broadcaster without reshaping it.
struct StreamingIngestionClient {
    /// Unary fallback, used for checkpoints below the streamed window.
    rpc: RpcIngestionClient,
    buffer: Arc<StreamBuffer>,
}

impl StreamingIngestionClient {
    fn new(url: &Url) -> IngestionResult<Self> {
        let rpc = RpcIngestionClient::new(url)?;
        let buffer = Arc::new(StreamBuffer {
            checkpoints: Mutex::new(BTreeMap::new()),
            tip: AtomicU64::new(0),
            notify: Notify::new(),
        });

        let endpoint = tonic::transport::Endpoint::from_shared(url.to_string())
            .map_err(|e| IngestionError::FetchError(0, e.into()))?;
        tokio::spawn(Self::subscription_loop(endpoint, buffer.clone()));

        Ok(Self { rpc, buffer })
    }

    /// Background task: keep a `subscribe_checkpoints` stream open, reconnecting forever on
    /// error, and feed every checkpoint it yields into the shared buffer.
    async fn subscription_loop(endpoint: tonic::transport::Endpoint, buffer: Arc<StreamBuffer>) {
        use rpc::proto::soma::subscription_service_client::SubscriptionServiceClient;
        use rpc::proto::soma::SubscribeCheckpointsRequest;

        loop {
            let channel = match endpoint.connect().await {
                Ok(c) => c,
                Err(e) => {
                    warn!("checkpoint subscription connect failed, retrying: {e}");
                    tokio::time::sleep(Duration::from_secs(1)).await;
                    continue;
                }
            };
            let mut client = SubscriptionServiceClient::new(channel)
                // Checkpoints can be large; lift the 4MB default decode cap.
                .max_decoding_message_size(256 * 1024 * 1024);

            let mut request = SubscribeCheckpointsRequest::default();
            request.read_mask = Some(prost_types::FieldMask {
                paths: [
                    "sequence_number",
                    "digest",
                    "summary",
                    "signature",
                    "contents",
                    "transactions",
                    "objects",
                ]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            });

            let mut stream = match client.subscribe_checkpoints(request).await {
                Ok(response) => response.into_inner(),
                Err(status) => {
                    warn!("subscribe_checkpoints failed, retrying: {status}");
                    tokio::time::sleep(Duration::from_secs(1)).await;
                    continue;
                }
            };

            info!("checkpoint subscription established");
            loop {
                match stream.message().await {
                    Ok(Some(response)) => {
                        let Some(cursor) = response.cursor else { continue };
                        let Some(proto_checkpoint) = response.checkpoint else { continue };
                        match Checkpoint::try_from(&proto_checkpoint) {
                            Ok(checkpoint) => {
                                let mut buf = buffer.checkpoints.lock().unwrap();
                                if buf.len() >= STREAM_BUFFER_CAP {
                                    buf.pop_first();
                                }
                                buf.insert(cursor, Arc::new(checkpoint));
                                drop(buf);
                                buffer.tip.fetch_max(cursor, Ordering::SeqCst);
                                buffer.notify.notify_waiters();
                            }
                            Err(e) => {
                                // A malformed streamed checkpoint is permanent — log and skip;
                                // the unary backfill path will re-fetch it if still needed.
                                error!(cursor, "failed to decode streamed checkpoint: {e}");
                            }
                        }
                    }
                    Ok(None) => {
                        warn!("checkpoint subscription stream ended, reconnecting");
                        break;
                    }
                    Err(status) => {
                        warn!("checkpoint subscription stream error, reconnecting: {status}");
                        break;
                    }
                }
            }
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
    }
}

#[async_trait]
impl IngestionClientTrait for StreamingIngestionClient {
    async fn fetch(&self, checkpoint: u64) -> FetchResult {
        let mut ticks: u32 = 0;
        loop {
            // Register for notification before checking the buffer, so a checkpoint landing
            // between the check and the await is not missed.
            let notified = self.buffer.notify.notified();

            if let Some(cp) = self.buffer.checkpoints.lock().unwrap().remove(&checkpoint) {
                return Ok(FetchData::Checkpoint((*cp).clone()));
            }

            let tip = self.buffer.tip.load(Ordering::SeqCst);

            // Below the streamed window (or stream not yet producing): backfill via unary RPC.
            if tip == 0 || checkpoint < tip {
                return self.rpc.fetch(checkpoint).await;
            }

            // At or ahead of the tip — the stream will deliver it. Wait, but don't wait
            // forever: if the subscription has silently stalled, fall back to unary RPC.
            tokio::select! {
                _ = notified => {}
                _ = tokio::time::sleep(STREAM_WAIT_TICK) => {
                    ticks += 1;
                    if ticks >= STREAM_WAIT_TICKS_BEFORE_FALLBACK {
                        return self.rpc.fetch(checkpoint).await;
                    }
                }
            }
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
        } else if let Some(url) = args.rpc_api_url.as_ref() {
            let client = Arc::new(RpcIngestionClient::new(url)?);
            IngestionClient::new_impl(client, metrics.clone())
        } else if let Some(url) = args.streaming_url.as_ref() {
            let client = Arc::new(StreamingIngestionClient::new(url)?);
            IngestionClient::new_impl(client, metrics.clone())
        } else {
            panic!(
                "One of remote_store_url, remote_store_s3, remote_store_gcs, \
                local_ingestion_path, rpc_api_url, or streaming_url must be provided"
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

                        // Decode failures (zstd, prost, proto→Checkpoint conversion) are
                        // permanent: the bytes in the bucket are what they are, and no number
                        // of retries will make them decodable. Treating them as transient
                        // burns CPU forever on a single bad blob and freezes the watermark
                        // (the v0.1.21 incident). Sui upstream has the same bug
                        // (sui-indexer-alt-framework/src/ingestion/ingestion_client.rs);
                        // PR the fix there once we have stability data.
                        decode::checkpoint(&bytes).map_err(|e| {
                            let reason = e.reason();
                            error!(
                                checkpoint,
                                reason,
                                "Permanent decode error: {e}"
                            );
                            self.metrics
                                .total_ingested_permanent_errors
                                .with_label_values(&[reason])
                                .inc();
                            BE::permanent(IngestionError::DeserializationError(
                                checkpoint,
                                e.into(),
                            ))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::tests::test_ingestion_metrics;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;
    use std::time::Duration;

    /// A test client that always returns the same raw bytes and counts how often it is called.
    struct StubBytesClient {
        bytes: Bytes,
        calls: AtomicUsize,
    }

    #[async_trait]
    impl IngestionClientTrait for StubBytesClient {
        async fn fetch(&self, _checkpoint: u64) -> FetchResult {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(FetchData::Raw(self.bytes.clone()))
        }
    }

    /// Decode failures (zstd, prost, proto→Checkpoint) must terminate the fetch — not retry
    /// forever. Regression for the v0.1.21 incident: a truncated blob in the bucket caused the
    /// indexer to burn CPU on infinite retries and freeze the watermark until operators
    /// bounced it.
    #[tokio::test(start_paused = true)]
    async fn decode_failures_terminate_fetch() {
        // Invalid zstd magic bytes — `decode::checkpoint` calls `decode_checkpoint` which
        // attempts zstd decompression first, so this fails at the Decompression stage.
        let client = Arc::new(StubBytesClient {
            bytes: Bytes::from_static(b"not-zstd-not-prost-just-garbage"),
            calls: AtomicUsize::new(0),
        });

        let metrics = test_ingestion_metrics();
        let ingestion_client =
            IngestionClient::new_impl(client.clone(), metrics.clone());

        // If decode were transient (the bug), the exponential backoff would retry up to
        // MAX_TRANSIENT_RETRY_INTERVAL forever. Cap the test with a deadline that is more
        // than enough for a permanent error to short-circuit, but small enough that the
        // bug-regressed code would visibly hang against it.
        let result = tokio::time::timeout(
            Duration::from_secs(5),
            ingestion_client.fetch(42),
        )
        .await
        .expect("fetch should terminate quickly on decode failure, not retry forever");

        let err = result.expect_err("expected decode failure");
        assert!(
            matches!(err, IngestionError::DeserializationError(42, _)),
            "expected DeserializationError(42, _), got: {err:?}",
        );

        // Permanent classification means the client is called exactly once — no retries.
        assert_eq!(
            client.calls.load(Ordering::SeqCst),
            1,
            "decode failures must not trigger retries",
        );
    }
}
