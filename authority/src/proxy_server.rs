//! Validator Proxy Server for serving submission data and model weights.
//!
//! This HTTP server runs on validators to serve data/model downloads to clients:
//! - `GET /data/{target_id}` - Serve winning submission data for a filled target
//! - `GET /model/{model_id}` - Serve model weights for an active model
//!
//! Key features:
//! - **Singleflight pattern**: Concurrent requests for the same resource share a single download
//! - **Availability reports**: On fetch failure, submits `ReportSubmission` or `ReportModel` tx
//! - **Size-based timeouts**: Timeout scales with expected data size
//!
//! Fullnodes use `ProxyClient` to connect to validators via this proxy.

use std::{sync::Arc, time::Duration};

use axum::{
    Router,
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    routing::get,
};
use bytes::Bytes;
use dashmap::DashMap;
use tokio::sync::{broadcast, mpsc};
use tracing::{info, warn};

use blobs::BlobPath;
use blobs::downloader::{BlobDownloader, HttpBlobDownloader};
use object_store::ObjectStore;
use types::{
    SYSTEM_STATE_OBJECT_ID,
    base::SomaAddress,
    consensus::ConsensusTransaction,
    crypto::{Signature, SomaKeyPair},
    intent::{Intent, IntentMessage},
    metadata::{Manifest, ManifestAPI as _, MetadataAPI as _},
    model::ModelId,
    object::ObjectID,
    parameters::HttpParameters,
    target::TargetId,
    transaction::{Transaction, TransactionData, TransactionKind},
};

use crate::authority::AuthorityState;
use crate::authority_per_epoch_store::AuthorityPerEpochStore;
use crate::consensus_adapter::ConsensusAdapter;

// ===========================================================================
// Availability Reports
// ===========================================================================

/// Report sent when a resource fetch fails.
/// The proxy server sends these to a report handler which submits the
/// appropriate `ReportSubmission` or `ReportModel` transaction.
#[derive(Debug, Clone)]
pub enum AvailabilityReport {
    /// Submission data was unavailable for the given target
    Submission(TargetId),
    /// Model weights were unavailable for the given model
    Model(ModelId),
}

/// Channel for sending availability reports to the report handler.
pub type ReportSender = mpsc::Sender<AvailabilityReport>;

// ===========================================================================
// Proxy Errors
// ===========================================================================

/// Errors that can occur in the proxy server.
#[derive(Debug, Clone)]
pub enum ProxyError {
    /// Target not found or not filled
    TargetNotFound(TargetId),
    /// Model not found or not active
    ModelNotFound(ModelId),
    /// Failed to download from source URL
    DownloadFailed(String),
    /// Request was cancelled (e.g., singleflight subscriber dropped)
    Cancelled,
    /// Internal error
    Internal(String),
}

impl std::fmt::Display for ProxyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProxyError::TargetNotFound(id) => write!(f, "Target not found or not filled: {}", id),
            ProxyError::ModelNotFound(id) => write!(f, "Model not found or not active: {}", id),
            ProxyError::DownloadFailed(msg) => write!(f, "Download failed: {}", msg),
            ProxyError::Cancelled => write!(f, "Request cancelled"),
            ProxyError::Internal(msg) => write!(f, "Internal error: {}", msg),
        }
    }
}

impl IntoResponse for ProxyError {
    fn into_response(self) -> axum::response::Response {
        let (status, message) = match &self {
            ProxyError::TargetNotFound(_) => (StatusCode::NOT_FOUND, self.to_string()),
            ProxyError::ModelNotFound(_) => (StatusCode::NOT_FOUND, self.to_string()),
            ProxyError::DownloadFailed(_) => (StatusCode::BAD_GATEWAY, self.to_string()),
            ProxyError::Cancelled => (StatusCode::INTERNAL_SERVER_ERROR, self.to_string()),
            ProxyError::Internal(_) => (StatusCode::INTERNAL_SERVER_ERROR, self.to_string()),
        };
        (status, message).into_response()
    }
}

// ===========================================================================
// Proxy Configuration
// ===========================================================================

/// Configuration for the proxy server.
#[derive(Debug, Clone)]
pub struct ProxyConfig {
    /// Base timeout for downloads (before size-based adjustment)
    pub base_download_timeout: Duration,
    /// Nanoseconds per byte for timeout calculation
    /// Default: 50ns/byte (~20MB/s minimum expected throughput)
    pub nanoseconds_per_byte: u64,
}

impl Default for ProxyConfig {
    fn default() -> Self {
        Self { base_download_timeout: Duration::from_secs(30), nanoseconds_per_byte: 50 }
    }
}

impl ProxyConfig {
    /// Calculate download timeout based on data size.
    pub fn download_timeout_for_size(&self, data_size: u64) -> Duration {
        let size_based_nanos = data_size.saturating_mul(self.nanoseconds_per_byte);
        self.base_download_timeout + Duration::from_nanos(size_based_nanos)
    }
}

// ===========================================================================
// Singleflight State
// ===========================================================================

/// State for in-flight requests (singleflight pattern).
/// Keyed by resource identifier (e.g., "data:{target_id}" or "model:{model_id}").
type InflightMap = DashMap<String, broadcast::Sender<Result<Bytes, ProxyError>>>;

// ===========================================================================
// Proxy Server
// ===========================================================================

/// Validator proxy server for serving submission data and model weights.
///
/// # Architecture
///
/// ```text
/// Client Request
///       │
///       ▼
/// ProxyServer::serve_data / serve_model
///       │
///       ├─ Check singleflight map
///       │   ├─ Request in flight? Subscribe and wait
///       │   └─ New request? Start download, insert into map
///       │
///       ▼
/// Download from source URL (with size-based timeout)
///       │
///       ├─ Success: Return bytes, broadcast to waiters
///       └─ Failure: Send AvailabilityReport, return error
/// ```
pub struct ProxyServer {
    /// Authority state for looking up targets and models
    state: Arc<AuthorityState>,
    /// Blob downloader for downloading from source URLs into the local store
    downloader: HttpBlobDownloader,
    /// Local object store used as download destination / cache
    store: Arc<dyn ObjectStore>,
    /// Channel to send reports when fetch fails
    report_tx: ReportSender,
    /// In-flight request deduplication (singleflight pattern)
    inflight: InflightMap,
    /// Configuration
    config: ProxyConfig,
}

impl ProxyServer {
    /// Create a new proxy server.
    pub fn new(
        state: Arc<AuthorityState>,
        report_tx: ReportSender,
        store: Arc<dyn ObjectStore>,
        config: ProxyConfig,
    ) -> Result<Self, ProxyError> {
        let http_params = HttpParameters::default();
        let semaphore = Arc::new(tokio::sync::Semaphore::new(16));
        let downloader = HttpBlobDownloader::new(
            &http_params,
            store.clone(),
            semaphore,
            blobs::MIN_PART_SIZE,
            50, // ns_per_byte
        )
        .map_err(|e| ProxyError::Internal(e.to_string()))?;

        Ok(Self { state, downloader, store, report_tx, inflight: DashMap::new(), config })
    }

    /// Build an axum router for the proxy server.
    pub fn router(self: Arc<Self>) -> Router {
        Router::new()
            .route("/data/{target_id}", get(Self::serve_data))
            .route("/model/{model_id}", get(Self::serve_model))
            .with_state(self)
    }

    /// Serve submission data for a filled target.
    async fn serve_data(
        State(server): State<Arc<ProxyServer>>,
        Path(target_id_str): Path<String>,
    ) -> Result<impl IntoResponse, ProxyError> {
        // Parse target ID
        let target_id = parse_object_id(&target_id_str)
            .ok_or_else(|| ProxyError::TargetNotFound(ObjectID::random()))?;

        // Look up target from authority state
        let target = server
            .state
            .get_object(&target_id)
            .await
            .ok_or(ProxyError::TargetNotFound(target_id))?;

        // Deserialize target
        let target: types::target::Target = bcs::from_bytes(target.as_inner().data.contents())
            .map_err(|e| ProxyError::Internal(format!("Failed to deserialize target: {}", e)))?;

        // Get the winning data manifest (only available if target is filled)
        let manifest =
            target.winning_data_manifest.as_ref().ok_or(ProxyError::TargetNotFound(target_id))?;

        // Fetch with singleflight
        let checksum = manifest.manifest.metadata().checksum();
        let epoch = server.state.load_epoch_store_one_call_per_task().epoch();
        let blob_path = BlobPath::Data(epoch, checksum);
        let key = format!("data:{}", target_id);
        match server.fetch_with_singleflight(&key, &manifest.manifest, blob_path).await {
            Ok(data) => {
                info!("Served {} bytes for target {}", data.len(), target_id);
                Ok((StatusCode::OK, data))
            }
            Err(e) => {
                // Report unavailable
                let _ = server.report_tx.send(AvailabilityReport::Submission(target_id)).await;
                Err(e)
            }
        }
    }

    /// Serve model weights for an active model.
    async fn serve_model(
        State(server): State<Arc<ProxyServer>>,
        Path(model_id_str): Path<String>,
    ) -> Result<impl IntoResponse, ProxyError> {
        // Parse model ID
        let model_id = parse_object_id(&model_id_str)
            .ok_or_else(|| ProxyError::ModelNotFound(ObjectID::random()))?;

        // Look up model from SystemState via model registry
        let manifest = server.get_model_manifest(&model_id).await?;

        // Fetch with singleflight
        let checksum = manifest.metadata().checksum();
        let epoch = server.state.load_epoch_store_one_call_per_task().epoch();
        let blob_path = BlobPath::Weights(epoch, checksum);
        let key = format!("model:{}", model_id);
        match server.fetch_with_singleflight(&key, &manifest, blob_path).await {
            Ok(data) => {
                info!("Served {} bytes for model {}", data.len(), model_id);
                Ok((StatusCode::OK, data))
            }
            Err(e) => {
                // Report unavailable
                let _ = server.report_tx.send(AvailabilityReport::Model(model_id)).await;
                Err(e)
            }
        }
    }

    /// Get model manifest from the model registry in SystemState.
    async fn get_model_manifest(&self, model_id: &ModelId) -> Result<Manifest, ProxyError> {
        // Load SystemState
        let state_object = self
            .state
            .get_object(&SYSTEM_STATE_OBJECT_ID)
            .await
            .ok_or_else(|| ProxyError::Internal("SystemState not found".into()))?;

        let system_state: types::system_state::SystemState =
            bcs::from_bytes(state_object.as_inner().data.contents()).map_err(|e| {
                ProxyError::Internal(format!("Failed to deserialize SystemState: {}", e))
            })?;

        // Look up model in registry
        let model = system_state
            .model_registry
            .active_models
            .get(model_id)
            .ok_or(ProxyError::ModelNotFound(*model_id))?;

        // Get weights manifest (only available if model is revealed/active)
        let weights_manifest =
            model.weights_manifest.as_ref().ok_or(ProxyError::ModelNotFound(*model_id))?;

        Ok(weights_manifest.manifest.clone())
    }

    /// Fetch with singleflight pattern to deduplicate concurrent requests.
    ///
    /// If a request for the same key is already in flight, we subscribe to it
    /// instead of starting a new download. This prevents thundering herd when
    /// many clients request the same uncached resource.
    async fn fetch_with_singleflight(
        &self,
        key: &str,
        manifest: &Manifest,
        blob_path: BlobPath,
    ) -> Result<Bytes, ProxyError> {
        // Check if request is already in flight
        if let Some(sender) = self.inflight.get(key) {
            // Subscribe to existing request
            let mut rx = sender.subscribe();
            drop(sender); // Release lock before awaiting
            return rx.recv().await.map_err(|_| ProxyError::Cancelled)?;
        }

        // Start new request
        let (tx, _) = broadcast::channel(1);
        self.inflight.insert(key.to_string(), tx.clone());

        // Fetch into local store, then read back
        let result = self.fetch_and_read(manifest, blob_path).await;

        // Broadcast result to all waiters (ignore send errors - no subscribers is fine)
        let _ = tx.send(result.clone());

        // Clean up inflight map
        self.inflight.remove(key);

        result
    }

    /// Download blob into the local object store via the blob downloader, then read it back.
    async fn fetch_and_read(
        &self,
        manifest: &Manifest,
        blob_path: BlobPath,
    ) -> Result<Bytes, ProxyError> {
        self.downloader
            .download(manifest, blob_path.clone())
            .await
            .map_err(|e| ProxyError::DownloadFailed(e.to_string()))?;

        let result = self
            .store
            .get(&blob_path.path())
            .await
            .map_err(|e| ProxyError::Internal(format!("Failed to read from store: {}", e)))?;

        result
            .bytes()
            .await
            .map_err(|e| ProxyError::Internal(format!("Failed to read bytes from store: {}", e)))
    }
}

// ===========================================================================
// Report Handler
// ===========================================================================

/// Spawns a task to handle availability reports by submitting transactions.
///
/// When the proxy server fails to fetch a resource, it sends a report through
/// the channel. This handler receives reports and submits the appropriate
/// `ReportSubmission` or `ReportModel` transaction.
///
/// # Arguments
/// * `report_rx` - Receiver for availability reports
/// * `validator_address` - This validator's account address (transaction sender)
/// * `account_keypair` - This validator's account keypair for signing transactions
/// * `state` - Authority state for validator name
/// * `consensus_adapter` - For submitting transactions to consensus
/// * `epoch_store` - Current epoch store for consensus context
pub async fn spawn_report_handler(
    mut report_rx: mpsc::Receiver<AvailabilityReport>,
    validator_address: SomaAddress,
    account_keypair: Arc<SomaKeyPair>,
    state: Arc<AuthorityState>,
    consensus_adapter: Arc<ConsensusAdapter>,
    epoch_store: Arc<AuthorityPerEpochStore>,
) {
    use fastcrypto::traits::KeyPair as _;

    while let Some(report) = report_rx.recv().await {
        let tx_kind = match report {
            AvailabilityReport::Submission(target_id) => {
                info!("Submitting ReportSubmission for target {}", target_id);
                // challenger=None for availability issues (no fraud, just data unavailable)
                TransactionKind::ReportSubmission { target_id, challenger: None }
            }
            AvailabilityReport::Model(model_id) => {
                info!("Submitting ReportModel for model {}", model_id);
                TransactionKind::ReportModel { model_id }
            }
        };

        // Create and sign transaction with validator's account keypair
        let tx_data = TransactionData::new(
            tx_kind.clone(),
            validator_address,
            vec![], // No gas payment for validator transactions
        );

        let intent_msg = IntentMessage::new(Intent::soma_transaction(), &tx_data);
        let sig = Signature::new_secure(&intent_msg, account_keypair.as_ref());
        let tx = Transaction::from_data(tx_data, vec![sig]);

        // Wrap in ConsensusTransaction and submit
        let consensus_tx = ConsensusTransaction::new_user_transaction_message(&state.name, tx);

        match consensus_adapter.submit(
            consensus_tx,
            None, // No reconfig lock
            &epoch_store,
            None, // No position tracking
            None, // No client address
        ) {
            Ok(_) => {
                info!("Validator transaction submitted successfully: {:?}", tx_kind);
            }
            Err(e) => {
                warn!("Failed to submit validator transaction {:?}: {:?}", tx_kind, e);
            }
        }
    }
    warn!("Report handler shutting down - report channel closed");
}

// ===========================================================================
// Helpers
// ===========================================================================

/// Parse an ObjectID from a hex string.
fn parse_object_id(s: &str) -> Option<ObjectID> {
    // ObjectID is 32 bytes, represented as 64 hex chars (with optional 0x prefix)
    let hex = s.strip_prefix("0x").unwrap_or(s);
    if hex.len() != 64 {
        return None;
    }
    let bytes: [u8; 32] = hex::decode(hex).ok()?.try_into().ok()?;
    ObjectID::from_bytes(bytes).ok()
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proxy_config_default() {
        let config = ProxyConfig::default();
        assert_eq!(config.base_download_timeout, Duration::from_secs(30));
        assert_eq!(config.nanoseconds_per_byte, 50);
    }

    #[test]
    fn test_proxy_config_timeout_calculation() {
        let config = ProxyConfig::default();

        // 1 MB should add ~50ms
        let timeout = config.download_timeout_for_size(1024 * 1024);
        assert!(timeout > Duration::from_secs(30));
        assert!(timeout < Duration::from_secs(31));

        // 100 MB should add ~5s
        let timeout = config.download_timeout_for_size(100 * 1024 * 1024);
        assert!(timeout > Duration::from_secs(34));
        assert!(timeout < Duration::from_secs(36));
    }

    #[test]
    fn test_parse_object_id() {
        // Valid 64-char hex
        let hex = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        assert!(parse_object_id(hex).is_some());

        // With 0x prefix
        let hex_prefixed = format!("0x{}", hex);
        assert!(parse_object_id(&hex_prefixed).is_some());

        // Invalid length
        assert!(parse_object_id("0123456789abcdef").is_none());

        // Invalid characters
        assert!(
            parse_object_id("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdeg")
                .is_none()
        );
    }

    #[test]
    fn test_availability_report_variants() {
        let target_id = ObjectID::random();
        let model_id = ObjectID::random();

        let report1 = AvailabilityReport::Submission(target_id);
        let report2 = AvailabilityReport::Model(model_id);

        // Just verify they can be created and matched
        match report1 {
            AvailabilityReport::Submission(id) => assert_eq!(id, target_id),
            _ => panic!("Expected Submission"),
        }
        match report2 {
            AvailabilityReport::Model(id) => assert_eq!(id, model_id),
            _ => panic!("Expected Model"),
        }
    }

    // =========================================================================
    // ObjectID Parsing Edge Cases
    // =========================================================================

    #[test]
    fn test_parse_object_id_uppercase() {
        // Uppercase hex should also work (hex crate handles it)
        let hex = "0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF";
        assert!(parse_object_id(hex).is_some());
    }

    #[test]
    fn test_parse_object_id_mixed_case() {
        let hex = "0123456789AbCdEf0123456789aBcDeF0123456789ABcdef0123456789abcdef";
        assert!(parse_object_id(hex).is_some());
    }

    #[test]
    fn test_parse_object_id_too_short() {
        // 63 characters (1 short)
        let hex = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcde";
        assert!(parse_object_id(hex).is_none());
    }

    #[test]
    fn test_parse_object_id_too_long() {
        // 65 characters (1 too many)
        let hex = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdeff";
        assert!(parse_object_id(hex).is_none());
    }

    #[test]
    fn test_parse_object_id_empty() {
        assert!(parse_object_id("").is_none());
    }

    #[test]
    fn test_parse_object_id_just_prefix() {
        assert!(parse_object_id("0x").is_none());
    }

    #[test]
    fn test_parse_object_id_leading_zeros() {
        let hex = "0000000000000000000000000000000000000000000000000000000000000000";
        let result = parse_object_id(hex);
        assert!(result.is_some());
    }

    // =========================================================================
    // ProxyError Display Tests
    // =========================================================================

    #[test]
    fn test_proxy_error_display_target_not_found() {
        let id = ObjectID::random();
        let err = ProxyError::TargetNotFound(id);
        let msg = err.to_string();
        assert!(msg.contains("Target not found"));
        assert!(msg.contains(&id.to_string()));
    }

    #[test]
    fn test_proxy_error_display_model_not_found() {
        let id = ObjectID::random();
        let err = ProxyError::ModelNotFound(id);
        let msg = err.to_string();
        assert!(msg.contains("Model not found"));
        assert!(msg.contains(&id.to_string()));
    }

    #[test]
    fn test_proxy_error_display_download_failed() {
        let err = ProxyError::DownloadFailed("connection timeout".into());
        let msg = err.to_string();
        assert!(msg.contains("Download failed"));
        assert!(msg.contains("connection timeout"));
    }

    #[test]
    fn test_proxy_error_display_cancelled() {
        let err = ProxyError::Cancelled;
        assert!(err.to_string().contains("cancelled"));
    }

    #[test]
    fn test_proxy_error_display_internal() {
        let err = ProxyError::Internal("unexpected state".into());
        let msg = err.to_string();
        assert!(msg.contains("Internal error"));
        assert!(msg.contains("unexpected state"));
    }

    // =========================================================================
    // ProxyConfig Tests
    // =========================================================================

    #[test]
    fn test_proxy_config_custom() {
        let config = ProxyConfig {
            base_download_timeout: Duration::from_secs(60),
            nanoseconds_per_byte: 100,
        };

        assert_eq!(config.base_download_timeout, Duration::from_secs(60));
        assert_eq!(config.nanoseconds_per_byte, 100);
    }

    #[test]
    fn test_proxy_config_timeout_large_file() {
        let config = ProxyConfig::default();

        // 1 GB file should have substantial timeout
        let timeout = config.download_timeout_for_size(1024 * 1024 * 1024);
        // 30s + (1GB * 50ns/byte) = 30s + ~53.69s = ~83.69s
        assert!(timeout > Duration::from_secs(80));
        assert!(timeout < Duration::from_secs(90));
    }

    #[test]
    fn test_proxy_config_timeout_zero_size() {
        let config = ProxyConfig::default();
        let timeout = config.download_timeout_for_size(0);
        assert_eq!(timeout, config.base_download_timeout);
    }

    #[test]
    fn test_proxy_config_timeout_overflow_safety() {
        let config = ProxyConfig::default();
        // u64::MAX should not panic (uses saturating_mul)
        let timeout = config.download_timeout_for_size(u64::MAX);
        // Should be capped at some reasonable value due to Duration limits
        assert!(timeout >= config.base_download_timeout);
    }
}
