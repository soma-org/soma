//! Fullnode Thin Proxy for forwarding data/model requests to validators.
//!
//! Fullnodes run this proxy to forward client requests to validators. This allows
//! clients to connect to a single fullnode endpoint rather than directly to validators.
//!
//! The thin proxy:
//! - Forwards `/data/{target_id}` and `/model/{model_id}` requests to validators
//! - Uses the ProxyClient for shuffle + round-robin retry across validators
//! - Caches the validator list from SystemState (refreshed on epoch change)
//! - Does NOT cache actual data (validators handle caching via singleflight)
//!
//! # Architecture
//!
//! ```text
//! Client → Fullnode Thin Proxy → Validator Proxy Server
//!                │                       │
//!                └── ProxyClient ────────┘
//!                    (shuffle + retry)
//! ```

use std::sync::Arc;

use arc_swap::ArcSwap;
use axum::{
    Router,
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    routing::get,
};
use tokio::sync::RwLock;
use tracing::{info, warn};

use sdk::proxy_client::{ProxyClient, ProxyClientConfig};

use crate::authority::AuthorityState;

// ===========================================================================
// Fullnode Proxy Errors
// ===========================================================================

/// Errors that can occur in the fullnode proxy.
#[derive(Debug, Clone)]
pub enum FullnodeProxyError {
    /// No validators have proxy addresses
    NoValidators,
    /// Proxy client not initialized
    NotInitialized,
    /// Failed to fetch from validators
    FetchFailed(String),
    /// Invalid resource ID format
    InvalidId(String),
}

impl std::fmt::Display for FullnodeProxyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FullnodeProxyError::NoValidators => {
                write!(f, "No validators have proxy addresses configured")
            }
            FullnodeProxyError::NotInitialized => {
                write!(f, "Proxy client not initialized")
            }
            FullnodeProxyError::FetchFailed(msg) => {
                write!(f, "Failed to fetch from validators: {}", msg)
            }
            FullnodeProxyError::InvalidId(msg) => {
                write!(f, "Invalid resource ID: {}", msg)
            }
        }
    }
}

impl IntoResponse for FullnodeProxyError {
    fn into_response(self) -> axum::response::Response {
        let (status, message) = match &self {
            FullnodeProxyError::NoValidators => (StatusCode::SERVICE_UNAVAILABLE, self.to_string()),
            FullnodeProxyError::NotInitialized => {
                (StatusCode::SERVICE_UNAVAILABLE, self.to_string())
            }
            FullnodeProxyError::FetchFailed(_) => (StatusCode::BAD_GATEWAY, self.to_string()),
            FullnodeProxyError::InvalidId(_) => (StatusCode::BAD_REQUEST, self.to_string()),
        };
        (status, message).into_response()
    }
}

// ===========================================================================
// Fullnode Proxy
// ===========================================================================

/// Fullnode thin proxy that forwards requests to validators.
///
/// This is designed to be run on fullnodes to provide a single endpoint for
/// clients to fetch data/model from, while the fullnode handles validator
/// discovery and load balancing.
pub struct FullnodeProxy {
    /// Authority state for reading SystemState
    state: Arc<AuthorityState>,
    /// Cached proxy client (refreshed on epoch change)
    proxy_client: ArcSwap<Option<ProxyClient>>,
    /// Configuration for the proxy client
    config: ProxyClientConfig,
    /// Last epoch for which we refreshed the client
    last_epoch: RwLock<u64>,
}

impl FullnodeProxy {
    /// Create a new fullnode proxy.
    pub fn new(state: Arc<AuthorityState>, config: ProxyClientConfig) -> Self {
        Self {
            state,
            proxy_client: ArcSwap::from_pointee(None),
            config,
            last_epoch: RwLock::new(0),
        }
    }

    /// Create with default configuration.
    pub fn with_default_config(state: Arc<AuthorityState>) -> Self {
        Self::new(state, ProxyClientConfig::default())
    }

    /// Build an axum router for the fullnode proxy.
    pub fn router(self: Arc<Self>) -> Router {
        Router::new()
            .route("/data/{target_id}", get(Self::forward_data))
            .route("/model/{model_id}", get(Self::forward_model))
            .with_state(self)
    }

    /// Ensure the proxy client is initialized and up-to-date.
    async fn ensure_client(&self) -> Result<ProxyClient, FullnodeProxyError> {
        // Check if we need to refresh the client
        let current_epoch = self.get_current_epoch().await?;

        {
            let last_epoch = self.last_epoch.read().await;
            if *last_epoch == current_epoch {
                if let Some(client) = self.proxy_client.load().as_ref() {
                    return Ok(client.clone());
                }
            }
        }

        // Need to refresh the client
        self.refresh_client(current_epoch).await
    }

    /// Refresh the proxy client from SystemState.
    async fn refresh_client(&self, epoch: u64) -> Result<ProxyClient, FullnodeProxyError> {
        let mut last_epoch = self.last_epoch.write().await;

        // Double-check after acquiring write lock
        if *last_epoch == epoch {
            if let Some(client) = self.proxy_client.load().as_ref() {
                return Ok(client.clone());
            }
        }

        // Get SystemState and create ProxyClient
        let system_state = self
            .state
            .get_object_cache_reader()
            .get_system_state_object()
            .map_err(|e| FullnodeProxyError::FetchFailed(e.to_string()))?;

        let client = ProxyClient::from_system_state_with_config(&system_state, self.config.clone())
            .map_err(|_| FullnodeProxyError::NoValidators)?;

        info!(
            "Refreshed fullnode proxy with {} validators for epoch {}",
            client.validator_count(),
            epoch
        );

        self.proxy_client.store(Arc::new(Some(client.clone())));
        *last_epoch = epoch;

        Ok(client)
    }

    /// Get current epoch from AuthorityState.
    async fn get_current_epoch(&self) -> Result<u64, FullnodeProxyError> {
        let epoch_store = self.state.load_epoch_store_one_call_per_task();
        Ok(epoch_store.epoch())
    }

    /// Forward a data request to validators.
    async fn forward_data(
        State(proxy): State<Arc<FullnodeProxy>>,
        Path(target_id_str): Path<String>,
    ) -> Result<impl IntoResponse, FullnodeProxyError> {
        let target_id = parse_object_id(&target_id_str)
            .ok_or_else(|| FullnodeProxyError::InvalidId(target_id_str.clone()))?;

        let client = proxy.ensure_client().await?;

        let data = client
            .fetch_submission_data(&target_id)
            .await
            .map_err(|e| FullnodeProxyError::FetchFailed(e.to_string()))?;

        info!(
            "Forwarded {} bytes for target {} through fullnode proxy",
            data.len(),
            target_id
        );

        Ok((StatusCode::OK, data))
    }

    /// Forward a model request to validators.
    async fn forward_model(
        State(proxy): State<Arc<FullnodeProxy>>,
        Path(model_id_str): Path<String>,
    ) -> Result<impl IntoResponse, FullnodeProxyError> {
        let model_id = parse_object_id(&model_id_str)
            .ok_or_else(|| FullnodeProxyError::InvalidId(model_id_str.clone()))?;

        let client = proxy.ensure_client().await?;

        let data = client
            .fetch_model(&model_id)
            .await
            .map_err(|e| FullnodeProxyError::FetchFailed(e.to_string()))?;

        info!(
            "Forwarded {} bytes for model {} through fullnode proxy",
            data.len(),
            model_id
        );

        Ok((StatusCode::OK, data))
    }
}

// ===========================================================================
// Helpers
// ===========================================================================

/// Parse an ObjectID from a hex string.
fn parse_object_id(s: &str) -> Option<types::object::ObjectID> {
    let hex = s.strip_prefix("0x").unwrap_or(s);
    if hex.len() != 64 {
        return None;
    }
    let bytes: [u8; 32] = hex::decode(hex).ok()?.try_into().ok()?;
    types::object::ObjectID::from_bytes(bytes).ok()
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

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
    }

    #[test]
    fn test_fullnode_proxy_error_display() {
        let err = FullnodeProxyError::NoValidators;
        assert!(err.to_string().contains("No validators"));

        let err = FullnodeProxyError::FetchFailed("timeout".into());
        assert!(err.to_string().contains("timeout"));
    }
}
