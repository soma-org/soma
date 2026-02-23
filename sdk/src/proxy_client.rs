//! Proxy client for fetching submission data and model weights from validators.
//!
//! The `ProxyClient` fetches data from validator proxy servers, with:
//! - **Shuffle for load balancing**: Validators are shuffled before each request
//! - **Round-robin retry**: On failure, try the next validator until all exhausted
//! - **Size-based timeouts**: Timeout scales with expected data size
//!
//! # Usage
//!
//! ```no_run
//! use sdk::proxy_client::ProxyClient;
//! use types::system_state::SystemState;
//!
//! # async fn example(system_state: &SystemState) -> Result<(), sdk::proxy_client::ProxyError> {
//! let client = ProxyClient::from_system_state(system_state)?;
//!
//! // Fetch submission data
//! let target_id = types::object::ObjectID::random();
//! let data = client.fetch_submission_data(&target_id).await?;
//!
//! // Fetch model weights
//! let model_id = types::object::ObjectID::random();
//! let weights = client.fetch_model(&model_id).await?;
//! # Ok(())
//! # }
//! ```

use std::time::Duration;

use rand::seq::SliceRandom as _;
use tracing::{info, warn};
use url::Url;

use types::{
    base::AuthorityName, model::ModelId, multiaddr::Multiaddr, object::ObjectID,
    system_state::SystemState, target::TargetId,
};

// ===========================================================================
// Errors
// ===========================================================================

/// Errors that can occur in the proxy client.
#[derive(Debug, Clone)]
pub enum ProxyError {
    /// No validators have proxy addresses configured
    NoValidators,
    /// All validators failed to serve the request
    AllValidatorsFailed { attempts: usize, last_error: String },
    /// Failed to build HTTP client
    ClientBuildError(String),
    /// Network error during request
    NetworkError(String),
    /// Validator returned an error status
    ValidatorError(u16, String),
    /// Invalid proxy address format
    InvalidProxyAddress(String),
}

impl std::fmt::Display for ProxyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProxyError::NoValidators => write!(f, "No validators have proxy addresses configured"),
            ProxyError::AllValidatorsFailed { attempts, last_error } => {
                write!(f, "All {} validators failed, last error: {}", attempts, last_error)
            }
            ProxyError::ClientBuildError(msg) => write!(f, "Failed to build HTTP client: {}", msg),
            ProxyError::NetworkError(msg) => write!(f, "Network error: {}", msg),
            ProxyError::ValidatorError(status, msg) => {
                write!(f, "Validator returned error {}: {}", status, msg)
            }
            ProxyError::InvalidProxyAddress(addr) => {
                write!(f, "Invalid proxy address format: {}", addr)
            }
        }
    }
}

impl std::error::Error for ProxyError {}

// ===========================================================================
// Validator Proxy Info
// ===========================================================================

/// Information about a validator's proxy server.
#[derive(Debug, Clone)]
pub struct ValidatorProxyInfo {
    /// Validator's authority name (for logging)
    pub name: AuthorityName,
    /// Proxy server base URL (e.g., "http://validator1.soma.io:8080")
    pub proxy_url: Url,
}

// ===========================================================================
// Proxy Client Configuration
// ===========================================================================

/// Configuration for the proxy client.
#[derive(Debug, Clone)]
pub struct ProxyClientConfig {
    /// Base timeout for downloads (before size-based adjustment)
    pub base_timeout: Duration,
    /// Nanoseconds per byte for timeout calculation
    /// Default: 50ns/byte (~20MB/s minimum expected throughput)
    pub nanoseconds_per_byte: u64,
    /// Maximum timeout for any single request
    pub max_timeout: Duration,
}

impl Default for ProxyClientConfig {
    fn default() -> Self {
        Self {
            base_timeout: Duration::from_secs(30),
            nanoseconds_per_byte: 50,
            max_timeout: Duration::from_secs(600), // 10 minutes max
        }
    }
}

impl ProxyClientConfig {
    /// Calculate timeout based on expected data size.
    pub fn timeout_for_size(&self, size_bytes: u64) -> Duration {
        let size_based = Duration::from_nanos(size_bytes.saturating_mul(self.nanoseconds_per_byte));
        std::cmp::min(self.base_timeout + size_based, self.max_timeout)
    }
}

// ===========================================================================
// Proxy Client
// ===========================================================================

/// Client for fetching submission data and model weights from validator proxies.
///
/// # Load Balancing
///
/// The client shuffles validators before each request to distribute load.
/// If a validator fails, it tries the next one in the shuffled order.
///
/// # Thread Safety
///
/// `ProxyClient` is `Clone` and thread-safe. The underlying HTTP client uses
/// connection pooling for efficiency.
#[derive(Clone)]
pub struct ProxyClient {
    /// List of validators with proxy addresses
    validators: Vec<ValidatorProxyInfo>,
    /// HTTP client (with connection pooling)
    http_client: reqwest::Client,
    /// Configuration
    config: ProxyClientConfig,
}

impl ProxyClient {
    /// Create a new proxy client with the given validators.
    pub fn new(
        validators: Vec<ValidatorProxyInfo>,
        config: ProxyClientConfig,
    ) -> Result<Self, ProxyError> {
        if validators.is_empty() {
            return Err(ProxyError::NoValidators);
        }

        let http_client = reqwest::Client::builder()
            .timeout(config.max_timeout)
            .pool_max_idle_per_host(10)
            .http2_prior_knowledge() // Use HTTP/2 for better performance
            .build()
            .map_err(|e| ProxyError::ClientBuildError(e.to_string()))?;

        Ok(Self { validators, http_client, config })
    }

    /// Create a proxy client from SystemState.
    ///
    /// Extracts validator proxy addresses from the validator set.
    pub fn from_system_state(state: &SystemState) -> Result<Self, ProxyError> {
        let validators: Vec<ValidatorProxyInfo> = state
            .validators()
            .validators
            .iter()
            .filter_map(|v| {
                let proxy_url = multiaddr_to_http_url(&v.metadata.proxy_address)?;
                Some(ValidatorProxyInfo { name: (&v.metadata.protocol_pubkey).into(), proxy_url })
            })
            .collect();

        Self::new(validators, ProxyClientConfig::default())
    }

    /// Create a proxy client that fetches from a single fullnode proxy URL.
    ///
    /// The fullnode exposes `/data/{target_id}` and `/model/{model_id}` routes
    /// on the same HTTP server as its RPC endpoint. This is the simplest way
    /// to create a proxy client — just pass the fullnode URL.
    pub fn from_url(url: impl AsRef<str>) -> Result<Self, ProxyError> {
        Self::from_url_with_config(url, ProxyClientConfig::default())
    }

    /// Create a proxy client from a fullnode proxy URL with custom config.
    pub fn from_url_with_config(
        url: impl AsRef<str>,
        config: ProxyClientConfig,
    ) -> Result<Self, ProxyError> {
        let proxy_url = Url::parse(url.as_ref())
            .map_err(|e| ProxyError::InvalidProxyAddress(format!("{}: {}", url.as_ref(), e)))?;
        let validators = vec![ValidatorProxyInfo { name: AuthorityName::default(), proxy_url }];
        Self::new(validators, config)
    }

    /// Create a proxy client with custom config from SystemState.
    pub fn from_system_state_with_config(
        state: &SystemState,
        config: ProxyClientConfig,
    ) -> Result<Self, ProxyError> {
        let validators: Vec<ValidatorProxyInfo> = state
            .validators()
            .validators
            .iter()
            .filter_map(|v| {
                let proxy_url = multiaddr_to_http_url(&v.metadata.proxy_address)?;
                Some(ValidatorProxyInfo { name: (&v.metadata.protocol_pubkey).into(), proxy_url })
            })
            .collect();

        Self::new(validators, config)
    }

    /// Returns the number of validators with proxy addresses.
    pub fn validator_count(&self) -> usize {
        self.validators.len()
    }

    /// Fetch submission data for a filled target.
    ///
    /// Shuffles validators and tries each in order until one succeeds.
    pub async fn fetch_submission_data(&self, target_id: &TargetId) -> Result<Vec<u8>, ProxyError> {
        self.fetch_with_retry(&format!("/data/{}", target_id)).await
    }

    /// Fetch model weights for an active model.
    ///
    /// Shuffles validators and tries each in order until one succeeds.
    pub async fn fetch_model(&self, model_id: &ModelId) -> Result<Vec<u8>, ProxyError> {
        self.fetch_with_retry(&format!("/model/{}", model_id)).await
    }

    /// Fetch submission data with a custom timeout.
    ///
    /// Use this when you know the expected data size.
    pub async fn fetch_submission_data_with_size(
        &self,
        target_id: &TargetId,
        expected_size: u64,
    ) -> Result<Vec<u8>, ProxyError> {
        let timeout = self.config.timeout_for_size(expected_size);
        self.fetch_with_retry_and_timeout(&format!("/data/{}", target_id), timeout).await
    }

    /// Fetch model weights with a custom timeout.
    ///
    /// Use this when you know the expected model size.
    pub async fn fetch_model_with_size(
        &self,
        model_id: &ModelId,
        expected_size: u64,
    ) -> Result<Vec<u8>, ProxyError> {
        let timeout = self.config.timeout_for_size(expected_size);
        self.fetch_with_retry_and_timeout(&format!("/model/{}", model_id), timeout).await
    }

    /// Internal method to fetch with retry across validators.
    async fn fetch_with_retry(&self, path: &str) -> Result<Vec<u8>, ProxyError> {
        self.fetch_with_retry_and_timeout(path, self.config.base_timeout).await
    }

    /// Internal method to fetch with retry and custom timeout.
    async fn fetch_with_retry_and_timeout(
        &self,
        path: &str,
        timeout: Duration,
    ) -> Result<Vec<u8>, ProxyError> {
        // Shuffle validators for load balancing
        let mut validators = self.validators.clone();
        validators.shuffle(&mut rand::thread_rng());

        let mut last_error = String::new();
        let mut attempts = 0;

        for validator in &validators {
            attempts += 1;

            let url = match validator.proxy_url.join(path) {
                Ok(u) => u,
                Err(e) => {
                    last_error = format!("Invalid URL: {}", e);
                    continue;
                }
            };

            match self.fetch_from_url(&url, timeout).await {
                Ok(data) => {
                    info!(
                        "Successfully fetched {} bytes from validator {:?}",
                        data.len(),
                        validator.name
                    );
                    return Ok(data);
                }
                Err(e) => {
                    warn!("Failed to fetch from validator {:?}: {}", validator.name, e);
                    last_error = e.to_string();
                }
            }
        }

        Err(ProxyError::AllValidatorsFailed { attempts, last_error })
    }

    /// Fetch data from a specific URL.
    async fn fetch_from_url(&self, url: &Url, timeout: Duration) -> Result<Vec<u8>, ProxyError> {
        let response = self
            .http_client
            .get(url.as_str())
            .timeout(timeout)
            .send()
            .await
            .map_err(|e| ProxyError::NetworkError(e.to_string()))?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(ProxyError::ValidatorError(status.as_u16(), body));
        }

        let bytes = response.bytes().await.map_err(|e| ProxyError::NetworkError(e.to_string()))?;

        Ok(bytes.to_vec())
    }
}

// ===========================================================================
// Helpers
// ===========================================================================

/// Convert a Multiaddr to an HTTP URL.
///
/// Expects a Multiaddr like `/ip4/127.0.0.1/tcp/8080/http` or `/dns/example.com/tcp/8080/http`.
/// Returns `http://{hostname}:{port}` as a URL.
fn multiaddr_to_http_url(addr: &Multiaddr) -> Option<Url> {
    let hostname = addr.hostname()?;
    let port = addr.port()?;

    // Check if it has /http or /https protocol
    let scheme = if addr.to_string().contains("/https") { "https" } else { "http" };

    let url_str = format!("{}://{}:{}", scheme, hostname, port);
    Url::parse(&url_str).ok()
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proxy_client_config_default() {
        let config = ProxyClientConfig::default();
        assert_eq!(config.base_timeout, Duration::from_secs(30));
        assert_eq!(config.nanoseconds_per_byte, 50);
        assert_eq!(config.max_timeout, Duration::from_secs(600));
    }

    #[test]
    fn test_proxy_client_config_timeout_calculation() {
        let config = ProxyClientConfig::default();

        // 1 MB should add ~50ms
        let timeout = config.timeout_for_size(1024 * 1024);
        assert!(timeout > Duration::from_secs(30));
        assert!(timeout < Duration::from_secs(31));

        // 100 MB should add ~5s
        let timeout = config.timeout_for_size(100 * 1024 * 1024);
        assert!(timeout > Duration::from_secs(34));
        assert!(timeout < Duration::from_secs(36));

        // Very large size should cap at max_timeout
        let timeout = config.timeout_for_size(u64::MAX);
        assert_eq!(timeout, config.max_timeout);
    }

    #[test]
    fn test_proxy_client_no_validators() {
        let result = ProxyClient::new(vec![], ProxyClientConfig::default());
        assert!(matches!(result, Err(ProxyError::NoValidators)));
    }

    #[test]
    fn test_validator_proxy_info_creation() {
        let url = Url::parse("http://validator1.soma.io:8080").unwrap();
        let info = ValidatorProxyInfo { name: AuthorityName::default(), proxy_url: url.clone() };
        assert_eq!(info.proxy_url, url);
    }

    #[test]
    fn test_proxy_error_display() {
        let err = ProxyError::NoValidators;
        assert_eq!(err.to_string(), "No validators have proxy addresses configured");

        let err = ProxyError::AllValidatorsFailed {
            attempts: 3,
            last_error: "connection refused".into(),
        };
        assert!(err.to_string().contains("3 validators failed"));
        assert!(err.to_string().contains("connection refused"));
    }

    // =========================================================================
    // Multiaddr Conversion Tests
    // =========================================================================

    #[test]
    fn test_multiaddr_to_http_url_basic() {
        use std::str::FromStr;
        use types::multiaddr::Multiaddr;

        let addr = Multiaddr::from_str("/ip4/127.0.0.1/tcp/8080/http").unwrap();
        let url = multiaddr_to_http_url(&addr);

        assert!(url.is_some());
        let url = url.unwrap();
        assert_eq!(url.scheme(), "http");
        assert_eq!(url.host_str(), Some("127.0.0.1"));
        assert_eq!(url.port(), Some(8080));
    }

    #[test]
    fn test_multiaddr_to_http_url_https() {
        use std::str::FromStr;
        use types::multiaddr::Multiaddr;

        let addr = Multiaddr::from_str("/ip4/192.168.1.1/tcp/443/https").unwrap();
        let url = multiaddr_to_http_url(&addr);

        assert!(url.is_some());
        let url = url.unwrap();
        assert_eq!(url.scheme(), "https");
        // port() returns None for default ports (443 for https), use port_or_known_default()
        assert_eq!(url.port_or_known_default(), Some(443));
    }

    #[test]
    fn test_multiaddr_to_http_url_dns() {
        use std::str::FromStr;
        use types::multiaddr::Multiaddr;

        let addr = Multiaddr::from_str("/dns/example.com/tcp/8080/http").unwrap();
        let url = multiaddr_to_http_url(&addr);

        assert!(url.is_some());
        let url = url.unwrap();
        assert_eq!(url.host_str(), Some("example.com"));
    }

    #[test]
    fn test_multiaddr_to_http_url_missing_port() {
        use std::str::FromStr;
        use types::multiaddr::Multiaddr;

        // Multiaddr without port should return None
        let addr = Multiaddr::from_str("/ip4/127.0.0.1").unwrap();
        let url = multiaddr_to_http_url(&addr);

        assert!(url.is_none());
    }

    // =========================================================================
    // Timeout Calculation Edge Cases
    // =========================================================================

    #[test]
    fn test_timeout_for_size_zero() {
        let config = ProxyClientConfig::default();
        let timeout = config.timeout_for_size(0);
        assert_eq!(timeout, config.base_timeout);
    }

    #[test]
    fn test_timeout_for_size_small() {
        let config = ProxyClientConfig::default();
        // 1 KB should add very little time
        let timeout = config.timeout_for_size(1024);
        let expected_nanos = 1024 * config.nanoseconds_per_byte;
        assert!(timeout < config.base_timeout + Duration::from_millis(1));
    }

    #[test]
    fn test_timeout_for_size_overflow_safety() {
        let config = ProxyClientConfig::default();
        // u64::MAX should not overflow and should cap at max_timeout
        let timeout = config.timeout_for_size(u64::MAX);
        assert_eq!(timeout, config.max_timeout);
    }

    // =========================================================================
    // ProxyError Variants Tests
    // =========================================================================

    #[test]
    fn test_proxy_error_client_build_error() {
        let err = ProxyError::ClientBuildError("TLS initialization failed".into());
        assert!(err.to_string().contains("TLS initialization failed"));
    }

    #[test]
    fn test_proxy_error_network_error() {
        let err = ProxyError::NetworkError("connection timeout".into());
        assert!(err.to_string().contains("connection timeout"));
    }

    #[test]
    fn test_proxy_error_validator_error() {
        let err = ProxyError::ValidatorError(404, "not found".into());
        assert!(err.to_string().contains("404"));
        assert!(err.to_string().contains("not found"));
    }

    #[test]
    fn test_proxy_error_invalid_proxy_address() {
        let err = ProxyError::InvalidProxyAddress("/invalid/multiaddr".into());
        assert!(err.to_string().contains("Invalid proxy address"));
    }

    // =========================================================================
    // ProxyClient with validators Tests
    // =========================================================================

    #[test]
    fn test_proxy_client_validator_count() {
        let validators = vec![
            ValidatorProxyInfo {
                name: AuthorityName::default(),
                proxy_url: Url::parse("http://v1.soma.io:8080").unwrap(),
            },
            ValidatorProxyInfo {
                name: AuthorityName::default(),
                proxy_url: Url::parse("http://v2.soma.io:8080").unwrap(),
            },
        ];

        let client = ProxyClient::new(validators, ProxyClientConfig::default()).unwrap();
        assert_eq!(client.validator_count(), 2);
    }

    #[test]
    fn test_proxy_client_single_validator() {
        let validators = vec![ValidatorProxyInfo {
            name: AuthorityName::default(),
            proxy_url: Url::parse("http://single.soma.io:8080").unwrap(),
        }];

        let client = ProxyClient::new(validators, ProxyClientConfig::default());
        assert!(client.is_ok());
        assert_eq!(client.unwrap().validator_count(), 1);
    }

    #[test]
    fn test_proxy_client_config_custom() {
        let config = ProxyClientConfig {
            base_timeout: Duration::from_secs(60),
            nanoseconds_per_byte: 100,
            max_timeout: Duration::from_secs(1200),
        };

        assert_eq!(config.base_timeout, Duration::from_secs(60));
        assert_eq!(config.nanoseconds_per_byte, 100);
        assert_eq!(config.max_timeout, Duration::from_secs(1200));
    }
}
