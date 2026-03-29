//! Ethereum RPC client wrapper.
//!
//! Uses raw JSON-RPC via reqwest instead of alloy to avoid version compatibility
//! issues. Only needs two RPC methods: `eth_getBlockByNumber` and `eth_getLogs`.

use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU32, Ordering};
use tracing::{debug, info, warn};

use crate::error::{BridgeError, BridgeResult};
use crate::types::DepositEvent;

/// Ethereum RPC client for bridge operations.
pub struct EthClient {
    client: reqwest::Client,
    bridge_contract: String,
    /// Current RPC endpoint index (for rotation on failure).
    current_endpoint_idx: AtomicU32,
    /// All configured RPC endpoints.
    rpc_urls: Vec<String>,
    /// Consecutive failure count per endpoint.
    failure_counts: Vec<AtomicU32>,
}

/// JSON-RPC request.
#[derive(Serialize)]
struct JsonRpcRequest<'a> {
    jsonrpc: &'a str,
    method: &'a str,
    params: serde_json::Value,
    id: u64,
}

/// JSON-RPC response.
#[derive(Deserialize)]
struct JsonRpcResponse<T> {
    result: Option<T>,
    error: Option<JsonRpcError>,
}

#[derive(Deserialize, Debug)]
struct JsonRpcError {
    code: i64,
    message: String,
}

/// Minimal Ethereum block header for finalized block queries.
#[derive(Deserialize)]
struct EthBlock {
    #[serde(deserialize_with = "deserialize_hex_u64")]
    number: u64,
}

/// Ethereum log entry from eth_getLogs.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct EthLog {
    address: String,
    topics: Vec<String>,
    data: String,
    #[serde(default, deserialize_with = "deserialize_optional_hex_u64")]
    block_number: Option<u64>,
    #[serde(default)]
    transaction_hash: Option<String>,
}

fn deserialize_hex_u64<'de, D>(deserializer: D) -> Result<u64, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let s: String = String::deserialize(deserializer)?;
    let s = s.strip_prefix("0x").unwrap_or(&s);
    u64::from_str_radix(s, 16).map_err(serde::de::Error::custom)
}

fn deserialize_optional_hex_u64<'de, D>(deserializer: D) -> Result<Option<u64>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let opt: Option<String> = Option::deserialize(deserializer)?;
    match opt {
        Some(s) => {
            let s = s.strip_prefix("0x").unwrap_or(&s);
            Ok(Some(
                u64::from_str_radix(s, 16).map_err(serde::de::Error::custom)?,
            ))
        }
        None => Ok(None),
    }
}

impl EthClient {
    /// Create a new EthClient.
    pub async fn new(
        rpc_urls: Vec<String>,
        bridge_contract_address: &str,
    ) -> BridgeResult<Self> {
        if rpc_urls.is_empty() {
            return Err(BridgeError::ConfigError(
                "At least one RPC URL required".into(),
            ));
        }

        let failure_counts = rpc_urls.iter().map(|_| AtomicU32::new(0)).collect();

        let client = Self {
            client: reqwest::Client::new(),
            bridge_contract: bridge_contract_address.to_lowercase(),
            current_endpoint_idx: AtomicU32::new(0),
            rpc_urls,
            failure_counts,
        };

        // Verify connectivity
        let chain_id = client.get_chain_id().await?;
        info!(chain_id, "EthClient connected to Ethereum");

        Ok(client)
    }

    fn current_url(&self) -> &str {
        let idx = self.current_endpoint_idx.load(Ordering::Relaxed) as usize;
        &self.rpc_urls[idx]
    }

    async fn rpc_call<T: serde::de::DeserializeOwned>(
        &self,
        method: &str,
        params: serde_json::Value,
    ) -> BridgeResult<T> {
        let req = JsonRpcRequest {
            jsonrpc: "2.0",
            method,
            params,
            id: 1,
        };

        let resp = self
            .client
            .post(self.current_url())
            .json(&req)
            .send()
            .await
            .map_err(|e| BridgeError::ProviderError(e.to_string()))?;

        let body: JsonRpcResponse<T> = resp
            .json()
            .await
            .map_err(|e| BridgeError::ProviderError(format!("Failed to parse response: {e}")))?;

        if let Some(err) = body.error {
            let msg = format!("RPC error {}: {}", err.code, err.message);
            // Detect the "too many results" error
            if err.code == -32005
                || err.message.contains("query returned more than")
                || err.message.contains("Log response size exceeded")
            {
                return Err(BridgeError::TransientProviderError(msg));
            }
            return Err(BridgeError::ProviderError(msg));
        }

        body.result
            .ok_or_else(|| BridgeError::ProviderError("RPC returned null result".into()))
    }

    /// Get the Ethereum chain ID.
    pub async fn get_chain_id(&self) -> BridgeResult<u64> {
        let result: String = self
            .rpc_call("eth_chainId", serde_json::json!([]))
            .await?;
        let s = result.strip_prefix("0x").unwrap_or(&result);
        u64::from_str_radix(s, 16)
            .map_err(|e| BridgeError::ProviderError(format!("Invalid chain ID: {e}")))
    }

    /// Get the latest finalized block number.
    pub async fn get_last_finalized_block_id(&self) -> BridgeResult<u64> {
        let block: EthBlock = self
            .rpc_call(
                "eth_getBlockByNumber",
                serde_json::json!(["finalized", false]),
            )
            .await?;
        Ok(block.number)
    }

    /// Query deposit events from the bridge contract in a block range.
    pub async fn get_deposit_events_in_range(
        &self,
        from_block: u64,
        to_block: u64,
    ) -> BridgeResult<Vec<DepositEvent>> {
        debug!(from_block, to_block, "Querying deposit events");

        let logs: Vec<EthLog> = self
            .rpc_call(
                "eth_getLogs",
                serde_json::json!([{
                    "address": self.bridge_contract,
                    "fromBlock": format!("0x{from_block:x}"),
                    "toBlock": format!("0x{to_block:x}")
                }]),
            )
            .await?;

        let mut events = Vec::new();
        for log in &logs {
            if let Some(event) = self.parse_deposit_log(log)? {
                events.push(event);
            }
        }

        debug!(count = events.len(), "Parsed deposit events");
        Ok(events)
    }

    /// Parse a raw Ethereum log into a DepositEvent.
    fn parse_deposit_log(&self, log: &EthLog) -> BridgeResult<Option<DepositEvent>> {
        // Verify the log is from our bridge contract
        if log.address.to_lowercase() != self.bridge_contract {
            return Ok(None);
        }

        // Decode hex data
        let data_hex = log.data.strip_prefix("0x").unwrap_or(&log.data);
        let data = hex::decode(data_hex)
            .map_err(|e| BridgeError::Internal(format!("Invalid hex in log data: {e}")))?;

        if data.len() < 128 {
            warn!(
                data_len = data.len(),
                "Deposit event data too short, skipping"
            );
            return Ok(None);
        }

        // ABI-encoded fields are padded to 32 bytes each:
        // word 0 (bytes 0..32): uint64 nonce (right-aligned)
        // word 1 (bytes 32..64): address sender (right-aligned, 20 bytes)
        // word 2 (bytes 64..96): bytes32 somaRecipient
        // word 3 (bytes 96..128): uint64 amount (right-aligned)
        let nonce = u64::from_be_bytes(data[24..32].try_into().unwrap());
        let mut eth_sender = [0u8; 20];
        eth_sender.copy_from_slice(&data[44..64]);
        let mut soma_recipient = [0u8; 32];
        soma_recipient.copy_from_slice(&data[64..96]);
        let amount = u64::from_be_bytes(data[120..128].try_into().unwrap());

        let tx_hash = log
            .transaction_hash
            .as_ref()
            .and_then(|h| {
                let h = h.strip_prefix("0x").unwrap_or(h);
                hex::decode(h).ok()
            })
            .map(|bytes| {
                let mut arr = [0u8; 32];
                let len = bytes.len().min(32);
                arr[..len].copy_from_slice(&bytes[..len]);
                arr
            })
            .unwrap_or([0; 32]);

        let block_number = log.block_number.unwrap_or(0);

        Ok(Some(DepositEvent {
            nonce,
            eth_sender,
            soma_recipient,
            amount,
            tx_hash,
            block_number,
        }))
    }

    /// Rotate to the next RPC endpoint after a failure.
    /// Returns true if all endpoints have exceeded the failure threshold.
    pub fn rotate_endpoint(&self, failure_threshold: u32) -> bool {
        let current = self.current_endpoint_idx.load(Ordering::Relaxed);
        let count = &self.failure_counts[current as usize];
        let failures = count.fetch_add(1, Ordering::Relaxed) + 1;

        if failures >= failure_threshold {
            warn!(
                endpoint = current,
                failures, "Endpoint exceeded failure threshold"
            );
        }

        let next = (current + 1) % self.rpc_urls.len() as u32;
        self.current_endpoint_idx.store(next, Ordering::Relaxed);

        // Check if ALL endpoints have exceeded threshold
        self.failure_counts
            .iter()
            .all(|c| c.load(Ordering::Relaxed) >= failure_threshold)
    }

    /// Reset failure count for the current endpoint (on success).
    pub fn reset_failure_count(&self) {
        let current = self.current_endpoint_idx.load(Ordering::Relaxed);
        self.failure_counts[current as usize].store(0, Ordering::Relaxed);
    }

    /// Get the latest finalized block number with retry and backoff.
    pub async fn get_last_finalized_block_id_with_retry(
        &self,
        max_elapsed: std::time::Duration,
    ) -> BridgeResult<u64> {
        crate::retry::retry_with_backoff(
            "get_last_finalized_block_id",
            max_elapsed,
            || self.get_last_finalized_block_id(),
        )
        .await
    }

    /// Query deposit events with retry and backoff.
    pub async fn get_deposit_events_in_range_with_retry(
        &self,
        from_block: u64,
        to_block: u64,
        max_elapsed: std::time::Duration,
    ) -> BridgeResult<Vec<DepositEvent>> {
        crate::retry::retry_with_backoff(
            "get_deposit_events_in_range",
            max_elapsed,
            || self.get_deposit_events_in_range(from_block, to_block),
        )
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deposit_log_parsing() {
        let client = EthClient {
            client: reqwest::Client::new(),
            bridge_contract: "0xabcdef1234567890abcdef1234567890abcdef12".to_string(),
            current_endpoint_idx: AtomicU32::new(0),
            rpc_urls: vec!["http://localhost:8545".into()],
            failure_counts: vec![AtomicU32::new(0)],
        };

        // Construct ABI-encoded event data
        let mut data = vec![0u8; 128];
        // nonce = 42
        data[24..32].copy_from_slice(&42u64.to_be_bytes());
        // eth_sender
        data[44..64].copy_from_slice(&[0xAA; 20]);
        // soma_recipient
        data[64..96].copy_from_slice(&[0xBB; 32]);
        // amount = 1_000_000
        data[120..128].copy_from_slice(&1_000_000u64.to_be_bytes());

        let log = EthLog {
            address: "0xabcdef1234567890abcdef1234567890abcdef12".to_string(),
            topics: vec![],
            data: format!("0x{}", hex::encode(&data)),
            block_number: Some(12345),
            transaction_hash: Some(format!("0x{}", hex::encode([0xCC; 32]))),
        };

        let event = client.parse_deposit_log(&log).unwrap().unwrap();
        assert_eq!(event.nonce, 42);
        assert_eq!(event.eth_sender, [0xAA; 20]);
        assert_eq!(event.soma_recipient, [0xBB; 32]);
        assert_eq!(event.amount, 1_000_000);
        assert_eq!(event.block_number, 12345);
        assert_eq!(event.tx_hash, [0xCC; 32]);
    }

    #[test]
    fn test_wrong_contract_address_returns_none() {
        let client = EthClient {
            client: reqwest::Client::new(),
            bridge_contract: "0x1111111111111111111111111111111111111111".to_string(),
            current_endpoint_idx: AtomicU32::new(0),
            rpc_urls: vec!["http://localhost:8545".into()],
            failure_counts: vec![AtomicU32::new(0)],
        };

        let log = EthLog {
            address: "0x2222222222222222222222222222222222222222".to_string(),
            topics: vec![],
            data: format!("0x{}", hex::encode([0u8; 128])),
            block_number: Some(1),
            transaction_hash: None,
        };

        let result = client.parse_deposit_log(&log).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_short_data_returns_none() {
        let client = EthClient {
            client: reqwest::Client::new(),
            bridge_contract: "0xabcd".to_string(),
            current_endpoint_idx: AtomicU32::new(0),
            rpc_urls: vec!["http://localhost:8545".into()],
            failure_counts: vec![AtomicU32::new(0)],
        };

        let log = EthLog {
            address: "0xabcd".to_string(),
            topics: vec![],
            data: "0x0000".to_string(),
            block_number: None,
            transaction_hash: None,
        };

        let result = client.parse_deposit_log(&log).unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_get_last_finalized_block() {
        use serde_json::json;
        use wiremock::matchers::{body_partial_json, method};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;

        // Mock chain_id (for constructor)
        Mock::given(method("POST"))
            .and(body_partial_json(json!({"method": "eth_chainId"})))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "jsonrpc": "2.0", "id": 1, "result": "0x1"
            })))
            .mount(&server)
            .await;

        // Mock finalized block
        Mock::given(method("POST"))
            .and(body_partial_json(
                json!({"method": "eth_getBlockByNumber"}),
            ))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "jsonrpc": "2.0", "id": 1,
                "result": { "number": "0x1a4" }
            })))
            .mount(&server)
            .await;

        let client = EthClient::new(vec![server.uri()], "0x0001")
            .await
            .unwrap();
        let block = client.get_last_finalized_block_id().await.unwrap();
        assert_eq!(block, 0x1a4); // 420
    }

    #[tokio::test]
    async fn test_get_deposit_events() {
        use serde_json::json;
        use wiremock::matchers::{body_partial_json, method};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        let contract_addr = "0x0000000000000000000000000000000000000001";

        Mock::given(method("POST"))
            .and(body_partial_json(json!({"method": "eth_chainId"})))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "jsonrpc": "2.0", "id": 1, "result": "0x1"
            })))
            .mount(&server)
            .await;

        // Build deposit event data
        let mut data = vec![0u8; 128];
        data[24..32].copy_from_slice(&7u64.to_be_bytes()); // nonce=7
        data[44..64].copy_from_slice(&[0xAA; 20]); // eth_sender
        data[64..96].copy_from_slice(&[0xBB; 32]); // soma_recipient
        data[120..128].copy_from_slice(&5_000_000u64.to_be_bytes()); // amount

        Mock::given(method("POST"))
            .and(body_partial_json(json!({"method": "eth_getLogs"})))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "jsonrpc": "2.0", "id": 1,
                "result": [{
                    "address": contract_addr,
                    "topics": [],
                    "data": format!("0x{}", hex::encode(&data)),
                    "blockNumber": "0x64",
                    "transactionHash": format!("0x{}", hex::encode([0xCC; 32]))
                }]
            })))
            .mount(&server)
            .await;

        let client = EthClient::new(vec![server.uri()], contract_addr)
            .await
            .unwrap();
        let events = client
            .get_deposit_events_in_range(100, 200)
            .await
            .unwrap();

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].nonce, 7);
        assert_eq!(events[0].amount, 5_000_000);
        assert_eq!(events[0].block_number, 100);
    }

    #[tokio::test]
    async fn test_transient_error_detection() {
        use serde_json::json;
        use wiremock::matchers::{body_partial_json, method};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(body_partial_json(json!({"method": "eth_chainId"})))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "jsonrpc": "2.0", "id": 1, "result": "0x1"
            })))
            .mount(&server)
            .await;

        // Return -32005 error for getLogs
        Mock::given(method("POST"))
            .and(body_partial_json(json!({"method": "eth_getLogs"})))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "jsonrpc": "2.0", "id": 1,
                "error": {
                    "code": -32005,
                    "message": "query returned more than 10000 results"
                }
            })))
            .mount(&server)
            .await;

        let client = EthClient::new(vec![server.uri()], "0x0001")
            .await
            .unwrap();
        let result = client.get_deposit_events_in_range(0, 10000).await;

        assert!(
            matches!(result, Err(BridgeError::TransientProviderError(_))),
            "should return TransientProviderError, got: {:?}",
            result
        );
    }

    #[test]
    fn test_endpoint_rotation() {
        let client = EthClient {
            client: reqwest::Client::new(),
            bridge_contract: "0x0000".to_string(),
            current_endpoint_idx: AtomicU32::new(0),
            rpc_urls: vec![
                "http://rpc1".into(),
                "http://rpc2".into(),
                "http://rpc3".into(),
            ],
            failure_counts: vec![
                AtomicU32::new(0),
                AtomicU32::new(0),
                AtomicU32::new(0),
            ],
        };

        assert_eq!(client.current_url(), "http://rpc1");

        // Rotate
        assert!(!client.rotate_endpoint(3));
        assert_eq!(client.current_url(), "http://rpc2");

        // Reset and check
        client.reset_failure_count();
        assert_eq!(
            client.failure_counts[1].load(Ordering::Relaxed),
            0
        );
    }
}
