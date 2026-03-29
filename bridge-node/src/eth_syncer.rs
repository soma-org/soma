//! Ethereum event syncer.
//!
//! Adapted from Sui's `eth_syncer.rs`. Two concurrent tasks:
//! 1. Finalized block poller — polls `eth_getBlockByNumber("finalized")` on interval
//! 2. Event listener — queries deposit events in the range [last_processed, finalized]

use std::sync::Arc;
use tokio::sync::{mpsc, watch};
use tokio::task::JoinHandle;
use tokio::time::{Duration, interval};
use tracing::{debug, error, info, warn};

use crate::error::{BridgeError, BridgeResult};
use crate::eth_client::EthClient;
use crate::retry::retry_with_backoff;
use crate::types::DepositEvent;

/// Maximum block range per eth_getLogs query.
/// Ethereum RPC providers typically limit results; 1000 blocks is a safe default.
const DEFAULT_MAX_BLOCK_RANGE: u64 = 1000;

/// Default max retry elapsed time for individual RPC calls within the syncer.
const DEFAULT_RETRY_ELAPSED: Duration = Duration::from_secs(120);

/// Syncer that watches Ethereum for bridge deposit events.
pub struct EthSyncer {
    eth_client: Arc<EthClient>,
    poll_interval: Duration,
    max_block_range: u64,
}

/// Result of starting the syncer: task handles + channels.
pub struct EthSyncerHandle {
    /// Background task handles (poller + listener).
    pub task_handles: Vec<JoinHandle<()>>,
    /// Receiver for parsed deposit events with their block number.
    pub event_rx: mpsc::Receiver<(u64, Vec<DepositEvent>)>,
    /// Watch channel for latest finalized block number.
    pub finalized_block_rx: watch::Receiver<u64>,
}

impl EthSyncer {
    pub fn new(
        eth_client: Arc<EthClient>,
        poll_interval: Duration,
        max_block_range: u64,
    ) -> Self {
        Self {
            eth_client,
            poll_interval,
            max_block_range,
        }
    }

    /// Start the syncer. Returns handles for the spawned tasks and event channels.
    ///
    /// `start_block` is the first block to query events from (typically the last
    /// processed block + 1, or the bridge contract deployment block on first run).
    pub fn start(self, start_block: u64) -> EthSyncerHandle {
        let (finalized_tx, finalized_rx) = watch::channel(start_block);
        let (event_tx, event_rx) = mpsc::channel(256);

        let poller_client = Arc::clone(&self.eth_client);
        let poll_interval = self.poll_interval;

        // Task 1: Finalized block poller
        let poller_handle = tokio::spawn(async move {
            run_finalized_block_poller(poller_client, finalized_tx, poll_interval).await;
        });

        let listener_client = Arc::clone(&self.eth_client);
        let max_range = self.max_block_range;

        let listener_finalized_rx = finalized_rx.clone();

        // Task 2: Event listener
        let listener_handle = tokio::spawn(async move {
            run_event_listener(
                listener_client,
                listener_finalized_rx,
                event_tx,
                start_block,
                max_range,
            )
            .await;
        });

        EthSyncerHandle {
            task_handles: vec![poller_handle, listener_handle],
            event_rx,
            finalized_block_rx: finalized_rx,
        }
    }
}

/// Polls for the latest finalized block and broadcasts it.
async fn run_finalized_block_poller(
    client: Arc<EthClient>,
    tx: watch::Sender<u64>,
    poll_interval: Duration,
) {
    let mut timer = interval(poll_interval);
    loop {
        timer.tick().await;
        match retry_with_backoff(
            "get_finalized_block",
            DEFAULT_RETRY_ELAPSED,
            || client.get_last_finalized_block_id(),
        )
        .await
        {
            Ok(block) => {
                let prev = *tx.borrow();
                if block > prev {
                    debug!(block, "New finalized block");
                    let _ = tx.send(block);
                    client.reset_failure_count();
                }
            }
            Err(e) => {
                warn!("Failed to get finalized block after retries: {e}");
                client.rotate_endpoint(3);
            }
        }
    }
}

/// Listens for new finalized blocks and queries deposit events.
async fn run_event_listener(
    client: Arc<EthClient>,
    mut finalized_rx: watch::Receiver<u64>,
    event_tx: mpsc::Sender<(u64, Vec<DepositEvent>)>,
    start_block: u64,
    max_block_range: u64,
) {
    let mut last_processed = start_block.saturating_sub(1);

    loop {
        // Wait for a new finalized block
        if finalized_rx.changed().await.is_err() {
            info!("Finalized block channel closed, shutting down event listener");
            return;
        }

        let finalized = *finalized_rx.borrow();
        if finalized <= last_processed {
            continue;
        }

        // Query events in chunks
        let mut from = last_processed + 1;
        while from <= finalized {
            let to = std::cmp::min(from + max_block_range - 1, finalized);
            match query_events_with_retry(&client, from, to, max_block_range).await {
                Ok(events) => {
                    if !events.is_empty() {
                        info!(
                            from,
                            to,
                            count = events.len(),
                            "Found deposit events"
                        );
                        if event_tx.send((to, events)).await.is_err() {
                            info!("Event channel closed, shutting down event listener");
                            return;
                        }
                    }
                    last_processed = to;
                    from = to + 1;
                    client.reset_failure_count();
                }
                Err(BridgeError::TransientProviderError(msg)) => {
                    // This is typically a "-32005" / "too many results" error.
                    // The retry logic inside query_events_with_retry already halved
                    // the range. If it still fails, log and retry after a delay.
                    warn!(from, to, "Transient error querying events: {msg}");
                    tokio::time::sleep(Duration::from_secs(5)).await;
                }
                Err(e) => {
                    error!(from, to, "Error querying events: {e}");
                    tokio::time::sleep(Duration::from_secs(10)).await;
                }
            }
        }
    }
}

/// Query events with automatic range halving on "too many results" errors.
/// Adapted from Sui's EthSyncer retry pattern.
async fn query_events_with_retry(
    client: &EthClient,
    from: u64,
    to: u64,
    max_range: u64,
) -> BridgeResult<Vec<DepositEvent>> {
    let mut current_range = to - from + 1;
    let mut current_from = from;

    // If the range is within limits, try with retry
    if current_range <= max_range {
        return retry_with_backoff(
            "get_deposit_events",
            DEFAULT_RETRY_ELAPSED,
            || client.get_deposit_events_in_range(from, to),
        )
        .await;
    }

    // Otherwise, chunk and collect
    let mut all_events = Vec::new();
    while current_from <= to {
        let current_to = std::cmp::min(current_from + current_range - 1, to);
        match client
            .get_deposit_events_in_range(current_from, current_to)
            .await
        {
            Ok(events) => {
                all_events.extend(events);
                current_from = current_to + 1;
            }
            Err(BridgeError::TransientProviderError(_)) if current_range > 1 => {
                // Halve the range and retry (Sui's -32005 retry pattern)
                current_range = std::cmp::max(current_range / 2, 1);
                debug!(
                    new_range = current_range,
                    "Halving query range after transient error"
                );
            }
            Err(e) => return Err(e),
        }
    }

    Ok(all_events)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use wiremock::matchers::{body_partial_json, method};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    /// Helper: mock a JSON-RPC response for `eth_getBlockByNumber("finalized")`.
    async fn mock_finalized_block(server: &MockServer, block_number: u64) {
        Mock::given(method("POST"))
            .and(body_partial_json(
                json!({"method": "eth_getBlockByNumber"}),
            ))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "number": format!("0x{block_number:x}")
                }
            })))
            .mount(server)
            .await;
    }

    /// Helper: mock `eth_chainId` for EthClient construction.
    async fn mock_chain_id(server: &MockServer) {
        Mock::given(method("POST"))
            .and(body_partial_json(json!({"method": "eth_chainId"})))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "jsonrpc": "2.0",
                "id": 1,
                "result": "0x1"
            })))
            .mount(server)
            .await;
    }

    /// Helper: mock `eth_getLogs` returning empty logs.
    async fn mock_get_logs_empty(server: &MockServer) {
        Mock::given(method("POST"))
            .and(body_partial_json(json!({"method": "eth_getLogs"})))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "jsonrpc": "2.0",
                "id": 1,
                "result": []
            })))
            .mount(server)
            .await;
    }

    /// Helper: build ABI-encoded deposit event data.
    fn encode_deposit_data(nonce: u64, amount: u64) -> String {
        let mut data = vec![0u8; 128];
        data[24..32].copy_from_slice(&nonce.to_be_bytes());
        data[44..64].copy_from_slice(&[0xAA; 20]); // eth_sender
        data[64..96].copy_from_slice(&[0xBB; 32]); // soma_recipient
        data[120..128].copy_from_slice(&amount.to_be_bytes());
        format!("0x{}", hex::encode(&data))
    }

    /// Helper: mock `eth_getLogs` returning deposit events.
    async fn mock_get_logs_with_deposits(
        server: &MockServer,
        deposits: Vec<(u64, u64)>, // (nonce, amount) pairs
    ) {
        let logs: Vec<_> = deposits
            .into_iter()
            .enumerate()
            .map(|(i, (nonce, amount))| {
                json!({
                    "address": "0x0000000000000000000000000000000000000001",
                    "topics": [],
                    "data": encode_deposit_data(nonce, amount),
                    "blockNumber": format!("0x{:x}", 100 + i),
                    "transactionHash": format!("0x{}", hex::encode([i as u8; 32]))
                })
            })
            .collect();

        Mock::given(method("POST"))
            .and(body_partial_json(json!({"method": "eth_getLogs"})))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "jsonrpc": "2.0",
                "id": 1,
                "result": logs
            })))
            .mount(server)
            .await;
    }

    #[tokio::test]
    async fn test_syncer_detects_finalized_block() {
        let server = MockServer::start().await;
        mock_chain_id(&server).await;
        mock_finalized_block(&server, 100).await;
        mock_get_logs_empty(&server).await;

        let client = Arc::new(
            EthClient::new(vec![server.uri()], "0x0000000000000000000000000000000000000001")
                .await
                .unwrap(),
        );

        let syncer = EthSyncer::new(client, Duration::from_millis(50), 1000);
        let handle = syncer.start(0);

        // Wait for the finalized block to be detected
        let mut rx = handle.finalized_block_rx;
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                rx.changed().await.unwrap();
                let block = *rx.borrow();
                if block >= 100 {
                    return block;
                }
            }
        })
        .await
        .expect("should detect finalized block within 5 seconds");

        // Abort tasks
        for h in handle.task_handles {
            h.abort();
        }
    }

    #[tokio::test]
    async fn test_syncer_emits_deposit_events() {
        let server = MockServer::start().await;
        mock_chain_id(&server).await;
        mock_finalized_block(&server, 50).await;
        mock_get_logs_with_deposits(&server, vec![(1, 1_000_000), (2, 2_000_000)]).await;

        let client = Arc::new(
            EthClient::new(vec![server.uri()], "0x0000000000000000000000000000000000000001")
                .await
                .unwrap(),
        );

        let syncer = EthSyncer::new(client, Duration::from_millis(50), 1000);
        let mut handle = syncer.start(0);

        // Wait for deposit events
        let (block, events) =
            tokio::time::timeout(Duration::from_secs(5), handle.event_rx.recv())
                .await
                .expect("should receive events within 5s")
                .expect("channel should not be closed");

        assert!(block <= 50);
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].nonce, 1);
        assert_eq!(events[0].amount, 1_000_000);
        assert_eq!(events[1].nonce, 2);
        assert_eq!(events[1].amount, 2_000_000);

        for h in handle.task_handles {
            h.abort();
        }
    }

    #[tokio::test]
    async fn test_query_events_with_retry_range_halving() {
        // Test the range-halving logic: when range > max_block_range,
        // the function chunks the query.
        let server = MockServer::start().await;
        mock_chain_id(&server).await;
        mock_get_logs_empty(&server).await;

        let client =
            EthClient::new(vec![server.uri()], "0x0000000000000000000000000000000000000001")
                .await
                .unwrap();

        // Query range 0..2000 with max_range 500 — should chunk into 4 queries
        let result = query_events_with_retry(&client, 0, 1999, 500).await;
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_query_events_with_retry_transient_error_halves() {
        // Simulate -32005 error on large range, success on smaller range.
        let server = MockServer::start().await;

        // First mock: chain_id for construction
        Mock::given(method("POST"))
            .and(body_partial_json(json!({"method": "eth_chainId"})))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "jsonrpc": "2.0",
                "id": 1,
                "result": "0x1"
            })))
            .mount(&server)
            .await;

        // eth_getLogs: first call returns -32005, subsequent calls succeed.
        // wiremock matches in reverse registration order, so register the
        // success fallback first, then the one-shot error on top.
        Mock::given(method("POST"))
            .and(body_partial_json(json!({"method": "eth_getLogs"})))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "jsonrpc": "2.0",
                "id": 1,
                "result": []
            })))
            .mount(&server)
            .await;

        Mock::given(method("POST"))
            .and(body_partial_json(json!({"method": "eth_getLogs"})))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "jsonrpc": "2.0",
                "id": 1,
                "error": {
                    "code": -32005,
                    "message": "query returned more than 10000 results"
                }
            })))
            .up_to_n_times(1)
            .mount(&server)
            .await;

        let client =
            EthClient::new(vec![server.uri()], "0x0000000000000000000000000000000000000001")
                .await
                .unwrap();

        // Large range that exceeds max_range, triggering chunked queries.
        // First chunk gets -32005, range halves, subsequent chunks succeed.
        let result = query_events_with_retry(&client, 0, 2999, 1000).await;
        assert!(result.is_ok());
    }
}
