//! Ethereum RPC client wrapper.
//!
//! Uses raw JSON-RPC via reqwest instead of alloy to avoid version compatibility
//! issues. Only needs two RPC methods: `eth_getBlockByNumber` and `eth_getLogs`.

use futures::stream::StreamExt;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU32, Ordering};
use tracing::{debug, info, warn};

use crate::error::{BridgeError, BridgeResult};
use crate::types::DepositEvent;

/// Ethereum RPC client for bridge operations.
pub struct EthClient {
    client: reqwest::Client,
    bridge_contract: String,
    /// Vestigial — kept so the legacy `rotate_endpoint` /
    /// `reset_failure_count` /\ `current_url` API surface doesn't break
    /// callers during the rotation→quorum migration. The new
    /// quorum-based [`Self::rpc_call`] ignores this entirely.
    current_endpoint_idx: AtomicU32,
    /// All configured RPC endpoints. Calls fan out to every endpoint
    /// in parallel and require quorum on the response (see
    /// [`Self::rpc_call`]) — a single compromised or buggy RPC can no
    /// longer dictate what the bridge thinks is on chain.
    rpc_urls: Vec<String>,
    /// Per-endpoint consecutive failure count. Kept for the auto-pause
    /// watchdog and operator observability; the quorum logic doesn't
    /// gate on it.
    failure_counts: Vec<AtomicU32>,
    /// Strict-majority threshold: a response must match at least this
    /// many endpoints to be accepted. Default `ceil(N/2)+1`. Setting
    /// this to `1` (e.g. for tests or single-provider dev) opts out
    /// of cross-checking.
    quorum_threshold: usize,
    /// Per-call deadline applied across the whole fan-out (not per
    /// endpoint — that's `reqwest::Client`'s own timeout). Once the
    /// deadline elapses, whatever responses arrived are considered
    /// for quorum; outstanding requests are abandoned.
    rpc_deadline: std::time::Duration,
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
///
/// `log_index` is the log's position *within the transaction's receipt*
/// — matches Sui's `log_index_in_tx` convention. This is the routing
/// identifier carried in the bridge URL path
/// (`/sign/bridge_tx/eth/soma/{tx_hash}/{event_idx}`) so peers re-fetch
/// the same log; it's not part of the signed message payload.
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
    /// Position of this log within its tx receipt. Wide-format `u64`
    /// for serde tolerance; cast down to `u16` at use sites (a tx
    /// with >65k logs is implausible).
    #[serde(default, deserialize_with = "deserialize_optional_hex_u64")]
    log_index: Option<u64>,
}

/// Subset of `eth_getTransactionReceipt` response we care about.
/// `logs` carries the event log array; `blockNumber` lets us assert
/// finalization before signing.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct EthTransactionReceipt {
    #[serde(default, deserialize_with = "deserialize_optional_hex_u64")]
    block_number: Option<u64>,
    #[serde(default)]
    logs: Vec<EthLog>,
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

/// Default per-call deadline for the quorum fan-out. Beyond this,
/// whatever responses have arrived are scored for quorum; outstanding
/// requests are abandoned. Long enough to absorb a single slow RPC,
/// short enough that the executor's retry budget isn't dominated by
/// one stalled call.
pub const DEFAULT_RPC_DEADLINE: std::time::Duration = std::time::Duration::from_secs(10);

/// Single endpoint call. Returns `(canonical_json_of_result, deserialized_value)`
/// on success so the caller can both vote on shape and consume the
/// strongly-typed value. Module-private; the public surface is
/// [`EthClient::rpc_call`] which fans this out across endpoints +
/// requires quorum.
async fn single_call<T: serde::de::DeserializeOwned>(
    client: &reqwest::Client,
    url: &str,
    body: Vec<u8>,
) -> BridgeResult<(String, T)> {
    let resp = client
        .post(url)
        .header("content-type", "application/json")
        .body(body)
        .send()
        .await
        .map_err(|e| BridgeError::ProviderError(format!("{url}: {e}")))?;

    // Read the body as raw `Value` first so we can hash a canonical
    // form. Going straight to `T` loses information about how other
    // providers represented the same logical value.
    let bytes = resp
        .bytes()
        .await
        .map_err(|e| BridgeError::ProviderError(format!("{url}: {e}")))?;
    let raw: JsonRpcResponse<serde_json::Value> = serde_json::from_slice(&bytes)
        .map_err(|e| BridgeError::ProviderError(format!("{url}: bad json: {e}")))?;

    if let Some(err) = raw.error {
        let msg = format!("RPC error {}: {}", err.code, err.message);
        if err.code == -32005
            || err.message.contains("query returned more than")
            || err.message.contains("Log response size exceeded")
        {
            return Err(BridgeError::TransientProviderError(msg));
        }
        return Err(BridgeError::ProviderError(msg));
    }

    let value = raw
        .result
        .ok_or_else(|| BridgeError::ProviderError(format!("{url}: null result")))?;
    let canonical = serde_json::to_string(&value)
        .map_err(|e| BridgeError::Internal(format!("re-encode: {e}")))?;
    let typed: T = serde_json::from_value(value).map_err(|e| {
        BridgeError::ProviderError(format!("{url}: response type mismatch: {e}"))
    })?;
    Ok((canonical, typed))
}

/// Default strict-majority quorum: `ceil(N/2) + 1`. For N=1 returns 1
/// (single-provider deployments don't gain cross-checking but don't
/// error either); for N=3 returns 2; for N=5 returns 3.
fn default_quorum(n: usize) -> usize {
    if n <= 1 { 1 } else { n / 2 + 1 }
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

        let n = rpc_urls.len();
        let failure_counts = rpc_urls.iter().map(|_| AtomicU32::new(0)).collect();

        let client = Self {
            client: reqwest::Client::new(),
            bridge_contract: bridge_contract_address.to_lowercase(),
            current_endpoint_idx: AtomicU32::new(0),
            rpc_urls,
            failure_counts,
            quorum_threshold: default_quorum(n),
            rpc_deadline: DEFAULT_RPC_DEADLINE,
        };

        // Verify connectivity. Note: a misconfigured cluster where the
        // configured RPC providers don't agree on `eth_chainId` will
        // fail here — that's the right behavior, the bridge would have
        // produced split-brain answers otherwise.
        let chain_id = client.get_chain_id().await?;
        info!(
            chain_id,
            providers = client.rpc_urls.len(),
            quorum = client.quorum_threshold,
            "EthClient connected to Ethereum"
        );

        Ok(client)
    }

    /// Override the quorum threshold post-construction. Mostly for
    /// tests; in production the `ceil(N/2)+1` default is what you
    /// want. Set to `1` to accept a single endpoint's response (i.e.
    /// disable cross-checking; e.g. for a single-provider devnet).
    pub fn with_quorum_threshold(mut self, threshold: usize) -> Self {
        self.quorum_threshold = threshold.max(1).min(self.rpc_urls.len().max(1));
        self
    }

    /// Test-only constructor that skips the connectivity probe. Used
    /// by unit tests and the multi-peer integration test in
    /// `bridge-node/tests/` (which is a separate crate — `cfg(test)`
    /// is not set when it compiles against the bridge-node lib, so
    /// this must be `pub`, not `#[cfg(test)] pub`). The
    /// `_for_test` suffix flags that production code should not use
    /// this; a forthcoming `test-utils` feature gate is the cleaner
    /// long-term home.
    pub fn new_for_test(bridge_contract_address: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            bridge_contract: bridge_contract_address.to_lowercase(),
            current_endpoint_idx: AtomicU32::new(0),
            rpc_urls: vec!["http://127.0.0.1:0".to_string()],
            failure_counts: vec![AtomicU32::new(0)],
            quorum_threshold: 1,
            rpc_deadline: DEFAULT_RPC_DEADLINE,
        }
    }

    /// Test-only constructor that takes explicit endpoint URLs +
    /// quorum threshold without doing a connectivity probe. Used by
    /// the quorum unit tests below.
    pub fn new_for_test_with_endpoints(
        bridge_contract_address: String,
        rpc_urls: Vec<String>,
        quorum_threshold: usize,
    ) -> Self {
        let failure_counts = rpc_urls.iter().map(|_| AtomicU32::new(0)).collect();
        Self {
            client: reqwest::Client::new(),
            bridge_contract: bridge_contract_address.to_lowercase(),
            current_endpoint_idx: AtomicU32::new(0),
            rpc_urls,
            failure_counts,
            quorum_threshold,
            rpc_deadline: std::time::Duration::from_secs(2),
        }
    }

    fn current_url(&self) -> &str {
        let idx = self.current_endpoint_idx.load(Ordering::Relaxed) as usize;
        &self.rpc_urls[idx]
    }

    /// Issue a JSON-RPC call to **every** configured endpoint in
    /// parallel and accept the response only when at least
    /// `quorum_threshold` endpoints agree byte-for-byte on the result.
    ///
    /// Comparison is on the canonical JSON form of the `result` field
    /// (not the deserialized `T` — most RPCs return primitive types
    /// where `serde_json::Value`-equality is the right notion of
    /// agreement; if `T` deserialization succeeds for one provider but
    /// fails for another, the differing form is treated as
    /// disagreement, which is the conservative read).
    ///
    /// ## Error semantics
    /// - Quorum reached on `Ok`: deserialize that value and return.
    /// - Quorum reached on `Err` (same error message): propagate as
    ///   `ProviderError` / `TransientProviderError`. This catches the
    ///   "everyone agrees the tx isn't finalized" steady-state for the
    ///   `TxNotFinalized` path.
    /// - No quorum (split decision or too many failures): error with a
    ///   tally summary — operators want to see which endpoints
    ///   disagreed.
    async fn rpc_call<T: serde::de::DeserializeOwned>(
        &self,
        method: &str,
        params: serde_json::Value,
    ) -> BridgeResult<T> {
        use std::collections::HashMap;

        let req = JsonRpcRequest {
            jsonrpc: "2.0",
            method,
            params,
            id: 1,
        };
        let body = serde_json::to_vec(&req)
            .map_err(|e| BridgeError::Internal(format!("encode JSON-RPC: {e}")))?;

        // Fire off one request per endpoint. `FuturesUnordered` so we
        // observe completions in arrival order — we can short-circuit
        // the moment quorum is met without waiting for stragglers.
        let mut futures = futures::stream::FuturesUnordered::new();
        for (idx, url) in self.rpc_urls.iter().enumerate() {
            let client = self.client.clone();
            let url = url.clone();
            let body = body.clone();
            futures.push(async move {
                let res = single_call::<T>(&client, &url, body).await;
                (idx, res)
            });
        }

        // Group responses by their canonical form so we can count
        // votes. `Ok` results are keyed by the JSON-serialized form of
        // the original `result`; `Err` results are keyed by the error
        // variant tag + message.
        let mut ok_tally: HashMap<String, (T, usize)> = HashMap::new();
        let mut err_tally: HashMap<String, (BridgeError, usize)> = HashMap::new();
        let mut total: usize = 0;

        let deadline = tokio::time::sleep(self.rpc_deadline);
        tokio::pin!(deadline);
        loop {
            tokio::select! {
                biased;
                _ = &mut deadline => {
                    warn!(method, deadline_ms = self.rpc_deadline.as_millis() as u64,
                          "rpc_call deadline elapsed before quorum");
                    break;
                }
                next = futures.next() => {
                    match next {
                        Some((idx, Ok((canonical, value)))) => {
                            total += 1;
                            self.failure_counts[idx].store(0, Ordering::Relaxed);
                            let entry = ok_tally.entry(canonical).or_insert((value, 0));
                            entry.1 += 1;
                            if entry.1 >= self.quorum_threshold {
                                // Move the value out — drop the tally,
                                // discard outstanding requests.
                                let (winning_key, _) = ok_tally
                                    .iter()
                                    .max_by_key(|(_, (_, c))| *c)
                                    .map(|(k, _)| (k.clone(), 0))
                                    .unwrap();
                                let (value, _) = ok_tally.remove(&winning_key).unwrap();
                                return Ok(value);
                            }
                        }
                        Some((idx, Err(err))) => {
                            total += 1;
                            self.failure_counts[idx].fetch_add(1, Ordering::Relaxed);
                            let key = format!("{err}");
                            let entry = err_tally.entry(key).or_insert((err, 0));
                            entry.1 += 1;
                            if entry.1 >= self.quorum_threshold {
                                let winning_key = err_tally
                                    .iter()
                                    .max_by_key(|(_, (_, c))| *c)
                                    .map(|(k, _)| k.clone())
                                    .unwrap();
                                let (err, _) = err_tally.remove(&winning_key).unwrap();
                                return Err(err);
                            }
                        }
                        None => break,
                    }
                }
            }
        }

        // No quorum: summarize what we saw, surface as ProviderError.
        let ok_vote_str = ok_tally
            .iter()
            .map(|(_, (_, c))| c.to_string())
            .collect::<Vec<_>>()
            .join("/");
        let err_vote_str = err_tally
            .iter()
            .map(|(_, (_, c))| c.to_string())
            .collect::<Vec<_>>()
            .join("/");
        Err(BridgeError::ProviderError(format!(
            "no quorum on {method}: {} of {} providers responded ({} ok-groups: [{ok_vote_str}], {} err-groups: [{err_vote_str}]); need {}",
            total,
            self.rpc_urls.len(),
            ok_tally.len(),
            err_tally.len(),
            self.quorum_threshold,
        )))
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

    /// Read the USDC balance of an arbitrary address from the configured
    /// USDC contract. Used by the watchdog to read the bridge contract's
    /// locked-USDC balance for the conservation invariant. The selector
    /// `balanceOf(address)` is `0x70a08231`.
    ///
    /// L8: queried at the **`finalized`** block tag rather than `latest`,
    /// so reorgs of the latest few blocks can't trigger a transient false
    /// "soma supply > eth locked" reading and false-pause the bridge.
    pub async fn get_erc20_balance(
        &self,
        token_contract: &str,
        holder: &str,
    ) -> BridgeResult<u128> {
        // ABI-encoded call: selector(4) || padded address(32) = 36 bytes hex.
        let holder_no_prefix = holder.strip_prefix("0x").unwrap_or(holder);
        if holder_no_prefix.len() != 40 {
            return Err(BridgeError::Internal(
                "ERC20 balanceOf: holder address must be 20 bytes".into(),
            ));
        }
        // selector + 12 zero bytes + 20-byte address.
        let calldata = format!("0x70a08231000000000000000000000000{}", holder_no_prefix);
        let result: String = self
            .rpc_call(
                "eth_call",
                serde_json::json!([
                    {
                        "to": token_contract,
                        "data": calldata,
                    },
                    "finalized"
                ]),
            )
            .await?;
        let s = result.strip_prefix("0x").unwrap_or(&result);
        // The return is a uint256 — parse as u128 (USDC fits in u128 trivially;
        // total supply is ~50e9 * 10^6 = 5e16, well under 2^128).
        // If the high 128 bits are non-zero we error out rather than truncate.
        let bytes = hex::decode(s)
            .map_err(|e| BridgeError::ProviderError(format!("bad balanceOf hex: {e}")))?;
        if bytes.len() != 32 {
            return Err(BridgeError::ProviderError(format!(
                "balanceOf return must be 32 bytes, got {}",
                bytes.len()
            )));
        }
        if bytes[..16].iter().any(|&b| b != 0) {
            return Err(BridgeError::ProviderError(
                "ERC20 balance exceeds u128 (impossible for USDC; check token contract)"
                    .into(),
            ));
        }
        let mut buf = [0u8; 16];
        buf.copy_from_slice(&bytes[16..]);
        Ok(u128::from_be_bytes(buf))
    }

    /// Call `paused()` on the Eth bridge contract. Returns `true` when
    /// the OpenZeppelin `PausableUpgradeable` flag is set (the contract
    /// is currently rejecting deposits + withdrawals).
    ///
    /// Used by the watchdog's `EthBridgeStatus` observable to detect
    /// committee-signed pause/unpause actions that landed on Eth but
    /// not Soma (or vice versa).
    pub async fn get_bridge_paused(&self, bridge_contract: &str) -> BridgeResult<bool> {
        // selector for `paused()` is keccak256("paused()")[..4] = 0x5c975abb.
        let calldata = "0x5c975abb";
        let result: String = self
            .rpc_call(
                "eth_call",
                serde_json::json!([
                    {
                        "to": bridge_contract,
                        "data": calldata,
                    },
                    "finalized"
                ]),
            )
            .await?;
        let s = result.strip_prefix("0x").unwrap_or(&result);
        let bytes = hex::decode(s)
            .map_err(|e| BridgeError::ProviderError(format!("bad paused() hex: {e}")))?;
        if bytes.len() != 32 {
            return Err(BridgeError::ProviderError(format!(
                "paused() return must be 32 bytes, got {}",
                bytes.len()
            )));
        }
        // bool occupies the right-aligned last byte; everything else
        // must be zero. Reject non-canonical encodings rather than
        // accept a maybe-malformed reply.
        if bytes[..31].iter().any(|&b| b != 0) || bytes[31] > 1 {
            return Err(BridgeError::ProviderError(
                "paused() return is not a canonical ABI-encoded bool".into(),
            ));
        }
        Ok(bytes[31] == 1)
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

    /// Fetch the `BridgeAction` corresponding to an Eth tx + event index,
    /// asserting the tx is in a finalized block. Mirrors Sui's
    /// `EthClient::get_finalized_bridge_action_maybe`. This is the
    /// **fetch-and-sign primitive** used by the bridge HTTP server: a
    /// peer asks `GET /sign/bridge_tx/eth/soma/{tx_hash}/{event_idx}`,
    /// the server queries Eth itself via this function to verify the
    /// deposit actually happened on a finalized block, and only then
    /// produces a signature. No peer can poison the server with a sig
    /// for a non-existent or reorged deposit.
    pub async fn get_finalized_bridge_action_maybe(
        &self,
        tx_hash: [u8; 32],
        event_idx: u16,
    ) -> BridgeResult<crate::types::BridgeAction> {
        // 1. Pin the finalized head first so a receipt observed before
        //    finalization can't slip through.
        let last_finalized = self.get_last_finalized_block_id().await?;

        // 2. Fetch the receipt. JSON-RPC returns `null` if the tx isn't found.
        let tx_hash_hex = format!("0x{}", hex::encode(tx_hash));
        let receipt: Option<EthTransactionReceipt> = self
            .rpc_call(
                "eth_getTransactionReceipt",
                serde_json::json!([&tx_hash_hex]),
            )
            .await?;
        let receipt = receipt.ok_or(BridgeError::Internal(format!(
            "eth tx not found: {tx_hash_hex}"
        )))?;

        // 3. Finalization check.
        let receipt_block = receipt.block_number.ok_or_else(|| {
            BridgeError::Internal("receipt missing blockNumber".into())
        })?;
        if receipt_block > last_finalized {
            return Err(BridgeError::Internal(format!(
                "tx not finalized: receipt block {receipt_block} > finalized {last_finalized}"
            )));
        }

        // 4. Pull the log at the claimed event index.
        let log = receipt.logs.get(event_idx as usize).ok_or_else(|| {
            BridgeError::Internal(format!(
                "no log at event_idx {event_idx} (receipt has {} logs)",
                receipt.logs.len()
            ))
        })?;

        // 5. Reject logs emitted from contracts other than the configured
        //    bridge contract. Mirrors Sui's `BridgeEventInUnrecognizedEthContract`.
        if log.address.to_lowercase() != self.bridge_contract {
            return Err(BridgeError::Internal(format!(
                "log emitted from unrecognized contract: {} (expected {})",
                log.address, self.bridge_contract
            )));
        }

        // 6. Parse the log as a deposit. Returns None for logs we can't
        //    decode (e.g. emitted from the same contract but a different
        //    event signature).
        let deposit = self.parse_deposit_log(log)?.ok_or_else(|| {
            BridgeError::Internal(
                "log at event_idx is not a recognized bridge event".into(),
            )
        })?;

        // 7. Force the tx_hash from the receipt-derived deposit to equal
        //    the caller's `tx_hash` — the parser pulls it from the log,
        //    but on a receipt-fetch path that should always be the same.
        //    Belt-and-suspenders.
        if deposit.tx_hash != tx_hash {
            return Err(BridgeError::Internal(format!(
                "tx_hash mismatch between caller and receipt: {:?} vs {:?}",
                tx_hash, deposit.tx_hash,
            )));
        }
        // 8. Overwrite event_idx with the caller's value. The receipt
        //    response sometimes omits `logIndex` per-log (RPC providers
        //    vary), so prefer the caller's index — which is what
        //    selected this log out of the receipt in step (6) above.
        let mut deposit = deposit;
        deposit.event_idx = event_idx;

        Ok(deposit.to_bridge_action())
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

        // V2 Soma `TokensDeposited` event carries 7 ABI words (224 bytes):
        //   word 0  bytes 0..32   : uint64 nonce               (right-aligned)
        //   word 1  bytes 32..64  : address sender             (right-aligned, 20)
        //   word 2  bytes 64..96  : uint8 destinationChainID   (right-aligned)
        //   word 3  bytes 96..128 : bytes32 somaRecipient
        //   word 4  bytes 128..160: uint8 tokenType            (right-aligned)
        //   word 5  bytes 160..192: uint64 amount              (right-aligned)
        //   word 6  bytes 192..224: uint64 timestampMs         (right-aligned)
        //
        // Order matches `ISomaBridge.TokensDeposited` in `bridge/evm/`
        // repo — see the inline event doc on the Solidity side. Field
        // adds vs. the pre-Sui-parity layout: `destinationChainID`,
        // `tokenType` (Sui parity).
        if data.len() < 224 {
            warn!(
                data_len = data.len(),
                "Deposit event data too short (expected 224 bytes), skipping"
            );
            return Ok(None);
        }

        let nonce = u64::from_be_bytes(data[24..32].try_into().unwrap());
        let mut eth_sender = [0u8; 20];
        eth_sender.copy_from_slice(&data[44..64]);
        let destination_chain_byte = data[95];
        let destination_chain_id = types::bridge::BridgeChainId::from_u8(destination_chain_byte)
            .ok_or_else(|| {
                BridgeError::Internal(format!(
                    "Deposit event destinationChainID {destination_chain_byte} is not a known BridgeChainId",
                ))
            })?;
        let mut soma_recipient = [0u8; 32];
        soma_recipient.copy_from_slice(&data[96..128]);
        let token_type = data[159];
        let amount = u64::from_be_bytes(data[184..192].try_into().unwrap());
        let timestamp_ms = u64::from_be_bytes(data[216..224].try_into().unwrap());

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
        // u64 → u16: tx receipts with >65k logs do not exist in
        // practice (block-level gas cap is far below that). Saturating
        // cast preserves the property "event_idx fits in the URL path".
        let event_idx = log.log_index.unwrap_or(0).min(u16::MAX as u64) as u16;

        Ok(Some(DepositEvent {
            nonce,
            eth_sender,
            destination_chain_id,
            soma_recipient,
            token_type,
            amount,
            tx_hash,
            event_idx,
            block_number,
            timestamp_ms,
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
            quorum_threshold: 1,
            rpc_deadline: std::time::Duration::from_secs(2),
        };

        // Construct ABI-encoded event data — 7 words, 224 bytes total.
        // See `parse_deposit_log`'s field-layout comment for the offset map.
        let mut data = vec![0u8; 224];
        // word 0: nonce (right-aligned u64 in 32-byte slot)
        data[24..32].copy_from_slice(&42u64.to_be_bytes());
        // word 1: sender address (right-aligned 20 in 32)
        data[44..64].copy_from_slice(&[0xAA; 20]);
        // word 2: destinationChainID (right-aligned u8) — SomaCustom = 2
        data[95] = types::bridge::BridgeChainId::SomaCustom.as_u8();
        // word 3: somaRecipient (full 32-byte slot)
        data[96..128].copy_from_slice(&[0xBB; 32]);
        // word 4: tokenType (right-aligned u8) — USDC = 3
        data[159] = types::bridge::USDC_TOKEN_TYPE;
        // word 5: amount (right-aligned u64)
        data[184..192].copy_from_slice(&1_000_000u64.to_be_bytes());
        // word 6: timestampMs (right-aligned u64)
        data[216..224].copy_from_slice(&1_700_000_000_000u64.to_be_bytes());

        let log = EthLog {
            address: "0xabcdef1234567890abcdef1234567890abcdef12".to_string(),
            topics: vec![],
            data: format!("0x{}", hex::encode(&data)),
            block_number: Some(12345),
            transaction_hash: Some(format!("0x{}", hex::encode([0xCC; 32]))),
            log_index: Some(3),
        };

        let event = client.parse_deposit_log(&log).unwrap().unwrap();
        assert_eq!(event.nonce, 42);
        assert_eq!(event.eth_sender, [0xAA; 20]);
        assert_eq!(
            event.destination_chain_id,
            types::bridge::BridgeChainId::SomaCustom
        );
        assert_eq!(event.soma_recipient, [0xBB; 32]);
        assert_eq!(event.token_type, types::bridge::USDC_TOKEN_TYPE);
        assert_eq!(event.amount, 1_000_000);
        assert_eq!(event.timestamp_ms, 1_700_000_000_000);
        assert_eq!(event.block_number, 12345);
        assert_eq!(event.tx_hash, [0xCC; 32]);
        assert_eq!(event.event_idx, 3);
    }

    #[test]
    fn test_wrong_contract_address_returns_none() {
        let client = EthClient {
            client: reqwest::Client::new(),
            bridge_contract: "0x1111111111111111111111111111111111111111".to_string(),
            current_endpoint_idx: AtomicU32::new(0),
            rpc_urls: vec!["http://localhost:8545".into()],
            failure_counts: vec![AtomicU32::new(0)],
            quorum_threshold: 1,
            rpc_deadline: std::time::Duration::from_secs(2),
        };

        let log = EthLog {
            address: "0x2222222222222222222222222222222222222222".to_string(),
            topics: vec![],
            data: format!("0x{}", hex::encode([0u8; 128])),
            block_number: Some(1),
            transaction_hash: None,
            log_index: None,
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
            quorum_threshold: 1,
            rpc_deadline: std::time::Duration::from_secs(2),
        };

        let log = EthLog {
            address: "0xabcd".to_string(),
            topics: vec![],
            data: "0x0000".to_string(),
            block_number: None,
            transaction_hash: None,
            log_index: None,
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

        // Build deposit event data — 7 ABI words (224 bytes) for the
        // Sui-parity `TokensDeposited` layout. See `parse_deposit_log`.
        let mut data = vec![0u8; 224];
        data[24..32].copy_from_slice(&7u64.to_be_bytes()); // nonce=7
        data[44..64].copy_from_slice(&[0xAA; 20]); // sender
        data[95] = types::bridge::BridgeChainId::SomaCustom.as_u8(); // destChain
        data[96..128].copy_from_slice(&[0xBB; 32]); // somaRecipient
        data[159] = types::bridge::USDC_TOKEN_TYPE; // tokenType
        data[184..192].copy_from_slice(&5_000_000u64.to_be_bytes()); // amount
        // word 6 (timestampMs) left zero for this test fixture.

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
            quorum_threshold: 2,
            rpc_deadline: std::time::Duration::from_secs(2),
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

    // -----------------------------------------------------------------------
    // Multi-provider quorum tests. Each test spins up 3 wiremock servers
    // that act as independent JSON-RPC providers, points an EthClient at
    // all three with quorum_threshold=2, and verifies the agree/disagree
    // behavior. Catches a single rogue provider lying about chain state.
    // -----------------------------------------------------------------------

    use wiremock::matchers::{body_json, method};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    /// Mount a JSON-RPC `eth_chainId` response on `server`. Each provider
    /// returns its own `chain_id_hex` so we can test agree/disagree.
    async fn mount_chain_id(server: &MockServer, chain_id_hex: &str) {
        Mock::given(method("POST"))
            .and(body_json(serde_json::json!({
                "jsonrpc": "2.0",
                "method": "eth_chainId",
                "params": [],
                "id": 1,
            })))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "application/json")
                    .set_body_json(serde_json::json!({
                        "jsonrpc": "2.0",
                        "id": 1,
                        "result": chain_id_hex,
                    })),
            )
            .mount(server)
            .await;
    }

    /// 3/3 agreement: quorum trivially met; client returns the agreed value.
    #[tokio::test]
    async fn quorum_three_of_three_agree() {
        let s1 = MockServer::start().await;
        let s2 = MockServer::start().await;
        let s3 = MockServer::start().await;
        mount_chain_id(&s1, "0x1").await;
        mount_chain_id(&s2, "0x1").await;
        mount_chain_id(&s3, "0x1").await;

        let client = EthClient::new_for_test_with_endpoints(
            "0x0".to_string(),
            vec![s1.uri(), s2.uri(), s3.uri()],
            2,
        );
        let id = client.get_chain_id().await.unwrap();
        assert_eq!(id, 1);
    }

    /// 2/3 agree on `0x1`, 1 lies about `0x2`: the majority wins, the
    /// honest providers carry the day.
    #[tokio::test]
    async fn quorum_majority_overrules_one_dissenter() {
        let s1 = MockServer::start().await;
        let s2 = MockServer::start().await;
        let s3 = MockServer::start().await;
        mount_chain_id(&s1, "0x1").await;
        mount_chain_id(&s2, "0x1").await;
        mount_chain_id(&s3, "0x99").await; // rogue / compromised RPC

        let client = EthClient::new_for_test_with_endpoints(
            "0x0".to_string(),
            vec![s1.uri(), s2.uri(), s3.uri()],
            2,
        );
        // Should return 1, not 0x99 — quorum rejects the rogue.
        let id = client.get_chain_id().await.unwrap();
        assert_eq!(id, 1);
    }

    /// All 3 providers disagree (1/1/1): no quorum, must error.
    /// Without this guard, the bridge would silently pick whichever
    /// response happened to arrive first — undefined trust behavior.
    #[tokio::test]
    async fn quorum_three_way_split_errors() {
        let s1 = MockServer::start().await;
        let s2 = MockServer::start().await;
        let s3 = MockServer::start().await;
        mount_chain_id(&s1, "0x1").await;
        mount_chain_id(&s2, "0x2").await;
        mount_chain_id(&s3, "0x3").await;

        let client = EthClient::new_for_test_with_endpoints(
            "0x0".to_string(),
            vec![s1.uri(), s2.uri(), s3.uri()],
            2,
        );
        let err = client.get_chain_id().await.unwrap_err();
        match err {
            BridgeError::ProviderError(msg) => {
                assert!(msg.contains("no quorum"), "{msg}");
            }
            other => panic!("expected no-quorum ProviderError, got {other:?}"),
        }
    }

    /// 1 provider succeeds; 2 are down. Quorum_threshold=2 means the
    /// single happy response can't carry the call. Either the two-down
    /// providers form an Err-tally quorum (and we return their error,
    /// fine — executor retries) OR no quorum at all (also an error).
    /// What MUST NOT happen: a successful return based on the single
    /// honest provider's response — that would silently downgrade
    /// security to single-provider trust.
    #[tokio::test]
    async fn quorum_one_alone_with_two_down_does_not_silently_succeed() {
        let s1 = MockServer::start().await;
        mount_chain_id(&s1, "0x1").await;
        // 127.0.0.1:1 is conventionally refused, used as "endpoint down".
        let client = EthClient::new_for_test_with_endpoints(
            "0x0".to_string(),
            vec![
                s1.uri(),
                "http://127.0.0.1:1".to_string(),
                "http://127.0.0.1:1".to_string(),
            ],
            2,
        );
        let err = client
            .get_chain_id()
            .await
            .expect_err("must not succeed on a single honest provider");
        // Either error variant is acceptable — both correctly signal
        // "can't be trusted; retry".
        match err {
            BridgeError::ProviderError(_) | BridgeError::TransientProviderError(_) => {}
            other => panic!("expected ProviderError, got {other:?}"),
        }
    }

    /// Single-provider deployment with quorum=1: the only-honest-RPC
    /// case must still work. (Test catches regressions where the
    /// quorum logic incorrectly demands >=2 for any chain_id call.)
    #[tokio::test]
    async fn quorum_one_provider_with_threshold_one() {
        let s1 = MockServer::start().await;
        mount_chain_id(&s1, "0x42").await;
        let client = EthClient::new_for_test_with_endpoints(
            "0x0".to_string(),
            vec![s1.uri()],
            1,
        );
        let id = client.get_chain_id().await.unwrap();
        assert_eq!(id, 0x42);
    }
}
