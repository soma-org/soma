use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;
use types::base::SomaAddress;

use crate::types::BridgeAction;

/// Watchdog block on `BridgeNodeConfig`. Opt-in (`watchdog = None` means
/// no watchdog spawned); when present, the bridge node spawns the
/// conservation-invariant watchdog at startup. See [`crate::watchdog`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WatchdogConfigBlock {
    /// Eth-side USDC ERC20 contract address. The watchdog reads the
    /// SomaBridge contract's USDC balance via `balanceOf` against
    /// this token.
    pub usdc_contract_address: String,
    /// Eth-side SomaBridge proxy address — the "locked pool" balance
    /// the watchdog reads. Often same value as `bridge_contract_address`
    /// at the top level, but specified here so a misconfigured node
    /// can't accidentally read its bridge's balance against the
    /// wrong contract.
    pub eth_bridge_contract_address: String,
    /// Poll cadence in milliseconds.
    #[serde(default = "default_watchdog_poll_ms")]
    pub poll_interval_ms: u64,
    /// Consecutive violation polls before auto-pause fires. Default 6
    /// at 5s cadence = 30s of sustained violation.
    #[serde(default = "default_watchdog_failure_threshold")]
    pub failure_threshold: u32,
    /// USDC-micro tolerance for in-flight transfers. Production sets
    /// to cover realistic burst volume.
    ///
    /// Note: kept as `u64` rather than `u128` because the `toml` crate
    /// does not support `u128` deserialization. `u64::MAX` USDC-micro
    /// is ~1.8e13 USDC, well above any plausible bridge supply, so the
    /// narrower type is harmless here. The comparison in
    /// `ConservationInvariantObservable::check_once` widens to `u128`.
    #[serde(default = "default_watchdog_tolerance_micro")]
    pub in_flight_tolerance_micro: u64,
}

impl WatchdogConfigBlock {
    pub fn poll_interval(&self) -> Duration {
        Duration::from_millis(self.poll_interval_ms)
    }
}

fn default_watchdog_poll_ms() -> u64 {
    5_000
}
fn default_watchdog_failure_threshold() -> u32 {
    6
}
fn default_watchdog_tolerance_micro() -> u64 {
    // 1 USDC = 1_000_000 micro. Default to 1k USDC of in-flight slack.
    1_000_000 * 1_000
}

/// Configuration for a bridge node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeNodeConfig {
    /// Path to the ECDSA bridge key file (Secp256k1 private key).
    pub bridge_key_path: PathBuf,

    /// Ethereum RPC endpoints (multiple for fallback).
    /// The node rotates through these on failure.
    pub eth_rpc_urls: Vec<String>,

    /// Address of the Soma bridge contract on Ethereum.
    pub bridge_contract_address: String,

    /// Soma fullnode RPC URL for checkpoint subscription.
    pub soma_rpc_url: String,

    /// Deprecated. Sig exchange happens over HTTP via
    /// `http_listen_address` (the gRPC bridge server was retired when
    /// the fetch-and-sign HTTP path landed). Kept here so existing
    /// config files don't fail to parse — drop it at your leisure.
    #[serde(default = "default_deprecated_grpc_address")]
    pub grpc_listen_address: SocketAddr,

    /// Address to listen on for the HTTP REST sig-exchange surface
    /// (Sui parity). Peers fetch signatures by GETing routes off this
    /// endpoint. See [`crate::http_server`] for the route table.
    /// Defaults to the gRPC port + 1 so single-validator dev clusters
    /// don't clash.
    #[serde(default = "default_http_listen_address")]
    pub http_listen_address: SocketAddr,

    /// Operator's pre-approved governance whitelist. The HTTP server's
    /// `GovernanceVerifier` rejects sig requests for any governance
    /// action not byte-identical to one in this list. Token transfers
    /// are server-verified via on-chain state and do NOT belong here.
    /// Empty by default — most deployments don't need governance
    /// sigs in steady state.
    #[serde(default)]
    pub approved_governance_actions: Vec<BridgeAction>,

    /// Addresses of peer bridge nodes for signature exchange.
    /// Discovered from validator set, but can be overridden.
    #[serde(default)]
    pub peer_addresses: Vec<String>,

    /// Ethereum chain ID (1 for mainnet, 11155111 for Sepolia).
    #[serde(default = "default_eth_chain_id")]
    pub eth_chain_id: u64,

    /// Finalized block poll interval in milliseconds.
    #[serde(default = "default_poll_interval_ms")]
    pub eth_poll_interval_ms: u64,

    /// Maximum block range per eth_getLogs query.
    #[serde(default = "default_max_log_query_range")]
    pub max_log_query_range: u64,

    /// Number of consecutive RPC failures before triggering auto-pause.
    #[serde(default = "default_auto_pause_threshold")]
    pub auto_pause_failure_threshold: u32,

    /// Maximum retry elapsed time in seconds for transient errors.
    #[serde(default = "default_max_retry_elapsed_secs")]
    pub max_retry_elapsed_secs: u64,

    /// Path to the bridge node's RocksDB WAL directory. Stores pending
    /// actions + per-contract Eth cursor + Soma checkpoint cursor so a
    /// restart picks up exactly where the previous run left off, instead
    /// of re-scanning from genesis. Mirrors Sui's `db_path/client`.
    #[serde(default = "default_wal_path")]
    pub wal_path: PathBuf,

    /// Block height to start scanning Ethereum from on first run (when
    /// the WAL has no recorded cursor for the bridge contract). Pick a
    /// value at-or-after the bridge contract's deployment block;
    /// otherwise the first scan downloads useless logs back to genesis.
    #[serde(default)]
    pub eth_start_block_fallback: u64,

    /// Optional watchdog config. When `Some`, the bridge node spawns
    /// the conservation-invariant watchdog (and auto-pause emitter)
    /// at startup. `None` = no watchdog (the default; opt-in so dev
    /// clusters don't need an Eth USDC contract address).
    #[serde(default)]
    pub watchdog: Option<WatchdogConfigBlock>,

    /// Optional Eth-side outbound relayer. When `Some`, the bridge
    /// node polls Soma for cert-attached `PendingWithdrawal` objects
    /// and submits the release tx to Ethereum. Disabled by default
    /// because it needs an Eth wallet keypair + the contract ABI
    /// integration (the contracts themselves live in-tree at
    /// [`../../bridge/evm/`](../../bridge/evm/), mirroring Sui's
    /// `bridge/evm/` layout). See [`crate::outbound_relayer`].
    #[serde(default)]
    pub outbound_relayer: Option<OutboundRelayerConfigBlock>,

    /// Optional Soma-side relayer (cert-submission wallet). When
    /// `Some`, the bridge node assembles the quorum-signed cert for
    /// each observed deposit / withdrawal and submits the resulting
    /// Soma transaction from this wallet — flipping the node out of
    /// sig-cache-only mode into full end-to-end participation. `None`
    /// keeps the legacy behavior: observe + sign, but rely on an
    /// external process to assemble + submit the cert.
    #[serde(default)]
    pub soma_relayer: Option<SomaRelayerConfigBlock>,
}

/// Soma-side relayer (wired via `BridgeNode::with_relayer`). The
/// wallet pays gas for the inbound-cert submission tx; in practice
/// a single relayer key is shared across all bridge nodes — the
/// first node to assemble the cert wins the submission race, and
/// duplicate submissions are rejected as replay on-chain.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SomaRelayerConfigBlock {
    /// Soma address derived from the relayer keypair. Surfaced
    /// separately so operators can validate the on-chain balance
    /// without decoding the keypair file.
    pub address: SomaAddress,
    /// Path to the relayer's SomaKeyPair file (base64-encoded
    /// flag-prefixed bytes, same format as `soma keytool generate`
    /// emits). Loaded at startup; the contents are zeroized when
    /// the node shuts down.
    pub key_path: PathBuf,
}

/// Outbound-relayer config block. Triggers spinning up an
/// `EthSubmitter` at bridge-node startup and connecting it to the
/// outbound polling loop. Requires an Eth wallet key the operator
/// can fund + the deployed `SomaBridge` contract address.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutboundRelayerConfigBlock {
    /// Deployed `SomaBridge` proxy address on Ethereum (the contract
    /// that owns the vault + receives release calls). Required.
    pub bridge_contract_address: String,
    /// Operator wallet private key (32 bytes hex, with or without
    /// `0x` prefix). This wallet pays gas for every release tx and
    /// signs the EIP-1559 envelope. **NOT** the validator's bridge
    /// committee key — that one signs the inner cert.
    pub operator_private_key_hex: String,
    /// Poll cadence in milliseconds.
    #[serde(default = "default_outbound_poll_ms")]
    pub poll_interval_ms: u64,
    /// Max withdrawal nonces to scan per poll. Without a chain-state
    /// reader for `next_withdrawal_nonce`, the relayer walks a fixed
    /// window each tick. Production swaps this for a chain read.
    #[serde(default = "default_scan_window")]
    pub scan_window: u64,
    /// Override the Eth RPC URL for the submitter. When `None`, the
    /// submitter reuses `bridge_node.eth_rpc_urls[0]` (top-level
    /// config). A separate field lets operators send writes through a
    /// flashbots-style protected RPC while reads still fan out across
    /// the multi-provider quorum.
    #[serde(default)]
    pub eth_submit_rpc_url: Option<String>,
}

fn default_outbound_poll_ms() -> u64 {
    10_000
}
fn default_scan_window() -> u64 {
    1024
}

fn default_eth_chain_id() -> u64 {
    11155111 // Sepolia testnet
}

fn default_poll_interval_ms() -> u64 {
    5000 // 5 seconds
}

fn default_max_log_query_range() -> u64 {
    1000
}

fn default_auto_pause_threshold() -> u32 {
    10
}

fn default_max_retry_elapsed_secs() -> u64 {
    120
}

fn default_wal_path() -> PathBuf {
    PathBuf::from("./bridge-wal")
}

fn default_deprecated_grpc_address() -> SocketAddr {
    "0.0.0.0:0".parse().expect("static SocketAddr")
}

fn default_http_listen_address() -> SocketAddr {
    // 0.0.0.0:9191 — matches the on-chain `BridgeMember.http_url`
    // convention used by single-validator dev clusters. Operators
    // override this for production deployments.
    "0.0.0.0:9191".parse().expect("static SocketAddr")
}

impl BridgeNodeConfig {
    /// Validate configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.eth_rpc_urls.is_empty() {
            return Err("At least one Ethereum RPC URL is required".into());
        }
        if self.bridge_contract_address.is_empty() {
            return Err("Bridge contract address is required".into());
        }
        if self.soma_rpc_url.is_empty() {
            return Err("Soma RPC URL is required".into());
        }
        if !self.bridge_key_path.exists() {
            return Err(format!("Bridge key file not found: {}", self.bridge_key_path.display()));
        }
        Ok(())
    }
}
