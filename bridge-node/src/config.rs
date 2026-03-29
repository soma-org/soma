use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::path::PathBuf;

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

    /// Address to listen on for gRPC signature exchange.
    pub grpc_listen_address: SocketAddr,

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
            return Err(format!(
                "Bridge key file not found: {}",
                self.bridge_key_path.display()
            ));
        }
        Ok(())
    }
}
