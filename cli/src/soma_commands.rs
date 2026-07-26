// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::fs;
use std::io::{self, Write as _, stdout};
use std::net::{AddrParseError, IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context as _, anyhow, bail, ensure};
use clap::{Command, CommandFactory as _, Parser};
use colored::Colorize;
use fastcrypto::traits::KeyPair as _;
use rand::rngs::OsRng;
use sdk::SomaClient;
use sdk::client_config::{SomaClientConfig, SomaEnv};
use sdk::wallet_context::{DEFAULT_WALLET_TIMEOUT_SEC, WalletContext, create_wallet_context};
use soma_keys::key_derive::generate_new_key;
use soma_keys::key_identity::KeyIdentity;
use soma_keys::keystore::{AccountKeystore as _, FileBasedKeystore, Keystore};
use test_cluster::swarm::Swarm;
use tokio::time::interval;
use tracing::info;
use types::base::SomaAddress;
use types::committee::CommitteeTrait as _;
use types::config::genesis_config::{GenesisConfig, ValidatorGenesisConfigBuilder};
use types::config::network_config::{ConfigBuilder, NetworkConfig};
use types::config::node_config::{FullnodeConfigBuilder, Genesis, default_json_rpc_address};
use types::config::p2p_config::SeedPeer;
use types::config::{
    Config, FULL_NODE_DB_PATH, PersistedConfig, SOMA_CLIENT_CONFIG, SOMA_FULLNODE_CONFIG,
    SOMA_GENESIS_FILENAME, SOMA_KEYSTORE_FILENAME, SOMA_NETWORK_CONFIG, genesis_blob_exists,
    soma_config_dir,
};
use types::crypto::{SignatureScheme, SomaKeyPair};
use types::digests::TransactionDigest;
use types::object::ObjectID;
use types::peer_id::PeerId;
use types::system_state::SystemStateTrait as _;
use url::Url;

use crate::client_commands::{SomaClientCommands, TxProcessingArgs};
use crate::commands;
use crate::commands::{
    ChannelCommand, EnvCommand, ModelCommand, ObjectCommand, OfferingCommand, ProviderArgs,
    ProviderCommand, ProxyArgs, SomaValidatorCommand, StakeCommand, TransferCommand, WalletCommand,
};
use crate::keytool::KeyToolCommand;

const DEFAULT_EPOCH_DURATION_MS: u64 = 86_400_000; // 24 hours; use admin endpoint to advance

pub(crate) const SOMA_BANNER: &str =
    "   ██████████      █████████████       ██████       █████          ████      
  ███      ██     ███       ██████      █████      █████           ████      
  ████      █   ███           █████     █ ████    ██████          █ ████     
  ████████      ██             ████     █  ████   █ ████         ██  ████    
   ██████████   ██              ████    █  ████  █  ████         █   ████    
       ███████  ██             ████     █   ██████   ███        ██████████   
  █       ████  ███           █████     █    ████    ████      ██     █████  
  ██      ███    ████       ██████     ██    ███     ████      █       ████  
  ██████████       █████████████      ████          ██████   ████     ███████";

/// Print the SOMA ASCII banner with a subtitle line underneath.
pub(crate) fn print_banner(subtitle: &str) {
    let banner_width = SOMA_BANNER.lines().map(|l| l.chars().count()).max().unwrap_or(68);
    eprintln!();
    for line in SOMA_BANNER.lines() {
        eprintln!("{line}");
    }
    eprintln!();
    eprintln!("  {}", "─".repeat(banner_width - 2).dimmed());
    eprintln!("  {}", subtitle.bold());
    eprintln!("  {}", "─".repeat(banner_width - 2).dimmed());
    eprintln!();
}

/// Print a key-value info panel inside a box.
fn print_info_panel(rows: &[(&str, &str)]) {
    let label_w = rows.iter().map(|(l, _)| l.len()).max().unwrap_or(10) + 2;
    let value_w = rows.iter().map(|(_, v)| v.len()).max().unwrap_or(20).max(20);
    let inner_w = 2 + label_w + value_w + 1;
    eprintln!("  {}", format!("┌{}┐", "─".repeat(inner_w)).dimmed());
    for (label, value) in rows {
        eprintln!(
            "  {}  {:<lw$}{:<vw$}{}",
            "│".dimmed(),
            label,
            value,
            "│".dimmed(),
            lw = label_w,
            vw = value_w + 1,
        );
    }
    eprintln!("  {}", format!("└{}┘", "─".repeat(inner_w)).dimmed());
}

#[derive(Parser)]
#[derive(Default)]
#[clap(rename_all = "kebab-case")]
pub struct SomaEnvConfig {
    /// Sets the file storing the state of our user accounts (an empty one will be created if missing)
    #[clap(long = "client.config")]
    config: Option<PathBuf>,
    /// The SOMA environment to use. This must be present in the current config file.
    #[clap(long = "client.env")]
    env: Option<String>,
    /// Create a new soma config without prompting if none exists
    #[clap(short = 'y', long = "yes")]
    accept_defaults: bool,
}

impl SomaEnvConfig {
    pub fn new(config: Option<PathBuf>, env: Option<String>) -> Self {
        Self { config, env, accept_defaults: false }
    }
}

#[allow(clippy::large_enum_variant)]
#[derive(Parser)]
#[clap(name = "soma", rename_all = "kebab-case")]
pub enum SomaCommand {
    // =========================================================================
    // COMMON USER ACTIONS (Top-level for convenience)
    // =========================================================================
    /// Check SOMA balance for an address
    #[clap(
        name = "balance",
        after_help = "\
EXAMPLES:
    soma balance
    soma balance 0x1234...5678"
    )]
    Balance {
        /// Address to check (defaults to active address)
        address: Option<KeyIdentity>,
        #[clap(long, global = true, help = "Output as JSON")]
        json: bool,
    },

    /// Transfer SOMA or USDC to a recipient
    #[clap(
        name = "transfer",
        after_help = "\
EXAMPLES:
    soma transfer soma 10 0x1234...5678
    soma transfer usdc 1.50 alice

For non-fungible transfers (objects), use `soma object transfer`."
    )]
    Transfer {
        #[clap(subcommand)]
        cmd: TransferCommand,
        #[clap(long, global = true, help = "Output as JSON")]
        json: bool,
    },

    /// Stake SOMA with a validator: add, remove, or list delegations
    #[clap(
        name = "stake",
        after_help = "\
EXAMPLES:
    soma stake add --validator 0xVAL... --amount 10
    soma stake remove --pool 0xPOOL_ID
    soma stake list"
    )]
    Stake {
        #[clap(subcommand)]
        cmd: StakeCommand,
        #[clap(long, global = true, help = "Output as JSON")]
        json: bool,
    },

    /// Show network connection status, version info, and active address
    #[clap(
        name = "status",
        after_help = "\
EXAMPLES:
    soma status
    soma status --json"
    )]
    Status {
        #[clap(long, global = true, help = "Output as JSON")]
        json: bool,
    },

    // =========================================================================
    // QUERY COMMANDS
    // =========================================================================
    /// Query on-chain objects by owner or ID, or transfer one to a recipient
    #[clap(
        name = "object",
        after_help = "\
EXAMPLES:
    soma object list
    soma object get 0xOBJECT_ID
    soma object transfer 0xOBJECT_ID 0xRECIPIENT"
    )]
    Object {
        #[clap(subcommand)]
        cmd: ObjectCommand,
        #[clap(long, global = true, help = "Output as JSON")]
        json: bool,
    },

    /// Get transaction details or execute serialized transactions
    #[clap(
        name = "tx",
        after_help = "\
EXAMPLES:
    soma tx DIGEST_BASE58
    soma tx execute-serialized <TX_BYTES>
    soma tx execute-signed --tx-bytes <BYTES> --signatures <SIGS>"
    )]
    Tx {
        #[clap(flatten)]
        config: SomaEnvConfig,
        #[clap(subcommand)]
        cmd: Option<TxCommand>,
        #[clap(long, global = true, help = "Output as JSON")]
        json: bool,
    },

    // =========================================================================
    // MANAGEMENT COMMANDS
    // =========================================================================
    /// Manage wallet addresses and keys
    #[clap(
        name = "wallet",
        after_help = "\
EXAMPLES:
    soma wallet list
    soma wallet new --alias my-wallet
    soma wallet switch 0x1234...5678"
    )]
    Wallet {
        #[clap(subcommand)]
        cmd: WalletCommand,
        #[clap(long, global = true, help = "Output as JSON")]
        json: bool,
    },

    /// Manage network environments (switch, add, list)
    #[clap(
        name = "env",
        after_help = "\
EXAMPLES:
    soma env list
    soma env switch testnet
    soma env new --alias mynet --rpc http://..."
    )]
    Env {
        #[clap(subcommand)]
        cmd: EnvCommand,
        #[clap(long, global = true, help = "Output as JSON")]
        json: bool,
    },

    /// Run the local agent-facing inference proxy
    ///
    /// Listens on `127.0.0.1:<port>` and speaks the OpenAI API. Agents
    /// point at it via `OPENAI_BASE_URL`. The proxy discovers providers,
    /// picks one per model, and signs vouchers per request.
    #[clap(
        name = "proxy",
        after_help = "EXAMPLE:\n    soma proxy --listen 127.0.0.1:7662 --indexer-url ..."
    )]
    Proxy(ProxyArgs),

    // =========================================================================
    // OPERATOR COMMANDS
    // =========================================================================
    /// Manage on-chain payment channels (open, settle, top-up, close).
    #[clap(
        name = "channel",
        after_help = "\
EXAMPLES:
    soma channel open --payee 0xabc... --deposit 1000000
    soma channel show --channel-id 0xdef..."
    )]
    Channel {
        #[clap(subcommand)]
        cmd: ChannelCommand,
    },

    /// Soma ↔ Eth USDC bridge: initiate withdrawals, inspect the live
    /// `BridgeState`. Inbound deposits go through the Eth-side
    /// `SomaBridge.deposit(...)` call — there's no Soma-side
    /// command for that direction.
    #[clap(
        name = "bridge",
        after_help = "\
EXAMPLES:
    soma bridge status
    soma bridge withdraw --amount 1.0 \\
        --recipient 0x7B42d2B6F94fDF3c2Fe62e0aAf451487FA2DAB6e \\
        --target-chain base-sepolia"
    )]
    Bridge {
        #[clap(subcommand)]
        cmd: crate::commands::BridgeCommand,
    },

    /// Manage on-chain provider registry (register, update, show).
    #[clap(
        name = "provider",
        after_help = "\
EXAMPLES:
    soma provider register --endpoint https://my.provider:8080
    soma provider show
    soma provider update --endpoint https://new.endpoint:8080"
    )]
    Provider {
        #[clap(subcommand)]
        cmd: ProviderCommand,
    },

    /// Manage per-(provider, model) on-chain offerings.
    #[clap(
        name = "offering",
        after_help = "\
EXAMPLES:
    soma offering register --model-id anthropic/claude-haiku-4.5 \\
        --prompt-micros-per-1k 1000 --completion-micros-per-1k 5000
    soma offering show --model-id anthropic/claude-haiku-4.5
    soma offering deactivate --model-id anthropic/claude-haiku-4.5"
    )]
    Offering {
        #[clap(subcommand)]
        cmd: OfferingCommand,
    },

    /// Read the protocol-config `ModelRegistry`.
    #[clap(
        name = "model",
        after_help = "\
EXAMPLES:
    soma model list
    soma model list --ids-only
    soma model show --model-id anthropic/claude-sonnet-4.6"
    )]
    Model {
        #[clap(subcommand)]
        cmd: ModelCommand,
    },

    /// Manage validators (register, set gas price, commission)
    #[clap(
        name = "validator",
        after_help = "\
EXAMPLES:
    soma validator display-metadata
    soma validator list"
    )]
    Validator {
        #[clap(flatten)]
        config: SomaEnvConfig,
        #[clap(subcommand)]
        cmd: Option<SomaValidatorCommand>,
        #[clap(long, global = true, help = "Output as JSON")]
        json: bool,
    },

    // =========================================================================
    // NODE OPERATIONS
    // =========================================================================
    /// Start a long-running service (localnet, validator, provider)
    #[clap(
        name = "start",
        after_help = "\
EXAMPLES:
    soma start localnet --force-regenesis
    soma start validator --config validator.yaml
    soma start provider --config provider.toml"
    )]
    Start {
        #[clap(subcommand)]
        cmd: StartCommand,
    },

    /// Inspect local network configuration and validator addresses
    #[clap(name = "network")]
    Network {
        #[clap(long = "network.config")]
        config: Option<PathBuf>,
        #[clap(short, long, help = "Dump the public keys of all authorities")]
        dump_addresses: bool,
    },

    /// Bootstrap and initialize a new SOMA network
    #[clap(name = "genesis")]
    Genesis {
        #[clap(subcommand)]
        cmd: Option<GenesisCommand>,
        #[clap(long, help = "Start genesis with a given config file")]
        from_config: Option<PathBuf>,
        #[clap(long, help = "Build a genesis config, write it to the specified path, and exit")]
        write_config: Option<PathBuf>,
        #[clap(long)]
        working_dir: Option<PathBuf>,
        #[clap(short, long, help = "Forces overwriting existing configuration")]
        force: bool,
        #[clap(long = "epoch-duration-ms")]
        epoch_duration_ms: Option<u64>,
        /// Set number of validators in the network.
        #[clap(long)]
        committee_size: Option<usize>,
    },

    /// Low-level keystore operations (generate, import, export keys)
    #[clap(name = "keytool")]
    KeyTool {
        #[clap(long)]
        keystore_path: Option<PathBuf>,
        #[clap(long, global = true, help = "Output as JSON")]
        json: bool,
        #[clap(subcommand)]
        cmd: KeyToolCommand,
    },

    /// Generate shell completion scripts
    #[clap(
        name = "completions",
        hide = true,
        after_help = "\
EXAMPLES:
    soma completions bash > /usr/local/etc/bash_completion.d/soma
    soma completions zsh > ~/.zfunc/_soma
    soma completions fish > ~/.config/fish/completions/soma.fish"
    )]
    Completions {
        /// Shell to generate completions for
        shell: clap_complete::Shell,
    },
}

/// Subcommands for `soma start` — all long-running services.
#[derive(Parser)]
#[clap(rename_all = "kebab-case")]
pub enum StartCommand {
    /// Start a local SOMA network for development and testing
    ///
    /// Launches local validators and a fullnode.
    /// State is persisted in ~/.soma/ by default, or use --force-regenesis
    /// for an ephemeral network that starts fresh each time.
    #[clap(
        name = "localnet",
        after_help = "\
EXAMPLES:
    soma start localnet --force-regenesis"
    )]
    Localnet {
        /// Config directory that will be used to store network config, node db, keystore.
        #[clap(long = "network.config")]
        config_dir: Option<std::path::PathBuf>,

        /// A new genesis is created each time this flag is set, and state is not persisted between
        /// runs.
        #[clap(long)]
        force_regenesis: bool,

        /// Port to start the Fullnode RPC server on. Default port is 9000.
        #[clap(long, default_value = "9000")]
        fullnode_rpc_port: u16,

        /// Set the epoch duration. Can only be used when `--force-regenesis` flag is passed.
        #[clap(long)]
        epoch_duration_ms: Option<u64>,

        /// Start the network without a fullnode
        #[clap(long = "no-full-node")]
        no_full_node: bool,

        /// Set the number of validators in the network.
        #[clap(long)]
        committee_size: Option<usize>,

        /// Log level for CLI output (trace, debug, info, warn, error).
        #[clap(long, default_value = "info")]
        log_level: String,
    },

    /// Start a validator node from a config file
    #[clap(
        name = "validator",
        after_help = "\
EXAMPLES:
    soma start validator --config validator.yaml"
    )]
    Validator {
        /// Path to the validator config file (YAML)
        #[clap(long = "config", short = 'c')]
        config: PathBuf,
    },

    /// Run the provider-side inference daemon (formerly `soma inference serve`)
    ///
    /// Fronts an OpenAI-compatible upstream behind SomaPay-authorized
    /// `/v1/chat/completions`. Backends today: openrouter, vast.
    #[clap(
        name = "provider",
        after_help = "EXAMPLE:\n    soma start provider --config provider.toml"
    )]
    Provider(ProviderArgs),
}

/// Subcommands for `soma tx` — transaction queries and raw execution.
#[derive(Parser)]
#[clap(rename_all = "kebab-case")]
pub enum TxCommand {
    /// Get transaction details by digest
    #[clap(name = "info")]
    Info {
        /// Transaction digest
        digest: TransactionDigest,
    },

    /// Execute from serialized transaction bytes
    #[clap(name = "execute-serialized")]
    ExecuteSerialized {
        /// Base64-encoded BCS-serialized TransactionData
        tx_bytes: String,
        #[clap(flatten)]
        processing: crate::client_commands::TxProcessingArgs,
    },

    /// Execute using pre-signed transaction bytes and signatures
    #[clap(name = "execute-signed")]
    ExecuteSigned {
        /// Base64-encoded unsigned transaction data
        #[clap(long)]
        tx_bytes: String,
        /// Base64-encoded signatures (flag || signature || pubkey)
        #[clap(long)]
        signatures: Vec<String>,
    },

    /// Execute a combined sender-signed transaction
    #[clap(name = "execute-combined-signed")]
    ExecuteCombinedSigned {
        /// Base64-encoded SenderSignedData
        #[clap(long)]
        signed_tx_bytes: String,
    },
}

/// Subcommands for `soma genesis`.
#[derive(Parser)]
#[clap(rename_all = "kebab-case")]
pub enum GenesisCommand {
    /// Coordinate multi-validator genesis for network launches
    #[clap(name = "ceremony")]
    Ceremony(crate::genesis_ceremony::Ceremony),

    /// Inspect a genesis.blob file for key parameters
    #[clap(name = "inspect")]
    Inspect {
        /// Path to genesis.blob file
        #[clap(name = "genesis-blob-path")]
        file: PathBuf,
    },
}

impl SomaCommand {
    pub fn log_level(&self) -> tracing::Level {
        match self {
            SomaCommand::Start { cmd } => match cmd {
                StartCommand::Localnet { log_level, .. } => {
                    log_level.parse().unwrap_or(tracing::Level::INFO)
                }
                StartCommand::Validator { .. } | StartCommand::Provider(_) => tracing::Level::INFO,
            },
            SomaCommand::Proxy(_) => tracing::Level::INFO,
            _ => tracing::Level::ERROR,
        }
    }

    pub async fn execute(self) -> Result<(), anyhow::Error> {
        match self {
            // =================================================================
            // COMMON USER ACTIONS
            // =================================================================
            SomaCommand::Balance { address, json } => {
                let context = get_wallet_context(&SomaEnvConfig::default()).await?;
                let result = commands::balance::execute(&context, address).await?;
                result.print(json);
                Ok(())
            }

            SomaCommand::Transfer { cmd, json } => {
                let mut context = get_wallet_context(&SomaEnvConfig::default()).await?;
                let result = cmd.execute(&mut context).await?;
                result.print(json);
                if result.has_failed_transaction() {
                    std::process::exit(1);
                }
                Ok(())
            }

            SomaCommand::Stake { cmd, json } => {
                let mut context = get_wallet_context(&SomaEnvConfig::default()).await?;
                let result = cmd.execute(&mut context, json).await?;
                result.print(json);
                if result.has_failed_transaction() {
                    std::process::exit(1);
                }
                Ok(())
            }

            SomaCommand::Status { json } => {
                let mut context = get_wallet_context(&SomaEnvConfig::default()).await?;
                let active_address = context.active_address().ok();
                let active_env = context.config.active_env.clone();
                let rpc_url =
                    context.config.get_active_env().map(|e| e.rpc.clone()).unwrap_or_default();

                let (
                    server_version,
                    chain_id,
                    epoch,
                    epoch_start_timestamp_ms,
                    epoch_duration_ms,
                    protocol_version,
                    soma_balance,
                    usdc_balance,
                    server_unreachable,
                ) = match context.get_client().await {
                    Ok(client) => {
                        let chain_id = client.get_chain_identifier().await.ok();
                        let server_version = client.get_server_version().await.ok();
                        let state = client.get_latest_system_state().await.ok();
                        let epoch = state.as_ref().map(|s| s.epoch());
                        let epoch_start_ms = state.as_ref().map(|s| s.epoch_start_timestamp_ms());
                        let epoch_dur_ms = state.as_ref().map(|s| s.epoch_duration_ms());
                        let protocol_version = client.get_protocol_version().await.ok();
                        let (soma_balance, usdc_balance) = if let Some(addr) = &active_address {
                            (
                                client
                                    .get_balance_by_coin_type(addr, types::object::CoinType::Soma)
                                    .await
                                    .ok(),
                                client
                                    .get_balance_by_coin_type(addr, types::object::CoinType::Usdc)
                                    .await
                                    .ok(),
                            )
                        } else {
                            (None, None)
                        };
                        let unreachable =
                            server_version.is_none() && chain_id.is_none() && epoch.is_none();
                        (
                            server_version,
                            chain_id,
                            epoch,
                            epoch_start_ms,
                            epoch_dur_ms,
                            protocol_version,
                            soma_balance,
                            usdc_balance,
                            unreachable,
                        )
                    }
                    Err(_) => (None, None, None, None, None, None, None, None, true),
                };

                let next_epoch_in = epoch_start_timestamp_ms
                    .zip(epoch_duration_ms)
                    .and_then(|(s, d)| crate::response::format_next_epoch_hint(s, d));

                let output = crate::response::StatusOutput {
                    network: active_env,
                    rpc_url,
                    server_version,
                    chain_id,
                    epoch,
                    epoch_start_timestamp_ms,
                    epoch_duration_ms,
                    next_epoch_in,
                    protocol_version,
                    active_address: active_address.map(|a| a.to_string()),
                    soma_balance,
                    usdc_balance,
                    server_reachable: !server_unreachable,
                };
                output.print(json);
                Ok(())
            }

            // =================================================================
            // QUERY COMMANDS
            // =================================================================
            SomaCommand::Object { cmd, json } => {
                let mut context = get_wallet_context(&SomaEnvConfig::default()).await?;
                let result = commands::objects::execute(&mut context, cmd).await?;
                result.print(json);
                if result.has_failed_transaction() {
                    std::process::exit(1);
                }
                Ok(())
            }

            SomaCommand::Tx { config, cmd, json } => {
                match cmd {
                    Some(TxCommand::Info { digest }) => {
                        let context = get_wallet_context(&config).await?;
                        let result = commands::tx::execute(&context, digest).await?;
                        result.print(json);
                    }
                    Some(TxCommand::ExecuteSerialized { tx_bytes, processing }) => {
                        let mut context = get_wallet_context(&config).await?;
                        if let Ok(client) = context.get_client().await {
                            if let Err(e) = client.check_api_version().await {
                                eprintln!("{}", format!("[warning] {e}").yellow().bold());
                            }
                        }
                        SomaClientCommands::ExecuteSerialized { tx_bytes, processing }
                            .execute(&mut context)
                            .await?
                            .print(json);
                    }
                    Some(TxCommand::ExecuteSigned { tx_bytes, signatures }) => {
                        let mut context = get_wallet_context(&config).await?;
                        if let Ok(client) = context.get_client().await {
                            if let Err(e) = client.check_api_version().await {
                                eprintln!("{}", format!("[warning] {e}").yellow().bold());
                            }
                        }
                        SomaClientCommands::ExecuteSignedTx { tx_bytes, signatures }
                            .execute(&mut context)
                            .await?
                            .print(json);
                    }
                    Some(TxCommand::ExecuteCombinedSigned { signed_tx_bytes }) => {
                        let mut context = get_wallet_context(&config).await?;
                        if let Ok(client) = context.get_client().await {
                            if let Err(e) = client.check_api_version().await {
                                eprintln!("{}", format!("[warning] {e}").yellow().bold());
                            }
                        }
                        SomaClientCommands::ExecuteCombinedSignedTx { signed_tx_bytes }
                            .execute(&mut context)
                            .await?
                            .print(json);
                    }
                    None => {
                        let mut app: Command = SomaCommand::command();
                        app.build();
                        app.find_subcommand_mut("tx").unwrap().print_help()?;
                    }
                }
                Ok(())
            }

            // =================================================================
            // MANAGEMENT COMMANDS
            // =================================================================
            SomaCommand::Wallet { cmd, json } => {
                let mut context = get_wallet_context(&SomaEnvConfig::default()).await?;
                let result = commands::wallet::execute(&mut context, cmd).await?;
                result.print(json);
                Ok(())
            }

            SomaCommand::Env { cmd, json } => {
                let mut context = get_wallet_context(&SomaEnvConfig::default()).await?;
                let result = commands::env::execute(&mut context, cmd).await?;
                result.print(json);
                Ok(())
            }

            SomaCommand::Proxy(args) => commands::inference::run_proxy(args).await,

            SomaCommand::Channel { cmd } => cmd.execute().await,

            SomaCommand::Bridge { cmd } => {
                let mut context = get_wallet_context(&SomaEnvConfig::default()).await?;
                // The `json` flag lives inside each subcommand; pull it
                // back out for `print()`. Withdraw + Status set it via
                // a global `--json`; default is the human-readable view.
                let json_flag = matches!(
                    &cmd,
                    crate::commands::BridgeCommand::Withdraw { json: true, .. }
                        | crate::commands::BridgeCommand::Status { json: true }
                );
                let result = cmd.execute(&mut context).await?;
                result.print(json_flag);
                if result.has_failed_transaction() {
                    std::process::exit(1);
                }
                Ok(())
            }

            SomaCommand::Provider { cmd } => cmd.execute().await,

            SomaCommand::Offering { cmd } => cmd.execute().await,

            SomaCommand::Model { cmd } => cmd.execute().await,

            // =================================================================
            // OPERATOR COMMANDS
            // =================================================================
            SomaCommand::Validator { config, cmd, json } => {
                let mut context = get_wallet_context(&config).await?;
                if let Some(cmd) = cmd {
                    if let Ok(client) = context.get_client().await {
                        if let Err(e) = client.check_api_version().await {
                            eprintln!("{}", format!("[warning] {e}").yellow().bold());
                        }
                    }
                    cmd.execute(&mut context).await?.print(json);
                } else {
                    let mut app: Command = SomaCommand::command();
                    app.build();
                    app.find_subcommand_mut("validator").unwrap().print_help()?;
                }
                Ok(())
            }

            // =================================================================
            // NODE OPERATIONS
            // =================================================================
            SomaCommand::Network { config, dump_addresses } => {
                let config_path = config.unwrap_or(soma_config_dir()?.join(SOMA_NETWORK_CONFIG));
                let config: NetworkConfig = PersistedConfig::read(&config_path).map_err(|err| {
                    err.context(format!(
                        "Cannot open SOMA network config file at {:?}",
                        config_path
                    ))
                })?;

                if dump_addresses {
                    for validator in config.validator_configs() {
                        println!(
                            "{} - {}",
                            validator.network_address(),
                            validator.protocol_key_pair().public(),
                        );
                    }
                }
                Ok(())
            }

            SomaCommand::Start { cmd } => {
                match cmd {
                    StartCommand::Localnet {
                        config_dir,
                        force_regenesis,
                        fullnode_rpc_port,
                        no_full_node,
                        epoch_duration_ms,
                        committee_size,
                        log_level: _,
                    } => {
                        start(
                            config_dir.clone(),
                            force_regenesis,
                            epoch_duration_ms,
                            fullnode_rpc_port,
                            no_full_node,
                            committee_size,
                        )
                        .await?;
                    }
                    StartCommand::Validator { config } => {
                        commands::validator::start_validator_node(config).await?;
                    }
                    StartCommand::Provider(args) => {
                        commands::inference::run_provider(args).await?;
                    }
                }
                Ok(())
            }

            SomaCommand::Genesis {
                cmd,
                working_dir,
                force,
                from_config,
                write_config,
                epoch_duration_ms,
                committee_size,
            } => {
                match cmd {
                    Some(GenesisCommand::Ceremony(ceremony)) => {
                        return crate::genesis_ceremony::run(ceremony);
                    }
                    Some(GenesisCommand::Inspect { file }) => {
                        return crate::genesis_ceremony::inspect_genesis_blob(&file);
                    }
                    None => {}
                }
                genesis(
                    from_config,
                    write_config,
                    working_dir,
                    force,
                    epoch_duration_ms,
                    committee_size,
                )
                .await
            }

            SomaCommand::KeyTool { keystore_path, json, cmd } => {
                let keystore_path =
                    keystore_path.unwrap_or(soma_config_dir()?.join(SOMA_KEYSTORE_FILENAME));
                let mut keystore =
                    Keystore::from(FileBasedKeystore::load_or_create(&keystore_path)?);
                cmd.execute(&mut keystore).await?.print(json);
                Ok(())
            }

            SomaCommand::Completions { shell } => {
                use clap::CommandFactory as _;
                let mut cmd = crate::soma_commands::SomaCommand::command();
                clap_complete::generate(shell, &mut cmd, "soma", &mut io::stdout());
                Ok(())
            }
        }
    }
}

// =============================================================================
// Helper functions (start, genesis, get_wallet_context, etc.)
// =============================================================================

/// Starts a local network with the given configuration.
async fn start(
    config: Option<PathBuf>,
    force_regenesis: bool,
    epoch_duration_ms: Option<u64>,
    fullnode_rpc_port: u16,
    no_full_node: bool,
    committee_size: Option<usize>,
) -> Result<(), anyhow::Error> {
    if force_regenesis {
        ensure!(
            config.is_none(),
            "Cannot pass `--force-regenesis` and `--network.config` at the same time."
        );
    }

    if epoch_duration_ms.is_some() && genesis_blob_exists(config.clone()) && !force_regenesis {
        bail!(
            "Epoch duration can only be set when passing the `--force-regenesis` flag, or when \
            there is no genesis configuration in the default SOMA configuration folder or the given \
            network.config argument.",
        );
    }

    let mut swarm_builder = Swarm::builder();

    let config_dir = if force_regenesis {
        let committee_size = match committee_size {
            Some(x) => NonZeroUsize::new(x),
            None => NonZeroUsize::new(1),
        }
        .ok_or_else(|| anyhow!("Committee size must be at least 1."))?;
        swarm_builder = swarm_builder.committee_size(committee_size);
        let genesis_config = GenesisConfig::for_local_testing();
        swarm_builder = swarm_builder.with_genesis_config(genesis_config);
        let epoch_duration_ms = epoch_duration_ms.unwrap_or(DEFAULT_EPOCH_DURATION_MS);
        swarm_builder = swarm_builder.with_epoch_duration_ms(epoch_duration_ms);
        tempfile::tempdir()?.keep()
    } else {
        let (network_config_path, soma_config_path) = match config {
            Some(config)
                if config.is_file()
                    && config.extension().is_some_and(|e| e == "yml" || e == "yaml") =>
            {
                if committee_size.is_some() {
                    eprintln!(
                        "{}",
                        "[warning] The committee-size arg will be ignored as a network \
                            configuration already exists."
                            .yellow()
                            .bold()
                    );
                }
                (config, soma_config_dir()?)
            }

            Some(config) => {
                if committee_size.is_some() {
                    eprintln!(
                        "{}",
                        "[warning] The committee-size arg will be ignored as a network \
                            configuration already exists."
                            .yellow()
                            .bold()
                    );
                }
                (config.join(SOMA_NETWORK_CONFIG), config)
            }

            None => {
                let soma_config = soma_config_dir()?;
                let network_config = soma_config.join(SOMA_NETWORK_CONFIG);

                if !network_config.exists() {
                    genesis(None, None, None, false, epoch_duration_ms, committee_size)
                        .await
                        .map_err(|_| {
                            anyhow!(
                                "Cannot run genesis with non-empty SOMA config directory: {}.\n\n\
                                If you are trying to run a local network without persisting the \
                                data, use --force-regenesis flag.",
                                soma_config.display(),
                            )
                        })?;
                } else if committee_size.is_some() {
                    eprintln!(
                        "{}",
                        "[warning] The committee-size arg will be ignored as a network \
                            configuration already exists."
                            .yellow()
                            .bold()
                    );
                }

                (network_config, soma_config)
            }
        };

        let network_config: NetworkConfig =
            PersistedConfig::read(&network_config_path).map_err(|err| {
                err.context(format!(
                    "Cannot open SOMA network config file at {:?}",
                    network_config_path
                ))
            })?;

        swarm_builder =
            swarm_builder.dir(soma_config_path.clone()).with_network_config(network_config);

        soma_config_path
    };

    let mut fullnode_rpc_address = types::config::node_config::default_json_rpc_address();
    fullnode_rpc_address.set_port(fullnode_rpc_port);

    if no_full_node {
        swarm_builder = swarm_builder.with_fullnode_count(0);
    } else {
        let rpc_config = types::config::rpc_config::RpcConfig {
            enable_indexing: Some(true),
            ..Default::default()
        };

        swarm_builder = swarm_builder
            .with_fullnode_count(1)
            .with_fullnode_rpc_addr(fullnode_rpc_address)
            .with_fullnode_rpc_config(rpc_config);
    }

    let num_validators = committee_size.unwrap_or(1);

    // -- Build & launch -------------------------------------------------------
    const STATUS_WIDTH: usize = 50;
    print_banner("Local Network");

    let msg = "Generating genesis...";
    eprint!("  {msg:<width$}", width = STATUS_WIDTH);
    let mut swarm = swarm_builder.build();
    eprintln!("{}", "done".green());

    let msg = format!("Starting validators ({num_validators})...");
    eprint!("  {msg:<width$}", width = STATUS_WIDTH);
    swarm.launch().await?;
    eprintln!("{}", "done".green());

    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;

    let fullnode_rpc_url =
        socket_addr_to_url(fullnode_rpc_address)?.to_string().trim_end_matches("/").to_string();

    if !no_full_node {
        let msg = "Starting fullnode...";
        eprintln!("  {msg:<width$}{done}", width = STATUS_WIDTH, done = "done".green());
    }

    if config_dir.join(SOMA_CLIENT_CONFIG).exists() {
        let _ = update_wallet_config_rpc(config_dir.clone(), fullnode_rpc_url.clone())?;
    }

    if force_regenesis && soma_config_dir()?.join(SOMA_CLIENT_CONFIG).exists() {
        let _ = update_wallet_config_rpc(soma_config_dir()?, fullnode_rpc_url.clone())?;
    }

    // -- Network ready banner -------------------------------------------------
    let epoch_ms = epoch_duration_ms.unwrap_or(DEFAULT_EPOCH_DURATION_MS);
    let state_dir = config_dir.display().to_string();
    let persistence = if force_regenesis { "ephemeral" } else { "enabled" };

    eprintln!();
    eprintln!("  {}", "Network ready.".green().bold());
    eprintln!();
    let epoch_display = format!("{}s", epoch_ms / 1000);
    let rows: Vec<(&str, &str)> = vec![
        ("RPC URL", &fullnode_rpc_url),
        ("Epoch", &epoch_display),
        ("Persistence", persistence),
    ];
    print_info_panel(&rows);
    eprintln!();
    eprintln!("  State dir: {}", state_dir.dimmed());
    eprintln!();
    eprintln!("  Press {} to stop the network.", "Ctrl+C".bold());

    // -- Main loop ------------------------------------------------------------
    let mut interval = interval(Duration::from_secs(3));

    loop {
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {
                break;
            }
            _ = interval.tick() => {}
        }
    }

    // -- Graceful shutdown ----------------------------------------------------
    eprintln!();
    eprintln!("  {}", "Shutting down...".yellow());
    for node in swarm.validator_nodes() {
        node.stop();
    }
    let msg = "Stopping validators...";
    eprintln!("  {msg:<width$}{done}", width = STATUS_WIDTH, done = "done".green());
    for node in swarm.fullnodes() {
        node.stop();
    }
    if !no_full_node {
        let msg = "Stopping fullnode...";
        eprintln!("  {msg:<width$}{done}", width = STATUS_WIDTH, done = "done".green());
    }
    if force_regenesis {
        eprintln!("  Ephemeral state discarded.");
    } else {
        eprintln!("  Network state saved to {}", state_dir.dimmed());
    }
    eprintln!("  {}", "Done.".green().bold());

    Ok(())
}

async fn genesis(
    from_config: Option<PathBuf>,
    write_config: Option<PathBuf>,
    working_dir: Option<PathBuf>,
    force: bool,
    epoch_duration_ms: Option<u64>,
    committee_size: Option<usize>,
) -> Result<(), anyhow::Error> {
    let soma_config_dir = &match working_dir {
        Some(v) => v,
        None => {
            let config_path = soma_config_dir()?;
            fs::create_dir_all(&config_path)?;
            config_path
        }
    };

    let dir = soma_config_dir.read_dir().map_err(|err| {
        anyhow!(err).context(format!("Cannot open SOMA config dir {:?}", soma_config_dir))
    })?;
    let files = dir.collect::<Result<Vec<_>, _>>()?;

    let client_path = soma_config_dir.join(SOMA_CLIENT_CONFIG);
    let keystore_path = soma_config_dir.join(SOMA_KEYSTORE_FILENAME);

    if write_config.is_none() && !files.is_empty() {
        if force {
            let is_compatible = FileBasedKeystore::load_or_create(&keystore_path).is_ok()
                && PersistedConfig::<SomaClientConfig>::read(&client_path).is_ok();
            if is_compatible {
                for file in files {
                    let path = file.path();
                    if path != client_path && path != keystore_path {
                        if path.is_file() {
                            fs::remove_file(path)
                        } else {
                            fs::remove_dir_all(path)
                        }
                        .map_err(|err| {
                            anyhow!(err).context(format!("Cannot remove file {:?}", file.path()))
                        })?;
                    }
                }
            } else {
                fs::remove_dir_all(soma_config_dir).map_err(|err| {
                    anyhow!(err)
                        .context(format!("Cannot remove SOMA config dir {:?}", soma_config_dir))
                })?;
                fs::create_dir(soma_config_dir).map_err(|err| {
                    anyhow!(err)
                        .context(format!("Cannot create SOMA config dir {:?}", soma_config_dir))
                })?;
            }
        } else if files.len() != 2 || !client_path.exists() || !keystore_path.exists() {
            bail!(
                "Cannot run genesis with non-empty SOMA config directory {}, please use the --force/-f option to remove the existing configuration",
                soma_config_dir.to_str().unwrap()
            );
        }
    }

    let network_path = soma_config_dir.join(SOMA_NETWORK_CONFIG);
    let genesis_path = soma_config_dir.join(SOMA_GENESIS_FILENAME);

    let mut genesis_conf = match from_config {
        Some(path) => PersistedConfig::read(&path)?,
        None => {
            if keystore_path.exists() {
                let existing_keys = FileBasedKeystore::load_or_create(&keystore_path)?.addresses();
                GenesisConfig::for_local_testing_with_addresses(existing_keys)
            } else {
                GenesisConfig::for_local_testing()
            }
        }
    };

    if let Some(path) = write_config {
        let persisted = genesis_conf.persisted(&path);
        persisted.save()?;
        return Ok(());
    }

    let validator_info = genesis_conf.validator_config_info.take();

    let mut builder = ConfigBuilder::new(soma_config_dir);
    if let Some(epoch_duration_ms) = epoch_duration_ms {
        genesis_conf.parameters.epoch_duration_ms = epoch_duration_ms;
    }

    let committee_size = match committee_size {
        Some(x) => NonZeroUsize::new(x),
        None => NonZeroUsize::new(1),
    }
    .ok_or_else(|| anyhow!("Committee size must be at least 1."))?;

    let mut network_config = if let Some(validators) = validator_info {
        builder.with_genesis_config(genesis_conf).with_validators(validators).build()
    } else {
        builder.committee_size(committee_size).with_genesis_config(genesis_conf).build()
    };

    let mut keystore = FileBasedKeystore::load_or_create(&keystore_path)?;
    for key in &network_config.account_keys {
        keystore.import(None, key.copy()).await?;
    }
    let active_address = keystore.addresses().pop();

    network_config.genesis.save(&genesis_path)?;
    for validator in &mut network_config.validator_configs {
        validator.genesis = Genesis::new_from_file(&genesis_path);
    }

    info!("Network genesis completed.");
    network_config.save(&network_path)?;
    info!("Network config file is stored in {:?}.", network_path);
    info!("Client keystore is stored in {:?}.", keystore_path);

    for (i, validator) in network_config.validator_configs().iter().enumerate() {
        let path = soma_config_dir.join(format!("validator_{}.yaml", i));
        validator.save(&path)?;
        info!("Validator config saved to {:?}", path);
    }

    // Build a separate fullnode config using FullnodeConfigBuilder
    let seed_peers: Vec<SeedPeer> = network_config
        .validator_configs()
        .iter()
        .filter_map(|config| {
            let p2p_address = config.p2p_config.external_address.clone()?;
            Some(SeedPeer {
                peer_id: Some(PeerId(config.network_key_pair().public().into_inner().0.to_bytes())),
                address: p2p_address,
            })
        })
        .collect();

    let fullnode_config = FullnodeConfigBuilder::new()
        .with_config_directory(FULL_NODE_DB_PATH.into())
        .with_rpc_addr(default_json_rpc_address())
        .build(network_config.genesis.clone(), seed_peers);

    fullnode_config.save(soma_config_dir.join(SOMA_FULLNODE_CONFIG))?;
    info!("Fullnode config saved in {:?}", soma_config_dir.join(SOMA_FULLNODE_CONFIG));

    let mut client_config = if client_path.exists() {
        PersistedConfig::read(&client_path)?
    } else {
        SomaClientConfig::new(keystore.into())
    };

    if client_config.active_address.is_none() {
        client_config.active_address = active_address;
    }

    let rpc = socket_addr_to_url(fullnode_config.rpc_address)?
        .to_string()
        .trim_end_matches("/")
        .to_string();

    client_config.add_env(SomaEnv {
        alias: "localnet".to_string(),
        rpc,
        basic_auth: None,
        chain_id: None,
    });
    // client_config.add_env(SomaEnv::devnet());  // devnet removed

    if client_config.active_env.is_none() {
        client_config.active_env = client_config.envs.first().map(|env| env.alias.clone());
    }

    client_config.save(&client_path)?;
    info!("Client config file is stored in {:?}.", client_path);

    Ok(())
}

/// If `wallet_conf_file` doesn't exist, prompt the user and create it.
async fn prompt_if_no_config(
    wallet_conf_file: &Path,
    accept_defaults: bool,
) -> Result<(), anyhow::Error> {
    if wallet_conf_file.exists() {
        return Ok(());
    }

    if !accept_defaults {
        println!(
            "No soma config found in `{}`, create one [Y/n]?",
            wallet_conf_file.to_string_lossy()
        );
        let response = read_line()?.trim().to_lowercase();
        if !response.is_empty() && response != "y" {
            bail!("No config found, aborting");
        }
    }

    let config_dir = wallet_conf_file
        .parent()
        .ok_or_else(|| anyhow!("Error: {wallet_conf_file:?} is an invalid file path"))?;

    let (keystore, address) =
        create_default_keystore(&config_dir.join(SOMA_KEYSTORE_FILENAME)).await?;

    let default_env = SomaEnv::testnet();
    let default_env_name = default_env.alias.clone();
    SomaClientConfig {
        keystore,
        envs: vec![default_env, /* SomaEnv::mainnet(), */ SomaEnv::localnet()],
        external_keys: None,
        active_address: Some(address),
        active_env: Some(default_env_name.clone()),
    }
    .persisted(wallet_conf_file)
    .save()?;
    println!("Created {wallet_conf_file:?}");
    println!("Set active environment to {default_env_name}");

    Ok(())
}

async fn create_default_keystore(keystore_file: &Path) -> anyhow::Result<(Keystore, SomaAddress)> {
    let mut keystore =
        Keystore::from(FileBasedKeystore::load_or_create(&keystore_file.to_path_buf())?);
    let key_scheme = SignatureScheme::ED25519;
    let (new_address, key_pair, scheme, phrase) = generate_new_key(key_scheme, None, None)?;
    keystore.import(None, key_pair).await?;
    let alias = keystore.get_alias(&new_address)?;

    println!(
        "Generated new keypair and alias for address with scheme {:?} [{alias}: {new_address}]",
        scheme.to_string()
    );
    println!("  secret recovery phrase : [{phrase}]");

    Ok((keystore, new_address))
}

fn read_line() -> Result<String, anyhow::Error> {
    let mut s = String::new();
    let _ = stdout().flush();
    io::stdin().read_line(&mut s)?;
    Ok(s.trim_end().to_string())
}

pub async fn get_wallet_context(
    client_config: &SomaEnvConfig,
) -> Result<WalletContext, anyhow::Error> {
    let wallet_conf_file =
        client_config.config.clone().unwrap_or(soma_config_dir()?.join(SOMA_CLIENT_CONFIG));

    prompt_if_no_config(&wallet_conf_file, client_config.accept_defaults).await?;
    let mut context = WalletContext::new(&wallet_conf_file)?;

    if let Some(env_override) = &client_config.env {
        context = context.with_env_override(env_override.clone());
    }

    Ok(context)
}

fn socket_addr_to_url(addr: SocketAddr) -> Result<Url, anyhow::Error> {
    let ip = normalize_bind_addr(addr);
    Url::parse(&format!("http://{ip}:{}", addr.port()))
        .with_context(|| format!("Failed to parse {addr} into a Url"))
}

fn normalize_bind_addr(addr: SocketAddr) -> IpAddr {
    match addr.ip() {
        IpAddr::V4(v4) if v4.is_unspecified() => IpAddr::V4(Ipv4Addr::LOCALHOST),
        IpAddr::V6(v6) if v6.is_unspecified() => IpAddr::V6(Ipv6Addr::LOCALHOST),
        ip => ip,
    }
}

fn update_wallet_config_rpc(
    config_dir: PathBuf,
    fullnode_rpc_url: String,
) -> anyhow::Result<WalletContext, anyhow::Error> {
    let mut wallet_context = create_wallet_context(DEFAULT_WALLET_TIMEOUT_SEC, config_dir.clone())?;
    if let Some(env) = wallet_context.config.envs.iter_mut().find(|env| env.alias == "localnet") {
        env.rpc = fullnode_rpc_url;
    }
    wallet_context.config.save()?;

    Ok(wallet_context)
}
