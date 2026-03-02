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
    EnvCommand, ModelCommand, ObjectsCommand, SomaValidatorCommand, TargetCommand, WalletCommand,
};
use crate::keytool::KeyToolCommand;
use crate::soma_amount::SomaAmount;

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
    soma balance 0x1234...5678
    soma balance --with-coins"
    )]
    Balance {
        /// Address to check (defaults to active address)
        address: Option<KeyIdentity>,
        /// Show individual coin details
        #[clap(long)]
        with_coins: bool,
        #[clap(long, global = true, help = "Output as JSON")]
        json: bool,
    },

    /// Send SOMA to a recipient
    #[clap(
        name = "send",
        after_help = "\
EXAMPLES:
    soma send --to 0x1234...5678 --amount 1
    soma send --to my-alias --amount 0.5 --coin 0xABCD..."
    )]
    Send {
        /// Recipient address or alias
        #[clap(long)]
        to: KeyIdentity,
        /// Amount to send in SOMA (e.g., 1 or 0.5)
        #[clap(long)]
        amount: SomaAmount,
        /// Specific coin to send from (auto-selected if not provided)
        #[clap(long)]
        coin: Option<ObjectID>,
        #[clap(flatten)]
        tx_args: TxProcessingArgs,
        #[clap(long, global = true, help = "Output as JSON")]
        json: bool,
    },

    /// Transfer an object to a recipient
    #[clap(
        name = "transfer",
        after_help = "\
EXAMPLES:
    soma transfer --to 0x1234...5678 --object-id 0xABCD...
    soma transfer --to my-alias --object-id 0xABCD... --gas 0xGAS..."
    )]
    Transfer {
        /// Recipient address or alias
        #[clap(long)]
        to: KeyIdentity,
        /// Object ID to transfer
        #[clap(long)]
        object_id: ObjectID,
        /// Gas object (auto-selected if not provided)
        #[clap(long)]
        gas: Option<ObjectID>,
        #[clap(flatten)]
        tx_args: TxProcessingArgs,
        #[clap(long, global = true, help = "Output as JSON")]
        json: bool,
    },

    /// Pay SOMA to multiple recipients
    #[clap(
        name = "pay",
        after_help = "\
EXAMPLES:
    soma pay --recipients 0xABC... --amounts 1
    soma pay --recipients 0xABC... 0xDEF... --amounts 1 0.5
    soma pay --recipients 0xABC... 0xDEF... --amounts 1 2 --coins 0xCOIN..."
    )]
    Pay {
        /// Recipient addresses
        #[clap(long, required = true, num_args(1..))]
        recipients: Vec<KeyIdentity>,
        /// Amounts to send to each recipient (in SOMA, e.g., 1 or 0.5)
        #[clap(long, required = true, num_args(1..))]
        amounts: Vec<SomaAmount>,
        /// Input coin object IDs (auto-selected if not provided)
        #[clap(long, num_args(1..))]
        coins: Option<Vec<ObjectID>>,
        #[clap(flatten)]
        tx_args: TxProcessingArgs,
        #[clap(long, global = true, help = "Output as JSON")]
        json: bool,
    },

    /// Stake SOMA with a validator or model
    #[clap(
        name = "stake",
        after_help = "\
EXAMPLES:
    soma stake --validator 0xVAL... --amount 10
    soma stake --model 0xMODEL... --amount 5
    soma stake --validator 0xVAL... --coin 0xCOIN..."
    )]
    Stake {
        /// Validator address to stake with
        #[clap(long, group = "stake_target", required_unless_present = "model")]
        validator: Option<SomaAddress>,
        /// Model ID to stake with
        #[clap(long, group = "stake_target", required_unless_present = "validator")]
        model: Option<ObjectID>,
        /// Amount to stake in SOMA (uses entire coin if not specified)
        #[clap(long)]
        amount: Option<SomaAmount>,
        /// Coin to use for staking (auto-selected if not provided)
        #[clap(long)]
        coin: Option<ObjectID>,
        #[clap(flatten)]
        tx_args: TxProcessingArgs,
        #[clap(long, global = true, help = "Output as JSON")]
        json: bool,
    },

    /// Withdraw staked SOMA
    #[clap(
        name = "unstake",
        after_help = "\
EXAMPLES:
    soma unstake 0xSTAKED_SOMA_ID"
    )]
    Unstake {
        /// StakedSoma object ID to withdraw
        staked_soma_id: ObjectID,
        #[clap(flatten)]
        tx_args: TxProcessingArgs,
        #[clap(long, global = true, help = "Output as JSON")]
        json: bool,
    },

    /// Request test tokens from a faucet server
    #[clap(
        name = "faucet",
        after_help = "\
EXAMPLES:
    soma faucet
    soma faucet --address 0x1234...5678
    soma faucet --url http://127.0.0.1:9123"
    )]
    Faucet {
        /// Address to receive tokens (defaults to active address)
        #[clap(long)]
        address: Option<soma_keys::key_identity::KeyIdentity>,
        /// The URL of the faucet server
        #[clap(long)]
        url: Option<String>,
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
    /// Query on-chain objects by owner or ID
    #[clap(
        name = "objects",
        after_help = "\
EXAMPLES:
    soma objects list
    soma objects get 0xOBJECT_ID"
    )]
    Objects {
        #[clap(subcommand)]
        cmd: ObjectsCommand,
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

    // =========================================================================
    // SUBMISSION COMMANDS
    // =========================================================================
    /// Manage models (commit, reveal, update, deactivate, query)
    #[clap(
        name = "model",
        after_help = "\
EXAMPLES:
    soma model list
    soma model info 0xMODEL_ID
    soma model commit 0xMODEL_ID --weights-url-commitment 0xHEX... \\
        --weights-commitment 0xHEX... --architecture-version 1 \\
        --stake-amount 100 --staking-pool-id 0xPOOL_ID
    soma model reveal 0xMODEL_ID --weights-url https://... \\
        --weights-checksum 0xHEX... --weights-size 1024 \\
        --decryption-key 0xHEX... --embedding 0.1,0.2,0.3
    soma model deactivate 0xMODEL_ID
    soma model download 0xMODEL_ID --output ./weights.bin"
    )]
    Model {
        #[clap(subcommand)]
        cmd: ModelCommand,
        #[clap(long, global = true, help = "Output as JSON")]
        json: bool,
    },

    /// Manage targets: query, submit data, claim rewards, download data
    #[clap(
        name = "target",
        after_help = "\
EXAMPLES:
    soma target list
    soma target list --status open
    soma target info 0xTARGET_ID
    soma target submit --target-id 0xTARGET... --data-commitment 0xHEX... ...
    soma target claim 0xTARGET_ID
    soma target download 0xTARGET_ID"
    )]
    Target {
        #[clap(subcommand)]
        cmd: TargetCommand,
        #[clap(long, global = true, help = "Output as JSON")]
        json: bool,
    },

    // =========================================================================
    // OPERATOR COMMANDS
    // =========================================================================
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
    /// Start a long-running service (localnet, validator, faucet, scoring)
    #[clap(
        name = "start",
        after_help = "\
EXAMPLES:
    soma start localnet --force-regenesis --small-model
    soma start validator --config validator.yaml
    soma start faucet --port 9123
    soma start scoring --port 9124"
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
        #[clap(long, help = "Creates an extra faucet configuration for soma persisted runs.")]
        with_faucet: bool,
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
    /// Launches local validators, a fullnode, and optionally a faucet and scoring service.
    /// State is persisted in ~/.soma/ by default, or use --force-regenesis
    /// for an ephemeral network that starts fresh each time.
    #[clap(
        name = "localnet",
        after_help = "\
EXAMPLES:
    soma start localnet --force-regenesis --small-model
    soma start localnet --force-regenesis
    soma start localnet --no-faucet --no-scoring"
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

        /// Directory for data ingestion.
        #[clap(long, value_name = "DATA_INGESTION_DIR")]
        data_ingestion_dir: Option<PathBuf>,

        /// Start the network without a fullnode
        #[clap(long = "no-full-node")]
        no_full_node: bool,

        /// Set the number of validators in the network.
        #[clap(long)]
        committee_size: Option<usize>,

        /// Log level for CLI output (trace, debug, info, warn, error).
        #[clap(long, default_value = "info")]
        log_level: String,

        /// Faucet host:port (default: 0.0.0.0:9123). Use --no-faucet to disable.
        #[clap(
            long,
            default_missing_value = "0.0.0.0:9123",
            num_args = 0..=1,
            require_equals = true,
            value_name = "FAUCET_HOST_PORT",
        )]
        with_faucet: Option<String>,

        /// Disable the faucet server.
        #[clap(long)]
        no_faucet: bool,

        /// Disable the scoring server.
        #[clap(long)]
        no_scoring: bool,

        /// Use small model config for the scoring server (embedding_dim=16, num_layers=2).
        #[clap(long)]
        small_model: bool,
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

    /// Start a standalone faucet gRPC server
    #[clap(
        name = "faucet",
        after_help = "\
EXAMPLES:
    soma start faucet
    soma start faucet --port 9999 --host 0.0.0.0
    soma start faucet --amount 5000000000 --num-coins 2"
    )]
    Faucet {
        /// Port to listen on
        #[clap(long, default_value_t = faucet::faucet_config::DEFAULT_FAUCET_PORT)]
        port: u16,
        /// Host IP to bind to
        #[clap(long, default_value = "0.0.0.0")]
        host: String,
        /// Amount of shannons to send per coin
        #[clap(long, default_value_t = faucet::faucet_config::DEFAULT_AMOUNT)]
        amount: u64,
        /// Number of coins to send per request
        #[clap(long, default_value_t = faucet::faucet_config::DEFAULT_NUM_COINS)]
        num_coins: usize,
        /// Path to the client config directory
        #[clap(long)]
        config_dir: Option<PathBuf>,
    },

    /// Start the scoring service for computing embeddings and distances
    ///
    /// Runs an HTTP server that accepts scoring requests via POST /score.
    #[clap(
        name = "scoring",
        after_help = "\
EXAMPLES:
    soma start scoring
    soma start scoring --host 127.0.0.1 --port 8080
    soma start scoring --device cuda
    soma start scoring --small-model
    soma start scoring --data-dir /tmp/soma-data"
    )]
    Scoring {
        /// Host to bind the scoring service to
        #[clap(long, default_value = "0.0.0.0")]
        host: String,

        /// Port to listen on
        #[clap(long, default_value_t = 9124)]
        port: u16,

        /// Directory for cached blob storage (data and model weights)
        #[clap(long)]
        data_dir: Option<PathBuf>,

        /// Use a small model for testing (embedding_dim=16, num_layers=2)
        #[clap(long)]
        small_model: bool,

        /// Compute device backend: cpu, wgpu, or cuda (default: wgpu).
        /// CUDA requires the NVIDIA CUDA toolkit to be installed.
        #[clap(long, default_value = "wgpu")]
        device: String,
    },
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
}

impl SomaCommand {
    pub fn log_level(&self) -> tracing::Level {
        match self {
            SomaCommand::Start { cmd } => match cmd {
                StartCommand::Localnet { log_level, .. } => {
                    log_level.parse().unwrap_or(tracing::Level::INFO)
                }
                StartCommand::Scoring { .. } => tracing::Level::INFO,
                StartCommand::Validator { .. } => tracing::Level::INFO,
                StartCommand::Faucet { .. } => tracing::Level::INFO,
            },
            _ => tracing::Level::ERROR,
        }
    }

    pub async fn execute(self) -> Result<(), anyhow::Error> {
        match self {
            // =================================================================
            // COMMON USER ACTIONS
            // =================================================================
            SomaCommand::Balance { address, with_coins, json } => {
                let context = get_wallet_context(&SomaEnvConfig::default()).await?;
                let result = commands::balance::execute(&context, address, with_coins).await?;
                result.print(json);
                Ok(())
            }

            SomaCommand::Send { to, amount, coin, tx_args, json } => {
                ensure!(amount.shannons() > 0, "Amount must be greater than 0");
                let mut context = get_wallet_context(&SomaEnvConfig::default()).await?;
                let result =
                    commands::send::execute(&mut context, to, amount.shannons(), coin, tx_args)
                        .await?;
                result.print(json);
                if result.has_failed_transaction() {
                    std::process::exit(1);
                }
                Ok(())
            }

            SomaCommand::Transfer { to, object_id, gas, tx_args, json } => {
                let mut context = get_wallet_context(&SomaEnvConfig::default()).await?;
                let result =
                    commands::transfer::execute(&mut context, to, object_id, gas, tx_args).await?;
                result.print(json);
                if result.has_failed_transaction() {
                    std::process::exit(1);
                }
                Ok(())
            }

            SomaCommand::Pay { recipients, amounts, coins, tx_args, json } => {
                ensure!(
                    recipients.len() == amounts.len(),
                    "Number of recipients ({}) must match number of amounts ({})",
                    recipients.len(),
                    amounts.len()
                );
                for (i, a) in amounts.iter().enumerate() {
                    ensure!(a.shannons() > 0, "Amount {} must be greater than 0", i + 1);
                }
                let mut context = get_wallet_context(&SomaEnvConfig::default()).await?;
                let amounts_shannons: Vec<u64> = amounts.iter().map(|a| a.shannons()).collect();
                let result = commands::pay::execute(
                    &mut context,
                    recipients,
                    amounts_shannons,
                    coins,
                    tx_args,
                )
                .await?;
                result.print(json);
                if result.has_failed_transaction() {
                    std::process::exit(1);
                }
                Ok(())
            }

            SomaCommand::Stake { validator, model, amount, coin, tx_args, json } => {
                let mut context = get_wallet_context(&SomaEnvConfig::default()).await?;
                let result = commands::stake::execute_stake(
                    &mut context,
                    validator,
                    model,
                    amount.map(|a| a.shannons()),
                    coin,
                    tx_args,
                )
                .await?;
                result.print(json);
                if result.has_failed_transaction() {
                    std::process::exit(1);
                }
                Ok(())
            }

            SomaCommand::Unstake { staked_soma_id, tx_args, json } => {
                let mut context = get_wallet_context(&SomaEnvConfig::default()).await?;
                let result =
                    commands::stake::execute_unstake(&mut context, staked_soma_id, tx_args).await?;
                result.print(json);
                if result.has_failed_transaction() {
                    std::process::exit(1);
                }
                Ok(())
            }

            SomaCommand::Faucet { address, url, json: _ } => {
                let mut context = get_wallet_context(&SomaEnvConfig::default()).await?;
                commands::faucet::execute_request(&mut context, address, url).await?;
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
                    balance,
                    server_unreachable,
                ) = match context.get_client().await {
                    Ok(client) => {
                        let chain_id = client.get_chain_identifier().await.ok();
                        let server_version = client.get_server_version().await.ok();
                        let state = client.get_latest_system_state().await.ok();
                        let epoch = state.as_ref().map(|s| s.epoch());
                        let epoch_start_ms = state.as_ref().map(|s| s.epoch_start_timestamp_ms());
                        let epoch_dur_ms = state.as_ref().map(|s| s.epoch_duration_ms());
                        let balance = if let Some(addr) = &active_address {
                            client.get_balance(addr).await.ok()
                        } else {
                            None
                        };
                        let unreachable =
                            server_version.is_none() && chain_id.is_none() && epoch.is_none();
                        (
                            server_version,
                            chain_id,
                            epoch,
                            epoch_start_ms,
                            epoch_dur_ms,
                            balance,
                            unreachable,
                        )
                    }
                    Err(_) => (None, None, None, None, None, None, true),
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
                    active_address: active_address.map(|a| a.to_string()),
                    balance,
                    server_reachable: !server_unreachable,
                };
                output.print(json);
                Ok(())
            }

            // =================================================================
            // QUERY COMMANDS
            // =================================================================
            SomaCommand::Objects { cmd, json } => {
                let mut context = get_wallet_context(&SomaEnvConfig::default()).await?;
                let result = commands::objects::execute(&mut context, cmd).await?;
                result.print(json);
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

            SomaCommand::Model { cmd, json } => {
                let mut context = get_wallet_context(&SomaEnvConfig::default()).await?;
                cmd.execute(&mut context).await?.print(json);
                Ok(())
            }

            SomaCommand::Target { cmd, json } => {
                let mut context = get_wallet_context(&SomaEnvConfig::default()).await?;
                cmd.execute(&mut context).await?.print(json);
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
                        data_ingestion_dir,
                        no_full_node,
                        epoch_duration_ms,
                        committee_size,
                        log_level: _,
                        with_faucet,
                        no_faucet,
                        no_scoring,
                        small_model,
                    } => {
                        // Faucet: on by default unless --no-faucet
                        let faucet = if no_faucet {
                            None
                        } else {
                            Some(with_faucet.unwrap_or_else(|| "0.0.0.0:9123".to_string()))
                        };
                        // Scoring: on by default unless --no-scoring
                        let scoring = !no_scoring;
                        start(
                            config_dir.clone(),
                            force_regenesis,
                            epoch_duration_ms,
                            fullnode_rpc_port,
                            data_ingestion_dir,
                            no_full_node,
                            committee_size,
                            faucet,
                            scoring,
                            small_model,
                        )
                        .await?;
                    }
                    StartCommand::Validator { config } => {
                        commands::validator::start_validator_node(config).await?;
                    }
                    StartCommand::Faucet { port, host, amount, num_coins, config_dir } => {
                        commands::faucet::execute_start(port, host, amount, num_coins, config_dir)
                            .await?;
                    }
                    StartCommand::Scoring { host, port, data_dir, small_model, device } => {
                        use types::config::node_config::DeviceConfig;

                        let device = match device.as_str() {
                            "cpu" => DeviceConfig::Cpu,
                            "wgpu" => DeviceConfig::Wgpu,
                            "cuda" => DeviceConfig::Cuda,
                            other => {
                                bail!("Unknown device: {other}. Valid options: cpu, wgpu, cuda")
                            }
                        };

                        let data_dir = data_dir.unwrap_or_else(|| {
                            soma_config_dir()
                                .unwrap_or_else(|_| PathBuf::from("."))
                                .join("scoring-data")
                        });
                        fs::create_dir_all(&data_dir)?;

                        let model_config = if small_model {
                            scoring::model_config_small()
                        } else {
                            runtime::ModelConfig::new()
                        };

                        let engine = std::sync::Arc::new(
                            scoring::scoring::ScoringEngine::new(&data_dir, model_config, &device)
                                .map_err(|e| anyhow!("Failed to create scoring engine: {e}"))?,
                        );

                        print_banner("Scoring Service");

                        let display_host = if host == "0.0.0.0" { "127.0.0.1" } else { &host };
                        let device_str = device.to_string();
                        let url = format!("http://{display_host}:{port}");
                        let score_ep = format!("POST {url}/score");
                        let data_display = data_dir.display().to_string();
                        print_info_panel(&[
                            ("URL", &url),
                            ("Score endpoint", &score_ep),
                            ("Device", &device_str),
                            ("Data dir", &data_display),
                        ]);
                        eprintln!();
                        eprintln!("  Press {} to stop.", "Ctrl+C".bold());

                        scoring::server::start_scoring_server(&host, port, engine)
                            .await
                            .map_err(|e| anyhow!("Scoring server error: {e}"))?;
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
                with_faucet,
                committee_size,
            } => {
                if let Some(GenesisCommand::Ceremony(ceremony)) = cmd {
                    return crate::genesis_ceremony::run(ceremony);
                }
                genesis(
                    from_config,
                    write_config,
                    working_dir,
                    force,
                    epoch_duration_ms,
                    with_faucet,
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
    mut data_ingestion_dir: Option<PathBuf>,
    no_full_node: bool,
    committee_size: Option<usize>,
    with_faucet: Option<String>,
    with_scoring: bool,
    small_model: bool,
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
        let mut genesis_config = if with_faucet.is_some() {
            GenesisConfig::for_local_testing().add_faucet_account()
        } else {
            GenesisConfig::for_local_testing()
        };
        if small_model {
            // Small model uses embedding_dim=16; override protocol config default of 2048
            genesis_config.parameters.target_embedding_dim_override = Some(16);
        }
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
                    genesis(None, None, None, false, epoch_duration_ms, false, committee_size)
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

    if let Some(ref dir) = data_ingestion_dir {
        swarm_builder = swarm_builder.with_data_ingestion_dir(dir.clone());
    }

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

    // -- Scoring service (must start before validators so they can connect) ----
    let mut scoring_url: Option<String> = None;
    if with_scoring {
        use types::config::node_config::DeviceConfig;

        let msg = "Starting scoring service...";
        eprint!("  {msg:<width$}", width = STATUS_WIDTH);

        let model_config =
            if small_model { scoring::model_config_small() } else { runtime::ModelConfig::new() };

        let scoring_data_dir = config_dir.join("scoring-data");
        fs::create_dir_all(&scoring_data_dir)?;
        let device = DeviceConfig::Wgpu;

        let engine = std::sync::Arc::new(
            scoring::scoring::ScoringEngine::new(&scoring_data_dir, model_config, &device)
                .map_err(|e| anyhow!("Failed to create scoring engine: {e}"))?,
        );

        tokio::spawn(async move {
            if let Err(e) = scoring::server::start_scoring_server("0.0.0.0", 9124, engine).await {
                tracing::error!("Scoring server error: {}", e);
            }
        });

        let url = "http://127.0.0.1:9124".to_string();
        swarm_builder = swarm_builder.with_scoring_url(url.clone());
        scoring_url = Some(url);
        eprintln!("{}", "done".green());
    }

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

    // -- Faucet ---------------------------------------------------------------
    // Faucet startup logic derived from MystenLabs/sui (Apache-2.0)
    // See: https://github.com/MystenLabs/sui/blob/main/crates/sui/src/sui_commands.rs
    let mut faucet_url: Option<String> = None;
    if let Some(input) = with_faucet {
        let msg = "Starting faucet...";
        eprint!("  {msg:<width$}", width = STATUS_WIDTH);
        let (host, port) = parse_faucet_host_port(&input)?;

        // Extract the last account key as the faucet key (added by add_faucet_account)
        let faucet_key = if force_regenesis {
            let keys = &swarm.config().account_keys;
            if keys.is_empty() {
                bail!("No account keys found in swarm config for faucet");
            }
            Some(keys.last().expect("account_keys is not empty").copy())
        } else {
            None
        };

        // Set up a wallet context for the faucet
        let keystore_path = tempfile::tempdir()?.keep().join(SOMA_KEYSTORE_FILENAME);
        let mut faucet_keystore = FileBasedKeystore::load_or_create(&keystore_path)?;

        if let Some(key) = faucet_key {
            faucet_keystore.import(None, key).await?;
        } else {
            // For persisted runs, import all keys from the network config
            for key in &swarm.config().account_keys {
                faucet_keystore.import(None, key.copy()).await?;
            }
        }

        let active_address = faucet_keystore.addresses().pop();

        let mut client_config = SomaClientConfig::new(Keystore::from(faucet_keystore));
        client_config.active_address = active_address;
        client_config.add_env(SomaEnv {
            alias: "localnet".to_string(),
            rpc: fullnode_rpc_url.clone(),
            basic_auth: None,
            chain_id: None,
        });
        client_config.active_env = Some("localnet".to_string());

        let faucet_config_path =
            keystore_path.parent().expect("keystore path has a parent").join(SOMA_CLIENT_CONFIG);
        client_config.save(&faucet_config_path)?;

        let wallet_context = create_wallet_context(
            60,
            faucet_config_path.parent().expect("config path has a parent").to_path_buf(),
        )?;

        let faucet_config = faucet::faucet_config::FaucetConfig {
            port,
            host_ip: host.clone(),
            ..Default::default()
        };

        let faucet_instance =
            faucet::local_faucet::LocalFaucet::new(wallet_context, faucet_config.clone())
                .await
                .map_err(|e| anyhow!("Failed to initialize faucet: {e}"))?;

        let app_state = std::sync::Arc::new(faucet::app_state::AppState {
            faucet: std::sync::Arc::new(faucet_instance),
            config: faucet_config,
        });

        tokio::spawn(async move {
            if let Err(e) = faucet::server::start_faucet(app_state).await {
                tracing::error!("Faucet server error: {}", e);
            }
        });

        let display_host = if host == "0.0.0.0" { "127.0.0.1" } else { &host };
        faucet_url = Some(format!("http://{display_host}:{port}/gas"));
        eprintln!("{}", "done".green());
    }

    // -- Admin server (epoch advance) -----------------------------------------
    let swarm = std::sync::Arc::new(swarm);
    {
        let admin_swarm = swarm.clone();
        tokio::spawn(async move {
            let svc = AdminServiceImpl::new(admin_swarm);
            let addr: std::net::SocketAddr = "127.0.0.1:9125".parse().unwrap();
            if let Err(e) = admin::tonic::transport::Server::builder()
                .add_service(admin::admin_gen::admin_server::AdminServer::new(svc))
                .serve(addr)
                .await
            {
                tracing::error!("Admin server error: {}", e);
            }
        });
    }
    let admin_url = "http://127.0.0.1:9125".to_string();

    // -- Network ready banner -------------------------------------------------
    let epoch_ms = epoch_duration_ms.unwrap_or(DEFAULT_EPOCH_DURATION_MS);
    let state_dir = config_dir.display().to_string();
    let persistence = if force_regenesis { "ephemeral" } else { "enabled" };

    eprintln!();
    eprintln!("  {}", "Network ready.".green().bold());
    eprintln!();
    let faucet_display = faucet_url.as_deref().unwrap_or("disabled");
    let scoring_display = scoring_url.as_deref().unwrap_or("disabled");
    let epoch_display = format!("{}s", epoch_ms / 1000);
    let rows: Vec<(&str, &str)> = vec![
        ("RPC URL", &fullnode_rpc_url),
        ("Faucet", faucet_display),
        ("Scoring", scoring_display),
        ("Admin", &admin_url),
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

/// gRPC admin service for epoch advancement on localnet.
struct AdminServiceImpl {
    swarm: std::sync::Arc<Swarm>,
}

impl AdminServiceImpl {
    fn new(swarm: std::sync::Arc<Swarm>) -> Self {
        Self { swarm }
    }
}

#[admin::tonic::async_trait]
impl admin::admin_gen::admin_server::Admin for AdminServiceImpl {
    async fn advance_epoch(
        &self,
        _request: admin::tonic::Request<admin::admin_types::AdvanceEpochRequest>,
    ) -> Result<
        admin::tonic::Response<admin::admin_types::AdvanceEpochResponse>,
        admin::tonic::Status,
    > {
        info!("[admin] advance_epoch called");

        let fullnode = self
            .swarm
            .fullnodes()
            .next()
            .ok_or_else(|| admin::tonic::Status::internal("No fullnode available"))?;

        let fullnode_handle = fullnode
            .get_node_handle()
            .ok_or_else(|| admin::tonic::Status::internal("Fullnode handle unavailable"))?;

        let cur_committee = fullnode_handle.with(|node| node.state().clone_committee_for_testing());
        let target_epoch = cur_committee.epoch + 1;
        info!("[admin] current epoch={}, target={}", cur_committee.epoch, target_epoch);

        // Wait for all validators to reach the current fullnode epoch first.
        let deadline = tokio::time::Instant::now() + Duration::from_secs(120);
        loop {
            let mut all_ready = true;
            for node in self.swarm.validator_nodes() {
                if let Some(handle) = node.get_node_handle() {
                    let v_epoch = handle.with(|n| n.state().epoch_store_for_testing().epoch());
                    if v_epoch < cur_committee.epoch {
                        all_ready = false;
                        break;
                    }
                }
            }
            if all_ready {
                break;
            }
            if tokio::time::Instant::now() > deadline {
                return Err(admin::tonic::Status::deadline_exceeded(
                    "Validators did not reach current epoch",
                ));
            }
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
        info!("[admin] all validators ready");

        // Close epoch on all validator nodes sequentially.
        // Important: do NOT use timeout here — dropping the close_epoch future
        // mid-lock-acquisition aborts it entirely, and the epoch never advances.
        info!("[admin] calling close_epoch_for_testing...");
        for node in self.swarm.validator_nodes() {
            if let Some(handle) = node.get_node_handle() {
                if let Err(e) =
                    handle.with_async(|n| async move { n.close_epoch_for_testing().await }).await
                {
                    tracing::warn!("[admin] close_epoch_for_testing failed: {e}");
                }
            }
        }
        info!("[admin] all close_epoch calls done, polling for epoch {target_epoch}...");

        // Wait for ALL nodes to fully reconfigure.
        loop {
            tokio::time::sleep(Duration::from_millis(200)).await;

            let mut all_ready = true;

            // Check fullnode: epoch_store AND authority aggregator
            if let Some(handle) = fullnode.get_node_handle() {
                let ready = handle.with(|node| {
                    let epoch = node.state().epoch_store_for_testing().epoch();
                    if epoch < target_epoch {
                        return false;
                    }
                    if let Some(agg) = node.clone_authority_aggregator() {
                        agg.committee.epoch() >= target_epoch
                    } else {
                        true
                    }
                });
                if !ready {
                    all_ready = false;
                }
            }

            // Check all validators: epoch_store only
            if all_ready {
                for node in self.swarm.validator_nodes() {
                    if let Some(handle) = node.get_node_handle() {
                        let v_epoch = handle.with(|n| n.state().epoch_store_for_testing().epoch());
                        if v_epoch < target_epoch {
                            all_ready = false;
                            break;
                        }
                    }
                }
            }

            if all_ready {
                // Grace period for consensus startup
                tokio::time::sleep(Duration::from_millis(1000)).await;
                info!("[admin] epoch advanced to {target_epoch}");
                return Ok(admin::tonic::Response::new(admin::admin_types::AdvanceEpochResponse {
                    epoch: target_epoch,
                }));
            }

            if tokio::time::Instant::now() > deadline {
                info!("[admin] TIMEOUT waiting for epoch {target_epoch}");
                return Err(admin::tonic::Status::deadline_exceeded(
                    "Epoch did not advance within 120s",
                ));
            }
        }
    }
}

async fn genesis(
    from_config: Option<PathBuf>,
    write_config: Option<PathBuf>,
    working_dir: Option<PathBuf>,
    force: bool,
    epoch_duration_ms: Option<u64>,
    with_faucet: bool,
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

    if with_faucet {
        info!("Adding faucet account to genesis config...");
        genesis_conf = genesis_conf.add_faucet_account();
    }

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

/// Parse a faucet host:port string like "0.0.0.0:9123" or just a port number.
fn parse_faucet_host_port(input: &str) -> Result<(String, u16), anyhow::Error> {
    if let Ok(port) = input.parse::<u16>() {
        return Ok(("0.0.0.0".to_string(), port));
    }

    if let Ok(addr) = input.parse::<SocketAddr>() {
        return Ok((addr.ip().to_string(), addr.port()));
    }

    // Try host:port format
    if let Some((host, port_str)) = input.rsplit_once(':') {
        let port: u16 =
            port_str.parse().map_err(|_| anyhow!("Invalid port in faucet address: {input}"))?;
        return Ok((host.to_string(), port));
    }

    bail!("Invalid faucet address format: {input}. Expected host:port or just a port number.")
}
