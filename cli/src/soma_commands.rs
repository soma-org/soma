use crate::{
    client_commands::SomaClientCommands,
    commands::{
        ChallengeCommand, ClaimCommand, DataCommand, EnvCommand, ModelCommand, ObjectsCommand,
        SomaValidatorCommand, SubmitCommand, TargetCommand, WalletCommand,
    },
    keytool::KeyToolCommand,
};
use anyhow::{Context as _, anyhow, bail, ensure};
use clap::{Command, CommandFactory as _, Parser};
use colored::Colorize;
use fastcrypto::traits::KeyPair as _;
use rand::rngs::OsRng;
use sdk::{
    SomaClient,
    client_config::{SomaClientConfig, SomaEnv},
    wallet_context::{DEFAULT_WALLET_TIMEOUT_SEC, WalletContext, create_wallet_context},
};
use soma_keys::{
    key_derive::generate_new_key,
    key_identity::KeyIdentity,
    keystore::{AccountKeystore as _, FileBasedKeystore, Keystore},
};
use std::{
    fs,
    io::{self, Write as _, stdout},
    net::{AddrParseError, IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr},
    path::{Path, PathBuf},
};
use std::{num::NonZeroUsize, time::Duration};
use test_cluster::swarm::Swarm;
use tokio::time::interval;
use tracing::info;
use types::{
    base::SomaAddress,
    config::{
        FULL_NODE_DB_PATH, PersistedConfig, SOMA_CLIENT_CONFIG, SOMA_FULLNODE_CONFIG,
        SOMA_KEYSTORE_FILENAME, SOMA_NETWORK_CONFIG, genesis_blob_exists,
        genesis_config::{GenesisConfig, ValidatorGenesisConfigBuilder},
        network_config::NetworkConfig,
        node_config::{FullnodeConfigBuilder, Genesis, default_json_rpc_address},
        p2p_config::SeedPeer,
        soma_config_dir,
    },
    crypto::SignatureScheme,
    digests::TransactionDigest,
    object::ObjectID,
    peer_id::PeerId,
};
use types::{
    config::{Config, SOMA_GENESIS_FILENAME, network_config::ConfigBuilder},
    crypto::SomaKeyPair,
};
use url::Url;

use crate::client_commands::TxProcessingArgs;
use crate::commands;

const DEFAULT_EPOCH_DURATION_MS: u64 = 60_000;

#[derive(Parser)]
#[clap(rename_all = "kebab-case")]
pub struct SomaEnvConfig {
    /// Sets the file storing the state of our user accounts (an empty one will be created if missing)
    #[clap(long = "client.config")]
    config: Option<PathBuf>,
    /// The Soma environment to use. This must be present in the current config file.
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

impl Default for SomaEnvConfig {
    fn default() -> Self {
        Self { config: None, env: None, accept_defaults: false }
    }
}

#[allow(clippy::large_enum_variant)]
#[derive(Parser)]
#[clap(rename_all = "kebab-case")]
pub enum SomaCommand {
    // =========================================================================
    // COMMON USER ACTIONS (Top-level for convenience)
    // =========================================================================
    /// Check SOMA balance for an address
    #[clap(name = "balance")]
    Balance {
        /// Address to check (defaults to active address)
        address: Option<KeyIdentity>,
        /// Show individual coin details
        #[clap(long)]
        with_coins: bool,
        #[clap(long, global = true)]
        json: bool,
    },

    /// Send SOMA to a recipient
    #[clap(name = "send")]
    Send {
        /// Recipient address or alias
        #[clap(long)]
        to: KeyIdentity,
        /// Amount to send in shannons
        #[clap(long)]
        amount: u64,
        /// Specific coin to send from (auto-selected if not provided)
        #[clap(long)]
        coin: Option<ObjectID>,
        #[clap(flatten)]
        tx_args: TxProcessingArgs,
        #[clap(long, global = true)]
        json: bool,
    },

    /// Transfer an object to a recipient
    #[clap(name = "transfer")]
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
        #[clap(long, global = true)]
        json: bool,
    },

    /// Pay SOMA to multiple recipients
    #[clap(name = "pay")]
    Pay {
        /// Recipient addresses
        #[clap(long, num_args(1..))]
        recipients: Vec<KeyIdentity>,
        /// Amounts to send to each recipient (in shannons)
        #[clap(long, num_args(1..))]
        amounts: Vec<u64>,
        /// Input coin object IDs (auto-selected if not provided)
        #[clap(long, num_args(1..))]
        coins: Option<Vec<ObjectID>>,
        #[clap(flatten)]
        tx_args: TxProcessingArgs,
        #[clap(long, global = true)]
        json: bool,
    },

    /// Stake SOMA with a validator or model
    #[clap(name = "stake")]
    Stake {
        /// Validator address to stake with
        #[clap(long, group = "stake_target")]
        validator: Option<SomaAddress>,
        /// Model ID to stake with
        #[clap(long, group = "stake_target")]
        model: Option<ObjectID>,
        /// Amount to stake (uses entire coin if not specified)
        #[clap(long)]
        amount: Option<u64>,
        /// Coin to use for staking (auto-selected if not provided)
        #[clap(long)]
        coin: Option<ObjectID>,
        #[clap(flatten)]
        tx_args: TxProcessingArgs,
        #[clap(long, global = true)]
        json: bool,
    },

    /// Withdraw staked SOMA
    #[clap(name = "unstake")]
    Unstake {
        /// StakedSoma object ID to withdraw
        staked_soma_id: ObjectID,
        #[clap(flatten)]
        tx_args: TxProcessingArgs,
        #[clap(long, global = true)]
        json: bool,
    },

    // =========================================================================
    // QUERY COMMANDS
    // =========================================================================
    /// Query objects
    #[clap(name = "objects")]
    Objects {
        #[clap(subcommand)]
        cmd: ObjectsCommand,
        #[clap(long, global = true)]
        json: bool,
    },

    /// Get transaction details by digest
    #[clap(name = "tx")]
    Tx {
        /// Transaction digest
        digest: TransactionDigest,
        #[clap(long, global = true)]
        json: bool,
    },

    // =========================================================================
    // MANAGEMENT COMMANDS
    // =========================================================================
    /// Manage wallet addresses and keys
    #[clap(name = "wallet")]
    Wallet {
        #[clap(subcommand)]
        cmd: WalletCommand,
        #[clap(long, global = true)]
        json: bool,
    },

    /// Manage network environments
    #[clap(name = "env")]
    Env {
        #[clap(subcommand)]
        cmd: EnvCommand,
        #[clap(long, global = true)]
        json: bool,
    },

    // =========================================================================
    // OPERATOR COMMANDS
    // =========================================================================
    /// A tool for validators and validator candidates
    #[clap(name = "validator")]
    Validator {
        #[clap(flatten)]
        config: SomaEnvConfig,
        #[clap(subcommand)]
        cmd: Option<SomaValidatorCommand>,
        #[clap(long, global = true)]
        json: bool,
    },

    /// Manage models (commit, reveal, update, deactivate, query)
    #[clap(name = "model")]
    Model {
        #[clap(subcommand)]
        cmd: ModelCommand,
        #[clap(long, global = true)]
        json: bool,
    },

    /// Query targets in the mining competition
    #[clap(name = "target")]
    Target {
        #[clap(subcommand)]
        cmd: TargetCommand,
        #[clap(long, global = true)]
        json: bool,
    },

    /// Download submission data from filled targets
    ///
    /// Fetches the winning submission's data via the validator proxy network.
    #[clap(name = "data")]
    Data {
        #[clap(subcommand)]
        cmd: DataCommand,
        #[clap(long, global = true)]
        json: bool,
    },

    /// Submit data to fill a target
    #[clap(name = "submit")]
    Submit {
        #[clap(flatten)]
        cmd: SubmitCommand,
        #[clap(long, global = true)]
        json: bool,
    },

    /// Claim rewards from a filled target
    ///
    /// Claims the reward pool from a target that was successfully filled.
    /// The challenge window must have closed (fill_epoch + 1 must have ended).
    #[clap(name = "claim")]
    Claim {
        #[clap(flatten)]
        cmd: ClaimCommand,
        #[clap(long, global = true)]
        json: bool,
    },

    /// Challenge a filled target's submission
    ///
    /// Initiates a fraud challenge against a miner's submission for a filled target.
    /// Requires a bond proportional to the data size.
    #[clap(name = "challenge")]
    Challenge {
        #[clap(subcommand)]
        cmd: ChallengeCommand,
        #[clap(long, global = true)]
        json: bool,
    },

    // =========================================================================
    // ADVANCED CLIENT OPERATIONS (backward compatibility)
    // =========================================================================
    /// Advanced client operations
    #[clap(name = "client")]
    Client {
        #[clap(flatten)]
        config: SomaEnvConfig,
        #[clap(subcommand)]
        cmd: Option<SomaClientCommands>,
        #[clap(long, global = true)]
        json: bool,
    },

    // =========================================================================
    // NODE OPERATIONS
    // =========================================================================
    /// Start a local network in two modes: saving state between re-runs and not saving state
    /// between re-runs. Please use (--help) to see the full description.
    ///
    /// This is for development and testing. To run a single validator node
    /// that connects to an existing network, use `soma validator start`.
    #[clap(name = "start", verbatim_doc_comment)]
    Start {
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
    },

    #[clap(name = "network")]
    Network {
        #[clap(long = "network.config")]
        config: Option<PathBuf>,
        #[clap(short, long, help = "Dump the public keys of all authorities")]
        dump_addresses: bool,
    },

    /// Bootstrap and initialize a new soma network
    #[clap(name = "genesis")]
    Genesis {
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

    GenesisCeremony(crate::genesis_ceremony::Ceremony),

    /// Soma keystore tool.
    #[clap(name = "keytool")]
    KeyTool {
        #[clap(long)]
        keystore_path: Option<PathBuf>,
        #[clap(long, global = true)]
        json: bool,
        #[clap(subcommand)]
        cmd: KeyToolCommand,
    },
}

impl SomaCommand {
    pub async fn execute(self) -> Result<(), anyhow::Error> {
        match self {
            // =================================================================
            // COMMON USER ACTIONS
            // =================================================================
            SomaCommand::Balance { address, with_coins, json } => {
                let context = get_wallet_context(&SomaEnvConfig::default()).await?;
                let result = commands::balance::execute(&context, address, with_coins).await?;
                result.print(!json);
                Ok(())
            }

            SomaCommand::Send { to, amount, coin, tx_args, json } => {
                let mut context = get_wallet_context(&SomaEnvConfig::default()).await?;
                let result =
                    commands::send::execute(&mut context, to, amount, coin, tx_args).await?;
                result.print(!json);
                Ok(())
            }

            SomaCommand::Transfer { to, object_id, gas, tx_args, json } => {
                let mut context = get_wallet_context(&SomaEnvConfig::default()).await?;
                let result =
                    commands::transfer::execute(&mut context, to, object_id, gas, tx_args).await?;
                result.print(!json);
                Ok(())
            }

            SomaCommand::Pay { recipients, amounts, coins, tx_args, json } => {
                let mut context = get_wallet_context(&SomaEnvConfig::default()).await?;
                let result =
                    commands::pay::execute(&mut context, recipients, amounts, coins, tx_args)
                        .await?;
                result.print(!json);
                Ok(())
            }

            SomaCommand::Stake { validator, model, amount, coin, tx_args, json } => {
                let mut context = get_wallet_context(&SomaEnvConfig::default()).await?;
                let result = commands::stake::execute_stake(
                    &mut context,
                    validator,
                    model,
                    amount,
                    coin,
                    tx_args,
                )
                .await?;
                result.print(!json);
                Ok(())
            }

            SomaCommand::Unstake { staked_soma_id, tx_args, json } => {
                let mut context = get_wallet_context(&SomaEnvConfig::default()).await?;
                let result =
                    commands::stake::execute_unstake(&mut context, staked_soma_id, tx_args).await?;
                result.print(!json);
                Ok(())
            }

            // =================================================================
            // QUERY COMMANDS
            // =================================================================
            SomaCommand::Objects { cmd, json } => {
                let mut context = get_wallet_context(&SomaEnvConfig::default()).await?;
                let result = commands::objects::execute(&mut context, cmd).await?;
                result.print(!json);
                Ok(())
            }

            SomaCommand::Tx { digest, json } => {
                let context = get_wallet_context(&SomaEnvConfig::default()).await?;
                let result = commands::tx::execute(&context, digest).await?;
                result.print(!json);
                Ok(())
            }

            // =================================================================
            // MANAGEMENT COMMANDS
            // =================================================================
            SomaCommand::Wallet { cmd, json } => {
                let mut context = get_wallet_context(&SomaEnvConfig::default()).await?;
                let result = commands::wallet::execute(&mut context, cmd).await?;
                result.print(!json);
                Ok(())
            }

            SomaCommand::Env { cmd, json } => {
                let mut context = get_wallet_context(&SomaEnvConfig::default()).await?;
                let result = commands::env::execute(&mut context, cmd).await?;
                result.print(!json);
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
                    cmd.execute(&mut context).await?.print(!json);
                } else {
                    let mut app: Command = SomaCommand::command();
                    app.build();
                    app.find_subcommand_mut("validator").unwrap().print_help()?;
                }
                Ok(())
            }

            SomaCommand::Model { cmd, json } => {
                let mut context = get_wallet_context(&SomaEnvConfig::default()).await?;
                cmd.execute(&mut context).await?.print(!json);
                Ok(())
            }

            SomaCommand::Target { cmd, json } => {
                let mut context = get_wallet_context(&SomaEnvConfig::default()).await?;
                cmd.execute(&mut context).await?.print(!json);
                Ok(())
            }

            SomaCommand::Data { cmd, json } => {
                let mut context = get_wallet_context(&SomaEnvConfig::default()).await?;
                cmd.execute(&mut context).await?.print(json);
                Ok(())
            }

            SomaCommand::Submit { cmd, json } => {
                let mut context = get_wallet_context(&SomaEnvConfig::default()).await?;
                cmd.execute(&mut context).await?.print(!json);
                Ok(())
            }

            SomaCommand::Claim { cmd, json } => {
                let mut context = get_wallet_context(&SomaEnvConfig::default()).await?;
                cmd.execute(&mut context).await?.print(json);
                Ok(())
            }

            SomaCommand::Challenge { cmd, json } => {
                let mut context = get_wallet_context(&SomaEnvConfig::default()).await?;
                cmd.execute(&mut context).await?.print(json);
                Ok(())
            }

            // =================================================================
            // ADVANCED CLIENT OPERATIONS
            // =================================================================
            SomaCommand::Client { config, cmd, json } => {
                if let Some(cmd) = cmd {
                    let mut context = get_wallet_context(&config).await?;

                    if let Ok(client) = context.get_client().await {
                        if let Err(e) = client.check_api_version().await {
                            eprintln!("{}", format!("[warning] {e}").yellow().bold());
                        }
                    }
                    cmd.execute(&mut context).await?.print(!json);
                } else {
                    let mut app: Command = SomaCommand::command();
                    app.build();
                    app.find_subcommand_mut("client").unwrap().print_help()?;
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
                        "Cannot open Soma network config file at {:?}",
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

            SomaCommand::Start {
                config_dir,
                force_regenesis,
                fullnode_rpc_port,
                data_ingestion_dir,
                no_full_node,
                epoch_duration_ms,
                committee_size,
            } => {
                start(
                    config_dir.clone(),
                    force_regenesis,
                    epoch_duration_ms,
                    fullnode_rpc_port,
                    data_ingestion_dir,
                    no_full_node,
                    committee_size,
                )
                .await?;
                Ok(())
            }

            SomaCommand::Genesis {
                working_dir,
                force,
                from_config,
                write_config,
                epoch_duration_ms,
                with_faucet,
                committee_size,
            } => {
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

            SomaCommand::GenesisCeremony(cmd) => crate::genesis_ceremony::run(cmd),

            SomaCommand::KeyTool { keystore_path, json, cmd } => {
                let keystore_path =
                    keystore_path.unwrap_or(soma_config_dir()?.join(SOMA_KEYSTORE_FILENAME));
                let mut keystore =
                    Keystore::from(FileBasedKeystore::load_or_create(&keystore_path)?);
                cmd.execute(&mut keystore).await?.print(!json);
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
            there is no genesis configuration in the default Soma configuration folder or the given \
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
        let genesis_config = GenesisConfig::custom_genesis(1, 100);
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
                                "Cannot run genesis with non-empty Soma config directory: {}.\n\n\
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
                    "Cannot open Soma network config file at {:?}",
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

    let mut swarm = swarm_builder.build();
    swarm.launch().await?;
    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
    info!("Cluster started");

    let fullnode_rpc_url =
        socket_addr_to_url(fullnode_rpc_address)?.to_string().trim_end_matches("/").to_string();
    info!("Fullnode RPC URL: {fullnode_rpc_url}");

    if config_dir.join(SOMA_CLIENT_CONFIG).exists() {
        let _ = update_wallet_config_rpc(config_dir.clone(), fullnode_rpc_url.clone())?;
    }

    if force_regenesis && soma_config_dir()?.join(SOMA_CLIENT_CONFIG).exists() {
        let _ = update_wallet_config_rpc(soma_config_dir()?, fullnode_rpc_url.clone())?;
    }

    let mut interval = interval(Duration::from_secs(3));

    loop {
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {
                info!("Received Ctrl+C, shutting down...");
                break;
            }
            _ = interval.tick() => {}
        }
    }
    Ok(())
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
        anyhow!(err).context(format!("Cannot open Soma config dir {:?}", soma_config_dir))
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
                        .context(format!("Cannot remove Soma config dir {:?}", soma_config_dir))
                })?;
                fs::create_dir(soma_config_dir).map_err(|err| {
                    anyhow!(err)
                        .context(format!("Cannot create Soma config dir {:?}", soma_config_dir))
                })?;
            }
        } else if files.len() != 2 || !client_path.exists() || !keystore_path.exists() {
            bail!(
                "Cannot run genesis with non-empty Soma config directory {}, please use the --force/-f option to remove the existing configuration",
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
    client_config.add_env(SomaEnv::devnet());

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
        envs: vec![default_env, SomaEnv::mainnet(), SomaEnv::devnet(), SomaEnv::localnet()],
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
