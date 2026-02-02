use anyhow::{Result, anyhow, bail};
use std::{
    collections::{BTreeMap, HashSet},
    fmt::{self, Debug, Display, Formatter, Write},
    fs,
    path::PathBuf,
    sync::Arc,
};
use url::{ParseError, Url};

use clap::*;
use colored::Colorize;
use fastcrypto::traits::ToFromBytes;
use fastcrypto::{
    encoding::{Base64, Encoding},
    traits::KeyPair,
};
use serde::Serialize;
use tap::tap::TapOptional;
use tokio::time::Duration;
use tracing::info;
use types::{
    base::SomaAddress,
    config::node_config::DEFAULT_COMMISSION_RATE,
    crypto::{AuthorityPublicKey, NetworkPublicKey, Signable},
    multiaddr::Multiaddr,
    object::{ObjectID, ObjectRef, Owner},
    system_state::{SystemState, validator::Validator},
    transaction::{
        AddValidatorArgs, RemoveValidatorArgs, TransactionKind, UpdateValidatorMetadataArgs,
    },
};
use types::{
    config::{PersistedConfig, node_config::NodeConfig},
    intent::{Intent, IntentMessage, IntentScope},
    validator_info::GenesisValidatorInfo,
};

use node::SomaNode;
use sdk::SomaClient;
use sdk::{
    transaction_builder::{ExecutionOptions, TransactionBuilder},
    wallet_context::WalletContext,
};
use soma_keys::{
    key_derive::generate_new_key,
    keypair_file::{
        read_authority_keypair_from_file, read_keypair_from_file, read_network_keypair_from_file,
        write_authority_keypair_to_file, write_keypair_to_file,
    },
};
use soma_keys::{keypair_file::read_key, keystore::AccountKeystore};
use types::crypto::{AuthorityKeyPair, NetworkKeyPair, SignatureScheme, SomaKeyPair};
use types::crypto::{AuthorityPublicKeyBytes, get_authority_key_pair};
use types::transaction::{Transaction, TransactionData};

use crate::response::{
    TransactionResponse, ValidatorCommandResponse, ValidatorStatus, ValidatorSummary,
};

/// Arguments related to transaction processing
#[derive(Args, Debug, Default)]
pub struct TxProcessingArgs {
    /// Instead of executing the transaction, serialize the bcs bytes of the unsigned transaction data
    /// (TransactionData) using base64 encoding, and print out the string <TX_BYTES>. The string can
    /// be used to execute transaction with `soma client execute-signed-tx --tx-bytes <TX_BYTES>`.
    #[arg(long)]
    pub serialize_unsigned_transaction: bool,
}

impl From<TxProcessingArgs> for ExecutionOptions {
    fn from(args: TxProcessingArgs) -> Self {
        let mut opts = ExecutionOptions::new();
        if args.serialize_unsigned_transaction {
            opts = opts.serialize_unsigned();
        }
        opts
    }
}

#[derive(Parser)]
#[clap(rename_all = "kebab-case")]
pub enum SomaValidatorCommand {
    /// Start a validator node from a config file
    #[clap(name = "start")]
    Start {
        /// Path to the validator config file (YAML)
        #[clap(long = "config", short = 'c')]
        config: PathBuf,
    },

    /// Generate validator key files and info
    #[clap(name = "make-validator-info")]
    MakeValidatorInfo {
        /// Hostname for the validator (e.g., validator.example.com)
        host_name: String,
        /// Commission rate in basis points (100 = 1%)
        #[clap(default_value_t = DEFAULT_COMMISSION_RATE)]
        commission_rate: u64,
    },

    /// Request to join the validator committee
    #[clap(name = "join-committee")]
    JoinCommittee {
        /// Path to the validator.info file
        #[clap(name = "validator-info-path")]
        file: PathBuf,
        #[clap(flatten)]
        tx_args: TxProcessingArgs,
    },

    /// Request to leave the validator committee
    #[clap(name = "leave-committee")]
    LeaveCommittee {
        #[clap(flatten)]
        tx_args: TxProcessingArgs,
    },

    /// Display validator metadata
    #[clap(name = "display-metadata")]
    DisplayMetadata {
        /// Validator address (defaults to active address)
        #[clap(name = "validator-address")]
        validator_address: Option<SomaAddress>,
        /// Output as JSON
        #[clap(long, default_value_t = false)]
        json: bool,
    },

    /// Update validator metadata
    #[clap(name = "update-metadata")]
    UpdateMetadata {
        #[clap(subcommand)]
        metadata: MetadataUpdate,
        #[clap(flatten)]
        tx_args: TxProcessingArgs,
    },

    /// Set commission rate for the next epoch
    #[clap(name = "set-commission-rate")]
    SetCommissionRate {
        /// Commission rate in basis points (100 = 1%, max 10000 = 100%)
        #[clap(name = "commission-rate")]
        commission_rate: u64,
        #[clap(flatten)]
        tx_args: TxProcessingArgs,
    },

    /// Report or un-report a validator
    #[clap(name = "report-validator")]
    ReportValidator {
        /// The Soma address of the validator being reported
        #[clap(name = "reportee-address")]
        reportee_address: SomaAddress,
        /// If true, undo an existing report
        #[clap(long, default_value_t = false)]
        undo_report: bool,
        #[clap(flatten)]
        tx_args: TxProcessingArgs,
    },
}

#[derive(Subcommand, Clone)]
#[clap(rename_all = "kebab-case")]
pub enum MetadataUpdate {
    /// Update network address (takes effect next epoch)
    NetworkAddress { network_address: Multiaddr },
    /// Update primary address (takes effect next epoch)
    PrimaryAddress { primary_address: Multiaddr },
    /// Update P2P address (takes effect next epoch)
    P2pAddress { p2p_address: Multiaddr },
    /// Update network public key (takes effect next epoch)
    NetworkPubKey {
        #[clap(name = "network-key-path")]
        file: PathBuf,
    },
    /// Update worker public key (takes effect next epoch)
    WorkerPubKey {
        #[clap(name = "worker-key-path")]
        file: PathBuf,
    },
    /// Update protocol public key (takes effect next epoch)
    ProtocolPubKey {
        #[clap(name = "protocol-key-path")]
        file: PathBuf,
    },
}

impl SomaValidatorCommand {
    pub async fn execute(
        self,
        context: &mut WalletContext,
    ) -> Result<ValidatorCommandResponse, anyhow::Error> {
        let sender = context.active_address()?;

        match self {
            SomaValidatorCommand::Start { config } => {
                start_validator_node(config).await?;
                Ok(ValidatorCommandResponse::Started)
            }

            SomaValidatorCommand::MakeValidatorInfo {
                host_name,
                commission_rate,
            } => {
                make_validator_info(context, &host_name, commission_rate)?;
                Ok(ValidatorCommandResponse::MakeValidatorInfo)
            }

            SomaValidatorCommand::JoinCommittee { file, tx_args } => {
                let kind = build_join_committee_tx(&file)?;
                execute_or_serialize(context, sender, kind, tx_args.into()).await
            }

            SomaValidatorCommand::LeaveCommittee { tx_args } => {
                // Verify sender is an active validator before building tx
                check_status(context, HashSet::from([ValidatorStatus::Active])).await?;

                let kind = TransactionKind::RemoveValidator(RemoveValidatorArgs {
                    pubkey_bytes: vec![], // The signer is inferred from tx sender
                });
                execute_or_serialize(context, sender, kind, tx_args.into()).await
            }

            SomaValidatorCommand::DisplayMetadata {
                validator_address,
                json,
            } => {
                let address = validator_address.unwrap_or(sender);
                display_metadata(context, address, json).await?;
                Ok(ValidatorCommandResponse::DisplayMetadata)
            }

            SomaValidatorCommand::UpdateMetadata { metadata, tx_args } => {
                // Verify sender is active or pending
                check_status(
                    context,
                    HashSet::from([ValidatorStatus::Active, ValidatorStatus::Pending]),
                )
                .await?;

                let kind = build_update_metadata_tx(metadata)?;
                execute_or_serialize(context, sender, kind, tx_args.into()).await
            }

            SomaValidatorCommand::SetCommissionRate {
                commission_rate,
                tx_args,
            } => {
                // Verify sender is active or pending
                check_status(
                    context,
                    HashSet::from([ValidatorStatus::Active, ValidatorStatus::Pending]),
                )
                .await?;

                // Validate commission rate (max 10000 = 100%)
                if commission_rate > 10000 {
                    bail!("Commission rate cannot exceed 10000 (100%)");
                }

                let kind = TransactionKind::SetCommissionRate {
                    new_rate: commission_rate,
                };
                execute_or_serialize(context, sender, kind, tx_args.into()).await
            }

            SomaValidatorCommand::ReportValidator {
                reportee_address,
                undo_report,
                tx_args,
            } => {
                // Only active validators can report
                check_status(context, HashSet::from([ValidatorStatus::Active])).await?;

                // Can't report yourself
                if sender == reportee_address {
                    bail!("Cannot report yourself");
                }

                let kind = if undo_report {
                    TransactionKind::UndoReportValidator {
                        reportee: reportee_address,
                    }
                } else {
                    TransactionKind::ReportValidator {
                        reportee: reportee_address,
                    }
                };
                execute_or_serialize(context, sender, kind, tx_args.into()).await
            }
        }
    }
}

/// Start a validator node from a config file
async fn start_validator_node(config_path: PathBuf) -> Result<()> {
    info!("Loading validator config from {:?}", config_path);

    let node_config: NodeConfig = PersistedConfig::read(&config_path).map_err(|err| {
        anyhow!(
            "Cannot open validator config file at {:?}: {}",
            config_path,
            err
        )
    })?;

    info!(
        "Starting validator node with protocol key: {:?}",
        node_config.protocol_public_key()
    );

    // Start the validator node
    let node = SomaNode::start(node_config)
        .await
        .map_err(|err| anyhow!("Failed to start validator node: {}", err))?;

    info!("Validator node started successfully");

    // Keep the node running until Ctrl+C
    let mut interval = tokio::time::interval(Duration::from_secs(5));
    loop {
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {
                info!("Received Ctrl+C, shutting down validator...");
                break;
            }
            _ = interval.tick() => {
                // Health check or status logging could go here
            }
        }
    }

    // Node will be dropped here, triggering graceful shutdown
    drop(node);
    info!("Validator node shut down");

    Ok(())
}

fn check_address(
    active_address: SomaAddress,
    validator_address: Option<SomaAddress>,
    print_unsigned_transaction_only: bool,
) -> Result<SomaAddress, anyhow::Error> {
    if !print_unsigned_transaction_only {
        if let Some(validator_address) = validator_address {
            if validator_address != active_address {
                bail!(
                    "`--validator-address` must be the same as the current active address: {}",
                    active_address
                );
            }
        }
        Ok(active_address)
    } else {
        validator_address
            .ok_or_else(|| anyhow!("--validator-address must be provided when `print_unsigned_transaction_only` is true"))
    }
}

/// Execute a transaction or serialize it for offline signing
async fn execute_or_serialize(
    context: &mut WalletContext,
    sender: SomaAddress,
    kind: TransactionKind,
    options: ExecutionOptions,
) -> Result<ValidatorCommandResponse> {
    let builder = TransactionBuilder::new(context);

    if options.serialize_unsigned {
        let serialized = builder
            .build_serialized_unsigned(sender, kind, options.gas)
            .await?;
        Ok(ValidatorCommandResponse::SerializedTransaction {
            serialized_unsigned_transaction: serialized,
        })
    } else {
        let tx = builder.build_transaction(sender, kind, options.gas).await?;
        drop(builder); // Release the borrow before execute_transaction
        let response = execute_transaction(context, tx).await?;
        Ok(ValidatorCommandResponse::Transaction(response))
    }
}

/// Execute a signed transaction and wait for checkpoint
async fn execute_transaction(
    context: &WalletContext,
    tx: Transaction,
) -> Result<TransactionResponse> {
    // Execute and wait for checkpoint finality
    let response = context.execute_transaction_may_fail(tx).await?;

    Ok(TransactionResponse::from_effects_with_balance_changes(
        &response.effects,
        Some(response.checkpoint_sequence_number),
        response.balance_changes,
    ))
}

/// Create a key file if it doesn't exist
fn make_key_file(path: &PathBuf, is_protocol_key: bool, key: Option<SomaKeyPair>) -> Result<()> {
    if path.exists() {
        println!("Using existing key file: {:?}", path);
        return Ok(());
    }

    if is_protocol_key {
        let (_, keypair) = get_authority_key_pair();
        write_authority_keypair_to_file(&keypair, path)?;
    } else {
        let kp = match key {
            Some(k) => k,
            None => {
                let (_, kp, _, _) = generate_new_key(SignatureScheme::ED25519, None, None)?;
                kp
            }
        };
        write_keypair_to_file(&kp, path)?;
    }

    println!("Generated new key file: {:?}", path);
    Ok(())
}

/// Generate validator info files
fn make_validator_info(
    context: &mut WalletContext,
    host_name: &str,
    commission_rate: u64,
) -> Result<()> {
    let sender = context.active_address()?;
    let dir = std::env::current_dir()?;

    // Key file paths
    let protocol_key_file = dir.join("protocol.key");
    let account_key_file = dir.join("account.key");
    let network_key_file = dir.join("network.key");
    let worker_key_file = dir.join("worker.key");

    // Generate keys
    let account_key = match context.config.keystore.export(&sender)? {
        SomaKeyPair::Ed25519(key) => SomaKeyPair::Ed25519(key.copy()),
        _ => bail!("Only Ed25519 account keys are currently supported"),
    };

    make_key_file(&protocol_key_file, true, None)?;
    make_key_file(&account_key_file, false, Some(account_key))?;
    make_key_file(&network_key_file, false, None)?;
    make_key_file(&worker_key_file, false, None)?;

    // Read back keys to build validator info
    let protocol_keypair: AuthorityKeyPair = read_authority_keypair_from_file(&protocol_key_file)?;
    let account_keypair: SomaKeyPair = read_keypair_from_file(&account_key_file)?;
    let network_keypair: NetworkKeyPair = read_network_keypair_from_file(&network_key_file)?;
    let worker_keypair: NetworkKeyPair = read_network_keypair_from_file(&worker_key_file)?;

    let validator_info = GenesisValidatorInfo {
        info: types::validator_info::ValidatorInfo {
            protocol_key: protocol_keypair.public().into(),
            worker_key: worker_keypair.public().clone(),
            account_address: SomaAddress::from(&account_keypair.public()),
            network_key: network_keypair.public().clone(),
            commission_rate,
            network_address: Multiaddr::try_from(format!("/dns/{}/tcp/8080/http", host_name))?,
            p2p_address: Multiaddr::try_from(format!("/dns/{}/tcp/8084/http", host_name))?,
            primary_address: Multiaddr::try_from(format!("/dns/{}/tcp/8081/http", host_name))?,
        },
    };

    let validator_info_file = dir.join("validator.info");
    let validator_info_yaml = serde_yaml::to_string(&validator_info)?;
    fs::write(&validator_info_file, validator_info_yaml)?;

    println!("Generated key files in: {}", dir.display());
    println!(
        "Generated validator info: {}",
        validator_info_file.display()
    );

    Ok(())
}

/// Build a JoinCommittee (AddValidator) transaction from validator info file
fn build_join_committee_tx(file: &PathBuf) -> Result<TransactionKind> {
    let validator_info_bytes = fs::read(file)?;
    let validator_info: GenesisValidatorInfo = serde_yaml::from_slice(&validator_info_bytes)?;
    let info = validator_info.info;

    Ok(TransactionKind::AddValidator(AddValidatorArgs {
        pubkey_bytes: info.protocol_key.as_bytes().to_vec(),
        network_pubkey_bytes: info.network_key.to_bytes().to_vec(),
        worker_pubkey_bytes: info.worker_key.to_bytes().to_vec(),
        net_address: bcs::to_bytes(&info.network_address.to_string())?,
        p2p_address: bcs::to_bytes(&info.p2p_address.to_string())?,
        primary_address: bcs::to_bytes(&info.primary_address.to_string())?,
    }))
}

/// Build an UpdateValidatorMetadata transaction
fn build_update_metadata_tx(metadata: MetadataUpdate) -> Result<TransactionKind> {
    let args = match metadata {
        MetadataUpdate::NetworkAddress { network_address } => {
            // Validate TCP address
            if !network_address.is_loosely_valid_tcp_addr() {
                bail!("Network address must be a TCP address");
            }
            UpdateValidatorMetadataArgs {
                next_epoch_network_address: Some(bcs::to_bytes(&network_address.to_string())?),
                ..Default::default()
            }
        }
        MetadataUpdate::PrimaryAddress { primary_address } => UpdateValidatorMetadataArgs {
            next_epoch_primary_address: Some(bcs::to_bytes(&primary_address.to_string())?),
            ..Default::default()
        },
        MetadataUpdate::P2pAddress { p2p_address } => UpdateValidatorMetadataArgs {
            next_epoch_p2p_address: Some(bcs::to_bytes(&p2p_address.to_string())?),
            ..Default::default()
        },
        MetadataUpdate::NetworkPubKey { file } => {
            let keypair: NetworkKeyPair = read_network_keypair_from_file(file)?;
            UpdateValidatorMetadataArgs {
                next_epoch_network_pubkey: Some(keypair.public().to_bytes().to_vec()),
                ..Default::default()
            }
        }
        MetadataUpdate::WorkerPubKey { file } => {
            let keypair: NetworkKeyPair = read_network_keypair_from_file(file)?;
            UpdateValidatorMetadataArgs {
                next_epoch_worker_pubkey: Some(keypair.public().to_bytes().to_vec()),
                ..Default::default()
            }
        }
        MetadataUpdate::ProtocolPubKey { file } => {
            let keypair: AuthorityKeyPair = read_authority_keypair_from_file(file)?;
            UpdateValidatorMetadataArgs {
                next_epoch_protocol_pubkey: Some(keypair.public().as_bytes().to_vec()),
                ..Default::default()
            }
        }
    };

    Ok(TransactionKind::UpdateValidatorMetadata(args))
}

/// Get validator summary from the current system state
///
/// Queries the latest SystemState and searches for the validator in:
/// 1. consensus_validators (ValidatorStatus::Consensus)
/// 2. networking_validators (ValidatorStatus::Networking)  
/// 3. pending_validators (ValidatorStatus::Pending)
pub async fn get_validator_summary(
    client: &SomaClient,
    address: SomaAddress,
) -> Result<Option<(ValidatorStatus, ValidatorSummary)>> {
    // Get the latest system state from the RPC
    let system_state = client
        .get_latest_system_state()
        .await
        .map_err(|e| anyhow!("Failed to get system state: {}", e))?;

    // Search for the validator in the system state
    Ok(find_validator_in_system_state(&system_state, address))
}

/// Display validator metadata
async fn display_metadata(
    context: &mut WalletContext,
    address: SomaAddress,
    json: bool,
) -> Result<()> {
    let client = context.get_client().await?;

    match get_validator_summary(&client, address).await? {
        Some((status, summary)) => {
            println!("{}'s validator status: {}", address, status);
            if json {
                println!("{}", serde_json::to_string_pretty(&summary)?);
            } else {
                println!("{}", summary);
            }
        }
        None => {
            println!(
                "{} is not an active, networking, or pending validator.",
                address
            );
        }
    }
    Ok(())
}

/// Check that the sender has the required validator status
async fn check_status(
    context: &mut WalletContext,
    allowed_statuses: HashSet<ValidatorStatus>,
) -> Result<ValidatorStatus> {
    let validator_address = context.active_address()?;
    let client = context.get_client().await?;

    match get_validator_summary(&client, validator_address).await? {
        Some((status, _)) if allowed_statuses.contains(&status) => Ok(status),
        Some((status, _)) => bail!(
            "Validator {} is {:?}, but this operation requires one of: {:?}",
            validator_address,
            status,
            allowed_statuses
        ),
        None => bail!("{} is not a validator", validator_address),
    }
}

/// Convert a Validator to a ValidatorSummary for display
fn validator_to_summary(validator: &Validator, status: ValidatorStatus) -> ValidatorSummary {
    let metadata = &validator.metadata;

    ValidatorSummary {
        address: metadata.soma_address,
        status,
        voting_power: validator.voting_power,
        commission_rate: validator.commission_rate,
        network_address: metadata.net_address.to_string(),
        p2p_address: metadata.p2p_address.to_string(),
        primary_address: metadata.primary_address.to_string(),
        protocol_pubkey: metadata.protocol_pubkey.to_string(),
        network_pubkey: metadata.network_pubkey.clone().into_inner().to_string(),
        worker_pubkey: metadata.worker_pubkey.clone().into_inner().to_string(),
    }
}

/// Find a validator by address in the SystemState and return its summary
fn find_validator_in_system_state(
    system_state: &SystemState,
    address: SomaAddress,
) -> Option<(ValidatorStatus, ValidatorSummary)> {
    // Check consensus validators
    for validator in &system_state.validators.validators {
        if validator.metadata.soma_address == address {
            return Some((
                ValidatorStatus::Active,
                validator_to_summary(validator, ValidatorStatus::Active),
            ));
        }
    }

    // Check pending validators
    for validator in &system_state.validators.pending_validators {
        if validator.metadata.soma_address == address {
            return Some((
                ValidatorStatus::Pending,
                validator_to_summary(validator, ValidatorStatus::Pending),
            ));
        }
    }

    None
}
