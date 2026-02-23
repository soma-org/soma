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
    model::ModelId,
    multiaddr::Multiaddr,
    object::{ObjectID, ObjectRef, Owner},
    system_state::{SystemState, SystemStateTrait as _, validator::Validator},
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
use sdk::wallet_context::WalletContext;
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

use crate::client_commands::TxProcessingArgs;

#[derive(Parser)]
#[clap(rename_all = "kebab-case")]
pub enum SomaValidatorCommand {
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

    /// List all active and pending validators
    #[clap(name = "list")]
    List,

    /// Display validator metadata
    #[clap(name = "display-metadata")]
    DisplayMetadata {
        /// Validator address (defaults to active address)
        #[clap(name = "validator-address")]
        validator_address: Option<SomaAddress>,
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

    /// Report or un-report a model (validators only)
    ///
    /// Active validators can report misbehaving models. If reports reach
    /// 2f+1 quorum at epoch boundary, the model is slashed and deactivated.
    #[clap(name = "report-model")]
    ReportModel {
        /// The model ID to report
        #[clap(name = "model-id")]
        model_id: ModelId,
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
            SomaValidatorCommand::MakeValidatorInfo { host_name, commission_rate } => {
                let output = make_validator_info(context, &host_name, commission_rate)?;
                Ok(ValidatorCommandResponse::MakeValidatorInfo(output))
            }

            SomaValidatorCommand::JoinCommittee { file, tx_args } => {
                let kind = build_join_committee_tx(&file)?;
                execute_tx(context, sender, kind, tx_args).await
            }

            SomaValidatorCommand::LeaveCommittee { tx_args } => {
                // Verify sender is an active validator before building tx
                check_status(context, HashSet::from([ValidatorStatus::Active])).await?;

                let kind = TransactionKind::RemoveValidator(RemoveValidatorArgs {
                    pubkey_bytes: vec![], // The signer is inferred from tx sender
                });
                execute_tx(context, sender, kind, tx_args).await
            }

            SomaValidatorCommand::List => {
                let client = context.get_client().await?;
                let system_state = client
                    .get_latest_system_state()
                    .await
                    .map_err(|e| anyhow!("Failed to get system state: {}", e.message()))?;
                let validators = list_all_validators(&system_state);
                Ok(ValidatorCommandResponse::List(crate::response::ValidatorListOutput {
                    validators,
                }))
            }

            SomaValidatorCommand::DisplayMetadata { validator_address } => {
                let address = validator_address.unwrap_or(sender);
                let output = display_metadata(context, address).await?;
                Ok(ValidatorCommandResponse::DisplayMetadata(output))
            }

            SomaValidatorCommand::UpdateMetadata { metadata, tx_args } => {
                // Verify sender is active or pending
                check_status(
                    context,
                    HashSet::from([ValidatorStatus::Active, ValidatorStatus::Pending]),
                )
                .await?;

                let kind = build_update_metadata_tx(metadata, sender)?;
                execute_tx(context, sender, kind, tx_args).await
            }

            SomaValidatorCommand::SetCommissionRate { commission_rate, tx_args } => {
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

                let kind = TransactionKind::SetCommissionRate { new_rate: commission_rate };
                execute_tx(context, sender, kind, tx_args).await
            }

            SomaValidatorCommand::ReportValidator { reportee_address, undo_report, tx_args } => {
                // Only active validators can report
                check_status(context, HashSet::from([ValidatorStatus::Active])).await?;

                // Can't report yourself
                if sender == reportee_address {
                    bail!("Cannot report yourself");
                }

                let kind = if undo_report {
                    TransactionKind::UndoReportValidator { reportee: reportee_address }
                } else {
                    TransactionKind::ReportValidator { reportee: reportee_address }
                };
                execute_tx(context, sender, kind, tx_args).await
            }

            SomaValidatorCommand::ReportModel { model_id, undo_report, tx_args } => {
                // Only active validators can report models
                check_status(context, HashSet::from([ValidatorStatus::Active])).await?;

                let kind = if undo_report {
                    TransactionKind::UndoReportModel { model_id }
                } else {
                    TransactionKind::ReportModel { model_id }
                };
                execute_tx(context, sender, kind, tx_args).await
            }
        }
    }
}

/// Start a validator node from a config file
pub async fn start_validator_node(config_path: PathBuf) -> Result<()> {
    crate::soma_commands::print_banner("Validator");

    info!("Loading validator config from {:?}", config_path);

    let node_config: NodeConfig = PersistedConfig::read(&config_path).map_err(|err| {
        anyhow!("Cannot open validator config file at {:?}: {}", config_path, err)
    })?;

    let protocol_key = format!("{:?}", node_config.protocol_public_key());
    eprint!("  {:<50}", "Starting validator...");

    // Start the validator node
    let node = SomaNode::start(node_config)
        .await
        .map_err(|err| anyhow!("Failed to start validator node: {}", err))?;

    eprintln!("{}", "done".green());
    eprintln!();
    eprintln!("  Protocol key: {}", protocol_key.dimmed());
    eprintln!();
    eprintln!("  Press {} to stop.", "Ctrl+C".bold());

    // Keep the node running until Ctrl+C or SIGTERM
    let mut interval = tokio::time::interval(Duration::from_secs(5));
    let mut sigterm = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        .expect("failed to register SIGTERM handler");
    loop {
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {
                info!("Received Ctrl+C, shutting down validator...");
                break;
            }
            _ = sigterm.recv() => {
                info!("Received SIGTERM, shutting down validator...");
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

/// Execute a validator transaction, delegating to the shared client_commands helper.
async fn execute_tx(
    context: &mut WalletContext,
    sender: SomaAddress,
    kind: TransactionKind,
    tx_args: TxProcessingArgs,
) -> Result<ValidatorCommandResponse> {
    let result =
        crate::client_commands::execute_or_serialize(context, sender, kind, None, tx_args).await?;

    // Convert ClientCommandResponse to ValidatorCommandResponse
    match result {
        crate::response::ClientCommandResponse::Transaction(tx) => {
            Ok(ValidatorCommandResponse::Transaction(tx))
        }
        crate::response::ClientCommandResponse::SerializedUnsignedTransaction(s) => {
            Ok(ValidatorCommandResponse::SerializedTransaction { serialized_transaction: s })
        }
        crate::response::ClientCommandResponse::SerializedSignedTransaction(s) => {
            Ok(ValidatorCommandResponse::SerializedTransaction { serialized_transaction: s })
        }
        crate::response::ClientCommandResponse::TransactionDigest(d) => {
            Ok(ValidatorCommandResponse::TransactionDigest(d))
        }
        crate::response::ClientCommandResponse::Simulation(sim) => {
            Ok(ValidatorCommandResponse::Simulation(sim))
        }
        _ => bail!("Unexpected response type from transaction execution"),
    }
}

/// Create a key file if it doesn't exist
fn make_key_file(path: &PathBuf, is_protocol_key: bool, key: Option<SomaKeyPair>) -> Result<()> {
    if path.exists() {
        eprintln!("Using existing key file: {:?}", path);
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

    eprintln!("Generated new key file: {:?}", path);
    Ok(())
}

/// Generate validator info files
fn make_validator_info(
    context: &mut WalletContext,
    host_name: &str,
    commission_rate: u64,
) -> Result<MakeValidatorInfoOutput> {
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

    let account_address = SomaAddress::from(&account_keypair.public());
    let pop = types::crypto::generate_proof_of_possession(&protocol_keypair, account_address);
    let proof_of_possession = pop.as_ref().to_vec();

    let validator_info = GenesisValidatorInfo {
        info: types::validator_info::ValidatorInfo {
            protocol_key: protocol_keypair.public().into(),
            worker_key: worker_keypair.public().clone(),
            account_address,
            network_key: network_keypair.public().clone(),
            proof_of_possession,
            commission_rate,
            network_address: Multiaddr::try_from(format!("/dns/{}/tcp/8080/http", host_name))?,
            p2p_address: Multiaddr::try_from(format!("/dns/{}/tcp/8084/http", host_name))?,
            primary_address: Multiaddr::try_from(format!("/dns/{}/tcp/8081/http", host_name))?,
            proxy_address: Multiaddr::try_from(format!("/dns/{}/tcp/8090/http", host_name))?,
        },
    };

    let validator_info_file = dir.join("validator.info");
    let validator_info_yaml = serde_yaml::to_string(&validator_info)?;
    fs::write(&validator_info_file, validator_info_yaml)?;

    Ok(MakeValidatorInfoOutput {
        output_dir: dir.display().to_string(),
        validator_info_file: validator_info_file.display().to_string(),
        files: vec![
            "protocol.key".to_string(),
            "account.key".to_string(),
            "network.key".to_string(),
            "worker.key".to_string(),
            "validator.info".to_string(),
        ],
    })
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
        proof_of_possession: info.proof_of_possession,
        net_address: bcs::to_bytes(&info.network_address.to_string())?,
        p2p_address: bcs::to_bytes(&info.p2p_address.to_string())?,
        primary_address: bcs::to_bytes(&info.primary_address.to_string())?,
        proxy_address: bcs::to_bytes(&info.proxy_address.to_string())?,
    }))
}

/// Build an UpdateValidatorMetadata transaction
fn build_update_metadata_tx(
    metadata: MetadataUpdate,
    sender: SomaAddress,
) -> Result<TransactionKind> {
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
            let pop = types::crypto::generate_proof_of_possession(&keypair, sender);
            UpdateValidatorMetadataArgs {
                next_epoch_protocol_pubkey: Some(keypair.public().as_bytes().to_vec()),
                next_epoch_proof_of_possession: Some(pop.as_ref().to_vec()),
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
        .map_err(|e| anyhow!("Failed to get system state: {}", e.message()))?;

    // Search for the validator in the system state
    Ok(find_validator_in_system_state(&system_state, address))
}

/// Display validator metadata
async fn display_metadata(
    context: &mut WalletContext,
    address: SomaAddress,
) -> Result<DisplayMetadataOutput> {
    let client = context.get_client().await?;

    match get_validator_summary(&client, address).await? {
        Some((status, summary)) => {
            Ok(DisplayMetadataOutput { address, status: Some(status), summary: Some(summary) })
        }
        None => Ok(DisplayMetadataOutput { address, status: None, summary: None }),
    }
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

/// List all validators from the SystemState
fn list_all_validators(system_state: &SystemState) -> Vec<ValidatorSummary> {
    let mut validators = Vec::new();
    for validator in &system_state.validators().validators {
        validators.push(validator_to_summary(validator, ValidatorStatus::Active));
    }
    for validator in &system_state.validators().pending_validators {
        validators.push(validator_to_summary(validator, ValidatorStatus::Pending));
    }
    validators
}

/// Find a validator by address in the SystemState and return its summary
fn find_validator_in_system_state(
    system_state: &SystemState,
    address: SomaAddress,
) -> Option<(ValidatorStatus, ValidatorSummary)> {
    // Check consensus validators
    for validator in &system_state.validators().validators {
        if validator.metadata.soma_address == address {
            return Some((
                ValidatorStatus::Active,
                validator_to_summary(validator, ValidatorStatus::Active),
            ));
        }
    }

    // Check pending validators
    for validator in &system_state.validators().pending_validators {
        if validator.metadata.soma_address == address {
            return Some((
                ValidatorStatus::Pending,
                validator_to_summary(validator, ValidatorStatus::Pending),
            ));
        }
    }

    None
}

// =============================================================================
// Output types
// =============================================================================

/// Output from `make-validator-info` command
#[derive(Debug, Serialize)]
pub struct MakeValidatorInfoOutput {
    pub output_dir: String,
    pub validator_info_file: String,
    pub files: Vec<String>,
}

/// Output from `display-metadata` command
#[derive(Debug, Serialize)]
pub struct DisplayMetadataOutput {
    pub address: SomaAddress,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<ValidatorStatus>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<ValidatorSummary>,
}
