use anyhow::{anyhow, bail, Result};
use clap::{Parser, Subcommand};
use fastcrypto::bls12381::min_sig::{BLS12381KeyPair, BLS12381PrivateKey};
use fastcrypto::hash::HashFunction as _;
use fastcrypto::traits::{KeyPair, ToFromBytes};
use rand::rngs::StdRng;
use rand::SeedableRng as _;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::time::Duration;
use tracing::info;
use url::Url;

use encoder::core::encoder_node::EncoderNode;
use sdk::wallet_context::WalletContext;
use sdk::SomaClient;
use soma_keys::key_derive::generate_new_key;
use soma_keys::keypair_file::{
    read_keypair_from_file, read_network_keypair_from_file, write_keypair_to_file,
};
use soma_keys::keystore::AccountKeystore;
use types::base::SomaAddress;
use types::checksum::Checksum;
use types::config::{encoder_config::EncoderConfig, PersistedConfig};
use types::crypto::{DefaultHash, NetworkKeyPair, SignatureScheme, SomaKeyPair};
use types::metadata::{
    DefaultDownloadMetadata, DefaultDownloadMetadataV1, DownloadMetadata, Metadata, MetadataV1,
};
use types::multiaddr::Multiaddr;
use types::shard_crypto::keys::EncoderKeyPair;
use types::transaction::{
    AddEncoderArgs, RemoveEncoderArgs, TransactionKind, UpdateEncoderMetadataArgs,
};

use crate::client_commands::TxProcessingArgs;
use crate::response::{EncoderCommandResponse, EncoderStatus, EncoderSummary, TransactionResponse};

#[derive(Parser)]
#[clap(rename_all = "kebab-case")]
pub enum SomaEncoderCommand {
    /// Start an encoder node from a config file
    #[clap(name = "start")]
    Start {
        /// Path to the encoder config file (YAML)
        #[clap(long = "config", short = 'c')]
        config: PathBuf,
        /// Working directory for the encoder (defaults to current directory)
        #[clap(long = "working-dir", short = 'w')]
        working_dir: Option<PathBuf>,
    },

    /// Generate encoder key files and registration info
    #[clap(name = "make-encoder-info")]
    MakeEncoderInfo {
        /// Hostname for the encoder (e.g., encoder.example.com)
        host_name: String,
    },

    /// Request to join the encoder committee
    #[clap(name = "join-committee")]
    JoinCommittee {
        /// Path to the encoder.info file
        #[clap(name = "encoder-info-path")]
        file: PathBuf,
        /// URL to the probe data (will be downloaded to compute checksum)
        #[clap(long)]
        probe_url: String,
        #[clap(flatten)]
        tx_args: TxProcessingArgs,
    },

    /// Request to leave the encoder committee
    #[clap(name = "leave-committee")]
    LeaveCommittee {
        #[clap(flatten)]
        tx_args: TxProcessingArgs,
    },

    /// Display encoder metadata
    #[clap(name = "display-metadata")]
    DisplayMetadata {
        /// Encoder address (defaults to active address)
        encoder_address: Option<SomaAddress>,
    },

    /// Update encoder metadata
    #[clap(name = "update-metadata")]
    UpdateMetadata {
        #[clap(subcommand)]
        metadata: EncoderMetadataUpdate,
        #[clap(flatten)]
        tx_args: TxProcessingArgs,
    },

    /// Set commission rate for the next epoch
    #[clap(name = "set-commission-rate")]
    SetCommissionRate {
        /// Commission rate in basis points (100 = 1%, max 10000 = 100%)
        commission_rate: u64,
        #[clap(flatten)]
        tx_args: TxProcessingArgs,
    },

    /// Set byte price for encoding services
    #[clap(name = "set-byte-price")]
    SetBytePrice {
        /// Price per byte in shannons
        price: u64,
        #[clap(flatten)]
        tx_args: TxProcessingArgs,
    },

    /// Report or un-report an encoder
    #[clap(name = "report-encoder")]
    ReportEncoder {
        /// Encoder address to report
        reportee_address: SomaAddress,
        /// Undo an existing report
        #[clap(long)]
        undo: bool,
        #[clap(flatten)]
        tx_args: TxProcessingArgs,
    },
}

#[derive(Subcommand, Clone)]
#[clap(rename_all = "kebab-case")]
pub enum EncoderMetadataUpdate {
    /// Update external network address
    ExternalAddress { address: Multiaddr },
    /// Update internal network address
    InternalAddress { address: Multiaddr },
    /// Update object server address
    ObjectServerAddress { address: Multiaddr },
    /// Update network public key
    NetworkPubKey {
        #[clap(name = "key-path")]
        file: PathBuf,
    },
    /// Update probe data (downloads from URL to compute checksum)
    Probe {
        /// URL to the new probe data
        #[clap(long)]
        url: String,
    },
}

/// Encoder info for registration (without probe - probe is provided at join time)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncoderInfo {
    pub encoder_pubkey: String,
    pub network_pubkey: String,
    pub account_address: SomaAddress,
    pub external_network_address: String,
    pub internal_network_address: String,
    pub object_server_address: String,
}

impl SomaEncoderCommand {
    pub async fn execute(self, context: &mut WalletContext) -> Result<EncoderCommandResponse> {
        let sender = context.active_address()?;

        match self {
            SomaEncoderCommand::Start {
                config,
                working_dir,
            } => {
                start_encoder_node(config, working_dir).await?;
                Ok(EncoderCommandResponse::Started)
            }

            SomaEncoderCommand::MakeEncoderInfo { host_name } => {
                make_encoder_info(context, &host_name)?;
                Ok(EncoderCommandResponse::MakeEncoderInfo)
            }

            SomaEncoderCommand::JoinCommittee {
                file,
                probe_url,
                tx_args,
            } => {
                let kind = build_join_encoder_tx(&file, &probe_url).await?;
                execute_or_serialize(context, sender, kind, tx_args).await
            }

            SomaEncoderCommand::LeaveCommittee { tx_args } => {
                let kind = TransactionKind::RemoveEncoder(RemoveEncoderArgs {
                    encoder_pubkey_bytes: vec![], // Inferred from sender
                });
                execute_or_serialize(context, sender, kind, tx_args).await
            }

            SomaEncoderCommand::DisplayMetadata { encoder_address } => {
                let address = encoder_address.unwrap_or(sender);
                display_encoder_metadata(context, address).await?;
                Ok(EncoderCommandResponse::DisplayMetadata)
            }

            SomaEncoderCommand::UpdateMetadata { metadata, tx_args } => {
                let kind = build_update_encoder_metadata_tx(metadata).await?;
                execute_or_serialize(context, sender, kind, tx_args).await
            }

            SomaEncoderCommand::SetCommissionRate {
                commission_rate,
                tx_args,
            } => {
                if commission_rate > 10000 {
                    bail!("Commission rate cannot exceed 10000 (100%)");
                }

                let kind = TransactionKind::SetEncoderCommissionRate {
                    new_rate: commission_rate,
                };
                execute_or_serialize(context, sender, kind, tx_args).await
            }

            SomaEncoderCommand::SetBytePrice { price, tx_args } => {
                let kind = TransactionKind::SetEncoderBytePrice { new_price: price };
                execute_or_serialize(context, sender, kind, tx_args).await
            }

            SomaEncoderCommand::ReportEncoder {
                reportee_address,
                undo,
                tx_args,
            } => {
                if sender == reportee_address {
                    bail!("Cannot report yourself");
                }

                let kind = if undo {
                    TransactionKind::UndoReportEncoder {
                        reportee: reportee_address,
                    }
                } else {
                    TransactionKind::ReportEncoder {
                        reportee: reportee_address,
                    }
                };
                execute_or_serialize(context, sender, kind, tx_args).await
            }
        }
    }
}

/// Start an encoder node from a config file
async fn start_encoder_node(config_path: PathBuf, working_dir: Option<PathBuf>) -> Result<()> {
    info!("Loading encoder config from {:?}", config_path);

    let encoder_config: EncoderConfig = PersistedConfig::read(&config_path).map_err(|err| {
        anyhow!(
            "Cannot open encoder config file at {:?}: {}",
            config_path,
            err
        )
    })?;

    // Use provided working dir or default to current directory
    let working_dir = working_dir
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

    info!(
        "Starting encoder node with working directory: {:?}",
        working_dir
    );
    info!(
        "Encoder external address: {}",
        encoder_config.external_network_address
    );

    // Start the encoder node (without shared object store for production use)
    let node = Arc::new(EncoderNode::start(encoder_config, working_dir, None).await);

    info!("Encoder node started successfully");

    // Keep the node running until Ctrl+C
    let mut interval = tokio::time::interval(Duration::from_secs(5));
    loop {
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {
                info!("Received Ctrl+C, shutting down encoder...");
                break;
            }
            _ = interval.tick() => {
                // Health check or status logging could go here
            }
        }
    }

    // Node will be dropped here, triggering graceful shutdown
    drop(node);
    info!("Encoder node shut down");

    Ok(())
}

/// Execute a transaction or serialize it for offline signing
async fn execute_or_serialize(
    context: &mut WalletContext,
    sender: SomaAddress,
    kind: TransactionKind,
    tx_args: TxProcessingArgs,
) -> Result<EncoderCommandResponse> {
    use fastcrypto::encoding::{Base64, Encoding};
    use sdk::transaction_builder::TransactionBuilder;

    let builder = TransactionBuilder::new(context);

    if tx_args.serialize_unsigned_transaction {
        let tx_data = builder.build_transaction_data(sender, kind, None).await?;
        let bytes = bcs::to_bytes(&tx_data)?;
        let encoded = Base64::encode(&bytes);

        Ok(EncoderCommandResponse::SerializedTransaction {
            serialized_unsigned_transaction: encoded,
        })
    } else {
        let tx = builder.build_transaction(sender, kind, None).await?;
        drop(builder);

        let response = context.execute_transaction_may_fail(tx).await?;
        Ok(EncoderCommandResponse::Transaction(
            TransactionResponse::from_response(&response),
        ))
    }
}

/// Generate encoder info files
fn make_encoder_info(context: &mut WalletContext, host_name: &str) -> Result<()> {
    let sender = context.active_address()?;
    let dir = std::env::current_dir()?;

    // Key file paths
    let encoder_key_file = dir.join("encoder.key");
    let account_key_file = dir.join("account.key");
    let network_key_file = dir.join("network.key");

    // Generate keys if they don't exist
    let account_key = match context.config.keystore.export(&sender)? {
        SomaKeyPair::Ed25519(key) => SomaKeyPair::Ed25519(key.copy()),
        _ => bail!("Only Ed25519 account keys are currently supported"),
    };

    make_key_file(&account_key_file, Some(account_key))?;
    make_key_file(&network_key_file, None)?;
    make_encoder_key_file(&encoder_key_file)?;

    // Read back keys to build encoder info
    let account_keypair: SomaKeyPair = read_keypair_from_file(&account_key_file)?;
    let network_keypair: NetworkKeyPair = read_network_keypair_from_file(&network_key_file)?;
    let encoder_keypair = read_encoder_keypair_from_file(&encoder_key_file)?;

    // Generate addresses from hostname (similar to validator commands)
    let external_network_address = format!("/dns/{}/tcp/9000/http", host_name);
    let internal_network_address = format!("/dns/{}/tcp/9001/http", host_name);
    let object_server_address = format!("https://{}/objects", host_name);

    let encoder_info = EncoderInfo {
        encoder_pubkey: hex::encode(encoder_keypair.public().to_bytes()),
        network_pubkey: hex::encode(network_keypair.public().to_bytes()),
        account_address: SomaAddress::from(&account_keypair.public()),
        external_network_address,
        internal_network_address,
        object_server_address,
    };

    let encoder_info_file = dir.join("encoder.info");
    let encoder_info_yaml = serde_yaml::to_string(&encoder_info)?;
    fs::write(&encoder_info_file, encoder_info_yaml)?;

    println!("Generated key files in: {}", dir.display());
    println!("Generated encoder info: {}", encoder_info_file.display());
    println!();
    println!("Next steps:");
    println!("  1. Upload your probe data to your object server");
    println!(
        "  2. Run: soma encoder join-committee {} --probe-url <PROBE_URL>",
        encoder_info_file.display()
    );

    Ok(())
}

/// Create a key file if it doesn't exist
fn make_key_file(path: &PathBuf, key: Option<SomaKeyPair>) -> Result<()> {
    if path.exists() {
        println!("Using existing key file: {:?}", path);
        return Ok(());
    }

    let kp = match key {
        Some(k) => k,
        None => {
            let (_, kp, _, _) = generate_new_key(SignatureScheme::ED25519, None, None)?;
            kp
        }
    };
    write_keypair_to_file(&kp, path)?;

    println!("Generated new key file: {:?}", path);
    Ok(())
}

/// Create an encoder key file (BLS)
fn make_encoder_key_file(path: &PathBuf) -> Result<()> {
    if path.exists() {
        println!("Using existing encoder key file: {:?}", path);
        return Ok(());
    }

    let keypair = EncoderKeyPair::generate(&mut StdRng::from_entropy());
    let private = keypair.inner().copy().private();
    fs::write(path, hex::encode(private.as_bytes()))?;

    println!("Generated new encoder key file: {:?}", path);
    Ok(())
}

/// Read encoder keypair from file
fn read_encoder_keypair_from_file(path: &PathBuf) -> Result<EncoderKeyPair> {
    let contents = fs::read_to_string(path)?;
    let bytes = hex::decode(contents.trim())?;
    let kp = BLS12381KeyPair::from(
        BLS12381PrivateKey::from_bytes(&bytes)
            .map_err(|e| anyhow!("Invalid encoder key: {}", e))?,
    );
    Ok(EncoderKeyPair::new(kp))
}

/// Download data from a URL and create DownloadMetadata
async fn download_and_create_metadata(url_str: &str) -> Result<DownloadMetadata> {
    let parsed_url = Url::parse(url_str).map_err(|e| anyhow!("Invalid URL: {}", e))?;

    println!("Downloading data from {}...", url_str);
    let data = download_data(url_str).await?;
    let size = data.len();
    println!("Downloaded {} bytes", size);

    let checksum = compute_checksum(&data);
    let metadata = Metadata::V1(MetadataV1::new(checksum, size));

    Ok(DownloadMetadata::Default(DefaultDownloadMetadata::V1(
        DefaultDownloadMetadataV1::new(parsed_url, metadata),
    )))
}

/// Download data from a URL
async fn download_data(url: &str) -> Result<Vec<u8>> {
    let response = reqwest::get(url)
        .await
        .map_err(|e| anyhow!("Failed to download data: {}", e))?;

    if !response.status().is_success() {
        return Err(anyhow!("HTTP error: {}", response.status()));
    }

    response
        .bytes()
        .await
        .map(|b| b.to_vec())
        .map_err(|e| anyhow!("Failed to read response body: {}", e))
}

/// Compute checksum of data
fn compute_checksum(data: &[u8]) -> Checksum {
    let mut hasher = DefaultHash::default();
    hasher.update(data);
    Checksum::new_from_hash(hasher.finalize().into())
}

/// Build a JoinCommittee (AddEncoder) transaction from encoder info file
async fn build_join_encoder_tx(file: &PathBuf, probe_url: &str) -> Result<TransactionKind> {
    let encoder_info_bytes = fs::read(file)?;
    let encoder_info: EncoderInfo = serde_yaml::from_slice(&encoder_info_bytes)?;

    // Download probe data and create metadata
    let probe = download_and_create_metadata(probe_url).await?;

    Ok(TransactionKind::AddEncoder(AddEncoderArgs {
        encoder_pubkey_bytes: hex::decode(&encoder_info.encoder_pubkey)?,
        network_pubkey_bytes: hex::decode(&encoder_info.network_pubkey)?,
        external_network_address: bcs::to_bytes(&encoder_info.external_network_address)?,
        internal_network_address: bcs::to_bytes(&encoder_info.internal_network_address)?,
        object_server_address: bcs::to_bytes(&encoder_info.object_server_address)?,
        probe: bcs::to_bytes(&probe)?,
    }))
}

/// Build an UpdateEncoderMetadata transaction
async fn build_update_encoder_metadata_tx(
    metadata: EncoderMetadataUpdate,
) -> Result<TransactionKind> {
    let args = match metadata {
        EncoderMetadataUpdate::ExternalAddress { address } => UpdateEncoderMetadataArgs {
            next_epoch_external_network_address: Some(bcs::to_bytes(&address.to_string())?),
            next_epoch_internal_network_address: None,
            next_epoch_network_pubkey: None,
            next_epoch_object_server_address: None,
            next_epoch_probe: None,
        },
        EncoderMetadataUpdate::InternalAddress { address } => UpdateEncoderMetadataArgs {
            next_epoch_external_network_address: None,
            next_epoch_internal_network_address: Some(bcs::to_bytes(&address.to_string())?),
            next_epoch_network_pubkey: None,
            next_epoch_object_server_address: None,
            next_epoch_probe: None,
        },
        EncoderMetadataUpdate::ObjectServerAddress { address } => UpdateEncoderMetadataArgs {
            next_epoch_external_network_address: None,
            next_epoch_internal_network_address: None,
            next_epoch_network_pubkey: None,
            next_epoch_object_server_address: Some(bcs::to_bytes(&address.to_string())?),
            next_epoch_probe: None,
        },
        EncoderMetadataUpdate::NetworkPubKey { file } => {
            let keypair: NetworkKeyPair = read_network_keypair_from_file(file)?;
            UpdateEncoderMetadataArgs {
                next_epoch_external_network_address: None,
                next_epoch_internal_network_address: None,
                next_epoch_network_pubkey: Some(keypair.public().to_bytes().to_vec()),
                next_epoch_object_server_address: None,
                next_epoch_probe: None,
            }
        }
        EncoderMetadataUpdate::Probe { url } => {
            let probe = download_and_create_metadata(&url).await?;
            UpdateEncoderMetadataArgs {
                next_epoch_external_network_address: None,
                next_epoch_internal_network_address: None,
                next_epoch_network_pubkey: None,
                next_epoch_object_server_address: None,
                next_epoch_probe: Some(bcs::to_bytes(&probe)?),
            }
        }
    };

    Ok(TransactionKind::UpdateEncoderMetadata(args))
}

/// Display encoder metadata
async fn display_encoder_metadata(context: &mut WalletContext, address: SomaAddress) -> Result<()> {
    let client = context.get_client().await?;

    match get_encoder_summary(&client, address).await? {
        Some((status, summary)) => {
            println!("{}'s encoder status: {}", address, status);
            println!("{}", summary);
        }
        None => {
            println!("{} is not an active or pending encoder.", address);
        }
    }
    Ok(())
}

/// Get encoder summary from the current system state
pub async fn get_encoder_summary(
    client: &SomaClient,
    address: SomaAddress,
) -> Result<Option<(EncoderStatus, EncoderSummary)>> {
    let system_state = client
        .get_latest_system_state()
        .await
        .map_err(|e| anyhow!("Failed to get system state: {}", e))?;

    // Search for the encoder in the system state
    for encoder in &system_state.encoders.active_encoders {
        if encoder.metadata.soma_address == address {
            return Ok(Some((
                EncoderStatus::Active,
                EncoderSummary {
                    address: encoder.metadata.soma_address,
                    status: EncoderStatus::Active,
                    commission_rate: encoder.commission_rate,
                    byte_price: encoder.byte_price,
                    external_address: encoder.metadata.external_network_address.to_string(),
                    internal_address: encoder.metadata.internal_network_address.to_string(),
                    object_server_address: encoder.metadata.object_server_address.to_string(),
                    encoder_pubkey: hex::encode(encoder.metadata.encoder_pubkey.to_bytes()),
                    network_pubkey: hex::encode(
                        encoder
                            .metadata
                            .network_pubkey
                            .clone()
                            .into_inner()
                            .as_bytes(),
                    ),
                },
            )));
        }
    }

    for encoder in &system_state.encoders.pending_active_encoders {
        if encoder.metadata.soma_address == address {
            return Ok(Some((
                EncoderStatus::Pending,
                EncoderSummary {
                    address: encoder.metadata.soma_address,
                    status: EncoderStatus::Pending,
                    commission_rate: encoder.commission_rate,
                    byte_price: encoder.byte_price,
                    external_address: encoder.metadata.external_network_address.to_string(),
                    internal_address: encoder.metadata.internal_network_address.to_string(),
                    object_server_address: encoder.metadata.object_server_address.to_string(),
                    encoder_pubkey: hex::encode(encoder.metadata.encoder_pubkey.to_bytes()),
                    network_pubkey: hex::encode(
                        encoder
                            .metadata
                            .network_pubkey
                            .clone()
                            .into_inner()
                            .as_bytes(),
                    ),
                },
            )));
        }
    }

    Ok(None)
}
