use anyhow::{anyhow, bail, Result};
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
use types::{
    base::SomaAddress,
    config::node_config::DEFAULT_COMMISSION_RATE,
    crypto::{AuthorityPublicKey, NetworkPublicKey, Signable},
    multiaddr::Multiaddr,
    object::{ObjectID, ObjectRef, Owner},
};
use types::{
    intent::{Intent, IntentMessage, IntentScope},
    validator_info::GenesisValidatorInfo,
};

use sdk::wallet_context::WalletContext;
use sdk::SomaClient;
use soma_keys::{
    key_derive::generate_new_key,
    keypair_file::{
        read_authority_keypair_from_file, read_keypair_from_file, read_network_keypair_from_file,
        write_authority_keypair_to_file, write_keypair_to_file,
    },
};
use soma_keys::{keypair_file::read_key, keystore::AccountKeystore};
use types::crypto::{get_authority_key_pair, AuthorityPublicKeyBytes};
use types::crypto::{AuthorityKeyPair, NetworkKeyPair, SignatureScheme, SomaKeyPair};
use types::transaction::{Transaction, TransactionData};

/// Arguments related to transaction processing
#[derive(Args, Debug, Default)]
pub struct TxProcessingArgs {
    /// Instead of executing the transaction, serialize the bcs bytes of the unsigned transaction data
    /// (TransactionData) using base64 encoding, and print out the string <TX_BYTES>. The string can
    /// be used to execute transaction with `soma client execute-signed-tx --tx-bytes <TX_BYTES>`.
    #[arg(long)]
    pub serialize_unsigned_transaction: bool,
}

#[derive(Parser)]
#[clap(rename_all = "kebab-case")]
pub enum SomaValidatorCommand {
    #[clap(name = "make-validator-info")]
    MakeValidatorInfo {
        host_name: String,
        commission_rate: u64,
    },
    #[clap(name = "join-committee")]
    AddValidator {
        #[clap(name = "validator-info-path")]
        file: PathBuf,
        #[clap(flatten)]
        tx_args: TxProcessingArgs,
    },
    #[clap(name = "leave-committee")]
    RemoveValidator {
        #[clap(flatten)]
        tx_args: TxProcessingArgs,
    },
    #[clap(name = "display-metadata")]
    DisplayMetadata {
        #[clap(name = "validator-address")]
        validator_address: Option<SomaAddress>,
        #[clap(name = "json", long)]
        json: Option<bool>,
    },
    #[clap(name = "update-metadata")]
    UpdateMetadata {
        #[clap(subcommand)]
        metadata: MetadataUpdate,
        #[clap(flatten)]
        tx_args: TxProcessingArgs,
    },
    /// Set commission rate
    #[clap(name = "set-commission-rate")]
    SetCommissionRate {
        #[clap(name = "commission-rate")]
        commission_rate: u64,
        #[clap(flatten)]
        tx_args: TxProcessingArgs,
    },
    /// Report or un-report a validator.
    #[clap(name = "report-validator")]
    ReportValidator {
        /// Optional when sender is reporter validator itself and it holds the Cap object.
        /// Required when sender is not the reporter validator itself.
        /// Validator's OperationCap ID can be found by using the `display-metadata` subcommand.
        #[clap(name = "operation-cap-id", long)]
        operation_cap_id: Option<ObjectID>,
        /// The Soma Address of the validator is being reported or un-reported
        #[clap(name = "reportee-address")]
        reportee_address: SomaAddress,
        /// If true, undo an existing report.
        #[clap(name = "undo-report", long)]
        undo_report: Option<bool>,
        #[clap(flatten)]
        tx_args: TxProcessingArgs,
    },
}

#[derive(Serialize)]
#[serde(untagged)]
pub enum SomaValidatorCommandResponse {
    MakeValidatorInfo,
    DisplayMetadata,
    AddValidator {
        response: Option<SomaTransactionBlockResponse>,
        serialized_unsigned_transaction: Option<String>,
    },
    RemoveValidator {
        response: Option<SomaTransactionBlockResponse>,
        serialized_unsigned_transaction: Option<String>,
    },
    UpdateMetadata {
        response: Option<SomaTransactionBlockResponse>,
        serialized_unsigned_transaction: Option<String>,
    },
    SetCommissionRate {
        response: Option<SomaTransactionBlockResponse>,
        serialized_unsigned_transaction: Option<String>,
    },
    ReportValidator {
        response: Option<SomaTransactionBlockResponse>,
        serialized_unsigned_transaction: Option<String>,
    },
}

fn make_key_files(
    file_name: PathBuf,
    is_protocol_key: bool,
    key: Option<SomaKeyPair>,
) -> Result<()> {
    if file_name.exists() {
        println!("Use existing {:?} key file.", file_name);
        return Ok(());
    } else if is_protocol_key {
        let (_, keypair) = get_authority_key_pair();
        write_authority_keypair_to_file(&keypair, file_name.clone())?;
        println!("Generated new key file: {:?}.", file_name);
    } else {
        let kp = match key {
            Some(key) => {
                println!(
                    "Generated new key file {:?} based on soma.keystore file.",
                    file_name
                );
                key
            }
            None => {
                let (_, kp, _, _) = generate_new_key(SignatureScheme::ED25519, None, None)?;
                println!("Generated new key file: {:?}.", file_name);
                kp
            }
        };
        write_keypair_to_file(&kp, &file_name)?;
    }
    Ok(())
}

impl SomaValidatorCommand {
    pub async fn execute(
        self,
        context: &mut WalletContext,
    ) -> Result<SomaValidatorCommandResponse, anyhow::Error> {
        let soma_address = context.active_address()?;

        Ok(match self {
            SomaValidatorCommand::MakeValidatorInfo {
                host_name,
                commission_rate,
            } => {
                let dir = std::env::current_dir()?;
                let protocol_key_file_name = dir.join("protocol.key");
                let account_key = match context.config.keystore.export(&soma_address)? {
                    SomaKeyPair::Ed25519(account_key) => SomaKeyPair::Ed25519(account_key.copy()),
                    _ => panic!(
                        "Other account key types supported yet, please use Ed25519 keys for now."
                    ),
                };
                let account_key_file_name = dir.join("account.key");
                let network_key_file_name = dir.join("network.key");
                let worker_key_file_name = dir.join("worker.key");
                make_key_files(protocol_key_file_name.clone(), true, None)?;
                make_key_files(account_key_file_name.clone(), false, Some(account_key))?;
                make_key_files(network_key_file_name.clone(), false, None)?;
                make_key_files(worker_key_file_name.clone(), false, None)?;

                let keypair: AuthorityKeyPair =
                    read_authority_keypair_from_file(protocol_key_file_name)?;
                let account_keypair: SomaKeyPair = read_keypair_from_file(account_key_file_name)?;
                let worker_keypair: NetworkKeyPair =
                    read_network_keypair_from_file(worker_key_file_name)?;
                let network_keypair: NetworkKeyPair =
                    read_network_keypair_from_file(network_key_file_name)?;
                let validator_info = GenesisValidatorInfo {
                    info: types::validator_info::ValidatorInfo {
                        protocol_key: keypair.public().into(),
                        worker_key: worker_keypair.public().clone(),
                        account_address: SomaAddress::from(&account_keypair.public()),
                        network_key: network_keypair.public().clone(),
                        commission_rate: DEFAULT_COMMISSION_RATE,
                        network_address: Multiaddr::try_from(format!(
                            "/dns/{}/tcp/8080/http",
                            host_name
                        ))?,
                        p2p_address: Multiaddr::try_from(format!(
                            "/dns/{}/tcp/8084/http",
                            host_name
                        ))?,
                        primary_address: Multiaddr::try_from(format!(
                            "/dns/{}/tcp/8081/http",
                            host_name
                        ))?,
                        worker_address: Multiaddr::try_from(format!(
                            "/dns/{}/tcp/8082/http",
                            host_name
                        ))?,
                    },
                };
                // TODO set key files permission
                let validator_info_file_name = dir.join("validator.info");
                let validator_info_bytes = serde_yaml::to_string(&validator_info)?;
                fs::write(validator_info_file_name.clone(), validator_info_bytes)?;
                println!(
                    "Generated validator info file: {:?}.",
                    validator_info_file_name
                );
                SomaValidatorCommandResponse::MakeValidatorInfo
            }
            SomaValidatorCommand::AddValidator { file, tx_args } => {
                let gas_budget = tx_args.gas_budget.unwrap_or(DEFAULT_GAS_BUDGET);
                let validator_info_bytes = fs::read(file)?;
                // Note: we should probably rename the struct or evolve it accordingly.
                let validator_info: GenesisValidatorInfo =
                    serde_yaml::from_slice(&validator_info_bytes)?;
                let validator = validator_info.info;

                let args = vec![
                    CallArg::Pure(
                        bcs::to_bytes(&AuthorityPublicKeyBytes::from_bytes(
                            validator.protocol_key().as_bytes(),
                        )?)
                        .unwrap(),
                    ),
                    CallArg::Pure(
                        bcs::to_bytes(&validator.network_key().as_bytes().to_vec()).unwrap(),
                    ),
                    CallArg::Pure(
                        bcs::to_bytes(&validator.worker_key().as_bytes().to_vec()).unwrap(),
                    ),
                    CallArg::Pure(
                        bcs::to_bytes(&validator_info.proof_of_possession.as_ref().to_vec())
                            .unwrap(),
                    ),
                    CallArg::Pure(
                        bcs::to_bytes(&validator.name().to_owned().into_bytes()).unwrap(),
                    ),
                    CallArg::Pure(
                        bcs::to_bytes(&validator.description.clone().into_bytes()).unwrap(),
                    ),
                    CallArg::Pure(
                        bcs::to_bytes(&validator.image_url.clone().into_bytes()).unwrap(),
                    ),
                    CallArg::Pure(
                        bcs::to_bytes(&validator.project_url.clone().into_bytes()).unwrap(),
                    ),
                    CallArg::Pure(bcs::to_bytes(validator.network_address()).unwrap()),
                    CallArg::Pure(bcs::to_bytes(validator.p2p_address()).unwrap()),
                    CallArg::Pure(bcs::to_bytes(validator.narwhal_primary_address()).unwrap()),
                    CallArg::Pure(bcs::to_bytes(validator.narwhal_worker_address()).unwrap()),
                    CallArg::Pure(bcs::to_bytes(&validator.gas_price()).unwrap()),
                    CallArg::Pure(bcs::to_bytes(&validator.commission_rate()).unwrap()),
                ];
                let (response, serialized_unsigned_transaction) = call_0x5(
                    context,
                    "request_add_validator_candidate",
                    args,
                    gas_budget,
                    tx_args.serialize_unsigned_transaction,
                )
                .await?;
                SomaValidatorCommandResponse::AddValidator {
                    response,
                    serialized_unsigned_transaction,
                }
            }

            SomaValidatorCommand::RemoveValidator { tx_args } => {
                // Only an active validator can leave committee.
                let _status =
                    check_status(context, HashSet::from([ValidatorStatus::Active])).await?;
                let gas_budget = tx_args.gas_budget.unwrap_or(DEFAULT_GAS_BUDGET);
                let (response, serialized_unsigned_transaction) = call_0x5(
                    context,
                    "request_remove_validator",
                    vec![],
                    gas_budget,
                    tx_args.serialize_unsigned_transaction,
                )
                .await?;
                SomaValidatorCommandResponse::RemoveValidator {
                    response,
                    serialized_unsigned_transaction,
                }
            }

            SomaValidatorCommand::DisplayMetadata {
                validator_address,
                json,
            } => {
                let validator_address = validator_address.unwrap_or(context.active_address()?);
                // Default display with json serialization for better UX.
                let soma_client = context.get_client().await?;
                display_metadata(&soma_client, validator_address, json.unwrap_or(true)).await?;
                SomaValidatorCommandResponse::DisplayMetadata
            }

            SomaValidatorCommand::UpdateMetadata { metadata, tx_args } => {
                let gas_budget = tx_args.gas_budget.unwrap_or(DEFAULT_GAS_BUDGET);
                let (response, serialized_unsigned_transaction) = update_metadata(
                    context,
                    metadata,
                    gas_budget,
                    tx_args.serialize_unsigned_transaction,
                )
                .await?;
                SomaValidatorCommandResponse::UpdateMetadata {
                    response,
                    serialized_unsigned_transaction,
                }
            }

            SomaValidatorCommand::SetCommissionRate {
                commission_rate,
                tx_args,
            } => {
                let gas_budget = tx_args.gas_budget.unwrap_or(DEFAULT_GAS_BUDGET);
                let (response, serialized_unsigned_transaction) = update_gas_price(
                    context,
                    operation_cap_id,
                    gas_price,
                    gas_budget,
                    tx_args.serialize_unsigned_transaction,
                )
                .await?;
                SomaValidatorCommandResponse::SetCommissionRate {
                    response,
                    serialized_unsigned_transaction,
                }
            }

            SomaValidatorCommand::ReportValidator {
                operation_cap_id,
                reportee_address,
                undo_report,
                tx_args,
            } => {
                let gas_budget = tx_args.gas_budget.unwrap_or(DEFAULT_GAS_BUDGET);
                let undo_report = undo_report.unwrap_or(false);
                let (response, serialized_unsigned_transaction) = report_validator(
                    context,
                    reportee_address,
                    operation_cap_id,
                    undo_report,
                    gas_budget,
                    tx_args.serialize_unsigned_transaction,
                )
                .await?;
                SomaValidatorCommandResponse::ReportValidator {
                    response,
                    serialized_unsigned_transaction,
                }
            }
        })
    }
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

async fn update_commission_rate(
    context: &mut WalletContext,
    commission_rate: u64,
    serialize_unsigned_transaction: bool,
) -> Result<(Option<SomaTransactionBlockResponse>, Option<String>)> {
    // TODO: Only active/pending validators can set commission rate.

    let args = vec![
        CallArg::Object(ObjectArg::ImmOrOwnedObject(cap_obj_ref)),
        CallArg::Pure(bcs::to_bytes(&gas_price).unwrap()),
    ];
    call_0x5(
        context,
        "request_set_gas_price",
        args,
        gas_budget,
        serialize_unsigned_transaction,
    )
    .await
}

async fn report_validator(
    context: &mut WalletContext,
    reportee_address: SomaAddress,
    undo_report: bool,
    serialize_unsigned_transaction: bool,
) -> Result<(Option<SomaTransactionBlockResponse>, Option<String>)> {
    let validator_address = summary.soma_address;
    // Only active validators can report/un-report.
    if !matches!(status, ValidatorStatus::Active) {
        anyhow::bail!(
            "Only active Validator can report/un-report Validators, but {} is {:?}.",
            validator_address,
            status
        );
    }
    let args = vec![
        CallArg::Object(ObjectArg::ImmOrOwnedObject(cap_obj_ref)),
        CallArg::Pure(bcs::to_bytes(&reportee_address).unwrap()),
    ];
    let function_name = if undo_report {
        "undo_report_validator"
    } else {
        "report_validator"
    };
    call_0x5(
        context,
        function_name,
        args,
        gas_budget,
        serialize_unsigned_transaction,
    )
    .await
}

impl Display for SomaValidatorCommandResponse {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut writer = String::new();
        match self {
            SomaValidatorCommandResponse::MakeValidatorInfo => {}
            SomaValidatorCommandResponse::DisplayMetadata => {}
            SomaValidatorCommandResponse::BecomeCandidate {
                response,
                serialized_unsigned_transaction,
            }
            | SomaValidatorCommandResponse::JoinCommittee {
                response,
                serialized_unsigned_transaction,
            }
            | SomaValidatorCommandResponse::LeaveCommittee {
                response,
                serialized_unsigned_transaction,
            }
            | SomaValidatorCommandResponse::UpdateMetadata {
                response,
                serialized_unsigned_transaction,
            }
            | SomaValidatorCommandResponse::UpdateGasPrice {
                response,
                serialized_unsigned_transaction,
            }
            | SomaValidatorCommandResponse::ReportValidator {
                response,
                serialized_unsigned_transaction,
            } => {
                if let Some(response) = response {
                    write!(writer, "{}", write_transaction_response(response)?)?;
                } else {
                    write!(
                        writer,
                        "Serialized transaction for signing: {:?}",
                        serialized_unsigned_transaction
                    )?;
                }
            }
            SomaValidatorCommandResponse::SerializedPayload(response) => {
                write!(writer, "Serialized payload: {}", response)?;
            }
            SomaValidatorCommandResponse::DisplayGasPriceUpdateRawTxn {
                data,
                serialized_data,
            } => {
                write!(
                    writer,
                    "Transaction: {:?}, \nSerialized transaction: {:?}",
                    data, serialized_data
                )?;
            }
            SomaValidatorCommandResponse::RegisterBridgeCommittee {
                execution_response,
                serialized_unsigned_transaction,
            }
            | SomaValidatorCommandResponse::UpdateBridgeCommitteeURL {
                execution_response,
                serialized_unsigned_transaction,
            } => {
                if let Some(response) = execution_response {
                    write!(writer, "{}", write_transaction_response(response)?)?;
                } else {
                    write!(
                        writer,
                        "Serialized transaction for signing: {:?}",
                        serialized_unsigned_transaction
                    )?;
                }
            }
        }
        write!(f, "{}", writer.trim_end_matches('\n'))
    }
}

pub fn write_transaction_response(
    response: &SomaTransactionBlockResponse,
) -> Result<String, fmt::Error> {
    // we requested with for full_content, so the following content should be available.
    let success = response.status_ok().unwrap();
    let lines = vec![
        String::from("----- Transaction Digest ----"),
        response.digest.to_string(),
        String::from("\n----- Transaction Data ----"),
        response.transaction.as_ref().unwrap().to_string(),
        String::from("----- Transaction Effects ----"),
        response.effects.as_ref().unwrap().to_string(),
    ];
    let mut writer = String::new();
    for line in lines {
        let colorized_line = if success { line.green() } else { line.red() };
        writeln!(writer, "{}", colorized_line)?;
    }
    Ok(writer)
}

impl Debug for SomaValidatorCommandResponse {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let string = serde_json::to_string_pretty(self);
        let s = match string {
            Ok(s) => s,
            Err(err) => format!("{err}").red().to_string(),
        };
        write!(f, "{}", s)
    }
}

impl SomaValidatorCommandResponse {
    pub fn print(&self, pretty: bool) {
        match self {
            // Don't print empty responses
            SomaValidatorCommandResponse::MakeValidatorInfo
            | SomaValidatorCommandResponse::DisplayMetadata => {}
            other => {
                let line = if pretty {
                    format!("{other}")
                } else {
                    format!("{:?}", other)
                };
                // Log line by line
                for line in line.lines() {
                    println!("{line}");
                }
            }
        }
    }
}

#[derive(Debug, Hash, PartialEq, Eq)]
pub enum ValidatorStatus {
    Active,
    Pending,
}

pub async fn get_validator_summary(
    client: &SomaClient,
    validator_address: SomaAddress,
) -> anyhow::Result<Option<(ValidatorStatus, SomaValidatorSummary)>> {
    let SomaSystemStateSummary {
        active_validators,
        pending_active_validators_id,
        ..
    } = client
        .governance_api()
        .get_latest_soma_system_state()
        .await?;
    let mut status = None;
    let mut active_validators = active_validators
        .into_iter()
        .map(|s| (s.soma_address, s))
        .collect::<BTreeMap<_, _>>();
    let validator_info = if active_validators.contains_key(&validator_address) {
        status = Some(ValidatorStatus::Active);
        Some(active_validators.remove(&validator_address).unwrap())
    } else {
        // Check panding validators
        get_pending_candidate_summary(validator_address, client, pending_active_validators_id)
            .await?
            .map(|v| v.into_soma_validator_summary())
            .tap_some(|_s| status = Some(ValidatorStatus::Pending))

        // TODO also check candidate and inactive valdiators
    };
    if validator_info.is_none() {
        return Ok(None);
    }
    // status is safe unwrap because it has to be Some when the code recahes here
    // validator_info is safe to unwrap because of the above check
    Ok(Some((status.unwrap(), validator_info.unwrap())))
}

async fn display_metadata(
    client: &SomaClient,
    validator_address: SomaAddress,
    json: bool,
) -> anyhow::Result<()> {
    match get_validator_summary(client, validator_address).await? {
        None => println!(
            "{} is not an active or pending Validator.",
            validator_address
        ),
        Some((status, info)) => {
            println!("{}'s valdiator status: {:?}", validator_address, status);
            if json {
                println!("{}", serde_json::to_string_pretty(&info)?);
            } else {
                println!("{:#?}", info);
            }
        }
    }
    Ok(())
}

#[derive(Subcommand)]
#[clap(rename_all = "kebab-case")]
pub enum MetadataUpdate {
    /// Update Network Address. Effectuate from next epoch.
    NetworkAddress { network_address: Multiaddr },
    /// Update Primary Address. Effectuate from next epoch.
    PrimaryAddress { primary_address: Multiaddr },
    /// Update Worker Address. Effectuate from next epoch.
    WorkerAddress { worker_address: Multiaddr },
    /// Update P2P Address. Effectuate from next epoch.
    P2pAddress { p2p_address: Multiaddr },
    /// Update Network Public Key. Effectuate from next epoch.
    NetworkPubKey {
        #[clap(name = "network-key-path")]
        file: PathBuf,
    },
    /// Update Worker Public Key. Effectuate from next epoch.
    WorkerPubKey {
        #[clap(name = "worker-key-path")]
        file: PathBuf,
    },
    /// Update Protocol Public Key and Proof and Possession. Effectuate from next epoch.
    ProtocolPubKey {
        #[clap(name = "protocol-key-path")]
        file: PathBuf,
    },
}

async fn update_metadata(
    context: &mut WalletContext,
    metadata: MetadataUpdate,
    gas_budget: u64,
    serialize_unsigned_transaction: bool,
) -> anyhow::Result<(Option<SomaTransactionBlockResponse>, Option<String>)> {
    use ValidatorStatus::*;
    match metadata {
        MetadataUpdate::NetworkAddress { network_address } => {
            // Check the network address to be in TCP.
            if !network_address.is_loosely_valid_tcp_addr() {
                bail!("Network address must be a TCP address");
            }
            let _status = check_status(context, HashSet::from([Pending, Active])).await?;
            let args = vec![CallArg::Pure(bcs::to_bytes(&network_address).unwrap())];
            call_0x5(
                context,
                "update_validator_next_epoch_network_address",
                args,
                gas_budget,
                serialize_unsigned_transaction,
            )
            .await
        }
        MetadataUpdate::PrimaryAddress { primary_address } => {
            let _status = check_status(context, HashSet::from([Pending, Active])).await?;
            let args = vec![CallArg::Pure(bcs::to_bytes(&primary_address).unwrap())];
            call_0x5(
                context,
                "update_validator_next_epoch_primary_address",
                args,
                gas_budget,
                serialize_unsigned_transaction,
            )
            .await
        }
        MetadataUpdate::WorkerAddress { worker_address } => {
            // Only an active validator can leave committee.
            let _status = check_status(context, HashSet::from([Pending, Active])).await?;
            let args = vec![CallArg::Pure(bcs::to_bytes(&worker_address).unwrap())];
            call_0x5(
                context,
                "update_validator_next_epoch_worker_address",
                args,
                gas_budget,
                serialize_unsigned_transaction,
            )
            .await
        }
        MetadataUpdate::P2pAddress { p2p_address } => {
            let _status = check_status(context, HashSet::from([Pending, Active])).await?;
            let args = vec![CallArg::Pure(bcs::to_bytes(&p2p_address).unwrap())];
            call_0x5(
                context,
                "update_validator_next_epoch_p2p_address",
                args,
                gas_budget,
                serialize_unsigned_transaction,
            )
            .await
        }
        MetadataUpdate::NetworkPubKey { file } => {
            let _status = check_status(context, HashSet::from([Pending, Active])).await?;
            let network_pub_key: NetworkPublicKey =
                read_network_keypair_from_file(file)?.public().clone();
            let args = vec![CallArg::Pure(
                bcs::to_bytes(&network_pub_key.as_bytes().to_vec()).unwrap(),
            )];
            call_0x5(
                context,
                "update_validator_next_epoch_network_pubkey",
                args,
                gas_budget,
                serialize_unsigned_transaction,
            )
            .await
        }
        MetadataUpdate::WorkerPubKey { file } => {
            let _status = check_status(context, HashSet::from([Pending, Active])).await?;
            let worker_pub_key: NetworkPublicKey =
                read_network_keypair_from_file(file)?.public().clone();
            let args = vec![CallArg::Pure(
                bcs::to_bytes(&worker_pub_key.as_bytes().to_vec()).unwrap(),
            )];
            call_0x5(
                context,
                "update_validator_next_epoch_worker_pubkey",
                args,
                gas_budget,
                serialize_unsigned_transaction,
            )
            .await
        }
        MetadataUpdate::ProtocolPubKey { file } => {
            let _status = check_status(context, HashSet::from([Pending, Active])).await?;
            let soma_address = context.active_address()?;
            let protocol_key_pair: AuthorityKeyPair = read_authority_keypair_from_file(file)?;
            let protocol_pub_key: AuthorityPublicKey = protocol_key_pair.public().clone();
            let pop = generate_proof_of_possession(&protocol_key_pair, soma_address);
            let args = vec![
                CallArg::Pure(
                    bcs::to_bytes(&AuthorityPublicKeyBytes::from_bytes(
                        protocol_pub_key.as_bytes(),
                    )?)
                    .unwrap(),
                ),
                CallArg::Pure(bcs::to_bytes(&pop.as_ref().to_vec()).unwrap()),
            ];
            call_0x5(
                context,
                "update_validator_next_epoch_protocol_pubkey",
                args,
                gas_budget,
                serialize_unsigned_transaction,
            )
            .await
        }
    }
}

async fn check_status(
    context: &mut WalletContext,
    allowed_status: HashSet<ValidatorStatus>,
) -> Result<ValidatorStatus> {
    let soma_client = context.get_client().await?;
    let validator_address = context.active_address()?;
    let summary = get_validator_summary(&soma_client, validator_address).await?;
    if summary.is_none() {
        bail!("{validator_address} is not a Validator.");
    }
    let (status, _summary) = summary.unwrap();
    if allowed_status.contains(&status) {
        return Ok(status);
    }
    bail!(
        "Validator {validator_address} is {:?}, this operation is not supported in this tool or prohibited.",
        status
    )
}

async fn construct_unsigned_0x5_txn(
    context: &mut WalletContext,
    sender: SomaAddress,
    function: &'static str,
    call_args: Vec<CallArg>,
    gas_budget: u64,
) -> anyhow::Result<TransactionData> {
    let soma_client = context.get_client().await?;
    let mut args = vec![CallArg::SOMA_SYSTEM_MUT];
    args.extend(call_args);
    let rgp = soma_client
        .governance_api()
        .get_reference_gas_price()
        .await?;

    let gas_obj_ref = get_gas_obj_ref(sender, &soma_client, gas_budget).await?;
    TransactionData::new_move_call(
        sender,
        SOMA_SYSTEM_PACKAGE_ID,
        ident_str!("soma_system").to_owned(),
        ident_str!(function).to_owned(),
        vec![],
        gas_obj_ref,
        args,
        gas_budget,
        rgp,
    )
}

async fn call_0x5(
    context: &mut WalletContext,
    function: &'static str,
    call_args: Vec<CallArg>,
    gas_budget: u64,
    serialize_unsigned_transaction: bool,
) -> anyhow::Result<(Option<SomaTransactionBlockResponse>, Option<String>)> {
    let sender = context.active_address()?;
    let tx_data =
        construct_unsigned_0x5_txn(context, sender, function, call_args, gas_budget).await?;
    if serialize_unsigned_transaction {
        let serialized_data = Base64::encode(bcs::to_bytes(&tx_data)?);
        return Ok((None, Some(serialized_data)));
    }
    let signature = context
        .config
        .keystore
        .sign_secure(&sender, &tx_data, Intent::soma_transaction())
        .await?;
    let transaction = Transaction::from_data(tx_data, vec![signature]);
    let soma_client = context.get_client().await?;
    let response = soma_client
        .quorum_driver_api()
        .execute_transaction_block(
            transaction,
            SomaTransactionBlockResponseOptions::new()
                .with_input()
                .with_effects(),
            Some(soma_types::quorum_driver_types::ExecuteTransactionRequestType::WaitForLocalExecution),
        )
        .await
        .map_err(|err| anyhow::anyhow!(err.to_string()))?;
    Ok((Some(response), None))
}

async fn get_pending_candidate_summary(
    validator_address: SomaAddress,
    soma_client: &SomaClient,
    pending_active_validators_id: ObjectID,
) -> anyhow::Result<Option<ValidatorV1>> {
    let pending_validators = soma_client
        .read_api()
        .get_dynamic_fields(pending_active_validators_id, None, None)
        .await?
        .data
        .into_iter()
        .map(|dyi| dyi.object_id)
        .collect::<Vec<_>>();
    let resps = soma_client
        .read_api()
        .multi_get_object_with_options(
            pending_validators,
            SomaObjectDataOptions::default().with_bcs(),
        )
        .await?;
    for resp in resps {
        // We always expect an objectId from the response as one of data/error should be included.
        let object_id = resp.object_id()?;
        let bcs = resp.move_object_bcs().ok_or_else(|| {
            anyhow::anyhow!(
                "Object {} does not exist or does not return bcs bytes",
                object_id
            )
        })?;
        let field = bcs::from_bytes::<Field<u64, ValidatorV1>>(bcs).map_err(|e| {
            anyhow::anyhow!(
                "Can't convert bcs bytes of object {} to ValidatorV1: {}",
                object_id,
                e,
            )
        })?;
        if field.value.verified_metadata().soma_address == validator_address {
            return Ok(Some(field.value));
        }
    }
    Ok(None)
}
