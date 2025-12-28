use crate::response::{
    AddressesOutput, BalanceOutput, ChainInfoOutput, ClientCommandResponse, EnvsOutput,
    GasCoinsOutput, NewAddressOutput, NewEnvOutput, ObjectOutput, ObjectsOutput,
    RemoveAddressOutput, SimulationResponse, SwitchOutput, TransactionQueryResponse,
    TransactionResponse, TransactionStatus,
};
use anyhow::{anyhow, bail, ensure, Result};
use bip32::DerivationPath;
use clap::*;
use colored::Colorize;
use fastcrypto::encoding::{Base64, Encoding};
use fastcrypto::traits::ToFromBytes;
use futures::TryStreamExt;
use protocol_config::ProtocolVersion;
use rpc::types::ObjectType;
use rpc::utils::field::{FieldMask, FieldMaskUtil};
use sdk::SomaClient;
use std::path::PathBuf;
use std::str::FromStr as _;
use types::effects::{ExecutionStatus, TransactionEffectsAPI as _};

use sdk::{
    client_config::SomaEnv,
    transaction_builder::{ExecutionOptions, TransactionBuilder},
    wallet_context::WalletContext,
};
use soma_keys::key_derive;
use soma_keys::key_identity::KeyIdentity;
use soma_keys::keystore::AccountKeystore;
use types::{
    base::SomaAddress,
    crypto::{GenericSignature, SignatureScheme},
    digests::TransactionDigest,
    envelope::Envelope,
    object::{ObjectID, ObjectRef},
    transaction::{SenderSignedData, Transaction, TransactionData, TransactionKind},
};

#[derive(Parser)]
#[clap(rename_all = "kebab-case")]
pub enum SomaClientCommands {
    // =========================================================================
    // ADDRESS MANAGEMENT
    // =========================================================================
    /// Display the current active address
    #[clap(name = "active-address")]
    ActiveAddress,

    /// List all addresses managed by the wallet
    #[clap(name = "addresses")]
    Addresses {
        /// Sort by alias instead of address
        #[clap(long, short = 's')]
        sort_by_alias: bool,
    },

    /// Generate a new address and keypair
    #[clap(name = "new-address")]
    NewAddress {
        /// Key scheme: ed25519, secp256k1, or secp256r1
        key_scheme: SignatureScheme,
        /// Optional alias (must start with letter, can contain letters/digits/hyphens/underscores)
        alias: Option<String>,
        /// Word length: word12, word15, word18, word21, word24 (default: word12)
        word_length: Option<String>,
        /// Custom derivation path
        derivation_path: Option<DerivationPath>,
    },

    /// Remove an address from the wallet
    #[clap(name = "remove-address")]
    RemoveAddress {
        /// Address or alias to remove
        alias_or_address: String,
    },

    /// Switch active address or environment
    #[clap(name = "switch")]
    Switch {
        /// Address or alias to make active
        #[clap(long)]
        address: Option<KeyIdentity>,
        /// Environment alias to make active
        #[clap(long)]
        env: Option<String>,
    },

    // =========================================================================
    // ENVIRONMENT MANAGEMENT
    // =========================================================================
    /// Display the current active environment
    #[clap(name = "active-env")]
    ActiveEnv,

    /// List all configured environments
    #[clap(name = "envs")]
    Envs,

    /// Add a new environment
    #[clap(name = "new-env")]
    NewEnv {
        /// Alias for the environment
        #[clap(long)]
        alias: String,
        /// RPC URL
        #[clap(long, value_hint = ValueHint::Url)]
        rpc: String,
        /// Basic auth (format: username:password)
        #[clap(long)]
        basic_auth: Option<String>,
    },

    // =========================================================================
    // CHAIN INFORMATION
    // =========================================================================
    /// Query the chain identifier from the RPC endpoint
    #[clap(name = "chain-identifier")]
    ChainIdentifier,

    // =========================================================================
    // OBJECT QUERIES
    // =========================================================================
    /// Get information about a specific object
    #[clap(name = "object")]
    Object {
        /// Object ID to fetch
        #[clap(name = "object_id")]
        id: ObjectID,
        /// Return BCS serialized data
        #[clap(long)]
        bcs: bool,
    },

    /// List all objects owned by an address
    #[clap(name = "objects")]
    Objects {
        /// Owner address (defaults to active address)
        #[clap(name = "owner_address")]
        address: Option<KeyIdentity>,
    },

    /// List gas coins owned by an address
    #[clap(name = "gas")]
    Gas {
        /// Owner address (defaults to active address)
        #[clap(name = "owner_address")]
        address: Option<KeyIdentity>,
    },

    /// Display coin balance for an address
    #[clap(name = "balance")]
    Balance {
        /// Address to check (defaults to active address)
        #[arg(value_parser)]
        address: Option<KeyIdentity>,
        /// Show individual coin details
        #[clap(long)]
        with_coins: bool,
    },

    // =========================================================================
    // TRANSACTION QUERIES
    // =========================================================================
    /// Get details of an executed transaction
    #[clap(name = "tx-block")]
    GetTransaction {
        /// Transaction digest
        #[clap(name = "digest")]
        digest: TransactionDigest,
    },

    // =========================================================================
    // TRANSFER TRANSACTIONS
    // =========================================================================
    /// Transfer an object to a recipient
    #[clap(name = "transfer")]
    Transfer {
        /// Recipient address or alias
        #[clap(long)]
        to: KeyIdentity,
        /// Object ID to transfer
        #[clap(long)]
        object_id: ObjectID,
        #[clap(flatten)]
        payment: PaymentArgs,
        #[clap(flatten)]
        processing: TxProcessingArgs,
    },

    /// Transfer SOMA coins to a recipient
    #[clap(name = "transfer-soma")]
    TransferSoma {
        /// Recipient address or alias
        #[clap(long)]
        to: KeyIdentity,
        /// Coin object ID (also used for gas)
        #[clap(long)]
        soma_coin_object_id: ObjectID,
        /// Amount to transfer (if not specified, transfers entire coin)
        #[clap(long)]
        amount: Option<u64>,
        #[clap(flatten)]
        processing: TxProcessingArgs,
    },

    /// Pay SOMA to multiple recipients
    #[clap(name = "pay")]
    PaySoma {
        /// Input coin object IDs (first is also used for gas)
        #[clap(long, num_args(1..))]
        input_coins: Vec<ObjectID>,
        /// Recipient addresses or aliases
        #[clap(long, num_args(1..))]
        recipients: Vec<KeyIdentity>,
        /// Amounts to send to each recipient
        #[clap(long, num_args(1..))]
        amounts: Vec<u64>,
        #[clap(flatten)]
        processing: TxProcessingArgs,
    },

    // =========================================================================
    // TRANSACTION EXECUTION HELPERS
    // =========================================================================
    /// Execute an already-serialized transaction
    #[clap(name = "serialized-tx")]
    SerializedTx {
        /// Base64-encoded BCS-serialized TransactionData
        tx_bytes: String,
        #[clap(flatten)]
        processing: TxProcessingArgs,
    },

    /// Execute using pre-signed transaction bytes and signatures
    #[clap(name = "execute-signed-tx")]
    ExecuteSignedTx {
        /// Base64-encoded unsigned transaction data
        #[clap(long)]
        tx_bytes: String,
        /// Base64-encoded signatures (flag || signature || pubkey)
        #[clap(long)]
        signatures: Vec<String>,
    },

    /// Execute a combined sender-signed transaction
    #[clap(name = "execute-combined-signed-tx")]
    ExecuteCombinedSignedTx {
        /// Base64-encoded SenderSignedData
        #[clap(long)]
        signed_tx_bytes: String,
    },
}

/// Arguments related to transaction processing
#[derive(Args, Debug, Default, Clone)]
pub struct TxProcessingArgs {
    /// Compute the transaction digest and print it out, but do not execute.
    #[arg(long)]
    pub tx_digest: bool,

    /// Perform a dry run (simulation) of the transaction, without executing it.
    #[arg(long)]
    pub simulate: bool,

    /// Serialize the unsigned transaction data (base64) instead of executing.
    /// Use with `soma client execute-signed-tx --tx-bytes <TX_BYTES>`.
    #[arg(long)]
    pub serialize_unsigned_transaction: bool,

    /// Serialize the signed transaction data (base64) instead of executing.
    /// Use with `soma client execute-combined-signed-tx --signed-tx-bytes <SIGNED_TX_BYTES>`.
    #[arg(long)]
    pub serialize_signed_transaction: bool,
}

/// Arguments for providing gas payment
#[derive(Args, Debug, Default, Clone)]
pub struct PaymentArgs {
    /// Object ID of the gas coin to use. If not provided, one is selected automatically.
    #[clap(long)]
    pub gas: Option<ObjectID>,
}

impl SomaClientCommands {
    pub async fn execute(self, context: &mut WalletContext) -> Result<ClientCommandResponse> {
        match self {
            // =================================================================
            // ADDRESS MANAGEMENT
            // =================================================================
            SomaClientCommands::ActiveAddress => {
                let address = context.active_address().ok();
                Ok(ClientCommandResponse::ActiveAddress(address))
            }

            SomaClientCommands::Addresses { sort_by_alias } => {
                let active_address = context.active_address()?;
                let mut addresses: Vec<(String, SomaAddress)> = context
                    .config
                    .keystore
                    .addresses_with_alias()
                    .into_iter()
                    .map(|(address, alias)| (alias.alias.to_string(), *address))
                    .collect();

                if sort_by_alias {
                    addresses.sort_by(|a, b| a.0.cmp(&b.0));
                }

                Ok(ClientCommandResponse::Addresses(AddressesOutput {
                    active_address,
                    addresses,
                }))
            }

            SomaClientCommands::NewAddress {
                key_scheme,
                alias,
                derivation_path,
                word_length,
            } => {
                let (address, keypair, scheme, phrase) =
                    key_derive::generate_new_key(key_scheme, derivation_path, word_length)
                        .map_err(|e| anyhow!("Failed to generate new key: {}", e))?;

                context
                    .config
                    .keystore
                    .import(alias.clone(), keypair)
                    .await?;

                let alias = match alias {
                    Some(a) => a,
                    None => context.config.keystore.get_alias(&address)?,
                };

                Ok(ClientCommandResponse::NewAddress(NewAddressOutput {
                    alias,
                    address,
                    key_scheme: scheme,
                    recovery_phrase: phrase,
                }))
            }

            SomaClientCommands::RemoveAddress { alias_or_address } => {
                let identity = KeyIdentity::from_str(&alias_or_address)
                    .map_err(|e| anyhow!("Invalid address or alias: {}", e))?;
                let address: SomaAddress = context.config.keystore.get_by_identity(&identity)?;

                context.config.keystore.remove(address).await?;

                Ok(ClientCommandResponse::RemoveAddress(RemoveAddressOutput {
                    alias_or_address,
                    address,
                }))
            }

            SomaClientCommands::Switch { address, env } => {
                if address.is_none() && env.is_none() {
                    bail!("Please specify --address or --env (or both)");
                }

                let mut switched_address = None;
                let mut switched_env = None;

                if let Some(addr) = address {
                    let resolved = context.get_identity_address(Some(addr))?;
                    if !context.config.keystore.addresses().contains(&resolved) {
                        bail!("Address {} not managed by wallet", resolved);
                    }
                    context.config.active_address = Some(resolved);
                    switched_address = Some(resolved);
                }

                if let Some(ref env_alias) = env {
                    ensure!(
                        context.config.get_env(&Some(env_alias.clone())).is_some(),
                        "Environment '{}' not found. Use 'soma client new-env' to add it.",
                        env_alias
                    );
                    context.config.active_env = Some(env_alias.clone());
                    switched_env = Some(env_alias.clone());
                }

                context.config.save()?;

                Ok(ClientCommandResponse::Switch(SwitchOutput {
                    address: switched_address,
                    env: switched_env,
                }))
            }

            // =================================================================
            // ENVIRONMENT MANAGEMENT
            // =================================================================
            SomaClientCommands::ActiveEnv => {
                let env = context.get_active_env().ok().map(|e| e.alias.clone());
                Ok(ClientCommandResponse::ActiveEnv(env))
            }

            SomaClientCommands::Envs => {
                let envs = context.config.envs.clone();
                let active = context.get_active_env().ok().map(|e| e.alias.clone());
                Ok(ClientCommandResponse::Envs(EnvsOutput { envs, active }))
            }

            SomaClientCommands::NewEnv {
                alias,
                rpc,
                basic_auth,
            } => {
                if context.config.envs.iter().any(|e| e.alias == alias) {
                    bail!("Environment '{}' already exists", alias);
                }

                let env = SomaEnv {
                    alias: alias.clone(),
                    rpc,
                    basic_auth,
                    chain_id: None,
                };

                // Verify connection
                env.create_rpc_client(None).await?;

                context.config.envs.push(env.clone());
                context.config.save()?;

                // Cache chain ID
                let client = context.get_client().await?;
                let chain_id = context.cache_chain_id(&client).await?;

                Ok(ClientCommandResponse::NewEnv(NewEnvOutput {
                    alias,
                    chain_id,
                }))
            }

            // =================================================================
            // CHAIN INFORMATION
            // =================================================================
            SomaClientCommands::ChainIdentifier => {
                let client = context.get_client().await?;
                let chain_id = context.cache_chain_id(&client).await?;
                let server_version = client.get_server_version().await.ok();

                Ok(ClientCommandResponse::ChainInfo(ChainInfoOutput {
                    chain_id,
                    server_version,
                }))
            }

            // =================================================================
            // OBJECT QUERIES
            // =================================================================
            SomaClientCommands::Object { id, bcs } => {
                let client = context.get_client().await?;
                let object = client
                    .get_object(id)
                    .await
                    .map_err(|e| anyhow!("Failed to get object: {}", e))?;

                Ok(ClientCommandResponse::Object(ObjectOutput::from_object(
                    &object, bcs,
                )))
            }

            SomaClientCommands::Objects { address } => {
                let address = context.get_identity_address(address)?;
                let client = context.get_client().await?;

                let mut request = rpc::proto::soma::ListOwnedObjectsRequest::default();
                request.owner = Some(address.to_string());
                request.page_size = Some(100);
                request.read_mask = Some(FieldMask::from_paths([
                    "object_id",
                    "version",
                    "digest",
                    "object_type",
                    "owner",
                ]));

                let stream = client.list_owned_objects(request).await;
                tokio::pin!(stream);

                let mut objects = Vec::new();
                while let Some(obj) = stream.try_next().await? {
                    objects.push(ObjectOutput::from_object(&obj, false));
                }

                Ok(ClientCommandResponse::Objects(ObjectsOutput {
                    address,
                    objects,
                }))
            }

            SomaClientCommands::Gas { address } => {
                let address = context.get_identity_address(address)?;
                let gas_objects = context
                    .get_all_gas_objects_owned_by_address(address)
                    .await?;

                // Fetch full objects to get balances
                let client = context.get_client().await?;
                let mut coins = Vec::new();
                for obj_ref in gas_objects {
                    if let Ok(obj) = client.get_object(obj_ref.0).await {
                        if let Some(balance) = obj.as_coin() {
                            coins.push((obj_ref, balance));
                        }
                    }
                }

                Ok(ClientCommandResponse::Gas(GasCoinsOutput {
                    address,
                    coins,
                }))
            }

            SomaClientCommands::Balance {
                address,
                with_coins,
            } => {
                let address = context.get_identity_address(address)?;
                let client = context.get_client().await?;

                let mut request = rpc::proto::soma::ListOwnedObjectsRequest::default();
                request.owner = Some(address.to_string());
                request.object_type = Some(ObjectType::Coin.into());
                request.page_size = Some(1000);
                request.read_mask = Some(FieldMask::from_paths([
                    "object_id",
                    "version",
                    "digest",
                    "contents",
                ]));

                let stream = client.list_owned_objects(request).await;
                tokio::pin!(stream);

                let mut total_balance: u128 = 0;
                let mut coin_details = Vec::new();

                while let Some(obj) = stream.try_next().await? {
                    if let Some(coin) = obj.as_coin() {
                        total_balance += coin as u128;
                        if with_coins {
                            coin_details.push((obj.id(), coin));
                        }
                    }
                }

                Ok(ClientCommandResponse::Balance(BalanceOutput {
                    address,
                    total_balance,
                    coin_count: coin_details.len(),
                    coins: if with_coins { Some(coin_details) } else { None },
                }))
            }

            // =================================================================
            // TRANSACTION QUERIES
            // =================================================================
            SomaClientCommands::GetTransaction { digest } => {
                let client = context.get_client().await?;

                let result = client
                    .get_transaction(digest)
                    .await
                    .map_err(|e| anyhow!("Failed to get transaction: {}", e))?;

                Ok(ClientCommandResponse::TransactionQuery(
                    TransactionQueryResponse::from_query_result(&result),
                ))
            }

            // =================================================================
            // TRANSFER TRANSACTIONS
            // =================================================================
            SomaClientCommands::Transfer {
                to,
                object_id,
                payment,
                processing,
            } => {
                let sender = context.get_object_owner(&object_id).await?;
                let recipient = context.get_identity_address(Some(to))?;
                let client = context.get_client().await?;

                // Get the object reference
                let object = client
                    .get_object(object_id)
                    .await
                    .map_err(|e| anyhow!("Failed to get object: {}", e))?;
                let object_ref = object.compute_object_reference();

                let kind = TransactionKind::TransferObjects {
                    objects: vec![object_ref],
                    recipient,
                };

                let gas = resolve_gas_payment(context, &payment).await?;
                execute_or_serialize(context, sender, kind, gas, processing).await
            }

            SomaClientCommands::TransferSoma {
                to,
                soma_coin_object_id,
                amount,
                processing,
            } => {
                let sender = context.get_object_owner(&soma_coin_object_id).await?;
                let recipient = context.get_identity_address(Some(to))?;
                let client = context.get_client().await?;

                // Get coin reference
                let coin = client
                    .get_object(soma_coin_object_id)
                    .await
                    .map_err(|e| anyhow!("Failed to get coin: {}", e))?;
                let coin_ref = coin.compute_object_reference();

                let kind = TransactionKind::TransferCoin {
                    coin: coin_ref,
                    amount,
                    recipient,
                };

                // Use the coin itself as gas
                execute_or_serialize(context, sender, kind, Some(coin_ref), processing).await
            }

            SomaClientCommands::PaySoma {
                input_coins,
                recipients,
                amounts,
                processing,
            } => {
                ensure!(
                    !input_coins.is_empty(),
                    "At least one input coin is required"
                );
                ensure!(!recipients.is_empty(), "At least one recipient is required");
                ensure!(
                    recipients.len() == amounts.len(),
                    "Number of recipients ({}) must match number of amounts ({})",
                    recipients.len(),
                    amounts.len()
                );

                let sender = context.get_object_owner(&input_coins[0]).await?;
                let client = context.get_client().await?;

                // Resolve recipient addresses
                let recipient_addresses: Vec<SomaAddress> = recipients
                    .into_iter()
                    .map(|r| context.get_identity_address(Some(r)))
                    .collect::<Result<Vec<_>>>()?;

                // Get coin references
                let mut coin_refs = Vec::new();
                for coin_id in &input_coins {
                    let coin = client
                        .get_object(*coin_id)
                        .await
                        .map_err(|e| anyhow!("Failed to get coin {}: {}", coin_id, e))?;
                    coin_refs.push(coin.compute_object_reference());
                }

                let kind = TransactionKind::PayCoins {
                    coins: coin_refs.clone(),
                    amounts: Some(amounts),
                    recipients: recipient_addresses,
                };

                // Use first coin as gas
                execute_or_serialize(context, sender, kind, Some(coin_refs[0]), processing).await
            }

            // =================================================================
            // TRANSACTION EXECUTION HELPERS
            // =================================================================
            SomaClientCommands::SerializedTx {
                tx_bytes,
                processing,
            } => {
                let bytes =
                    Base64::decode(&tx_bytes).map_err(|_| anyhow!("Invalid Base64 encoding"))?;

                let tx_data: TransactionData = bcs::from_bytes(&bytes)
                    .map_err(|_| anyhow!("Failed to parse TransactionData"))?;

                let sender = tx_data.sender();
                let kind = tx_data.kind().clone();
                let gas = tx_data.gas().first().cloned();

                execute_or_serialize(context, sender, kind, gas, processing).await
            }

            SomaClientCommands::ExecuteSignedTx {
                tx_bytes,
                signatures,
            } => {
                let data_bytes = Base64::decode(&tx_bytes)
                    .map_err(|_| anyhow!("Invalid Base64 encoding for tx_bytes"))?;

                let data: TransactionData = bcs::from_bytes(&data_bytes).map_err(|_| {
                    anyhow!(
                        "Failed to parse tx_bytes. \
                         Ensure it matches output from --serialize-unsigned-transaction"
                    )
                })?;

                let mut sigs = Vec::new();
                for sig_str in signatures {
                    let sig_bytes = Base64::decode(&sig_str)
                        .map_err(|_| anyhow!("Invalid Base64 encoding for signature"))?;
                    let sig = GenericSignature::from_bytes(&sig_bytes)
                        .map_err(|_| anyhow!("Invalid signature format"))?;
                    sigs.push(sig);
                }

                let transaction = Transaction::from_generic_sig_data(data, sigs);
                let response = context.execute_transaction_may_fail(transaction).await?;

                Ok(ClientCommandResponse::Transaction(
                    TransactionResponse::from_response(&response),
                ))
            }

            SomaClientCommands::ExecuteCombinedSignedTx { signed_tx_bytes } => {
                let bytes = Base64::decode(&signed_tx_bytes)
                    .map_err(|_| anyhow!("Invalid Base64 encoding"))?;

                let sender_signed: SenderSignedData = bcs::from_bytes(&bytes).map_err(|_| {
                    anyhow!(
                        "Failed to parse SenderSignedData. \
                         Ensure it matches output from --serialize-signed-transaction"
                    )
                })?;

                let transaction =
                    Envelope::<SenderSignedData, types::crypto::EmptySignInfo>::new(sender_signed);
                let response = context.execute_transaction_may_fail(transaction).await?;

                Ok(ClientCommandResponse::Transaction(
                    TransactionResponse::from_response(&response),
                ))
            }
        }
    }
}

/// Resolve optional gas payment argument to an ObjectRef
async fn resolve_gas_payment(
    context: &WalletContext,
    payment: &PaymentArgs,
) -> Result<Option<ObjectRef>> {
    match &payment.gas {
        Some(gas_id) => {
            let client = context.get_client().await?;
            let gas_obj = client
                .get_object(*gas_id)
                .await
                .map_err(|e| anyhow!("Failed to get gas object: {}", e))?;
            Ok(Some(gas_obj.compute_object_reference()))
        }
        None => Ok(None),
    }
}

/// Execute a transaction or serialize it based on processing args
async fn execute_or_serialize(
    context: &mut WalletContext,
    sender: SomaAddress,
    kind: TransactionKind,
    gas: Option<ObjectRef>,
    processing: TxProcessingArgs,
) -> Result<ClientCommandResponse> {
    ensure!(
        !(processing.serialize_unsigned_transaction && processing.serialize_signed_transaction),
        "Cannot specify both --serialize-unsigned-transaction and --serialize-signed-transaction"
    );

    // Check protocol version compatibility
    let client = context.get_client().await?;
    // Check protocol version compatibility (non-blocking on failure)
    let _ = check_protocol_version_and_warn(&client).await;

    // Build transaction data
    let builder = TransactionBuilder::new(context);
    let tx_data = builder.build_transaction_data(sender, kind, gas).await?;
    drop(builder); // Release the borrow before execute_transaction

    // Handle tx-digest-only mode
    if processing.tx_digest {
        return Ok(ClientCommandResponse::TransactionDigest(tx_data.digest()));
    }

    // Handle simulation mode (no signature required)
    if processing.simulate {
        let result = client
            .simulate_transaction(&tx_data)
            .await
            .map_err(|e| anyhow!("Simulation failed: {}", e))?;

        let status = match result.effects.status() {
            ExecutionStatus::Success => TransactionStatus::Success,
            ExecutionStatus::Failure { error } => TransactionStatus::Failure {
                error: format!("{}", error),
            },
        };

        return Ok(ClientCommandResponse::Simulation(SimulationResponse {
            status,
            gas_used: result.effects.transaction_fee().total_fee,
            created: result
                .effects
                .created()
                .into_iter()
                .map(Into::into)
                .collect(),
            mutated: result
                .effects
                .mutated_excluding_gas()
                .into_iter()
                .map(Into::into)
                .collect(),
            deleted: result
                .effects
                .deleted()
                .into_iter()
                .map(Into::into)
                .collect(),
            balance_changes: result.balance_changes,
        }));
    }

    // Handle serialize-unsigned mode
    if processing.serialize_unsigned_transaction {
        let bytes = bcs::to_bytes(&tx_data)?;
        let encoded = Base64::encode(&bytes);
        return Ok(ClientCommandResponse::SerializedUnsignedTransaction(
            encoded,
        ));
    }

    // Sign the transaction
    let tx = context.sign_transaction(&tx_data).await;

    // Handle serialize-signed mode
    if processing.serialize_signed_transaction {
        let bytes = bcs::to_bytes(tx.data())?;
        let encoded = Base64::encode(&bytes);
        return Ok(ClientCommandResponse::SerializedSignedTransaction(encoded));
    }

    // Execute the transaction
    let response = context.execute_transaction_may_fail(tx).await?;

    Ok(ClientCommandResponse::Transaction(
        TransactionResponse::from_response(&response),
    ))
}

/// Warn the user if the CLI falls behind more than 2 protocol versions.
async fn check_protocol_version_and_warn(client: &SomaClient) -> Result<()> {
    let on_chain_version = match client.get_protocol_version().await {
        Ok(v) => v,
        Err(_) => return Ok(()), // Silently skip if we can't fetch
    };

    let cli_version = protocol_config::ProtocolVersion::MAX.as_u64();

    if (cli_version + 2) < on_chain_version {
        eprintln!(
            "{}",
            format!(
                // TODO: replace this with the actual url when docs are ready
                "[warning] CLI's protocol version is {cli_version}, but the active \
                network's protocol version is {on_chain_version}. \
                \nConsider installing the latest version of the CLI - \
                https://docs.soma.org/guides/getting-started/install"
            )
            .yellow()
            .bold()
        );
    }

    Ok(())
}

// impl Display for SomaClientCommandResult {
//     fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
//         let mut writer = String::new();
//         match self {
//             SomaClientCommandResult::Addresses(addresses) => {
//                 let mut builder = TableBuilder::default();
//                 builder.set_header(vec!["alias", "address", "active address"]);
//                 for (alias, address) in &addresses.addresses {
//                     let active_address = if address == &addresses.active_address {
//                         "*".to_string()
//                     } else {
//                         "".to_string()
//                     };
//                     builder.push_record([alias.to_string(), address.to_string(), active_address]);
//                 }
//                 let mut table = builder.build();
//                 let style = TableStyle::rounded();
//                 table.with(style);
//                 write!(f, "{}", table)?
//             }
//             SomaClientCommandResult::Balance(coins, with_coins) => {
//                 if coins.is_empty() {
//                     return write!(f, "No coins found for this address.");
//                 }
//                 let mut builder = TableBuilder::default();
//                 pretty_print_balance(coins, &mut builder, *with_coins);
//                 let mut table = builder.build();
//                 table.with(TablePanel::header("Balance of coins owned by this address"));
//                 table.with(TableStyle::rounded().horizontals([HorizontalLine::new(
//                     1,
//                     TableStyle::modern().get_horizontal(),
//                 )]));
//                 table.with(tabled::settings::style::BorderSpanCorrection);
//                 write!(f, "{}", table)?;
//             }

//             SomaClientCommandResult::Gas(gas_coins) => {
//                 let gas_coins = gas_coins
//                     .iter()
//                     .map(GasCoinOutput::from)
//                     .collect::<Vec<_>>();
//                 if gas_coins.is_empty() {
//                     write!(f, "No gas coins are owned by this address")?;
//                     return Ok(());
//                 }

//                 let mut builder = TableBuilder::default();
//                 builder.set_header(vec![
//                     "gasCoinId",
//                     "shannonsBalance (SHNS)",
//                     "somaBalance (SOMA)",
//                 ]);
//                 for coin in &gas_coins {
//                     builder.push_record(vec![
//                         coin.gas_coin_id.to_string(),
//                         coin.shannons_balance.to_string(),
//                         coin.soma_balance.to_string(),
//                     ]);
//                 }
//                 let mut table = builder.build();
//                 table.with(TableStyle::rounded());
//                 if gas_coins.len() > 10 {
//                     table.with(TablePanel::header(format!(
//                         "Showing {} gas coins and their balances.",
//                         gas_coins.len()
//                     )));
//                     table.with(TablePanel::footer(format!(
//                         "Showing {} gas coins and their balances.",
//                         gas_coins.len()
//                     )));
//                     table.with(TableStyle::rounded().horizontals([
//                         HorizontalLine::new(1, TableStyle::modern().get_horizontal()),
//                         HorizontalLine::new(2, TableStyle::modern().get_horizontal()),
//                         HorizontalLine::new(
//                             gas_coins.len() + 2,
//                             TableStyle::modern().get_horizontal(),
//                         ),
//                     ]));
//                     table.with(tabled::settings::style::BorderSpanCorrection);
//                 }
//                 write!(f, "{}", table)?;
//             }
//             SomaClientCommandResult::NewAddress(new_address) => {
//                 let mut builder = TableBuilder::default();
//                 builder.push_record(vec!["alias", new_address.alias.as_str()]);
//                 builder.push_record(vec!["address", new_address.address.to_string().as_str()]);
//                 builder.push_record(vec![
//                     "keyScheme",
//                     new_address.key_scheme.to_string().as_str(),
//                 ]);
//                 builder.push_record(vec![
//                     "recoveryPhrase",
//                     new_address.recovery_phrase.to_string().as_str(),
//                 ]);

//                 let mut table = builder.build();
//                 table.with(TableStyle::rounded());
//                 table.with(TablePanel::header(
//                     "Created new keypair and saved it to keystore.",
//                 ));

//                 table.with(
//                     TableModify::new(TableCell::new(0, 0))
//                         .with(TableBorder::default().corner_bottom_right('┬')),
//                 );
//                 table.with(
//                     TableModify::new(TableCell::new(0, 0))
//                         .with(TableBorder::default().corner_top_right('─')),
//                 );

//                 write!(f, "{}", table)?
//             }
//             SomaClientCommandResult::RemoveAddress(remove_address) => {
//                 let mut builder = TableBuilder::default();
//                 builder.push_record(vec![remove_address.alias_or_address.as_str()]);

//                 let mut table = builder.build();
//                 table.with(TableStyle::rounded());
//                 table.with(TablePanel::header("removed the keypair from keystore."));

//                 table.with(
//                     TableModify::new(TableCell::new(0, 0))
//                         .with(TableBorder::default().corner_bottom_right('┬')),
//                 );
//                 table.with(
//                     TableModify::new(TableCell::new(0, 0))
//                         .with(TableBorder::default().corner_top_right('─')),
//                 );

//                 write!(f, "{}", table)?
//             }
//             SomaClientCommandResult::Object(object_read) => match object_read.object() {
//                 Ok(obj) => {
//                     let object = ObjectOutput::from(obj);
//                     let json_obj = json!(&object);
//                     let mut table = json_to_table(&json_obj);
//                     table.with(TableStyle::rounded().horizontals([]));
//                     writeln!(f, "{}", table)?
//                 }
//                 Err(e) => writeln!(f, "Internal error, cannot read the object: {e}")?,
//             },
//             SomaClientCommandResult::Objects(object_refs) => {
//                 if object_refs.is_empty() {
//                     writeln!(f, "This address has no owned objects.")?
//                 } else {
//                     let objects = ObjectsOutput::from_vec(object_refs.to_vec());
//                     match objects {
//                         Ok(objs) => {
//                             let json_obj = json!(objs);
//                             let mut table = json_to_table(&json_obj);
//                             table.with(TableStyle::rounded().horizontals([]));
//                             writeln!(f, "{}", table)?
//                         }
//                         Err(e) => write!(f, "Internal error: {e}")?,
//                     }
//                 }
//             }
//             SomaClientCommandResult::TransactionBlock(response) => {
//                 write!(writer, "{}", response)?;
//             }
//             SomaClientCommandResult::RawObject(raw_object_read) => {
//                 let raw_object = match raw_object_read.object() {
//                     Ok(v) => match &v.bcs {
//                         Some(SomaRawData::Object(o)) => {
//                             format!("{:?}\nNumber of bytes: {}", o.bcs_bytes, o.bcs_bytes.len())
//                         }
//                         None => "Bcs field is None".to_string().red().to_string(),
//                     },
//                     Err(err) => format!("{err}").red().to_string(),
//                 };
//                 writeln!(writer, "{}", raw_object)?;
//             }
//             SomaClientCommandResult::ComputeTransactionDigest(tx_data) => {
//                 writeln!(writer, "{}", tx_data.digest())?;
//             }
//             SomaClientCommandResult::SerializedUnsignedTransaction(tx_data) => {
//                 writeln!(
//                     writer,
//                     "{}",
//                     fastcrypto::encoding::Base64::encode(bcs::to_bytes(tx_data).unwrap())
//                 )?;
//             }
//             SomaClientCommandResult::SerializedSignedTransaction(sender_signed_tx) => {
//                 writeln!(
//                     writer,
//                     "{}",
//                     fastcrypto::encoding::Base64::encode(bcs::to_bytes(sender_signed_tx).unwrap())
//                 )?;
//             }
//             SomaClientCommandResult::SyncClientState => {
//                 writeln!(writer, "Client state sync complete.")?;
//             }
//             SomaClientCommandResult::ChainIdentifier(ci) => {
//                 writeln!(writer, "{}", ci)?;
//             }
//             SomaClientCommandResult::Switch(response) => {
//                 write!(writer, "{}", response)?;
//             }
//             SomaClientCommandResult::ActiveAddress(response) => {
//                 match response {
//                     Some(r) => write!(writer, "{}", r)?,
//                     None => write!(writer, "None")?,
//                 };
//             }
//             SomaClientCommandResult::ActiveEnv(env) => {
//                 write!(writer, "{}", env.as_deref().unwrap_or("None"))?;
//             }
//             SomaClientCommandResult::NewEnv(env) => {
//                 writeln!(writer, "Added new Soma env [{}] to config.", env.alias)?;
//             }
//             SomaClientCommandResult::Envs(envs, active) => {
//                 let mut builder = TableBuilder::default();
//                 builder.set_header(["alias", "url", "active"]);
//                 for env in envs {
//                     builder.push_record(vec![env.alias.clone(), env.rpc.clone(), {
//                         if Some(env.alias.as_str()) == active.as_deref() {
//                             "*".to_string()
//                         } else {
//                             "".to_string()
//                         }
//                     }]);
//                 }
//                 let mut table = builder.build();
//                 table.with(TableStyle::rounded());
//                 write!(f, "{}", table)?
//             }
//             SomaClientCommandResult::NoOutput => {}
//             SomaClientCommandResult::DryRun(response) => {
//                 writeln!(f, "{}", Pretty(response))?;
//             }
//         }
//         write!(f, "{}", writer.trim_end_matches('\n'))
//     }
// }

// fn convert_number_to_string(value: Value) -> Value {
//     match value {
//         Value::Number(n) => Value::String(n.to_string()),
//         Value::Array(a) => Value::Array(a.into_iter().map(convert_number_to_string).collect()),
//         Value::Object(o) => Value::Object(
//             o.into_iter()
//                 .map(|(k, v)| (k, convert_number_to_string(v)))
//                 .collect(),
//         ),
//         _ => value,
//     }
// }

// impl Debug for SomaClientCommandResult {
//     fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
//         let s = unwrap_err_to_string(|| match self {
//             SomaClientCommandResult::Gas(gas_coins) => {
//                 let gas_coins = gas_coins
//                     .iter()
//                     .map(GasCoinOutput::from)
//                     .collect::<Vec<_>>();
//                 Ok(serde_json::to_string_pretty(&gas_coins)?)
//             }
//             SomaClientCommandResult::Object(object_read) => {
//                 let object = object_read.object()?;
//                 Ok(serde_json::to_string_pretty(&object)?)
//             }
//             SomaClientCommandResult::RawObject(raw_object_read) => {
//                 let raw_object = raw_object_read.object()?;
//                 Ok(serde_json::to_string_pretty(&raw_object)?)
//             }
//             _ => Ok(serde_json::to_string_pretty(self)?),
//         });
//         write!(f, "{}", s)
//     }
// }

// fn unwrap_err_to_string<T: Display, F: FnOnce() -> Result<T, anyhow::Error>>(func: F) -> String {
//     match func() {
//         Ok(s) => format!("{s}"),
//         Err(err) => format!("{err}").red().to_string(),
//     }
// }

// impl SomaClientCommandResult {
//     pub fn objects_response(&self) -> Option<Vec<SomaObjectResponse>> {
//         use SomaClientCommandResult::*;
//         match self {
//             Object(o) | RawObject(o) => Some(vec![o.clone()]),
//             Objects(o) => Some(o.clone()),
//             _ => None,
//         }
//     }

//     pub fn print(&self, pretty: bool) {
//         let line = if pretty {
//             format!("{self}")
//         } else {
//             format!("{:?}", self)
//         };
//         // Log line by line
//         for line in line.lines() {
//             // Logs write to a file on the side.  Print to stdout and also log to file, for tests to pass.
//             println!("{line}");
//             info!("{line}")
//         }
//     }

//     pub fn tx_block_response(&self) -> Option<&SomaTransactionBlockResponse> {
//         use SomaClientCommandResult::*;
//         match self {
//             TransactionBlock(b) => Some(b),
//             _ => None,
//         }
//     }

//     pub async fn prerender_clever_errors(mut self, context: &mut WalletContext) -> Self {
//         match &mut self {
//             SomaClientCommandResult::DryRun(DryRunTransactionBlockResponse { effects, .. })
//             | SomaClientCommandResult::TransactionBlock(SomaTransactionBlockResponse {
//                 effects: Some(effects),
//                 ..
//             }) => {
//                 let client = context.get_client().await.expect("Cannot connect to RPC");
//                 prerender_clever_errors(effects, client.read_api()).await
//             }

//             SomaClientCommandResult::TransactionBlock(SomaTransactionBlockResponse {
//                 effects: None,
//                 ..
//             }) => (),
//             SomaClientCommandResult::ActiveAddress(_)
//             | SomaClientCommandResult::ActiveEnv(_)
//             | SomaClientCommandResult::Addresses(_)
//             | SomaClientCommandResult::Balance(_, _)
//             | SomaClientCommandResult::ComputeTransactionDigest(_)
//             | SomaClientCommandResult::ChainIdentifier(_)
//             | SomaClientCommandResult::Envs(_, _)
//             | SomaClientCommandResult::Gas(_)
//             | SomaClientCommandResult::NewAddress(_)
//             | SomaClientCommandResult::NewEnv(_)
//             | SomaClientCommandResult::NoOutput
//             | SomaClientCommandResult::Object(_)
//             | SomaClientCommandResult::Objects(_)
//             | SomaClientCommandResult::RemoveAddress(_)
//             | SomaClientCommandResult::RawObject(_)
//             | SomaClientCommandResult::SerializedSignedTransaction(_)
//             | SomaClientCommandResult::SerializedUnsignedTransaction(_)
//             | SomaClientCommandResult::Switch(_)
//             | SomaClientCommandResult::SyncClientState
//         }
//         self
//     }
// }

// #[derive(Serialize)]
// #[serde(rename_all = "camelCase")]
// pub struct AddressesOutput {
//     pub active_address: SomaAddress,
//     pub addresses: Vec<(String, SomaAddress)>,
// }

// #[derive(Serialize)]
// #[serde(rename_all = "camelCase")]
// pub struct NewAddressOutput {
//     pub alias: String,
//     pub address: SomaAddress,
//     pub key_scheme: SignatureScheme,
//     pub recovery_phrase: String,
// }

// #[derive(Serialize)]
// #[serde(rename_all = "camelCase")]
// pub struct RemoveAddressOutput {
//     pub alias_or_address: String,
// }

// #[derive(Serialize)]
// #[serde(rename_all = "camelCase")]
// pub struct ObjectOutput {
//     pub object_id: ObjectID,
//     pub version: Version,
//     pub digest: String,
//     pub obj_type: String,
//     #[serde(skip_serializing_if = "Option::is_none")]
//     pub owner: Option<Owner>,
//     #[serde(skip_serializing_if = "Option::is_none")]
//     pub prev_tx: Option<TransactionDigest>,
//     #[serde(skip_serializing_if = "Option::is_none")]
//     pub content: Option<SomaParsedData>,
// }

// impl From<&SomaObjectData> for ObjectOutput {
//     fn from(obj: &SomaObjectData) -> Self {
//         let obj_type = match obj.type_.as_ref() {
//             Some(x) => x.to_string(),
//             None => "unknown".to_string(),
//         };
//         Self {
//             object_id: obj.object_id,
//             version: obj.version,
//             digest: obj.digest.to_string(),
//             obj_type,
//             owner: obj.owner.clone(),
//             prev_tx: obj.previous_transaction,
//             content: obj.content.clone(),
//         }
//     }
// }

// #[derive(Serialize)]
// #[serde(rename_all = "camelCase")]
// pub struct GasCoinOutput {
//     pub gas_coin_id: ObjectID,
//     pub shannons_balance: u64,
//     pub soma_balance: String,
// }

// impl From<&Coin> for GasCoinOutput {
//     fn from(gas_coin: &Coin) -> Self {
//         Self {
//             gas_coin_id: *gas_coin.id(),
//             shannons_balance: gas_coin.value(),
//             soma_balance: format_balance(gas_coin.value() as u128, 9, 2, None),
//         }
//     }
// }

// #[derive(Serialize)]
// #[serde(rename_all = "camelCase")]
// pub struct ObjectsOutput {
//     pub object_id: ObjectID,
//     pub version: Version,
//     pub digest: String,
//     pub object_type: String,
// }

// impl ObjectsOutput {
//     fn from(obj: SomaObjectResponse) -> Result<Self, anyhow::Error> {
//         let obj = obj.into_object()?;
//         // this replicates the object type display as in the soma explorer
//         let object_type = match obj.type_ {
//             Some(types::object::ObjectType::Struct(x)) => {
//                 let address = x.address().to_string();
//                 // check if the address has length of 64 characters
//                 // otherwise, keep it as it is
//                 let address = if address.len() == 64 {
//                     format!("0x{}..{}", &address[..4], &address[address.len() - 4..])
//                 } else {
//                     address
//                 };
//                 format!("{}::{}::{}", address, x.module(), x.name(),)
//             }
//             None => "unknown".to_string(),
//         };
//         Ok(Self {
//             object_id: obj.object_id,
//             version: obj.version,
//             digest: Base64::encode(obj.digest),
//             object_type,
//         })
//     }
//     fn from_vec(objs: Vec<SomaObjectResponse>) -> Result<Vec<Self>, anyhow::Error> {
//         objs.into_iter()
//             .map(ObjectsOutput::from)
//             .collect::<Result<Vec<_>, _>>()
//     }
// }

// #[derive(Serialize)]
// #[serde(untagged)]
// pub enum SomaClientCommandResult {
//     ActiveAddress(Option<SomaAddress>),
//     ActiveEnv(Option<String>),
//     Addresses(AddressesOutput),
//     Balance(Vec<Coin>, bool),
//     ChainIdentifier(String),
//     ComputeTransactionDigest(TransactionData),
//     DryRun(DryRunTransactionBlockResponse),
//     Envs(Vec<SomaEnv>, Option<String>),
//     Gas(Vec<Coin>),
//     NewAddress(NewAddressOutput),
//     NewEnv(SomaEnv),
//     NoOutput,
//     Object(SomaObjectResponse),
//     Objects(Vec<SomaObjectResponse>),
//     RawObject(SomaObjectResponse),
//     RemoveAddress(RemoveAddressOutput),
//     SerializedSignedTransaction(SenderSignedData),
//     SerializedUnsignedTransaction(TransactionData),
//     Switch(SwitchResponse),
//     SyncClientState,
//     TransactionBlock(SomaTransactionBlockResponse),
// }

// #[derive(Serialize, Clone)]
// pub struct SwitchResponse {
//     /// Active address
//     pub address: Option<String>,
//     pub env: Option<String>,
// }

// impl Display for SwitchResponse {
//     fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
//         let mut writer = String::new();

//         if let Some(addr) = &self.address {
//             writeln!(writer, "Active address switched to {addr}")?;
//         }
//         if let Some(env) = &self.env {
//             writeln!(writer, "Active environment switched to [{env}]")?;
//         }
//         write!(f, "{}", writer)
//     }
// }

// fn pretty_print_balance(coins: &Vec<Coin>, builder: &mut TableBuilder, with_coins: bool) {
//     let format_decmials = 2;
//     let mut table_builder = TableBuilder::default();
//     if !with_coins {
//         table_builder.set_header(vec!["coin", "balance (raw)", "balance", ""]);
//     }

//     let balance = coins.iter().map(|x| x.balance as u128).sum::<u128>();
//     let mut inner_table = TableBuilder::default();
//     inner_table.set_header(vec!["coinId", "balance (raw)", "balance", ""]);

//     if with_coins {
//         let coin_numbers = if coins.len() != 1 { "coins" } else { "coin" };
//         let balance_formatted = format!(
//             "({} {})",
//             format_balance(balance, coin_decimals, format_decmials, Some(symbol)),
//             symbol
//         );
//         let summary = format!(
//             "{}: {} {coin_numbers}, Balance: {} {}",
//             name,
//             coins.len(),
//             balance,
//             balance_formatted
//         );
//         for c in coins {
//             inner_table.push_record(vec![
//                 c.coin_object_id.to_string().as_str(),
//                 c.balance.to_string().as_str(),
//                 format_balance(
//                     c.balance as u128,
//                     coin_decimals,
//                     format_decmials,
//                     Some(symbol),
//                 )
//                 .as_str(),
//             ]);
//         }
//         let mut table = inner_table.build();
//         table.with(TablePanel::header(summary));
//         table.with(
//             TableStyle::rounded()
//                 .horizontals([
//                     HorizontalLine::new(1, TableStyle::modern().get_horizontal()),
//                     HorizontalLine::new(2, TableStyle::modern().get_horizontal()),
//                 ])
//                 .remove_vertical(),
//         );
//         table.with(tabled::settings::style::BorderSpanCorrection);
//         builder.push_record(vec![table.to_string()]);
//     } else {
//         table_builder.push_record(vec![
//             name,
//             balance.to_string().as_str(),
//             format_balance(balance, coin_decimals, format_decmials, Some(symbol)).as_str(),
//         ]);
//     }

//     let mut table = table_builder.build();
//     table.with(
//         TableStyle::rounded()
//             .horizontals([HorizontalLine::new(
//                 1,
//                 TableStyle::modern().get_horizontal(),
//             )])
//             .remove_vertical(),
//     );
//     table.with(tabled::settings::style::BorderSpanCorrection);
//     builder.push_record(vec![table.to_string()]);
// }

// fn divide(value: u128, divisor: u128) -> (u128, u128) {
//     let integer_part = value / divisor;
//     let fractional_part = value % divisor;
//     (integer_part, fractional_part)
// }

// fn format_balance(
//     value: u128,
//     coin_decimals: u8,
//     format_decimals: usize,
//     symbol: Option<&str>,
// ) -> String {
//     let mut suffix = if let Some(symbol) = symbol {
//         format!(" {symbol}")
//     } else {
//         "".to_string()
//     };

//     let mut coin_decimals = coin_decimals as u32;
//     let billions = 10u128.pow(coin_decimals + 9);
//     let millions = 10u128.pow(coin_decimals + 6);
//     let thousands = 10u128.pow(coin_decimals + 3);
//     let units = 10u128.pow(coin_decimals);

//     let (whole, fractional) = if value > billions {
//         coin_decimals += 9;
//         suffix = format!("B{suffix}");
//         divide(value, billions)
//     } else if value > millions {
//         coin_decimals += 6;
//         suffix = format!("M{suffix}");
//         divide(value, millions)
//     } else if value > thousands {
//         coin_decimals += 3;
//         suffix = format!("K{suffix}");
//         divide(value, thousands)
//     } else {
//         divide(value, units)
//     };

//     let mut fractional = format!("{fractional:0width$}", width = coin_decimals as usize);
//     fractional.truncate(format_decimals);

//     format!("{whole}.{fractional}{suffix}")
// }
