use crate::displays::Pretty;
use std::{
    collections::{btree_map::Entry, BTreeMap, BTreeSet},
    fmt::{Debug, Display, Formatter, Write},
    fs,
    path::{Path, PathBuf},
    str::FromStr,
    sync::Arc,
};

use anyhow::{anyhow, bail, ensure, Context};
use bip32::DerivationPath;
use clap::*;
use colored::Colorize;
use fastcrypto::{
    encoding::{Base64, Encoding},
    traits::ToFromBytes,
};
use reqwest::StatusCode;

use protocol_config::{Chain, ProtocolConfig, ProtocolVersion};
use serde::Serialize;
use serde_json::{json, Value};

use types::intent::Intent;

use soma_keys::key_identity::KeyIdentity;
use soma_keys::keystore::AccountKeystore;

use sdk::{
    client_config::{SomaClientConfig, SomaEnv},
    wallet_context::WalletContext,
    SomaClient, SOMA_DEVNET_URL, SOMA_LOCAL_NETWORK_URL, SOMA_LOCAL_NETWORK_URL_0,
    SOMA_TESTNET_URL,
};
use types::{
    base::{FullObjectID, SomaAddress},
    crypto::GenericSignature,
    crypto::{EmptySignInfo, SignatureScheme},
    digests::TransactionDigest,
    envelope::Envelope,
    error::SomaError,
    object::Owner,
    object::{ObjectID, ObjectRef, ObjectType, Version},
    transaction::{
        InputObjectKind, SenderSignedData, Transaction, TransactionData, TransactionKind,
    },
    tx_fee::TransactionFee,
};

use json_to_table::json_to_table;
use tabled::{
    builder::Builder as TableBuilder,
    settings::{
        object::{Cell as TableCell, Columns as TableCols, Rows as TableRows},
        span::Span as TableSpan,
        style::HorizontalLine,
        Alignment as TableAlignment, Border as TableBorder, Modify as TableModify,
        Panel as TablePanel, Style as TableStyle,
    },
};

use soma_keys::key_derive;
use tracing::{debug, info};
use types::digests::ChainIdentifier;

#[derive(Parser)]
#[clap(rename_all = "kebab-case")]
pub enum SomaClientCommands {
    /// Default address used for commands when none specified
    #[clap(name = "active-address")]
    ActiveAddress,
    /// Default environment used for commands when none specified
    #[clap(name = "active-env")]
    ActiveEnv,
    /// Obtain the Addresses managed by the client.
    #[clap(name = "addresses")]
    Addresses {
        /// Sort by alias instead of address
        #[clap(long, short = 's')]
        sort_by_alias: bool,
    },
    /// List the coin balance of an address
    #[clap(name = "balance")]
    Balance {
        /// Address (or its alias)
        #[arg(value_parser)]
        address: Option<KeyIdentity>,
        /// Show a list with each coin's object ID and balance
        #[clap(long, required = false)]
        with_coins: bool,
    },

    /// Query the chain identifier from the rpc endpoint.
    #[clap(name = "chain-identifier")]
    ChainIdentifier,

    /// List all Soma environments
    Envs,

    /// Execute a Signed Transaction. This is useful when the user prefers to sign elsewhere and use this command to execute.
    ExecuteSignedTx {
        /// BCS serialized transaction data bytes without its type tag, as base64 encoded string. This is the output of soma client command using --serialize-unsigned-transaction.
        #[clap(long)]
        tx_bytes: String,

        /// A list of Base64 encoded signatures `flag || signature || pubkey`.
        #[clap(long)]
        signatures: Vec<String>,
    },
    /// Execute a combined serialized SenderSignedData string.
    ExecuteCombinedSignedTx {
        /// BCS serialized sender signed data, as base64 encoded string. This is the output of soma client command using --serialize-signed-transaction.
        #[clap(long)]
        signed_tx_bytes: String,
    },

    /// TODO: Request gas coin from faucet. By default, it will use the active address and the active network.
    // #[clap[name = "faucet"]]
    // Faucet {
    //     /// Address (or its alias)
    //     #[clap(long)]
    //     #[arg(value_parser)]
    //     address: Option<KeyIdentity>,
    //     /// The url to the faucet
    //     #[clap(long)]
    //     url: Option<String>,
    // },

    /// Obtain all gas objects owned by the address.
    /// An address' alias can be used instead of the address.
    #[clap(name = "gas")]
    Gas {
        /// Address (or its alias) owning the objects
        #[clap(name = "owner_address")]
        #[arg(value_parser)]
        address: Option<KeyIdentity>,
    },

    /// Generate new address and keypair with keypair scheme flag {ed25519 | secp256k1 | secp256r1}
    /// with optional derivation path, default to m/44'/784'/0'/0'/0' for ed25519 or
    /// m/54'/784'/0'/0/0 for secp256k1 or m/74'/784'/0'/0/0 for secp256r1. Word length can be
    /// { word12 | word15 | word18 | word21 | word24} default to word12 if not specified.
    #[clap(name = "new-address")]
    NewAddress {
        key_scheme: SignatureScheme,
        /// The alias must start with a letter and can contain only letters, digits, hyphens (-), or underscores (_).
        alias: Option<String>,
        word_length: Option<String>,
        derivation_path: Option<DerivationPath>,
    },

    /// Add new Soma environment.
    #[clap(name = "new-env")]
    NewEnv {
        #[clap(long)]
        alias: String,
        #[clap(long, value_hint = ValueHint::Url)]
        rpc: String,
        #[clap(long, help = "Basic auth in the format of username:password")]
        basic_auth: Option<String>,
    },

    /// Get object info
    #[clap(name = "object")]
    Object {
        /// Object ID of the object to fetch
        #[clap(name = "object_id")]
        id: ObjectID,

        /// Return the bcs serialized version of the object
        #[clap(long)]
        bcs: bool,
    },
    /// Obtain all objects owned by the address. It also accepts an address by its alias.
    #[clap(name = "objects")]
    Objects {
        /// Address owning the object. If no address is provided, it will show all
        /// objects owned by `soma client active-address`.
        #[clap(name = "owner_address")]
        address: Option<KeyIdentity>,
    },

    /// Pay SOMA coins to recipients following specified amounts, with input coins.
    /// Length of recipients must be the same as that of amounts.
    /// The input coins also include the coin for gas payment, so no extra gas coin is required.
    #[clap(name = "pay")]
    PaySoma {
        /// The input coins to be used for pay recipients, including the gas coin.
        #[clap(long, num_args(1..))]
        input_coins: Vec<ObjectID>,

        /// The recipient addresses, must be of same length as amounts.
        /// Aliases of addresses are also accepted as input.
        #[clap(long, num_args(1..))]
        recipients: Vec<KeyIdentity>,

        /// The amounts to be paid, following the order of recipients.
        #[clap(long, num_args(1..))]
        amounts: Vec<u64>,

        #[clap(flatten)]
        processing: TxProcessingArgs,
    },

    /// Execute, dry-run, or otherwise inspect an already serialized transaction.
    SerializedTx {
        /// Base64-encoded BCS-serialized TransactionData.
        tx_bytes: String,

        #[clap(flatten)]
        processing: TxProcessingArgs,
    },

    /// Execute, dry-run, or otherwise inspect an already serialized transaction kind.
    SerializedTxKind {
        /// Base64-encoded BCS-serialized TransactionKind.
        tx_bytes: String,

        #[clap(flatten)]
        payment: PaymentArgs,

        #[clap(flatten)]
        processing: TxProcessingArgs,
    },

    /// Switch active address and network(e.g., devnet, local rpc server).
    #[clap(name = "switch")]
    Switch {
        /// An address to be used as the active address for subsequent
        /// commands. It accepts also the alias of the address.
        #[clap(long)]
        address: Option<KeyIdentity>,
        /// The RPC server URL (e.g., local rpc server, devnet rpc server, etc) to be
        /// used for subsequent commands.
        #[clap(long)]
        env: Option<String>,
    },

    /// Get the effects of executing the given transaction block
    #[clap(name = "tx-block")]
    TransactionBlock {
        /// Digest of the transaction block
        #[clap(name = "digest")]
        digest: TransactionDigest,
    },

    /// Transfer object
    #[clap(name = "transfer")]
    Transfer {
        /// Recipient address (or its alias if it's an address in the keystore)
        #[clap(long)]
        to: KeyIdentity,

        /// ID of the object to transfer
        #[clap(long)]
        object_id: ObjectID,

        #[clap(flatten)]
        payment: PaymentArgs,

        #[clap(flatten)]
        processing: TxProcessingArgs,
    },

    /// Transfer SOMA, and pay gas with the same SOMA coin object.
    /// If amount is specified, only the amount is transferred; otherwise the entire object
    /// is transferred.
    #[clap(name = "transfer-soma")]
    TransferSoma {
        /// Recipient address (or its alias if it's an address in the keystore)
        #[clap(long)]
        to: KeyIdentity,

        /// ID of the coin to transfer. This is also the gas object.
        #[clap(long)]
        soma_coin_object_id: ObjectID,

        /// The amount to transfer, if not specified, the entire coin object will be transferred.
        #[clap(long)]
        amount: Option<u64>,

        #[clap(flatten)]
        processing: TxProcessingArgs,
    },

    /// Remove an existing address by its alias or hexadecimal string.
    #[clap(name = "remove-address")]
    RemoveAddress { alias_or_address: String },
}

/// Arguments related to providing coins for gas payment
#[derive(Args, Debug, Default)]
pub struct PaymentArgs {
    /// IDs of gas objects to be used for gas payment. If none are provided, coins are selected
    /// automatically to cover the gas budget.
    #[clap(long, num_args(1..))]
    pub gas: Vec<ObjectID>,
}

/// Arguments related to what to do to a transaction after it has been built.
#[derive(Args, Debug, Default)]
pub struct TxProcessingArgs {
    /// Compute the transaction digest and print it out, but do not execute the transaction.
    #[arg(long)]
    pub tx_digest: bool,
    /// Perform a dry run of the transaction, without executing it.
    #[arg(long)]
    pub dry_run: bool,
    /// Instead of executing the transaction, serialize the bcs bytes of the unsigned transaction data
    /// (TransactionData) using base64 encoding, and print out the string <TX_BYTES>. The string can
    /// be used to execute transaction with `soma client execute-signed-tx --tx-bytes <TX_BYTES>`.
    #[arg(long)]
    pub serialize_unsigned_transaction: bool,
    /// Instead of executing the transaction, serialize the bcs bytes of the signed transaction data
    /// (SenderSignedData) using base64 encoding, and print out the string <SIGNED_TX_BYTES>. The
    /// string can be used to execute transaction with
    /// `soma client execute-combined-signed-tx --signed-tx-bytes <SIGNED_TX_BYTES>`.
    #[arg(long)]
    pub serialize_signed_transaction: bool,
    /// Set the transaction sender to this address. When not specified, the sender is inferred
    /// by finding the owner of the gas payment. Note that when setting this field, the
    /// transaction will fail to execute if the sender's private key is not in the keystore;
    /// similarly, it will fail when using this with `--serialize-signed-transaction` flag if the
    /// private key corresponding to this address is not in keystore.
    #[arg(long, required = false, value_parser)]
    pub sender: Option<SomaAddress>,
}

impl SomaClientCommands {
    pub async fn execute(
        self,
        context: &mut WalletContext,
    ) -> Result<SomaClientCommandResult, anyhow::Error> {
        let ret = match self {
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
                    addresses.sort();
                }

                let output = AddressesOutput {
                    active_address,
                    addresses,
                };
                SomaClientCommandResult::Addresses(output)
            }
            SomaClientCommands::Balance {
                address,
                with_coins,
            } => {
                let address = context.get_identity_address(address)?;
                let client = context.get_client().await?;
                let _ = context.cache_chain_id(&client).await?;

                let mut objects: Vec<Coin> = Vec::new();
                let mut cursor = None;
                loop {
                    let response = match coin_type {
                        Some(ref coin_type) => {
                            client
                                .coin_read_api()
                                .get_coins(address, Some(coin_type.clone()), cursor, None)
                                .await?
                        }
                        None => {
                            client
                                .coin_read_api()
                                .get_all_coins(address, cursor, None)
                                .await?
                        }
                    };

                    objects.extend(response.data);

                    if response.has_next_page {
                        cursor = response.next_cursor;
                    } else {
                        break;
                    }
                }

                fn canonicalize_type(type_: &str) -> Result<String, anyhow::Error> {
                    Ok(TypeTag::from_str(type_)
                        .context("Cannot parse coin type")?
                        .to_canonical_string(/* with_prefix */ true))
                }

                let mut coins_by_type = BTreeMap::new();
                for c in objects {
                    let coins = match coins_by_type.entry(canonicalize_type(&c.coin_type)?) {
                        Entry::Vacant(entry) => {
                            let metadata = client
                                .coin_read_api()
                                .get_coin_metadata(c.coin_type.clone())
                                .await
                                .with_context(|| {
                                    format!(
                                        "Cannot fetch the coin metadata for coin {}",
                                        c.coin_type
                                    )
                                })?;

                            &mut entry.insert((metadata, vec![])).1
                        }
                        Entry::Occupied(entry) => &mut entry.into_mut().1,
                    };

                    coins.push(c);
                }
                let soma_type_tag = canonicalize_type(SOMA_COIN_TYPE)?;

                // show SOMA first
                let ordered_coins_soma_first = coins_by_type
                    .remove(&soma_type_tag)
                    .into_iter()
                    .chain(coins_by_type.into_values())
                    .collect();

                SomaClientCommandResult::Balance(ordered_coins_soma_first, with_coins)
            }

            SomaClientCommands::Object { id, bcs } => {
                // Fetch the object ref
                let client = context.get_client().await?;
                let _ = context.cache_chain_id(&client).await?;
                if !bcs {
                    let object_read = client
                        .read_api()
                        .get_object_with_options(id, SomaObjectDataOptions::full_content())
                        .await?;
                    SomaClientCommandResult::Object(object_read)
                } else {
                    let raw_object_read = client
                        .read_api()
                        .get_object_with_options(id, SomaObjectDataOptions::bcs_lossless())
                        .await?;
                    SomaClientCommandResult::RawObject(raw_object_read)
                }
            }

            SomaClientCommands::TransactionBlock { digest } => {
                let client = context.get_client().await?;
                let _ = context.cache_chain_id(&client).await?;
                let tx_read = client
                    .read_api()
                    .get_transaction_with_options(
                        digest,
                        SomaTransactionBlockResponseOptions {
                            show_input: true,
                            show_raw_input: false,
                            show_effects: true,
                            show_events: true,
                            show_object_changes: true,
                            show_balance_changes: false,
                            show_raw_effects: false,
                        },
                    )
                    .await?;
                SomaClientCommandResult::TransactionBlock(tx_read)
            }

            SomaClientCommands::Transfer {
                to,
                object_id,
                payment,
                processing,
            } => {
                let signer = context.get_object_owner(&object_id).await?;
                let to = context.get_identity_address(Some(to))?;
                let client = context.get_client().await?;
                let _ = context.cache_chain_id(&client).await?;

                let tx_kind = client
                    .transaction_builder()
                    .transfer_object_tx_kind(object_id, to)
                    .await?;

                let gas_payment = client
                    .transaction_builder()
                    .input_refs(&payment.gas)
                    .await?;

                dry_run_or_execute_or_serialize(signer, tx_kind, context, gas_payment, processing)
                    .await?
            }

            SomaClientCommands::TransferSoma {
                to,
                soma_coin_object_id: object_id,
                amount,
                processing,
            } => {
                let signer = context.get_object_owner(&object_id).await?;
                let to = context.get_identity_address(Some(to))?;
                let client = context.get_client().await?;
                let _ = context.cache_chain_id(&client).await?;

                let tx_kind = client
                    .transaction_builder()
                    .transfer_soma_tx_kind(to, amount);

                let gas_payment = client
                    .transaction_builder()
                    .input_refs(&[object_id])
                    .await?;

                dry_run_or_execute_or_serialize(
                    signer,
                    tx_kind,
                    context,
                    gas_payment,
                    gas_data,
                    processing,
                )
                .await?
            }

            SomaClientCommands::PaySoma {
                input_coins,
                recipients,
                amounts,
                processing,
            } => {
                ensure!(
                    !input_coins.is_empty(),
                    "PaySoma transaction requires a non-empty list of input coins"
                );
                ensure!(
                    !recipients.is_empty(),
                    "PaySoma transaction requires a non-empty list of recipient addresses"
                );
                ensure!(
                    recipients.len() == amounts.len(),
                    format!(
                        "Found {:?} recipient addresses, but {:?} recipient amounts",
                        recipients.len(),
                        amounts.len()
                    ),
                );
                let recipients = recipients
                    .into_iter()
                    .map(|x| context.get_identity_address(Some(x)))
                    .collect::<Result<Vec<SomaAddress>, anyhow::Error>>()
                    .map_err(|e| anyhow!("{e}"))?;
                let signer = context.get_object_owner(&input_coins[0]).await?;
                let client = context.get_client().await?;
                let _ = context.cache_chain_id(&client).await?;

                let tx_kind = client
                    .transaction_builder()
                    .pay_soma_tx_kind(recipients, amounts)?;

                let gas_payment = client
                    .transaction_builder()
                    .input_refs(&input_coins)
                    .await?;

                dry_run_or_execute_or_serialize(
                    signer,
                    tx_kind,
                    context,
                    gas_payment,
                    gas_data,
                    processing,
                )
                .await?
            }

            SomaClientCommands::Objects { address } => {
                let address = context.get_identity_address(address)?;
                let client = context.get_client().await?;
                let _ = context.cache_chain_id(&client).await?;
                let mut objects: Vec<SomaObjectResponse> = Vec::new();
                let mut cursor = None;
                loop {
                    let response = client
                        .read_api()
                        .get_owned_objects(
                            address,
                            Some(SomaObjectResponseQuery::new_with_options(
                                SomaObjectDataOptions::full_content(),
                            )),
                            cursor,
                            None,
                        )
                        .await?;
                    objects.extend(response.data);

                    if response.has_next_page {
                        cursor = response.next_cursor;
                    } else {
                        break;
                    }
                }
                SomaClientCommandResult::Objects(objects)
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
                    Some(x) => x,
                    None => context.config.keystore.get_alias(&address)?,
                };

                SomaClientCommandResult::NewAddress(NewAddressOutput {
                    alias,
                    address,
                    key_scheme: scheme,
                    recovery_phrase: phrase,
                })
            }

            SomaClientCommands::RemoveAddress { alias_or_address } => {
                let identity = KeyIdentity::from_str(&alias_or_address)
                    .map_err(|e| anyhow!("Invalid address or alias: {}", e))?;
                let address: SomaAddress = context.config.keystore.get_by_identity(&identity)?;

                context.config.keystore.remove(address).await?;

                SomaClientCommandResult::RemoveAddress(RemoveAddressOutput { alias_or_address })
            }

            SomaClientCommands::Gas { address } => {
                let address = context.get_identity_address(address)?;
                let coins = context
                    .gas_objects(address)
                    .await?
                    .iter()
                    // Ok to unwrap() since `get_gas_objects` guarantees gas
                    .map(|(_val, object)| GasCoin::try_from(object).unwrap())
                    .collect();
                let _ = context.cache_chain_id(&context.get_client().await?).await?;
                SomaClientCommandResult::Gas(coins)
            }
            // SomaClientCommands::Faucet { address, url } => {
            //     let address = context.get_identity_address(address)?;
            //     let url = if let Some(url) = url {
            //         ensure!(
            //             !url.starts_with("https://faucet.testnet.soma.io"),
            //             "For testnet tokens, please use the Web UI: https://faucet.soma.io/?address={address}"
            //         );
            //         url
            //     } else {
            //         let active_env = context.get_active_env();
            //         if let Ok(env) = active_env {
            //             find_faucet_url(address, &env.rpc)?
            //         } else {
            //             bail!("No URL for faucet was provided and there is no active network.")
            //         }
            //     };
            //     request_tokens_from_faucet(address, url).await?;
            //     let _ = context.cache_chain_id(&context.get_client().await?).await?;
            //     SomaClientCommandResult::NoOutput
            // }
            SomaClientCommands::ChainIdentifier => {
                let client = context.get_client().await?;
                let ci = context.cache_chain_id(&client).await?;
                SomaClientCommandResult::ChainIdentifier(ci)
            }
            SomaClientCommands::SerializedTx {
                tx_bytes,
                processing,
            } => {
                let Ok(bytes) = Base64::decode(&tx_bytes) else {
                    bail!("Invalid Base64 encoding");
                };

                let Ok(tx_data): Result<TransactionData, _> = bcs::from_bytes(&bytes) else {
                    bail!("Failed to parse --tx-bytes as TransactionData");
                };

                let sender = tx_data.sender();
                let gas_payment = tx_data.gas().to_owned();
                let tx_kind = tx_data.kind().clone();

                dry_run_or_execute_or_serialize(
                    sender,
                    tx_kind,
                    context,
                    gas_payment,
                    gas_data,
                    processing,
                )
                .await?
            }
            SomaClientCommands::SerializedTxKind {
                tx_bytes,
                payment,
                processing,
            } => {
                let Ok(bytes) = Base64::decode(&tx_bytes) else {
                    bail!("Invalid Base64 encoding");
                };

                let Ok(tx_kind): Result<TransactionKind, _> = bcs::from_bytes(&bytes) else {
                    bail!("Failed to parse --tx-bytes as TransactionKind");
                };

                let client = context.get_client().await?;
                let sender = context.infer_sender(&payment.gas).await?;
                let gas_payment = client
                    .transaction_builder()
                    .input_refs(&payment.gas)
                    .await?;

                dry_run_or_execute_or_serialize(
                    sender,
                    tx_kind,
                    context,
                    gas_payment,
                    gas_data,
                    processing,
                )
                .await?
            }
            SomaClientCommands::Switch { address, env } => {
                let mut addr = None;

                if address.is_none() && env.is_none() {
                    return Err(anyhow!(
                        "No address, an alias, or env specified. Please specify one."
                    ));
                }

                if let Some(address) = address {
                    let address = context.get_identity_address(Some(address))?;
                    if !context.config.keystore.addresses().contains(&address) {
                        return Err(anyhow!("Address {} not managed by wallet", address));
                    }
                    context.config.active_address = Some(address);
                    addr = Some(address.to_string());
                }

                if let Some(ref env) = env {
                    Self::switch_env(&mut context.config, env)?;
                }
                context.config.save()?;
                SomaClientCommandResult::Switch(SwitchResponse { address: addr, env })
            }
            SomaClientCommands::ActiveAddress => {
                SomaClientCommandResult::ActiveAddress(context.active_address().ok())
            }

            SomaClientCommands::ExecuteSignedTx {
                tx_bytes,
                signatures,
            } => {
                let data = bcs::from_bytes(
                    &Base64::try_from(tx_bytes)
                    .map_err(|_| anyhow!("Invalid Base64 encoding"))?
                    .to_vec()
                    .map_err(|_| anyhow!("Invalid Base64 encoding"))?
                ).map_err(|_| anyhow!("Failed to parse tx bytes, check if it matches the output of soma client commands with --serialize-unsigned-transaction"))?;

                let mut sigs = Vec::new();
                for sig in signatures {
                    sigs.push(
                        GenericSignature::from_bytes(
                            &Base64::try_from(sig)
                                .map_err(|_| anyhow!("Invalid Base64 encoding"))?
                                .to_vec()
                                .map_err(|e| anyhow!(e))?,
                        )
                        .map_err(|_| anyhow!("Invalid generic signature"))?,
                    );
                }
                let transaction = Transaction::from_generic_sig_data(data, sigs);

                let response = context.execute_transaction_may_fail(transaction).await?;
                SomaClientCommandResult::TransactionBlock(response)
            }
            SomaClientCommands::ExecuteCombinedSignedTx { signed_tx_bytes } => {
                let data: SenderSignedData = bcs::from_bytes(
                    &Base64::try_from(signed_tx_bytes)
                        .map_err(|_| anyhow!("Invalid Base64 encoding"))?
                        .to_vec()
                        .map_err(|_| anyhow!("Invalid Base64 encoding"))?
                ).map_err(|_| anyhow!("Failed to parse SenderSignedData bytes, check if it matches the output of soma client commands with --serialize-signed-transaction"))?;
                let transaction = Envelope::<SenderSignedData, EmptySignInfo>::new(data);
                let response = context.execute_transaction_may_fail(transaction).await?;
                SomaClientCommandResult::TransactionBlock(response)
            }
            SomaClientCommands::NewEnv {
                alias,
                rpc,
                basic_auth,
            } => {
                if context.config.envs.iter().any(|env| env.alias == alias) {
                    return Err(anyhow!(
                        "Environment config with name [{alias}] already exists."
                    ));
                }
                let mut env = SomaEnv {
                    alias,
                    rpc,
                    basic_auth,
                    chain_id: None,
                };

                // Check urls are valid and server is reachable
                env.create_rpc_client(None).await?;
                context.config.envs.push(env.clone());
                context.config.save()?;
                let chain_id = context.cache_chain_id(&context.get_client().await?).await?;
                env.chain_id = Some(chain_id);
                SomaClientCommandResult::NewEnv(env)
            }
            SomaClientCommands::ActiveEnv => SomaClientCommandResult::ActiveEnv(
                context.get_active_env().ok().map(|e| e.alias.clone()),
            ),
            SomaClientCommands::Envs => SomaClientCommandResult::Envs(
                context.config.envs.clone(),
                context.get_active_env().ok().map(|e| e.alias.clone()),
            ),
        };
        Ok(ret.prerender_clever_errors(context).await)
    }

    pub fn switch_env(config: &mut SomaClientConfig, env: &str) -> Result<(), anyhow::Error> {
        let env = Some(env.into());
        ensure!(
            config.get_env(&env).is_some(),
            "Environment config not found for [{env:?}], add new environment config using the `soma client new-env` command."
        );
        config.active_env = env;
        Ok(())
    }
}

impl Display for SomaClientCommandResult {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut writer = String::new();
        match self {
            SomaClientCommandResult::Addresses(addresses) => {
                let mut builder = TableBuilder::default();
                builder.set_header(vec!["alias", "address", "active address"]);
                for (alias, address) in &addresses.addresses {
                    let active_address = if address == &addresses.active_address {
                        "*".to_string()
                    } else {
                        "".to_string()
                    };
                    builder.push_record([alias.to_string(), address.to_string(), active_address]);
                }
                let mut table = builder.build();
                let style = TableStyle::rounded();
                table.with(style);
                write!(f, "{}", table)?
            }
            SomaClientCommandResult::Balance(coins, with_coins) => {
                if coins.is_empty() {
                    return write!(f, "No coins found for this address.");
                }
                let mut builder = TableBuilder::default();
                pretty_print_balance(coins, &mut builder, *with_coins);
                let mut table = builder.build();
                table.with(TablePanel::header("Balance of coins owned by this address"));
                table.with(TableStyle::rounded().horizontals([HorizontalLine::new(
                    1,
                    TableStyle::modern().get_horizontal(),
                )]));
                table.with(tabled::settings::style::BorderSpanCorrection);
                write!(f, "{}", table)?;
            }

            SomaClientCommandResult::Gas(gas_coins) => {
                let gas_coins = gas_coins
                    .iter()
                    .map(GasCoinOutput::from)
                    .collect::<Vec<_>>();
                if gas_coins.is_empty() {
                    write!(f, "No gas coins are owned by this address")?;
                    return Ok(());
                }

                let mut builder = TableBuilder::default();
                builder.set_header(vec![
                    "gasCoinId",
                    "shannonsBalance (SHNS)",
                    "somaBalance (SOMA)",
                ]);
                for coin in &gas_coins {
                    builder.push_record(vec![
                        coin.gas_coin_id.to_string(),
                        coin.shannons_balance.to_string(),
                        coin.soma_balance.to_string(),
                    ]);
                }
                let mut table = builder.build();
                table.with(TableStyle::rounded());
                if gas_coins.len() > 10 {
                    table.with(TablePanel::header(format!(
                        "Showing {} gas coins and their balances.",
                        gas_coins.len()
                    )));
                    table.with(TablePanel::footer(format!(
                        "Showing {} gas coins and their balances.",
                        gas_coins.len()
                    )));
                    table.with(TableStyle::rounded().horizontals([
                        HorizontalLine::new(1, TableStyle::modern().get_horizontal()),
                        HorizontalLine::new(2, TableStyle::modern().get_horizontal()),
                        HorizontalLine::new(
                            gas_coins.len() + 2,
                            TableStyle::modern().get_horizontal(),
                        ),
                    ]));
                    table.with(tabled::settings::style::BorderSpanCorrection);
                }
                write!(f, "{}", table)?;
            }
            SomaClientCommandResult::NewAddress(new_address) => {
                let mut builder = TableBuilder::default();
                builder.push_record(vec!["alias", new_address.alias.as_str()]);
                builder.push_record(vec!["address", new_address.address.to_string().as_str()]);
                builder.push_record(vec![
                    "keyScheme",
                    new_address.key_scheme.to_string().as_str(),
                ]);
                builder.push_record(vec![
                    "recoveryPhrase",
                    new_address.recovery_phrase.to_string().as_str(),
                ]);

                let mut table = builder.build();
                table.with(TableStyle::rounded());
                table.with(TablePanel::header(
                    "Created new keypair and saved it to keystore.",
                ));

                table.with(
                    TableModify::new(TableCell::new(0, 0))
                        .with(TableBorder::default().corner_bottom_right('┬')),
                );
                table.with(
                    TableModify::new(TableCell::new(0, 0))
                        .with(TableBorder::default().corner_top_right('─')),
                );

                write!(f, "{}", table)?
            }
            SomaClientCommandResult::RemoveAddress(remove_address) => {
                let mut builder = TableBuilder::default();
                builder.push_record(vec![remove_address.alias_or_address.as_str()]);

                let mut table = builder.build();
                table.with(TableStyle::rounded());
                table.with(TablePanel::header("removed the keypair from keystore."));

                table.with(
                    TableModify::new(TableCell::new(0, 0))
                        .with(TableBorder::default().corner_bottom_right('┬')),
                );
                table.with(
                    TableModify::new(TableCell::new(0, 0))
                        .with(TableBorder::default().corner_top_right('─')),
                );

                write!(f, "{}", table)?
            }
            SomaClientCommandResult::Object(object_read) => match object_read.object() {
                Ok(obj) => {
                    let object = ObjectOutput::from(obj);
                    let json_obj = json!(&object);
                    let mut table = json_to_table(&json_obj);
                    table.with(TableStyle::rounded().horizontals([]));
                    writeln!(f, "{}", table)?
                }
                Err(e) => writeln!(f, "Internal error, cannot read the object: {e}")?,
            },
            SomaClientCommandResult::Objects(object_refs) => {
                if object_refs.is_empty() {
                    writeln!(f, "This address has no owned objects.")?
                } else {
                    let objects = ObjectsOutput::from_vec(object_refs.to_vec());
                    match objects {
                        Ok(objs) => {
                            let json_obj = json!(objs);
                            let mut table = json_to_table(&json_obj);
                            table.with(TableStyle::rounded().horizontals([]));
                            writeln!(f, "{}", table)?
                        }
                        Err(e) => write!(f, "Internal error: {e}")?,
                    }
                }
            }
            SomaClientCommandResult::TransactionBlock(response) => {
                write!(writer, "{}", response)?;
            }
            SomaClientCommandResult::RawObject(raw_object_read) => {
                let raw_object = match raw_object_read.object() {
                    Ok(v) => match &v.bcs {
                        Some(SomaRawData::Object(o)) => {
                            format!("{:?}\nNumber of bytes: {}", o.bcs_bytes, o.bcs_bytes.len())
                        }
                        None => "Bcs field is None".to_string().red().to_string(),
                    },
                    Err(err) => format!("{err}").red().to_string(),
                };
                writeln!(writer, "{}", raw_object)?;
            }
            SomaClientCommandResult::ComputeTransactionDigest(tx_data) => {
                writeln!(writer, "{}", tx_data.digest())?;
            }
            SomaClientCommandResult::SerializedUnsignedTransaction(tx_data) => {
                writeln!(
                    writer,
                    "{}",
                    fastcrypto::encoding::Base64::encode(bcs::to_bytes(tx_data).unwrap())
                )?;
            }
            SomaClientCommandResult::SerializedSignedTransaction(sender_signed_tx) => {
                writeln!(
                    writer,
                    "{}",
                    fastcrypto::encoding::Base64::encode(bcs::to_bytes(sender_signed_tx).unwrap())
                )?;
            }
            SomaClientCommandResult::SyncClientState => {
                writeln!(writer, "Client state sync complete.")?;
            }
            SomaClientCommandResult::ChainIdentifier(ci) => {
                writeln!(writer, "{}", ci)?;
            }
            SomaClientCommandResult::Switch(response) => {
                write!(writer, "{}", response)?;
            }
            SomaClientCommandResult::ActiveAddress(response) => {
                match response {
                    Some(r) => write!(writer, "{}", r)?,
                    None => write!(writer, "None")?,
                };
            }
            SomaClientCommandResult::ActiveEnv(env) => {
                write!(writer, "{}", env.as_deref().unwrap_or("None"))?;
            }
            SomaClientCommandResult::NewEnv(env) => {
                writeln!(writer, "Added new Soma env [{}] to config.", env.alias)?;
            }
            SomaClientCommandResult::Envs(envs, active) => {
                let mut builder = TableBuilder::default();
                builder.set_header(["alias", "url", "active"]);
                for env in envs {
                    builder.push_record(vec![env.alias.clone(), env.rpc.clone(), {
                        if Some(env.alias.as_str()) == active.as_deref() {
                            "*".to_string()
                        } else {
                            "".to_string()
                        }
                    }]);
                }
                let mut table = builder.build();
                table.with(TableStyle::rounded());
                write!(f, "{}", table)?
            }
            SomaClientCommandResult::NoOutput => {}
            SomaClientCommandResult::DryRun(response) => {
                writeln!(f, "{}", Pretty(response))?;
            }
        }
        write!(f, "{}", writer.trim_end_matches('\n'))
    }
}

fn convert_number_to_string(value: Value) -> Value {
    match value {
        Value::Number(n) => Value::String(n.to_string()),
        Value::Array(a) => Value::Array(a.into_iter().map(convert_number_to_string).collect()),
        Value::Object(o) => Value::Object(
            o.into_iter()
                .map(|(k, v)| (k, convert_number_to_string(v)))
                .collect(),
        ),
        _ => value,
    }
}

impl Debug for SomaClientCommandResult {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let s = unwrap_err_to_string(|| match self {
            SomaClientCommandResult::Gas(gas_coins) => {
                let gas_coins = gas_coins
                    .iter()
                    .map(GasCoinOutput::from)
                    .collect::<Vec<_>>();
                Ok(serde_json::to_string_pretty(&gas_coins)?)
            }
            SomaClientCommandResult::Object(object_read) => {
                let object = object_read.object()?;
                Ok(serde_json::to_string_pretty(&object)?)
            }
            SomaClientCommandResult::RawObject(raw_object_read) => {
                let raw_object = raw_object_read.object()?;
                Ok(serde_json::to_string_pretty(&raw_object)?)
            }
            _ => Ok(serde_json::to_string_pretty(self)?),
        });
        write!(f, "{}", s)
    }
}

fn unwrap_err_to_string<T: Display, F: FnOnce() -> Result<T, anyhow::Error>>(func: F) -> String {
    match func() {
        Ok(s) => format!("{s}"),
        Err(err) => format!("{err}").red().to_string(),
    }
}

impl SomaClientCommandResult {
    pub fn objects_response(&self) -> Option<Vec<SomaObjectResponse>> {
        use SomaClientCommandResult::*;
        match self {
            Object(o) | RawObject(o) => Some(vec![o.clone()]),
            Objects(o) => Some(o.clone()),
            _ => None,
        }
    }

    pub fn print(&self, pretty: bool) {
        let line = if pretty {
            format!("{self}")
        } else {
            format!("{:?}", self)
        };
        // Log line by line
        for line in line.lines() {
            // Logs write to a file on the side.  Print to stdout and also log to file, for tests to pass.
            println!("{line}");
            info!("{line}")
        }
    }

    pub fn tx_block_response(&self) -> Option<&SomaTransactionBlockResponse> {
        use SomaClientCommandResult::*;
        match self {
            TransactionBlock(b) => Some(b),
            _ => None,
        }
    }

    pub async fn prerender_clever_errors(mut self, context: &mut WalletContext) -> Self {
        match &mut self {
            SomaClientCommandResult::DryRun(DryRunTransactionBlockResponse { effects, .. })
            | SomaClientCommandResult::TransactionBlock(SomaTransactionBlockResponse {
                effects: Some(effects),
                ..
            }) => {
                let client = context.get_client().await.expect("Cannot connect to RPC");
                prerender_clever_errors(effects, client.read_api()).await
            }

            SomaClientCommandResult::TransactionBlock(SomaTransactionBlockResponse {
                effects: None,
                ..
            }) => (),
            SomaClientCommandResult::ActiveAddress(_)
            | SomaClientCommandResult::ActiveEnv(_)
            | SomaClientCommandResult::Addresses(_)
            | SomaClientCommandResult::Balance(_, _)
            | SomaClientCommandResult::ComputeTransactionDigest(_)
            | SomaClientCommandResult::ChainIdentifier(_)
            | SomaClientCommandResult::Envs(_, _)
            | SomaClientCommandResult::Gas(_)
            | SomaClientCommandResult::NewAddress(_)
            | SomaClientCommandResult::NewEnv(_)
            | SomaClientCommandResult::NoOutput
            | SomaClientCommandResult::Object(_)
            | SomaClientCommandResult::Objects(_)
            | SomaClientCommandResult::RemoveAddress(_)
            | SomaClientCommandResult::RawObject(_)
            | SomaClientCommandResult::SerializedSignedTransaction(_)
            | SomaClientCommandResult::SerializedUnsignedTransaction(_)
            | SomaClientCommandResult::Switch(_)
            | SomaClientCommandResult::SyncClientState
        }
        self
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AddressesOutput {
    pub active_address: SomaAddress,
    pub addresses: Vec<(String, SomaAddress)>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NewAddressOutput {
    pub alias: String,
    pub address: SomaAddress,
    pub key_scheme: SignatureScheme,
    pub recovery_phrase: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RemoveAddressOutput {
    pub alias_or_address: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ObjectOutput {
    pub object_id: ObjectID,
    pub version: Version,
    pub digest: String,
    pub obj_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub owner: Option<Owner>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prev_tx: Option<TransactionDigest>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<SomaParsedData>,
}

impl From<&SomaObjectData> for ObjectOutput {
    fn from(obj: &SomaObjectData) -> Self {
        let obj_type = match obj.type_.as_ref() {
            Some(x) => x.to_string(),
            None => "unknown".to_string(),
        };
        Self {
            object_id: obj.object_id,
            version: obj.version,
            digest: obj.digest.to_string(),
            obj_type,
            owner: obj.owner.clone(),
            prev_tx: obj.previous_transaction,
            content: obj.content.clone(),
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GasCoinOutput {
    pub gas_coin_id: ObjectID,
    pub shannons_balance: u64,
    pub soma_balance: String,
}

impl From<&Coin> for GasCoinOutput {
    fn from(gas_coin: &Coin) -> Self {
        Self {
            gas_coin_id: *gas_coin.id(),
            shannons_balance: gas_coin.value(),
            soma_balance: format_balance(gas_coin.value() as u128, 9, 2, None),
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ObjectsOutput {
    pub object_id: ObjectID,
    pub version: Version,
    pub digest: String,
    pub object_type: String,
}

impl ObjectsOutput {
    fn from(obj: SomaObjectResponse) -> Result<Self, anyhow::Error> {
        let obj = obj.into_object()?;
        // this replicates the object type display as in the soma explorer
        let object_type = match obj.type_ {
            Some(types::object::ObjectType::Struct(x)) => {
                let address = x.address().to_string();
                // check if the address has length of 64 characters
                // otherwise, keep it as it is
                let address = if address.len() == 64 {
                    format!("0x{}..{}", &address[..4], &address[address.len() - 4..])
                } else {
                    address
                };
                format!("{}::{}::{}", address, x.module(), x.name(),)
            }
            None => "unknown".to_string(),
        };
        Ok(Self {
            object_id: obj.object_id,
            version: obj.version,
            digest: Base64::encode(obj.digest),
            object_type,
        })
    }
    fn from_vec(objs: Vec<SomaObjectResponse>) -> Result<Vec<Self>, anyhow::Error> {
        objs.into_iter()
            .map(ObjectsOutput::from)
            .collect::<Result<Vec<_>, _>>()
    }
}

#[derive(Serialize)]
#[serde(untagged)]
pub enum SomaClientCommandResult {
    ActiveAddress(Option<SomaAddress>),
    ActiveEnv(Option<String>),
    Addresses(AddressesOutput),
    Balance(Vec<Coin>, bool),
    ChainIdentifier(String),
    ComputeTransactionDigest(TransactionData),
    DryRun(DryRunTransactionBlockResponse),
    Envs(Vec<SomaEnv>, Option<String>),
    Gas(Vec<Coin>),
    NewAddress(NewAddressOutput),
    NewEnv(SomaEnv),
    NoOutput,
    Object(SomaObjectResponse),
    Objects(Vec<SomaObjectResponse>),
    RawObject(SomaObjectResponse),
    RemoveAddress(RemoveAddressOutput),
    SerializedSignedTransaction(SenderSignedData),
    SerializedUnsignedTransaction(TransactionData),
    Switch(SwitchResponse),
    SyncClientState,
    TransactionBlock(SomaTransactionBlockResponse),
}

#[derive(Serialize, Clone)]
pub struct SwitchResponse {
    /// Active address
    pub address: Option<String>,
    pub env: Option<String>,
}

impl Display for SwitchResponse {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut writer = String::new();

        if let Some(addr) = &self.address {
            writeln!(writer, "Active address switched to {addr}")?;
        }
        if let Some(env) = &self.env {
            writeln!(writer, "Active environment switched to [{env}]")?;
        }
        write!(f, "{}", writer)
    }
}

fn pretty_print_balance(coins: &Vec<Coin>, builder: &mut TableBuilder, with_coins: bool) {
    let format_decmials = 2;
    let mut table_builder = TableBuilder::default();
    if !with_coins {
        table_builder.set_header(vec!["coin", "balance (raw)", "balance", ""]);
    }

    let balance = coins.iter().map(|x| x.balance as u128).sum::<u128>();
    let mut inner_table = TableBuilder::default();
    inner_table.set_header(vec!["coinId", "balance (raw)", "balance", ""]);

    if with_coins {
        let coin_numbers = if coins.len() != 1 { "coins" } else { "coin" };
        let balance_formatted = format!(
            "({} {})",
            format_balance(balance, coin_decimals, format_decmials, Some(symbol)),
            symbol
        );
        let summary = format!(
            "{}: {} {coin_numbers}, Balance: {} {}",
            name,
            coins.len(),
            balance,
            balance_formatted
        );
        for c in coins {
            inner_table.push_record(vec![
                c.coin_object_id.to_string().as_str(),
                c.balance.to_string().as_str(),
                format_balance(
                    c.balance as u128,
                    coin_decimals,
                    format_decmials,
                    Some(symbol),
                )
                .as_str(),
            ]);
        }
        let mut table = inner_table.build();
        table.with(TablePanel::header(summary));
        table.with(
            TableStyle::rounded()
                .horizontals([
                    HorizontalLine::new(1, TableStyle::modern().get_horizontal()),
                    HorizontalLine::new(2, TableStyle::modern().get_horizontal()),
                ])
                .remove_vertical(),
        );
        table.with(tabled::settings::style::BorderSpanCorrection);
        builder.push_record(vec![table.to_string()]);
    } else {
        table_builder.push_record(vec![
            name,
            balance.to_string().as_str(),
            format_balance(balance, coin_decimals, format_decmials, Some(symbol)).as_str(),
        ]);
    }

    let mut table = table_builder.build();
    table.with(
        TableStyle::rounded()
            .horizontals([HorizontalLine::new(
                1,
                TableStyle::modern().get_horizontal(),
            )])
            .remove_vertical(),
    );
    table.with(tabled::settings::style::BorderSpanCorrection);
    builder.push_record(vec![table.to_string()]);
}

fn divide(value: u128, divisor: u128) -> (u128, u128) {
    let integer_part = value / divisor;
    let fractional_part = value % divisor;
    (integer_part, fractional_part)
}

fn format_balance(
    value: u128,
    coin_decimals: u8,
    format_decimals: usize,
    symbol: Option<&str>,
) -> String {
    let mut suffix = if let Some(symbol) = symbol {
        format!(" {symbol}")
    } else {
        "".to_string()
    };

    let mut coin_decimals = coin_decimals as u32;
    let billions = 10u128.pow(coin_decimals + 9);
    let millions = 10u128.pow(coin_decimals + 6);
    let thousands = 10u128.pow(coin_decimals + 3);
    let units = 10u128.pow(coin_decimals);

    let (whole, fractional) = if value > billions {
        coin_decimals += 9;
        suffix = format!("B{suffix}");
        divide(value, billions)
    } else if value > millions {
        coin_decimals += 6;
        suffix = format!("M{suffix}");
        divide(value, millions)
    } else if value > thousands {
        coin_decimals += 3;
        suffix = format!("K{suffix}");
        divide(value, thousands)
    } else {
        divide(value, units)
    };

    let mut fractional = format!("{fractional:0width$}", width = coin_decimals as usize);
    fractional.truncate(format_decimals);

    format!("{whole}.{fractional}{suffix}")
}

pub(crate) async fn prerender_clever_errors(
    effects: &mut SomaTransactionBlockEffects,
    read_api: &ReadApi,
) {
    let SomaTransactionBlockEffects::V1(effects) = effects;
    if let SomaExecutionStatus::Failure { error } = &mut effects.status {
        if let Some(rendered) = render_clever_error_opt(error, read_api).await {
            *error = rendered;
        }
    }
}

/// Warn the user if the CLI falls behind more than 2 protocol versions.
async fn check_protocol_version_and_warn(read_api: &ReadApi) -> Result<(), anyhow::Error> {
    let protocol_cfg = read_api.get_protocol_config(None).await?;
    let on_chain_protocol_version = protocol_cfg.protocol_version.as_u64();
    let cli_protocol_version = ProtocolVersion::MAX.as_u64();
    if (cli_protocol_version + 2) < on_chain_protocol_version {
        // TODO: modify this message according to actual docs url
        eprintln!(
            "{}",
            format!(
                "[warning] CLI's protocol version is {cli_protocol_version}, but the active \
                network's protocol version is {on_chain_protocol_version}. \
                \n Consider installing the latest version of the CLI - \
                https://docs.soma.org/guides/developer/getting-started/soma-install \n\n \
                If publishing/upgrading returns a dependency verification error, then install the \
                latest CLI version."
            )
            .yellow()
            .bold()
        );
    }

    Ok(())
}

/// Dry run, execute, or serialize a transaction.
///
/// This basically extracts the logical code for each command that deals with dry run, executing,
/// or serializing a transaction and puts it in a function to reduce code duplication.
pub(crate) async fn dry_run_or_execute_or_serialize(
    signer: SomaAddress,
    tx_kind: TransactionKind,
    context: &mut WalletContext,
    gas_payment: Vec<ObjectRef>,
    processing: TxProcessingArgs,
) -> Result<SomaClientCommandResult, anyhow::Error> {
    let TxProcessingArgs {
        tx_digest,
        dry_run,
        serialize_unsigned_transaction,
        serialize_signed_transaction,
        sender,
    } = processing;

    ensure!(
        !serialize_unsigned_transaction || !serialize_signed_transaction,
        "Cannot specify both flags: --serialize-unsigned-transaction and --serialize-signed-transaction."
    );

    let client = context.get_client().await?;

    let signer = sender.unwrap_or(signer);

    if dry_run {
        return execute_dry_run(context, signer, tx_kind, gas_payment.clone(), None).await;
    }

    let gas_payment = if !gas_payment.is_empty() {
        gas_payment
    } else {
        let input_objects: Vec<_> = tx_kind
            .input_objects()?
            .iter()
            .filter_map(|o| match o {
                InputObjectKind::ImmOrOwnedObject((id, _, _)) => Some(*id),
                _ => None,
            })
            .collect();

        let gas_payment = client
            .transaction_builder()
            .select_gas(signer, None, input_objects)
            .await?;

        vec![gas_payment]
    };

    debug!("Preparing transaction data");
    let tx_data = TransactionData::new(tx_kind, signer, gas_payment);
    debug!("Finished preparing transaction data");

    if serialize_unsigned_transaction {
        Ok(SomaClientCommandResult::SerializedUnsignedTransaction(
            tx_data,
        ))
    } else if tx_digest {
        Ok(SomaClientCommandResult::ComputeTransactionDigest(tx_data))
    } else {
        let mut signatures = vec![context
            .config
            .keystore
            .sign_secure(&signer, &tx_data, Intent::soma_transaction())
            .await?
            .into()];

        let sender_signed_data = SenderSignedData::new(tx_data, signatures);
        if serialize_signed_transaction {
            Ok(SomaClientCommandResult::SerializedSignedTransaction(
                sender_signed_data,
            ))
        } else {
            let transaction = Transaction::new(sender_signed_data);
            debug!("Executing transaction: {:?}", transaction);
            let mut response = context
                .execute_transaction_may_fail(transaction.clone())
                .await?;
            debug!("Transaction executed: {:?}", transaction);
            if let Some(effects) = response.effects.as_mut() {
                prerender_clever_errors(effects, client.read_api()).await;
            }
            let effects = response.effects.as_ref().ok_or_else(|| {
                anyhow!("Effects from SomaTransactionBlockResult should not be empty")
            })?;
            if let SomaExecutionStatus::Failure { error } = effects.status() {
                return Err(anyhow!(
                    "Error executing transaction '{}': {error}",
                    response.digest
                ));
            }
            Ok(SomaClientCommandResult::TransactionBlock(response))
        }
    }
}
