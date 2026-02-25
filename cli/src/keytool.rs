// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use anyhow::anyhow;
use bip32::DerivationPath;
use clap::Subcommand;
use fastcrypto::{
    encoding::{Base64, Encoding, Hex},
    hash::HashFunction as _,
    traits::KeyPair as _,
};
use json_to_table::{Orientation, json_to_table};
use serde::Serialize;
use serde_json::json;
use soma_keys::{
    key_derive::generate_new_key,
    key_identity::KeyIdentity,
    keypair_file::{
        read_authority_keypair_from_file, read_keypair_from_file, write_authority_keypair_to_file,
        write_keypair_to_file,
    },
    keystore::{AccountKeystore, Keystore},
};
use std::{
    fmt::{Debug, Display, Formatter},
    path::{Path, PathBuf},
};
use tabled::{
    builder::Builder,
    settings::{Modify, Rotate, Width, object::Rows},
};
use tracing::info;
use types::{
    base::SomaAddress,
    crypto::{
        DefaultHash, EncodeDecodeBase64, GenericSignature, PublicKey, SignatureScheme, SomaKeyPair,
    },
    error::SomaResult,
    intent::{Intent, IntentMessage},
    multisig::{MultiSig, MultiSigPublicKey, ThresholdUnit, WeightUnit},
    transaction::TransactionData,
};

#[allow(clippy::large_enum_variant)]
#[derive(Subcommand)]
#[clap(rename_all = "kebab-case")]
pub enum KeyToolCommand {
    /// Update an old alias to a new one.
    /// If a new alias is not provided, a random one will be generated.
    #[clap(name = "update-alias")]
    Alias {
        old_alias: String,
        /// The alias must start with a letter and can contain only letters, digits, dots, hyphens (-), or underscores (_).
        new_alias: Option<String>,
    },
    /// Given a Base64 encoded transaction bytes, decode its components. If a signature is provided,
    /// verify the signature against the transaction and output the result.
    DecodeOrVerifyTx {
        #[clap(long)]
        tx_bytes: String,
        #[clap(long)]
        sig: Option<GenericSignature>,
        #[clap(long, default_value = "0")]
        cur_epoch: u64,
    },
    /// Given a Base64 encoded MultiSig signature, decode its components.
    /// If tx_bytes is passed in, verify the multisig.
    DecodeMultiSig {
        #[clap(long)]
        multisig: MultiSig,
        #[clap(long)]
        tx_bytes: Option<String>,
        #[clap(long, default_value = "0")]
        cur_epoch: u64,
    },
    /// Generate a new keypair with key scheme flag {ed25519 | secp256k1 | secp256r1}
    /// with optional derivation path, default to m/44'/784'/0'/0'/0' for ed25519 or
    /// m/54'/784'/0'/0/0 for secp256k1 or m/74'/784'/0'/0/0 for secp256r1. Word
    /// length can be { word12 | word15 | word18 | word21 | word24} default to word12
    /// if not specified.
    ///
    /// The keypair file is output to the current directory. The content of the file is
    /// a Base64 encoded string of 33-byte `flag || privkey`.
    ///
    /// Use `soma wallet new` if you want to generate and save the key into soma.keystore.
    Generate {
        key_scheme: SignatureScheme,
        derivation_path: Option<DerivationPath>,
        word_length: Option<String>,
    },
    /// Add a new key to SOMA CLI Keystore using either the input mnemonic phrase or a Bech32 encoded 33-byte
    /// `flag || privkey` starting with "somaprivkey", the key scheme flag {ed25519 | secp256k1 | secp256r1}
    /// and an optional derivation path, default to m/44'/784'/0'/0'/0' for ed25519 or m/54'/784'/0'/0/0
    /// for secp256k1 or m/74'/784'/0'/0/0 for secp256r1. Supports mnemonic phrase of word length 12, 15,
    /// 18, 21, 24. Set an alias for the key with the --alias flag. If no alias is provided, the tool will
    /// automatically generate one.
    Import {
        /// Sets an alias for this address. The alias must start with a letter and can contain only letters, digits, hyphens (-), or underscores (_).
        #[clap(long)]
        alias: Option<String>,
        input_string: String,
        key_scheme: SignatureScheme,
        derivation_path: Option<DerivationPath>,
    },
    /// Output the private key of the given key identity in SOMA CLI Keystore as Bech32
    /// encoded string starting with `somaprivkey`.
    Export {
        #[clap(long)]
        key_identity: KeyIdentity,
    },
    /// List all keys by its SOMA address, Base64 encoded public key, key scheme name in
    /// soma.keystore.
    List {
        /// Sort by alias
        #[clap(long, short = 's')]
        sort_by_alias: bool,
    },
    /// This reads the content at the provided file path. The accepted format can be
    /// [enum SomaKeyPair] (Base64 encoded of 33-byte `flag || privkey`) or `type AuthorityKeyPair`
    /// (Base64 encoded `privkey`). This prints out the account keypair as Base64 encoded `flag || privkey`,
    /// the network keypair, worker keypair, protocol keypair as Base64 encoded `privkey`.
    LoadKeypair { file: PathBuf },
    /// To MultiSig SOMA Address. Pass in a list of all public keys `flag || pk` in Base64.
    /// See `keytool list` for example public keys.
    MultiSigAddress {
        #[clap(long)]
        threshold: ThresholdUnit,
        #[clap(long, num_args(1..))]
        pks: Vec<PublicKey>,
        #[clap(long, num_args(1..))]
        weights: Vec<WeightUnit>,
    },
    /// Provides a list of participating signatures (`flag || sig || pk` encoded in Base64),
    /// threshold, a list of all public keys and a list of their weights that define the
    /// MultiSig address. Returns a valid MultiSig signature and its sender address. The
    /// result can be used as signature field for `soma tx execute-signed`. The sum
    /// of weights of all signatures must be >= the threshold.
    ///
    /// The order of `sigs` must be the same as the order of `pks`.
    /// e.g. for [pk1, pk2, pk3, pk4, pk5], [sig1, sig2, sig5] is valid, but
    /// [sig2, sig1, sig5] is invalid.
    MultiSigCombinePartialSig {
        #[clap(long, num_args(1..))]
        sigs: Vec<GenericSignature>,
        #[clap(long, num_args(1..))]
        pks: Vec<PublicKey>,
        #[clap(long, num_args(1..))]
        weights: Vec<WeightUnit>,
        #[clap(long)]
        threshold: ThresholdUnit,
    },
    /// Read the content at the provided file path. The accepted format can be
    /// [enum SomaKeyPair] (Base64 encoded of 33-byte `flag || privkey`) or `type AuthorityKeyPair`
    /// (Base64 encoded `privkey`). It prints its Base64 encoded public key and the key scheme flag.
    Show { file: PathBuf },
    /// Create signature using the private key for the given address (or its alias) in soma keystore.
    /// Any signature commits to a [struct IntentMessage] consisting of the Base64 encoded
    /// of the BCS serialized transaction bytes itself and its intent. If intent is absent,
    /// default will be used.
    Sign {
        #[clap(long)]
        address: KeyIdentity,
        #[clap(long)]
        data: String,
        #[clap(long)]
        intent: Option<Intent>,
    },
    /// This takes [enum SomaKeyPair] of Base64 encoded of 33-byte `flag || privkey`). It
    /// outputs the keypair into a file at the current directory where the address is the filename,
    /// and prints out its SOMA address, Base64 encoded public key, the key scheme, and the key scheme flag.
    Unpack { keypair: String },
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Key {
    alias: Option<String>,
    soma_address: SomaAddress,
    public_base64_key: String,
    key_scheme: String,
    flag: u8,
    #[serde(skip_serializing_if = "Option::is_none")]
    mnemonic: Option<String>,
}

// Command Output types
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AliasUpdate {
    old_alias: String,
    new_alias: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DecodedMultiSig {
    public_base64_key: String,
    sig_base64: String,
    weight: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DecodedMultiSigOutput {
    multisig_address: SomaAddress,
    participating_keys_signatures: Vec<DecodedMultiSig>,
    pub_keys: Vec<MultiSigOutput>,
    threshold: usize,
    sig_verify_result: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DecodeOrVerifyTxOutput {
    tx: TransactionData,
    result: Option<SomaResult>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ExportedKey {
    exported_private_key: String,
    key: Key,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct KeypairData {
    account_keypair: String,
    network_keypair: Option<String>,
    worker_keypair: Option<String>,
    key_scheme: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MultiSigAddress {
    multisig_address: String,
    multisig: Vec<MultiSigOutput>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MultiSigCombinePartialSig {
    multisig_address: SomaAddress,
    multisig_parsed: GenericSignature,
    multisig_serialized: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MultiSigOutput {
    address: SomaAddress,
    public_base64_key: String,
    weight: u8,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PrivateKeyBase64 {
    base64: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SerializedSig {
    serialized_sig_base64: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SignData {
    soma_address: SomaAddress,
    // Base64 encoded string of serialized transaction data.
    raw_tx_data: String,
    // Intent struct used, see [struct Intent] for field definitions.
    intent: Intent,
    // Base64 encoded [struct IntentMessage] consisting of (intent || message)
    // where message can be `TransactionData` etc.
    raw_intent_msg: String,
    // Base64 encoded blake2b hash of the intent message, this is what the signature commits to.
    digest: String,
    // Base64 encoded `flag || signature || pubkey` for a complete
    // serialized SOMA signature to be send for executing the transaction.
    soma_signature: String,
}

#[derive(Serialize)]
#[serde(untagged)]
#[allow(clippy::large_enum_variant)]
pub enum CommandOutput {
    Alias(AliasUpdate),
    DecodeMultiSig(DecodedMultiSigOutput),
    DecodeOrVerifyTx(DecodeOrVerifyTxOutput),
    Error(String),
    Generate(Key),
    Import(Key),
    Export(ExportedKey),
    List(Vec<Key>),
    LoadKeypair(KeypairData),
    MultiSigAddress(MultiSigAddress),
    MultiSigCombinePartialSig(MultiSigCombinePartialSig),
    PrivateKeyBase64(PrivateKeyBase64),
    Show(Key),
    Sign(SignData),
}

impl KeyToolCommand {
    pub async fn execute(self, keystore: &mut Keystore) -> Result<CommandOutput, anyhow::Error> {
        Ok(match self {
            KeyToolCommand::Alias { old_alias, new_alias } => {
                let new_alias = keystore.update_alias(&old_alias, new_alias.as_deref()).await?;
                CommandOutput::Alias(AliasUpdate { old_alias, new_alias })
            }
            KeyToolCommand::DecodeMultiSig { multisig, tx_bytes, cur_epoch } => {
                let pks = multisig.get_pk().pubkeys();
                let sigs = multisig.get_sigs();
                let bitmap = multisig.get_indices()?;
                let address = SomaAddress::from(multisig.get_pk());

                let pub_keys = pks
                    .iter()
                    .map(|(pk, w)| MultiSigOutput {
                        address: (pk).into(),
                        public_base64_key: pk.encode_base64(),
                        weight: *w,
                    })
                    .collect::<Vec<MultiSigOutput>>();

                let threshold = *multisig.get_pk().threshold() as usize;

                let mut output = DecodedMultiSigOutput {
                    multisig_address: address,
                    participating_keys_signatures: vec![],
                    pub_keys,
                    threshold,
                    sig_verify_result: "".to_string(),
                };

                for (sig, i) in sigs.iter().zip(bitmap) {
                    let (pk, w) = pks
                        .get(i as usize)
                        .ok_or(anyhow!("Invalid public keys index".to_string()))?;
                    output.participating_keys_signatures.push(DecodedMultiSig {
                        public_base64_key: pk.encode_base64().clone(),
                        sig_base64: Base64::encode(sig.as_ref()),
                        weight: w.to_string(),
                    })
                }

                if let Some(tx_bytes_val) = tx_bytes {
                    let tx_bytes = Base64::decode(&tx_bytes_val)
                        .map_err(|e| anyhow!("Invalid base64 tx bytes: {:?}", e))?;
                    let tx_data: TransactionData = bcs::from_bytes(&tx_bytes)?;
                    let s = GenericSignature::MultiSig(multisig);
                    let res = s.verify_authenticator(
                        &IntentMessage::new(Intent::soma_transaction(), tx_data),
                        address,
                    );

                    match res {
                        Ok(()) => output.sig_verify_result = "OK".to_string(),
                        Err(e) => output.sig_verify_result = format!("{:?}", e),
                    };
                };

                CommandOutput::DecodeMultiSig(output)
            }

            KeyToolCommand::DecodeOrVerifyTx { tx_bytes, sig, cur_epoch } => {
                let tx_bytes = Base64::decode(&tx_bytes)
                    .map_err(|e| anyhow!("Invalid base64 key: {:?}", e))?;
                let tx_data: TransactionData = bcs::from_bytes(&tx_bytes)?;
                match sig {
                    None => CommandOutput::DecodeOrVerifyTx(DecodeOrVerifyTxOutput {
                        tx: tx_data,
                        result: None,
                    }),
                    Some(s) => {
                        let res = s.verify_authenticator(
                            &IntentMessage::new(Intent::soma_transaction(), tx_data.clone()),
                            tx_data.sender(),
                        );
                        CommandOutput::DecodeOrVerifyTx(DecodeOrVerifyTxOutput {
                            tx: tx_data,
                            result: Some(res),
                        })
                    }
                }
            }
            KeyToolCommand::Generate { key_scheme, derivation_path, word_length } => {
                match key_scheme {
                    SignatureScheme::BLS12381 => {
                        let (soma_address, kp) = types::crypto::get_authority_key_pair();
                        let file_name = format!("bls-{soma_address}.key");
                        write_authority_keypair_to_file(&kp, file_name)?;
                        CommandOutput::Generate(Key {
                            alias: None,
                            soma_address,
                            public_base64_key: kp.public().encode_base64(),
                            key_scheme: key_scheme.to_string(),
                            flag: SignatureScheme::BLS12381.flag(),
                            mnemonic: None,
                        })
                    }
                    _ => {
                        let (soma_address, skp, _scheme, phrase) =
                            generate_new_key(key_scheme, derivation_path, word_length)?;
                        let file = format!("{soma_address}.key");
                        write_keypair_to_file(&skp, file)?;
                        let mut key = Key::from(&skp);
                        key.mnemonic = Some(phrase);
                        CommandOutput::Generate(key)
                    }
                }
            }
            KeyToolCommand::Import { alias, input_string, key_scheme, derivation_path } => {
                match SomaKeyPair::decode(&input_string) {
                    Ok(skp) => {
                        info!("Importing Bech32 encoded private key to keystore");
                        let mut key = Key::from(&skp);
                        keystore.import(alias.clone(), skp).await?;

                        let alias = match alias {
                            Some(x) => x,
                            None => keystore.get_alias(&key.soma_address)?,
                        };

                        key.alias = Some(alias);
                        CommandOutput::Import(key)
                    }
                    Err(_) => {
                        info!("Importing mneomonics to keystore");
                        let soma_address = keystore
                            .import_from_mnemonic(
                                &input_string,
                                key_scheme,
                                derivation_path,
                                alias.clone(),
                            )
                            .await?;
                        let skp = keystore.export(&soma_address)?;
                        let mut key = Key::from(skp);

                        let alias = match alias {
                            Some(x) => x,
                            None => keystore.get_alias(&key.soma_address)?,
                        };

                        key.alias = Some(alias);
                        CommandOutput::Import(key)
                    }
                }
            }
            KeyToolCommand::Export { key_identity } => {
                let address = keystore.get_by_identity(&key_identity)?;
                let skp = keystore.export(&address)?;
                let mut key = Key::from(skp);
                key.alias = keystore.get_alias(&key.soma_address).ok();
                let key = ExportedKey {
                    exported_private_key: skp
                        .encode()
                        .map_err(|_| anyhow!("Cannot decode keypair"))?,
                    key,
                };
                CommandOutput::Export(key)
            }
            KeyToolCommand::List { sort_by_alias } => {
                let mut keys = keystore
                    .entries()
                    .into_iter()
                    .map(|pk| {
                        let mut key = Key::from(pk);
                        key.alias = keystore.get_alias(&key.soma_address).ok();
                        key
                    })
                    .collect::<Vec<Key>>();
                if sort_by_alias {
                    keys.sort_unstable();
                }
                CommandOutput::List(keys)
            }

            KeyToolCommand::LoadKeypair { file } => {
                let output = match read_keypair_from_file(&file) {
                    Ok(keypair) => {
                        // Account keypair is encoded with the key scheme flag {},
                        // and network and worker keypair are not.
                        let network_worker_keypair = match &keypair {
                            SomaKeyPair::Ed25519(kp) => kp.encode_base64(),
                        };
                        KeypairData {
                            account_keypair: keypair.encode_base64(),
                            network_keypair: Some(network_worker_keypair.clone()),
                            worker_keypair: Some(network_worker_keypair),
                            key_scheme: keypair.public().scheme().to_string(),
                        }
                    }
                    Err(_) => {
                        // Authority keypair file is not stored with the flag, it will try read as BLS keypair..
                        match read_authority_keypair_from_file(&file) {
                            Ok(keypair) => KeypairData {
                                account_keypair: keypair.encode_base64(),
                                network_keypair: None,
                                worker_keypair: None,
                                key_scheme: SignatureScheme::BLS12381.to_string(),
                            },
                            Err(e) => {
                                return Err(anyhow!(format!(
                                    "Failed to read keypair at path {:?} err: {:?}",
                                    file, e
                                )));
                            }
                        }
                    }
                };
                CommandOutput::LoadKeypair(output)
            }

            KeyToolCommand::MultiSigAddress { threshold, pks, weights } => {
                let multisig_pk = MultiSigPublicKey::new(pks.clone(), weights.clone(), threshold)?;
                let address: SomaAddress = (&multisig_pk).into();
                let mut output =
                    MultiSigAddress { multisig_address: address.to_string(), multisig: vec![] };

                for (pk, w) in pks.into_iter().zip(weights.into_iter()) {
                    output.multisig.push(MultiSigOutput {
                        address: Into::<SomaAddress>::into(&pk),
                        public_base64_key: pk.encode_base64(),
                        weight: w,
                    });
                }
                CommandOutput::MultiSigAddress(output)
            }

            KeyToolCommand::MultiSigCombinePartialSig { sigs, pks, weights, threshold } => {
                let multisig_pk = MultiSigPublicKey::new(pks, weights, threshold)?;
                let address: SomaAddress = (&multisig_pk).into();
                let multisig = MultiSig::combine(sigs, multisig_pk)?;
                let generic_sig: GenericSignature = multisig.into();
                let multisig_serialized = generic_sig.encode_base64();
                CommandOutput::MultiSigCombinePartialSig(MultiSigCombinePartialSig {
                    multisig_address: address,
                    multisig_parsed: generic_sig,
                    multisig_serialized,
                })
            }

            KeyToolCommand::Show { file } => {
                let res = read_keypair_from_file(&file);
                match res {
                    Ok(skp) => {
                        let key = Key::from(&skp);
                        CommandOutput::Show(key)
                    }
                    Err(_) => match read_authority_keypair_from_file(&file) {
                        Ok(keypair) => {
                            let public_base64_key = keypair.public().encode_base64();
                            CommandOutput::Show(Key {
                                alias: None, // alias does not get stored in key files
                                soma_address: (keypair.public()).into(),
                                public_base64_key,
                                key_scheme: SignatureScheme::BLS12381.to_string(),
                                flag: SignatureScheme::BLS12381.flag(),

                                mnemonic: None,
                            })
                        }
                        Err(e) => CommandOutput::Error(format!(
                            "Failed to read keypair at path {:?}, err: {e}",
                            file
                        )),
                    },
                }
            }

            KeyToolCommand::Sign { address, data, intent } => {
                let address = keystore.get_by_identity(&address)?;
                let intent = intent.unwrap_or_else(Intent::soma_transaction);
                let intent_clone = intent.clone();
                let msg: TransactionData =
                    bcs::from_bytes(&Base64::decode(&data).map_err(|e| {
                        anyhow!("Cannot deserialize data as TransactionData {:?}", e)
                    })?)?;
                let intent_msg = IntentMessage::new(intent, msg);
                let raw_intent_msg: String = Base64::encode(bcs::to_bytes(&intent_msg)?);
                let mut hasher = DefaultHash::default();
                hasher.update(bcs::to_bytes(&intent_msg)?);
                let digest = hasher.finalize().digest;
                let soma_signature =
                    keystore.sign_secure(&address, &intent_msg.value, intent_msg.intent).await?;
                CommandOutput::Sign(SignData {
                    soma_address: address,
                    raw_tx_data: data,
                    intent: intent_clone,
                    raw_intent_msg,
                    digest: Base64::encode(digest),
                    soma_signature: soma_signature.encode_base64(),
                })
            }

            KeyToolCommand::Unpack { keypair } => {
                let keypair = SomaKeyPair::decode_base64(&keypair)
                    .map_err(|_| anyhow!("Invalid Base64 encode keypair"))?;

                let key = Key::from(&keypair);
                let path_str = format!("{}.key", key.soma_address).to_lowercase();
                let path = Path::new(&path_str);
                let out_str = format!(
                    "address: {}\nkeypair: {}\nflag: {}",
                    key.soma_address,
                    keypair.encode_base64(),
                    key.flag
                );
                std::fs::write(path, out_str).unwrap();
                CommandOutput::Show(key)
            }
        })
    }
}

impl Display for CommandOutput {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            CommandOutput::Alias(update) => {
                write!(
                    formatter,
                    "Old alias {} was updated to {}",
                    update.old_alias, update.new_alias
                )
            }
            // Sign needs to be manually built because we need to wrap the very long
            // rawTxData string and rawIntentMsg strings into multiple rows due to
            // their lengths, which we cannot do with a JsonTable
            CommandOutput::Sign(data) => {
                let intent_table = json_to_table(&json!(&data.intent))
                    .with(tabled::settings::Style::rounded().horizontals([]))
                    .to_string();

                let mut builder = Builder::default();
                builder
                    .set_header([
                        "somaSignature",
                        "digest",
                        "rawIntentMsg",
                        "intent",
                        "rawTxData",
                        "somaAddress",
                    ])
                    .push_record([
                        &data.soma_signature,
                        &data.digest,
                        &data.raw_intent_msg,
                        &intent_table,
                        &data.raw_tx_data,
                        &data.soma_address.to_string(),
                    ]);
                let mut table = builder.build();
                table.with(Rotate::Left);
                table.with(tabled::settings::Style::rounded().horizontals([]));
                table.with(Modify::new(Rows::new(0..)).with(Width::wrap(160).keep_words()));
                write!(formatter, "{}", table)
            }
            _ => {
                let json_obj = json![self];
                let mut table = json_to_table(&json_obj);
                let style = tabled::settings::Style::rounded().horizontals([]);
                table.with(style);
                table.array_orientation(Orientation::Column);
                write!(formatter, "{}", table)
            }
        }
    }
}

impl CommandOutput {
    pub fn print(&self, pretty: bool) {
        let line = if pretty { format!("{self}") } else { format!("{:?}", self) };
        // Log line by line
        for line in line.lines() {
            // Logs write to a file on the side.  Print to stdout and also log to file, for tests to pass.
            println!("{line}");
            info!("{line}")
        }
    }
}

// when --json flag is used, any output result is transformed into a JSON pretty string and sent to std output
impl Debug for CommandOutput {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match serde_json::to_string_pretty(self) {
            Ok(json) => write!(f, "{json}"),
            Err(err) => write!(f, "Error serializing JSON: {err}"),
        }
    }
}

impl From<&SomaKeyPair> for Key {
    fn from(skp: &SomaKeyPair) -> Self {
        Key::from(skp.public())
    }
}

impl From<PublicKey> for Key {
    fn from(pk: PublicKey) -> Self {
        Key {
            alias: None, // this is retrieved later
            soma_address: SomaAddress::from(&pk),
            public_base64_key: pk.encode_base64(),
            key_scheme: pk.scheme().to_string(),
            mnemonic: None,
            flag: pk.flag(),
        }
    }
}
