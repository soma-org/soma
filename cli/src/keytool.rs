use std::{
    fmt::{Debug, Display, Formatter},
    path::PathBuf,
};

use crate::{
    error::{CliError, CliResult},
    key_identity::{get_identity_address_from_keystore, KeyIdentity},
};
use bip32::DerivationPath;
use clap::Subcommand;
use json_to_table::{json_to_table, Orientation};
use serde::Serialize;
use serde_json::json;
use soma_keys::{
    key_derive::generate_new_key,
    keypair_file::{read_keypair_from_file, write_keypair_to_file},
    keystore::{AccountKeystore, Keystore},
};
use tracing::info;
use types::{
    base::SomaAddress,
    crypto::{EncodeDecodeBase64, PublicKey, SignatureScheme, SomaKeyPair},
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
    Generate {
        key_scheme: SignatureScheme,
        derivation_path: Option<DerivationPath>,
        word_length: Option<String>,
    },
    /// Add a new key to Sui CLI Keystore using either the input mnemonic phrase or a Bech32 encoded 33-byte
    /// `flag || privkey` starting with "suiprivkey", the key scheme flag {ed25519 | secp256k1 | secp256r1}
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
    /// Output the private key of the given key identity in Sui CLI Keystore as Bech32
    /// encoded string starting with `suiprivkey`.
    Export {
        #[clap(long)]
        key_identity: KeyIdentity,
    },
    /// List all keys by its Sui address, Base64 encoded public key, key scheme name in
    /// sui.keystore.
    List {
        /// Sort by alias
        #[clap(long, short = 's')]
        sort_by_alias: bool,
    },
    /// This reads the content at the provided file path. The accepted format can be
    /// [enum SuiKeyPair] (Base64 encoded of 33-byte `flag || privkey`) or `type AuthorityKeyPair`
    /// (Base64 encoded `privkey`). This prints out the account keypair as Base64 encoded `flag || privkey`,
    /// the network keypair, worker keypair, protocol keypair as Base64 encoded `privkey`.
    LoadKeypair { file: PathBuf },
    /// Read the content at the provided file path. The accepted format can be
    /// [enum SuiKeyPair] (Base64 encoded of 33-byte `flag || privkey`) or `type AuthorityKeyPair`
    /// (Base64 encoded `privkey`). It prints its Base64 encoded public key and the key scheme flag.
    Show { file: PathBuf },
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
pub struct PrivateKeyBase64 {
    base64: String,
}

#[derive(Serialize)]
#[serde(untagged)]
pub enum CommandOutput {
    Alias(AliasUpdate),
    Generate(Key),
    Import(Key),
    Export(ExportedKey),
    List(Vec<Key>),
    LoadKeypair(KeypairData),
    PrivateKeyBase64(PrivateKeyBase64),
    Show(Key),
}

impl KeyToolCommand {
    pub async fn execute(self, keystore: &mut Keystore) -> CliResult<CommandOutput> {
        Ok(match self {
            KeyToolCommand::Alias {
                old_alias,
                new_alias,
            } => {
                let new_alias = keystore
                    .update_alias(&old_alias, new_alias.as_deref())
                    .map_err(CliError::SomaKey)?;
                CommandOutput::Alias(AliasUpdate {
                    old_alias,
                    new_alias,
                })
            }
            KeyToolCommand::Generate {
                key_scheme,
                derivation_path,
                word_length,
            } => match key_scheme {
                _ => {
                    let (soma_address, skp, _scheme, phrase) =
                        generate_new_key(key_scheme, derivation_path, word_length)
                            .map_err(CliError::SomaKey)?;
                    let file = format!("{soma_address}.key");
                    write_keypair_to_file(&skp, file).map_err(CliError::SomaKey)?;
                    let mut key = Key::from(&skp);
                    key.mnemonic = Some(phrase);
                    CommandOutput::Generate(key)
                }
            },
            KeyToolCommand::Import {
                alias,
                input_string,
                key_scheme,
                derivation_path,
            } => match SomaKeyPair::decode(&input_string) {
                Ok(skp) => {
                    info!("Importing Bech32 encoded private key to keystore");
                    let mut key = Key::from(&skp);
                    keystore
                        .add_key(alias.clone(), skp)
                        .map_err(CliError::SomaKey)?;

                    let alias = match alias {
                        Some(x) => x,
                        None => keystore
                            .get_alias_by_address(&key.soma_address)
                            .map_err(CliError::SomaKey)?,
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
                        .map_err(CliError::SomaKey)?;
                    let skp = keystore.get_key(&soma_address).map_err(CliError::SomaKey)?;
                    let mut key = Key::from(skp);

                    let alias = match alias {
                        Some(x) => x,
                        None => keystore
                            .get_alias_by_address(&key.soma_address)
                            .map_err(CliError::SomaKey)?,
                    };

                    key.alias = Some(alias);
                    CommandOutput::Import(key)
                }
            },
            KeyToolCommand::Export { key_identity } => {
                let address = get_identity_address_from_keystore(key_identity, keystore)?;
                let skp = keystore.get_key(&address).map_err(CliError::SomaKey)?;
                let key = ExportedKey {
                    exported_private_key: skp
                        .encode()
                        .map_err(|_| CliError::ErrorDecodingKeyPair)?,
                    key: Key::from(skp),
                };
                CommandOutput::Export(key)
            }
            KeyToolCommand::List { sort_by_alias } => {
                let mut keys = keystore
                    .keys()
                    .into_iter()
                    .map(|pk| {
                        let mut key = Key::from(pk);
                        key.alias = keystore.get_alias_by_address(&key.soma_address).ok();
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
                        Err(CliError::ErrorReadingKeyPair)?
                        // Authority keypair file is not stored with the flag, it will try read as BLS keypair..
                        // match read_authority_keypair_from_file(&file) {
                        //     Ok(keypair) => KeypairData {
                        //         account_keypair: keypair.encode_base64(),
                        //         network_keypair: None,
                        //         worker_keypair: None,
                        //         key_scheme: SignatureScheme::BLS12381.to_string(),
                        //     },
                        //     Err(e) => {
                        //         return Err(anyhow!(format!(
                        //             "Failed to read keypair at path {:?} err: {:?}",
                        //             file, e
                        //         )));
                        //     }
                        // }
                    }
                };
                CommandOutput::LoadKeypair(output)
            }

            KeyToolCommand::Show { file } => {
                let res = read_keypair_from_file(&file);
                match res {
                    Ok(skp) => {
                        let key = Key::from(&skp);
                        CommandOutput::Show(key)
                    }
                    Err(_) => {
                        Err(CliError::ErrorReadingKeyPair)?
                        // match read_authority_keypair_from_file(&file) {
                        //         Ok(keypair) => {
                        //             let public_base64_key = keypair.public().encode_base64();
                        //             CommandOutput::Show(Key {
                        //                 alias: None, // alias does not get stored in key files
                        //                 sui_address: (keypair.public()).into(),
                        //                 public_base64_key,
                        //                 key_scheme: SignatureScheme::BLS12381.to_string(),
                        //                 flag: SignatureScheme::BLS12381.flag(),
                        //                 peer_id: None,
                        //                 mnemonic: None,
                        //             })
                        //         }
                        //         Err(e) => CommandOutput::Error(format!(
                        //             "Failed to read keypair at path {:?}, err: {e}",
                        //             file
                        //         )),
                    }
                }
            }
        })
    }
}

impl Display for CommandOutput {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            // CommandOutput::Alias(update) => {
            //     write!(
            //         formatter,
            //         "Old alias {} was updated to {}",
            //         update.old_alias, update.new_alias
            //     )
            // }
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
