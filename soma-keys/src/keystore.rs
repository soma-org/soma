// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

use crate::error::{SomaKeyError, SomaKeyResult};
use crate::key_derive::{derive_key_pair_from_path, generate_new_key};
use crate::key_identity::KeyIdentity;
use crate::random_names::{random_name, random_names};
use anyhow::anyhow;
use async_trait::async_trait;
use bip32::DerivationPath;
use bip39::{Language, Mnemonic, Seed};
use enum_dispatch::enum_dispatch;
use rand::{SeedableRng, rngs::StdRng};
use regex::Regex;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::{BTreeMap, HashSet};
use std::fmt::Write;
use std::fmt::{Display, Formatter};
use std::fs;
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use types::base::SomaAddress;
use types::crypto::Signature;
use types::crypto::get_key_pair_from_rng;
use types::crypto::{EncodeDecodeBase64, PublicKey, SignatureScheme, SomaKeyPair};
use types::intent::{Intent, IntentMessage};

#[derive(Serialize, Deserialize)]
#[enum_dispatch(AccountKeystore)]
pub enum Keystore {
    File(FileBasedKeystore),
    InMem(InMemKeystore),
}

#[async_trait]
#[enum_dispatch]
pub trait AccountKeystore: Send + Sync {
    fn add_key(&mut self, alias: Option<String>, keypair: SomaKeyPair) -> SomaKeyResult<()>;
    fn remove_key(&mut self, address: SomaAddress) -> SomaKeyResult<()>;
    fn keys(&self) -> Vec<PublicKey>;
    fn get_key(&self, address: &SomaAddress) -> SomaKeyResult<&SomaKeyPair>;

    fn addresses(&self) -> Vec<SomaAddress> {
        self.keys().iter().map(|k| k.into()).collect()
    }
    fn addresses_with_alias(&self) -> Vec<(&SomaAddress, &Alias)>;
    fn aliases(&self) -> Vec<&Alias>;
    fn aliases_mut(&mut self) -> Vec<&mut Alias>;
    fn alias_names(&self) -> Vec<&str> {
        self.aliases()
            .into_iter()
            .map(|a| a.alias.as_str())
            .collect()
    }
    /// Return an array of `PublicKey`, consisting of every public key in the keystore.
    fn entries(&self) -> Vec<PublicKey>;
    /// Get alias of address
    fn get_alias_by_address(&self, address: &SomaAddress) -> SomaKeyResult<String>;
    fn get_address_by_alias(&self, alias: String) -> SomaKeyResult<&SomaAddress>;
    /// Check if an alias exists by its name
    fn alias_exists(&self, alias: &str) -> bool {
        self.alias_names().contains(&alias)
    }

    /// Import a keypair into the keystore from a `SuiKeyPair` and optional alias.
    async fn import(
        &mut self,
        alias: Option<String>,
        keypair: SomaKeyPair,
    ) -> Result<(), anyhow::Error>;

    /// Sign a message with the keypair corresponding to the given address with the given intent.
    async fn sign_secure<T>(
        &self,
        address: &SomaAddress,
        msg: &T,
        intent: Intent,
    ) -> Result<Signature, signature::Error>
    where
        T: Serialize + Sync;

    /// Get address by its identity: a type which is either an address or an alias.
    fn get_by_identity(&self, key_identity: &KeyIdentity) -> Result<SomaAddress, anyhow::Error> {
        match key_identity {
            KeyIdentity::Address(addr) => Ok(*addr),
            KeyIdentity::Alias(alias) => Ok(*self
                .addresses_with_alias()
                .iter()
                .find(|(_, a)| a.alias == *alias)
                .ok_or_else(|| anyhow!("Cannot resolve alias {alias} to an address"))?
                .0),
        }
    }

    fn create_alias(&self, alias: Option<String>) -> SomaKeyResult<String>;

    fn update_alias(&mut self, old_alias: &str, new_alias: Option<&str>) -> SomaKeyResult<String>;

    // Internal function. Use update_alias instead
    fn update_alias_value(
        &mut self,
        old_alias: &str,
        new_alias: Option<&str>,
    ) -> SomaKeyResult<String> {
        if !self.alias_exists(old_alias) {
            Err(SomaKeyError::AliasError(format!(
                "The provided alias {old_alias} does not exist"
            )))?
        }
        let new_alias_name = self.create_alias(new_alias.map(str::to_string))?;
        for a in self.aliases_mut() {
            if a.alias == old_alias {
                let pk = &a.public_key_base64;
                *a = Alias {
                    alias: new_alias_name.clone(),
                    public_key_base64: pk.clone(),
                };
            }
        }
        Ok(new_alias_name)
    }

    fn generate_and_add_new_key(
        &mut self,
        key_scheme: SignatureScheme,
        alias: Option<String>,
        derivation_path: Option<DerivationPath>,
        word_length: Option<String>,
    ) -> SomaKeyResult<(SomaAddress, String, SignatureScheme)> {
        let (address, kp, scheme, phrase) =
            generate_new_key(key_scheme, derivation_path, word_length)?;
        self.add_key(alias, kp)?;
        Ok((address, phrase, scheme))
    }

    fn import_from_mnemonic(
        &mut self,
        phrase: &str,
        key_scheme: SignatureScheme,
        derivation_path: Option<DerivationPath>,
        alias: Option<String>,
    ) -> SomaKeyResult<SomaAddress> {
        let mnemonic = Mnemonic::from_phrase(phrase, Language::English)
            .map_err(|e| SomaKeyError::InvalidMnemonic(e.to_string()))?;
        let seed = Seed::new(&mnemonic, "");
        match derive_key_pair_from_path(seed.as_bytes(), derivation_path, &key_scheme) {
            Ok((address, kp)) => {
                self.add_key(alias, kp)?;
                Ok(address)
            }
            Err(_) => Err(SomaKeyError::InvalidDerivationPath),
        }
    }
}

impl Display for Keystore {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut writer = String::new();
        match self {
            Keystore::File(file) => {
                writeln!(writer, "Keystore Type : File")?;
                write!(writer, "Keystore Path : {:?}", file.path)?;
                write!(f, "{}", writer)
            }
            Keystore::InMem(_) => {
                writeln!(writer, "Keystore Type : InMem")?;
                write!(f, "{}", writer)
            }
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Alias {
    pub alias: String,
    pub public_key_base64: String,
}

#[derive(Default)]
pub struct FileBasedKeystore {
    keys: BTreeMap<SomaAddress, SomaKeyPair>,
    aliases: BTreeMap<SomaAddress, Alias>,
    path: Option<PathBuf>,
}

impl Serialize for FileBasedKeystore {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(
            self.path
                .as_ref()
                .unwrap_or(&PathBuf::default())
                .to_str()
                .unwrap_or(""),
        )
    }
}

impl<'de> Deserialize<'de> for FileBasedKeystore {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        use serde::de::Error;
        FileBasedKeystore::new(&PathBuf::from(String::deserialize(deserializer)?))
            .map_err(D::Error::custom)
    }
}

#[async_trait]
impl AccountKeystore for FileBasedKeystore {
    fn add_key(&mut self, alias: Option<String>, keypair: SomaKeyPair) -> SomaKeyResult<()> {
        let address: SomaAddress = (&keypair.public()).into();
        let alias = self.create_alias(alias)?;
        self.aliases.insert(
            address,
            Alias {
                alias,
                public_key_base64: keypair.public().encode_base64(),
            },
        );
        self.keys.insert(address, keypair);
        self.save()?;
        Ok(())
    }

    fn remove_key(&mut self, address: SomaAddress) -> SomaKeyResult<()> {
        self.aliases.remove(&address);
        self.keys.remove(&address);
        self.save()?;
        Ok(())
    }

    fn entries(&self) -> Vec<PublicKey> {
        self.keys.values().map(|key| key.public()).collect()
    }

    async fn import(
        &mut self,
        alias: Option<String>,
        keypair: SomaKeyPair,
    ) -> Result<(), anyhow::Error> {
        let address: SomaAddress = (&keypair.public()).into();
        let alias = self.create_alias(alias)?;
        self.aliases.insert(
            address,
            Alias {
                alias,
                public_key_base64: keypair.public().encode_base64(),
            },
        );
        self.keys.insert(address, keypair);
        self.save()?;
        Ok(())
    }

    /// Return an array of `Alias`, consisting of every alias and its corresponding public key.
    fn aliases(&self) -> Vec<&Alias> {
        self.aliases.values().collect()
    }

    fn addresses_with_alias(&self) -> Vec<(&SomaAddress, &Alias)> {
        self.aliases.iter().collect::<Vec<_>>()
    }

    /// Return an array of `Alias`, consisting of every alias and its corresponding public key.
    fn aliases_mut(&mut self) -> Vec<&mut Alias> {
        self.aliases.values_mut().collect()
    }

    fn keys(&self) -> Vec<PublicKey> {
        self.keys.values().map(|key| key.public()).collect()
    }

    /// This function returns an error if the provided alias already exists. If the alias
    /// has not already been used, then it returns the alias.
    /// If no alias has been passed, it will generate a new alias.
    fn create_alias(&self, alias: Option<String>) -> SomaKeyResult<String> {
        match alias {
            Some(a) if self.alias_exists(&a) => Err(SomaKeyError::AliasError(format!(
                "Alias {a} already exists. Please choose another alias."
            )))?,
            Some(a) => validate_alias(&a),
            None => Ok(random_name(
                &self
                    .alias_names()
                    .into_iter()
                    .map(|x| x.to_string())
                    .collect::<HashSet<_>>(),
            )),
        }
    }

    /// Get the address by its alias
    fn get_address_by_alias(&self, alias: String) -> SomaKeyResult<&SomaAddress> {
        self.addresses_with_alias()
            .iter()
            .find(|x| x.1.alias == alias)
            .ok_or_else(|| {
                SomaKeyError::AliasError(format!("Cannot resolve alias {alias} to an address"))
            })
            .map(|x| x.0)
    }

    /// Get the alias if it exists, or return an error if it does not exist.
    fn get_alias_by_address(&self, address: &SomaAddress) -> SomaKeyResult<String> {
        match self.aliases.get(address) {
            Some(alias) => Ok(alias.alias.clone()),
            None => Err(SomaKeyError::AliasError(format!(
                "Cannot find alias for address {address}"
            ))),
        }
    }

    fn get_key(&self, address: &SomaAddress) -> SomaKeyResult<&SomaKeyPair> {
        match self.keys.get(address) {
            Some(key) => Ok(key),
            None => Err(SomaKeyError::AliasError(format!(
                "Cannot find key for address: [{address}]"
            ))),
        }
    }

    /// Updates an old alias to the new alias and saves it to the alias file.
    /// If the new_alias is None, it will generate a new random alias.
    fn update_alias(&mut self, old_alias: &str, new_alias: Option<&str>) -> SomaKeyResult<String> {
        let new_alias_name = self.update_alias_value(old_alias, new_alias)?;
        self.save_aliases()?;
        Ok(new_alias_name)
    }

    async fn sign_secure<T>(
        &self,
        address: &SomaAddress,
        msg: &T,
        intent: Intent,
    ) -> Result<Signature, signature::Error>
    where
        T: Serialize + Sync,
    {
        Ok(Signature::new_secure(
            &IntentMessage::new(intent, msg),
            self.keys.get(address).ok_or_else(|| {
                signature::Error::from_source(format!("Cannot find key for address: [{address}]"))
            })?,
        ))
    }
}

impl FileBasedKeystore {
    pub fn new(path: &PathBuf) -> SomaKeyResult<Self> {
        let keys = if path.exists() {
            let reader = BufReader::new(File::open(path).map_err(|_| {
                SomaKeyError::KeyStoreError(format!(
                    "Cannot open the keystore file: {}",
                    path.display()
                ))
            })?);
            let kp_strings: Vec<String> = serde_json::from_reader(reader).map_err(|_| {
                SomaKeyError::KeyStoreError(format!(
                    "Cannot deserialize the keystore file: {}",
                    path.display()
                ))
            })?;
            kp_strings
                .iter()
                .map(|kpstr| {
                    let key = SomaKeyPair::decode_base64(kpstr);
                    key.map(|k| (SomaAddress::from(&k.public()), k))
                })
                .collect::<Result<BTreeMap<_, _>, _>>()
                .map_err(|e| {
                    SomaKeyError::KeyStoreError(format!(
                        "Invalid keystore file: {}. {}",
                        path.display(),
                        e
                    ))
                })?
        } else {
            BTreeMap::new()
        };

        // check aliases
        let mut aliases_path = path.clone();
        aliases_path.set_extension("aliases");

        let aliases = if aliases_path.exists() {
            let reader = BufReader::new(File::open(&aliases_path).map_err(|_| {
                SomaKeyError::AliasError(format!(
                    "Cannot open aliases file in keystore: {}",
                    aliases_path.display()
                ))
            })?);

            let aliases: Vec<Alias> = serde_json::from_reader(reader).map_err(|_| {
                SomaKeyError::AliasError(format!(
                    "Cannot deserialize aliases file in keystore: {}",
                    aliases_path.display()
                ))
            })?;

            aliases
                .into_iter()
                .map(|alias| {
                    let key = PublicKey::decode_base64(&alias.public_key_base64);
                    key.map(|k| (Into::<SomaAddress>::into(&k), alias))
                })
                .collect::<Result<BTreeMap<_, _>, _>>()
                .map_err(|e| {
                    SomaKeyError::AliasError(format!(
                        "Invalid aliases file in keystore: {}. {}",
                        aliases_path.display(),
                        e
                    ))
                })?
        } else if keys.is_empty() {
            BTreeMap::new()
        } else {
            let names: Vec<String> = random_names(HashSet::new(), keys.len());
            let aliases = keys
                .iter()
                .zip(names)
                .map(|((sui_address, skp), alias)| {
                    let public_key_base64 = skp.public().encode_base64();
                    (
                        *sui_address,
                        Alias {
                            alias,
                            public_key_base64,
                        },
                    )
                })
                .collect::<BTreeMap<_, _>>();
            let aliases_store = serde_json::to_string_pretty(&aliases.values().collect::<Vec<_>>())
                .map_err(|_| {
                    SomaKeyError::AliasError(format!(
                        "Cannot serialize aliases to file in keystore: {}",
                        aliases_path.display(),
                    ))
                })?;
            fs::write(aliases_path, aliases_store)
                .map_err(|e| SomaKeyError::FileSystemError(e.to_string()))?;
            aliases
        };

        Ok(Self {
            keys,
            aliases,
            path: Some(path.to_path_buf()),
        })
    }

    pub fn set_path(&mut self, path: &Path) {
        self.path = Some(path.to_path_buf());
    }

    pub fn save_aliases(&self) -> SomaKeyResult<()> {
        if let Some(path) = &self.path {
            let aliases_store = serde_json::to_string_pretty(
                &self.aliases.values().collect::<Vec<_>>(),
            )
            .map_err(|_| {
                SomaKeyError::AliasError(format!(
                    "Cannot serialize aliases to file in keystore: {}",
                    path.display(),
                ))
            })?;

            let mut aliases_path = path.clone();
            aliases_path.set_extension("aliases");
            fs::write(aliases_path, aliases_store)
                .map_err(|e| SomaKeyError::FileSystemError(e.to_string()))?;
        }
        Ok(())
    }

    /// Keys saved as Base64 with 33 bytes `flag || privkey` ($BASE64_STR).
    /// To see Bech32 format encoding, use `sui keytool export $SUI_ADDRESS` where
    /// $SUI_ADDRESS can be found with `sui keytool list`. Or use `sui keytool convert $BASE64_STR`
    pub fn save_keystore(&self) -> SomaKeyResult<()> {
        if let Some(path) = &self.path {
            let store = serde_json::to_string_pretty(
                &self
                    .keys
                    .values()
                    .map(|k| k.encode_base64())
                    .collect::<Vec<_>>(),
            )
            .map_err(|_| {
                SomaKeyError::KeyStoreError(format!(
                    "Cannot serialize keystore to file: {}",
                    path.display()
                ))
            })?;
            fs::write(path, store).map_err(|e| SomaKeyError::FileSystemError(e.to_string()))?;
        }
        Ok(())
    }

    pub fn save(&self) -> SomaKeyResult<()> {
        self.save_aliases()?;
        self.save_keystore()?;
        Ok(())
    }

    pub fn key_pairs(&self) -> Vec<&SomaKeyPair> {
        self.keys.values().collect()
    }
}

#[derive(Default, Serialize, Deserialize)]
pub struct InMemKeystore {
    aliases: BTreeMap<SomaAddress, Alias>,
    keys: BTreeMap<SomaAddress, SomaKeyPair>,
}

#[async_trait]
impl AccountKeystore for InMemKeystore {
    fn add_key(&mut self, alias: Option<String>, keypair: SomaKeyPair) -> SomaKeyResult<()> {
        let address: SomaAddress = (&keypair.public()).into();
        let alias = alias.unwrap_or_else(|| {
            random_name(
                &self
                    .aliases()
                    .iter()
                    .map(|x| x.alias.clone())
                    .collect::<HashSet<_>>(),
            )
        });

        let public_key_base64 = keypair.public().encode_base64();
        let alias = Alias {
            alias,
            public_key_base64,
        };
        self.aliases.insert(address, alias);
        self.keys.insert(address, keypair);
        Ok(())
    }

    async fn import(
        &mut self,
        alias: Option<String>,
        keypair: SomaKeyPair,
    ) -> Result<(), anyhow::Error> {
        let address: SomaAddress = (&keypair.public()).into();
        let alias = alias.unwrap_or_else(|| {
            random_name(
                &self
                    .aliases()
                    .iter()
                    .map(|x| x.alias.clone())
                    .collect::<HashSet<_>>(),
            )
        });

        let public_key_base64 = keypair.public().encode_base64();
        let alias = Alias {
            alias,
            public_key_base64,
        };
        self.aliases.insert(address, alias);
        self.keys.insert(address, keypair);
        Ok(())
    }

    fn entries(&self) -> Vec<PublicKey> {
        self.keys.values().map(|key| key.public()).collect()
    }

    fn remove_key(&mut self, address: SomaAddress) -> SomaKeyResult<()> {
        self.aliases.remove(&address);
        self.keys.remove(&address);
        Ok(())
    }

    /// Get all aliases objects
    fn aliases(&self) -> Vec<&Alias> {
        self.aliases.values().collect()
    }

    fn addresses_with_alias(&self) -> Vec<(&SomaAddress, &Alias)> {
        self.aliases.iter().collect::<Vec<_>>()
    }

    fn keys(&self) -> Vec<PublicKey> {
        self.keys.values().map(|key| key.public()).collect()
    }

    fn get_key(&self, address: &SomaAddress) -> SomaKeyResult<&SomaKeyPair> {
        match self.keys.get(address) {
            Some(key) => Ok(key),
            None => Err(SomaKeyError::AddressError(format!(
                "Cannot find key for address: [{address}]"
            ))),
        }
    }

    /// Get alias of address
    fn get_alias_by_address(&self, address: &SomaAddress) -> SomaKeyResult<String> {
        match self.aliases.get(address) {
            Some(alias) => Ok(alias.alias.clone()),
            None => Err(SomaKeyError::AliasError(format!(
                "Cannot find alias for address {address}"
            ))),
        }
    }

    /// Get the address by its alias
    fn get_address_by_alias(&self, alias: String) -> SomaKeyResult<&SomaAddress> {
        self.addresses_with_alias()
            .iter()
            .find(|x| x.1.alias == alias)
            .ok_or_else(|| {
                SomaKeyError::AliasError(format!("Cannot resolve alias {alias} to an address"))
            })
            .map(|x| x.0)
    }

    /// This function returns an error if the provided alias already exists. If the alias
    /// has not already been used, then it returns the alias.
    /// If no alias has been passed, it will generate a new alias.
    fn create_alias(&self, alias: Option<String>) -> SomaKeyResult<String> {
        match alias {
            Some(a) if self.alias_exists(&a) => Err(SomaKeyError::AliasError(format!(
                "Alias {a} already exists. Please choose another alias."
            ))),
            Some(a) => validate_alias(&a),
            None => Ok(random_name(
                &self
                    .alias_names()
                    .into_iter()
                    .map(|x| x.to_string())
                    .collect::<HashSet<_>>(),
            )),
        }
    }

    fn aliases_mut(&mut self) -> Vec<&mut Alias> {
        self.aliases.values_mut().collect()
    }

    /// Updates an old alias to the new alias. If the new_alias is None,
    /// it will generate a new random alias.
    fn update_alias(&mut self, old_alias: &str, new_alias: Option<&str>) -> SomaKeyResult<String> {
        self.update_alias_value(old_alias, new_alias)
    }

    async fn sign_secure<T>(
        &self,
        address: &SomaAddress,
        msg: &T,
        intent: Intent,
    ) -> Result<Signature, signature::Error>
    where
        T: Serialize + Sync,
    {
        Ok(Signature::new_secure(
            &IntentMessage::new(intent, msg),
            self.keys.get(address).ok_or_else(|| {
                signature::Error::from_source(format!("Cannot find key for address: [{address}]"))
            })?,
        ))
    }
}

impl InMemKeystore {
    pub fn new_insecure_for_tests(initial_key_number: usize) -> Self {
        let mut rng = StdRng::from_seed([0; 32]);
        let keys = (0..initial_key_number)
            .map(|_| get_key_pair_from_rng(&mut rng))
            .map(|(ad, k)| (ad, SomaKeyPair::Ed25519(k)))
            .collect::<BTreeMap<SomaAddress, SomaKeyPair>>();

        let aliases = keys
            .iter()
            .zip(random_names(HashSet::new(), keys.len()))
            .map(|((sui_address, skp), alias)| {
                let public_key_base64 = skp.public().encode_base64();
                (
                    *sui_address,
                    Alias {
                        alias,
                        public_key_base64,
                    },
                )
            })
            .collect::<BTreeMap<_, _>>();

        Self { aliases, keys }
    }
}

fn validate_alias(alias: &str) -> SomaKeyResult<String> {
    let re = Regex::new(r"^[A-Za-z][A-Za-z0-9-_\.]*$").map_err(|_| {
        SomaKeyError::RegexError(
            "Cannot build the regex needed to validate the alias naming".to_string(),
        )
    })?;
    let alias = alias.trim();
    if !re.is_match(alias) {
        return Err(SomaKeyError::AliasError(
            "Invalid alias. A valid alias must start with a letter and can contain only letters, digits, hyphens (-), dots (.), or underscores (_).".to_string(),
        ));
    }
    Ok(alias.to_string())
}

#[cfg(test)]
mod tests {
    use crate::keystore::validate_alias;

    #[test]
    fn validate_alias_test() {
        // OK
        assert!(validate_alias("A.B_dash").is_ok());
        assert!(validate_alias("A.B-C1_dash").is_ok());
        assert!(validate_alias("abc_123.sui").is_ok());
        // Not allowed
        assert!(validate_alias("A.B-C_dash!").is_err());
        assert!(validate_alias(".B-C_dash!").is_err());
        assert!(validate_alias("_test").is_err());
        assert!(validate_alias("123").is_err());
        assert!(validate_alias("@@123").is_err());
        assert!(validate_alias("@_Ab").is_err());
        assert!(validate_alias("_Ab").is_err());
        assert!(validate_alias("^A").is_err());
        assert!(validate_alias("-A").is_err());
    }
}
